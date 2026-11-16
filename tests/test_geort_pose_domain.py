"""Sampling and network outputs must exclude native hyperextension branches."""
import copy
import json
import numpy as np
import pytest
import torch
from retarget.baseline.geort.adapter import RobotAdapter,digest,fk_cache_key
from retarget.baseline.geort.pose_domain import extension_domain,sample_poses,SAMPLING_POLICY
from retarget.baseline.geort.constrained_network import create_ik,DomainIK
from retarget.baseline.geort.workspace import domain_poses,training_reference_spec


def test_uniform_sampling_excludes_hyperextension_without_changing_robot_limits():
    robot=RobotAdapter();native=robot.lower.copy();spec=extension_domain(robot.lower,robot.upper)
    q=sample_poses(spec,100000,np.random.default_rng(714))
    assert domain_poses(q,spec).all() and q.shape==(100000,20)
    # Independent angle expectations: MCP/PIP/DIP and thumb MCP/IP.
    for ids,degrees in (([4,8,12,16],0),([6,10,14,18],0),([7,11,15,19],0),([2,3],0)):
        assert (np.rad2deg(q[:,ids])>=-degrees-1e-5).all()
        np.testing.assert_allclose(np.rad2deg(np.array(spec['lower'])[ids]),-degrees,atol=1e-5)
    unchanged=[0,1,5,9,13,17]
    np.testing.assert_array_equal(np.array(spec['lower'])[unchanged],native[unchanged])
    np.testing.assert_array_equal(spec['upper'],robot.upper)
    np.testing.assert_array_equal(robot.lower,native)
    # Direct uniform draws do not accumulate formerly negative poses at zero.
    assert (q[:,6]>0).all() and len(np.unique(q[:,6]))>99000
    assert not (q[:,[2,3,4,6,7,8,10,11,12,14,15,16,18,19]]<0).any()
    np.testing.assert_allclose(q.mean(0),(np.array(spec['lower'])+spec['upper'])/2,atol=.012)


@pytest.mark.parametrize('mode',['official','contact_v1'])
def test_both_objectives_save_and_restore_structurally_bounded_outputs(mode):
    robot=RobotAdapter();spec=extension_domain(robot.lower,robot.upper,dict(pip_extension_deg=4,dip_extension_deg=6))
    net=create_ik(robot,dict(training_mode=mode,output_domain=spec)).eval()
    assert isinstance(net,DomainIK)
    x=torch.randn(128,5,3)*10
    with torch.no_grad():q=robot.unnormalize(net(x).numpy())
    assert domain_poses(q,spec).all()
    loaded=create_ik(robot,dict(training_mode=mode,output_domain=spec)).eval()
    loaded.load_state_dict(copy.deepcopy(net.state_dict()))
    torch.testing.assert_close(net(x),loaded(x),rtol=0,atol=0)
    # Existing published weights remain loadable without silently remapping.
    assert not isinstance(create_ik(robot,dict(training_mode='official')),DomainIK)


def test_reference_uses_bounded_cache_and_rejects_legacy_hyperextension(tmp_path):
    robot=RobotAdapter();native=robot.spec;spec=extension_domain(robot.lower,robot.upper)
    cache=digest(dict(robot=native,sampling_policy=SAMPLING_POLICY,lower=spec['lower'],upper=spec['upper']))
    assert cache!=robot.key
    config=dict(robot=native,robot_samples=3,fk_cache_key=cache)
    assert fk_cache_key(config)==cache and fk_cache_key(dict(robot=native))==robot.key
    with pytest.raises(ValueError):fk_cache_key(dict(robot=native,fk_cache_key='../invalid'))
    model=tmp_path/'models/current';model.mkdir(parents=True)
    (model/'config.json').write_text(json.dumps(config))
    folder=tmp_path/'fk'/cache;folder.mkdir(parents=True)
    q=np.zeros((3,20),np.float32);q[1,6]=np.deg2rad(-59);q[2,7]=np.deg2rad(-59)
    before=q.copy();np.savez(folder/'kinematics_3.npz',qpos=q,keypoints=np.zeros((3,5,3)))
    reference=training_reference_spec(tmp_path)
    assert reference['labels']==str(folder/'kinematics_3.npz')
    assert domain_poses(q,reference['output_domain']).all() # legacy native domain
    assert domain_poses(q,reference['reference_domain']).tolist()==[True,False,False]
    np.testing.assert_array_equal(q,before)
    stricter=extension_domain(robot.lower,robot.upper,dict(dip_extension_deg=5))
    # A source-glove extension allowance cannot reopen robot hyperextension.
    assert stricter['lower']==spec['lower']
    assert stricter['robot_extension_policy']['dip_extension_deg']==0


def test_publication_keeps_new_domain_fk_checkpoint_instead_of_legacy_cache(tmp_path):
    from retarget.baseline.geort.model_store import prune_checkpoints
    robot=RobotAdapter();cache='a'*64
    model=tmp_path/'models/current';model.mkdir(parents=True)
    config=dict(robot=robot.spec,fk_cache_key=cache,robot_samples=100000,fk_epochs=200)
    (model/'config.json').write_text(json.dumps(config))
    for key in (cache,robot.key):
        folder=tmp_path/'fk'/key;folder.mkdir(parents=True)
        (folder/'fk_100000_200.pth').write_bytes(b'checkpoint')
    prune_checkpoints(tmp_path)
    assert (tmp_path/'fk'/cache/'fk_100000_200.pth').is_file()
    assert not (tmp_path/'fk'/robot.key/'fk_100000_200.pth').exists()


def test_reported_pose_14105_is_rejected_despite_glove_extension_allowance():
    robot=RobotAdapter()
    # Actual saved sample reported by the user; both signs matter independently.
    q=np.deg2rad([[.221,-52.921,3.906,55.69,19.966,25.501,-1.802,-17.731,
                  89.245,-38.558,116.65,20.924,71.423,15.516,110.709,39.297,
                  32.012,3.835,40.983,27.112]])
    spec=extension_domain(robot.lower,robot.upper,dict(pip_extension_deg=10,dip_extension_deg=20))
    assert not domain_poses(q,spec)[0]
    for index in (6,7):
        single=q.copy();single[0,[6,7]]=0;single[0,index]=q[0,index]
        assert not domain_poses(single,spec)[0]


def test_pose_cleanup_preserves_published_reference_and_skips_symlinks(tmp_path):
    from retarget.baseline.geort.model_store import prune_pose_caches
    from retarget.baseline.geort.workspace import REVISION
    robot=RobotAdapter();cache='b'*64;source={'serial':'hand'}
    config=dict(robot=robot.spec,fk_cache_key=cache,robot_samples=2,source=source)
    model=tmp_path/'models/current';model.mkdir(parents=True)
    (model/'config.json').write_text(json.dumps(config))
    labels=tmp_path/'fk'/cache;labels.mkdir(parents=True)
    np.savez(labels/'kinematics_2.npz',qpos=np.zeros((2,20)),keypoints=np.zeros((2,5,3)))
    reference=digest([source,REVISION,training_reference_spec(tmp_path)])
    (tmp_path/'workspace'/reference).mkdir(parents=True)
    for area in ('fk','workspace','robot_contact_targets'):
        folder=tmp_path/area/('a'*64);folder.mkdir(parents=True)
        (folder/'bad.npz').write_bytes(b'legacy pose')
        if area=='workspace':(folder/'reference_spec.json').write_text('{"revision":"legacy"}')
    other=tmp_path/'workspace'/('d'*64);other.mkdir()
    (other/'reference_spec.json').write_text(json.dumps(training_reference_spec(tmp_path)))
    external=tmp_path/'external';external.mkdir();(external/'keep.npy').write_bytes(b'keep')
    (tmp_path/'fk'/('c'*64)).symlink_to(external,target_is_directory=True)
    recording=tmp_path/'recordings/original';recording.mkdir(parents=True)
    (recording/'points.npy').write_bytes(b'original')
    removed=prune_pose_caches(tmp_path)
    assert set(removed)=={area+'/'+('a'*64) for area in ('fk','workspace','robot_contact_targets')}
    assert labels.is_dir() and (tmp_path/'workspace'/reference).is_dir()
    assert other.is_dir()
    assert (external/'keep.npy').read_bytes()==b'keep'
    assert (recording/'points.npy').read_bytes()==b'original'
