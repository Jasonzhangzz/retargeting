"""A fixed reachable-space denominator, density, publication and rollback."""
from collections import Counter
from pathlib import Path
import json
import numpy as np
import pytest
from retarget.baseline.geort.workspace import WorkspaceReference
from retarget.baseline.geort.coverage import WorkspaceCoverage
from retarget.baseline.geort.model_store import publish


def test_reachable_fraction_is_against_fixed_reference_and_density_penalizes_repeats():
    source={'serial':'hand'};cells=np.array([[i,0,0] for i in range(4)])
    reference=WorkspaceReference(source,[cells]*5,dict(samples=100))
    c=WorkspaceCoverage(source)
    c.add(np.tile([.001,.001,.001],(5,1)),1,now=0)
    c.add(np.tile([.006,.001,.001],(5,1)),2,now=0)
    first=reference.evaluate(c.coarse_counts,'thumb')['fingers'][0]
    assert first['coverage_percent']==50 and first['covered_cells']==2 and first['missing_cells']==2
    assert first['uniformity_percent']==pytest.approx(100)
    projection=first['projections'][0]['cells']
    assert sum(row[2] for row in projection)==4 and sum(row[3] for row in projection)==2
    for i in range(3,103):c.add(np.tile([.006,.001,.001],(5,1)),i,now=0)
    after=reference.evaluate(c.coarse_counts)['fingers'][0]
    assert after['coverage_percent']==50 and after['uniformity_percent']<60
    c.add(np.tile([.101,.001,.001],(5,1)),103,now=0)
    out=reference.evaluate(c.coarse_counts)['fingers'][0]
    assert out['coverage_percent']==50 and out['outside_reference_cells']==1


def staged(root,key,marker):
    folder=root/'training'/key;folder.mkdir(parents=True)
    (folder/'status.json').write_text(json.dumps(dict(status='completed',run_id=key)))
    (folder/'config.json').write_text(json.dumps(dict(run_id=key)))
    (folder/'last.pth').write_bytes(marker)
    return folder


def test_success_replaces_model_and_removes_previous_versions(tmp_path):
    old=tmp_path/'models/previous';old.mkdir(parents=True);(old/'last.pth').write_bytes(b'old')
    first=publish(tmp_path,staged(tmp_path,'a',b'A'),'a')
    assert first.name=='current' and not old.exists()
    second=publish(tmp_path,staged(tmp_path,'b',b'B'),'b')
    assert second==first and (second/'last.pth').read_bytes()==b'B'
    assert [p.name for p in (tmp_path/'models').iterdir()]==['current']
    assert json.loads((tmp_path/'active.json').read_text())==dict(model='current',run_id='b')


def test_failed_or_incomplete_run_cannot_replace_model(tmp_path):
    current=publish(tmp_path,staged(tmp_path,'a',b'A'),'a')
    failed=staged(tmp_path,'failed',b'bad');(failed/'status.json').write_text('{"status":"cancelled"}')
    with pytest.raises(ValueError):publish(tmp_path,failed,'failed')
    assert (current/'last.pth').read_bytes()==b'A'
    assert json.loads((tmp_path/'active.json').read_text())['run_id']=='a'


def test_publish_rolls_back_folder_if_pointer_update_fails(tmp_path,monkeypatch):
    import retarget.baseline.geort.training as training
    current=publish(tmp_path,staged(tmp_path,'a',b'A'),'a')
    candidate=staged(tmp_path,'b',b'B')
    monkeypatch.setattr(training,'atomic_json',lambda *a:(_ for _ in ()).throw(OSError('disk full')))
    with pytest.raises(OSError):publish(tmp_path,candidate,'b')
    assert (current/'last.pth').read_bytes()==b'A'
    assert (candidate/'last.pth').read_bytes()==b'B'
    assert json.loads((tmp_path/'active.json').read_text())['run_id']=='a'


def test_service_detects_publication_with_same_model_id_and_new_run(tmp_path):
    from moxian_web.geort_service import GeoRTService
    service=GeoRTService(tmp_path);seen=[];service.on_model_published=seen.append
    try:
        publish(tmp_path,staged(tmp_path,'a',b'A'),'a');service.refresh_published()
        publish(tmp_path,staged(tmp_path,'b',b'B'),'b');service.refresh_published()
        assert seen==['current','current'] and service.selected=='current' and service.published_run=='b'
    finally:service.close()


def test_publication_keeps_only_latest_ik_and_current_fk_weights(tmp_path):
    old=tmp_path/'training/failed';old.mkdir(parents=True);(old/'last.pth').write_bytes(b'bad')
    obsolete=tmp_path/'fk/obsolete';obsolete.mkdir(parents=True);(obsolete/'fk_100000_200.pth').write_bytes(b'obsolete')
    candidate=staged(tmp_path,'new',b'latest');(candidate/'epoch_000.pth').write_bytes(b'history')
    (candidate/'fk.pth').write_bytes(b'fk')
    current=publish(tmp_path,candidate,'new')
    assert (current/'last.pth').read_bytes()==b'latest' and (current/'fk.pth').read_bytes()==b'fk'
    assert not (current/'epoch_000.pth').exists() and not (old/'last.pth').exists()
    assert not (obsolete/'fk_100000_200.pth').exists()


def test_training_workspace_rejects_whole_pose_only_above_five_mm():
    from retarget.baseline.geort.workspace import accepted_poses,reference_cells,DENOMINATOR
    depths=np.array([0.,5.,5.0001,np.nan,1.,-1.])
    valid=np.array([True,True,True,True,False,True])
    accepted=accepted_poses(depths,valid)
    assert accepted.tolist()==[True,True,False,False,False,False]
    # Every rejected whole pose loses all five fingertips, including fingers
    # that did not participate in the colliding pair.
    points=np.zeros((6,5,3));points[:,:,0]=np.arange(6)[:,None]*.01+.001
    original=points.copy();cells=reference_cells(points,accepted)
    assert all(c.tolist()==[[0,0,0],[2,0,0]] for c in cells)
    assert np.array_equal(original,points)
    reference=WorkspaceReference({},cells,dict(samples=6,accepted_samples=2,
        rejected_penetration_samples=1,rejected_invalid_samples=3,
        penetration_threshold_mm=5.,self_collision_filtered=True))
    result=reference.evaluate([Counter({(0,0,0):7,(4,0,0):3})]*5,'thumb')
    assert result['denominator']==DENOMINATOR and result['self_collision_filtered']
    assert result['accepted_samples']==2 and result['rejected_invalid_samples']==3
    assert result['fingers'][0]['coverage_percent']==50
    assert result['fingers'][0]['outside_reference_cells']==1


def test_reference_uses_published_training_file_and_fingerprints_its_contents(tmp_path):
    from retarget.baseline.geort.workspace import training_reference_spec
    from retarget.baseline.geort.adapter import digest
    config=dict(robot={'native':'current','lower':[-1.]*20,'upper':[1.]*20},robot_samples=2)
    model=tmp_path/'models/current';model.mkdir(parents=True)
    (model/'config.json').write_text(json.dumps(config))
    folder=tmp_path/'fk'/digest(config['robot']);folder.mkdir(parents=True)
    labels=folder/'kinematics_2.npz'
    np.savez(labels,qpos=np.zeros((2,20)),keypoints=np.zeros((2,5,3)))
    before=labels.read_bytes();spec=training_reference_spec(tmp_path)
    assert Path(spec['labels'])==labels and spec['samples']==2
    assert spec['policy']['threshold_mm']==5 and spec['policy']['dense']
    assert before==labels.read_bytes()
    config['output_domain']=dict(lower=[-.1]*20,upper=[.5]*20)
    (model/'config.json').write_text(json.dumps(config))
    restricted=training_reference_spec(tmp_path)
    assert restricted['output_domain']!=spec['output_domain']
    assert restricted['labels_sha256']==spec['labels_sha256']
    np.savez(labels,qpos=np.ones((2,20)),keypoints=np.zeros((2,5,3)))
    assert training_reference_spec(tmp_path)['labels_sha256']!=spec['labels_sha256']
    labels.unlink()
    with pytest.raises(FileNotFoundError):training_reference_spec(tmp_path)


def test_workspace_without_training_samples_does_not_use_old_reference(tmp_path):
    from moxian_web.geort_service import GeoRTService
    service=GeoRTService(tmp_path)
    try:
        service.latest=({'source':{'serial':'hand'}},{})
        service.workspace_reference=WorkspaceReference({},[[]]*5)
        service.poll_workspace()
        assert service.workspace_reference is None
        assert service.workspace_status['status']=='waiting_training_samples'
    finally:service.close()


def test_service_invalidates_reference_when_published_fk_file_changes(tmp_path,monkeypatch):
    from moxian_web.geort_service import GeoRTService
    from retarget.baseline.geort.workspace import training_reference_spec,REVISION
    from retarget.baseline.geort.adapter import digest
    import moxian_web.geort_service as module
    launched=[]
    class Process:
        def __init__(self,args,**kwargs):launched.append(args)
        def poll(self):return None
        def terminate(self):pass
    monkeypatch.setattr(module.subprocess,'Popen',Process)
    def setup_config(robot):
        robot=dict(robot,lower=[-1.]*20,upper=[1.]*20)
        config=dict(robot=robot,robot_samples=2)
        model=tmp_path/'models/current';model.mkdir(parents=True,exist_ok=True)
        (model/'config.json').write_text(json.dumps(config))
        folder=tmp_path/'fk'/digest(robot);folder.mkdir(parents=True)
        labels=folder/'kinematics_2.npz'
        np.savez(labels,qpos=np.zeros((2,20)),keypoints=np.zeros((2,5,3)))
        return labels
    old_labels=setup_config({'native':'a'})
    source={'serial':'hand'};spec=training_reference_spec(tmp_path)
    folder=tmp_path/'workspace'/digest([source,REVISION,spec]);folder.mkdir(parents=True)
    (folder/'status.json').write_text(json.dumps(dict(status='completed',revision=REVISION,source=source)))
    np.savez(folder/'reference.npz',**{f:np.array([[0,0,0]]) for f in ('thumb','index','middle','ring','pinky')})
    service=GeoRTService(tmp_path)
    try:
        service.latest=({'source':source},{})
        service.poll_workspace()
        assert service.workspace_reference is not None and not launched
        setup_config({'native':'new-model'})
        old_labels.unlink()  # Publishing may remove an obsolete FK cache.
        service.poll_workspace()
        assert service.workspace_folder!=folder and service.workspace_reference is None
        assert len(launched)==1 and service.workspace_status['status']=='starting'
    finally:service.close()


def test_reference_domain_filters_network_limits_not_only_native_limits():
    from retarget.baseline.geort.workspace import domain_poses,accepted_poses
    q=np.zeros((5,20),np.float32);q[1,6]=-.4;q[2,6]=-.1;q[3,6]=.5;q[4,0]=np.nan
    limits=dict(lower=[-.1]*20,upper=[.5]*20)
    # A collision-free pose with excessive extension is still invalid for IK.
    combined=domain_poses(q,limits)&accepted_poses(np.zeros(5),np.ones(5,bool))
    assert combined.tolist()==[True,False,True,True,False]
    with pytest.raises(ValueError):domain_poses(q,dict(lower=[0]*20,upper=[0]*20))


def test_exact_reference_cloud_and_independent_recorded_cloud():
    from retarget.baseline.geort.workspace import reference_cells
    source={'serial':'hand'};points=np.tile([[[.0012,.0034,.0041]]],(1,5,1)).astype(np.float32)
    cells=reference_cells(points,np.array([True]))
    reference=WorkspaceReference(source,cells,points=points)
    actual=reference.evaluate([{}]*5,'thumb')['fingers'][0]['reference_cloud_mm']
    assert np.allclose(actual,np.array(points[:,0])*1000)
    assert not np.allclose(actual,[[2.5,2.5,2.5]])
    recording=WorkspaceCoverage(source);recording.add(np.tile([.1002,.0012,.0022],(5,1)),1)
    cloud=recording.summary(include_cloud=True)['fingers'][0]['cloud_mm']
    assert cloud==[[100.5,1.5,2.5]]  # Recorded points outside the robot reference survive.


def test_clickable_reference_keeps_exact_fk_sample_pose_and_finger_coverage():
    from retarget.baseline.geort.adapter import RobotAdapter
    from retarget.baseline.geort.workspace import kinematic_spec,reference_cells
    import subprocess
    robot=RobotAdapter()
    q=np.stack([(robot.lower+robot.upper)/2,robot.lower*.4+robot.upper*.6])
    code="""import {workspaceFrames} from './moxian_web/static/geort-workspace-fk.js';
let text='';for await(const chunk of process.stdin)text+=chunk;
const {spec,poses}=JSON.parse(text);console.log(JSON.stringify(poses.map(q=>workspaceFrames(spec,q))));"""
    result=subprocess.run(['node','--input-type=module','-e',code],cwd=Path(__file__).resolve().parents[1],
        input=json.dumps(dict(spec=kinematic_spec(robot),poses=q.tolist())),text=True,capture_output=True,check=True)
    browser_frames=json.loads(result.stdout)
    points=[]
    for i,value in enumerate(q):
        robot.fk(value)
        expected=(robot.data.site_xpos[robot.tip_ids]-robot.data.xpos[robot.body_ids[0]])@robot.basis
        # The browser computes FK only for the selected pose. Compare against
        # native MuJoCo, including every link frame and tip site.
        p=np.array(browser_frames[i]['link_positions_m']);r=np.array(browser_frames[i]['link_rotations'])
        native_p=robot.data.xpos[robot.body_ids]
        np.testing.assert_allclose(p,(native_p-native_p[0])@robot.basis,atol=1e-12)
        np.testing.assert_allclose(r,robot.basis.T@robot.data.xmat[robot.body_ids].reshape(-1,3,3),atol=1e-12)
        found=[]
        for f,site in enumerate(robot.tip_ids):
            owner=robot.body_ids.index(int(robot.model.site_bodyid[site]))
            found.append(p[owner]+r[owner]@robot.model.site_pos[site])
        np.testing.assert_allclose(found,expected,atol=1e-12)
        points.append(expected)
    points=np.asarray(points);cells=reference_cells(points,np.ones(2,bool))
    ref=WorkspaceReference({},cells,dict(reference_id='reference'),points,np.array([7,99]),q)
    counts=[{} for _ in range(5)]
    cell=tuple(np.floor(points[0,0]/.005+1e-7).astype(int));counts[0][cell]=4
    cloud=ref.evaluate(counts,'thumb')['fingers'][0]
    assert cloud['reference_sample_ids']==[7,99]
    pose=ref.pose(7,'thumb',counts)
    np.testing.assert_array_equal(pose['joint_positions_rad'],q[0])
    np.testing.assert_array_equal(pose['point_mm'],cloud['reference_cloud_mm'][0])
    assert pose['cell_observed'] and pose['recorded_cell_frames']==4
    assert not ref.pose(7,'index',counts)['cell_observed']
    assert pose['coverage_scope']=='selected_finger_cell_only'
    assert 'link_positions_m' not in pose and 'link_rotations' not in pose
    for sample in (0,8,True,'7'):
        with pytest.raises(ValueError):ref.pose(sample,'thumb',counts)


def test_workspace_preview_rejects_stale_reference_and_leaves_recording_untouched(tmp_path):
    from moxian_web.geort_service import GeoRTService
    source={'serial':'hand'};folder=tmp_path/'workspace/reference';folder.mkdir(parents=True)
    (folder/'robot_topology.json').write_text('{"ready":true}')
    points=np.zeros((1,5,3));q=np.zeros((1,20))
    ref=WorkspaceReference(source,[np.zeros((1,3),int)]*5,dict(reference_id='reference'),points,np.array([5]),q)
    service=GeoRTService(tmp_path)
    try:
        service.latest=({'source':source},{})
        service.workspace_reference=ref;service.workspace_folder=folder
        service.recording=dict(active=True,frames=42)
        assert service.workspace_preview('reference',5,'thumb')['sample_id']==5
        assert service.workspace_preview('reference')['ready']
        assert service.recording==dict(active=True,frames=42)
        with pytest.raises(ValueError,match='已更新'):service.workspace_preview('old',5,'thumb')
        service.latest=({'source':{'serial':'other'}},{})
        with pytest.raises(ValueError,match='已更新'):service.workspace_preview('reference',5,'thumb')
    finally:service.close()


def test_preview_subset_indexes_original_points_instead_of_preview_row():
    points=np.zeros((5,5,3));points[:,0,0]=np.arange(5)*.01
    q=np.zeros((2,20));q[1]=.2
    ref=WorkspaceReference({},[np.zeros((1,3),int)]*5,dict(reference_id='ref'),
        points,np.array([10,20,30,40,50]),q,
        preview_sample_indices=np.array([10,50]))
    pose=ref.pose(50,'thumb')
    assert pose['point_mm']==[40.,0.,0.]
    assert pose['joint_positions_rad']==[.2]*20
    with pytest.raises(ValueError):ref.pose(20,'thumb')
