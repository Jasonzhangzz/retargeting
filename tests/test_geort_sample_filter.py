import json
import numpy as np
import pytest
from retarget.baseline.geort.adapter import RobotAdapter,digest
from retarget.baseline.geort.pose_domain import extension_domain,SAMPLING_POLICY
from retarget.baseline.geort.sample_filter import retained_arrays,geometry_spec,certified_depths,cache_key


def test_bad_whole_poses_are_removed_from_saved_table_without_clipping(tmp_path):
    robot=RobotAdapter();domain=extension_domain(robot.lower,robot.upper)
    q=np.zeros((6,20),np.float32);q[1,7]=-.1;q[2,7]=.2;q[3,7]=.3;q[4,7]=.4;q[5,7]=.5
    p=np.arange(90,dtype=np.float32).reshape(6,5,3)*.001
    depth=np.array([0,0,6,1,np.nan,5],np.float32);valid=np.array([1,1,1,0,1,1],bool)
    arrays,summary=retained_arrays(q,p,domain,depth,valid,geometry_spec(robot.key,domain))
    assert summary['deleted_samples']==4 and summary['retained_samples']==2
    np.testing.assert_array_equal(arrays['qpos'],q[[0,5]])
    np.testing.assert_array_equal(arrays['keypoints'],p[[0,5]])
    path=tmp_path/'labels.npz';np.savez(path,**arrays)
    actual=certified_depths(path,robot.key,domain)
    np.testing.assert_array_equal(actual[0],[0,5]);assert actual[1].all()
    arrays['qpos'][1,7]=-.2;np.savez(path,**arrays)
    with pytest.raises(ValueError,match='违规姿态'):certified_depths(path,robot.key,domain)


def test_filter_policy_changes_cache_identity_and_rejects_foreign_certificates(tmp_path):
    robot=RobotAdapter();domain=extension_domain(robot.lower,robot.upper)
    legacy=digest(dict(robot=robot.spec,sampling_policy=SAMPLING_POLICY,lower=domain['lower'],upper=domain['upper']))
    assert cache_key(robot.spec,domain,SAMPLING_POLICY)!=legacy
    spec=geometry_spec(robot.key,domain);spec['threshold_mm']=100
    path=tmp_path/'labels.npz';np.savez(path,screening_spec=np.array(json.dumps(spec)))
    with pytest.raises(ValueError,match='策略已变化'):certified_depths(path,robot.key,domain)


def test_verified_labels_are_contact_targets_without_second_geometry_query(tmp_path,monkeypatch):
    import retarget.baseline.card.collision as module
    from retarget.baseline.geort.contact_sampling import robot_targets
    robot=RobotAdapter();domain=extension_domain(robot.lower,robot.upper)
    q=np.zeros((2,20),np.float32);p=np.zeros((2,5,3),np.float32);p[1]=.01
    arrays,_=retained_arrays(q,p,domain,np.array([0,1]),np.ones(2,bool),geometry_spec(robot.key,domain))
    path=tmp_path/'labels.npz';np.savez(path,**arrays)
    class Forbidden:
        def __init__(self,*a):raise AssertionError('must reuse certified geometry')
    monkeypatch.setattr(module,'WujiMeshCollision',Forbidden)
    targets,meta=robot_targets(tmp_path,robot,domain,path,'cpu',lambda *a,**k:None,lambda:None)
    np.testing.assert_array_equal(targets,p.transpose(1,0,2));assert meta['accepted_samples']==2
