"""Coordinate regression, native GPU FK, and voxel geometric parity."""
import numpy as np
import pytest
from retarget.baseline.geort.adapter import SourceAdapter, RobotAdapter, anatomical_basis


def test_mcp_frame_ignores_bent_distal_display_normal_and_mirrors_left():
    from tests.test_geort import topology
    right=topology();p=np.zeros((21,3))
    p[5]=[0,.03,.07];p[9]=[0,0,.08];p[13]=[0,-.03,.07]
    p[[4,8,12,16,20]]=[[.01,.05,.12],[0,.03,.17],[0,0,.18],[0,-.03,.17],[0,-.05,.14]]
    right['rest_joints_m']=p.tolist()
    right['palm_view']['normal']=[.5,0,-.8660254]
    a=SourceAdapter(right)
    np.testing.assert_allclose(a.basis,np.eye(3),atol=1e-7)
    np.testing.assert_allclose(a.points(p),p[[4,8,12,16,20]],atol=1e-7)
    left=topology('left');left['rest_joints_m']=(p*[1,-1,1]).tolist()
    np.testing.assert_allclose(SourceAdapter(left).points(p*[1,-1,1]),a.points(p),atol=1e-7)
    for i in (8,12,16,20):p[i]=[.1,0,.06]
    np.testing.assert_allclose(anatomical_basis(p),np.eye(3),atol=1e-7)


def test_gpu_fk_matches_mujoco_native_ranges_and_flexion_direction():
    import torch
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from retarget.baseline.geort.gpu import RobotFK
    r=RobotAdapter();q=np.random.default_rng(714).uniform(r.lower,r.upper,(128,20))
    actual=RobotFK(r)(q).cpu().numpy();expected=np.stack([r.fk(v) for v in q])
    np.testing.assert_allclose(actual,expected,atol=1e-7,rtol=0)
    zero=r.fk(np.zeros(20))
    for finger in range(1,5):
        q=np.zeros(20);q[4*finger+2]=.2
        assert (r.fk(q)-zero)[finger,0]>0 # positive flex moves toward palmar X


def test_cuda_voxel_centroids_match_open3d_including_negative_bounds():
    import torch
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    import open3d as o3d
    from scipy.spatial import cKDTree
    from retarget.baseline.geort.gpu import voxel_centroids,human_dataset
    p=np.random.default_rng(714).normal(0,.004,(2048,3)).astype(np.float32)
    cloud=o3d.geometry.PointCloud();cloud.points=o3d.utility.Vector3dVector(p)
    expected=np.asarray(cloud.voxel_down_sample(.001).points)
    actual=voxel_centroids(p).cpu().numpy()
    assert len(actual)==len(expected)
    assert cKDTree(expected).query(actual)[0].max()<1e-12
    np.random.seed(12);ds=human_dataset(np.repeat(p[:,None],5,axis=1),128)
    assert ds.points.shape==(5,128,3)
    assert not np.array_equal(ds.points[0],ds.points[1])


def test_legacy_recording_is_reprojected_without_modifying_raw_files(tmp_path):
    import hashlib,json
    from tests.test_geort import topology
    from retarget.baseline.geort.training import load_recordings
    from retarget.baseline.geort.adapter import REVISION
    t=topology();j=np.zeros((21,3));j[5]=[0,.03,.07];j[9]=[0,0,.08];j[13]=[0,-.03,.07]
    j[[4,8,12,16,20]]=[[.01,.04,.12],[.02,.03,.17],[.02,0,.18],[.02,-.03,.17],[.02,-.05,.14]]
    t['rest_joints_m']=j.tolist();t['palm_view']['normal']=[.5,0,-.8660254]
    folder=tmp_path/'recordings/legacy';folder.mkdir(parents=True)
    old=np.ones((2,5,3),np.float32);np.save(folder/'points.npy',old)
    np.savez(folder/'motion.npz',joints=np.stack((j,j+[1,2,3])),qpos=np.zeros((2,29)))
    (folder/'topology.json').write_text(json.dumps(t))
    checksum=hashlib.sha256((folder/'points.npy').read_bytes()).hexdigest()
    meta=dict(complete=True,frames=2,source=dict(revision='moxian_wuji2_geort_v1',serial='glove'),points_sha256=checksum)
    (folder/'manifest.json').write_text(json.dumps(meta))
    actual,source,records=load_recordings(tmp_path,['legacy'])
    np.testing.assert_allclose(actual[0],SourceAdapter(t).points(j),atol=1e-7)
    np.testing.assert_allclose(actual[0],actual[1],atol=1e-7)
    assert source['revision']==REVISION and records[0]['coordinate_reprojection']['from_revision']==meta['source']['revision']
    assert hashlib.sha256((folder/'points.npy').read_bytes()).hexdigest()==checksum
    assert json.loads((folder/'manifest.json').read_text())==meta
