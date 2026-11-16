import json
import hashlib
import time
import numpy as np
import pytest
from scipy.spatial import ConvexHull
from moxian_web.cart import fingertip_hull,CartEstimator


def test_volume_faces_and_area_match_qhull():
    rng=np.random.default_rng(822)
    for _ in range(80):
        p=rng.normal(size=(5,3))*.06
        actual=fingertip_hull(p);reference=ConvexHull(p)
        assert actual['dimension']==3 and actual['solid']
        assert actual['volume_m3']==pytest.approx(reference.volume,rel=1e-10)
        assert actual['surface_area_m2']==pytest.approx(reference.area,rel=1e-10)
        assert actual['volume_ml']==pytest.approx(reference.volume*1e6)
        for face in actual['faces']:
            a,b,c=p[face];normal=np.cross(b-a,c-a)
            assert ((p-a)@normal).max()<1e-12


def test_coplanar_facets_have_no_overlapping_triangles():
    p=np.array([[0,0,0],[.1,0,0],[.1,.1,0],[0,.1,0],[.05,.05,.1]])
    hull=fingertip_hull(p)
    assert len(hull['faces'])==6 and len(hull['edges'])==8
    assert hull['volume_m3']==pytest.approx(.1**3/3)
    np.testing.assert_allclose(hull['surface_area_m2'],ConvexHull(p).area)


def test_flat_collinear_and_coincident_tips_have_no_artificial_volume():
    square=np.array([[0,0,0],[.1,0,0],[.1,.1,0],[0,.1,0],[.05,.05,0]])
    flat=fingertip_hull(square)
    assert flat['dimension']==2 and flat['status']=='planar' and not flat['solid']
    assert flat['volume_ml']==0 and flat['surface_area_m2']==pytest.approx(.01)
    assert flat['hull_vertex_count']==4 and len(flat['faces'])==2
    line=fingertip_hull(np.c_[np.linspace(0,.1,5),np.zeros((5,2))])
    assert line['dimension']==1 and len(line['edges'])==1 and line['faces']==[]
    point=fingertip_hull(np.zeros((5,3)))
    assert point['dimension']==0 and point['volume_m3']==0 and point['edges']==[]


def test_current_frame_only_and_rigid_invariance():
    from scipy.spatial.transform import Rotation
    p=np.random.default_rng(4).normal(size=(5,3))*.03
    original=fingertip_hull(p)
    transformed=fingertip_hull(p@Rotation.from_rotvec([.2,.5,.8]).as_matrix().T+[1,2,3])
    assert transformed['volume_m3']==pytest.approx(original['volume_m3'],rel=1e-10)
    assert transformed['surface_area_m2']==pytest.approx(original['surface_area_m2'],rel=1e-10)
    shrunk=fingertip_hull(p*.5)
    assert shrunk['volume_m3']==pytest.approx(original['volume_m3']/8)
    assert not shrunk['temporal_accumulation']
    with pytest.raises(ValueError):fingertip_hull(np.full((5,3),np.nan))
    with pytest.raises(ValueError):fingertip_hull(np.zeros((4,3)))


@pytest.mark.parametrize('capture_mode',['cart','cart_v3'])
def test_cart_motion_recording_keeps_matching_per_frame_hulls(tmp_path,monkeypatch,capture_mode):
    import moxian_web.geort_service as service
    monkeypatch.setattr(service,'ROOT',tmp_path)
    serial='test-cart';home=tmp_path/'output/moxian/hand_model'/hashlib.sha256(serial.encode()).hexdigest()[:16]/'qnbot'
    home.mkdir(parents=True);(home/'current.json').write_text('{"model_hash":"model"}')
    topology=dict(surface_backend='qnbot',coordinate_frame='wrist_local',hand_side='right',geometry_version='g',
        binding_version='b',model_hash='model',palm_view=dict(normal=[1,0,0],forward=[0,0,1]),
        joint_chains=[[0,1+4*f,2+4*f,3+4*f,4+4*f] for f in range(5)])
    frame=dict(ready=True,calibrated=True,geometry_version='g',binding_version='b',model_hash='model',
        hand_side='right',calibration_fingerprint='c',skeleton_runtime_revision='r',source_joint_positions_rad=[0.]*29,
        timestamp_ms=time.time()*1000)
    joints=np.random.default_rng(6).normal(size=(21,3))*.04
    estimator=CartEstimator();recorder=service.GeoRTService(tmp_path/'geort')
    def feed(f,j):
        hull=estimator.observe(f,topology,j,serial)
        recorder.observe(f,topology,j,serial,cart_hull=hull if capture_mode=='cart' else None)
    try:
        feed(frame,joints);started=recorder.start_recording('cart',capture_mode=capture_mode,metadata=dict(protocol='cart_contact_v3',stage='precision_grasp',category=2) if capture_mode=='cart_v3' else None)
        for i in range(5):feed({**frame,'timestamp_ms':frame['timestamp_ms']+10*i},joints*(1+i*.1))
        recorder.stop_recording()
        for _ in range(100):
            if not recorder.snapshot()['recording']['saving']:break
            time.sleep(.01)
        folder=recorder.root/'recordings'/started['id']
        manifest=json.loads((folder/'manifest.json').read_text());assert manifest['capture_mode']==capture_mode and manifest['frames']==5
        rows=[json.loads(line) for line in (folder/'frames.jsonl').read_text().splitlines()]
        if capture_mode=='cart':
            for row in rows:
                assert row['cart_hull']['timestamp_ms']==row['timestamp_ms']
                assert row['cart_hull']['coordinate_frame']=='wrist_local'
            assert rows[-1]['cart_hull']['volume_m3']>rows[0]['cart_hull']['volume_m3']
        else:
            assert manifest['protocol']=='cart_contact_v3'
            assert all('cart_hull' not in row and len(row['qpos'])==29 for row in rows)
            np.testing.assert_allclose([r['timestamp_ms'] for r in rows],frame['timestamp_ms']+np.arange(5)*10)
        from retarget.baseline.geort.training import load_recordings
        points,source,records=load_recordings(recorder.root,[started['id']]);assert points.shape==(5,5,3)
        assert not estimator.snapshot(False)['ready']
    finally:recorder.close()
