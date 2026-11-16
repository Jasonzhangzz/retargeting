"""Contact/penetration counterexamples and read-only asynchronous contracts."""
import hashlib
import json
from types import SimpleNamespace
import numpy as np
import pytest
from retarget.baseline.geort.diagnostics import contact_state,source_allowed


def test_zero_unsigned_gap_does_not_hide_penetration_or_wrong_normal():
    assert contact_state(0,7.7,0)=='penetrating'
    assert contact_state(0,0,0)=='contact'
    assert contact_state(1,0,80)=='normal_mismatch'
    assert contact_state(5,0,0)=='separated'
    assert contact_state(0,0,0,False)=='unknown'
    assert contact_state(float('nan'),0,0)=='unknown'


def test_dense_region_distance_tracks_moving_original_triangles_without_editing():
    import torch,trimesh
    from retarget.baseline.geort.diagnostic_queries import SurfacePairs
    a=trimesh.creation.box(extents=[.02]*3);b=a.copy();b.apply_translation([.025,0,0])
    mesh=trimesh.util.concatenate((a,b));v=torch.tensor(np.asarray(mesh.vertices),dtype=torch.float32)
    f=torch.tensor(np.asarray(mesh.faces),dtype=torch.long)
    regions={'a':list(range(12)),'b':list(range(12,24))}
    query=SurfacePairs(v,f,regions,[{'regions':['a','b']}]);before=v.clone()
    gap,_=query.evaluate(v);assert gap[0]==pytest.approx(5.,abs=1e-4)
    torch.testing.assert_close(v,before,rtol=0,atol=0)
    moved=v.clone();moved[8:,0]+=.01
    gap,_=query.evaluate(moved);assert gap[0]==pytest.approx(15.,abs=1e-4)
    torch.testing.assert_close(v,before,rtol=0,atol=0)


def test_skin_seam_exclusions_keep_cross_finger_and_nonadjacent_contacts():
    names=['right_wrist']
    for f in ('thumb','index','middle','ring','little'):
        names += ['right_'+f+'_'+p for p in (('metacarpal','proximal','distal') if f=='thumb' else ('metacarpal','proximal','intermediate','distal'))]
    allowed=source_allowed(names)
    index=lambda name:names.index('right_'+name)
    assert not allowed[index('index_proximal'),index('index_intermediate')]
    assert allowed[index('index_proximal'),index('index_distal')]
    assert allowed[index('thumb_distal'),index('index_distal')]
    np.testing.assert_array_equal(allowed,allowed.T)


def test_diagnostic_expiry_clears_measurements_but_paused_replay_is_static(tmp_path):
    from moxian_web.geort_diagnostics_service import DiagnosticService
    service=DiagnosticService(lambda:None,tmp_path)
    try:
        service.configure(True)
        with service.lock:
            service.result=dict(ready=True,status='ready',input_kind='live',timestamp_ms=0,pairs=[1],source={'max_penetration_mm':4})
            service.received=0.
        stale=service.snapshot();assert stale['status']=='stale' and not stale['ready'] and stale['pairs']==[] and stale['source'] is None
        with service.lock:service.result['input_kind']='replay'
        assert service.snapshot()['ready']
        assert service.configure(False)==dict(enabled=False,ready=False,status='disabled')
        with pytest.raises(ValueError):service.configure('true')
    finally:service.close()


def test_diagnostic_provider_uses_paired_geort_source_not_a_newer_glove(tmp_path,monkeypatch):
    import moxian_web.server as server
    monkeypatch.setattr(server,'OUTPUT',tmp_path)
    serial='device';folder=tmp_path/'hand_model'/hashlib.sha256(serial.encode()).hexdigest()[:16]/'qnbot'
    folder.mkdir(parents=True);(folder/'current.json').write_text(json.dumps({'model_hash':'model'}))
    topology={'geometry_version':'geometry','binding_version':'binding','model_hash':'model'}
    frame=dict(ready=True,solver_backend='geort',source_timestamp_ms=100,geometry_version='geometry',source_binding_version='binding',joint_positions_rad=[0]*20,model_run_id='run')
    def no_newer_source(*a,**k):raise AssertionError('Must use the source carried by this GeoRT result')
    runtime=object.__new__(server.Runtime);runtime.wuji=SimpleNamespace(snapshot=lambda *a,**k:frame)
    runtime.hand=SimpleNamespace(serial=serial,model=lambda:topology,snapshot=no_newer_source)
    runtime.geort_diagnostic_context=None
    packet=runtime.geort_diagnostic_input()
    assert packet['frame'] is frame and packet['timestamp_ms']==100 and packet['robot']


def test_other_solver_is_not_reported_as_a_geort_prediction(tmp_path,monkeypatch):
    import base64,struct
    import moxian_web.server as server
    monkeypatch.setattr(server,'OUTPUT',tmp_path)
    serial='device';folder=tmp_path/'hand_model'/hashlib.sha256(serial.encode()).hexdigest()[:16]/'qnbot'
    folder.mkdir(parents=True);(folder/'current.json').write_text(json.dumps({'model_hash':'model'}))
    top=dict(geometry_version='g',binding_version='b',model_hash='model')
    hand=dict(ready=True,calibrated=True,timestamp_ms=123.,geometry_version='g',binding_version='b',mesh_frame={'sequence':1})
    payload=struct.pack('<4sIIIIId',b'QNB1',1,1,1,1,0,123.)+b'g'.ljust(32,b'\0')+b'b'.ljust(64,b'\0')+bytes(24)
    runtime=object.__new__(server.Runtime);runtime.geort_diagnostic_context=None
    runtime.wuji=SimpleNamespace(snapshot=lambda *a,**k:dict(ready=True,solver_backend='cem',joint_positions_rad=[1]*20))
    runtime.hand=SimpleNamespace(serial=serial,model=lambda:top,snapshot=lambda **k:hand,mesh_frame=lambda **k:payload)
    packet=runtime.geort_diagnostic_input()
    assert not packet['robot'] and 'joint_positions_rad' not in packet['frame']
    assert base64.b64decode(packet['frame']['mesh_f32_b64'])==payload
