"""Recording completeness, physical wrist coordinates and crash-safe geometry."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import threading
import time
import numpy as np
import pytest
import torch
from moxian_web.cart_capture_schema import CAPTURE_SCHEMA,binary_taxels
from moxian_web.cart_contacts import SourceContacts,transform_contacts
from moxian_web.cart_data import SourceGeometry
from moxian_web.cart_capture import capture_recording,capture_current
from moxian_web.cart_protocol import PROTOCOL


@pytest.fixture(autouse=True)
def one_thread():torch.set_num_threads(1)


def test_binary_taxels_preserve_bits_and_distinguish_unknown():
    topology=dict(sid_face_ids=[0]*460,sid_patch_indices=[0]*460)
    frame=dict(timestamp_ms=100.,tactile_timestamp_ms=100.,source_contact=dict(ready=True,timestamp_ms=100.,cell_contact=[1,0]*230))
    bits,valid,bound=binary_taxels(frame,topology)
    assert bits.dtype==np.uint8 and bits.sum()==230 and valid.all() and bound.all()
    frame['tactile_timestamp_ms']=70.
    old,valid,_=binary_taxels(frame,topology);assert not valid.any();np.testing.assert_array_equal(old,bits)
    frame['source_contact']['cell_contact']=None
    bits,valid,_=binary_taxels(frame,topology);assert not bits.any() and not valid.any()


def contact_fixture():
    import trimesh
    meshes=[]
    for x in (-.1,0.,.015):
        mesh=trimesh.creation.box(extents=(.02,.02,.02));mesh.apply_translation((x,0,0));meshes.append(mesh)
    vertices=np.vstack([m.vertices for m in meshes]).astype(np.float32)
    faces=np.vstack([m.faces+8*i for i,m in enumerate(meshes)])
    class Model:
        bone_names=['right_wrist','right_index_proximal','right_middle_proximal']
        weights=torch.tensor(np.repeat(np.eye(3),8,axis=0),dtype=torch.float32)
        q0=torch.zeros(1)
        def __call__(self,q):return torch.tensor(vertices)[None],None,None
    topology=dict(hand_side='right',faces=faces.tolist(),surface_policy='qnbot_original_triangles',
        sid_face_ids=[12]*460,sid_barycentric=[[1/3]*3]*460,sid_patch_indices=[0]*460)
    detector=SourceContacts(topology,Model())
    emf=dict(timestamp_ms=100.,coordinate_frame='wrist_local',positions_m=[[.01,.02,.03]]*5,
        rotations=np.tile(np.eye(3),(5,1,1)).tolist(),raw_positions_m=[[.01,.02,.03]]*5,
        raw_rotations=np.tile(np.eye(3),(5,1,1)).tolist(),valid=[True]*5)
    bits=[False]*460;bits[0]=True;bits[2]=True
    frame=dict(timestamp_ms=100.,tactile_timestamp_ms=100.,emf_observation=emf,
        source_contact=dict(ready=True,timestamp_ms=100.,cell_contact=bits,normalized=[.2]*460))
    return detector,vertices,faces,topology,frame


def test_all_penetration_witnesses_are_saved_in_both_directions():
    detector,v,faces,topology,frame=contact_fixture()
    evidence=detector.detect(v,np.zeros((21,3)),frame,include_all=True)
    samples=evidence['self_penetrations']
    assert len(evidence['self_pairs'])==1 and samples['count']>1 and samples['truncated'] is False
    assert set(zip(samples['source_regions'],samples['target_regions']))=={
        ('index_proximal','middle_proximal'),('middle_proximal','index_proximal')}
    np.testing.assert_allclose(np.asarray(samples['positions_m']),
        (v[np.asarray(samples['source_vertices'])]*np.asarray(samples['source_barycentric'])[...,None]).sum(1),atol=1.e-8)
    np.testing.assert_allclose(np.asarray(samples['surface_positions_m']),
        (v[faces[np.asarray(samples['target_face_indices'])]]*np.asarray(samples['target_barycentric'])[...,None]).sum(1),atol=1.e-8)
    distance=np.linalg.norm(np.asarray(samples['positions_m'])-samples['surface_positions_m'],axis=-1)
    np.testing.assert_allclose(samples['depths_m'],distance,atol=1.e-7)
    root=np.array([.1,.2,.3]);a=np.diag([-1.,1.,1.])
    restored=transform_contacts(transform_contacts(evidence,a,root),a,root,True)
    np.testing.assert_allclose(restored['self_penetrations']['positions_m'],samples['positions_m'],atol=1.e-8)


def test_capture_has_all_taxels_and_unmirrored_wrist_points():
    detector,v,faces,topology,frame=contact_fixture()
    geometry=SourceGeometry.__new__(SourceGeometry);geometry.detector=detector;geometry.topology=topology
    root=np.array([.001,.002,.003],np.float32);joints=np.broadcast_to(root,(21,3)).copy()
    geometry.adapter=SimpleNamespace(indices=np.arange(21),transform=np.diag([-1.,1.,1.]))
    geometry.reconstruct=lambda q:(v,joints)
    capture=geometry.capture_frame([0.],100.,frame)
    positions,normals=detector.taxels.evaluate(v)
    assert capture['schema']==CAPTURE_SCHEMA and capture['coordinate_frame']=='wrist_local' and capture['units']=='m'
    assert len(capture['tactile_state'])==460 and set(capture['tactile_state'])=={0,1}
    assert capture['tactile_signal_sids']==[0,2]
    np.testing.assert_allclose(capture['tactile_positions_m'],positions-root)
    np.testing.assert_allclose(capture['tactile_signal_positions_m'],(positions-root)[[0,2]])
    row=SourceGeometry.row_from_capture(capture,geometry.adapter)
    np.testing.assert_allclose(row['tactile_positions_m'],(positions-root)@geometry.adapter.transform)
    assert capture['contacts']['self_penetrations']['count']>1
    assert 'self_penetrations' not in row['contacts']  # Full journal remains intact, training stays bounded.


def recording_fixture(tmp_path,monkeypatch,complete=True):
    import moxian_web.cart_data as data
    import moxian_web.source_ik_initialization as initialization
    folder=tmp_path/'recordings/test';folder.mkdir(parents=True)
    source=dict(calibration_fingerprint='c',skeleton_runtime_revision='s',hand_side='right')
    meta=dict(id='test',protocol=PROTOCOL,complete=complete,frames=3 if complete else 0,source=source)
    (folder/'manifest.json').write_text(json.dumps(meta));(folder/'profile.json').write_text('{}');(folder/'topology.json').write_text('{}')
    rows=[dict(timestamp_ms=float(i*20),qpos=[0.]*29,joints_m=np.zeros((21,3)).tolist(),
        emf_observation=dict(calibration_fingerprint='c',skeleton_runtime_revision='s')) for i in range(3)]
    (folder/'frames.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows) if complete else '')
    calls=[]
    class Geometry:
        adapter=SimpleNamespace(indices=np.arange(21))
        def __init__(self,*args):pass
        def capture_frame(self,q,stamp,obs):
            calls.append(stamp)
            return dict(schema=CAPTURE_SCHEMA,timestamp_ms=stamp,source_wrist_origin_m=[0.,0.,0.],
                coordinate_frame='wrist_local',units='m',joints_m=np.zeros((21,3)).tolist(),tactile_signal_sids=[1],
                contacts=dict(self_penetrations=dict(count=2)))
    monkeypatch.setattr(data,'SourceGeometry',Geometry)
    monkeypatch.setattr(initialization,'validate_recorded_source',lambda *args:None)
    return folder,meta,rows,calls,Geometry


def test_capture_complete_cache_and_raw_immutability(tmp_path,monkeypatch):
    folder,meta,rows,calls,Geometry=recording_fixture(tmp_path,monkeypatch)
    original=(folder/'frames.jsonl').read_bytes()
    updates=[]
    def progress(phase,**values):
        # Exercise the real CLI/trainer callback contract, not only **kwargs.
        state=dict(id='job',pid=123,status='running',phase=phase,elapsed_s=0.,**values)
        assert state['id']=='job' and state['pid']==123
        updates.append((phase,values))
    result=capture_recording(tmp_path,'test',device='cpu',progress=progress)
    assert result['frames']==3 and result['penetration_samples_processed']==6 and result['signal_points_processed']==3
    assert capture_current(folder) and calls==[0.,20.,40.]
    capture_recording(tmp_path,'test',device='cpu');assert calls==[0.,20.,40.]
    assert (folder/'frames.jsonl').read_bytes()==original
    assert updates[0][0]=='capture_geometry' and updates[0][1]['capture_status']=='running'
    assert updates[0][1]['capture_phase']=='initializing'
    assert updates[-1][1]['capture_status']=='completed'


def test_capture_recovers_torn_tail_without_dropping_raw_frames(tmp_path,monkeypatch):
    folder,meta,rows,calls,Geometry=recording_fixture(tmp_path,monkeypatch)
    original=Geometry.capture_frame
    def fail(self,q,stamp,obs):
        if stamp==20.:raise InterruptedError('fixture interruption')
        return original(self,q,stamp,obs)
    monkeypatch.setattr(Geometry,'capture_frame',fail)
    with pytest.raises(InterruptedError):capture_recording(tmp_path,'test',device='cpu')
    with (folder/'cart_capture.pending.jsonl').open('ab') as stream:stream.write(b'{"timestamp_ms":')
    monkeypatch.setattr(Geometry,'capture_frame',original)
    result=capture_recording(tmp_path,'test',device='cpu')
    assert result['resumed_prefix_frames']==1 and calls==[0.,20.,40.]
    assert [json.loads(line)['timestamp_ms'] for line in (folder/'cart_capture.jsonl').open()]==[0.,20.,40.]
    assert not (folder/'cart_capture.pending.jsonl').exists()


def test_follower_catches_up_after_recording_stops(tmp_path,monkeypatch):
    folder,meta,rows,calls,Geometry=recording_fixture(tmp_path,monkeypatch,complete=False)
    errors=[]
    def run():
        try:capture_recording(tmp_path,'test',follow=True,device='cpu')
        except BaseException as error:errors.append(error)
    thread=threading.Thread(target=run,daemon=True);thread.start()
    with (folder/'frames.jsonl').open('a') as stream:
        for row in rows:stream.write(json.dumps(row)+'\n');stream.flush()
    from retarget.baseline.geort.training import atomic_json
    atomic_json(folder/'manifest.json',dict(meta,complete=True,frames=3))
    thread.join(timeout=5)
    assert not thread.is_alive() and not errors and calls==[0.,20.,40.]
    assert json.loads((folder/'cart_capture_status.json').read_text())['status']=='completed'


def test_service_starts_independent_geometry_capture_with_new_recording(tmp_path,monkeypatch):
    from moxian_web.cart_service import CartService
    service=CartService(tmp_path)
    try:
        service.recorder.latest=({'emf_observation':{'valid':[True]*5}}, {})
        captured={};started=[]
        def record(label,capture_mode,metadata):
            captured.update(mode=capture_mode,metadata=metadata);return dict(id='new_record')
        monkeypatch.setattr(service.recorder,'start_recording',record)
        monkeypatch.setattr(service,'start_preview',lambda:None)
        monkeypatch.setattr(service,'_start_geometry_capture',lambda key:started.append(key))
        result=service.start_recording(dict(split='train'))
        assert captured['mode']=='cart_v5' and captured['metadata']['capture_geometry_schema']==CAPTURE_SCHEMA
        assert captured['metadata']['coordinate_frame']=='wrist_local' and started==[result['id']]
    finally:service.close()


def test_cancelled_geometry_is_not_automatically_restarted(tmp_path,monkeypatch):
    from moxian_web.cart_service import CartService
    folder=tmp_path/'recordings/cancelled';folder.mkdir(parents=True)
    (folder/'manifest.json').write_text(json.dumps(dict(complete=True,protocol=PROTOCOL,capture_geometry_schema=CAPTURE_SCHEMA)))
    (folder/'cart_capture_status.json').write_text(json.dumps(dict(status='cancelled',frames=3)))
    service=CartService(tmp_path)
    try:
        started=[];monkeypatch.setattr(service,'_start_geometry_capture',lambda key:started.append(key))
        service._recover_geometry();assert not started and service.capture_proc is None
    finally:service.close()
