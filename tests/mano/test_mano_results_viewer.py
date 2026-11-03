"""Smoke real solved snapshots and the actual Viser scene/GUI API on the remote host."""
from pathlib import Path
from types import SimpleNamespace
import json,hashlib,time
import numpy as np
import pytest
from retarget.viewer.mano_results import *
from scripts.serve_mano_results import Viewer,closure_markdown,pose_markdown,palm_camera
pytestmark=pytest.mark.skipif(not DEFAULT_RESULTS.exists(),reason='Remote solved result fixture unavailable')

@pytest.fixture(scope='module')
def index():return SolvedIndex(DEFAULT_RESULTS).refresh()
@pytest.fixture(scope='module')
def row(index):
    refs,_=index.page(quality='几何达标',limit=1)
    if not refs:pytest.skip('No geometry-accepted example yet')
    return snapshot(refs[0]['path']).row(refs[0]['index'])

def test_pagination_and_filter(index):
    first,n=index.page(limit=100);second,_=index.page(offset=100,limit=100)
    assert n>=200 and len(first)==len(second)==100
    assert {(r['path'],r['index']) for r in first}.isdisjoint({(r['path'],r['index']) for r in second})
    yes,count=index.page(quality='几何达标',limit=7)
    assert count>0 and all(snapshot(r['path']).summary(r['index'])['geometry_accepted'] for r in yes)
    no,_=index.page(quality='几何未达标',limit=7)
    assert all(not snapshot(r['path']).summary(r['index'])['geometry_accepted'] for r in no)

def test_world_geometry_and_contacts(row):
    s=row['snapshot'];i=row['index'];v,f=s.mano_mesh(i);ov,of=s.object_mesh(i);c=s.contacts(i)
    assert v.shape==(778,3) and f.shape[1]==3 and np.isfinite(ov).all()
    reconstructed=np.einsum('ni,nij->nj',c['object_barycentric'],ov[of[c['object_face_ids']]])
    np.testing.assert_allclose(reconstructed,c['object_points_world'],atol=2e-6)
    np.testing.assert_allclose(v[c['hand_vertex_ids']],c['hand_points_world'],atol=1e-7)
    ids=s.poses['mapped_mano_face_ids'];b=s.poses['mapped_mano_barycentric']
    np.testing.assert_allclose(np.einsum('ni,nij->nj',b,v[f[ids]]),s.poses['mapped_hand_points'][i],atol=1e-7)

def test_camera_and_honest_labels(row):
    position,target,up=palm_camera(row)
    assert all(np.isfinite(a).all() for a in (position,target,up))
    assert np.linalg.norm(position-target)>.1 and abs(np.linalg.norm(up)-1)<1e-6
    assert '未评估' in closure_markdown(None)
    assert '自碰撞／关节限位：未评估' in pose_markdown(row)

def test_repository_closure_cached_witness(row):
    cache=ClosureCache();s=row['snapshot'];before=sha(s.path/'poses.npz')
    report=cache.evaluate(row)
    assert report['source_pose_modified'] is False and report['dynamic_evaluated'] is False
    assert report['projected']['witness_audit_passed']
    assert report==cache.cached(row)
    witness=cache.witness(row);m=report['projected']['closure_model']
    forces=witness['closure_response_forces_n'];torques=witness['closure_response_intrinsic_torques_nm']
    points=s.poses['mapped_object_points'][row['index']]
    wrench=np.concatenate((forces.sum(1),(np.cross(points[None]-m['center'],forces)+torques).sum(1)/m['radius_m']),axis=-1)
    residual=np.linalg.norm(wrench/m['test_force_n']+np.concatenate((np.zeros((1,6)),np.eye(6),-np.eye(6))),axis=-1)
    if report['projected']['certified']:assert residual[1:].max()<=m['certificate_tolerance']
    assert before==sha(s.path/'poses.npz')

def test_cache_rejects_changed_pose(row,tmp_path):
    cache=ClosureCache(tmp_path);path=cache.cache_path(row);write(path,dict(fingerprint='wrong'))
    assert cache.cached(row) is None

def test_real_viser_gui_scene_smoke(index,row):
    import viser
    server=viser.ViserServer(host='127.0.0.1',port=0,verbose=False)
    view=Viewer(server,index,ClosureCache())
    try:
        fake=SimpleNamespace(scene=server.scene,gui=server.gui,camera=SimpleNamespace(),atomic=server.atomic)
        state=view.connect(fake,start_timer=False,auto_closure=False)
        assert state['row'] is not None and state['handles']
        for key in ('near','normals','shadow'):
            state['controls'][key].value=True
        state['render']()
        state['row']=row;state['render']();state['camera']()
        state['controls']['forces'].value=True;state['controls']['scenario'].value=state['controls']['scenario'].options[4]
        state['render']()
        assert len(state['handles'])>=12
        state['closed']=True
    finally:
        view.executor.shutdown(wait=True,cancel_futures=True);server.stop()
