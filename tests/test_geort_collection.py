"""Collection guidance and destructive lifecycle operations on temporary data."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import threading
import numpy as np
import pytest
from retarget.baseline.geort.coverage import WorkspaceCoverage
from moxian_web.geort_service import GeoRTService
from moxian_web.saved_assets import asset_lock


def recording(root, key, points, source=None):
    folder=root/'recordings'/key;folder.mkdir(parents=True)
    np.save(folder/'points.npy',points)
    np.save(folder/'timestamps.npy',np.arange(len(points))*10.)
    (folder/'manifest.json').write_text(json.dumps(dict(id=key,complete=True,frames=len(points),
        source=source or {'serial':'glove'},points_sha256=hashlib.sha256((folder/'points.npy').read_bytes()).hexdigest())))
    return folder


def test_occupancy_does_not_reward_repeated_frames_and_tracks_three_dimensions():
    c=WorkspaceCoverage({'serial':'glove'});p=np.zeros((5,3))
    for i in range(100):c.add(p,i*10.,now=i*.01)
    before=c.summary(now=1)
    assert before['frames']==100
    assert [f['unique_voxels'] for f in before['fingers']]==[1]*5
    assert not before['absolute_workspace_percentage']
    for i in range(1,11):c.add(p+np.array([i*.001,i*.002,i*.003]),1000+i*10.,now=1+i*.01)
    result=c.summary(include_cloud=True,now=1.2)
    assert result['fingers'][0]['unique_voxels']==11
    assert result['fingers'][0]['span_mm']==[10.,20.,30.]
    assert len(result['fingers'][0]['cloud_mm'])>1
    assert c.summary(now=10)['fingers'][0]['added_last_5s']==0
    assert c.add(p,1100.,now=10) is False
    with pytest.raises(ValueError):c.add(np.full((5,3),np.nan),1200.)


def test_pinch_counter_has_hold_hysteresis_and_breaks_on_stream_gaps():
    c=WorkspaceCoverage({});p=np.zeros((5,3));p[1:,0]=.05
    c.add(p,0,now=0)
    p[1,0]=.01
    for stamp in (10,60,120):c.add(p,stamp,now=stamp/1000)
    assert c.summary(now=.12)['pinches'][0]['episodes']==1
    p[1,0]=.017;c.add(p,130,now=.13)
    p[1,0]=.01;c.add(p,140,now=.14)
    assert c.summary(now=.14)['pinches'][0]['episodes']==1
    p[1,0]=.03;c.add(p,150,now=.15)
    p[1,0]=.01
    for stamp in (160,220,280):c.add(p,stamp,now=stamp/1000)
    assert c.summary(now=.28)['pinches'][0]['episodes']==2
    c.add(p,1000,now=1)
    assert c.summary(now=1)['pinches'][0]['episodes']==2
    assert c.summary(now=1)['pinches'][0]['seconds']<.5


def test_baseline_union_compatibility_and_delete_invalidation(tmp_path):
    points=np.arange(30,dtype=np.float32).reshape(2,5,3)*.001
    recording(tmp_path,'a',points);recording(tmp_path,'b',points)
    service=GeoRTService(tmp_path)
    try:
        result=service.select_coverage(['a','b'])
        assert result['frames']==4
        assert [f['unique_voxels'] for f in result['fingers']]==[2]*5
        service.latest=({'source':{'serial':'other'}},{})
        with pytest.raises(ValueError,match='不兼容'):service.select_coverage(['a'])
        assert service.coverage_snapshot()['compatible'] is False
        service.latest=None
        service.delete_assets('recordings',['a'])
        assert (tmp_path/'recordings/b/points.npy').exists()
        assert not service.coverage_snapshot()['ready']
        assert '已删除' in service.coverage_snapshot()['notice']
    finally:service.close()


def test_bulk_delete_validates_all_ids_and_guards_links_and_live_data(tmp_path):
    points=np.zeros((2,5,3),np.float32);folder=recording(tmp_path,'a',points)
    service=GeoRTService(tmp_path)
    try:
        for ids in (['a','missing'],['a','a'],['../escape'],[],['/tmp']):
            with pytest.raises(ValueError):service.delete_assets('recordings',ids)
            assert folder.exists()
        outside=tmp_path/'outside';outside.mkdir();(tmp_path/'recordings/link').symlink_to(outside,target_is_directory=True)
        with pytest.raises(ValueError,match='链接'):service.delete_assets('recordings',['link'])
        assert outside.exists()
        service.recording['saving']=True
        with pytest.raises(ValueError,match='保存'):service.delete_assets('recordings',['a'])
        service.recording['saving']=False;service.replay_active=True
        with pytest.raises(ValueError,match='实时'):service.delete_assets('recordings',['a'])
        service.replay_active=False
        with asset_lock(tmp_path):
            with pytest.raises(ValueError,match='后台任务'):service.delete_assets('recordings',['a'])
        assert folder.exists()
    finally:service.close()


def test_delete_active_model_unloads_without_switching_solver(tmp_path):
    from moxian_web.server import Runtime
    folder=tmp_path/'models/active';folder.mkdir(parents=True);(folder/'last.pth').write_bytes(b'weights')
    (tmp_path/'active.json').write_text('{"model":"active"}')
    service=GeoRTService(tmp_path);stopped=[]
    wuji=SimpleNamespace(backend='geort',lock=threading.RLock(),state={'ready':True},
        stop=lambda:stopped.append(True),topology={'mesh':True},source_version='v',geort_model_path=str(folder))
    try:
        runtime=SimpleNamespace(wuji=wuji)
        result=Runtime.delete_learning_assets(runtime,service,'models',['active'],'geort')
        assert result['unloaded'] and stopped
        assert wuji.backend=='geort' and wuji.state['ready'] is False
        assert wuji.state['status']=='model_missing' and not wuji.geort_model_path
        assert service.selected is None and not (tmp_path/'active.json').exists() and not folder.exists()
        # Models cancelled before config.json exists remain visible and deletable.
        failed=tmp_path/'models/failed';failed.mkdir();(failed/'status.json').write_text('{"status":"cancelled"}')
        assert service.catalog('models')[0]['id']=='failed'
        service.delete_assets('models',['failed'])
    finally:service.close()


def test_busy_job_prevents_deletion_and_cart_uses_same_lifecycle(tmp_path):
    from moxian_web.cart_service import CartService
    service=CartService(tmp_path)
    folder=tmp_path/'models/current';folder.mkdir(parents=True)
    service.selected='current';(tmp_path/'active.json').write_text('{"model":"current"}')
    try:
        service.job=SimpleNamespace(poll=lambda:None)
        with pytest.raises(ValueError,match='后台任务'):service.delete_assets('models',['current'])
        assert folder.exists()
        service.job=None
        assert service.delete_assets('models',['current'])['unloaded']
        assert service.selected is None
    finally:service.close()


def test_deleting_last_recording_removes_its_coverage_and_counters(tmp_path):
    service=GeoRTService(tmp_path)
    recording(tmp_path,'last',np.zeros((2,5,3),np.float32))
    c=WorkspaceCoverage({});c.add(np.zeros((5,3)),1)
    service.coverage_total=service.coverage_segment=c;service.coverage_current_id='last'
    service.recording=dict(id='last',active=False,saving=False,frames=2)
    try:
        service.delete_assets('recordings',['last'])
        assert not service.coverage_snapshot()['ready']
        assert service.recording['frames']==0
    finally:service.close()


def test_cart_preview_packet_is_independent_of_robot_and_contains_same_frame_mesh(tmp_path):
    import base64,time
    from moxian_web.cart_service import CartService
    service=CartService(tmp_path)
    frame=dict(vertices_m=[[0,0,0],[.1,0,0],[0,.1,0]],joints_m=[[0,0,0]],
               geometry_version='g',binding_version='b',timestamp_ms=1000)
    service.preview=dict(preview_frame=frame,hull=dict(ready=False,contact_mode='hand',timestamp_ms=1000))
    service.preview_at=service.recorder.latest_at=time.monotonic()
    try:
        packet=service.preview_packet()
        assert packet['ready'] and packet['timestamp_ms']==packet['hull']['timestamp_ms']
        values=np.frombuffer(base64.b64decode(packet['mesh_f32']['data']),dtype='<f4').reshape(-1,3)
        np.testing.assert_allclose(values[:3],frame['vertices_m'])
        assert packet['mesh_f32']['vertices']==3 and packet['mesh_f32']['joints']==1
        service.recorder.latest_at=time.monotonic()-10
        assert not service.preview_packet()['ready']
    finally:service.close()
