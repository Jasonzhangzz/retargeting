"""Contract tests for GeoRT parity, capture identity and exact robot FK."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import textwrap
import time
import numpy as np
import pytest
from retarget.baseline.geort.adapter import SourceAdapter, RobotAdapter, ROOT
from retarget.baseline.geort.training import reference, training_losses, load_recordings


def topology(side='right'):
    return dict(surface_backend='qnbot',coordinate_frame='wrist_local',hand_side=side,
                geometry_version='geometry',binding_version='binding',model_hash='model',
                joint_chains=[[0,1+4*f,2+4*f,3+4*f,4+4*f] for f in range(5)],
                palm_view=dict(normal=[1,0,0],forward=[0,0,1]))


def frame():
    return dict(ready=True,calibrated=True,geometry_version='geometry',binding_version='binding',
                model_hash='model',hand_side='right',calibration_fingerprint='calibration',
                skeleton_runtime_revision='revision',source_joint_positions_rad=[0.]*29,
                timestamp_ms=time.time()*1000)


def test_fingertip_order_translation_and_left_mirror():
    rng=np.random.default_rng(1);j=rng.normal(size=(21,3))*.04
    adapter=SourceAdapter(topology())
    np.testing.assert_allclose(adapter.points(j),adapter.points(j+[2,3,4]),atol=1e-7)
    np.testing.assert_allclose(adapter.points(j),(j[[4,8,12,16,20]]-j[0]),atol=1e-7)
    left=j.copy();left[:,1]*=-1
    np.testing.assert_allclose(SourceAdapter(topology('left')).points(left),adapter.points(j),atol=1e-7)
    with pytest.raises(ValueError):adapter.identity('serial',{**frame(),'geometry_version':'changed'})


def test_wuji_exact_fk_and_normalization():
    robot=RobotAdapter();rng=np.random.default_rng(2)
    assert robot.joint_names[0]=='r_thumb_cmc_flex'
    assert robot.joint_names[-1]=='r_pinky_dip'
    for _ in range(8):
        q=rng.uniform(robot.lower,robot.upper)
        actual=robot.fk(q)
        expected=(robot.data.site_xpos[robot.tip_ids]-robot.data.xpos[robot.body_ids[0]])@robot.basis
        np.testing.assert_allclose(actual,expected,atol=1e-7)
        np.testing.assert_allclose(robot.unnormalize(robot.normalize(q)),q,atol=1e-7)
        assert np.all(robot.data.qpos[:6]==0)


def test_official_files_match_pinned_hashes():
    folder=ROOT/'retarget/baseline/geort/upstream';manifest=json.loads((folder/'SOURCE.json').read_text())
    for name,checksum in manifest['files'].items():assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==checksum


@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_loss_gradients_and_optimizer_step_match_upstream(device):
    import torch
    import torch.nn.functional as F
    if device.startswith('cuda') and not torch.cuda.is_available():pytest.skip('CUDA required')
    torch.set_num_threads(1)
    model=reference('model');indices=[list(range(i*4,i*4+4)) for i in range(5)]
    torch.manual_seed(81)
    ik=model.IKModel(indices).to(device);fk=model.FKModel(indices).to(device).eval()
    ik2=copy.deepcopy(ik);fk2=copy.deepcopy(fk)
    point=torch.randn(8,5,3,device=device)*.04;robot=np.random.default_rng(3).normal(size=(5,300,3)).astype(np.float32)*.04
    # Execute the original loss block, changing only CUDA device placement for CPU testing.
    source=(ROOT/'retarget/baseline/geort/upstream/trainer.py').read_text()
    block=source[source.index('                direction_loss = 0'):source.index('                ik_optim.zero_grad()')]
    block=textwrap.dedent(block).replace('.cuda()', '.to(point.device)')
    fn='def upstream(batch,point,ik_model,fk_model,robot_points):\n'+textwrap.indent(block,'    ')+'\n    return loss, dict(direction=direction_loss,chamfer=chamfer_loss,curvature=curvature_loss,pinch=pinch_loss,collision=collision_loss)\n'
    env=dict(torch=torch,np=np,F=F,chamfer_distance=reference('loss').chamfer_distance,
             n_keypoints=5,w_chamfer=80.,w_curvature=.1,w_collision=0.,w_pinch=1.)
    exec(fn,env)
    torch.manual_seed(14);np.random.seed(14)
    expected,ep=env['upstream'](point,point,ik,fk,robot)
    torch.manual_seed(14);np.random.seed(14)
    actual,ap=training_losses(point,ik2,fk2,robot)
    torch.testing.assert_close(actual,expected.reshape(()),rtol=0,atol=0)
    for k in ep:torch.testing.assert_close(ap[k],ep[k].reshape(()),rtol=0,atol=0)
    expected.backward();actual.backward()
    for a,b in zip(ik.parameters(),ik2.parameters()):torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)
    o1=torch.optim.AdamW(ik.parameters(),lr=1e-4);o2=torch.optim.AdamW(ik2.parameters(),lr=1e-4)
    o1.step();o2.step()
    for a,b in zip(ik.parameters(),ik2.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_official_voxel_resampling_is_preserved():
    dataset=reference('dataset');points=np.random.default_rng(42).normal(size=(5,120,3))*.04
    np.random.seed(99);a=dataset.MultiPointDataset.from_points(points,20000,resample_to=200)
    np.random.seed(99);b=reference('dataset').MultiPointDataset.from_points(points,20000,resample_to=200)
    np.testing.assert_array_equal(a.points,b.points)
    assert len(a)==200 and a[0].shape==(5,3)


def test_recorder_dedup_profile_change_and_dataset_compatibility(tmp_path,monkeypatch):
    import moxian_web.geort_service as service
    monkeypatch.setattr(service,'ROOT',tmp_path)
    serial='serial';home=tmp_path/'output/moxian/hand_model'/hashlib.sha256(serial.encode()).hexdigest()[:16]/'qnbot'
    home.mkdir(parents=True);(home/'current.json').write_text(json.dumps(dict(model_hash='model')))
    recorder=service.GeoRTService(tmp_path/'geort')
    try:
        f=frame();j=np.arange(63).reshape(21,3)*.001
        recorder.observe(f,topology(),j,serial)
        first=recorder.start_recording('test')
        for i in (0,0,1,2):recorder.observe({**f,'timestamp_ms':f['timestamp_ms']+i},topology(),j+i*.001,serial)
        recorder.observe({**f,'timestamp_ms':f['timestamp_ms']+3,'calibration_fingerprint':'changed'},topology(),j,serial)
        for _ in range(100):
            if not recorder.snapshot()['recording']['saving']:break
            time.sleep(.01)
        assert recorder.snapshot()['recording']['frames']==3
        assert '变化' in recorder.snapshot()['recording']['error']
        assert recorder.snapshot()['coverage']['segment']['frames']==3
        p,identity,records=load_recordings(tmp_path/'geort',[first['id']])
        assert p.shape==(3,5,3) and identity['serial']==serial
        assert recorder.catalog('recordings')[0]['complete']
        with pytest.raises(ValueError):load_recordings(tmp_path/'geort',['../bad'])
        folder=tmp_path/'geort/recordings'/first['id'];(folder/'points.npy').write_bytes(b'bad')
        with pytest.raises(ValueError):load_recordings(tmp_path/'geort',[first['id']])
    finally:recorder.close()


def test_new_recording_clears_previous_distribution_and_selected_baseline(tmp_path,monkeypatch):
    import moxian_web.geort_service as service
    monkeypatch.setattr(service,'ROOT',tmp_path)
    serial='serial';home=tmp_path/'output/moxian/hand_model'/hashlib.sha256(serial.encode()).hexdigest()[:16]/'qnbot'
    home.mkdir(parents=True);(home/'current.json').write_text(json.dumps(dict(model_hash='model')))
    recorder=service.GeoRTService(tmp_path/'geort')
    def saved():
        for _ in range(100):
            if not recorder.snapshot()['recording']['saving']:return
            time.sleep(.01)
        raise AssertionError('recording did not finish saving')
    try:
        f=frame();j=np.arange(63).reshape(21,3)*.001
        recorder.observe(f,topology(),j,serial)
        first=recorder.start_recording('first')
        recorder.observe(f,topology(),j,serial)
        recorder.observe({**f,'timestamp_ms':f['timestamp_ms']+1},topology(),j,serial)
        recorder.stop_recording();saved()
        assert recorder.select_coverage([first['id']])['frames']==2
        recorder.latest_at=0
        with pytest.raises(ValueError):recorder.start_recording('invalid')
        assert recorder.coverage_snapshot()['frames']==2
        recorder.latest_at=time.monotonic()
        second=recorder.start_recording('second')
        empty=recorder.coverage_snapshot(include_cloud=True)
        assert empty['frames']==empty['segment']['frames']==0
        assert empty['baseline_ids']==[] and empty['notice'] is None
        assert all(not finger['cloud_mm'] for finger in empty['fingers'])
        recorder.observe({**f,'timestamp_ms':f['timestamp_ms']+10},topology(),j*.5,serial)
        recorder.stop_recording();saved()
        current=recorder.coverage_snapshot(include_cloud=True)
        assert current['frames']==current['segment']['frames']==1
        assert all(len(finger['cloud_mm'])==1 for finger in current['fingers'])
        assert recorder.coverage_current_id==second['id']
        assert (tmp_path/'geort/recordings'/first['id']/'points.npy').is_file()
    finally:recorder.close()


def test_network_load_failure_does_not_select_partial_model(tmp_path):
    from moxian_web.geort_service import GeoRTService
    service=GeoRTService(tmp_path)
    try:
        folder=tmp_path/'models/unfinished';folder.mkdir(parents=True)
        (folder/'status.json').write_text('{"status":"cancelled"}')
        with pytest.raises(ValueError):service.model_folder('unfinished')
        assert service.selected is None
    finally:service.close()


def test_early_cancel_is_reported_and_does_not_replace_active_model(tmp_path):
    from types import SimpleNamespace
    from moxian_web.geort_service import GeoRTService
    service=GeoRTService(tmp_path)
    folder=tmp_path/'models/job';folder.mkdir(parents=True)
    (folder/'status.json').write_text('{"status":"starting","run_id":"job"}')
    service.job_id='job';service.selected='previous'
    process=SimpleNamespace(returncode=None)
    process.poll=lambda:process.returncode
    process.terminate=lambda:setattr(process,'returncode',-15)
    service.proc=process
    try:
        service.cancel_training()
        assert service.snapshot()['training']['status']=='cancelled'
        assert service.selected=='previous'
        with pytest.raises(ValueError):service.start_training(['missing'],epochs=0)
        assert service.selected=='previous'
    finally:service.close()


def test_geort_bridge_unknown_contacts_and_paused_playback(tmp_path):
    from types import SimpleNamespace
    from moxian_web.wuji_bridge import WujiBridge
    bridge=WujiBridge(tmp_path,tmp_path);bridge.proc=SimpleNamespace(poll=lambda:None)
    bridge.pose_at=time.monotonic()-60
    bridge.state=dict(solver_backend='geort',status='ready',display_available=True,solution_feasible=None,
                      playback=dict(active=True,playing=False),source_timestamp_ms=1,contact_ready=True)
    state=bridge.snapshot(False)
    assert state['ready'] and state['display_available'] and not state['contact_ready']
    assert state['contact_mask']==[None]*460
    bridge.state['playback']['active']=False
    assert not bridge.snapshot(False)['display_available']
    bridge.state['playback']['active']=True;bridge.state['status']='error'
    assert not bridge.snapshot()['ready']
