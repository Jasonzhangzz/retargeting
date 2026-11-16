"""A calibration artifact must reproduce source IK in a fresh runtime."""
import copy
import hashlib
import json
from pathlib import Path
import threading

import numpy as np
import pytest

from moxian_web.bone_extrinsics import validate_candidate, publish_active, read_active, digest
from moxian_web.source_ik_initialization import (with_calibration_seeds,
    runtime_revision, validate_recorded_source, BANK_REVISION)
from moxian_web.studio_emf_solver import StudioQPSkeleton


@pytest.fixture(scope='module')
def calibrated():
    data=json.loads((Path(__file__).parent/'fixtures/moxian_source_ik_branch.json').read_text())
    candidate=copy.deepcopy(data['candidate']);candidate['source_capture_hash']='raw-test-capture'
    rows=[dict(copy.deepcopy(row),partition='train',native_pose=stage,timestamp_ms=1000+i*20)
          for i,(stage,row) in enumerate(data['poses'].items())]
    return with_calibration_seeds(candidate,rows,threading.Event()),rows


def test_serialized_calibration_seeds_reproduce_fresh_engine_and_keep_geometry(calibrated,tmp_path):
    candidate,rows=calibrated
    bank=candidate['source_ik_initialization']
    assert bank['revision']==BANK_REVISION and len(bank['seeds'])==len(rows)
    publish_active(tmp_path,candidate)
    loaded=read_active(tmp_path)
    np.testing.assert_array_equal(loaded['source_ik_initialization']['seeds'][0]['joint_positions_rad'],bank['seeds'][0]['joint_positions_rad'])
    a=StudioQPSkeleton(candidate);b=StudioQPSkeleton(loaded)
    try:
        for row in rows:
            a.reset();b.reset()
            ra=a.step(row);rb=b.step(row)
            np.testing.assert_array_equal(ra['pose']['joint_positions_rad'],rb['pose']['joint_positions_rad'])
            assert rb['ik_diagnostics']['branch_recovery']['calibration_seed_sha256']==bank['sha256']
            assert rb['ik_diagnostics']['branch_recovery']['probe_calls']<=3
            if rb['ik_diagnostics']['branch_recovery']['probe_calls']:
                assert rb['ik_diagnostics']['branch_recovery']['calibration_seed_scores']==len(bank['seeds'])
        # Runtime never uses audit pose labels to choose or force the output.
        changed=copy.deepcopy(loaded)
        for s in changed['source_ik_initialization']['seeds']:s['calibration_pose']='unrelated'
        payload={k:v for k,v in changed['source_ik_initialization'].items() if k!='sha256'}
        changed['source_ik_initialization']['sha256']=digest(payload)
        c=StudioQPSkeleton(changed)
        try:
            a.reset()
            np.testing.assert_array_equal(a.step(rows[0])['pose']['joint_positions_rad'],c.step(rows[0])['pose']['joint_positions_rad'])
        finally:c.close()
    finally:a.close();b.close()


@pytest.mark.parametrize('kind',['checksum','model','limits','names','capture'])
def test_invalid_or_stale_seed_bank_rejected(calibrated,kind):
    candidate=copy.deepcopy(calibrated[0]);bank=candidate['source_ik_initialization']
    if kind=='checksum':bank['seeds'][0]['joint_positions_rad'][0]+=.001
    if kind=='model':bank['model_hash']='other'
    if kind=='limits':bank['seeds'][0]['joint_positions_rad'][0]=100.
    if kind=='names':bank['joint_names'][0]='other'
    if kind=='capture':bank['source_capture_hash']='other'
    if kind!='checksum':bank['sha256']=digest({k:v for k,v in bank.items() if k!='sha256'})
    with pytest.raises(ValueError,match='源IK'):validate_candidate(candidate)


def test_training_only_cancellation_and_runtime_identity(calibrated):
    candidate,rows=calibrated
    old=copy.deepcopy(candidate);old.pop('source_ik_initialization')
    assert old['model_hash']==candidate['model_hash']
    assert runtime_revision(old)!=runtime_revision(candidate)!=candidate['model_hash']
    validate_recorded_source(candidate,dict(skeleton_runtime_revision=runtime_revision(candidate)))
    with pytest.raises(ValueError,match='旧源IK'):
        validate_recorded_source(candidate,dict(skeleton_runtime_revision=candidate['model_hash']))
    held=copy.deepcopy(rows);held[0]['partition']='validation'
    with pytest.raises(ValueError,match='训练'):with_calibration_seeds(candidate,held,threading.Event())
    cancel=threading.Event();cancel.set()
    with pytest.raises(InterruptedError):with_calibration_seeds(candidate,rows,cancel)


def test_saved_old_skeleton_cannot_be_reused_for_training_or_replay(calibrated,tmp_path):
    from retarget.baseline.geort.training import load_recordings
    from retarget.baseline.geort.runtime import Replay
    candidate,_=calibrated
    folder=tmp_path/'recordings/old';folder.mkdir(parents=True)
    np.save(folder/'points.npy',np.zeros((2,5,3),np.float32))
    np.save(folder/'timestamps.npy',np.array([1000.,1020.]))
    np.savez(folder/'motion.npz',qpos=np.zeros((2,29)))
    (folder/'topology.json').write_text('{}')
    (folder/'profile.json').write_text(json.dumps(candidate))
    (folder/'manifest.json').write_text(json.dumps(dict(complete=True,frames=2,
        points_sha256=hashlib.sha256((folder/'points.npy').read_bytes()).hexdigest(),
        source=dict(skeleton_runtime_revision=candidate['model_hash']))))
    with pytest.raises(ValueError,match='旧源IK'):load_recordings(tmp_path,['old'])
    with pytest.raises(ValueError,match='旧源IK'):Replay(folder)


def test_saved_seed_warm_transition_still_uses_original_speed_box(calibrated):
    candidate,rows=calibrated
    data=json.loads((Path(__file__).parent/'fixtures/moxian_source_ik_branch.json').read_text())
    e=StudioQPSkeleton(candidate)
    try:
        e.previous=np.array(data['stuck_q']);e.previous_timestamp=1000.;e.frames=1
        for n in range(1,31):
            previous=e.previous.copy();row=dict(data['poses']['pinch_index'],timestamp_ms=1000+n*1000/60)
            result=e.step(row)
            assert np.max(abs(e.previous-previous))<=e.config['v_max']/60+1e-12
        assert result['pose']['tracking_errors']['index']['position_error_mm']<.5
    finally:e.close()


def test_six_pose_fit_builds_and_replays_serialized_bank_on_every_calibration(monkeypatch):
    import torch
    from scipy.spatial.transform import Rotation
    import moxian_web.moxian_six_calibration as six
    from moxian_web.qnbot_surface import QnbotSurface
    from moxian_web.contact import default_mapping
    from moxian_web.wuji_native_abi import SDK_POSES,FINGERS
    from moxian_web.wuji_native_capture import PROTOCOL
    torch.set_num_threads(1)
    m=QnbotSurface('right','cpu',finger_twist=True).double()
    mapping=dict(assignments=default_mapping('right'),confirmed=[]);m.bind_mapping(mapping)
    with torch.no_grad():p,r=m.sensor_prediction(m.q0[None].repeat(6,1))
    frames={stage:[[dict(position=p[i,f].tolist(),quaternion_xyzw=Rotation.from_matrix(r[i,f].numpy()).as_quat().tolist())
                      for f in range(5)]] for i,stage in enumerate(SDK_POSES)}
    doc=dict(protocol=PROTOCOL,completed_groups=list(SDK_POSES),state=dict(state='complete'),
             pose_frames=frames,side='right',calibration_fingerprint='capture',mapping=mapping)
    profile=dict(side='right',serial='TEST',calibration_fingerprint='capture',device='cpu',
                 objective_backend='eager',capture_session_id='six-poses')
    monkeypatch.setattr(six,'initial_rows',lambda *args:[])
    monkeypatch.setattr(six,'initialize_registration',lambda *args:None)
    # Isolate geometry optimization; exercise the real post-fit seed building,
    # serialization, cold replay and report path twice without a long fit.
    class Optimizer:
        provenance={}
        def __init__(self,*args):pass
        def set_lower_bounds(self,*args):pass
        def set_upper_bounds(self,*args):pass
        def set_ftol_rel(self,*args):pass
        def set_maxeval(self,*args):pass
        def set_min_objective(self,callback):self.callback=callback
        def optimize(self,x):self.callback(x,np.empty_like(x));return x
        def last_optimize_result(self):return 4
    monkeypatch.setattr(six,'WujiCalibrationLbfgs',Optimizer)
    values=[]
    for n in range(2):
        phases=[]
        artifact,report=six.fit_capture(profile,mapping,doc,lambda m:phases.append(m['phase']),threading.Event())
        assert 'source_ik_seeds' in phases and report['source_ik_initialization']['serialized_replay_validated']
        assert report['source_ik_initialization']['seeds']==6
        assert report['validation_source_ik']=='moxian_studio_native_qp_v3_calibration_seeds'
        validate_candidate(json.loads(json.dumps(artifact)))
        values.append(artifact['source_ik_initialization'])
    assert values[0]==values[1]
    from moxian_web.moxian_replay import MoxianReplayReconstructor
    from moxian_web.moxian_six_calibration import training_rows
    _,rows=training_rows(doc)
    observation=dict(rows[0],hand_side='right',mapping=mapping,calibration_fingerprint='capture')
    replay=MoxianReplayReconstructor(json.loads(json.dumps(artifact)))
    reconstructed=replay.frame(observation)
    assert reconstructed['skeleton_runtime_revision']==runtime_revision(artifact)
    assert reconstructed['quality']['source_ik_solver']==report['validation_source_ik']
