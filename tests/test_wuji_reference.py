"""Calibrated Moxian input must preserve the original robot numerical/state path."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from moxian_web.contact import default_mapping
from moxian_web.qnbot_surface import QnbotSurface
from retarget.baseline.wuji_retargeting.reference import RetargetSession, REFERENCE_ROOT
from retarget.baseline.wuji_retargeting.runtime import WujiReferenceRetarget, link_poses
from retarget_reference.preprocess import preprocess


@pytest.fixture(scope='module', params=['left', 'right'])
def source(request):
    torch.set_num_threads(1)
    m = QnbotSurface(request.param, 'cpu', finger_twist=True)
    m.bind_mapping(dict(assignments=default_mapping(m.side), confirmed=[]))
    return m


def skeleton_frame(source, index=0):
    q = source.q0.clone()
    q[7] += .12*np.sin(index*.4)
    q[-1] += .08*np.sin(index*.3)
    with torch.no_grad(): v, j, _ = source(q[None])
    return dict(hand_side=source.side, timestamp_ms=1000.+index*10, skeleton_timestamp_ms=1000.+index*10,
        geometry_version=source.version(), binding_version=source.binding_hash, surface_backend='qnbot',
        calibrated=True, preview=False, vertices_m=v[0].tolist(), joints_m=j[0].tolist())


@pytest.fixture(scope='module')
def engine(source):
    return WujiReferenceRetarget(source.topology())


def test_imported_source_and_all_assets_match_remote_hashes():
    manifest=json.loads((REFERENCE_ROOT/'SOURCE.json').read_text())
    assert len(manifest['files'])==45
    for name, expected in manifest['files'].items():
        assert hashlib.sha256((REFERENCE_ROOT/name).read_bytes()).hexdigest()==expected, name
    assert manifest['tree_sha256']==hashlib.sha256(json.dumps(manifest['files'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def test_same_initialization_warm_filter_cache_and_reset_as_upstream(engine, source):
    engine.reset()
    original=RetargetSession.for_hand('WujiHand2', source.side)
    # Select independently from semantic chains, including the actual wrist
    # index (2 in Qnbot), rather than assuming that joints[0] is the wrist.
    chains=source.topology()['joint_chains']
    indices=[chains[0][0], *[j for c in chains for j in c[-4:]]]
    assert engine.source.indices==indices and indices[0]==2
    frames=[skeleton_frame(source, i) for i in range(3)]
    first=None
    for i, f in enumerate(frames):
        direct=original.step(np.asarray(f['joints_m'])[indices]); wrapped=engine.step(f)
        np.testing.assert_array_equal(wrapped['joint_positions_rad'], direct)
        np.testing.assert_array_equal(engine.robot._warm, original._warm)
        np.testing.assert_array_equal(engine.robot._filtered, original._filtered)
        assert engine.robot.last_trace==original.last_trace
        assert wrapped['qpos'][6:]==wrapped['joint_positions_rad']
        assert [s['maxeval'] for s in wrapped['solver']['stages']]==([50, 80] if i==0 else [1, 10])
        assert 'human_q_rad' not in wrapped and 'emf_seq' not in wrapped
        if first is None:first=wrapped
    cached=engine.step(frames[-1]); assert engine.frames_computed==3
    assert cached['joint_positions_rad']==wrapped['joint_positions_rad']
    # A new observation with identical points must still advance the filter.
    repeated={**frames[-1], 'timestamp_ms':1040., 'skeleton_timestamp_ms':1040.}
    np.testing.assert_array_equal(engine.step(repeated)['joint_positions_rad'],
                                  original.step(np.asarray(repeated['joints_m'])[indices]))
    assert engine.frames_computed==4
    engine.reset(); again=engine.step(frames[0])
    assert again['joint_positions_rad']==first['joint_positions_rad']
    assert again['solver_session_id']!=first['solver_session_id']


def test_vertices_raw_emf_and_contact_cannot_change_solver_or_add_filter(engine, source):
    engine.reset(); f=skeleton_frame(source)
    first=engine.step(f); engine.reset()
    changed={**f, 'vertices_m':[[12.,22.,33.]], 'emf_input':{'invalid':'unused'},
             'source_contact':dict(ready=True,cell_contact=[True]*460), 'input_stream_epoch':72}
    second=engine.step(changed, contact_enabled=False)
    assert first['joint_positions_rad']==second['joint_positions_rad']
    assert second['contact_ready'] is False and second['solution_feasible'] is None
    assert second['postprocessing']=='upstream_float32_lowpass_alpha_0.2'
    later=engine.step({**skeleton_frame(source, 4), 'timestamp_ms':11000.,
                       'skeleton_timestamp_ms':11000., 'input_stream_epoch':73})
    assert later['solver']['first'] is False


def test_live_path_does_not_run_original_cqq_human_ik(engine, source, monkeypatch):
    from human_ik_reference import HumanIKSession
    def forbidden(*args, **kwargs):raise AssertionError('CQQ IK must not run online')
    monkeypatch.setattr(HumanIKSession, '__init__', forbidden)
    direct=WujiReferenceRetarget(source.topology())
    assert direct.step(skeleton_frame(source))['ready']


def test_display_is_inverse_of_original_preprocessing_and_uses_actual_wrist(engine, source):
    from scipy.spatial.transform import Rotation
    engine.reset(); f=skeleton_frame(source, 2); result=engine.step(f)
    points=np.asarray(f['joints_m'])[engine.source.indices]
    p32=points.astype(np.float32).astype(float)
    target=preprocess(points, 'WujiHand2', source.side, pinch=False)
    rotation=np.asarray(result['wrist_rotation'])
    origin=np.asarray(result['wrist_position_m'])
    expected=(p32-p32[0])@engine.mirror.T + origin
    actual=target@engine.native_mirror.T@rotation.T+origin
    np.testing.assert_allclose(actual,expected,atol=1e-15)
    np.testing.assert_array_equal(origin, points[0]@engine.mirror.T)
    rotations=np.asarray(result['link_rotations'])
    np.testing.assert_allclose(rotations@rotations.transpose(0,2,1),np.broadcast_to(np.eye(3),rotations.shape),atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(rotations),1.,atol=1e-12)
    # The displayed original wrist's local palmar axis follows the very same
    # source frame used by the objective, including r_base -> r_wrist Rx(pi).
    p, r=link_poses(engine.kinematics, result['joint_positions_rad'])
    expected_body=np.array([p[n] for n in engine.link_names])@engine.native_mirror.T@rotation.T+origin
    np.testing.assert_array_equal(result['link_positions_m'],expected_body)
    np.testing.assert_allclose(rotations[0],rotation@engine.native_mirror@
        Rotation.from_quat(r[engine.link_names[0]]).as_matrix()@engine.native_mirror,atol=1e-15)
    topology=engine.topology()
    assert topology['display_revision']=='inverse_original_preprocess_v3'
    assert len(topology['link_names'])==21 and len(topology['vertex_owners'])==len(topology['local_vertices_m'])


def test_rendering_preserves_original_fk_for_both_sides(engine):
    q=np.linspace(-.1,.3,20)
    p,_=link_poses(engine.kinematics,q)
    original,_=engine.kinematics.forward(q)
    np.testing.assert_array_equal([p[n] for n in engine.kinematics.landmark_names],original)


def test_invalid_uncalibrated_or_mismatched_input_does_not_advance_history(engine, source):
    engine.reset(); f=skeleton_frame(source)
    for change in ({'calibrated':False}, {'preview':True}, {'joints_m':[[0.,0.,0.]]},
                   {'hand_side':'invalid'}, {'binding_version':'other'}, {'model_hash':'other'},
                   {'skeleton_timestamp_ms':1026.}, {'timestamp_ms':float('nan')},
                   {'joints_m':np.full_like(f['joints_m'],np.nan).tolist()},
                   {'joints_m':np.zeros_like(f['joints_m']).tolist()}):
        with pytest.raises(ValueError):engine.step({**f, **change})
    assert engine.frames_computed==0 and engine.robot._warm is None
    good=engine.step(f);warm=engine.robot._warm.copy();trace=copy.deepcopy(engine.robot.last_trace)
    with pytest.raises(ValueError):engine.step({**f, 'calibrated':False})
    np.testing.assert_array_equal(engine.robot._warm,warm)
    assert engine.robot.last_trace==trace and engine.frames_computed==1 and good['ready']


def test_bridge_dispatches_current_calibrated_skeleton_timestamp(tmp_path):
    from moxian_web.wuji_bridge import WujiBridge
    bridge=WujiBridge(tmp_path,tmp_path);bridge.backend='wuji-retargeting'
    bridge.proc=SimpleNamespace(poll=lambda:None)
    source=dict(ready=True,timestamp_ms=1000.,skeleton_timestamp_ms=1000.,calibrated=True,
                hand_side='right',geometry_version='geometry',binding_version='binding',
                joints_m=[[0.,0.,0.]],vertices_m=[[0.,0.,0.]])
    hand=SimpleNamespace(snapshot=lambda **kwargs:source,model=lambda:dict(
        ready=True,rest_vertices_m=[[0.,0.,0.]],geometry_version='geometry',binding_version='binding'))
    bridge.dispatch(hand)
    sent=bridge.frames.get_nowait()['frame']
    assert sent['timestamp_ms']==sent['skeleton_timestamp_ms']==1000. and sent['calibrated']
    assert 'emf_input' not in sent and 'source_display_timestamp_ms' not in sent
    bridge.last_dispatch=0.;bridge.dispatch(hand)
    assert bridge.frames.empty()


def test_calibrated_skeleton_offline_replay_matches_original_without_tactile(engine, source, tmp_path):
    from scripts.retarget_moxian_wuji2 import replay
    recording=tmp_path/'input.jsonl'; destination=tmp_path/'output.jsonl'
    frames=[skeleton_frame(source,i) for i in range(3)]
    # A skipped frame interval must not silently reset the original robot.
    frames[2].update(timestamp_ms=2000., skeleton_timestamp_ms=2000.)
    rows=[dict(hand_model=f, hand_model_topology=source.topology() if i==0 else None)
          for i,f in enumerate(frames)]
    recording.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    report=replay(recording,destination,solver='wuji-retargeting')
    assert report['solved_frames']==3 and report['segments']==1 and report['contact_enabled'] is False
    assert report['postprocessing']=='upstream_float32_lowpass_alpha_0.2'
    original=RetargetSession.for_hand('WujiHand2',source.side)
    solved=[json.loads(line)['wuji2'] for line in destination.read_text().splitlines()]
    for f,result in zip(frames,solved):
        expected=original.step(np.asarray(f['joints_m'])[engine.source.indices])
        np.testing.assert_array_equal(result['joint_positions_rad'],expected)
        assert result['solver']['stages'][0]['maxeval']==(50 if f is frames[0] else 1)


def test_raw_emf_reconstruction_provides_same_frame_skeleton_metadata(source):
    from moxian_web.bone_extrinsics import definition_from_surface, model_hash
    from moxian_web.moxian_replay import MoxianReplayReconstructor
    from scipy.spatial.transform import Rotation
    candidate={**source.artifact(), 'bone_extrinsics':definition_from_surface(source),
               'serial':'test-device', 'calibration_fingerprint':'calibration'}
    candidate['model_hash']=model_hash(candidate)
    reconstructor=MoxianReplayReconstructor(candidate)
    with torch.no_grad():
        positions,rotations=source.sensor_prediction(source.q0[None])
        alignment,translation=source.alignment()
    positions=(positions[0].numpy()-translation.numpy())@alignment.numpy()
    rotations=alignment.numpy().T@rotations[0].numpy()
    observation=dict(timestamp_ms=1000.,calibration_fingerprint='calibration',
        hand_side=source.side,mapping=source.mapping,
        fingers={f:dict(position=p.tolist(),quaternion_xyzw=Rotation.from_matrix(r).as_quat().tolist(),valid=True)
                 for f,p,r in zip(('thumb','index','middle','ring','pinky'),positions,rotations)})
    f=reconstructor.frame(observation)
    assert f['timestamp_ms']==f['skeleton_timestamp_ms']==1000.
    wrapped=WujiReferenceRetarget(reconstructor.topology)
    expected=RetargetSession.for_hand('WujiHand2',source.side).step(np.asarray(f['joints_m'])[wrapped.source.indices])
    np.testing.assert_array_equal(wrapped.step(f)['joint_positions_rad'],expected)


def test_saved_motion_replay_has_source_identity_for_wuji_retargeting(source,tmp_path):
    from retarget.baseline.geort.runtime import Replay
    topology={**source.topology(), 'model_hash':'test-model'}
    meta=dict(complete=True,frames=2,id='replay',source=dict(model_hash='test-model',hand_side=source.side,
              serial='test-device',calibration_fingerprint='calibration',skeleton_runtime_revision='revision'))
    (tmp_path/'manifest.json').write_text(json.dumps(meta))
    (tmp_path/'topology.json').write_text(json.dumps(topology))
    (tmp_path/'profile.json').write_text(json.dumps(source.artifact()))
    np.savez(tmp_path/'motion.npz',qpos=np.stack([source.q0.numpy()]*2))
    np.save(tmp_path/'timestamps.npy',np.array([1000.,1016.]))
    replay=Replay(tmp_path);replay.pause(True)
    f=replay.frame()
    assert f['surface_backend']=='qnbot' and f['coordinate_frame']=='wrist_local'
    wrapped=WujiReferenceRetarget(topology)
    expected=RetargetSession.for_hand('WujiHand2',source.side).step(f['joints_m'][wrapped.source.indices])
    np.testing.assert_array_equal(wrapped.step(f)['joint_positions_rad'],expected)


def test_only_supported_solver_options_exist_in_page():
    from moxian_web.wuji_backends import SOLVERS
    import re
    html=(Path(__file__).resolve().parents[1]/'moxian_web/static/index.html').read_text()
    select=html.split('id="wujiSolver"', 1)[1].split('</select>', 1)[0]
    assert set(re.findall(r'<option value="([^"]+)"', select))==set(SOLVERS)
    assert not (Path(__file__).resolve().parents[1]/'moxian_web/card_slsqp.py').exists()
