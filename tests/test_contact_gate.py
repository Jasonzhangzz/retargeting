"""Contact-gated reference progression, including the MPC lookahead and replay."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np

from forceaware.contact_observation import loaded_fingers
from forceaware.time_grid import TimeGrid
from forceaware.targets import ContactGuidance, SequenceData
from retarget.contact_gate import ContactReferenceGate, prelift_frame


def sequence():
    n, contacts = 12, 2
    hand = np.zeros((n, 8), np.float32)
    hand[6:, 2] = np.arange(1, n-5) * .01
    hand[:, 6] = np.arange(n) * .1
    pose = np.zeros((n, 7), np.float32); pose[:, 3] = 1.
    pose[7:, 6] = np.arange(1, n-6) * .01
    mask = np.zeros((n, contacts), np.float32); mask[3:] = 1.
    contact = ContactGuidance(mask, np.ones_like(mask), np.zeros((n, contacts, 3)),
                              np.zeros((n, contacts), int), np.ones((n, contacts), int),
                              np.zeros((n, contacts, 3)))
    return SequenceData(hand, pose, contact)


def config():
    return SimpleNamespace(optimizer=SimpleNamespace(horizon=5),
                           simulator=SimpleNamespace(time_grid=TimeGrid.create(
                               mpc_dt=.01, action_dt=.03, knot_dt=.06,
                               exec_dt=.01, ref_dt=1/30)))


def test_hold_caps_dense_and_knot_lookahead_and_keeps_contact_posture():
    seq, cfg = sequence(), config()
    hold = prelift_frame(seq, 0., 11)
    assert hold == 5
    gate = ContactReferenceGate(4.5, hold, 11, .01)
    targets = gate.make_targets(seq, cfg)
    assert targets.frames.max() == targets.knot_frames.max() == hold
    assert np.all(targets.dense_actions[:, 2] == 0.)
    assert np.all(targets.actions[:, 2] == 0.)
    assert np.all(targets.contact_mask == 1.)
    gate.advance(targets.frames[:3]); gate.observe([(), (), ()])
    for _ in range(50):
        targets = gate.make_targets(seq, cfg)
        gate.advance(targets.frames[:3]); gate.observe([(), (), ()])
    assert gate.frame == hold
    assert not gate.ready


def test_two_distinct_fingers_must_be_continuously_loaded():
    gate = ContactReferenceGate(5., 5., 11., .01)
    gate.observe([(2, 2)] * 100)
    assert not gate.ready
    gate.observe([(1, 2)] * 8)
    assert not gate.ready
    gate.observe([()])
    gate.observe([(1, 2)] * 8)
    assert not gate.ready
    gate.observe([(1, 2)])
    assert gate.ready


def test_resume_from_held_frame_without_wall_clock_catchup_and_pause_on_loss():
    cfg, seq = config(), sequence()
    gate = ContactReferenceGate(5., 5., 11., .01)
    gate.observe([(1, 2)] * 9)
    targets = gate.make_targets(seq, cfg)
    np.testing.assert_allclose(targets.frames[:3], [5.3, 5.6, 5.9], atol=1e-6)
    gate.advance(targets.frames[:3])
    assert gate.released
    gate.observe([(1,)])
    frozen = gate.make_targets(seq, cfg)
    assert np.all(frozen.frames == gate.frame)
    assert np.all(frozen.knot_frames == gate.frame)


def test_contact_confirmation_excludes_separation_inactive_and_table_rows():
    assert loaded_fingers([1, 1, 2, 3, 4, -1],
                          [-.001, -.001, -.001, .001, -.001, -.001],
                          [1, 1, 1, 1, 0, 1], [.03, .03, .06, 100., 100., 100.]) == (1, 2)
    assert loaded_fingers([1, 2], [-.001, -.001], [1, 1], [.001, .001]) == ()


def test_bundled_sequence_gate_precedes_hand_and_object_lift():
    root = Path(__file__).resolve().parents[1] / 'data/example/contactaware'
    hand = np.load(root / 'wuji_right_qpos.npy')
    pose = np.load(root / 'object_pose_7.npy')
    with np.load(root / 'contact_guidance.npz') as data:
        seq = SimpleNamespace(hand=hand, object_pose_xyzw=pose, hand_index_per_frame=1.,
                              contact=SimpleNamespace(mask=data['contact_mask']))
        hold = int(prelift_frame(seq, 0., len(hand)-1))
        assert hold == 23
        assert np.count_nonzero(seq.contact.mask[hold]) > 0
        assert hand[hold+1, 2] > hand[hold, 2]
        assert pose[hold, 6] < pose[-1, 6] - .1


def test_replay_uses_held_reference_time(tmp_path):
    from retarget.viewer import reference_times
    (tmp_path / 'config.json').write_text(json.dumps({'simulator': {'ref_dt': 1/30}}))
    frames = np.r_[np.arange(24), np.full(100, 23)]
    saved = {'qpos_frames': frames, 'qpos_traj': np.zeros((124, 33))}
    result = reference_times(saved, tmp_path)
    np.testing.assert_allclose(result[23:], 23/30)
