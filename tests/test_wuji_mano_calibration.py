"""Calibration numerics and acquisition contracts; no hardware is accessed."""
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

SPEC = importlib.util.spec_from_file_location(
    "wuji_mano", Path(__file__).parents[1] / "scripts/calibrate_wuji_mano.py")
cal = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cal)


def test_rodrigues_matches_scipy_and_has_finite_gradient_at_zero():
    vectors = np.array([[0., 0., 0.], [1e-12, 0, 0], [.7, -.3, 1.2]])
    aa = torch.tensor(vectors, dtype=torch.float64, requires_grad=True)
    matrices = cal.rotmat(aa)
    np.testing.assert_allclose(matrices.detach(), Rotation.from_rotvec(vectors).as_matrix(), atol=1e-14)
    matrices[:, 0, 1].sum().backward()
    assert torch.isfinite(aa.grad).all()
    assert aa.grad[0, 2] == -1


def emf(seq=1, confidence=.95):
    q = SimpleNamespace(x=0., y=0., z=0., w=1.)
    return SimpleNamespace(
        header=SimpleNamespace(seq=seq, timestamp_us=seq*10000, frame_id="r_hand_emf_tx"),
        poses=[SimpleNamespace(pose=SimpleNamespace(position=[.01*i, .02, .03], orientation=q),
                               confidence=confidence) for i in range(5)])


def test_actual_sdk_field_layout_and_bad_frame_rejection():
    row = cal.decode_emf(emf())
    np.testing.assert_allclose(row["positions_m"][1], [.01, .02, .03])
    assert row["quaternions_xyzw"][0] == [0., 0., 0., 1.]
    frame = emf()
    frame.poses[0].pose.position[0] = np.nan
    with pytest.raises(ValueError, match="Non-finite"):
        cal.decode_emf(frame)
    with pytest.raises(ValueError, match="confidence"):
        cal.decode_emf(emf(confidence=2))


def test_stability_rejects_low_confidence_position_and_orientation_motion():
    rows = [cal.decode_emf(emf(i)) for i in range(20)]
    assert cal.stable_frames(rows)
    rows[0]["confidence"][2] = .2
    assert not cal.stable_frames(rows)
    rows[0]["confidence"][2] = .95
    rows[0]["positions_m"][0][0] += .1
    assert not cal.stable_frames(rows)
    rows[0]["positions_m"][0][0] -= .1
    rows[0]["quaternions_xyzw"][1] = Rotation.from_rotvec([2., 0, 0]).as_quat().tolist()
    assert not cal.stable_frames(rows)


def test_measurement_units_missing_values_and_unknown_fingers(tmp_path):
    path = tmp_path / "measurements.json"
    measurements = cal.measurements_template()
    measurements["widths_mm"]["index"] = 18.
    measurements["palm_width_mm"] = 80.
    cal.write_json(path, measurements)
    targets, sigma = cal.load_measurements(path)
    assert targets == {"index_width_mm": 18., "palm_width_mm": 80.}
    assert sigma == 2.
    measurements["widths_mm"]["index"] = .018  # metres accidentally entered as mm
    cal.write_json(path, measurements)
    with pytest.raises(ValueError, match="millimetres"):
        cal.load_measurements(path)
    measurements["widths_mm"] = {"unknown": 18.}
    cal.write_json(path, measurements)
    with pytest.raises(ValueError, match="Unknown finger"):
        cal.load_measurements(path)


def test_legacy_numeric_select_without_importing_chumpy():
    leaf = cal._ChValue()
    leaf.x = np.arange(24).reshape(2, 3, 4)
    selection = cal._ChValue()
    selection.a = leaf
    selection.idxs = np.arange(24).reshape(2, 3, 4)[:, :, :2].ravel()
    selection.preferred_shape = (2, 3, 2)
    np.testing.assert_array_equal(cal.as_array(selection), leaf.x[:, :, :2])


def test_capture_file_rejects_incomplete_or_changed_coordinate_frame(tmp_path):
    path = tmp_path / "capture.json"
    data = {"schema_version": 1, "units": "metres", "finger_order": list(cal.FINGERS),
            "side": "right", "status": "capture_incomplete", "poses": []}
    cal.write_json(path, data)
    with pytest.raises(ValueError, match="incomplete"):
        cal.observations(path, "right")
    data["status"] = "capture_complete"
    data["poses"] = [{"name": "flat_open" if i == 0 else f"pose{i}",
                      "frames": [cal.decode_emf(emf(i+1))]} for i in range(6)]
    cal.write_json(path, data)
    assert len(cal.observations(path, "right")[1]) == 6
    data["poses"][1]["frames"][0]["frame_id"] = "another_tx"
    cal.write_json(path, data)
    with pytest.raises(ValueError, match="transmitter frame"):
        cal.observations(path, "right")


def test_rigid_alignment_handles_rotations_without_reflections():
    source = np.random.default_rng(2).normal(size=(5, 3))
    expected = Rotation.from_rotvec([.8, -.4, .5]).as_matrix()
    target = source @ expected.T + [.1, -.05, .02]
    r, t = cal.rigid_alignment(source, target)
    np.testing.assert_allclose(source @ r.T + t, target, atol=1e-14)
    assert np.linalg.det(r) == pytest.approx(1.)


@pytest.fixture
def licensed_model():
    path = os.environ.get("MANO_TEST_MODEL")
    if not path:
        pytest.skip("Set MANO_TEST_MODEL to a locally licensed MANO_RIGHT.pkl")
    torch.set_num_threads(1)
    return cal.PersonalizedMANO(path)


def test_real_model_neutral_shape_and_morphology_gradients(licensed_model):
    m = licensed_model
    beta = torch.zeros(m.shapedirs.shape[-1], dtype=torch.float64, requires_grad=True)
    scale = torch.zeros((), dtype=torch.float64, requires_grad=True)
    length = torch.zeros(5, dtype=torch.float64, requires_grad=True)
    radial = torch.zeros(5, dtype=torch.float64, requires_grad=True)
    pose = torch.zeros((1, 15, 3), dtype=torch.float64, requires_grad=True)
    v, j, _ = m(beta, scale, length, radial, pose)
    np.testing.assert_allclose(v[0].detach(), m.template, atol=2e-15)
    dims = m.dimensions(v[0], j[0])
    (dims["index_width_mm"] + dims["index_length_mm"]).backward()
    assert radial.grad[1].abs() > 1
    assert length.grad[1].abs() > 1
    assert all(torch.isfinite(p.grad).all() for p in (beta, scale, length, radial, pose))
    # Extension moves bone positions along with surface lengths.
    with torch.no_grad():
        base_v, base_j = m.rest_shape(beta, scale, length, radial)
        length[1] = np.log(1.15)
        larger_v, larger_j = m.rest_shape(beta, scale, length, radial)
    assert torch.linalg.vector_norm(larger_j[3]-larger_j[1]) > torch.linalg.vector_norm(base_j[3]-base_j[1]) * 1.14
    assert not torch.allclose(larger_v, base_v)


def test_nonpositive_or_nan_budget_is_rejected():
    with pytest.raises(SystemExit):
        cal.main(["fit", "--steps", "0"])
    with pytest.raises(SystemExit):
        cal.main(["capture", "--position-scale", "nan"])


def test_live_capture_only_disconnects_its_glove_on_cancel(tmp_path, monkeypatch):
    calls = []
    resource = SimpleNamespace(subscribe_with_callback=lambda *a, **kw: SimpleNamespace(close=lambda: calls.append("close")))
    glove = SimpleNamespace(hand_side=lambda: SimpleNamespace(get=lambda: "right"),
                            emf_poses=lambda: resource, disconnect=lambda: calls.append("disconnect"))
    manager = SimpleNamespace(scan=lambda: [SimpleNamespace(device_type=1, sn="test")],
                              connect=lambda **kw: glove)
    fake = SimpleNamespace(SdkManager=SimpleNamespace(instance=lambda: manager),
                           DeviceType=SimpleNamespace(WujiGlove=1))
    monkeypatch.setitem(__import__("sys").modules, "wuji_sdk", fake)
    monkeypatch.setattr(cal.importlib.metadata, "version", lambda name: "2026.10.26")
    def cancel(prompt):
        raise KeyboardInterrupt
    monkeypatch.setattr("builtins.input", cancel)
    args = SimpleNamespace(output=tmp_path, sn=None, side="right", position_scale=1.)
    with pytest.raises(KeyboardInterrupt):
        cal.capture(args)
    assert calls == ["close", "disconnect"]
    assert json.loads((tmp_path / "capture.json").read_text())["status"] == "capture_incomplete"


def test_full_fit_exercises_contacts_and_rejects_a_false_pinch(licensed_model, tmp_path):
    m = licensed_model
    zero = torch.zeros((), dtype=torch.float64)
    with torch.no_grad():
        v, _, _ = m(torch.zeros(10, dtype=torch.float64), zero,
                    torch.zeros(5, dtype=torch.float64), torch.zeros(5, dtype=torch.float64),
                    torch.zeros((1, 15, 3), dtype=torch.float64))
    row = cal.decode_emf(emf())
    row["positions_m"] = v[0, list(cal.TIPS)].numpy().tolist()
    # All observed poses are flat, while the operator falsely declares pinches.
    # Good sensor residuals alone must not certify this as a successful fit.
    data = {"schema_version": 1, "status": "capture_complete", "side": "right",
            "units": "metres", "finger_order": list(cal.FINGERS),
            "source": "synthetic_test", "poses": [
                {"name": name, "frames": [row]} for name, _ in cal.POSES]}
    path = tmp_path / "capture.json"
    cal.write_json(path, data)
    assert cal.main(["fit", "--capture", str(path), "--mano-model", str(m.path),
                     "--output", str(tmp_path / "fit"), "--steps", "2"]) == 2
    report = json.loads((tmp_path / "fit/calibration.json").read_text())
    assert report["checks"]["declared_pinch_surface_gap"] is False
    assert report["anatomical_identification_verified"] is False
    assert np.isfinite(report["loss"])
    assert (tmp_path / "fit/personalized_rest.obj").is_file()


def test_visualization_updates_actual_viser_handles_and_separates_frames():
    pytest.importorskip("viser")
    from scripts.wuji_mano_viewer import LiveViewer
    args = SimpleNamespace(parameters=None, host="127.0.0.1", port=0, no_browser=True,
                           fps=20., min_confidence=.7, position_scale=1.)
    viewer = LiveViewer(args)
    try:
        # Deterministic render on this test thread, without a browser or hardware.
        viewer.stop_event.set()
        viewer.thread.join(timeout=2)
        viewer.push_emf(cal.decode_emf(emf()))
        received = viewer.latest["emf"][1]
        viewer.push_emf(cal.decode_emf(emf()))
        assert viewer.latest["emf"][1] == received
        skeleton = SimpleNamespace(header=SimpleNamespace(frame_id="sdk_palm"), joints=[
            SimpleNamespace(pose=SimpleNamespace(position=[.001*i, 0, 0])) for i in range(21)])
        viewer.push_skeleton(skeleton)
        viewer._render()
        assert viewer.error == ""
        assert viewer.sensor_points.name == "/emf/sensors"
        assert viewer.sdk_lines.name == "/sdk/bones"
        assert np.asarray(viewer.sdk_lines.points).shape == (20, 2, 3)
        assert "r_hand_emf_tx" in viewer.status.content
        viewer.show_sdk.value = False
        viewer._render()
        assert not viewer.sdk_lines.visible
        row = cal.decode_emf(emf(2))
        row["frame_id"] = "changed_frame"
        with pytest.raises(ValueError, match="coordinate frame changed"):
            viewer.push_emf(row)
    finally:
        viewer.close()


def test_online_tracking_keeps_shape_fixed_and_freezes_on_low_confidence(licensed_model, tmp_path):
    from scripts.wuji_mano_viewer import OnlineMANO
    m = licensed_model
    parameters = tmp_path / "personalized_mano.npz"
    np.savez(parameters, betas=np.zeros(10), global_scale=np.array(1.),
             finger_length_scales=np.ones(5), finger_radial_scales=np.ones(5),
             root_axis_angle=np.zeros(3), translation_m=np.zeros(3),
             sensor_offsets_local_m=np.zeros((5, 3)),
             sensor_rotations_local_axis_angle=np.zeros((5, 3)),
             poses_axis_angle=np.zeros((1, 15, 3)), finger_order=np.array(cal.FINGERS))
    args = SimpleNamespace(parameters=parameters, mano_model=m.path, side="right", model_unit="m",
                           threads=1, device="cpu", ik_steps=2, min_confidence=.7)
    tracker = OnlineMANO(args)
    row = cal.decode_emf(emf())
    row["positions_m"] = m.template[list(cal.TIPS)].numpy().tolist()
    beta_before = tracker.beta.clone()
    tracked = tracker.update(row)
    assert tracked["vertices"].shape == (778, 3)
    assert np.isfinite(tracked["rmse_mm"])
    torch.testing.assert_close(tracker.beta, beta_before, rtol=0, atol=0)
    previous = tracker.pose.detach().clone()
    row["confidence"][2] = .1
    assert tracker.update(row) is None
    torch.testing.assert_close(tracker.pose, previous, rtol=0, atol=0)


def test_live_view_subscribes_both_sdk_streams_and_cleans_up(monkeypatch):
    from scripts import wuji_mano_viewer as module
    calls = []
    class Viewer:
        def __init__(self, args):
            self.stage = self.error = ""
        def push_emf(self, row):
            calls.append(("emf", row["seq"]))
        def push_skeleton(self, row):
            calls.append(("skeleton", row))
        def close(self):
            calls.append("viewer_close")
    def resource(label, value):
        def subscribe(callback, **kwargs):
            callback(value)
            return SimpleNamespace(close=lambda: calls.append(label + "_close"))
        return SimpleNamespace(subscribe_with_callback=subscribe)
    glove = SimpleNamespace(hand_side=lambda: SimpleNamespace(get=lambda: "right"),
                            emf_poses=lambda: resource("emf", emf()),
                            hand_skeleton=lambda: resource("skeleton", "sdk_frame"),
                            disconnect=lambda: calls.append("glove_disconnect"))
    manager = SimpleNamespace(scan=lambda: [SimpleNamespace(device_type=1, sn="test")],
                              connect=lambda **kw: glove)
    sdk = SimpleNamespace(SdkManager=SimpleNamespace(instance=lambda: manager),
                          DeviceType=SimpleNamespace(WujiGlove=1))
    monkeypatch.setitem(__import__("sys").modules, "wuji_sdk", sdk)
    monkeypatch.setattr(module, "LiveViewer", Viewer)
    args = SimpleNamespace(parameters=None, capture=None, side="right", sn=None,
                           duration=.001, position_scale=1.)
    module.view(args)
    assert ("emf", 1) in calls and ("skeleton", "sdk_frame") in calls
    assert calls[-4:] == ["emf_close", "skeleton_close", "glove_disconnect", "viewer_close"]
