"""CPU fixtures validate SPIDER replay contracts, not GPU grasp success."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from retarget.env.grasp.spider_validation import audit_replay, object_errors


@pytest.fixture
def recorded_cpu_fixture(tmp_path):
    """26 scalar servos and a free object supported by a fixed thumb pad."""
    bodies, actuators = [], []
    for i in range(26):
        name = f"right_finger{i+1}_link4" if i < 5 else f"joint_body_{i}"
        position = "0 0 0" if i == 0 else f"{2.+i*.2} 0 0"
        pad = ('''<body name="r_thumb_tip_sensor_frame">
            <geom name="sensor_pad" type="box" size=".05 .05 .01" mass="0"/>
            </body>''' if i == 0 else '<geom type="sphere" size=".001" mass="0"/>')
        bodies.append(f'''<body name="{name}" pos="{position}">
          <inertial mass=".1" pos="0 0 0" diaginertia=".001 .001 .001"/>
          <joint name="j{i}" axis="0 0 1" range="-1 1" armature=".01"/>{pad}</body>''')
        actuators.append(f'<position joint="j{i}" kp="10" kv="1" ctrlrange="-1 1"/>')
    xml = tmp_path/"model.xml"
    xml.write_text('<mujoco><option timestep=".002" integrator="implicitfast"/>'
        '<worldbody>'+''.join(bodies)+'''<body name="grasp_object" pos="0 0 .029">
          <freejoint name="obj_joint"/><geom name="object_geom" type="sphere" size=".02" mass=".2"/>
          </body></worldbody><actuator>'''+''.join(actuators)+'</actuator></mujoco>')
    model = mujoco.MjModel.from_xml_path(str(xml))
    assert (model.nq, model.nv, model.nu) == (33, 32, 26)
    data = mujoco.MjData(model)
    data.qvel[26] = .01
    q, v, times, controls = [data.qpos.copy()], [data.qvel.copy()], [data.time], []
    for _ in range(10):
        data.ctrl[:] = 0.
        mujoco.mj_step(model, data)
        q.append(data.qpos.copy()); v.append(data.qvel.copy()); times.append(data.time)
        controls.append(data.ctrl.copy())
    values = dict(qpos=np.asarray(q), qvel=np.asarray(v), ctrl=np.asarray(controls), time=np.asarray(times))
    trajectory = tmp_path/"trajectory.npz"
    np.savez(trajectory, **values)
    return xml, trajectory, values, model


def test_object_errors_track_reference_and_ignore_quaternion_sign_and_hand_root():
    reference = np.zeros((3, 33)); reference[:, 29] = 1.
    reference[:, 26:29] = [[0., 0., 0.], [.003, .004, 0.], [.01, 0., 0.]]
    actual = reference.copy()
    actual[:, :26] = 6.7
    actual[:, 29] = -1.
    actual[1, 26] += .003
    actual[2, 29:33] = [np.cos(.15), np.sin(.15), 0., 0.]
    errors = object_errors(actual, reference)
    np.testing.assert_allclose(errors["position_mm"], [0., 3., 0.], atol=1e-12)
    np.testing.assert_allclose(errors["rotation_deg"], [0., 0., np.rad2deg(.3)], atol=1e-10)
    with pytest.raises(ValueError, match="Zero object quaternion"):
        object_errors(np.zeros((2, 33)), np.zeros(33))


def test_real_cpu_replay_retains_velocity_and_measures_fixed_pad_forces(recorded_cpu_fixture, tmp_path):
    xml, trajectory, values, model = recorded_cpu_fixture
    original = xml.read_bytes()
    report = audit_replay(xml, trajectory, tmp_path/"audit", reference_qpos=values["qpos"][0])
    assert report["finite"] and report["mujoco_warnings_zero"]
    assert report["physics_steps"] == 10 and report["duration_seconds"] == pytest.approx(.02)
    assert report["cpu"]["finger_normal_force_min_n"][0] > 0.
    assert report["cpu"]["finger_zero_load_steps"] == [0, 10, 10, 10, 10]
    assert report["gpu"]["available"] is False
    assert report["five_finger_loaded_contact_retained"] is None
    assert report["task_success"] is False
    assert report["controls_changed"] is False and report["extra_forward_calls"] is False
    assert report["cpu_object_contact_force_constraint_max_error_n"] < 1e-10
    assert report["gpu_cpu_qpos_max_abs_difference"] < 1e-12
    assert report["gpu_cpu_qvel_max_abs_difference"] < 1e-12
    with np.load(tmp_path/"audit/replay.npz") as result:
        np.testing.assert_array_equal(result["cpu_qvel"][0], values["qvel"][0])
        assert result["cpu_qvel"][0, 26] == .01
        np.testing.assert_array_equal(result["ctrl"], values["ctrl"])
        np.testing.assert_array_equal(result["cpu_contact_finger_id"], 0)
    assert xml.read_bytes() == original


def test_gpu_load_record_cannot_replace_independent_cpu_five_finger_check(recorded_cpu_fixture, tmp_path):
    xml, trajectory, values, model = recorded_cpu_fixture
    # Format-only positive GPU fixture; the real independent replay still
    # establishes that only a thumb pad supports this toy object's weight.
    np.savez(trajectory, **values, finger_normal_force_n=np.ones((10, 5)))
    report = audit_replay(xml, trajectory, tmp_path/"audit", reference_qpos=values["qpos"][0])
    assert report["five_finger_loaded_contact_retained"] is True
    assert report["independent_mujoco_five_finger_contact_retained"] is False
    assert report["task_success"] is False


def test_ca_subset_acceptance_checks_annotated_fingers_without_claiming_five(recorded_cpu_fixture, tmp_path):
    xml, trajectory, values, model = recorded_cpu_fixture
    # Format fixture only: independent CPU dynamics establish thumb support.
    measured = np.zeros((10, 5)); measured[:, 0] = 1.
    np.savez(trajectory, **values, finger_normal_force_n=measured)
    report = audit_replay(xml, trajectory, tmp_path/"thumb", reference_qpos=values["qpos"][0],
        required_contact_finger_mask=[1, 0, 0, 0, 0])
    assert report["required_finger_contact_retained"] is True
    assert report["independent_mujoco_required_finger_contact_retained"] is True
    assert report["five_finger_loaded_contact_retained"] is False
    assert report["independent_mujoco_five_finger_contact_retained"] is False
    assert report["task_success"] is True
    missing = audit_replay(xml, trajectory, tmp_path/"two", reference_qpos=values["qpos"][0],
        required_contact_finger_mask=[1, 1, 0, 0, 0])
    assert missing["task_success"] is False
    with pytest.raises(ValueError, match="five binary"):
        audit_replay(xml, trajectory, tmp_path/"bad", reference_qpos=values["qpos"][0],
            required_contact_finger_mask=[1, -1, 0, 0, 0])


def test_ca_gaps_use_reference_object_anchors_without_mutating_replay(recorded_cpu_fixture, tmp_path):
    xml, trajectory, values, model = recorded_cpu_fixture
    report = audit_replay(xml, trajectory, tmp_path/"audit", reference_qpos=values["qpos"][0],
        contact_body_ids=[model.body("right_finger1_link4").id],
        contact_local=[[0., 0., .01]], contact_object_local=[[0., 0., -.02]])
    assert report["contact_geometry"]["available"]
    assert report["gpu_cpu_qpos_max_abs_difference"] < 1e-12
    with np.load(tmp_path/"audit/replay.npz") as result:
        np.testing.assert_allclose(result["contact_target_world_m"],
            np.broadcast_to([[[0., 0., .009]]], (11, 1, 3)), atol=1e-12)
        np.testing.assert_allclose(result["cpu_contact_target_gap_mm"][0], [1.], atol=1e-12)


def test_moving_reference_is_not_silently_replaced_by_initial_object_pose(recorded_cpu_fixture, tmp_path):
    xml, trajectory, values, model = recorded_cpu_fixture
    reference = values["qpos"].copy()
    reference[5:, 26] += .1
    report = audit_replay(xml, trajectory, tmp_path/"audit", reference_qpos=reference)
    assert report["gpu"]["position_mm"]["max"] == pytest.approx(100.)
    assert report["native_hold_success"] is False
    assert report["independent_mujoco_hold_success"] is False


@pytest.mark.parametrize("bad", ["time", "ctrl", "state", "anchors"])
def test_invalid_trace_or_incomplete_anchor_input_is_rejected(recorded_cpu_fixture, tmp_path, bad):
    xml, trajectory, values, model = recorded_cpu_fixture
    kwargs = {}
    if bad == "time":
        values["time"][3] += .0003
    elif bad == "ctrl":
        values["ctrl"] = values["ctrl"][:, :25]
    elif bad == "state":
        values["qvel"][0, 0] = np.nan
    else:
        kwargs["contact_body_ids"] = [model.body("right_finger1_link4").id]
        kwargs["contact_local"] = [[0., 0., .01]]
    np.savez(trajectory, **values)
    with pytest.raises(ValueError):
        audit_replay(xml, trajectory, tmp_path/"audit", reference_qpos=values["qpos"][0], **kwargs)
