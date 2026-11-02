"""Actual CPU contact forces and exact replay-sampling audit boundaries."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from retarget.env.grasp.forceaware_mesh_audit import (
    compare_saved_replay, control_physics_indices, mesh_contact_snapshot,
)


def test_mesh_contact_force_groups_rigid_sensor_descendant_and_matches_object_constraint():
    model = mujoco.MjModel.from_xml_string('''<mujoco>
      <option timestep=".002" integrator="implicitfast"/>
      <worldbody><body name="right_finger1_link4">
        <body name="r_thumb_tip_sensor_frame"><geom name="sensor_pad" type="box" size=".05 .05 .01"/></body>
      </body><body name="grasp_object" pos="0 0 .029">
        <freejoint name="obj_joint"/><geom name="object_geom" type="sphere" size=".02" mass=".2"/>
      </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    mujoco.mj_step(model, data)
    result = mesh_contact_snapshot(model, data, object_body=model.body("grasp_object").id)
    assert data.ncon > 0 and result["finger_normal_force_n"][0] > 0
    np.testing.assert_array_equal(result["finger_normal_force_n"][1:], 0.)
    np.testing.assert_array_equal(result["contact_finger_id"], 0)
    assert result["object_contact_force_world_n"][2] > 0
    np.testing.assert_allclose(result["object_contact_force_world_n"], data.qfrc_constraint[:3], atol=1e-10)
    np.testing.assert_allclose(result["finger_force_on_object_world_n"][0], result["object_contact_force_world_n"])


def test_mesh_time_grid_retains_every_substep_and_control_boundary():
    np.testing.assert_array_equal(control_physics_indices(3, 5), np.repeat([0, 1, 2], 5))
    for count, substeps in ((0, 1), (1, 0), (1, 2.5)):
        with pytest.raises(ValueError, match="positive integer"):
            control_physics_indices(count, substeps)


def test_saved_mesh_replay_comparison_never_masks_a_state_or_time_mismatch():
    values = dict(qpos=np.zeros((3, 7), np.float32), qvel=np.zeros((3, 6), np.float32),
                  ctrl=np.zeros((2, 0), np.float32), time=np.array([0., .01, .02], np.float32))
    actual = SimpleNamespace(**values)
    saved = {name: value.copy() for name, value in values.items()}
    assert compare_saved_replay(actual, saved)["all_exact_equal"]
    saved["qpos"][1, 0] += 1e-6
    result = compare_saved_replay(actual, saved)
    assert not result["all_exact_equal"] and result["qpos"]["max_abs_difference"] > 0
    saved["qpos"] = values["qpos"].copy()
    saved["time"] = np.zeros(1, np.float32)
    result = compare_saved_replay(actual, saved)
    assert not result["all_exact_equal"] and result["time"]["max_abs_difference"] is None
