"""Native FA collision conversion preserves the actual Wuji mechanical model."""

from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from retarget.config.run import ROOT
from retarget.env.grasp.forceaware_scene import _canonical_tree, prepare_scene
from retarget.env.grasp.hand import HAND_BODIES, query_path


@pytest.fixture
def scene(tmp_path):
    paths = sorted((ROOT / "output/grasp_contactaware_mppi").glob("001*/*/input_00/runtime_scene.xml"))
    if not paths:
        pytest.skip("A successful primitive ContactAware input is required")
    return prepare_scene(paths[0], tmp_path / "native")


def test_native_conversion_preserves_body_order_fk_inertia_and_servos(scene):
    source, viewer = scene.source_model, scene.model
    assert (viewer.nq, viewer.nv, viewer.nu) == (33, 32, 26)
    assert [viewer.body(i).name for i in range(viewer.nbody)] == [source.body(i).name for i in range(source.nbody)]
    for field in ("body_pos", "body_quat", "body_mass", "body_inertia", "body_ipos", "body_iquat",
                  "body_gravcomp", "jnt_range", "jnt_axis", "jnt_pos", "dof_armature", "dof_damping",
                  "actuator_gainprm", "actuator_biasprm", "actuator_gear", "actuator_ctrlrange",
                  "actuator_forcerange", "actuator_ctrllimited", "actuator_forcelimited", "geom_friction",
                  "geom_contype", "geom_conaffinity"):
        np.testing.assert_allclose(getattr(viewer, field), getattr(source, field), rtol=0, atol=1e-14, err_msg=field)
    left, right = mujoco.MjData(source), mujoco.MjData(viewer)
    q = source.qpos0.copy()
    q[:26] += np.random.default_rng(23).normal(0, 0.03, 26)
    left.qpos[:] = right.qpos[:] = q
    mujoco.mj_kinematics(source, left)
    mujoco.mj_kinematics(viewer, right)
    np.testing.assert_allclose(left.xpos, right.xpos, rtol=0, atol=1e-14)
    np.testing.assert_allclose(left.xmat, right.xmat, rtol=0, atol=1e-14)


def test_mesh_replay_keeps_source_integrator_while_native_uses_euler(scene):
    source, viewer = scene.source_model, scene.model
    native_option = ET.parse(scene.xml).find("option")
    assert source.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert viewer.opt.integrator == source.opt.integrator
    assert native_option.get("integrator") == "Euler"
    assert scene.manifest["source_integrator"] == scene.manifest["viewer_integrator"] == "implicitfast"
    assert scene.manifest["native_integrator"] == "Euler"
    assert viewer.opt.timestep == scene.manifest["viewer_timestep"] == scene.manifest["native_timestep"] == .01
    assert float(native_option.get("timestep")) == .01
    for name in ("cone", "iterations", "ls_iterations", "solver", "disableflags", "enableflags"):
        assert getattr(viewer.opt, name) == getattr(source.opt, name)
    for name in ("gravity", "wind", "magnetic"):
        np.testing.assert_array_equal(getattr(viewer.opt, name), getattr(source.opt, name))


@pytest.mark.parametrize("integrator", ["Euler", "RK4", "implicit", "implicitfast"])
def test_canonical_viewer_preserves_mujoco_integrator_values(tmp_path, integrator):
    source = tmp_path / "original.xml"
    source.write_text(f'''<mujoco><option integrator="{integrator}" timestep=".003" cone="elliptic"/>
        <worldbody><body name="body"><freejoint name="free"/>
        <geom name="sphere" type="sphere" size=".01" mass="1"/></body></worldbody></mujoco>''')
    model = mujoco.MjModel.from_xml_path(str(source))
    destination = tmp_path / "prepared"; destination.mkdir()
    _, viewer, _, _ = _canonical_tree(source, model, destination, .002)
    prepared = mujoco.MjModel.from_xml_path(str(viewer))
    assert prepared.opt.integrator == model.opt.integrator
    assert prepared.opt.cone == model.opt.cone
    assert prepared.opt.timestep == .002


def test_all_source_skin_and_fixed_pad_components_are_tangent_spheres(scene):
    assert set(scene.sphere_assets) == set(HAND_BODIES)
    records = {row["owner_body"]: row for row in scene.manifest["converted_hand_surfaces"]}
    for name, path in scene.sphere_assets.items():
        with np.load(query_path(scene.source_model, name), allow_pickle=False) as raw, np.load(path) as cloud:
            original = raw["local_pos"]
            normals = raw["local_normal"].astype(np.float64)
            normals /= np.linalg.norm(normals, axis=1, keepdims=True)
            assert len(cloud["local_pos"]) == len(original)
            assert np.all(cloud["radius"] > 0)
            np.testing.assert_allclose(cloud["local_pos"] + cloud["radius"][:, None] * normals, original, atol=2e-8)
            np.testing.assert_array_equal(cloud["group_id"], 0)
        assert not records[name]["semantic_filter_applied"]
    assert "r_mount" in records["right_palm_link"]["rigid_component_bodies"]
    for finger in range(1, 6):
        assert any("tip_sensor_frame" in body for body in records[f"right_finger{finger}_link4"]["rigid_component_bodies"])


def test_native_contract_separates_viewer_masks_and_closes_object_mesh(scene):
    root = ET.parse(scene.xml).getroot()
    clouds = root.findall(".//geom[@type='gs']")
    assert len(clouds) == 21
    assert all((geom.get("contype"), geom.get("conaffinity")) == ("1", "4") for geom in clouds)
    target = root.find(".//geom[@type='sdfmesh']")
    assert target is not None and target.get("mesh") is None
    assert (target.get("contype"), target.get("conaffinity")) == ("4", "1")
    assert scene.manifest["object_mesh"]["signed_volume_m3"] > 0
    assert "Native GS hand-hand pairs disabled" in scene.manifest["hand_self_collision"]


def test_native_loader_has_unique_complete_object_contact_outputs(scene):
    if sys.version_info[:2] != (3, 10):
        pytest.skip("Run this native contract check in .venv-forceaware")
    from retarget.dynamics.dexforge_upstream import activate_upstream, require_environment

    require_environment()
    activate_upstream()
    import warp as wp
    import comfree_warp

    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("CUDA required for native FA load")
    device, native = comfree_warp.load_model(str(scene.xml), device="cuda:0")
    assert (native.nq, native.nv, native.nu) == (33, 32, 26)
    assert native.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER
    assert scene.model.opt.integrator == scene.source_model.opt.integrator
    for field in ("body_pos", "body_quat", "body_mass", "body_inertia", "body_ipos", "body_iquat",
                  "jnt_range", "actuator_gainprm", "actuator_biasprm", "actuator_forcerange"):
        np.testing.assert_allclose(getattr(native, field), getattr(scene.source_model, field), rtol=0, atol=1e-14)
    batches = device.gaussian_collision if isinstance(device.gaussian_collision, tuple) else (device.gaussian_collision,)
    assert len(batches) == 1
    batch = batches[0]
    assert int(batch.target_body_id) == native.body("grasp_object").id
    bodies = batch.source_body_ids.numpy()[batch.output_starts.numpy()]
    assert len(bodies) == len(np.unique(bodies)) == 21
    assert set(bodies) == {native.body(name).id for name in HAND_BODIES}
    assert list(device.mesh_targets) == [native.body("grasp_object").id]
    from comfree_warp.native_adjoint.kinematics import compile_kinematics, allocate_output, kinematics

    compiled = compile_kinematics(native, device)
    pose = allocate_output(compiled, 1, requires_grad=False)
    q = native.qpos0.copy().astype(np.float32)
    q[:26] += np.random.default_rng(31).normal(0, .02, 26).astype(np.float32)
    kinematics(compiled, wp.array(q[None], dtype=float, device="cuda:0"), pose)
    data = mujoco.MjData(scene.source_model)
    data.qpos[:] = q
    mujoco.mj_kinematics(scene.source_model, data)
    np.testing.assert_allclose(pose.body_position.numpy()[0], data.xpos, rtol=0, atol=2e-7)
    np.testing.assert_allclose(pose.body_matrix.numpy()[0], data.xmat.reshape(-1, 3, 3), rtol=0, atol=1e-6)
