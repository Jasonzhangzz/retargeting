"""Robot geometry/frame invariants required by the native GraspADMM bridge."""

import numpy as np
import pytest

pytest.importorskip("qpsolvers")
pytest.importorskip("rtree")
import mujoco
import trimesh

from retarget.env.grasp.hand import hand_tree, HAND_BODIES
from retarget.env.grasp.graspadmm_scene import (
    write_scene, WujiOptEnv, to_upstream_qpos, from_upstream_qpos,
    project_to_hand, required_body_groups,
)
from retarget.env.grasp.graspadmm_source import require_upstream


@pytest.fixture
def scene(tmp_path):
    mesh = tmp_path / "object.obj"
    trimesh.creation.icosphere(subdivisions=1, radius=.03).export(mesh)
    return write_scene(tmp_path / "scene", mesh)


def test_free_root_matches_existing_wuji2_fk(scene):
    import xml.etree.ElementTree as ET
    source = require_upstream()
    local = mujoco.MjModel.from_xml_string(ET.tostring(hand_tree("wuji2"), encoding="unicode"))
    data = mujoco.MjData(local)
    qpos = np.r_[[.12, -.03, .15, .4, -.2, .3], np.full(20, .12)]
    data.qpos[:] = qpos
    mujoco.mj_forward(local, data)
    env = WujiOptEnv(source.sim.HandCfg(str(scene.hand_xml), freejoint=True), scene.scene_cfg)
    env.reset_qpos(to_upstream_qpos(qpos))
    assert env._model.nq == 27
    assert env._model.nu == 20
    np.testing.assert_allclose(from_upstream_qpos(to_upstream_qpos(qpos)), qpos, atol=1e-12)
    for body in HAND_BODIES:
        original = local.body(body).id
        moved = env._model.body("hand-" + body).id
        np.testing.assert_allclose(env._data.xpos[moved], data.xpos[original], atol=1e-8)
        np.testing.assert_allclose(env._data.xmat[moved], data.xmat[original], atol=1e-8)


def test_projection_includes_fixed_sensor_pad(scene):
    source = require_upstream()
    env = WujiOptEnv(source.sim.HandCfg(str(scene.hand_xml), freejoint=True), scene.scene_cfg)
    env.reset_qpos(np.r_[np.zeros(3), 1., np.zeros(23)])
    meshes = source.core.create_hand_trimesh(env)
    proxies = [trimesh.proximity.ProximityQuery(mesh) for mesh in meshes]
    pad = env._model.body("hand-r_index_finger_tip_sensor_frame").id
    geom = np.flatnonzero((env._model.geom_bodyid == pad) & (env._model.geom_contype != 0))[0]
    mesh = int(env._model.geom_dataid[geom])
    start, count = env._model.mesh_vertadr[mesh], env._model.mesh_vertnum[mesh]
    vertices = env._model.mesh_vert[start:start + count]
    points = vertices @ env._data.geom_xmat[geom].reshape(3, 3).T + env._data.geom_xpos[geom]
    names = ["right_finger2_link4"] * len(points)
    metal, _ = source.core.project_to_hand(points, names, env, meshes, proxies, "collision")
    selected = int(np.linalg.norm(metal - points, axis=1).argmax())
    assert np.linalg.norm(metal[selected] - points[selected]) > .001
    projected, body_local = project_to_hand(points[selected:selected+1], names[:1], env, meshes, proxies, "collision")
    np.testing.assert_allclose(projected[0], points[selected], atol=1e-8)
    assert np.isfinite(body_local).all()


def test_required_groups_only_allow_annotated_links():
    contacts = ["right_finger1_link4", "right_finger2_link3", "right_finger2_link4", "right_palm_link"]
    assert required_body_groups(contacts) == [
        ["right_finger1_link4"], ["right_finger2_link3", "right_finger2_link4"], ["right_palm_link"]]


def test_fixed_finger_cache_matches_fresh_conversion_bitwise(tmp_path):
    import torch
    import pickle
    import random
    if not torch.cuda.is_available():
        pytest.skip("The original fixed-template conversion requires CUDA")
    import xml.etree.ElementTree as ET
    from types import SimpleNamespace
    from retarget.config.run import ROOT
    from retarget.task.grasp.dexonomy import load_template, anatomical_joints
    from retarget.env.grasp.graspadmm_inputs import fixed_finger_angles

    torch.set_num_threads(1)
    model = mujoco.MjModel.from_xml_string(ET.tostring(hand_tree("wuji2"), encoding="unicode"))
    args = SimpleNamespace(dexonomy_root=ROOT / "data/grasp/templates", init_device="cuda:0")
    shadow, posed, _, annotation = load_template(args.dexonomy_root, "11_Power_Sphere")
    joints = anatomical_joints(shadow, posed)[None]
    def random_state():
        return (pickle.dumps(np.random.get_state()), torch.get_rng_state().numpy().tobytes(),
                torch.cuda.get_rng_state(args.init_device).numpy().tobytes(), random.getstate())
    before = random_state()
    fresh, _ = fixed_finger_angles(model, joints, args, annotation, cache=False)
    first, first_info = fixed_finger_angles(model, joints, args, annotation, cache_root=tmp_path / "cache")
    cached, cached_info = fixed_finger_angles(model, joints, args, annotation, cache_root=tmp_path / "cache")
    assert not first_info["cache_hit"] and cached_info["cache_hit"]
    np.testing.assert_array_equal(fresh, first)
    np.testing.assert_array_equal(first, cached)
    assert cached_info["template_conversion_seconds"] == 0.
    assert before == random_state()
    assert list((tmp_path / "cache").glob("*/finger_qpos.npy"))
