"""Grasp input adapters and stationary reference integration checks."""

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch
import trimesh
import warp as wp
from scipy.spatial.transform import Rotation

from retarget.config.run import ROOT
from retarget.env.grasp.initialization import sample_original_poses
from retarget.initialization.dexonomy.data import hand_loader, obj_loader
from retarget.initialization.dexonomy.util.np_util import np_transform_points


def _mapped_template(tmp_path):
    contacts = np.array(
        [[-.04, 0, 0, 1, 0, 0], [.04, 0, 0, -1, 0, 0]], np.float32
    )
    path = tmp_path / "mapped_template.npy"
    np.save(
        path,
        dict(
            grasp_qpos=np.r_[np.zeros(3), 1., np.zeros(3), np.zeros(20)].astype(
                np.float32
            ),
            hand_cpn_w=contacts,
            n_evo=np.int64(0),
        ),
    )
    loader = object.__new__(hand_loader.HandTemplateLoader)
    loader.skt_data = np.empty((0, 6), np.float32)
    loader.skt_bid = np.empty(0, int)
    fk = SimpleNamespace(
        forward_kinematics=lambda _: (
            np.eye(3, dtype=np.float32)[None], np.zeros((1, 3), np.float32)
        )
    )
    previous = np.random.get_state()
    try:
        np.random.seed(42)
        prepared = loader._load_data(fk, str(path))
    finally:
        np.random.set_state(previous)
    return contacts, prepared


def test_retained_initialization_functions_match_recorded_upstream_ast():
    directory = ROOT / "retarget/initialization/dexonomy/data"
    source = json.loads((directory / "source.json").read_text())
    for qualified, record in source["functions"].items():
        tree = ast.parse((directory / record["path"]).read_text())
        function = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == qualified.rsplit(".", 1)[-1]
        )
        actual = hashlib.sha256(
            ast.dump(function, include_attributes=False).encode()
        ).hexdigest()
        assert actual == record["function_ast_sha256"]


def test_template_contact_frame_roundtrip_and_deterministic_primitive_sampling(tmp_path):
    contacts, prepared = _mapped_template(tmp_path)
    rotation = prepared["rot_c2h"].T
    restored = np_transform_points(
        prepared["hand_cpn_c"], rotation, -rotation @ prepared["trans_c2h"]
    )
    np.testing.assert_allclose(restored, contacts, atol=1e-8)
    assert prepared["hand_skt_h"].shape == (0, 6)

    mesh = trimesh.creation.box(extents=[.07, .07, .12])
    first = sample_original_poses(obj_loader, mesh, 4, 4, 13)
    second = sample_original_poses(obj_loader, mesh, 4, 4, 13)
    assert first[0].shape == (16, 3, 3)
    assert first[1].shape == (16, 3)
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)


def test_experiment_persists_fixed_template_points_and_stationary_object(tmp_path, monkeypatch):
    from retarget.env.grasp import hand as hand_module
    from retarget.env.grasp.runtime import prepare_experiment

    monkeypatch.setattr(hand_module, "model_hand", lambda _: "wuji2")
    monkeypatch.setattr(
        hand_module, "hand_tree",
        lambda _: ET.fromstring("<mujoco><option/><asset/><worldbody/></mujoco>"),
    )
    config = tmp_path / "base.yaml"
    config.write_text("sequence: {}\n")
    local = np.array([[.001, .002, .003], [.004, .005, .006]], np.float32)
    bodies = np.array([1, 2], np.int32)
    mesh = trimesh.creation.box(extents=[.07, .07, .12])
    hand = SimpleNamespace(
        model=object(), surface=SimpleNamespace(triangles=mesh.triangles),
        local=local.copy(), bodies=bodies.copy(),
        semantic_contact_masks=[np.array([True, False]), np.array([False, True, True])],
        semantic_patch_labels=["palm.palmar", "index.distal.palmar"],
        semantic_contact_local=local.copy(),
        semantic_contact_normals=np.array([[0, 1, 0], [0, 1, 0]]),
    )
    anchors = np.array([[-.035, 0, 0], [.035, 0, 0]], np.float32)
    guess = SimpleNamespace(
        hand_qpos=np.linspace(-.1, .1, 26), object_points=anchors,
        object_normals=np.array([[-1, 0, 0], [1, 0, 0]], np.float32),
    )
    args = SimpleNamespace(
        solver_config=config, object_mass=.2, friction=1., physics_reference=None,
    )
    experiment = prepare_experiment(tmp_path / "experiment", hand, guess, args)
    with np.load(experiment / "contactaware/contact_guidance.npz") as guidance:
        for frame in range(2):
            np.testing.assert_array_equal(guidance["hand_query_local_pos"][frame], local)
            np.testing.assert_array_equal(guidance["hand_query_body_ids"][frame], bodies)
            np.testing.assert_array_equal(guidance["contact_pos_obj"][frame], anchors)
    object_pose = np.load(experiment / "contactaware/object_pose_7.npy")
    np.testing.assert_array_equal(object_pose, np.tile([0, 0, 0, 1, 0, 0, 0], (2, 1)))
    np.testing.assert_array_equal(hand.local, local)
    np.testing.assert_array_equal(hand.bodies, bodies)
    with np.load(experiment / "card_input.npz", allow_pickle=False) as supplied:
        np.testing.assert_array_equal(supplied["semantic_patch_offsets"], [0, 2, 5])
        np.testing.assert_array_equal(supplied["semantic_patch_masks"], [True, False, False, True, True])
        np.testing.assert_array_equal(supplied["semantic_reference_local"], local)
        assert supplied["semantic_patch_labels"].tolist() == hand.semantic_patch_labels


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_retained_sampler_loader_and_matcher_execute_on_gpu(tmp_path):
    from retarget.initialization.dexonomy.op.gen_init import HandObjMatcher
    from retarget.initialization.dexonomy.util.warp_util import WarpCollisionEnv

    _, prepared = _mapped_template(tmp_path)
    mesh = trimesh.graph.smooth_shade(
        trimesh.creation.box(extents=[.07, .07, .12]),
        angle=np.pi / 4, facet_minarea=None,
    )
    rotations, translations = sample_original_poses(obj_loader, mesh, 4, 4, 0)
    count = len(translations)
    device = torch.device("cuda:0")

    def tensor(value):
        return torch.as_tensor(np.asarray(value).copy(), dtype=torch.float32, device=device)

    template = {
        key: tensor(prepared[key])[None, None]
        .expand(1, count, *np.shape(prepared[key])).contiguous()
        for key in ("grasp_qpos", "rot_c2h", "trans_c2h", "hand_cpn_c", "hand_skt_h", "n_evo")
    }
    target = dict(
        rot_o2c_init=tensor(rotations)[None],
        trans_o2c_init=tensor(translations)[None],
        obj_scale=tensor([[1, 1, 1]]),
        pose_w2so=tensor([[0, 0, 0, 1, 0, 0, 0]]),
        ext_center=tensor([mesh.center_mass]), ext_wrench=tensor(np.zeros((1, 6))),
    )
    environment = WarpCollisionEnv(device)
    environment.load(None, [("synthetic_cube", mesh)])
    result = HandObjMatcher(SimpleNamespace(opt_step=3), device, environment).forward(
        template, target
    )
    torch.cuda.synchronize()
    assert result["grasp_pose"].shape == (16, 7)
    assert all(bool(torch.isfinite(value).all()) for value in result.values())


def _measure_synthetic_gaps(device, offsets, rotation=None, translation=None):
    from retarget.task.grasp.tracking_cost import measure_gaps

    mesh_data = trimesh.creation.box(extents=[.07, .08, .1])
    mesh = wp.Mesh(
        points=wp.array(mesh_data.vertices, dtype=wp.vec3, device=device),
        indices=wp.array(mesh_data.faces.reshape(-1), dtype=int, device=device),
    )
    rotation = np.eye(3) if rotation is None else rotation
    translation = np.zeros(3) if translation is None else translation
    count = len(offsets)
    positions = np.zeros((count, 2, 3), np.float32)
    positions[:, 0] = translation
    positions[:, 1] = translation + np.asarray(offsets)[:, None] * rotation[:, 0]
    matrices = np.broadcast_to(rotation, (count, 2, 3, 3)).copy()
    distance = wp.empty((count, 1), dtype=float, device=device)
    active = wp.zeros((count, 1), dtype=int, device=device)
    wp.launch(
        measure_gaps, (count, 1),
        inputs=[
            mesh.id, 0,
            wp.array([1], dtype=int, device=device),
            wp.array([0], dtype=int, device=device),
            wp.array([3], dtype=int, device=device),
            wp.array([[0, 0, 0], [.001, 0, 0], [-.001, 0, 0]], dtype=wp.vec3, device=device),
            wp.array(positions, dtype=wp.vec3, device=device),
            wp.array(matrices, dtype=wp.mat33, device=device), distance, active,
        ],
        device=device,
    )
    return distance.numpy()[:, 0], active.numpy()[:, 0]


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_tracking_mesh_gaps_preserve_separation_and_object_frame_when_split(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    offsets = np.array([.04, .08, .02, .035])
    combined, active = _measure_synthetic_gaps(device, offsets)
    np.testing.assert_allclose(combined, [.004, .044, -.016, -.001], atol=3e-8)
    np.testing.assert_array_equal(active, np.ones(4))
    split = np.concatenate([
        _measure_synthetic_gaps(device, offsets[:2])[0],
        _measure_synthetic_gaps(device, offsets[2:])[0],
    ])
    np.testing.assert_array_equal(combined, split)
    transformed, _ = _measure_synthetic_gaps(
        device, offsets, Rotation.from_euler("xyz", [.4, -.6, .7]).as_matrix(),
        np.array([.2, -.3, .15]),
    )
    np.testing.assert_allclose(transformed, combined, atol=3e-8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_tracking_mesh_gap_cpu_and_gpu_agree():
    offsets = np.array([.04, .08, .02, .035])
    cpu, _ = _measure_synthetic_gaps("cpu", offsets)
    gpu, _ = _measure_synthetic_gaps("cuda:0", offsets)
    np.testing.assert_allclose(cpu, gpu, rtol=1e-5, atol=3e-8)
