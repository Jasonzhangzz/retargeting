"""Grasp migration invariants and independent force-closure checks."""
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from retarget.integrations.supermimic.dataset import BODY_NAMES, GraspDataset, relative_pose

ROOT = Path(__file__).resolve().parents[1]


def test_cached_relative_frame_round_trip():
    rotation = Rotation.from_euler("xyz", [.3, -.7, 1.1])
    q = np.r_[[.02, -.1, .06], rotation.as_quat(scalar_first=True), np.zeros(20)]
    result = relative_pose(q)
    np.testing.assert_allclose(rotation.apply(result[:3]) + q[:3], 0, atol=1e-12)
    np.testing.assert_allclose((rotation * Rotation.from_quat(result[3:], scalar_first=True)).as_matrix(), np.eye(3), atol=1e-12)


def test_complete_cache_split_and_contact_groups():
    path = ROOT / "output/supermimic_grasp_dataset_v2"
    if not path.exists():
        pytest.skip("Exported cache not present")
    ds = GraspDataset(path)
    assert len(ds.cases) == 161 and len(ds.records) == 3220
    assert len({c["type_id"] for c in ds.cases}) == 31
    assert not ds.manifest["success_filter"]
    for split, seeds in [("train", set(range(8))), ("validation", {8}), ("test", {9})]:
        for pool in ds.pools(split):
            assert {ds.records[i]["seed"] for i in pool} == seeds
            assert {ds.records[i]["method"] for i in pool} == {"admm", "mppi"}
    for i, case in enumerate(ds.cases):
        if case["type_id"] == 8:
            np.testing.assert_array_equal(ds.allowed[i], [1, 1, 1, 0, 0, 0])
        if case["type_id"] == 33:
            assert ds.required[i].sum() == 4  # Four phalanx groups, two fingers.
            assert ds.allowed[i].sum() == 2
    for record in ds.records:
        type_id = ds.cases[ds.ids[record["case_id"]]]["type_id"]
        assert (record["source"] == "soft_center_200_pi0") == (type_id in (6, 7, 8, 9))
        assert isinstance(record["historical_evaluation"]["task_success"], bool)
        assert len(record["historical_evaluation"]["metadata_sha256"]) == 64
    assert ds.arrays["original_grasp_qpos"].shape == (3220, 27)
    assert any(not r["historical_evaluation"]["task_success"] for r in ds.records)


def test_exact_closure_rejects_same_side_contacts_and_accepts_surrounding_contacts():
    pytest.importorskip("dex_supermimic")
    pytest.importorskip("mujoco_warp")
    from retarget.integrations.supermimic.evaluate import exact_closure
    points = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    result = exact_closure(points * .02, -points, .6, np.zeros(3), .02)
    assert result["success"] and result["rank"] == 6 and result["status"] == "Solved"
    bad = exact_closure(points * .02, np.tile([0., 0., 1.], (4, 1)), .6, np.zeros(3), .02)
    assert not bad["success"]
    assert not exact_closure(points[:2], -points[:2], .6, np.zeros(3), .02)["success"]


def test_fixed_pad_geometry_is_owned_by_distal_link():
    pytest.importorskip("dex_supermimic")
    pytest.importorskip("mujoco_warp")
    import mujoco
    from retarget.integrations.supermimic.contact import geom_labels
    from retarget.integrations.supermimic.environment import hand_spec
    model = hand_spec().compile()
    assert model.nq == 20 and model.nu == 20
    assert not np.any(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    owners, physical, _ = geom_labels(model)
    matched = 0
    for geom in range(model.ngeom):
        body = model.body(int(model.geom_bodyid[geom])).name
        if "tip_sensor_frame" in body:
            assert owners[geom] >= 0
            assert BODY_NAMES[owners[geom]].endswith("_link4")
            assert 0 <= physical[geom] < 5
            matched += 1
    assert matched >= 5


def test_reward_matches_recomputation_and_has_action_gradient():
    pytest.importorskip("dex_supermimic")
    pytest.importorskip("mujoco_warp")
    import torch
    from retarget.integrations.supermimic.environment import rewards, CONTEXT_DIM
    from dex_supermimic.rl.dex_supermimic.state import pack
    batch = 3
    x = pack(torch.zeros(batch, 20), torch.zeros(batch, 20), torch.zeros(batch, 3),
             torch.eye(3).expand(batch, 3, 3), torch.zeros(batch, 3), torch.zeros(batch, 3))
    context = torch.zeros(batch, CONTEXT_DIM)
    context[:, 200:209] = torch.eye(3).flatten()
    context[:, 171] = 1
    aux = torch.zeros(batch, 270)
    aux[:, 0] = .1
    aux[:, 12:268] = .1
    action = torch.full((batch, 20), .1, requires_grad=True)
    a, _ = rewards(x, context, aux, action, torch.zeros_like(action))
    b, _ = rewards(x, context, aux, action, torch.zeros_like(action))
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    (-a.mean()).backward()
    assert action.grad.isfinite().all() and action.grad.abs().sum() > 0


def test_control_respects_limits_even_with_invalid_cached_controller_memory():
    pytest.importorskip("dex_supermimic")
    pytest.importorskip("mujoco_warp")
    import torch
    from retarget.integrations.supermimic.environment import control, CONTEXT_DIM
    context = torch.zeros(2, CONTEXT_DIM)
    context[:, 177:197] = 2
    target = control(torch.zeros(2, 20), torch.full((2, 20), 3.), context,
                     torch.full((20,), -1.), torch.ones(20))
    torch.testing.assert_close(target, torch.ones_like(target))
