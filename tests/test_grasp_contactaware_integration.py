"""ContactAware frontend acceptance and solved-reference handoff to MPPI."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp
from scipy.spatial.transform import Rotation

from retarget.config.run import ROOT, load
from retarget.env.grasp.initialization import GraspInitialization


def test_contactaware_failure_cannot_be_accepted_by_shape_first_initialization(tmp_path, monkeypatch):
    from retarget.env.grasp import inputs, initialization, semantic_patches, seed, template
    from retarget.env.grasp import tip_patch, orientation, contactaware
    from retarget.task.grasp import wrist_prior

    seed_guess = GraspInitialization(
        np.zeros(26), np.zeros((1, 3)), np.array([[0., 1., 0.]]), {},
        dict(initialization_valid=True, contacts_optional=True, initialization_policy="shape_first"),
    )
    calls = []

    def fail_contactaware(hand, guess, args, **kwargs):
        calls.append(guess)
        result = copy.deepcopy(guess)
        result.metadata.update(initialization_valid=False, contacts_optional=False,
            initialization_policy="dexforge_contactaware_strict_required_fingers",
            contactaware=dict(success=False), failure_reasons=["contactaware_reference_infeasible"])
        return result

    def forbidden_old_path(*args, **kwargs):
        pytest.fail("ContactAware mode must not use optional shape-first acceptance")

    monkeypatch.setattr(inputs, "build_scene", lambda *a, **k: (object(), object(), "<mujoco/>", np.zeros(3)))
    monkeypatch.setattr(inputs, "load_template", lambda *a: (object(), object(), [("rh_thdistal", np.zeros(6))], Path("template.yaml")))
    monkeypatch.setattr(inputs, "anatomical_joints", lambda *a: np.zeros((21, 3)))
    monkeypatch.setattr(inputs, "StaticHand", lambda model, surface, bodies, local, normals:
        SimpleNamespace(model=model, surface=surface, bodies=bodies, local=local.copy(), normals=normals.copy()))
    monkeypatch.setattr(semantic_patches, "match_semantic_patches", lambda *a:
        (np.array([1]), np.zeros((1, 3)), np.array([[0., 1., 0.]]),
         [dict(query_index=0)], [dict(id="thumb.distal.palmar", mask=np.array([True]))]))
    monkeypatch.setattr(tip_patch, "centered_template_contacts", lambda root, name, contacts: (contacts, []))
    monkeypatch.setattr(tip_patch, "configure_tip_centers", lambda *a, **k: None)
    monkeypatch.setattr(seed, "bone_seed_cuda", lambda *a, **k: (np.zeros((1, 26)), dict(method="seed only")))
    monkeypatch.setattr(orientation, "configure_orientation", lambda *a: None)
    monkeypatch.setattr(wrist_prior, "initialize_references", lambda *a: None)
    monkeypatch.setattr(wrist_prior, "configure_reference", lambda *a: None)
    monkeypatch.setattr(initialization, "global_alignment", lambda *a, **k: [seed_guess])
    monkeypatch.setattr(initialization, "validate_initialization", forbidden_old_path)
    # A template fit may improve the numerical seed, but it cannot accept the
    # final reference after the ContactAware stage reports failure.
    monkeypatch.setattr(template, "fit_contact_template", lambda *a, **k:
        (np.asarray(a[4]).copy(), dict(method="template numerical seed only", surface_query_indices=[0])))
    monkeypatch.setattr(contactaware, "refine_grasp_reference", fail_contactaware)
    args = SimpleNamespace(
        reference_method="dexforge_contactaware", contact_mapping="semantic_patch",
        _tasks=[(Path("primitive.obj"), "1_Large_Diameter", None)], seed=0,
        scale=1., hand_model="wuji2", dexonomy_root=tmp_path, init_device="cuda:0",
    )
    rows = list(inputs.initialize_cases(args, tmp_path))
    assert len(calls) == 1 and len(rows) == 1
    result = rows[0][1][0]
    assert not result.metadata["initialization_valid"]
    assert result.metadata["failure_reasons"] == ["contactaware_reference_infeasible"]
    assert not result.metadata["contacts_optional"]
    assert not hasattr(result, "contactaware_reference_qpos")
    assert not hasattr(result, "hand_reference_qpos")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_solved_contactaware_reference_survives_real_wuji_runtime_and_hand_tracking(tmp_path):
    """Use the saved real SQP smoke result; do not rerun a large SQP or MPPI."""
    from retarget.env.grasp.inputs import StaticHand, build_scene
    from retarget.env.grasp.runtime import GraspRuntime, prepare_experiment
    from retarget.env.grasp.runner import arguments
    from retarget.env.grasp.rollout import GraspProblem
    from retarget.env.geometry import ObjectMeshSurface

    source_cases = list((ROOT / "output/grasp_tracking_mppi_semantic").glob("001*/*/input_00"))
    solved_archive = Path("/tmp/grasp_ca_result.npz")
    solved_report = Path("/tmp/grasp_ca_result.json")
    if not source_cases or not solved_archive.is_file() or not solved_report.is_file():
        pytest.skip("Run the real ContactAware primitive smoke to create the solved-reference artifacts")
    source = source_cases[0]
    report = json.loads(solved_report.read_text())
    ca = report["contactaware"]
    if not ca["success"]:
        pytest.skip("The available ContactAware smoke did not produce an accepted reference")
    assert ca["source_commit"] == "fa6d671948ef03d18514062748b94e445b63cd49"
    assert all(row["contact"] for row in ca["required_finger_contact"].values())
    solved = np.load(solved_archive, allow_pickle=False)
    model, surface, _, _ = build_scene(source / "object.obj", hand_model="wuji2")
    raw = np.load(source / "card_input.npz", allow_pickle=False)
    guidance = np.load(source / "contactaware/contact_guidance.npz", allow_pickle=False)
    matching = json.loads((source.parent / "semantic_matching.json").read_text())
    hand = StaticHand(model, surface, guidance["hand_query_body_ids"][0],
        guidance["hand_query_local_pos"][0], np.array([r["normal"] for r in matching["fitted_contacts"]]))
    offsets, packed = raw["semantic_patch_offsets"], raw["semantic_patch_masks"]
    hand.semantic_contact_masks = [packed[offsets[i]:offsets[i+1]].copy() for i in range(len(hand.bodies))]
    hand.semantic_patch_labels = raw["semantic_patch_labels"].tolist()
    hand.semantic_contact_local, hand.semantic_contact_normals = hand.local.copy(), hand.normals.copy()
    hand.contact_reference_normals = raw["region_reference_normals"].copy()
    q_reference = solved["q"].copy()
    template_seed = np.load(source / "contactaware/wuji_right_qpos.npy")[0]
    assert not np.allclose(q_reference, template_seed, atol=1e-3)
    actual = q_reference.copy()
    # The reference and actual initial configuration have distinct roles.
    joint = 6 + int(np.argmax(model.jnt_range[6:, 1] - q_reference[6:]))
    actual[joint] += .01
    guess = GraspInitialization(actual, solved["object_points"].copy(), solved["object_normals"].copy(), {}, report)
    guess.contactaware_reference_qpos = q_reference.copy()
    guess.hand_reference_qpos = q_reference.copy()
    guess.contact_mask = np.asarray([r["active"] for r in ca["per_patch"]], np.float32)
    args = arguments(load(ROOT / "configs/grasp_contactaware_mppi.yaml"))
    args.worlds, args.explicit_horizon = 1, 1
    experiment = prepare_experiment(tmp_path / "experiment", hand, guess, args)
    with np.load(experiment / "card_input.npz", allow_pickle=False) as persisted:
        np.testing.assert_allclose(persisted["hand_reference_qpos"], q_reference)
        np.testing.assert_allclose(persisted["hand_initial_qpos"], actual)
        np.testing.assert_allclose(persisted["object_points"], solved["object_points"])
        np.testing.assert_array_equal(persisted["reference_contact_mask"], guess.contact_mask)
    wp.init(); wp.set_device("cuda:0")
    runtime = GraspRuntime(experiment, args)
    np.testing.assert_allclose(runtime.qpos[runtime.index.hand_qpos], actual, atol=1e-7)
    with np.load(experiment / "contactaware/contact_guidance.npz", allow_pickle=False) as exported:
        np.testing.assert_array_equal(exported["contact_mask"][0], guess.contact_mask)
    np.testing.assert_array_equal(np.load(experiment / "contactaware/object_pose_7.npy"),
        np.tile([0, 0, 0, 1, 0, 0, 0], (2, 1)))
    runtime_surface = ObjectMeshSurface.from_model(runtime.model, runtime.index.object_body)
    vertices, inverse = np.unique(runtime_surface.triangles.reshape(-1, 3), axis=0, return_inverse=True)
    mesh = wp.Mesh(points=wp.array(vertices, dtype=wp.vec3, device="cuda:0"),
        indices=wp.array(inverse, dtype=int, device="cuda:0"))
    runtime.audit_context = (SimpleNamespace(mesh=mesh), hand, args)
    with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream())):
        problem = GraspProblem(runtime)
        candidate = problem.mean.clone()
        candidate[:26] = torch.as_tensor(q_reference, device=candidate.device)
        r_initial = Rotation.from_euler("XYZ", actual[3:6])
        candidate[3:6] = torch.as_tensor((r_initial.inv() * Rotation.from_euler("XYZ", q_reference[3:6])).as_rotvec(), device=candidate.device)
        accepted_reference = candidate.clone()
        problem.evaluate(candidate[None]); torch.cuda.synchronize()
        at_reference = float(problem.tracking.step_parts.numpy()[0, 0, 4])
        object_at_reference = float(problem.tracking.step_parts.numpy()[0, 0, 3])
        assert at_reference < 1e-3
        assert problem.tracking.metadata["hand_reference_source"] == "dexforge_contactaware"
        np.testing.assert_allclose(problem.tracking.job.target.action.numpy(), q_reference, atol=1e-7)
        candidate[joint] += .02
        problem.evaluate(candidate[None]); torch.cuda.synchronize()
        perturbed = float(problem.tracking.step_parts.numpy()[0, 0, 4])
        assert perturbed > at_reference + .1
        assert float(problem.tracking.step_parts.numpy()[0, 0, 3]) == pytest.approx(object_at_reference, abs=1e-3)
        np.testing.assert_allclose(problem.tracking.job.target.action.numpy(), q_reference, atol=1e-7)
        # The same geometrically valid reference is unacceptable when its
        # inner contact iteration is deliberately left numerically unfinished.
        problem.mean.copy_(accepted_reference)
        problem.explicit.settings["contact_iterations"] = 1
        problem.explicit.settings["contact_check_interval"] = 1
        problem.explicit.contact_tolerance_mm = 1e-9
        unfinished = problem.evaluate(problem.mean[None]); torch.cuda.synchronize()
        assert float(problem.cost.penetration.numpy()[0]) <= 3
        assert float(problem.cost.palm_gap.numpy()[0]) >= 6
        assert float(problem.cost.self_gap.numpy()[0]) >= problem.cost.self_clearance
        assert float(problem.cost.forbidden_gap.numpy()[0]) >= 1
        assert np.isfinite(problem.tracking.cost.numpy()[0])
        assert np.max(problem.explicit.contact_residual.numpy()[0]) > 1e-9
        assert not bool(torch.isfinite(unfinished.cost[0]))
        assert bool(torch.isfinite(unfinished.ranking[0]))
        # Disabling the convergence gate leaves this exact candidate finite,
        # proving rejection came from the numerical test rather than geometry.
        problem.explicit.contact_tolerance_mm = None
        unchecked = problem.evaluate(problem.mean[None]); torch.cuda.synchronize()
        assert bool(torch.isfinite(unchecked.cost[0]))
