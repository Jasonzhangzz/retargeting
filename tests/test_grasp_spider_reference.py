"""Real CA-input immutability and ordinary WUJI model serialization checks.

These tests do not rerun CA, MPPI or simulation, and never claim a modified
format probe is a newly accepted physical grasp. Everything runs on CPU.
"""

import hashlib
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys

import mujoco
import numpy as np
import pytest

from retarget.config.run import ROOT
from retarget.env.grasp import spider_reference
from retarget.env.grasp.forceaware_reference import _read_ca


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_snapshot(source):
    return {str(path.relative_to(source)): _hash(path)
            for path in source.rglob("*") if path.is_file()}


@pytest.fixture
def accepted_mid():
    candidates = [ROOT / "output/grasp_contactaware_feature_normals_final/001_radius_0.03500/1_Large_Diameter/input_00",
                  ROOT / "output/grasp_contactaware_feature_normals05/001_radius_0.03500/1_Large_Diameter/input_00"]
    candidates += sorted((ROOT / "output/grasp_contactaware_mppi").glob("001*/*/input_00"))
    for source in candidates:
        if (source / "card_input.npz").is_file() and (source / "contactaware_reference.json").is_file():
            report = json.loads((source / "contactaware_reference.json").read_text())
            if report.get("success") is True:
                return source.resolve()
    pytest.skip("An accepted real middle-size WUJI2 ContactAware input is required")


def test_reference_is_exact_ca_configuration_and_only_original_active_slots(accepted_mid):
    before = _source_snapshot(accepted_mid)
    _, card, report, guidance, poses = _read_ca(accepted_mid)
    reference = spider_reference.read_input(accepted_mid)
    slots = np.flatnonzero(card["reference_contact_mask"])
    assert reference.source == accepted_mid
    assert reference.qpos.shape == (33,) and reference.qvel.shape == (32,)
    np.testing.assert_array_equal(reference.qpos[:26], card["hand_reference_qpos"])
    np.testing.assert_array_equal(reference.qpos[26:], np.r_[poses[0, 4:7], poses[0, 3], poses[0, :3]])
    np.testing.assert_array_equal(reference.qvel, 0.)
    np.testing.assert_array_equal(reference.active_slots, slots)
    assert len(slots) == 5
    np.testing.assert_array_equal(reference.contact_bodies, guidance["hand_query_body_ids"][0, slots])
    np.testing.assert_array_equal(reference.contact_local, guidance["hand_query_local_pos"][0, slots])
    np.testing.assert_array_equal(reference.object_local, guidance["contact_pos_obj"][0, slots])
    assert all(row["contact"] for row in report["required_finger_contact"].values())
    assert _source_snapshot(accepted_mid) == before
    assert all(_hash(path) == digest for path, digest in reference.source_hashes.items())


def test_subset_and_palm_preload_preserves_ca_contacts(accepted_mid, tmp_path, monkeypatch):
    from retarget.env.grasp import forceaware_preload
    reference = spider_reference.read_input(accepted_mid)
    model = mujoco.MjModel.from_xml_path(str(reference.scene))
    # Private format probes; no new accepted physical grasp is asserted.
    original = reference.qpos.copy()
    calls = []
    def proposal(artifacts, cfg, **kwargs):
        calls.append(kwargs)
        return np.zeros(26), {"static_qp_only": True}
    monkeypatch.setattr(forceaware_preload, "prepare_preload", proposal)
    partial = replace(reference, contact_bodies=reference.contact_bodies[:2])
    spider_reference.preload(partial, tmp_path/"prepared.xml", 5)
    assert calls[-1]["require_all_fingers"] is False
    np.testing.assert_array_equal(calls[-1]["minimum_per_contact"], [.1, .1])
    palmar = replace(reference, contact_bodies=np.asarray([model.body("right_palm_link").id]))
    spider_reference.preload(palmar, tmp_path/"prepared.xml", 5)
    np.testing.assert_array_equal(calls[-1]["minimum_per_contact"], [.1])
    np.testing.assert_array_equal(reference.qpos, original)


def test_seed_angles_cannot_replace_ca_reference_and_missing_reference_fails(accepted_mid, tmp_path):
    source = tmp_path / "input"
    shutil.copytree(accepted_mid, source)
    with np.load(source / "card_input.npz", allow_pickle=False) as archive:
        card = {name: archive[name].copy() for name in archive.files}
    ca_reference = card["hand_reference_qpos"].copy()
    # Format-only perturbation of non-reference fields in an independent copy.
    card["hand_initial_qpos"] = ca_reference + .37
    card["template_qpos"] = np.full(26, -99.)
    np.savez_compressed(source / "card_input.npz", **card)
    np.testing.assert_array_equal(spider_reference.read_input(source).qpos[:26], ca_reference)
    card.pop("hand_reference_qpos")
    np.savez_compressed(source / "card_input.npz", **card)
    with pytest.raises(ValueError, match="ContactAware hand reference"):
        spider_reference.read_input(source)


def test_nonidentity_ca_pose_layout_converts_xyzw_xyz_to_xyz_wxyz(accepted_mid, monkeypatch):
    source, card, report, guidance, original_poses = _read_ca(accepted_mid)
    # Pure serialization probe. No files are changed, and this pose is not
    # submitted to CA acceptance or represented as a new physical grasp.
    quaternion_xyzw = np.asarray([.1, -.2, .3, np.sqrt(.86)])
    translation = np.asarray([.04, -.05, .06])
    poses = np.repeat(np.r_[quaternion_xyzw, translation][None], len(original_poses), axis=0)
    monkeypatch.setattr(spider_reference, "_read_ca", lambda path: (source, card, report, guidance, poses))
    reference = spider_reference.read_input(accepted_mid)
    np.testing.assert_array_equal(reference.qpos[:26], card["hand_reference_qpos"])
    np.testing.assert_array_equal(reference.qpos[26:29], translation)
    np.testing.assert_array_equal(reference.qpos[29:], quaternion_xyzw[[3, 0, 1, 2]])


@pytest.mark.skipif(sys.version_info[:2] != (3, 10), reason="Pinned WUJI/SPIDER integration environment is Python3.10")
@pytest.mark.parametrize("dt,iterations,ls_iterations", [(.002, 20, 50), (.004, 7, 11)])
def test_prepared_model_preserves_mechanics_materials_frames_and_ca_source(
    accepted_mid, tmp_path, dt, iterations, ls_iterations,
):
    before = _source_snapshot(accepted_mid)
    reference = spider_reference.read_input(accepted_mid)
    original = mujoco.MjModel.from_xml_path(str(reference.scene))
    original_option = {name: getattr(original.opt, name).copy() if isinstance(getattr(original.opt, name), np.ndarray)
                       else getattr(original.opt, name)
                       for name in dir(original.opt) if not name.startswith("_")
                       and isinstance(getattr(original.opt, name), (int, float, np.ndarray))}
    prepared, path = spider_reference.prepare_model(reference, tmp_path / "prepared", dt,
        solver_iterations=iterations, ls_iterations=ls_iterations)
    assert path.is_file() and path.parent == tmp_path / "prepared"
    assert (prepared.nq, prepared.nv, prepared.nu) == (33, 32, 26)
    assert [prepared.body(i).name for i in range(prepared.nbody)] == [original.body(i).name for i in range(original.nbody)]
    assert [prepared.joint(i).name for i in range(prepared.njnt)] == [original.joint(i).name for i in range(original.njnt)]
    assert [prepared.actuator(i).name for i in range(prepared.nu)] == [original.actuator(i).name for i in range(original.nu)]
    for field in ("body_pos", "body_quat", "body_ipos", "body_iquat", "body_mass", "body_inertia", "body_gravcomp",
                  "body_parentid", "jnt_type", "jnt_qposadr", "jnt_dofadr", "jnt_axis", "jnt_pos", "jnt_range",
                  "jnt_limited", "jnt_stiffness", "dof_damping", "dof_armature", "dof_frictionloss",
                  "actuator_gainprm", "actuator_biasprm", "actuator_dynprm", "actuator_gear", "actuator_trnid",
                  "actuator_ctrlrange", "actuator_forcerange", "actuator_ctrllimited", "actuator_forcelimited",
                  "geom_friction", "geom_solref", "geom_solimp", "geom_solmix", "geom_margin", "geom_gap", "geom_condim",
                  "geom_contype", "geom_conaffinity", "geom_bodyid", "geom_type", "geom_priority",
                  "geom_pos", "geom_quat", "geom_size", "geom_dataid", "mesh_vert", "mesh_face",
                  "site_pos", "site_quat", "qpos0", "qpos_spring"):
        np.testing.assert_allclose(getattr(prepared, field), getattr(original, field), rtol=0, atol=1e-12, err_msg=field)
    assert prepared.opt.timestep == dt
    assert prepared.opt.iterations == iterations and prepared.opt.ls_iterations == ls_iterations
    assert prepared.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    # The real accepted scene already uses implicitfast; other options must
    # not change just because the canonical scene was serialized elsewhere.
    assert original.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    for name, value in original_option.items():
        if name not in {"timestep", "iterations", "ls_iterations"}:
            np.testing.assert_allclose(getattr(prepared.opt, name), value, rtol=0, atol=1e-12, err_msg="opt."+name)
    left, right = mujoco.MjData(original), mujoco.MjData(prepared)
    generator = np.random.default_rng(44717007)
    for noise in (np.zeros(26), generator.normal(0., .01, 26), generator.normal(0., .03, 26)):
        qpos = reference.qpos.copy(); qpos[:26] += noise
        left.qpos[:] = right.qpos[:] = qpos
        mujoco.mj_kinematics(original, left); mujoco.mj_kinematics(prepared, right)
        np.testing.assert_allclose(left.xpos, right.xpos, rtol=0, atol=1e-12)
        np.testing.assert_allclose(left.xmat, right.xmat, rtol=0, atol=1e-12)
        local_left = left.xpos[reference.contact_bodies] + np.einsum("nij,nj->ni",
            left.xmat[reference.contact_bodies].reshape(-1, 3, 3), reference.contact_local)
        local_right = right.xpos[reference.contact_bodies] + np.einsum("nij,nj->ni",
            right.xmat[reference.contact_bodies].reshape(-1, 3, 3), reference.contact_local)
        np.testing.assert_allclose(local_left, local_right, rtol=0, atol=1e-12)
    assert _source_snapshot(accepted_mid) == before
    assert all(_hash(path) == digest for path, digest in reference.source_hashes.items())
