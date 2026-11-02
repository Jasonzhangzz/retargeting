"""Re-solve saved accepted ContactAware references at tighter geometry limits.

Only the contact/hand IK input stage is run.  Source files and semantic patch
identities remain fixed; all proposals, including failures, get a separate
output directory and an explicit acceptance report.
"""

from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import shutil
import traceback
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from retarget.common.results import write_json
from retarget.env.geometry import ObjectMeshSurface
from .contactaware import COMMIT, refine_grasp_reference, validate_settings, signed_distance_normal
from .forceaware_reference import _read_ca
from .hand import query_path
from .initialization import GraspInitialization
from .inputs import StaticHand


def saved_reference_inputs(source_experiment):
    """Recover original material points/normals and immutable semantic regions."""
    source, card, report, guidance, poses = _read_ca(source_experiment)
    scene = source/"runtime_scene.xml"
    if not scene.is_file():
        scene = source/"sequence/scene/wuji_right/primitive_mesh.xml"
    if not scene.is_file():
        scene = source/"sequence/scene/wuji_right/primitive_sdf.xml"
    model = mujoco.MjModel.from_xml_path(str(scene))
    if not np.allclose(poses, [0., 0., 0., 1., 0., 0., 0.], rtol=0, atol=1e-7):
        raise ValueError("Saved CA refinement expects its original fixed object-coordinate frame")
    data = mujoco.MjData(model); mujoco.mj_kinematics(model, data)
    obj = model.body("grasp_object").id
    if not np.allclose(data.xpos[obj], 0., rtol=0, atol=1e-7) or not np.allclose(
            data.xmat[obj].reshape(3, 3), np.eye(3), rtol=0, atol=1e-7):
        raise ValueError("Saved scene is not expressed in the fixed CA object frame")
    surface = ObjectMeshSurface.from_model(model, obj)
    bodies = guidance["hand_query_body_ids"][0].astype(int)
    local = guidance["hand_query_local_pos"][0].astype(np.float64)
    count = len(bodies)
    labels = np.asarray(card["semantic_patch_labels"])
    offsets, flat_masks = np.asarray(card["semantic_patch_offsets"]), np.asarray(card["semantic_patch_masks"], bool)
    if (len(report["per_patch"]) != count or labels.shape != (count,) or offsets.shape != (count+1,)
            or offsets[0] != 0 or offsets[-1] != len(flat_masks) or np.any(np.diff(offsets) <= 0)):
        raise ValueError("Saved CA semantic patch records do not match guidance slots")
    normals, masks, query_indices = [], [], []
    for i, (body, record) in enumerate(zip(bodies, report["per_patch"])):
        if record["label"] != str(labels[i]):
            raise ValueError("Saved CA patch labels changed order")
        index = int(record["query_index"])
        with np.load(query_path(model, int(body)), allow_pickle=False) as asset:
            points, query_normals = asset["local_pos"], asset["local_normal"]
            mask = flat_masks[int(offsets[i]):int(offsets[i+1])].copy()
            if (mask.shape != (len(points),) or not 0 <= index < len(points) or not mask[index]
                    or not np.allclose(points[index], local[i], rtol=0, atol=1e-7)):
                raise ValueError("Saved CA material point is outside its original semantic query patch")
            normals.append(query_normals[index].copy())
        masks.append(mask); query_indices.append(index)
    hand = StaticHand(model, surface, bodies.copy(), local.copy(), np.asarray(normals))
    hand.semantic_contact_masks = masks
    hand.semantic_patch_labels = labels.tolist()
    hand.semantic_contact_local = card["semantic_reference_local"].copy()
    hand.semantic_contact_normals = card["semantic_reference_normals"].copy()
    hand.contact_reference_normals = card["region_reference_normals"].copy()
    hand.ik_reference_qpos = card.get("gesture_reference_qpos", card["hand_reference_qpos"]).copy()
    for name in ("pose_intent_qpos", "pose_intent_radius", "pose_intent_center", "intent_palm_position",
                 "intent_palm_rotation", "cylinder_middle_slots", "cylinder_middle_half_band"):
        if name in card:
            setattr(hand, name, card[name].copy())
    seed = GraspInitialization(card["hand_reference_qpos"].copy(), card["object_points"].copy(),
        card["object_normals"].copy(), {}, dict(initialization_valid=False,
            contacts_optional=False, source="accepted CA reference used only as numerical seed"))
    physical = (model.geom_contype | model.geom_conaffinity) != 0
    friction = model.geom_friction[physical, 0]
    if not np.allclose(friction, friction[0], rtol=0, atol=1e-12):
        raise ValueError("Saved CA refinement needs explicit nonuniform contact-material handling")
    return dict(source=source, card=card, report=report, guidance=guidance, poses=poses,
        hand=hand, seed=seed, friction=float(friction[0]), scene=scene,
        query_indices=query_indices)


def active_reference_inputs(values):
    """Prune contact variables while preserving every whole-hand geometry row."""
    from .regions import hand_forbidden_mask

    full = values["hand"]
    full_forbidden = hand_forbidden_mask(full, full.contact_reference_normals)
    slots = np.flatnonzero(values["card"]["reference_contact_mask"].astype(bool))
    hand = StaticHand(full.model, full.surface, full.bodies[slots].copy(),
                      full.local[slots].copy(), full.normals[slots].copy())
    if not np.array_equal(hand.all_bodies, full.all_bodies) or not np.array_equal(hand.all_local, full.all_local):
        raise ValueError("Active-only ContactAware must preserve the complete hand-query ordering")
    hand.semantic_contact_masks = [full.semantic_contact_masks[i].copy() for i in slots]
    hand.semantic_patch_labels = [full.semantic_patch_labels[i] for i in slots]
    hand.semantic_contact_local = full.semantic_contact_local[slots].copy()
    hand.semantic_contact_normals = full.semantic_contact_normals[slots].copy()
    hand.contact_reference_normals = full.contact_reference_normals[slots].copy()
    hand.ik_reference_qpos = full.ik_reference_qpos.copy()
    for name in ("pose_intent_qpos", "pose_intent_radius", "pose_intent_center", "intent_palm_position",
                 "intent_palm_rotation", "cylinder_middle_half_band"):
        if hasattr(full, name):
            setattr(hand, name, copy.deepcopy(getattr(full, name)))
    if hasattr(full, "cylinder_middle_slots"):
        hand.cylinder_middle_slots = np.flatnonzero(np.isin(slots, full.cylinder_middle_slots))
        if not len(hand.cylinder_middle_slots):
            raise ValueError("Active-only CA cannot drop the required middle-cylinder contact")
    hand.contactaware_forbidden_mask = full_forbidden.copy()
    seed = copy.deepcopy(values["seed"])
    seed.object_points = seed.object_points[slots].copy()
    seed.object_normals = seed.object_normals[slots].copy()
    return hand, seed, slots, full_forbidden


def _slot_geometry(hand, qpos, anchors, device):
    import torch
    from .audit import GraspAudit

    hand.set(qpos)
    points = hand.points(hand.bodies, hand.local)
    normals = np.einsum("nij,nj->ni", hand.data.xmat[hand.bodies].reshape(-1, 3, 3), hand.normals)
    audit = GraspAudit(hand.surface, device)
    projected, face, phi = audit.query(torch.as_tensor(points.copy(), dtype=torch.float64, device=device))
    _, _, surface_gap = audit.query(torch.as_tensor(np.asarray(anchors).copy(), dtype=torch.float64, device=device))
    outward = signed_distance_normal(points, projected.cpu().numpy(), face.cpu().numpy(), phi.cpu().numpy())
    return np.linalg.norm(points-anchors, axis=1)*1000., -np.sum(normals*outward, axis=1), surface_gap.cpu().numpy()*1000.


def _expand_active_proposal(values, solved, slots, full_forbidden, args):
    """Restore the saved slot layout without reinstating released constraints."""
    result = copy.deepcopy(values["seed"])
    result.hand_qpos = np.asarray(solved.hand_qpos).copy()
    result.object_points = np.asarray(result.object_points, np.float64).copy()
    result.object_normals = np.asarray(result.object_normals, np.float64).copy()
    result.object_points[slots] = solved.object_points
    result.object_normals[slots] = solved.object_normals
    result.contact_mask = np.zeros(len(values["hand"].bodies), bool)
    result.contact_mask[slots] = np.asarray(solved.contact_mask, bool)
    result.metadata = copy.deepcopy(solved.metadata)
    report = result.metadata["contactaware"]
    report["optimization_per_patch"] = copy.deepcopy(report["per_patch"])
    gaps, facing, surface_gap = _slot_geometry(values["hand"], result.hand_qpos,
                                              result.object_points, args.init_device)
    active_slots = set(slots.tolist())
    report["per_patch"] = [dict(label=label, query_index=values["query_indices"][i],
        contact_error_mm=float(gaps[i]), facing_cosine=float(facing[i]),
        anchor_surface_gap_mm=float(surface_gap[i]), active=bool(result.contact_mask[i]),
        optimized_contact_slot=i in active_slots,
        release_reason="not_in_accepted_contact_mask" if i not in active_slots else
            None if result.contact_mask[i] else "contact_distance" if gaps[i]>args.contactaware["contact_tolerance_mm"] else "surface_facing")
        for i, label in enumerate(values["hand"].semantic_patch_labels)]
    report.update(active_contact_mask=result.contact_mask.tolist(), active_only=True,
        optimization_slot_indices=slots.tolist(), optimization_slot_count=len(slots),
        saved_slot_count=len(values["hand"].bodies),
        inactive_slot_policy="source anchors retained; excluded from contact residual, force closure and topology; guidance mask zero",
        full_hand_geometry_preserved=True, full_hand_query_count=len(values["hand"].all_bodies),
        forbidden_mask_source="computed before pruning from the original full annotation set and full hand-query ordering",
        full_forbidden_query_count=int(full_forbidden.sum()),
        full_forbidden_mask_sha256=hashlib.sha256(full_forbidden.tobytes()).hexdigest())
    for name in ("contactaware_reference_qpos", "hand_reference_qpos"):
        if hasattr(solved, name):
            setattr(result, name, np.asarray(getattr(solved, name)).copy())
    return result


def _write_proposal(destination, values, solved, report, *, candidate_origin):
    card, guidance = copy.deepcopy(values["card"]), copy.deepcopy(values["guidance"])
    qpos = np.asarray(solved.hand_qpos, np.float64)
    count, frames = len(values["hand"].bodies), len(values["poses"])
    accepted = report.get("success") is True
    card.update(object_points=np.asarray(solved.object_points), object_normals=np.asarray(solved.object_normals),
        hand_initial_qpos=qpos.copy(), hand_reference_source=np.asarray("dexforge_contactaware"),
        reference_contact_mask=np.asarray(getattr(solved, "contact_mask", np.zeros(count)), bool),
        reference_valid=np.asarray(accepted))
    if accepted:
        reference = np.asarray(solved.contactaware_reference_qpos)
        if reference.shape != (26,) or not np.isfinite(reference).all() or not np.array_equal(reference, qpos):
            raise ValueError("An accepted refinement must supply its explicit solved CA reference")
        card["hand_reference_qpos"] = reference.copy()
    else:
        card.pop("hand_reference_qpos", None)
        card["hand_candidate_qpos"] = qpos.copy()
    np.savez_compressed(destination/"card_input.npz", **card)
    contacts = destination/"contactaware"; contacts.mkdir(exist_ok=True)
    np.save(contacts/"wuji_right_qpos.npy", np.repeat(qpos[None], frames, axis=0).astype(np.float32))
    np.save(contacts/"object_pose_7.npy", values["poses"].copy())
    guidance.update(contact_mask=np.repeat(card["reference_contact_mask"][None], frames, axis=0).astype(np.float32),
        contact_pos_obj=np.repeat(np.asarray(solved.object_points)[None], frames, axis=0).astype(np.float32))
    np.savez_compressed(contacts/"contact_guidance.npz", **guidance)
    write_json(destination/"contactaware_reference.json", report)
    np.savez_compressed(destination/"refinement_candidate.npz", hand_qpos=qpos,
        object_points=solved.object_points, object_normals=solved.object_normals,
        active_contact_mask=card["reference_contact_mask"], accepted=accepted)
    return dict(success=accepted, candidate_origin=candidate_origin,
        source_seed_qpos=values["seed"].hand_qpos.tolist(), candidate_qpos=qpos.tolist(),
        object_pose_unchanged=True, semantic_material_points_unchanged=True,
        semantic_query_indices=values["query_indices"], template_angle_fallback=False)


def refine_saved_reference(source_experiment, destination, *, contact_tolerance_mm=.2,
                           penetration_tolerance_mm=.5, active_only=True, normal_cosine=.5):
    """Return a separate saved CA experiment, with strict success/failure metadata."""
    values = saved_reference_inputs(source_experiment)
    if not isinstance(active_only, bool):
        raise ValueError("active_only must be an explicit boolean")
    source = values["source"]
    destination = Path(destination).expanduser().resolve()
    if destination == source or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("CA refinement output must be separate from its immutable source")
    if destination.exists():
        raise FileExistsError(f"CA refinement destination already exists: {destination}")
    settings = dict(values["report"].get("settings", {}))
    settings.update(contact_tolerance_mm=contact_tolerance_mm,
                    penetration_tolerance_mm=penetration_tolerance_mm)
    if normal_cosine is not None:
        settings["normal_cosine"] = normal_cosine
    args = SimpleNamespace(contactaware=settings, friction=values["friction"],
        init_device=os.environ.get("FORCEAWARE_DEVICE", "cuda:0"))
    validate_settings(args)
    hand, seed = values["hand"], values["seed"]
    slots, full_forbidden = None, None
    if active_only:
        hand, seed, slots, full_forbidden = active_reference_inputs(values)
    shutil.copytree(source, destination)
    baseline = destination/"source_baseline"; baseline.mkdir(exist_ok=True)
    for name in ("card_input.npz", "contactaware_reference.json"):
        shutil.copy2(source/name, baseline/("source_"+name))
    # Point the independent copy at its byte-identical copied primitive mesh.
    for relative in ("runtime_scene.xml", "sequence/scene/wuji_right/primitive_sdf.xml",
                     "sequence/scene/wuji_right/primitive_mesh.xml"):
        scene = destination/relative
        if scene.is_file() and (destination/"object.obj").is_file():
            tree = ET.parse(scene)
            for asset in tree.findall("asset/mesh[@name='grasp_object_mesh']"):
                asset.set("file", str(destination/"object.obj"))
            tree.write(scene, encoding="utf-8")
    pending = copy.deepcopy(values["card"])
    pending.pop("hand_reference_qpos", None)
    pending["reference_valid"] = np.asarray(False)
    np.savez_compressed(destination/"card_input.npz", **pending)
    write_json(destination/"contactaware_reference.json", dict(success=False, source_commit=COMMIT,
        settings=settings, method="CA refinement pending; source reference is not the new result"))
    try:
        solved = refine_grasp_reference(hand, seed, args)
        if active_only:
            solved = _expand_active_proposal(values, solved, slots, full_forbidden, args)
        report = copy.deepcopy(solved.metadata["contactaware"])
        origin = "original DexForge solve_joint_grasp contact and hand IK refinement"
        report["saved_reference_refinement"] = dict(source_experiment=str(source),
            initial_state_source="accepted source CA qref; numerical seed only", object_pose_fixed=True,
            contact_tolerance_mm=contact_tolerance_mm, penetration_tolerance_mm=penetration_tolerance_mm,
            active_only=active_only, normal_cosine=settings.get("normal_cosine"))
        result = _write_proposal(destination, values, solved, report, candidate_origin=origin)
    except Exception as error:
        solved = copy.deepcopy(values["seed"])
        solved.contact_mask = np.zeros(len(values["hand"].bodies), bool)
        report = dict(success=False, source_commit=COMMIT, settings=settings,
            error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
            method="original DexForge CA refinement failed before returning a proposal")
        origin = "unchanged input seed for diagnostics only; no optimized reference returned"
        report["saved_reference_refinement"] = dict(source_experiment=str(source),
            initial_state_source="accepted source CA qref; numerical seed only", object_pose_fixed=True,
            contact_tolerance_mm=contact_tolerance_mm, penetration_tolerance_mm=penetration_tolerance_mm,
            active_only=active_only, normal_cosine=settings.get("normal_cosine"))
        result = _write_proposal(destination, values, solved, report, candidate_origin=origin)
    hashes = {name: hashlib.sha256((source/name).read_bytes()).hexdigest()
        for name in ("card_input.npz", "contactaware_reference.json", "contactaware/contact_guidance.npz")}
    write_json(destination/"ca_refinement_manifest.json", dict(**result,
        source_experiment=str(source), source_sha256=hashes, source_commit=COMMIT,
        requested_contact_tolerance_mm=contact_tolerance_mm,
        requested_penetration_tolerance_mm=penetration_tolerance_mm,
        requested_normal_cosine=normal_cosine, effective_normal_cosine=settings.get("normal_cosine"),
        active_only=active_only,
        optimizer="original CA elastic trust-region SQP + Clarabel; no MPPI/FA executed"))
    return destination
