"""Final-pose diagnostic coverage of immutable anatomical contact patches.

This module does not participate in MPPI scoring or force closure. A required
patch is a rigid body and anatomical surface region, classified from the fixed
Wuji template point and normal. Several anchors on that patch count once. All exposed
interior queries on that patch may witness contact; anchors are never moved.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import mujoco
import numpy as np
from scipy.spatial import cKDTree
import torch

from . import semantic_patches as semantic
from .hand import query_path, rigid_mesh_arrays

DEFINITION = "anatomical_patch_interior_v2"
EDGE_MARGIN_M = .001
_INTERIORS = {}


def _target_semantic(name):
    if name == "right_palm_link":
        return "palm", None
    finger = int(name.removeprefix("right_finger").split("_")[0])
    segment = {2: "proximal", 3: "middle", 4: "distal"}[int(name[-1])]
    return segment, ("thumb", "index", "middle", "ring", "pinky")[finger-1]


def _identity(model, template, slot):
    name = template.body_names[slot]
    body = model.body(name).id
    segment, finger = _target_semantic(name)
    pool = semantic._target_pool(model, body, segment, finger)
    normal = template.local_normals[slot] @ semantic._matrix(pool["frame"])
    chart = (template.local_points[slot]-pool["frame"]["origin"]) @ semantic._matrix(pool["frame"])
    # The point's patch is on Wuji2. A source Shadow tip annotation can map
    # onto a lateral Wuji surface; using the source label would change the
    # required contact surface, contrary to a metric-only redefinition.
    region, _ = semantic._region(normal, chart[2]/pool["length"], segment)
    label = f"{finger or segment}.{segment if finger else 'surface'}.{region}"
    return body, segment, finger, region, label


def sharp_edges(vertices, faces, angle_deg=45.):
    """Weld component seams and return genuine sharp/boundary segments."""
    vertices, inverse = np.unique(np.round(vertices, 8), axis=0, return_inverse=True)
    faces = inverse[np.asarray(faces)]
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    norm = np.linalg.norm(normals, axis=1)
    normals = normals/np.maximum(norm[:, None], 1.e-12)
    neighbors = {}
    for face_id, face in enumerate(faces):
        if norm[face_id] < 1.e-15:
            continue
        for a, b in zip(face, np.roll(face, -1)):
            neighbors.setdefault(tuple(sorted((int(a), int(b)))), []).append(face_id)
    threshold = np.cos(np.deg2rad(angle_deg))
    edges = [edge for edge, ids in neighbors.items() if len(ids) == 1 or any(
        normals[ids[0]] @ normals[i] < threshold for i in ids[1:])]
    return vertices[np.asarray(edges, int)].reshape(-1, 2, 3)


def edge_clearance(points, segments):
    result = np.full(len(points), np.inf)
    for start in range(0, len(segments), 512):
        chunk = segments[start:start+512]
        a, v = chunk[:, 0], chunk[:, 1]-chunk[:, 0]
        offset = points[:, None]-a
        t = np.clip((offset*v).sum(-1)/np.maximum((v*v).sum(-1), 1.e-20), 0, 1)
        distance = np.linalg.norm(offset-t[..., None]*v, axis=-1)
        result = np.minimum(result, distance.min(axis=1))
    return result


def interior_mask(pool, region, segment, *, edge_margin_m=EDGE_MARGIN_M, segments=()):
    """Full anatomical side, eroded at semantic borders and mesh creases.

    The margin concerns the hand patch, not object rims. Smooth surface
    triangulation diagonals are not edges. Longitudinal annotation windows
    from the correspondence matcher deliberately do not restrict this metric.
    """
    matrix = semantic._matrix(pool["frame"])
    normals = pool["normals"] @ matrix
    u = pool["chart"][:, 2]/pool["length"]
    regions = np.asarray([semantic._region(n, z, segment)[0] for n, z in zip(normals, u)])
    axes = {"palmar": [0, 1, 0], "dorsal": [0, -1, 0],
            "lateral_positive": [1, 0, 0], "lateral_negative": [-1, 0, 0], "tip": [0, 0, 1]}
    membership = pool["base"] & (regions == region) & (normals @ axes[region] >= np.cos(np.pi/4))
    outside = pool["base"] & ~membership
    boundary_distance = (cKDTree(pool["points"][outside]).query(pool["points"])[0]
                         if outside.any() else np.full(len(u), np.inf))
    geometric_distance = edge_clearance(pool["points"], segments)
    clearance = np.minimum(boundary_distance, geometric_distance)
    return membership & (clearance >= edge_margin_m), clearance


@dataclass(frozen=True)
class ContactPatch:
    body: str
    label: str
    slots: tuple[int, ...]
    local_points: np.ndarray
    local_normals: np.ndarray
    query_indices: np.ndarray
    clearance_m: np.ndarray
    query_sha256: str


def build_patches(model, template, *, edge_margin_m=EDGE_MARGIN_M):
    identities = [_identity(model, template, i) for i in range(len(template.body_names))]
    groups = {}
    for i, identity in enumerate(identities):
        groups.setdefault(identity, []).append(i)
    patches = []
    for (body, segment, finger, region, label), slots in groups.items():
        path = query_path(model, body)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        # Pool geometry and semantic frames depend on the static hand assets,
        # never on object size or current hand q.
        key = (str(path.resolve()), digest, segment, region, edge_margin_m)
        if key not in _INTERIORS:
            pool = semantic._target_pool(model, body, segment, finger)
            vertices, faces = rigid_mesh_arrays(model, body)
            mask, clearance = interior_mask(pool, region, segment,
                edge_margin_m=edge_margin_m, segments=sharp_edges(vertices, faces))
            _INTERIORS[key] = (pool, mask, clearance)
        pool, mask, clearance = _INTERIORS[key]
        if template.provenance.get('policy') == 'fixed_prismatic_pad_center_v1':
            # This is a new, explicitly fixed center-pad template. Its display
            # and metric must not count another part of the same long patch.
            from .contact_regions import region_mask
            mask = mask & region_mask(model.body(body).name, pool['points'], pool['normals'],
                                      template.provenance['fixed_contact_region'])
        patches.append(ContactPatch(model.body(body).name, label, tuple(slots),
            pool["points"][mask].copy(), pool["normals"][mask].copy(),
            np.flatnonzero(mask), clearance[mask], digest))
    return patches


def fitted_queries(gap, hand_normals, object_normals, *, distance_m, allowance_m, angle_deg):
    cosine = -(hand_normals*object_normals).sum(-1)
    finite = torch.isfinite(gap) & torch.isfinite(hand_normals).all(-1) & torch.isfinite(object_normals).all(-1)
    return finite & (gap >= -allowance_m) & (gap <= distance_m) & (cosine >= np.cos(np.deg2rad(angle_deg)))


def evaluate_patches(patches, positions, rotations, audit, settings, *, edge_margin_m=EDGE_MARGIN_M,
                     side_bodies=(), side_audit=None):
    """Query all patch interiors using existing FK frames, with no force solve.

    positions/rotations are in patch order for exactly one final pose. Projection
    is batched on the selected device; padding and MPPI populations are absent.
    """
    rows, point_batches, normal_batches = [], [], []
    for i, patch in enumerate(patches):
        local = torch.as_tensor(patch.local_points, dtype=positions.dtype, device=positions.device)
        normals = torch.as_tensor(patch.local_normals, dtype=positions.dtype, device=positions.device)
        point_batches.append(positions[i] + local @ rotations[i].T)
        normal_batches.append(normals @ rotations[i].T)
    sizes = [len(x) for x in point_batches]
    total = sum(sizes)
    if total:
        world, hand_normals = torch.cat(point_batches).float(), torch.cat(normal_batches).float()
        projected, outward, gap = audit.query(world)
        if side_audit is not None:
            projected, outward, gap = projected.clone(), outward.clone(), gap.clone()
            mask = torch.cat([torch.full((count,), p.body in side_bodies, device=world.device,
                                         dtype=torch.bool) for p, count in zip(patches, sizes)])
            if bool(mask.any()):
                side_points, side_normals, _ = side_audit.query(world[mask])
                distance = torch.linalg.vector_norm(world[mask]-side_points, dim=-1)
                projected[mask], outward[mask] = side_points, side_normals
                gap[mask] = torch.where(gap[mask] < 0, -distance, distance)
        fitted = fitted_queries(gap, hand_normals, outward,
            distance_m=settings.contact_tolerance_m, allowance_m=settings.hand_object_allowance_m,
            angle_deg=settings.normal_tolerance_deg)
        cosine = -(hand_normals*outward).sum(-1).clamp(-1, 1)
        score = ((gap.clamp_min(0) + (gap+settings.hand_object_allowance_m).clamp_max(0))/settings.contact_scale_m).square()
        score += (1-cosine)/(1-np.cos(np.deg2rad(settings.normal_tolerance_deg)))
    start = 0
    for patch, count in zip(patches, sizes):
        row = dict(body=patch.body, label=patch.label, template_slots=list(patch.slots),
            local_points=patch.local_points.tolist(), local_normals=patch.local_normals.tolist(),
            query_indices=patch.query_indices.tolist(), query_sha256=patch.query_sha256,
            eligible_query_count=count, edge_clearance_m=patch.clearance_m.tolist(), fitted=False,
            fitted_query_indices=[], witness=None)
        if count:
            fits = fitted[start:start+count]
            row["fitted"] = bool(fits.any())
            row["fitted_query_indices"] = patch.query_indices[fits.cpu().numpy()].tolist()
            quality = torch.where(torch.isfinite(score[start:start+count]), score[start:start+count], torch.inf)
            if row["fitted"]:
                quality = torch.where(fits, quality, torch.inf)
            index = int(quality.argmin())
            if torch.isfinite(quality[index]):
                j = start+index
                row["witness"] = dict(query_index=int(patch.query_indices[index]),
                    local_point=patch.local_points[index].tolist(), hand_point=world[j].cpu().tolist(),
                    projected_point=projected[j].cpu().tolist(), hand_normal=hand_normals[j].cpu().tolist(),
                    object_normal=outward[j].cpu().tolist(), signed_distance_m=float(gap[j]),
                    normal_error_deg=float(torch.rad2deg(torch.acos(cosine[j]))),
                    edge_clearance_m=float(patch.clearance_m[index]), fitted=row["fitted"])
        rows.append(row)
        start += count
    covered = sum(row["fitted"] for row in rows)
    slot_count = sum(len(p.slots) for p in patches)
    slot_to_patch = [None]*slot_count
    for i, patch in enumerate(patches):
        for slot in patch.slots:
            slot_to_patch[slot] = i
    side_fingers = {body.split('_link')[0] for body in side_bodies}
    fitted_side_fingers = {row['body'].split('_link')[0] for row in rows
                          if row['body'] in side_bodies and row['fitted']}
    return dict(coverage_definition=DEFINITION, coverage=covered/len(rows) if rows else 0.,
        required_patch_count=len(rows), covered_patch_count=covered, patches=rows,
        slot_to_patch=slot_to_patch, edge_margin_m=edge_margin_m,
        edge_scope="hand patch boundary and mesh creases over 45 degrees; object rims unchanged",
        distance_tolerance_m=settings.contact_tolerance_m,
        hand_object_allowance_m=settings.hand_object_allowance_m,
        normal_tolerance_deg=settings.normal_tolerance_deg,
        denominator="unique rigid body and anatomical patch, independent of anchor multiplicity",
        object_surface_regions={body: 'cylinder_side' for body in side_bodies},
        required_side_finger_count=len(side_fingers),
        covered_side_finger_count=len(fitted_side_fingers),
        patch_identity_source="fixed Wuji2 body, local point and local normal in its anatomical frame",
        empty_patches=[row["label"] for row in rows if not row["eligible_query_count"]],
        optimization_or_force_contacts_changed=False)


def diagnose_patch_coverage(model, template, q, audit, settings, *, device="cuda:0", side_bodies=(), side_audit=None):
    """Standalone final diagnostic, also usable on saved results only."""
    patches = build_patches(model, template)
    target = torch.device(device)
    if side_bodies and side_audit is None:
        from .template_mppi_size_ik import CylinderSideProjection
        from ..geometry import ObjectMeshSurface
        surface = ObjectMeshSurface.from_model(model, model.body('grasp_object').id)
        side_audit = CylinderSideProjection(surface, target, []).audit
    ids = [model.body(p.body).id for p in patches]
    with torch.no_grad():
        if target.type == "cuda":
            from .template import WujiContactKinematics
            fk = WujiContactKinematics(model, ids, target)
            p, r = fk.frames(fk.tensor(q))
        else:
            data = mujoco.MjData(model)
            data.qpos[:] = np.asarray(q)
            mujoco.mj_kinematics(model, data)
            p = torch.as_tensor(data.xpos[ids].copy(), device=target)
            r = torch.as_tensor(data.xmat[ids].copy().reshape(-1, 3, 3), device=target)
        result = evaluate_patches(patches, p, r, audit, settings, side_bodies=side_bodies, side_audit=side_audit)
    result["fk_backend"] = "torch_cuda_WujiContactKinematics" if target.type == "cuda" else "mujoco_cpu_reference"
    if template.provenance.get('policy') == 'fixed_prismatic_pad_center_v1':
        result['contact_region_policy'] = 'fixed_prismatic_pad_center_v1'
    return result


def attach_patch_coverage(diagnostic, coverage):
    """Change reporting only; leave costs, forces, slots and acceptance intact."""
    updated = dict(diagnostic)
    updated.setdefault("template_point_coverage", diagnostic["coverage"])
    updated.setdefault("template_point_geometry_success", diagnostic.get("geometry_success", False))
    updated.update(coverage=coverage["coverage"], coverage_definition=DEFINITION,
        required_patch_count=coverage["required_patch_count"], covered_patch_count=coverage["covered_patch_count"],
        object_surface_regions=coverage.get('object_surface_regions', {}))
    for key in ('required_side_finger_count', 'covered_side_finger_count'):
        if key in coverage:
            updated[key] = coverage[key]
    if coverage.get('contact_region_policy'):
        updated['contact_region_policy'] = coverage['contact_region_policy']
    required = diagnostic.get("settings", {}).get("contact_fraction", .8)
    updated["geometry_success"] = bool(diagnostic.get("soft_contact_success", False) and coverage["coverage"]+1.e-7 >= required)
    return updated
