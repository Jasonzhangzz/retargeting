"""Moxian SKN material taxels and patch graph. No MANO assets or inference.

All material coordinates belong to the source mesh. Reflection is applied to
positions AND physical normals, never inferred again from reflected winding.
"""
from __future__ import annotations

from itertools import combinations
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .contact import PATCH_IDS

POLICY = 'moxian_semantic_link_graph_v3'


def _patch_joint_indices(patch):
    """Return the two source skeleton endpoints for a sensor patch."""
    if patch >= 14:
        return (0, 9)
    if patch < 2:
        return (1, 2) if patch == 1 else (3, 4)
    finger = 1 + (patch - 2) // 3
    segment = 2 - (patch - 2) % 3
    return (1 + 4 * finger + segment, 2 + 4 * finger + segment)


def fit_patch_envelope(points, joints, normal, weights=None, margin=0.0008):
    """Fit a deterministic, convex-hull-like surface envelope to a patch.

    The envelope is deliberately a point/surface set rather than a one-to-one
    correspondence.  Degenerate patches are thickened in their PCA normal so
    the downstream Chamfer objective remains defined for a single taxel.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    joints = np.asarray(joints, dtype=np.float64)
    normal = unit(np.asarray(normal, dtype=np.float64))
    if weights is None or len(weights) != len(points):
        weights = np.ones(len(points), dtype=np.float64)
    weights = np.maximum(np.asarray(weights, dtype=np.float64), 0.)
    if not np.any(weights > 1e-12):
        weights = np.ones(len(points), dtype=np.float64)
    center = np.average(points, axis=0, weights=weights)
    # Keep the source skeleton as a weak geometric scaffold, not an orientation
    # target.  Endpoints are included only to prevent a sparse patch collapsing.
    # Only the local segment/hand-shape scaffold is allowed here.  In
    # particular, never add the wrist as a pseudo-contact point.
    extras = np.asarray(joints, dtype=np.float64).reshape(-1, 3) if len(joints) else np.empty((0, 3))
    cloud = np.vstack((points, extras)) if len(points) else extras.copy()
    if not len(cloud):
        cloud = np.zeros((1, 3), dtype=np.float64)
    # Use a local PCA frame to make the fallback deterministic.
    demean = cloud - cloud.mean(0)
    _, values, vectors = np.linalg.svd(demean, full_matrices=False)
    rank = int(np.sum(values > 1e-7)) if len(values) else 0
    valid = bool(rank >= 2 and len(cloud) >= 3)
    if rank == 0:
        tangent = np.array([1., 0., 0.]); bitangent = np.array([0., 0., 1.])
    else:
        tangent = unit(vectors[0])
        tangent -= normal * np.dot(tangent, normal)
        tangent = unit(tangent if np.linalg.norm(tangent) > 1e-8 else vectors[0])
        bitangent = unit(np.cross(normal, tangent))
    # A fixed oriented box is a fast convex envelope approximation.  It has
    # stable support planes and avoids Qhull inside every incoming frame.
    ext = float(np.ptp(cloud @ tangent)); wid = float(np.ptp(cloud @ bitangent))
    valid = valid and max(ext, wid) >= margin * .5
    ext = max(ext, margin); wid = max(wid, margin)
    thick = margin
    corners = np.array([center + a * tangent * ext / 2 + b * bitangent * wid / 2 + c * normal * thick
                        for a in (-1., 1.) for b in (-1., 1.) for c in (-1., 1.)])
    planes = []
    for axis, half in ((tangent, ext / 2), (bitangent, wid / 2), (normal, thick)):
        planes.append(np.r_[axis, -float(axis @ center + half)])
        planes.append(np.r_[-axis, float(axis @ center - half)])
    return dict(points=corners.tolist(), source_points=points.tolist(), center=center.tolist(),
                normal=normal.tolist(), planes=np.asarray(planes).tolist(), rank=rank,
                valid=bool(valid), fallback='oriented_box', margin=float(margin),
                scale=float(max(ext, wid, 2 * thick)))


def semantic_link(patch):
    """Sensor tiles share a graph class when they belong to one hand link."""
    return 'palm' if patch >= 14 else PATCH_IDS[patch]


def contact_edges(anchors):
    return np.asarray([(i, j) for i, j in combinations(range(len(anchors)), 2)
                       if anchors[i]['semantic_link'] != anchors[j]['semantic_link']], int).reshape(-1, 2)


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    length = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(length < 1e-12) or not np.isfinite(x).all():
        raise ValueError('Degenerate material normal or anatomical axis')
    return x / length


def closest_triangle(triangles, point):
    """Exact closest point on a triangle union, with face index and barycentrics."""
    triangles = np.asarray(triangles, dtype=np.float64)
    if not len(triangles):
        raise ValueError('Empty contact surface')
    a, b, c = triangles.transpose(1, 0, 2)
    ab, ac, d = b-a, c-a, point-a
    aa = (ab*ab).sum(-1); bb = (ab*ac).sum(-1); cc = (ac*ac).sum(-1)
    ad = (ab*d).sum(-1); cd = (ac*d).sum(-1); det = aa*cc-bb*bb
    safe = np.where(det > 1e-24, det, 1.)
    v = (cc*ad-bb*cd)/safe; w = (aa*cd-bb*ad)/safe
    bary = np.stack((1-v-w, v, w), -1)
    choices = [bary]
    scores = [np.where((bary.min(-1) >= 0) & (det > 1e-24),
                       (((triangles*bary[..., None]).sum(1)-point)**2).sum(-1), np.inf)]
    for i, j in ((0, 1), (1, 2), (2, 0)):
        start = triangles[:, i]; edge = triangles[:, j]-start
        t = np.clip(((point-start)*edge).sum(-1)/np.maximum((edge*edge).sum(-1), 1e-24), 0, 1)
        bc = np.zeros_like(bary); bc[:, i] = 1-t; bc[:, j] = t
        choices.append(bc); scores.append(((start+t[:, None]*edge-point)**2).sum(-1))
    face, choice = np.unravel_index(np.asarray(scores).T.argmin(), (len(triangles), 4))
    bc = choices[choice][face]
    return bc @ triangles[face], int(face), bc


def activation(contact, stamp, tactile_stamp, enabled=True):
    """Binary evidence, relative response and same-frame validity are separate."""
    raw_response = contact.get('normalized')
    response = np.zeros(460) if raw_response is None or isinstance(raw_response, list) and not raw_response else np.asarray(raw_response, dtype=np.float64)
    if response.shape != (460,):
        raise ValueError('460 tactile responses required')
    response = np.clip(np.nan_to_num(response), 0, 1)
    times = (stamp, tactile_stamp, contact.get('timestamp_ms', tactile_stamp))
    ready = bool(enabled and contact.get('ready') and all(
        isinstance(t, (int, float)) and np.isfinite(t) and abs(t-stamp) <= 25 for t in times))
    cells = contact.get('cell_contact') or []
    mask = np.array([x is True for x in cells], bool) if len(cells) == 460 else np.zeros(460, bool)
    if not ready: mask[:] = False
    return dict(ready=ready, mask=mask, response=response)


class MoxianContactSource:
    def __init__(self, topology):
        if topology.get('surface_backend') != 'qnbot' or topology.get('surface_policy') != 'qnbot_original_triangles':
            raise ValueError('SLSQP requires the Moxian glove mesh; reconstruct legacy MANO recordings from raw EMF first')
        self.topology = topology
        self.faces = np.asarray(topology['faces'], dtype=int)
        self.rest = np.asarray(topology['rest_vertices_m'], dtype=np.float64)
        self.sid_faces = np.asarray(topology['sid_face_ids'], dtype=int)
        self.bary = np.asarray(topology['sid_barycentric'], dtype=np.float64)
        self.offset = np.asarray(topology.get('sid_normal_offset_m', [0.]*460), dtype=np.float64)
        self.patches = np.asarray(topology['sid_patch_indices'], dtype=int)
        if (self.sid_faces.shape != (460,) or self.bary.shape != (460, 3) or self.offset.shape != (460,)
                or self.patches.shape != (460,) or self.faces.ndim != 2 or self.faces.shape[1] != 3
                or self.rest.ndim != 2 or self.rest.shape[1] != 3 or not np.isfinite(self.rest).all()
                or not np.isfinite(self.bary).all() or not np.isfinite(self.offset).all()
                or self.bary.min() < 0 or not np.allclose(self.bary.sum(1), 1, atol=1e-5)
                or self.faces.min() < 0 or self.faces.max() >= len(self.rest)
                or self.sid_faces.min() < 0 or self.sid_faces.max() >= len(self.faces)
                or self.patches.min() < 0 or self.patches.max() >= 20 or abs(self.offset).max() > .005):
            raise ValueError('Invalid Moxian material binding')
        self.regions = [np.asarray(topology['regions'][p], int) for p in PATCH_IDS]
        for sid, face in enumerate(self.sid_faces):
            if face not in self.regions[self.patches[sid]]:
                raise ValueError('SID is outside its Moxian patch')
        chains = topology['joint_chains']
        self.landmarks = np.array([chains[0][0]]+[i for c in chains for i in c[-4:]], int)
        rest_joints = np.asarray(topology['rest_joints_m'], float)
        self.rest_joints = rest_joints[self.landmarks]
        self.mirror = np.eye(3)
        if topology['hand_side'] == 'left':
            lateral = unit(np.cross(topology['palm_view']['forward'], topology['palm_view']['normal']))
            self.mirror -= 2*np.outer(lateral, lateral)
        self.identity = (topology['geometry_version'], topology['binding_version'], topology.get('model_hash'))

    def taxels(self, vertices):
        vertices = np.asarray(vertices, dtype=np.float64)
        if vertices.shape != self.rest.shape or not np.isfinite(vertices).all():
            raise ValueError('Invalid Moxian posed mesh')
        tri = vertices[self.faces[self.sid_faces]]
        normals = unit(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]))
        points = (tri*self.bary[..., None]).sum(1)+self.offset[:, None]*normals
        return points @ self.mirror.T, normals @ self.mirror.T

    def read(self, frame, enabled=True, valid_sids=None, build_envelopes=True):
        if (frame.get('geometry_version'), frame.get('binding_version')) != self.identity[:2]:
            raise ValueError('Moxian topology/binding differs from the posed frame')
        if 'model_hash' in self.topology and frame.get('model_hash') != self.identity[2]:
            raise ValueError('Moxian skeleton and mesh model hashes differ')
        if frame.get('hand_side') not in (None, self.topology['hand_side']):
            raise ValueError('Moxian hand side differs from topology')
        vertices = np.asarray(frame['vertices_m'], float)
        points, normals = self.taxels(vertices)
        joints = np.asarray(frame['joints_m'], float)
        if joints.ndim != 2 or joints.shape[1] != 3 or len(joints) <= self.landmarks.max() or not np.isfinite(joints).all():
            raise ValueError('Invalid Moxian skeleton')
        stamp = float(frame['timestamp_ms'])
        if not np.isfinite(stamp): raise ValueError('Invalid source timestamp')
        skeleton_stamp = frame.get('skeleton_timestamp_ms', stamp)
        if not isinstance(skeleton_stamp, (int, float)) or not np.isfinite(skeleton_stamp) or abs(skeleton_stamp-stamp) > 25:
            raise ValueError('Moxian skeleton and mesh timestamps differ')
        weights = activation(frame.get('source_contact') or {}, stamp, frame.get('tactile_timestamp_ms'), enabled)
        weights['observed_mask']=weights['mask'].copy()
        if valid_sids is None:valid_sids=getattr(self,'correspondence_valid_sids',None)
        if valid_sids is not None:
            valid_sids=np.asarray(valid_sids,bool)
            if valid_sids.shape!=(460,):raise ValueError('460 correspondence validity flags required')
            weights['mask'] &= valid_sids
        raw_points = points @ self.mirror
        posed_joints = joints[self.landmarks] @ self.mirror.T
        anchors = []
        for patch in range(20):
            ids = np.flatnonzero(weights['mask'] & (self.patches == patch))
            if not len(ids): continue
            response = weights['response'][ids]; force = float(response.sum())
            response = response if force > 1e-12 else np.ones(len(ids))
            centre = np.average(raw_points[ids], axis=0, weights=response)
            pool = self.regions[patch]
            point, index, bc = closest_triangle(vertices[self.faces[pool]], centre)
            face = int(pool[index]); offset = float(np.average(self.offset[ids], weights=response))
            if len(ids) == 1:
                face = int(self.sid_faces[ids[0]]); bc = self.bary[ids[0]]; offset = float(self.offset[ids[0]])
            tri = vertices[self.faces[face]]; normal = unit(np.cross(tri[1]-tri[0], tri[2]-tri[0]))
            point = bc @ tri+offset*normal
            j0, j1 = _patch_joint_indices(patch)
            segment_joints = posed_joints[[5, 9, 13, 17]] if patch >= 14 else posed_joints[[j0, j1]]
            # Classify the patch's dominant contact face in its anatomical frame.
            axis = unit(segment_joints[1] - segment_joints[0])
            physical_normal = normal @ self.mirror.T
            lateral = unit(np.cross(physical_normal, axis))
            side_score = float(np.mean((normals[ids] @ lateral) * response)) if len(ids) else 0.
            patch_side = 'front' if abs(side_score) < .18 else ('left' if side_score > 0 else 'right')
            envelope = fit_patch_envelope(points[ids], segment_joints, physical_normal, response) if build_envelopes else {}
            if build_envelopes:
                envelope['center'] = np.average(points[ids], axis=0, weights=response).tolist()
            anchors.append(dict(patch=patch, patch_id=PATCH_IDS[patch], semantic_link=semantic_link(patch), position_m=(point @ self.mirror.T).tolist(),
                direction=(normal @ self.mirror.T).tolist(), source_face=face, barycentric=bc.tolist(),
                normal_offset_m=offset, sids=ids.tolist(), force=force, patch_side=patch_side,
                envelope=envelope))
        patch_envelopes = []
        active_by_patch = {int(a['patch']): a for a in anchors}
        for patch in range(20):
            if patch in active_by_patch:
                patch_envelopes.append(dict(active_by_patch[patch]))
                continue
            ids = np.flatnonzero(self.patches == patch)
            if not len(ids):
                continue
            j0, j1 = _patch_joint_indices(patch)
            local_joints = posed_joints[[5, 9, 13, 17]] if patch >= 14 else posed_joints[[j0, j1]]
            patch_normal = unit(normals[ids].mean(0))
            axis = unit(local_joints[1] - local_joints[0])
            lateral = unit(np.cross(patch_normal, axis))
            side_score = float(np.mean(normals[ids] @ lateral))
            side = 'front' if abs(side_score) < .18 else ('left' if side_score > 0 else 'right')
            patch_envelopes.append(dict(patch=patch, patch_id=PATCH_IDS[patch], semantic_link=semantic_link(patch),
                                        sids=ids.tolist(), force=0., patch_side=side,
                                        envelope=(fit_patch_envelope(points[ids], local_joints, patch_normal) if build_envelopes else {})))
        patch_envelopes.sort(key=lambda row: int(row['patch']))
        return dict(vertices=vertices @ self.mirror.T, joints=joints[self.landmarks] @ self.mirror.T,
                    taxels=points, normals=normals, weights=weights, anchors=anchors,
                    patch_envelopes=patch_envelopes, edges=contact_edges(anchors))


def robot_mapping(source, surface, cache=None, rebuild=False, normal_limit_deg=None):
    """Static anatomical SID correspondence, independent of either optimizer."""
    from .wuji_surface import basis
    topo = source.topology
    signature = [POLICY, surface.asset_hash, source.identity, source.sid_faces.tolist(), source.bary.tolist(), source.offset.tolist()]
    if normal_limit_deg is not None:
        if not np.isfinite(normal_limit_deg) or not 0<normal_limit_deg<90:raise ValueError('Mapping normal limit must be between 0 and 90 degrees')
        signature+=['normal_compatible_material_v1',float(normal_limit_deg)]
    version = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:24]
    path = Path(cache)/(version+'.json') if cache else None
    if path and path.exists() and not rebuild:
        result = json.loads(path.read_text())
        if result.get('version') == version: return result
    points, normals = source.taxels(source.rest); joints = source.rest_joints @ source.mirror.T
    records = [None]*460; failed = []
    for patch, name in enumerate(PATCH_IDS):
        ids = np.flatnonzero(source.patches == patch)
        if not len(ids): continue
        if patch >= 14:
            owner = 0; origin = joints[0]; axis = joints[9]-origin; joint = 0
        else:
            finger = 0 if patch < 2 else 1+(patch-2)//3
            # Moxian thumb proximal is MCP->IP, i.e. native thumb_middle;
            # the robot's proximal_abd body belongs to the CMC->MCP segment.
            segment = (2 if patch == 0 else 1) if finger == 0 else 2-(patch-2)%3
            owner = 2+4*finger+segment; joint = 1+4*finger+segment
            origin = joints[joint]; axis = joints[joint+1]-origin
        pool = surface.candidates(owner); keep = np.arange(len(pool['points']))
        target_basis = surface.bases[owner]
        if patch >= 14:
            coords = pool['points'] @ target_basis
            scaled = (coords-coords.min(0))/np.maximum(np.ptp(coords, axis=0), 1e-6)
            row, col = divmod(patch-14, 2)
            # Physical thumb-side tile is determined in the source palm chart.
            palm = points @ basis(np.asarray(topo['palm_view']['forward']) @ source.mirror.T,
                                  np.asarray(topo['palm_view']['normal']) @ source.mirror.T)
            high = palm[ids, 0].mean() > np.median(palm[source.patches >= 14, 0])
            keep = np.flatnonzero((scaled[:, 2] >= row/3) & (scaled[:, 2] <= (row+1)/3)
                                 & ((scaled[:, 0] >= .5) if high else (scaled[:, 0] < .5)))
        if len(keep) < len(ids): failed.extend(ids.tolist()); continue
        sb = basis(axis, unit(normals[ids].mean(0)))
        sp = (points[ids]-origin) @ sb; sn = normals[ids] @ sb
        tp = pool['points'][keep] @ target_basis; tn = pool['normals'][keep] @ target_basis
        lo, hi = sp.min(0), sp.max(0); tl, th = tp.min(0), tp.max(0)
        # Preserve patch coordinates, without importing MANO skin weights.
        a = (sp-lo)/np.maximum(hi-lo, .001); b = (tp-tl)/np.maximum(th-tl, .001)
        dot = np.clip(sn @ tn.T, -1, 1)
        cost = 4*((a[:, None]-b[None])**2).sum(-1)+3*(1-dot)
        # The source's curved palm can have much wider normal variation than
        # a robot pad. Keep the anatomical material correspondence and expose
        # the discrepancy; a 60-degree gate would drop real source taxels.
        # Preserve the legacy mapping for CEM. SLSQP maximizes the number of
        # normal-compatible assignments first; impossible SID matches remain
        # visible diagnostics but are excluded from retargeting, never forced.
        assignment=cost if normal_limit_deg is None else np.where(dot>=np.cos(np.deg2rad(normal_limit_deg)),cost,1e9+cost)
        rows, cols = linear_sum_assignment(assignment)
        for row, col in zip(rows, cols):
            sid = int(ids[row]); k = int(keep[col])
            if cost[row, col] >= 1e8: failed.append(sid); continue
            records[sid] = dict(sid=sid, patch=patch, owner=owner, link=surface.model.body(surface.body_ids[owner]).name,
                source_joint_index=joint, face_id=int(surface.face_offsets[owner]+pool['face_ids'][k]),
                barycentric=pool['barycentric'][k].tolist(), local_position_m=pool['points'][k].tolist(),
                local_normal=pool['normals'][k].tolist(), normal_error_deg=float(np.degrees(np.arccos(dot[row, col]))))
            if normal_limit_deg is not None:
                records[sid]['normal_valid']=bool(dot[row,col]>=np.cos(np.deg2rad(normal_limit_deg))-1e-12)
    result = dict(version=version, policy=POLICY, ready=not failed and all(r is not None for r in records),
        records=records, failed_sids=sorted(set(failed)), geometry_version=topo['geometry_version'],
        normal_warning_sids=[r['sid'] for r in records if r is not None and r['normal_error_deg'] > 60],
        binding_version=topo['binding_version'], mirror_matrix=source.mirror.tolist(), asset_sha256=surface.asset_hash,
        source_surface='moxian_original_skn', virtual_sensors=True,
        source_note='Moxian material binding; robot correspondence is anatomical, not measured')
    if normal_limit_deg is not None:
        result.update(policy=POLICY+'_normal_compatible_v1',normal_limit_deg=float(normal_limit_deg),
            normal_rejected_sids=[r['sid'] for r in records if r is not None and not r['normal_valid']],
            correspondence_complete=all(r is not None and r['normal_valid'] for r in records))
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(result)); tmp.replace(path)
    return result


class MoxianCARDSource:
    """Geometry-only adapter for the unchanged, 16-region CARD/CEM objective.

    The SLSQP graph keeps separate sensor patches and semantic link labels.
    Legacy CARD intentionally retains its existing one-anchor-per-region rule.
    """
    def __init__(self, source):
        self.material = source; self.patches = source.patches; self.mirror = source.mirror
        self.topology = source.topology
        self.regions = np.zeros(460, int)
        for sid, patch in enumerate(self.patches):
            if patch >= 14: continue
            f = 0 if patch < 2 else 1+(patch-2)//3
            k = (2 if patch == 0 else 1) if f == 0 else 2-(patch-2)%3
            self.regions[sid] = 1+3*f+k
        self.region_faces = [np.unique(np.concatenate(source.regions[14:]))]
        for f in range(5):
            self.region_faces.extend(source.regions[(0 if k == 2 else 1) if f == 0 else 2+3*(f-1)+(2-k)] for k in range(3))

    def source(self, frame):
        result = self.material.read(frame, False)
        return result['vertices'], result['joints'], result['taxels'], result['normals']

    def chart(self, region, vertices, joints):
        if region == 0: origin = joints[0]; axis = unit(joints[9]-origin)
        else:
            f, k = divmod(region-1, 3); origin = joints[1+4*f+k]; axis = unit(joints[2+4*f+k]-origin)
        tri = vertices[self.material.faces[self.region_faces[region]]]
        normal = unit(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]).sum(0))
        if np.linalg.det(self.mirror) < 0: normal = -normal
        return origin, axis, normal, tri

    def extract(self, frame, enabled=True):
        from retarget.baseline.card.geometry import contact_weights
        from retarget.baseline.card.contact_force import estimate_normal_resultant
        v, j, p, n = self.source(frame)
        weights = contact_weights(frame.get('source_contact') or {}, self.patches, float(frame['timestamp_ms']),
            frame.get('tactile_timestamp_ms') if isinstance(frame.get('tactile_timestamp_ms'), (int, float)) else -1e20, enabled)
        anchors = []
        for region in range(16):
            ids = np.flatnonzero(weights['mask'] & (self.regions == region))
            if not len(ids): continue
            values = weights['response'][ids]; force = float(values.sum())
            values = values if force > 1e-12 else np.ones(len(ids))
            o, axis, normal, triangles = self.chart(region, v, j)
            point, face, bc = closest_triangle(triangles, np.average(p[ids], axis=0, weights=values))
            f, k = divmod(region-1, 3) if region else (-1, -1)
            u = float(np.clip((point-o) @ axis/max(np.linalg.norm(j[2+4*f+k]-o), 1e-12), 0, 1)) if region else 0.
            anchors.append(dict(region=region, finger=f, segment=k, owner=0 if region == 0 else 2+4*f+k,
                position_m=point.tolist(), direction=unit(np.average(n[ids], axis=0, weights=values)).tolist(), force=force,
                uv_m=[float((point-o) @ axis), float((point-o) @ unit(np.cross(normal, axis)))], u=u,
                source_face=int(self.region_faces[region][face]), barycentric=bc.tolist(), sids=ids.tolist(),
                force_estimate=estimate_normal_resultant(n[ids], weights['response'][ids]), source_material_registered=False))
        return (v, j, p, n), weights, anchors

    def relocate(self, anchor, uv, vertices, joints):
        from retarget.baseline.card.geometry import chart_point
        a = dict(anchor); o, x, n, tri = self.chart(a['region'], vertices, joints)
        point, f, bc = chart_point(tri, o, x, n, uv, np.asarray(a['position_m']))
        a.update(position_m=point.tolist(), uv_m=np.asarray(uv).tolist(),
                 source_face=int(self.region_faces[a['region']][f]), barycentric=bc.tolist())
        return a
