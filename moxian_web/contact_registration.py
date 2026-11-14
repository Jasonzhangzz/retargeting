"""Measured glove-to-MANO material registration, independent of retargeting.

Reference positions/normals must be measured in each posed mesh's wrist frame.
Only training frames select a face and barycentric coordinates; validation
frames never update them. Patch confirmation and manual clicks are not evidence.
"""
import copy
import hashlib
import json
import numpy as np

REVISION = 'measured_material_holdout_v1'
POSITION_LIMIT_MM = 2.
NORMAL_LIMIT_DEG = 30.
MIN_FRAMES = 2


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


def context(topology, serial):
    return dict(serial=serial, hand_side=topology['hand_side'],
                weights_hash=topology['weights_hash'], geometry_version=topology['geometry_version'],
                mesh_sha256=digest(topology['faces']),
                assignments_sha256=digest(topology['sid_patch_indices']))


def finite_array(value, shape, name):
    a = np.asarray(value, float)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError('Invalid ' + name)
    return a


def unit(value):
    value = np.asarray(value, float)
    length = np.linalg.norm(value, axis=-1, keepdims=True)
    if np.any(length < 1e-10):
        raise ValueError('Degenerate registration normal or triangle')
    return value / length


def closest_barycentric(triangles, targets):
    """Least squares on a simplex, also for stacked multi-pose coordinates."""
    a, b, c = np.moveaxis(triangles, -2, 0)
    ab, ac, d = b-a, c-a, targets-a
    aa = (ab*ab).sum(-1); bb = (ab*ac).sum(-1); cc = (ac*ac).sum(-1)
    ad = (ab*d).sum(-1); cd = (ac*d).sum(-1); det = aa*cc-bb*bb
    safe = np.where(det > 1e-24, det, 1.)
    v = (cc*ad-bb*cd)/safe; w = (aa*cd-bb*ad)/safe
    bc = np.stack((1-v-w, v, w), -1)
    choices = [bc]
    scores = [np.where((bc.min(-1) >= 0) & (det > 1e-24),
                       np.sum(((bc[..., None]*triangles).sum(-2)-targets)**2, axis=-1), np.inf)]
    for i, j in ((0, 1), (1, 2), (2, 0)):
        start = triangles[..., i, :]; edge = triangles[..., j, :]-start
        t = np.clip(((targets-start)*edge).sum(-1)/np.maximum((edge*edge).sum(-1), 1e-24), 0, 1)
        bary = np.zeros_like(bc); bary[..., i] = 1-t; bary[..., j] = t
        choices.append(bary); scores.append(((start+t[..., None]*edge-targets)**2).sum(-1))
    best = np.stack(scores, -1).argmin(-1)
    return np.take_along_axis(np.stack(choices, -2), best[..., None, None], axis=-2)[..., 0, :]


def fit_registration(topology, serial, measurements):
    expected = context(topology, serial)
    if measurements.get('context') != expected:
        raise ValueError('Registration measurements do not match device, MANO or SID assignments')
    if measurements.get('coordinate_frame') != 'mano_wrist_local' or measurements.get('units') != 'm':
        raise ValueError('Measured references must use mano_wrist_local metres')
    provenance = measurements.get('measurement_source')
    if not isinstance(provenance, str) or not provenance.strip():
        raise ValueError('Independent reference measurement provenance is required')
    faces = np.asarray(topology['faces'], int); patches = np.asarray(topology['face_patches'])
    groups = {}; seen = set(); frame_ids = {'train': [], 'validation': []}
    for frame in measurements.get('frames', []):
        fid = frame.get('id'); split = frame.get('split')
        if not isinstance(fid, str) or not fid or fid in seen or split not in frame_ids:
            raise ValueError('Unique frame IDs and explicit train/validation split required')
        seen.add(fid); frame_ids[split].append(fid)
        vertices = finite_array(frame['vertices_m'], (len(topology['rest_vertices_m']), 3), 'posed vertices')
        if np.max(abs(vertices)) > .6:
            raise ValueError('Registration mesh must use wrist-local metres')
        present = set()
        for observation in frame.get('observations', []):
            sid = observation.get('sid')
            if type(sid) is not int or not 0 <= sid < 460 or sid in present:
                raise ValueError('Unique SID observation required per frame')
            present.add(sid)
            point = finite_array(observation['position_m'], (3,), 'measured point')
            if np.max(abs(point)) > .6:raise ValueError('Measured point must use wrist-local metres')
            normal = unit(finite_array(observation['normal'], (3,), 'measured outward normal'))
            groups.setdefault(sid, {'train': [], 'validation': []})[split].append((vertices, point, normal))
    if not groups:raise ValueError('No independent measured SID observations')
    records = {}; failures = {}
    for sid, data in sorted(groups.items()):
        if any(len(data[k]) < MIN_FRAMES for k in data):
            failures[str(sid)] = 'At least two training and two held-out frames per SID required'; continue
        candidates = np.flatnonzero(patches == topology['sid_patch_indices'][sid])
        if not len(candidates):raise ValueError('SID has no anatomical surface')
        # Flatten all training poses into one least-squares system per face.
        tri = np.stack([v[faces[candidates]] for v, _, _ in data['train']])
        tri = tri.transpose(1, 2, 0, 3).reshape(len(candidates), 3, -1)
        target = np.concatenate([p for _, p, _ in data['train']])
        bc = closest_barycentric(tri, target)
        errors = ((tri*bc[..., None]).sum(1)-target).reshape(len(candidates), -1, 3)
        score = (errors**2).mean((1, 2))
        # Measured normals disambiguate nearby palmar/dorsal surfaces.
        for v, _, n in data['train']:
            t = v[faces[candidates]]; normals = unit(np.cross(t[:, 1]-t[:, 0], t[:, 2]-t[:, 0]))
            score[normals@n < np.cos(np.deg2rad(NORMAL_LIMIT_DEG))] = np.inf
        chosen = int(score.argmin())
        if not np.isfinite(score[chosen]):failures[str(sid)] = 'No surface matches measured training normals'; continue
        face = int(candidates[chosen]); bary = bc[chosen]
        metrics = {}
        for split, rows in data.items():
            distances = []; angles = []
            for v, p, n in rows:
                t = v[faces[face]]; distances.append(float(np.linalg.norm(bary@t-p)*1000))
                nn = unit(np.cross(t[1]-t[0], t[2]-t[0]))
                angles.append(float(np.rad2deg(np.arccos(np.clip(nn@n, -1, 1)))))
            metrics[split] = dict(frames=len(rows), max_error_mm=max(distances),
                                  rms_error_mm=float(np.sqrt(np.mean(np.square(distances)))),
                                  max_normal_error_deg=max(angles))
        if any(m['max_error_mm'] > POSITION_LIMIT_MM or m['max_normal_error_deg'] > NORMAL_LIMIT_DEG for m in metrics.values()):
            failures[str(sid)] = metrics; continue
        records[str(sid)] = dict(face=face, barycentric=bary.tolist(), metrics=metrics)
    result = dict(schema_version=1, revision=REVISION, context=expected,
                  measurement_source=provenance, evidence_sha256=digest(measurements),
                  frame_ids=frame_ids, records=records, failures=failures, accepted=not failures,
                  complete=len(records) == 460, position_limit_mm=POSITION_LIMIT_MM,
                  normal_limit_deg=NORMAL_LIMIT_DEG)
    result['version'] = digest(result)
    return result


def validate_registration(artifact, topology, serial):
    """Validate context and quantitative evidence, never just an accepted bit."""
    if not isinstance(artifact,dict):raise ValueError('Invalid measured registration artifact')
    if artifact.get('schema_version') != 1 or artifact.get('revision') != REVISION or artifact.get('context') != context(topology, serial):
        raise ValueError('Measured contact registration is stale or incompatible')
    payload = {k: v for k, v in artifact.items() if k != 'version'}
    if artifact.get('version') != digest(payload) or artifact.get('accepted') is not True or artifact.get('failures'):
        raise ValueError('Measured registration failed independent holdout acceptance')
    if not artifact.get('measurement_source') or len(artifact.get('evidence_sha256', '')) != 64:
        raise ValueError('Missing measured reference evidence')
    splits = artifact.get('frame_ids', {}); train = splits.get('train', []); valid = splits.get('validation', [])
    if len(set(train)) < MIN_FRAMES or len(set(valid)) < MIN_FRAMES or set(train) & set(valid):
        raise ValueError('Independent registration holdout frames required')
    records = artifact.get('records')
    if not isinstance(records, dict) or not records:raise ValueError('Empty material registration')
    for key, record in records.items():
        sid = int(key); face = record['face']; bary = finite_array(record['barycentric'], (3,), 'barycentric coordinates')
        if str(sid) != key or not 0 <= sid < 460 or type(face) is not int or not 0 <= face < len(topology['faces']):
            raise ValueError('Invalid registered SID or face')
        if bary.min() < 0 or abs(bary.sum()-1) > 1e-6 or topology['face_patches'][face] != topology['sid_patch_indices'][sid]:
            raise ValueError('Registered material point outside its anatomical region')
        for split in ('train', 'validation'):
            m = record['metrics'][split]
            if type(m['frames']) is not int or not MIN_FRAMES <= m['frames'] <= len(splits[split]):
                raise ValueError('Insufficient independent measured frames')
            for field, limit in (('max_error_mm', POSITION_LIMIT_MM), ('rms_error_mm', POSITION_LIMIT_MM), ('max_normal_error_deg', NORMAL_LIMIT_DEG)):
                v = m[field]
                if not np.isfinite(v) or not 0 <= v <= limit:raise ValueError('Contact registration holdout exceeds limits')
    return copy.deepcopy(records)


def registration_status(artifact, topology, serial, face_ids, barycentric):
    status = dict(status='unregistered', verified_sids=[], verified_count=0, complete=False,
                  physical_registration_verified=False, revision=REVISION)
    if not artifact:return status
    try:records = validate_registration(artifact, topology, serial)
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        return {**status, 'status': 'stale_or_invalid', 'reason': str(exc)}
    verified = [int(s) for s, r in records.items() if face_ids[int(s)] == r['face'] and
                np.allclose(barycentric[int(s)], r['barycentric'], atol=1e-7, rtol=0)]
    complete = len(verified) == 460
    return {**status, 'status': 'complete' if complete else 'partial', 'verified_sids': sorted(verified),
            'verified_count': len(verified), 'complete': complete, 'physical_registration_verified': complete,
            'version': artifact['version'], 'measurement_source': artifact['measurement_source'],
            'evidence_sha256': artifact['evidence_sha256']}


def material_records(artifact,topology,serial):
    """Shape edits keep material identity, but invalidate measured validation."""
    if not isinstance(artifact,dict):raise ValueError('Invalid measured registration artifact')
    if not isinstance(artifact.get('context'),dict):raise ValueError('Invalid registration context')
    compatible=dict(topology)
    compatible['geometry_version']=(artifact.get('context') or {}).get('geometry_version')
    return validate_registration(artifact,compatible,serial)
