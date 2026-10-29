"""Numeric-only grasp records: finger joints and object pose in wrist frame.

Schema 4 (reads schemas 1/2/3), metres/radians, stages IK/output, quaternion wxyz.
Hand geometry is shared; primitive vertices are regenerated, never serialized.
"""
from functools import lru_cache
from pathlib import Path
import copy
import hashlib
import json
import os
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from retarget.grasp_results import ROOT, METHODS, method_name

SHAPES = ('mesh', 'cylinder', 'sphere', 'cuboid', 'disk', 'plate')
OBJECT_ASSETS = ROOT/'assets/grasp_objects'
FILE = 'grasps.npy'
DTYPE_V1 = np.dtype([
    ('version', 'u1'), ('type_id', 'u1'), ('hand_id', 'u1'), ('method_id', 'u1'),
    ('shape_id', 'u1'), ('dimensions', '<f8', (4,)),
    ('mesh_id', 'u1', (32,)), ('mesh_scale', '<f8'),
    ('stage_valid', 'u1', (2,)), ('hand_config', '<f8', (2, 20)),
    ('object_pose_wrist', '<f8', (2, 7)),
    # FC, acceptance, required contacts, count, object/self penetration (mm),
    # optimizer hard validity, initial penetration, independent export/closure.
    ('checks', '<f8', (10,)), ('cost_groups', '<f8', (5,))])
DTYPE_V2 = np.dtype(DTYPE_V1.descr + [('initialization_failure_mask', '<u4')])
# shape-first flag, forbidden gap (mm), maximum bone deviation (degrees), IK validity.
DTYPE_V3 = np.dtype(DTYPE_V2.descr + [('ik_geometry_checks', '<f8', (4,))])
DTYPE = np.dtype(DTYPE_V3.descr + [('hold_offset', '<f8', (20,)),
    ('prediction_checks', '<f8', (4,))])  # enabled, maximum shift mm/angle deg, stable


def legacy_rows(directory):
    directory = Path(directory)
    sources = ([directory/'summary.json'] if (directory/'summary.json').is_file() else
        sorted([*directory.glob('node_*/summary.json'), *directory.glob('type_*/summary.json')]))
    return [row for source in sources for row in json.loads(source.read_text())]


@lru_cache(maxsize=8)
def records(path, stamp):
    # mmap avoids decompressing geometry/contact/history arrays on every load.
    array = np.load(path, mmap_mode='r', allow_pickle=False)
    version = 4 if array.dtype == DTYPE else 3 if array.dtype == DTYPE_V3 else 2 if array.dtype == DTYPE_V2 else 1 if array.dtype == DTYPE_V1 else None
    if version is None or array.ndim != 1 or not np.all(array['version'] == version):
        raise ValueError(f'Unsupported compact grasp schema: {path}')
    return array


def load_records(path):
    path = Path(path)
    return records(str(path), path.stat().st_mtime_ns)


def is_compact_case(case):
    case = Path(case)
    return case.name.startswith('case_') and (case.parent/FILE).is_file()


def record_for_case(case):
    case = Path(case)
    return load_records(case.parent/FILE)[int(case.name.removeprefix('case_'))]


def read_rows(directory):
    directory = Path(directory)
    array = load_records(directory/FILE)
    from retarget.grasp_primitives import catalog
    names = {r['type_id']: r['template'] for r in catalog(ROOT/'third_party/Dexonomy')}
    rows = []
    for i, record in enumerate(array):
        shape = SHAPES[int(record['shape_id'])]
        r, h, radial, axial = record['dimensions']
        c = record['checks']
        from retarget.grasp_initialization import INITIALIZATION_FAILURES
        mask = int(record['initialization_failure_mask']) if 'initialization_failure_mask' in record.dtype.names else 0
        failures = [name for bit, name in enumerate(INITIALIZATION_FAILURES) if mask & (1 << bit)]
        failed = bool(record['stage_valid'][0] and not record['stage_valid'][1])
        geometry = record['ik_geometry_checks'] if 'ik_geometry_checks' in record.dtype.names else None
        prediction = record['prediction_checks'] if 'prediction_checks' in record.dtype.names else None
        motion_failed = bool(prediction is not None and prediction[0] and not prediction[3])
        prediction_info = (dict(explicit_prediction=dict(predicted_max_displacement_mm=float(prediction[1]),
            predicted_max_rotation_deg=float(prediction[2]), predicted_hold_feasible=bool(prediction[3])))
            if prediction is not None and prediction[0] else {})
        shape_info = (dict(contacts_optional=True, initialization_policy='shape_first',
            forbidden_region_min_gap_mm=float(geometry[1]), forbidden_region_clearance_mm=1.,
            bone_direction_max_deviation_deg=float(geometry[2]), initialization_valid=bool(geometry[3]))
            if geometry is not None and geometry[0] else {})
        rows.append(dict(case=str(directory/f'case_{i:06d}'), hand_model=f"wuji{int(record['hand_id'])}",
            template=names.get(int(record['type_id']), f"{int(record['type_id'])}_Grasp"),
            primitive=dict(kind=shape, radius_m=float(r), height_m=float(h),
                           radial_scale=float(radial), axial_scale=float(axial)),
            force_closure=dict(force_closure=bool(c[0])), feasible=bool(c[1]),
            required_groups_satisfied=bool(c[2]), active_contacts=int(c[3]),
            max_object_penetration_mm=float(c[4]), max_self_penetration_mm=float(c[5]),
            optimizer_success=bool(c[6]), optimizer_hard_constraints_satisfied=bool(c[6]),
            initialization=dict(initial_penetration_mm=float(c[7]), failure_reasons=failures,
                **(dict(shape_info, initialization_valid=False) if failed else shape_info)), compact=True,
            termination='initialization_failed' if failed else 'predicted_motion_exceeds_tolerance' if motion_failed else 'static_solution',
            hold_offset=record['hold_offset'].tolist() if 'hold_offset' in record.dtype.names else None,
            predicted_hold_feasible=bool(prediction[3]) if prediction is not None and prediction[0] else None,
            solver_diagnostics=dict(**prediction_info, cost_groups=dict(zip(
                ('contact', 'style', 'smooth', 'collision', 'object_pose'), map(float, record['cost_groups']))))))
    return rows


def relative_pose(model, qpos):
    """Use actual FK wrist/object frames, including root translations/rotations."""
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    wrist, obj = model.body('right_palm_link').id, model.body('grasp_object').id
    rw = data.xmat[wrist].reshape(3, 3)
    ro = data.xmat[obj].reshape(3, 3)
    return np.r_[rw.T@(data.xpos[obj]-data.xpos[wrist]),
        Rotation.from_matrix(rw.T@ro).as_quat(scalar_first=True)]


def write_batch(rows, destination, *, method, audited=()):
    """Export a snapshot atomically; all NPY fields contain only numbers."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    proofs = {str(Path(r['case']).resolve()): r for r in audited}
    output = np.zeros(len(rows), dtype=DTYPE)
    for i, row in enumerate(rows):
        case = Path(row['case'])
        report = json.loads((case/'result.json').read_text())
        model = mujoco.MjModel.from_xml_path(str(case/'scene.xml'))
        record = output[i]
        record['version'] = 4
        record['type_id'] = int(row['template'].split('_')[0])
        record['hand_id'] = 2 if row.get('hand_model') == 'wuji2' else 1
        record['method_id'] = METHODS.index(method)
        primitive = row.get('primitive')
        if primitive:
            record['shape_id'] = SHAPES.index(primitive['kind'])
            record['dimensions'] = [primitive['radius_m'], primitive['height_m'],
                primitive['radial_scale'], primitive['axial_scale']]
        else:
            import trimesh
            mesh = Path(report['object'])
            digest = hashlib.sha256(mesh.read_bytes()).digest()
            record['mesh_id'] = np.frombuffer(digest, dtype=np.uint8)
            record['mesh_scale'] = report['scale']
            OBJECT_ASSETS.mkdir(parents=True, exist_ok=True)
            shared = OBJECT_ASSETS/(digest.hex()+'.obj')
            if not shared.exists():
                trimesh.load(mesh, force='mesh', process=False).export(shared)
            vertices = model.mesh_vert[model.mesh_vertadr[-1]:]
            extent = np.ptp(vertices, axis=0)
            record['dimensions'] = [max(extent[:2])/2, extent[2], 1., 1.]
        with np.load(case/'grasp.npz', allow_pickle=False) as saved:
            if 'hold_action' in saved.files and 'hand_qpos' in saved.files:
                record['hold_offset'] = saved['hold_action'][6:26]-saved['hand_qpos'][6:26]
            for stage, key in enumerate(('initial_qpos', 'qpos')):
                if key in saved.files:
                    qpos = saved[key]
                    record['hand_config'][stage] = qpos[6:26]
                    record['object_pose_wrist'][stage] = relative_pose(model, qpos)
                    record['stage_valid'][stage] = saved['stage_valid'][stage] if 'stage_valid' in saved.files else 1
        from retarget.grasp_initialization import INITIALIZATION_FAILURES
        for reason in row.get('initialization', {}).get('failure_reasons', []):
            record['initialization_failure_mask'] |= 1 << INITIALIZATION_FAILURES.index(reason)
        check = row
        initial = row.get('initialization', {})
        record['ik_geometry_checks'] = [initial.get('contacts_optional', False),
            initial.get('forbidden_region_min_gap_mm') or 0.,
            initial.get('bone_direction_max_deviation_deg') or 0., initial.get('initialization_valid', False)]
        proof = proofs.get(str(case.resolve()), {})
        record['checks'] = [check['force_closure']['force_closure'], check['feasible'],
            check['required_groups_satisfied'], check['active_contacts'],
            check['max_object_penetration_mm'], check['max_self_penetration_mm'],
            check.get('optimizer_hard_constraints_satisfied', check.get('optimizer_success', False)),
            initial.get('initial_penetration_mm', 0.),
            proof.get('saved_data_valid', False), proof.get('closure_agrees', False)]
        costs = row.get('solver_diagnostics', {}).get('cost_groups', {})
        prediction = row.get('solver_diagnostics', {}).get('explicit_prediction', {})
        if prediction:
            record['prediction_checks'] = [1., prediction.get('predicted_max_displacement_mm', prediction['predicted_displacement_mm']),
                prediction.get('predicted_max_rotation_deg', prediction['predicted_rotation_deg']), prediction.get('predicted_hold_feasible', False)]
        record['cost_groups'] = [costs.get(k, 0.) for k in
            ('contact', 'style', 'smooth', 'collision', 'object_pose')]
    temporary = destination/'grasps.pending.npy'
    np.save(temporary, output, allow_pickle=False)
    os.replace(temporary, destination/FILE)
    return destination/FILE


@lru_cache(maxsize=2)
def hand_model(hand):
    from retarget.grasp_hand import hand_tree
    tree = hand_tree(hand)
    for parent in tree.iter():
        for child in list(parent):
            if child.tag == 'geom' and child.get('type') == 'querypoint':
                parent.remove(child)
    return mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding='unicode'))


def object_geometry(record):
    if record['shape_id']:
        from retarget.grasp_primitives import primitive_geometry
        vertices, faces = primitive_geometry(SHAPES[int(record['shape_id'])], *record['dimensions'][:2])
    else:
        import trimesh
        path = OBJECT_ASSETS/(bytes(record['mesh_id']).hex()+'.obj')
        mesh = trimesh.load(path, force='mesh', process=False)
        vertices, faces = np.asarray(mesh.vertices)*record['mesh_scale'], np.asarray(mesh.faces)
    vertices = vertices-(vertices.min(0)+vertices.max(0))/2
    return vertices.astype(np.float32), faces.astype(np.uint32)


def run_compact(args):
    """Keep solver setup/export intermediates temporary, persist numbers only."""
    from retarget.grasp import run
    from scripts.validate_grasp_matrix import audit_batch
    destination = args.output.expanduser().resolve()/args.name
    destination.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='grasp_solve_') as workspace:
        temporary = copy.copy(args)
        temporary.output = Path(workspace)
        temporary._compact_output = destination
        source = run(temporary)
        audited = audit_batch(source)
        if any(not r.get('saved_data_valid') or not r.get('closure_agrees') for r in audited):
            raise RuntimeError('Independent archive audit failed before compact export')
        rows = legacy_rows(source)
        write_batch(rows, destination, method=method_name(args.grasp_optimizer, args.explicit_pose), audited=audited)
    print(f'[saved numeric grasps] {destination/FILE}', flush=True)
    return destination


def audit_records(directory):
    """Check numeric records; report the independent pre-compaction proofs."""
    output = []
    for row, record in zip(read_rows(directory), load_records(Path(directory)/FILE)):
        finite = all(np.isfinite(record[name]).all() for name in record.dtype.names)
        quat = record['object_pose_wrist'][:, 3:]
        normalized = np.allclose(np.linalg.norm(quat[record['stage_valid'].astype(bool)], axis=-1), 1., atol=1.e-7)
        output.append(dict(case=row['case'], template=row['template'],
            saved_data_valid=bool(finite and normalized and record['checks'][8]),
            closure_agrees=bool(record['checks'][9]),
            independent_proof_source='full export/FK/closure audit before numeric compaction',
            numeric_structure_checked=True))
    return output
