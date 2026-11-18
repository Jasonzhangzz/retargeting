"""Calibration-owned IK starting points, never pose targets or frame history."""
from __future__ import annotations
import copy
import hashlib
import json
import numpy as np

BANK_REVISION = 'moxian_calibration_ik_seeds_v1'
SOLVER_REVISION = 'moxian_studio_native_qp_v3_calibration_seeds'


def _digest(value):
    # Shared artifact encoding, kept independent of SciPy for the web process.
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,
        separators=(',',':')).encode()).hexdigest()


def runtime_revision(candidate):
    bank = candidate.get('source_ik_initialization')
    return _digest(dict(model_hash=candidate['model_hash'], solver=SOLVER_REVISION,
                       calibration_seeds=bank['sha256'] if bank else None))


def validate_bank(candidate):
    bank = candidate.get('source_ik_initialization')
    if bank is None:
        return
    definition = candidate['bone_extrinsics']
    if (bank.get('revision') != BANK_REVISION or
            bank.get('model_hash') != candidate['model_hash'] or
            bank.get('source_capture_hash') != candidate.get('source_capture_hash') or
            bank.get('joint_names') != definition['rig']['joint_names']):
        raise ValueError('源IK标定初值与模型或采集来源不匹配')
    payload = {k:v for k,v in bank.items() if k != 'sha256'}
    if bank.get('sha256') != _digest(payload):
        raise ValueError('源IK标定初值校验失败')
    seeds = bank.get('seeds', [])
    limits = np.asarray(definition['rig']['limits'], float)
    q = np.asarray([s['joint_positions_rad'] for s in seeds], float)
    if (not 1 <= len(seeds) <= 18 or q.shape != (len(seeds), len(limits)) or
            not np.isfinite(q).all() or np.any(q < limits[:, 0]-1e-10) or
            np.any(q > limits[:, 1]+1e-10)):
        raise ValueError('源IK标定初值维度或关节限位无效')


def validate_recorded_source(profile, source):
    if profile.get('bone_extrinsics'):
        validate_bank(profile)
        if source.get('skeleton_runtime_revision') != runtime_revision(profile):
            raise ValueError('旧源IK骨架数据已失效，请从原始EMF重建或重新采集')


def with_calibration_seeds(candidate, rows, cancel, progress=None, fitted_angles=None):
    """Run the same native objective from calibration/generic starts per pose.

    Only training observations are accepted. Persist refined starting points
    and audit residuals, then deserialize before validation replay so the fit
    cannot depend on an in-memory recovery cache.
    """
    from scipy.spatial.transform import Rotation
    from .bone_extrinsics import FINGERS, digest, validate_candidate
    from .studio_emf_solver import NativeStudioQP
    if not rows or any(r.get('partition') != 'train' for r in rows):
        raise ValueError('源IK初值仅允许从标定训练动作生成')
    result = copy.deepcopy(candidate)
    result.pop('source_ik_initialization', None)
    validate_candidate(result)
    native = NativeStudioQP(result['bone_extrinsics'])
    try:
        k = native.kinematics
        reference = np.asarray(result['bone_extrinsics']['rest_angles_rad'], float)
        generic = [k.limits[:, 0]+f*(k.limits[:, 1]-k.limits[:, 0]) for f in (.5, .25, .75)]
        if fitted_angles is not None:
            fitted_angles = np.asarray(fitted_angles, float)
            if fitted_angles.shape != (len(rows), native.nq) or not np.isfinite(fitted_angles).all():
                raise ValueError('标定姿态初值维度无效')
        seeds = []
        for index, row in enumerate(rows):
            if cancel.is_set():
                raise InterruptedError('标定已取消')
            if progress:
                progress(dict(state='running', phase='source_ik_seeds', progress=None,
                    completed=index, total=len(rows), message='生成并保存标定源IK分支初值'))
            p = np.asarray([row['fingers'][f]['position'] for f in FINGERS], float)
            r = Rotation.from_quat([row['fingers'][f]['quaternion_xyzw'] for f in FINGERS]).as_matrix()
            a = k.alignment
            p = p@a[:3, :3].T+a[:3, 3]; r = a[:3, :3]@r
            q, report = native.solve(reference, p, r, 1/60, False)
            starts = list(generic)
            if fitted_angles is not None:
                starts.insert(0, np.clip(fitted_angles[index], k.limits[:, 0], k.limits[:, 1]))
            for start in starts:
                other, score = native.solve(start, p, r, 1/60, False, reference=reference, probe=True)
                for f, ids in enumerate(k.finger_indices):
                    if score[f, 3] < report[f, 3]:
                        q[ids] = other[ids]; report[f] = score[f]
            score = native.score(q, p, r, reference, False)
            seeds.append(dict(calibration_pose=row.get('native_pose', row.get('static_key', str(index))),
                joint_positions_rad=q.tolist(), position_error_mm=(score[:, 0]*1000).tolist(),
                rotation_error_deg=np.rad2deg(score[:, 1]).tolist(), cost=score[:, 3].tolist()))
        bank = dict(revision=BANK_REVISION, model_hash=result['model_hash'],
            source_capture_hash=result.get('source_capture_hash'),
            joint_names=list(k.joint_names), source='calibration_training_emf_same_native_objective',
            selection='current_emf_objective_only', seeds=seeds)
        bank['sha256'] = digest(bank)
        result['source_ik_initialization'] = bank
        # Exercise the artifact boundary before its validation replay.
        result = json.loads(json.dumps(result, allow_nan=False))
        validate_candidate(result)
        return result
    finally:
        native.close()
