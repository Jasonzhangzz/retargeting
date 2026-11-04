"""Explicit experiment extensions to upstream GraspADMM, outside its frozen tree.

Penetration changes the existing filter configuration only. Wrench trials are
post-solve MuJoCo rollouts; they do not add a grasp optimization objective.
"""
from dataclasses import asdict, dataclass
import math
import time

import mujoco
import numpy as np


@dataclass(frozen=True)
class Settings:
    hand_object_penetration_m: float = .002
    self_penetration_m: float = .002
    torque_scale_mg_radius: float = 1.
    torque_nm: float | None = None
    translation_limit_m: float = .05
    rotation_limit_deg: float = 15.
    steps: int = 500
    check_every: int = 50

    def __post_init__(self):
        for name, value in asdict(self).items():
            if value is not None and not math.isfinite(value):
                raise ValueError(name + ' must be finite')
        if min(self.hand_object_penetration_m, self.self_penetration_m) < 0:
            raise ValueError('Penetration tolerances must be nonnegative')
        if min(self.torque_scale_mg_radius, self.translation_limit_m, self.rotation_limit_deg) <= 0:
            raise ValueError('Wrench scale and pose thresholds must be positive')
        if self.torque_nm is not None and self.torque_nm <= 0:
            raise ValueError('torque_nm must be positive')
        if self.steps < 1 or self.check_every < 1:
            raise ValueError('Rollout steps must be positive')


def configure_penetration(cfg, settings):
    """Use upstream's signed-distance filter; preserve pregrasp clearance."""
    cfg.op.grasp.filter.collision.ho_thre = -settings.hand_object_penetration_m
    cfg.op.grasp.filter.collision.hh_thre = -settings.self_penetration_m
    cfg.op.pregrasp.filter.collision.hh_thre = -settings.self_penetration_m
    return cfg


def world_wrench(model, data, body_id, force, torque):
    """Apply a world-frame wrench at body COM, without a force moment arm.

    xfrc_applied is Cartesian force then torque. MuJoCo converts it at each
    step, keeping a pure world-axis torque correct as the object rotates.
    """
    force, torque = np.asarray(force, dtype=float), np.asarray(torque, dtype=float)
    if force.shape != (3,) or torque.shape != (3,) or not np.isfinite(np.r_[force, torque]).all():
        raise ValueError('Expected finite three-vectors')
    if not 0 < body_id < model.nbody:
        raise ValueError('Wrench target must be a non-world body')
    data.qfrc_applied[:] = 0.
    data.xfrc_applied[:] = 0.
    data.xfrc_applied[body_id, :3] = force
    data.xfrc_applied[body_id, 3:] = torque


def displacement(reference, current):
    reference, current = np.asarray(reference), np.asarray(current)
    translation = float(np.linalg.norm(current[:3] - reference[:3]))
    a, b = reference[3:] / np.linalg.norm(reference[3:]), current[3:] / np.linalg.norm(current[3:])
    rotation = float(np.rad2deg(2 * np.arccos(np.clip(abs(a @ b), 0., 1.))))
    return translation, rotation


def evaluate_wrenches(env, ctrl_qpos, ctrl_type, ext_fdir, target_obj_pose,
                      object_radius, settings=Settings()):
    """Six force and six pure torque trials, each with an independent reset.

    The approach/grasp/squeeze trajectory is the original upstream trajectory.
    Every direction is recorded, even if a previous direction fails. End states
    are retained for replay; full optimization or simulation histories are not.
    """
    if not np.isfinite(object_radius) or object_radius <= 0:
        raise ValueError('A physical object radius in metres is required')
    model, data = env._model, env._data
    body = env._obj_active_bodyid
    mass = float(model.body_subtreemass[body])
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError('Object mass must be positive')
    force_n = mass * 9.8
    torque_nm = settings.torque_nm if settings.torque_nm is not None else settings.torque_scale_mg_radius * force_n * object_radius
    directions = [(axis, sign, np.eye(3)[axis] * sign) for axis in range(3) for sign in (1, -1)]
    rows = []
    start = time.perf_counter()
    for kind in ('force', 'torque'):
        for axis, sign, direction in directions:
            trial_start = time.perf_counter()
            row = dict(kind=kind, axis='XYZ'[axis], sign=sign, passed=False,
                       force_n=(direction * force_n if kind == 'force' else np.zeros(3)).tolist(),
                       torque_nm=(direction * torque_nm if kind == 'torque' else np.zeros(3)).tolist(),
                       simulated_steps=0, failure=None)
            try:
                env.reset_qpos(np.asarray(ctrl_qpos[0]).reshape(-1))
                world_wrench(model, data, body, np.zeros(3), np.zeros(3))
                env.run_trajectory(ctrl_qpos, ctrl_type, ext_fdir)
                prepared_pose = env.get_active_obj_pose().copy()
                row['prepared_object_pose'] = prepared_pose.tolist()
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    raise FloatingPointError('Nonfinite state after approach/grasp/squeeze')
                trans, rot = displacement(target_obj_pose, prepared_pose)
                if trans >= settings.translation_limit_m or rot >= settings.rotation_limit_deg:
                    row['failure'] = 'preload_drift'
                else:
                    world_wrench(model, data, body, row['force_n'], row['torque_nm'])
                    while row['simulated_steps'] < settings.steps:
                        count = min(settings.check_every, settings.steps-row['simulated_steps'])
                        env.step_sim(substep=count)
                        row['simulated_steps'] += count
                        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                            raise FloatingPointError('Nonfinite disturbance rollout')
                        trans, rot = displacement(target_obj_pose, env.get_active_obj_pose())
                        if trans >= settings.translation_limit_m or rot >= settings.rotation_limit_deg:
                            row['failure'] = 'translation_limit' if trans >= settings.translation_limit_m else 'rotation_limit'
                            break
                    row['passed'] = row['failure'] is None
                row.update(translation_m=trans, rotation_deg=rot,
                           final_object_pose=env.get_active_obj_pose().tolist(),
                           final_qpos=data.qpos.copy().tolist())
            except Exception as error:
                row['failure'] = type(error).__name__ + ': ' + str(error)
            finally:
                world_wrench(model, data, body, np.zeros(3), np.zeros(3))
            row['wall_seconds'] = time.perf_counter()-trial_start
            rows.append(row)
    return dict(settings=asdict(settings),mass_kg=mass,radius_m=object_radius,
                force_magnitude_n=force_n,torque_magnitude_nm=torque_nm,
                force_passed=all(r['passed'] for r in rows if r['kind']=='force'),
                torque_passed=all(r['passed'] for r in rows if r['kind']=='torque'),
                combined_passed=all(r['passed'] for r in rows),trials=rows,
                wall_seconds=time.perf_counter()-start,
                criterion='Extended wrench stability; not the unmodified upstream GSR')
