"""Lazy adapter for the pinned, unmodified DexForge ForceAware optimizer.

The production workflow uses ``ServoPriorOptimizer``. Its complete algorithm is
inherited; only the objective factory supplies floating-base humanoid losses.
Importing settings requires no numerical package or upstream module.
"""

from dataclasses import dataclass, fields
import math
from pathlib import Path


@dataclass(frozen=True)
class DexForgeSettings:
    horizon: int = 5
    n_iter: int = 40
    lr_max: float = 0.0015
    lr_min: float = 1e-5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    grad_clip: float = 50.0
    warm_shift_scale: float = 1.0
    multistart_count: int = 64
    multistart_seed: int = 0
    multistart_init_std: float = 0.03
    multistart_ctrl_std: float = 0.003
    multistart_init_raw_limit: float = 0.5
    mpc_dt: float = 0.01
    action_dt: float = 0.03
    knot_dt: float = 0.06
    exec_dt: float = 0.01
    init_joint_bound: float = 0.24
    root_pos_weight: float = 10.0
    root_rot_weight: float = 3.0
    joint_ref_weight: float = 0.03
    obj_pos_weight: float = 1.0
    obj_rot_weight: float = 0.3
    reference_pose_scale: float = 100.0
    terminal_weight: float = 2.0
    init_weight: float = 0.5
    soft_penetration_weight: float = 15.0
    penetration_weight: float = 75.0
    control_velocity_weight: float = 1.0
    control_acceleration_weight: float = 1.0
    joint_state_weight: float = 1.0
    control_velocity_scale: float = 1.0
    control_acceleration_scale: float = 500.0
    state_velocity_scale: float = 4.0
    state_acceleration_scale: float = 100.0
    joint_component_ratio: float = 0.3
    soft_penetration_limit: float = 0.004
    penetration_limit: float = 0.008
    contact_topk: int = 1

    def __post_init__(self):
        integer = {"horizon", "n_iter", "multistart_count", "multistart_seed", "contact_topk"}
        for name in integer:
            value = getattr(self, name)
            minimum = 0 if name == "multistart_seed" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"dexforge.{name} must be an integer >= {minimum}")
        positive = {"lr_max", "lr_min", "adam_epsilon", "grad_clip", "mpc_dt", "action_dt",
                    "knot_dt", "exec_dt", "init_joint_bound", "control_velocity_scale",
                    "control_acceleration_scale", "state_velocity_scale", "state_acceleration_scale"}
        for field in fields(self):
            if field.name in integer:
                continue
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"dexforge.{field.name} must be finite and nonnegative")
            if field.name in positive and value == 0:
                raise ValueError(f"dexforge.{field.name} must be positive")
        if not 0 < self.adam_beta1 < 1 or not 0 < self.adam_beta2 < 1:
            raise ValueError("dexforge Adam beta values must lie in (0, 1)")
        if self.lr_min > self.lr_max or self.contact_topk > 8:
            raise ValueError("Invalid DexForge learning-rate schedule or contact_topk")
        if self.soft_penetration_limit > self.penetration_limit:
            raise ValueError("dexforge.penetration_limit must be >= soft_penetration_limit")
        for outer, inner in ((self.action_dt, self.mpc_dt), (self.knot_dt, self.mpc_dt), (self.mpc_dt, self.exec_dt)):
            ratio = round(outer / inner)
            if ratio < 1 or not math.isclose(ratio * inner, outer, rel_tol=0, abs_tol=1e-9):
                raise ValueError("DexForge action, knot and execution timesteps must form integer grids")
        if self.action_dt > self.knot_dt:
            raise ValueError("dexforge.action_dt must not exceed knot_dt")


def make_config(settings, *, xml, ref_dt, device="cuda:0"):
    """Build official dataclasses in memory; no CA files or IK are loaded."""
    from retarget.dynamics.dexforge_upstream import activate_upstream

    root = activate_upstream()
    from forceaware import config as fa
    from forceaware.time_grid import TimeGrid

    if not isinstance(settings, DexForgeSettings):
        raise TypeError("make_config requires DexForgeSettings")
    # Device is selected by the native model loader, outside RetargetConfig.
    del device
    xml = Path(xml).resolve()
    grid = TimeGrid.create(mpc_dt=settings.mpc_dt, action_dt=settings.action_dt,
                           knot_dt=settings.knot_dt, exec_dt=settings.exec_dt, ref_dt=ref_dt)
    default = fa.load_config()
    loss = dict(default["loss"])
    loss.update(obj_pos=settings.obj_pos_weight, obj_rot=settings.obj_rot_weight,
                terminal=settings.terminal_weight, reference_pose_scale=settings.reference_pose_scale,
                init=settings.init_weight, action_ref_base=settings.joint_ref_weight,
                action_ref_base_rot=settings.joint_ref_weight, action_ref_finger=settings.joint_ref_weight,
                contact_robustness=0.0, marked_query_pos=0.0, finger_joint_collision=0.0,
                soft_penetration=settings.soft_penetration_weight, penetration=settings.penetration_weight)
    smoothing = dict(default["smoothing"])
    smoothing.update(control_velocity_weight=settings.control_velocity_weight,
                     control_acceleration_weight=settings.control_acceleration_weight,
                     hand_state_weight=settings.joint_state_weight)
    for group in ("base_position", "base_rotation", "finger"):
        smoothing[f"control_velocity_{group}_scale"] = settings.control_velocity_scale
        smoothing[f"control_acceleration_{group}_scale"] = settings.control_acceleration_scale
    optimizer = dict(default["optimizer"])
    for name in optimizer:
        if name != "contact_dynamics":
            optimizer[name] = getattr(settings, name)
    contact = dict(default["contact"])
    contact.update(target_phi=-settings.soft_penetration_limit, penetration_limit=settings.penetration_limit)
    cfg = fa.RetargetConfig(
        loss=fa.LossWeights(**loss), smoothing=fa.SmoothingConfig(**smoothing),
        bounds=fa.BoundsConfig(settings.init_joint_bound, settings.init_joint_bound, settings.init_joint_bound),
        optimizer=fa.OptimizerConfig(**optimizer), contact=fa.ContactConfig(**contact),
        simulator=fa.SimConfig(grid, "comfree", root / "forceaware/third_party/comfree_warp", 0.0, settings.contact_topk),
        sequence=fa.SeqConfig(xml.parent, "humanoid", xml, xml.parent / "unused_contact_guidance.npz", 0, 1, -1, 0.0, ""),
        video=fa.VideoConfig(float(default["video"]["fps"])),
    )
    return fa.validate_retarget_config(cfg)


def create_optimizer(cpu_model, device_model, index, *, cfg, settings=None,
                     root_pos_weight=None, root_rot_weight=None,
                     penetration_output_indices=None, guidance_count=4):
    """Construct the official production solver with an objective-only hook."""
    from retarget.dynamics.dexforge_upstream import activate_upstream, require_environment

    require_environment()
    activate_upstream()
    from retarget.dynamics.dexforge_loss import create_optimizer as create

    settings = DexForgeSettings() if settings is None else settings
    return create(cpu_model, device_model, index, cfg=cfg, settings=settings,
                  root_pos_weight=settings.root_pos_weight if root_pos_weight is None else root_pos_weight,
                  root_rot_weight=settings.root_rot_weight if root_rot_weight is None else root_rot_weight,
                  penetration_output_indices=penetration_output_indices, guidance_count=guidance_count)


@dataclass
class DexForgeProblem:
    optimizer: object
    qpos: object
    qvel: object
    targets: object
    warm: object
    previous_action: object
    allow_init: bool = False
    execution_steps: int | None = None
    start_time: float = 0.0


@dataclass
class DexForgeResult:
    parameters: object
    cost: float
    history: list
    sampled: int
    evaluated: int
    plan: object


def solve(problem, settings, seed=0):
    """Optional shared-solver dispatch; the humanoid runner can use Plan directly."""
    if not isinstance(problem, DexForgeProblem):
        raise TypeError("dexforge requires a DexForgeProblem wrapping the official optimizer")
    settings = settings.dexforge if hasattr(settings, "dexforge") else settings
    if not isinstance(settings, DexForgeSettings):
        raise TypeError("dexforge requires DexForgeSettings")
    import torch

    count = int(round(settings.action_dt / settings.mpc_dt)) if problem.execution_steps is None else problem.execution_steps
    plan = problem.optimizer.solve(
        problem.qpos, problem.qvel, problem.targets, warm=problem.warm,
        previous_action=problem.previous_action, allow_init=problem.allow_init,
        seed=seed, execution_steps=count, start_time=problem.start_time,
    )
    return DexForgeResult(torch.as_tensor(plan.raw).detach().clone(), float(plan.loss),
                         [dict(loss=float(plan.loss), grad_norm=float(plan.grad_norm), **plan.diag)],
                         settings.multistart_count, settings.multistart_count * (settings.n_iter + 1), plan)
