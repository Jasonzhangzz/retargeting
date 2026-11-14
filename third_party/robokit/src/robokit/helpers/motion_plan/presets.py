"""Motion planner presets; `dataclasses.replace` them per run and `deepcopy` before mutating shared nested configs.

Presets:
    - `offline_pose_lbfgs` runs goal IK, trajectory optimization, and fixed-endpoint finetuning with four candidates.
    - `offline_pose_lm` runs the same stages with LM.
    - `offline_joint_lbfgs` plans to joint-space goals.
    - `online_pose_lbfgs` and `online_pose_lm` solve fresh trajectories with executed-history smoothing.
    - `online_pose_mppi` uses one planner with caller-owned MPPI state.
    - `offline_pose_gd`, `offline_pose_adam` and `offline_pose_adamw` descend with first-order optimizers.
"""

import dataclasses
import math
from copy import deepcopy
from typing import Literal, Optional

import numpy as np

from robokit.helpers.ik import IKContext, IKSolverConfig
from robokit.helpers.motion_plan.gradient_trajectory_optimizer import (
    GradientTrajectoryOptimizerConfig,
    MotionPlanContext,
)
from robokit.helpers.motion_plan.motion_plan_score_task import MotionPlanScoreTask
from robokit.helpers.motion_plan.motion_planner import MotionPlannerConfig, build_goal_score
from robokit.helpers.motion_plan.mppi_trajectory_optimizer import (
    ColoredGaussianSamplerConfig,
    MppiTrajectoryOptimizerConfig,
)
from robokit.helpers.motion_plan.trajectory_retimer import RetimeConfig
from robokit.opt.gd_optimizer import GDOptimizerConfig
from robokit.opt.lbfgs_optimizer import LBFGSOptimizerConfig
from robokit.opt.lm_optimizer import LMOptimizerConfig
from robokit.opt.population_solver import PopulationSolverConfig, Stage
from robokit.terms.dense.com_position_task import ComPositionTask
from robokit.terms.dense.configuration_task import ConfigurationTask
from robokit.terms.dense.frame_task import FrameTask
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.position_task import PositionTask
from robokit.terms.dense.rotation_task import RotationTask
from robokit.terms.dense.scene_collision_task import SceneCollisionTask
from robokit.terms.dense.self_collision_task import SelfCollisionTask
from robokit.terms.dense.smoothness_task import SmoothnessTask
from robokit.terms.sparse.trajectory_collision_task import TrajectoryCollisionTask
from robokit.terms.sparse.trajectory_limit_task import TrajectoryLimitTask
from robokit.terms.sparse.trajectory_position_task import TrajectoryPositionTask
from robokit.terms.sparse.trajectory_self_collision_task import TrajectorySelfCollisionTask
from robokit.terms.sparse.trajectory_smoothness_task import TrajectorySmoothnessTask
from robokit.terms.trajectory_task import TrajectoryTask


ctx = MotionPlanContext  # the stage values that templates read
_derivative_limits = (ctx.VELOCITY_LIMITS, ctx.ACCELERATION_LIMIT, ctx.JERK_LIMIT)
# trajectory updates: LM damping bounds and an L-BFGS line search suited to trajectories
_lm = LMOptimizerConfig(lambda_min=1e-6, lambda_max=1e6, rho_min=1e-4)
_lbfgs = LBFGSOptimizerConfig(
    line_search_alphas=(0.001, 0.01, 0.05, 0.1), history_len=15, fold_line_search_gradient=True
)


# --- builders --------------------------------------------------------------
def build_offline_ik_config(
    num_candidates: int = 4,
    cuda_graph_mode: Literal["none", "full", "iter"] = "full",
    num_goal_candidates: Optional[int] = None,
    num_samples: int = 64,
    num_refine_samples: Optional[int] = None,
    collision_margin: float = 0.025,
    collision_weight: float = 20.0,
) -> IKSolverConfig:
    """Build the offline goal-IK multi-candidate funnel.

    Args:
        num_candidates: Default number of final IK goals.
        cuda_graph_mode: CUDA graph capture mode.
        num_goal_candidates: Number of final joint goals returned per query. Defaults to `num_candidates`.
        num_samples: Number of IK seeds sampled before the funnel narrows.
        num_refine_samples: Seeds retained for refinement. Defaults to `num_samples`.
        collision_margin: Clearance goal IK keeps from every scene geometry (goal IK has no self-collision term).
        collision_weight: Weight of that clearance: a cost, not a constraint, so answers can still end inside.

    Returns:
        A `samples -> refine -> goal` IK configuration.

    Examples:
        >>> config = build_offline_ik_config(num_samples=128, num_refine_samples=64)
        >>> config.solver.num_candidates
        [128, 64, 4]
    """
    num_goal_candidates = num_candidates if num_goal_candidates is None else num_goal_candidates
    if num_goal_candidates > num_candidates:
        raise ValueError("num_goal_candidates must not exceed num_candidates.")
    num_refine_samples = num_samples if num_refine_samples is None else num_refine_samples
    if num_refine_samples > num_samples:
        raise ValueError("num_refine_samples must not exceed num_samples.")
    return (
        IKSolverConfig(
            solver=PopulationSolverConfig(
                stages=[
                    Stage(4, max(num_samples, num_candidates * 2), LMOptimizerConfig(lm_lambda=10.0)),
                    Stage(12, max(num_refine_samples, num_candidates * 2), LMOptimizerConfig(lm_lambda=1.0)),
                    Stage(0, num_goal_candidates, LMOptimizerConfig(lm_lambda=1.0)),
                ],
                cuda_graph_mode=cuda_graph_mode,
            )
        )
        .add(PositionTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=20.0))
        .add(RotationTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=40.0))
        .add(PositionLimit(weight=50.0))
        .add(ConfigurationTask(rest_q=IKContext.REST_Q, weight=0.05))
        .add(
            SceneCollisionTask(
                scene=IKContext.SCENE,
                scene_indices=IKContext.SCENE_INDICES,
                weight=collision_weight,
                margin=collision_margin,
            ),
            name="ik_collision",
        )
        .add_score(build_goal_score(), name="goal_score")
    )


# --- offline pose-goal planning --------------------------------------------
def build_offline_pose_goal_config(
    num_timesteps: int = 30,
    dt: float = 0.25,
    max_iter: int = 100,
    num_candidates: int = 4,
    max_retries: int = 20,
    virtual_hold_frames: int = 2,
    jerk_limit: float = 150.0,
    use_cuda_graph: bool = True,
    num_ik_samples: int = 128,
) -> MotionPlannerConfig:
    """Build the configuration used by `offline_pose_lbfgs`, with explicit overrides.

    Stages:
    1. Solve goal IK.
    2. Balance goal error against joint/derivative limits and self/world collision; retime to shorten the step.
    3. Finetune every candidate with fixed endpoints and a higher collision weight on the shorter step.

    Args:
        num_timesteps: Waypoints per plan.
        dt: Step the trajectory stage plans at, before retiming.
        max_iter: Trajectory-stage iterations.
        num_candidates: Trajectories optimized in parallel.
        max_retries: Plans attempted before giving up.
        virtual_hold_frames: Still frames held at each end, so the plan ramps from and to rest.
        jerk_limit: Joint jerk limit of the plan and its retimes.
        use_cuda_graph: Capture the whole solve, retries included, in one graph; otherwise run eagerly.
        num_ik_samples: Initial goal-IK samples; retries may reserve a larger goal pool.

    Returns:
        A `MotionPlannerConfig` with the trajectory and finetune stages.

    Examples:
        >>> config = build_offline_pose_goal_config()
        >>> [stage.solver.num_candidates for stage in config.stages]
        [[4], [4]]
        >>> [stage.solver.num_iters for stage in config.stages]
        [100, 180]
    """
    trajopt = GradientTrajectoryOptimizerConfig(
        # the derivative bounds cover smoothness: no plain smoothness, rest, start or goal pulls
        terms=[
            # first, so its term streams fork first: the swept cost runs longest and otherwise waits for SMs
            TrajectoryCollisionTask(
                scene=ctx.SCENE,
                scene_indices=ctx.SCENE_INDICES,
                weight=500.0,
                margin=0.01,
                sweep_steps=4,
                penalty="smooth",  # quadratic inside the activation distance, then linear
                speed_dt=ctx.QUERY_STEP,  # scale by sphere speed
                name="collision",
            ),
            TrajectoryTask(
                FrameTask(
                    frame_index=ctx.TARGET_LINKS,
                    T_world_target=ctx.TARGET,
                    position_weight=11000.0,
                    orientation_weight=950.0,
                ),
                frame_indices=ctx.GOAL_FRAMES,
                name="target",
            ),
            TrajectoryTask(PositionLimit(weight=300.0, activation=0.05), name="position_limit"),
            TrajectoryLimitTask(
                order=(1, 2, 3),
                upper=_derivative_limits,
                weight=(300.0, 100.0, 3.0),
                dt=ctx.QUERY_STEP,
                activation=(0.0, 0.05, 0.05, 0.05),
                name="derivative_bounds",
            ),
            # the acceleration and jerk penalties fold dt^2 and dt^3 at the trajectory step
            TrajectoryLimitTask(
                order=(2, 3),
                lower=0.0,
                upper=0.0,
                weight=(300.0, 5.0),
                dt=ctx.QUERY_STEP,
                dt_power=(1.0, 1.5),
                name="derivative_smoothness",
            ),
            # the robot asset's link buffers inflate its spheres, as they do the score's check
            TrajectorySelfCollisionTask(weight=5000.0, margin=0.0, penalty="max_linear", name="self_collision"),
        ],
        solver=PopulationSolverConfig(
            stages=[
                Stage(
                    max_iter,
                    num_candidates,
                    dataclasses.replace(
                        _lbfgs,
                        line_search="approx_wolfe",
                        line_search_alphas=(0.3, 1.0),  # two trials cost less per iteration and plan as well as four
                        h0_scale=0.01,
                        wolfe_c1=1e-5,
                    ),
                )
            ]
        ),
        score_terms=[
            MotionPlanScoreTask(
                rotation_tolerance=2.0 * math.asin(0.05),  # the judge accepts sin(angle / 2) <= 0.05
                joint_limit_tolerance=0.2,  # as the judge allows
                interpolation_dt=0.025,  # the judge samples the plan at this step, so the score checks it there too
                self_collision=True,
                rank_retimed_path=True,  # rank by the retimed path length, which is what the plan costs to run
                keep_candidate_zero=False,
            )
        ],
        clamp_to_limits=True,  # every line-search trial is projected into the joint limits
        lock_frames=(0,),  # the plan starts exactly at the given configuration instead of being pulled towards it
    )
    finetune = dataclasses.replace(
        trajopt,
        solver=dataclasses.replace(trajopt.solver, stages=[dataclasses.replace(trajopt.solver.stages[0], iters=180)]),
        lock_frames=(0, -1),  # holds the terminal waypoint the trajectory stage reached, besides the start
        terms=[
            # first, so its term streams fork first: the swept cost runs longest and otherwise waits for SMs
            TrajectoryCollisionTask(
                scene=ctx.SCENE,
                scene_indices=ctx.SCENE_INDICES,
                weight=1500.0,
                margin=0.01,
                sweep_steps=4,
                penalty="smooth",  # quadratic inside the activation distance, then linear
                speed_dt=ctx.QUERY_STEP,  # scale by sphere speed
                name="collision",
            ),
            TrajectoryTask(
                FrameTask(
                    frame_index=ctx.TARGET_LINKS,
                    T_world_target=ctx.TARGET,
                    position_weight=11000.0,
                    orientation_weight=950.0,
                ),
                frame_indices=ctx.GOAL_FRAMES,
                name="target",
            ),
            TrajectoryTask(PositionLimit(weight=300.0, activation=0.05), name="position_limit"),
            TrajectoryLimitTask(
                order=(1, 2, 3),
                upper=_derivative_limits,
                weight=(300.0, 100.0, 3.0),
                dt=ctx.QUERY_STEP,
                activation=(0.0, 0.001, 0.001, 0.001),
                name="derivative_bounds",
            ),
            # the acceleration and jerk penalties fold dt^2 and dt^3 at the trajectory step
            TrajectoryLimitTask(
                order=(2, 3),
                lower=0.0,
                upper=0.0,
                weight=(300.0, 5.0),
                dt=ctx.QUERY_STEP,
                dt_power=(1.0, 1.5),
                name="derivative_smoothness",
            ),
            # the robot asset's link buffers inflate its spheres, as they do the score's check
            TrajectorySelfCollisionTask(weight=5000.0, margin=0.0, penalty="max_linear", name="self_collision"),
        ],
    )
    # goal IK keeps its own margin, wider than the trajectory objective's activation
    ik_config = build_offline_ik_config(
        num_candidates, "full" if use_cuda_graph else "none", num_samples=num_ik_samples
    )
    return MotionPlannerConfig(
        num_timesteps=num_timesteps,
        dt=dt,
        cuda_graph="solve" if use_cuda_graph else "none",
        stages=(trajopt, finetune),
        acceleration_limit=15.0,  # the judge's limit; planning to it keeps the retimed plan feasible
        jerk_limit=jerk_limit,
        # the finetune stage starts from this retime of the fastest feasible candidate, at the plan's limits
        stage_retime=RetimeConfig(speed_scale=1.0, minimum_dt=min(0.02, dt), virtual_hold_frames=virtual_hold_frames),
        retime=RetimeConfig(speed_scale=1.0, max_dt=1.0, virtual_hold_frames=virtual_hold_frames),
        stage_dt_scale=0.8,  # the finetune runs at 0.8 times the fastest feasible candidate's retimed step
        stage_retries=2,  # a failed finetune retries with the step grown by 1.025 each time
        stage_retry_iters=100,
        max_retries=max_retries,
        goal_ik=ik_config,
    )


offline_pose_lbfgs = build_offline_pose_goal_config()
offline_pose_lm = dataclasses.replace(
    offline_pose_lbfgs,
    cuda_graph="stages",
    max_retries=0,
    stage_retry_iters=50,
    stages=tuple(
        dataclasses.replace(
            stage,
            solver=dataclasses.replace(
                stage.solver,
                stages=[
                    dataclasses.replace(
                        stage.solver.stages[0], iters=50, optimizer=dataclasses.replace(_lm, lm_lambda=0.01)
                    )
                ],
            ),
            terms=deepcopy(stage.terms),
        )
        for stage in offline_pose_lbfgs.stages
    ),
)


# --- joint-space planning --------------------------------------------------
# Joint-space planning pins both endpoints and keeps the trajectory objective without its pose residual.
offline_joint_lbfgs = dataclasses.replace(
    offline_pose_lbfgs,
    stages=(
        dataclasses.replace(
            offline_pose_lbfgs.stages[0],
            terms=[term for term in deepcopy(offline_pose_lbfgs.stages[0].terms) if term.name != "target"],
            lock_frames=(0, -1),
        ),
    ),
)


# --- online gradient planning ----------------------------------------------
# Online L-BFGS replans fresh collision-aware trajectories from the current state.
online_pose_lbfgs = MotionPlannerConfig(
    num_timesteps=15,
    dt=0.1,
    max_retries=0,
    stages=(
        GradientTrajectoryOptimizerConfig(
            terms=[
                # first, so its term streams fork first: the swept cost runs longest and otherwise waits for SMs
                TrajectoryCollisionTask(
                    scene=ctx.SCENE,
                    scene_indices=ctx.SCENE_INDICES,
                    weight=1000.0,
                    margin=0.025,
                    goal_radius=3.0,
                    goal_distance=ctx.GOAL_DISTANCE,
                    name="collision",
                ),
                TrajectoryTask(ConfigurationTask(rest_q=ctx.START_Q, weight=2000.0), frame_indices=[0], name="start"),
                TrajectorySmoothnessTask(weight=200.0, smooth_boundary=False, name="smoothness"),
                TrajectorySmoothnessTask(
                    weight=0.25, order=3, dt=ctx.STEP, history=ctx.HISTORY, name="jerk_smoothness"
                ),
                TrajectorySmoothnessTask(
                    weight=3.0, order=2, dt=ctx.STEP, history=ctx.HISTORY, name="acceleration_smoothness"
                ),
                TrajectoryLimitTask(
                    order=1,
                    upper=ctx.VELOCITY_LIMITS,
                    weight=5.0,
                    dt=ctx.STEP,
                    stencil="forward",
                    name="velocity_limit",
                ),
                TrajectoryTask(PositionLimit(weight=100.0), name="position_limit"),
                TrajectoryTask(ConfigurationTask(rest_q="midrange", weight=0.01), name="rest"),
                TrajectoryTask(ConfigurationTask(rest_q=ctx.GOAL_Q, weight=5000.0), frame_indices=[-1], name="goal"),
            ],
            score_terms=[MotionPlanScoreTask(keep_candidate_zero=False)],
            solver=PopulationSolverConfig(
                stages=[
                    Stage(
                        10,
                        16,  # eight candidates lose success on the full reactive benchmark
                        dataclasses.replace(_lbfgs, normalize_direction=True),  # ten iterations: steps are lengths
                    )
                ]
            ),
            lock_frames=(0,),
        ),
    ),
    goal_ik=build_offline_ik_config(
        num_goal_candidates=1, num_samples=128, num_refine_samples=64, collision_margin=0.001, collision_weight=500.0
    ),
)

# Online LM solves the same fresh planning problem with sparse Gauss-Newton steps.
online_pose_lm = dataclasses.replace(
    online_pose_lbfgs,
    stages=(
        dataclasses.replace(
            online_pose_lbfgs.stages[0],
            solver=PopulationSolverConfig(
                stages=[Stage(5, 8, dataclasses.replace(_lm, lm_lambda=0.01, linear_solver="cg", cg_max_iter=4))]
            ),
            terms=deepcopy(online_pose_lbfgs.stages[0].terms),
        ),
    ),
)


# --- online MPPI planning --------------------------------------------------
# Reactive acceleration MPPI uses caller-owned state and live IK goals without a separate discovery planner.
online_pose_mppi = MotionPlannerConfig(
    num_timesteps=31,  # -> num_frames = 32 (the MPPI horizon)
    dt=0.03,  # control step + rollout base dt
    max_retries=0,
    # 128 seeds lose full reactive-benchmark success
    goal_ik=build_offline_ik_config(num_goal_candidates=1, collision_margin=0.005),
    stages=(
        MppiTrajectoryOptimizerConfig(
            terms=[
                # first, so its term streams fork first: the swept cost runs longest and otherwise waits for SMs
                TrajectoryCollisionTask(
                    scene=ctx.SCENE,
                    scene_indices=ctx.SCENE_INDICES,
                    weight=1500.0,
                    margin=0.005,
                    sweep_steps=4,
                    penalty="smooth",
                    name="collision",
                ),
                TrajectoryTask(
                    FrameTask(
                        frame_index=ctx.TARGET_LINKS,
                        T_world_target=ctx.TARGET,
                        position_weight=600.0,
                        orientation_weight=150.0,
                    ),
                    frame_indices=[-1],
                    name="target",
                ),
                TrajectorySmoothnessTask(weight=1.0, order=3, dt=ctx.STEP, name="jerk_smoothness"),
                TrajectorySmoothnessTask(weight=8.0, order=2, dt=ctx.STEP, name="acceleration_smoothness"),
                TrajectoryTask(PositionLimit(weight=300.0), name="position_limit"),
                TrajectoryTask(ConfigurationTask(rest_q=ctx.GOAL_Q, weight=40.0), name="goal_rest"),
                TrajectoryPositionTask(
                    frame_index=ctx.EE_LINK, target_positions=ctx.LINK_TARGETS, weight=300.0, name="link_position"
                ),
            ],
            solver=PopulationSolverConfig(
                stages=[Stage(5)],  # MPPI updates per tick
                samplers=[
                    ColoredGaussianSamplerConfig(
                        num_draws=384, temperature=0.2, init_std=1.5, min_std=0.005, knot_ratio=0.7
                    )
                ],
            ),
            settle_radius=0.05,
            control_space="acceleration",
            dt_max=0.05,
            q_limit_margin=0.1,
        ),
    ),
)


# --- first-order offline planning ------------------------------------------
# The LM preset's stages with both endpoints fixed, descending with cosine-annealed first-order steps.
offline_pose_gd, offline_pose_adam, offline_pose_adamw = (
    dataclasses.replace(
        offline_pose_lm,
        stages=tuple(
            dataclasses.replace(
                stage,
                lock_frames=(0, -1),
                terms=deepcopy(stage.terms),
                solver=dataclasses.replace(
                    stage.solver,
                    stages=[
                        dataclasses.replace(
                            stage.solver.stages[0],
                            optimizer=GDOptimizerConfig(
                                learning_rate=learning_rate,
                                # cosine annealing to `low` over the stage's 50 iterations
                                lr_schedule=lambda it, lr, low=low: (
                                    low + 0.5 * (lr - low) * (1.0 + math.cos(math.pi * min(it, 50) / 50))
                                ),
                                optimizer_type="adam" if adam else "gd",
                                weight_decay=weight_decay,
                                early_stopping_interval=0,
                            ),
                        )
                    ],
                ),
            )
            for stage in offline_pose_lm.stages
        ),
    )
    # the shared derivative residuals have large gradients: plain GD needs far smaller steps than Adam
    for adam, learning_rate, low, weight_decay in (
        (False, 1e-12, 1e-13, 0.0),
        (True, 0.02, 3e-3, 0.0),
        (True, 0.02, 3e-3, 1e-4),  # AdamW: Adam with decoupled weight decay
    )
)


# --- humanoids: floating-base goal IK and planted feet ---
# Complete foot, balance, and posture stubs from the standing robot; collision margins may be zeroed on feet.
def _humanoid_ik_config() -> IKSolverConfig:
    return IKSolverConfig(
        solver=PopulationSolverConfig(
            stages=[
                Stage(8, 128, LMOptimizerConfig(lm_lambda=10.0)),
                Stage(16, 64, LMOptimizerConfig(lm_lambda=1.0)),
                Stage(0, 1, LMOptimizerConfig(lm_lambda=1.0)),
            ],
            cuda_graph_mode="full",
        ),
        enable_T_world_base=True,
    ).add_score(build_goal_score(), name="goal_score")


humanoid_offline_lbfgs = MotionPlannerConfig(
    num_timesteps=12,
    dt=0.12,
    max_retries=0,
    jerk_limit=100.0,
    retime=RetimeConfig(speed_scale=1.0, max_dt=1.0, virtual_hold_frames=2),  # rest to rest
    goal_ik=_humanoid_ik_config()
    .add(PositionTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=20.0))
    .add(RotationTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=40.0))
    .add(PositionLimit(weight=50.0))
    .add(
        ConfigurationTask(
            rest_q=IKContext.REST_Q,
            T_world_base_rest=IKContext.REST_BASE,
            weight=0.05,
            base_weight=[0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
        )
    )  # upright pelvis: squat, not fold
    .add(SelfCollisionTask(weight=20.0, margin=0.005))
    # the goal keeps the feet planted, or the locked final waypoint snaps them to wherever IK left them
    .add(PositionTask(weight=100.0, fixed_target=True), name="feet_position")
    .add(RotationTask(weight=50.0, fixed_target=True), name="feet_rotation")
    .add(ComPositionTask(weight=[20.0, 20.0, 0.0]), name="balance")  # the goal keeps its CoM over the feet
    .add(
        SceneCollisionTask(scene=IKContext.SCENE, scene_indices=IKContext.SCENE_INDICES, weight=20.0, margin=0.001),
        name="ik_collision",
    ),
    stages=(
        GradientTrajectoryOptimizerConfig(
            solver=PopulationSolverConfig(
                stages=[
                    Stage(
                        120,
                        4,  # the goal IK solution, repeated with different initial noise
                        dataclasses.replace(
                            _lbfgs,
                            line_search_alphas=(1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 5e-1, 1.0),
                            history_len=10,
                            normalize_direction=True,  # the planted-feet weights dwarf the rest: steps are lengths
                        ),
                    )
                ]
            ),
            lock_frames=(0, -1),  # paths run from the start to the goal IK solution
            terms=[
                # first, so its term streams fork first: the swept cost runs longest and otherwise waits for SMs
                TrajectoryCollisionTask(
                    scene=ctx.SCENE,
                    scene_indices=ctx.SCENE_INDICES,
                    weight=100.0,
                    margin=0.03,
                    sweep_steps=4,
                    name="collision",
                ),
                TrajectoryLimitTask(
                    order=1,
                    upper=ctx.VELOCITY_LIMITS,
                    weight=10.0,
                    dt=ctx.STEP,
                    stencil="forward",
                    name="velocity_limit",
                ),
                TrajectorySmoothnessTask(weight=0.1, dt=ctx.STEP),
                # spreads the bend evenly over the path
                TrajectorySmoothnessTask(weight=3.0, order=2, dt=ctx.STEP, history=ctx.HISTORY),
                TrajectoryTask(PositionLimit(weight=100.0)),
                TrajectoryTask(SelfCollisionTask(weight=100.0, margin=0.005), name="self_collision"),
                TrajectoryTask(PositionTask(weight=2000.0, fixed_target=True), name="feet_position"),
                TrajectoryTask(RotationTask(weight=800.0, fixed_target=True), name="feet_rotation"),
                TrajectoryTask(ComPositionTask(weight=[100.0, 100.0, 0.0]), name="balance"),
                TrajectoryTask(ConfigurationTask(weight=0.05), name="posture"),
            ],
        ),
    ),
)

humanoid_online_lbfgs = MotionPlannerConfig(
    num_timesteps=15,
    dt=0.1,
    max_retries=0,
    init_sample_range=0.02,  # planted feet tolerate only small steps
    goal_ik=_humanoid_ik_config()
    .add(PositionTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=1.5))
    .add(RotationTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=0.3))
    .add(PositionLimit(weight=50.0))
    .add(SmoothnessTask(prev_var=IKContext.PREVIOUS_STATE, weight=0.03, base_weight=0.0))
    .add(SelfCollisionTask(weight=10.0, margin=0.005))
    .add(PositionTask(weight=5.0, fixed_target=True), name="feet_position")
    .add(RotationTask(weight=2.0, fixed_target=True), name="feet_rotation")
    .add(ComPositionTask(weight=[1.0, 1.0, 0.0]), name="balance")
    .add(
        ConfigurationTask(
            rest_q=IKContext.REST_Q,
            T_world_base_rest=IKContext.REST_BASE,
            weight=0.05,
            base_weight=[0.0, 0.0, 0.0, 100.0, 100.0, 0.0],
        ),
        name="posture",
    )
    .add(
        SceneCollisionTask(scene=IKContext.SCENE, scene_indices=IKContext.SCENE_INDICES, weight=500.0, margin=0.001),
        name="ik_collision",
    ),
    stages=(
        GradientTrajectoryOptimizerConfig(
            solver=PopulationSolverConfig(
                # the planted-feet weights dwarf the rest: steps are lengths
                stages=[Stage(10, 4, dataclasses.replace(_lbfgs, normalize_direction=True)), Stage(50, 1)],
            ),
            score_terms=[MotionPlanScoreTask(keep_candidate_zero=False)],
            lock_frames=(0,),
            terms=[
                # first, so its term streams fork first: the swept cost runs longest and otherwise waits for SMs
                TrajectoryCollisionTask(
                    scene=ctx.SCENE,
                    scene_indices=ctx.SCENE_INDICES,
                    weight=1e6,
                    margin=0.01,
                    sweep_steps=4,
                    name="collision",
                ),
                TrajectoryTask(ConfigurationTask(rest_q=ctx.START_Q, weight=2000.0), frame_indices=[0], name="start"),
                TrajectorySmoothnessTask(
                    weight=0.25,
                    order=3,
                    dt=ctx.STEP,
                    history=ctx.HISTORY,
                    name="jerk_smoothness",
                ),
                TrajectorySmoothnessTask(
                    weight=3.0,
                    order=2,
                    dt=ctx.STEP,
                    history=ctx.HISTORY,
                    name="acceleration_smoothness",
                ),
                TrajectoryLimitTask(
                    order=1,
                    upper=ctx.VELOCITY_LIMITS,
                    weight=5.0,
                    dt=ctx.STEP,
                    stencil="forward",
                    name="velocity_limit",
                ),
                TrajectoryTask(PositionLimit(weight=100.0), name="position_limit"),
                TrajectoryTask(ConfigurationTask(rest_q="midrange", weight=0.01), name="rest"),
                TrajectoryTask(ConfigurationTask(rest_q=ctx.GOAL_Q, weight=500.0), frame_indices=[-1], name="goal"),
                TrajectorySmoothnessTask(weight=200.0, smooth_boundary=False, base_weight=1000.0),  # smooth pelvis too
                TrajectoryTask(SelfCollisionTask(weight=1e4, margin=0.005), name="self_collision"),
                TrajectoryTask(PositionTask(weight=1e6, fixed_target=True), name="feet_position"),
                TrajectoryTask(RotationTask(weight=4e5, fixed_target=True), name="feet_rotation"),
                TrajectoryTask(ComPositionTask(weight=[1000.0, 1000.0, 0.0]), name="balance"),
                TrajectoryTask(
                    ConfigurationTask(weight=50.0, base_weight=[0.0, 0.0, 0.0, 1e5, 1e5, 0.0]), name="posture"
                ),
            ],
        ),
    ),
)


# Humanoid sampling controls:
# - Acceleration draws retain measured momentum and base twist; settled plans hold still.
# - A 0.5 rad/s soft cap and velocity cost brake momentum within 2 rad/s² draws.
# - The executed frame cannot enter the scene; colliding starts only back out.
# - Near goals, smoothness and velocity costs outweigh spread without a distance schedule.
humanoid_online_mppi = dataclasses.replace(
    humanoid_online_lbfgs,
    num_timesteps=9,
    dt=0.1,
    acceleration_limit=2.0,  # the rollouts' limit
    stages=(
        MppiTrajectoryOptimizerConfig(
            control_space="acceleration",
            solver=PopulationSolverConfig(
                stages=[Stage(50)],
                samplers=[
                    ColoredGaussianSamplerConfig(
                        num_draws=64, init_std=0.1, min_std=0.001, temperature=0.05, knot_ratio=0.9
                    )
                ],
            ),
            fresh_noise=True,
            terms=[
                # first, so its term streams fork first: the swept cost runs longest and otherwise waits for SMs
                TrajectoryCollisionTask(
                    scene=ctx.SCENE,
                    scene_indices=ctx.SCENE_INDICES,
                    weight=1e4,
                    margin=0.01,
                    sweep_steps=2,
                    penalty="smooth",
                    name="collision",
                ),
                TrajectoryCollisionTask(  # the executed frame, frame 1, never enters the scene
                    scene=ctx.SCENE,
                    scene_indices=ctx.SCENE_INDICES,
                    weight=1e6,
                    margin=0.0,
                    sweep_steps=2,
                    penalty="smooth",
                    frame_sphere_mask=np.eye(10, 1, -1, bool),
                ),
                TrajectoryLimitTask(
                    order=1,
                    upper=0.5,  # rad/s
                    weight=1e5,
                    dt=ctx.STEP,
                    stencil="forward",
                    name="velocity_limit",
                ),
                TrajectoryTask(PositionLimit(weight=1e6), name="position_limit"),
                TrajectoryTask(
                    FrameTask(
                        frame_index=ctx.TARGET_LINKS,
                        T_world_target=ctx.TARGET,
                        position_weight=800.0,
                        orientation_weight=200.0,
                    ),
                    frame_indices=list(range(1, 10)),
                    name="target",
                ),
                TrajectoryTask(SelfCollisionTask(weight=1e6, margin=0.003), name="self_collision"),
                TrajectoryTask(PositionTask(weight=2000.0, fixed_target=True), name="feet_position"),
                TrajectoryTask(RotationTask(weight=800.0, fixed_target=True), name="feet_rotation"),
                TrajectoryTask(ConfigurationTask(weight=0.05), name="posture"),
                TrajectorySmoothnessTask(weight=300.0),
                TrajectorySmoothnessTask(weight=0.01, order=2),
                TrajectorySmoothnessTask(weight=0.1, order=3),
            ],
        ),
    ),
)


__all__ = [
    "build_offline_ik_config",
    "build_offline_pose_goal_config",
    "humanoid_offline_lbfgs",
    "humanoid_online_lbfgs",
    "humanoid_online_mppi",
    "offline_pose_lm",
    "offline_pose_lbfgs",
    "offline_joint_lbfgs",
    "online_pose_lbfgs",
    "online_pose_lm",
    "online_pose_mppi",
    "offline_pose_gd",
    "offline_pose_adam",
    "offline_pose_adamw",
]
