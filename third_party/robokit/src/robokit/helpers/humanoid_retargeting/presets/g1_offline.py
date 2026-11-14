"""G1 offline optimizer preset paired with `g1_offline_mapping`."""

import numpy as np

from robokit.helpers.humanoid_retargeting.config import (
    HumanoidContext,
    HumanoidRetargeterOfflineConfig,
)
from robokit.helpers.humanoid_retargeting.presets.g1_offline_mapping import g1_offline_mapping
from robokit.opt.lm_optimizer import LMOptimizerConfig
from robokit.terms.dense.configuration_task import ConfigurationTask
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.rotation_task import RotationTask
from robokit.terms.sparse.trajectory_limit_task import TrajectoryLimitTask
from robokit.terms.sparse.trajectory_position_task import TrajectoryPositionTask
from robokit.terms.sparse.trajectory_self_collision_task import TrajectorySelfCollisionTask
from robokit.terms.sparse.trajectory_smoothness_task import TrajectorySmoothnessTask
from robokit.terms.trajectory_task import TrajectoryTask


g1_offline = HumanoidRetargeterOfflineConfig(
    solver=LMOptimizerConfig(
        linear_solver="cg",
        lm_lambda=0.5,
        max_iter=100,
        cuda_graph_mode="full",
        lambda_factor=2.0,
        lambda_min=1e-06,
        lambda_max=1000000.0,
        rho_min=0.0001,
    ),
    terms=[
        # track the mapping's links with its target weights
        TrajectoryPositionTask(
            frame_index=HumanoidContext.TARGET_LINKS,
            target_positions=HumanoidContext.TARGET_POSITIONS,
            weight=dict(g1_offline_mapping.task("target").position_weight),
            name="positions",
        ),
        TrajectoryTask(
            RotationTask(
                frame_index=HumanoidContext.TARGET_LINKS,
                T_world_target=HumanoidContext.TARGETS,
                weight=dict(g1_offline_mapping.task("target").orientation_weight),
            ),
            name="orientations",
        ),
        TrajectorySmoothnessTask(
            weight=[
                25.0,
                20.0,
                15.0,
                45.0,
                10.0,
                10.0,
                25.0,
                20.0,
                15.0,
                45.0,
                10.0,
                10.0,
                100.0,
                100.0,
                100.0,
                35.0,
                25.0,
                10.0,
                13.0,
                5.0,
                5.0,
                5.0,
                35.0,
                25.0,
                10.0,
                13.0,
                5.0,
                5.0,
                5.0,
            ],
            base_weight=np.array([10.0] * 3 + [5.0] * 3),
            name="smoothness",
        ),
        TrajectoryTask(PositionLimit(weight=10.0), name="position_limit"),
        TrajectoryTask(
            ConfigurationTask(
                rest_q="midrange",
                weight=[
                    0.5,
                    0.5,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.5,
                    0.5,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    15.0,
                    5.0,
                    0.5,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    15.0,
                    5.0,
                    0.5,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            ),
            name="rest",
        ),
        TrajectoryLimitTask(
            order=1,
            upper=HumanoidContext.VELOCITY_LIMITS,
            dt=HumanoidContext.STEP,
            stencil="forward",
            weight=0.2,
            name="velocity_limit",
        ),
        TrajectorySelfCollisionTask(weight=0.0, margin=0.01, name="self_collision"),
    ],
)
