"""Berkeley Humanoid Lite retargeting preset."""

import numpy as np

from robokit.assets.robots.humanoids import berkeley_humanoid_lite
from robokit.helpers.humanoid_retargeting.config import (
    HumanoidContext,
    HumanoidRetargeterOnlineConfig,
    LinkMapping,
)
from robokit.opt.lm_optimizer import LMOptimizerConfig
from robokit.opt.population_solver import PopulationSolverConfig, Stage
from robokit.terms.dense.configuration_task import ConfigurationTask
from robokit.terms.dense.frame_task import FrameTask
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.scene_collision_task import SceneCollisionTask
from robokit.terms.dense.self_collision_task import SelfCollisionTask
from robokit.terms.dense.smoothness_task import SmoothnessTask
from robokit.terms.dense.velocity_limit_task import VelocityLimitTask


bhl = HumanoidRetargeterOnlineConfig(
    urdf_path=str(berkeley_humanoid_lite.URDF_PATH),
    human_root_name="pelvis",
    human_height_assumption=1.66,
    velocity_limit_dt=0.03333333,
    velocity_clamp_scale=0.8,
    velocity_limit_override=[
        12.0,
        12.0,
        12.0,
        12.0,
        12.0,
        12.0,
        12.0,
        12.0,
        12.0,
        12.0,
        12.0,
        12.0,
        12.0,
        12.0,
        8.0,
        8.0,
        12.0,
        12.0,
        12.0,
        12.0,
        8.0,
        8.0,
    ],
    solver=PopulationSolverConfig(
        stages=[Stage(6, 4, LMOptimizerConfig(lm_lambda=1.5)), Stage(3, 1)],
        cuda_graph_mode="full",
    ),
    scale_table={
        "pelvis": 0.5,
        "left_hip": 0.5,
        "right_hip": 0.5,
        "left_knee": 0.5,
        "right_knee": 0.5,
        "left_foot": 0.5,
        "right_foot": 0.5,
        "left_elbow": 0.8,
        "right_elbow": 0.8,
        "left_wrist": 0.8,
        "right_wrist": 0.8,
    },
    link_mapping={
        "imu_2": LinkMapping(
            "pelvis", np.array([0.08, 0.0, 0.2], dtype=np.float32), np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float32)
        ),
        "leg_left_hip_roll": LinkMapping(
            "left_hip",
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.707, 0.0, 0.707], dtype=np.float32),
        ),
        "leg_left_knee_pitch": LinkMapping(
            "left_knee",
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.707, 0.0, -0.707, 0.0], dtype=np.float32),
        ),
        "leg_left_ankle_roll": LinkMapping(
            "left_foot", np.array([-0.05, 0.0, 0.0], dtype=np.float32), np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        ),
        "leg_right_hip_roll": LinkMapping(
            "right_hip",
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.707, 0.0, 0.707], dtype=np.float32),
        ),
        "leg_right_knee_pitch": LinkMapping(
            "right_knee",
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.707, 0.0, 0.707], dtype=np.float32),
        ),
        "leg_right_ankle_roll": LinkMapping(
            "right_foot", np.array([0.05, 0.0, 0.0], dtype=np.float32), np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        ),
        "arm_left_elbow_pitch": LinkMapping(
            "left_elbow", np.array([0.0, 0.0, 0.0], dtype=np.float32), np.array([0.5, 0.5, 0.5, -0.5], dtype=np.float32)
        ),
        "arm_left_hand_link": LinkMapping(
            "left_wrist",
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.707, 0.0, -0.707, 0.0], dtype=np.float32),
        ),
        "arm_right_elbow_pitch": LinkMapping(
            "right_elbow",
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.5, 0.5, -0.5, 0.5], dtype=np.float32),
        ),
        "arm_right_hand_link": LinkMapping(
            "right_wrist",
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.707, 0.0, 0.707, 0.0], dtype=np.float32),
        ),
    },
    terms=[
        FrameTask(
            frame_index=HumanoidContext.TARGET_LINKS,
            T_world_target=HumanoidContext.TARGETS,
            position_weight={
                "imu_2": 100,
                "leg_left_hip_roll": 1,
                "leg_left_knee_pitch": 0,
                "leg_left_ankle_roll": 100,
                "leg_right_hip_roll": 1,
                "leg_right_knee_pitch": 0,
                "leg_right_ankle_roll": 100,
                "arm_left_elbow_pitch": 30,
                "arm_left_hand_link": 50,
                "arm_right_elbow_pitch": 30,
                "arm_right_hand_link": 50,
            },
            orientation_weight={
                "imu_2": 60,
                "leg_left_hip_roll": 1,
                "leg_left_knee_pitch": 1,
                "leg_left_ankle_roll": 10,
                "leg_right_hip_roll": 1,
                "leg_right_knee_pitch": 1,
                "leg_right_ankle_roll": 10,
                "arm_left_elbow_pitch": 10,
                "arm_left_hand_link": 10,
                "arm_right_elbow_pitch": 10,
                "arm_right_hand_link": 10,
            },
            name="target",
        ),
        PositionLimit(weight=30.0, residual_mode="exp_barrier", frame=HumanoidContext.FRAME, name="position_limit"),
        ConfigurationTask(rest_q="midrange", weight=0.1, name="rest"),
        SmoothnessTask(
            prev_var=HumanoidContext.PREVIOUS_STATE,
            weight=[
                15.0,
                20.0,
                15.0,
                45.0,
                10.0,
                15.0,
                20.0,
                15.0,
                45.0,
                10.0,
                25.0,
                15.0,
                5.0,
                13.0,
                5.0,
                5.0,
                25.0,
                15.0,
                5.0,
                13.0,
                5.0,
                5.0,
            ],
            base_weight=1.5,
            name="smoothness",
        ),
        VelocityLimitTask(
            prev_state_var=HumanoidContext.PREVIOUS_STATE,
            dt=HumanoidContext.STEP,
            velocity_limits=HumanoidContext.VELOCITY_LIMITS,
            weight=0.2,
            name="velocity_limit",
        ),
        SceneCollisionTask(scene=HumanoidContext.SCENE, weight=0.0, margin=0.02, name="collision"),
        SelfCollisionTask(weight=0.0, margin=0.01, name="self_collision"),
    ],
)
