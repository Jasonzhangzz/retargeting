"""Unitree H1-2 retargeting preset."""

import numpy as np

from robokit.assets.robots.humanoids import unitree_h1_2
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


h1_2 = HumanoidRetargeterOnlineConfig(
    urdf_path=str(unitree_h1_2.URDF_PATH),
    human_root_name="pelvis",
    human_height_assumption=1.66,
    velocity_limit_dt=0.03333333,
    velocity_clamp_scale=0.8,
    velocity_limit_override=[
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
        4.0,
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
    ],
    solver=PopulationSolverConfig(
        stages=[Stage(6, 4, LMOptimizerConfig(lm_lambda=1.5)), Stage(3, 1)],
        cuda_graph_mode="full",
    ),
    scale_table={
        "pelvis": 1.1,
        "spine3": 1.1,
        "left_hip": 1.1,
        "right_hip": 1.1,
        "left_knee": 1.1,
        "right_knee": 1.1,
        "left_foot": 1.1,
        "right_foot": 1.1,
        "left_shoulder": 1.1,
        "right_shoulder": 1.1,
        "left_elbow": 1.1,
        "right_elbow": 1.1,
        "left_wrist": 1.1,
        "right_wrist": 1.1,
    },
    link_mapping={
        "pelvis": LinkMapping(
            "pelvis", np.array([0.0, 0.0, 0.0], dtype=np.float32), np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float32)
        ),
        "left_hip_roll_link": LinkMapping(
            "left_hip",
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array(
                [0.4267755048530407, -0.5637931078484661, -0.5637931078484661, -0.4267755048530407], dtype=np.float32
            ),
        ),
        "left_knee_link": LinkMapping(
            "left_knee",
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float32),
        ),
        "left_ankle_roll_link": LinkMapping(
            "left_foot",
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float32),
        ),
        "right_hip_roll_link": LinkMapping(
            "right_hip",
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array(
                [0.4267755048530407, -0.5637931078484661, -0.5637931078484661, -0.4267755048530407], dtype=np.float32
            ),
        ),
        "right_knee_link": LinkMapping(
            "right_knee",
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float32),
        ),
        "right_ankle_roll_link": LinkMapping(
            "right_foot",
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float32),
        ),
        "torso_link": LinkMapping(
            "spine3", np.array([0.0, 0.0, 0.0], dtype=np.float32), np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float32)
        ),
        "left_shoulder_roll_link": LinkMapping(
            "left_shoulder",
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.70710678, 0.0, -0.70710678, 0.0], dtype=np.float32),
        ),
        "left_elbow_link": LinkMapping(
            "left_elbow", np.array([0.0, 0.0, 0.0], dtype=np.float32), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        ),
        "left_wrist_yaw_link": LinkMapping(
            "left_wrist", np.array([0.0, 0.0, 0.0], dtype=np.float32), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        ),
        "right_shoulder_roll_link": LinkMapping(
            "right_shoulder",
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.70710678, 0.0, 0.70710678], dtype=np.float32),
        ),
        "right_elbow_link": LinkMapping(
            "right_elbow",
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32),
        ),
        "right_wrist_yaw_link": LinkMapping(
            "right_wrist",
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32),
        ),
    },
    terms=[
        FrameTask(
            frame_index=HumanoidContext.TARGET_LINKS,
            T_world_target=HumanoidContext.TARGETS,
            position_weight={
                "pelvis": 100,
                "left_hip_roll_link": 0,
                "left_knee_link": 0,
                "left_ankle_roll_link": 100,
                "right_hip_roll_link": 0,
                "right_knee_link": 0,
                "right_ankle_roll_link": 100,
                "torso_link": 0,
                "left_shoulder_roll_link": 0,
                "left_elbow_link": 30,
                "left_wrist_yaw_link": 30,
                "right_shoulder_roll_link": 0,
                "right_elbow_link": 30,
                "right_wrist_yaw_link": 30,
            },
            orientation_weight={
                "pelvis": 10,
                "left_hip_roll_link": 10,
                "left_knee_link": 10,
                "left_ankle_roll_link": 10,
                "right_hip_roll_link": 10,
                "right_knee_link": 10,
                "right_ankle_roll_link": 10,
                "torso_link": 10,
                "left_shoulder_roll_link": 10,
                "left_elbow_link": 10,
                "left_wrist_yaw_link": 0,
                "right_shoulder_roll_link": 10,
                "right_elbow_link": 10,
                "right_wrist_yaw_link": 0,
            },
            name="target",
        ),
        PositionLimit(weight=30.0, residual_mode="exp_barrier", frame=HumanoidContext.FRAME, name="position_limit"),
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
            name="rest",
        ),
        SmoothnessTask(
            prev_var=HumanoidContext.PREVIOUS_STATE,
            weight=[
                25.0,
                20.0,
                25.0,
                45.0,
                10.0,
                10.0,
                25.0,
                20.0,
                25.0,
                45.0,
                10.0,
                10.0,
                100.0,
                35.0,
                25.0,
                15.0,
                13.0,
                5.0,
                5.0,
                5.0,
                35.0,
                25.0,
                15.0,
                13.0,
                5.0,
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
