"""Fourier GR3 retargeting preset."""

import numpy as np

from robokit.assets.robots.humanoids import fourier_gr3
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


gr3 = HumanoidRetargeterOnlineConfig(
    urdf_path=str(fourier_gr3.URDF_PATH),
    human_root_name="pelvis",
    human_height_assumption=1.66,
    velocity_limit_dt=0.03333333,
    velocity_clamp_scale=0.8,
    velocity_limit_override=[
        4.0,
        4.0,
        4.0,
        8.0,
        8.0,
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
        "pelvis": 1.0,
        "spine3": 1.0,
        "left_hip": 1.0,
        "right_hip": 1.0,
        "left_knee": 1.0,
        "right_knee": 1.0,
        "left_foot": 1.0,
        "right_foot": 1.0,
        "left_shoulder": 1.0,
        "right_shoulder": 1.0,
        "left_elbow": 1.0,
        "right_elbow": 1.0,
        "left_wrist": 1.0,
        "right_wrist": 1.0,
    },
    link_mapping={
        "base_link": LinkMapping(
            "pelvis", np.array([0.0, 0.0, 0.0], dtype=np.float32), np.array([-0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        ),
        "left_thigh_roll_link": LinkMapping(
            "left_hip", np.array([0.0, 0.0, 0.0], dtype=np.float32), np.array([-0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        ),
        "left_shank_pitch_link": LinkMapping(
            "left_knee", np.array([0.0, 0.0, 0.0], dtype=np.float32), np.array([-0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        ),
        "left_foot_roll_link": LinkMapping(
            "left_foot", np.array([0.0, 0.0, 0.0], dtype=np.float32), np.array([-0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        ),
        "right_thigh_roll_link": LinkMapping(
            "right_hip", np.array([0.0, 0.0, 0.0], dtype=np.float32), np.array([-0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        ),
        "right_shank_pitch_link": LinkMapping(
            "right_knee", np.array([0.0, 0.0, 0.0], dtype=np.float32), np.array([-0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        ),
        "right_foot_roll_link": LinkMapping(
            "right_foot", np.array([0.0, 0.0, 0.0], dtype=np.float32), np.array([-0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        ),
        "torso_link": LinkMapping(
            "spine3", np.array([0.0, 0.0, 0.0], dtype=np.float32), np.array([-0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        ),
        "left_upper_arm_roll_link": LinkMapping(
            "left_shoulder",
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([-0.7071067811865475, 0.0, 0.7071067811865475, 0.0], dtype=np.float32),
        ),
        "left_lower_arm_pitch_link": LinkMapping(
            "left_elbow",
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([-0.7071067811865475, 0.0, 0.7071067811865475, 0.0], dtype=np.float32),
        ),
        "left_end_effector_link": LinkMapping(
            "left_wrist",
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([-0.7071067811865475, 0.0, 0.7071067811865475, 0.0], dtype=np.float32),
        ),
        "right_upper_arm_roll_link": LinkMapping(
            "right_shoulder",
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.7071067811865475, 0.0, 0.7071067811865475], dtype=np.float32),
        ),
        "right_lower_arm_pitch_link": LinkMapping(
            "right_elbow",
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.7071067811865475, 0.0, 0.7071067811865475], dtype=np.float32),
        ),
        "right_end_effector_link": LinkMapping(
            "right_wrist",
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.7071067811865475, 0.0, 0.7071067811865475], dtype=np.float32),
        ),
    },
    terms=[
        FrameTask(
            frame_index=HumanoidContext.TARGET_LINKS,
            T_world_target=HumanoidContext.TARGETS,
            position_weight={
                "base_link": 50,
                "left_thigh_roll_link": 0,
                "left_shank_pitch_link": 0,
                "left_foot_roll_link": 100,
                "right_thigh_roll_link": 0,
                "right_shank_pitch_link": 0,
                "right_foot_roll_link": 100,
                "torso_link": 0,
                "left_upper_arm_roll_link": 0,
                "left_lower_arm_pitch_link": 30,
                "left_end_effector_link": 30,
                "right_upper_arm_roll_link": 0,
                "right_lower_arm_pitch_link": 30,
                "right_end_effector_link": 30,
            },
            orientation_weight={
                "base_link": 10,
                "left_thigh_roll_link": 10,
                "left_shank_pitch_link": 10,
                "left_foot_roll_link": 50,
                "right_thigh_roll_link": 10,
                "right_shank_pitch_link": 10,
                "right_foot_roll_link": 50,
                "torso_link": 100,
                "left_upper_arm_roll_link": 10,
                "left_lower_arm_pitch_link": 10,
                "left_end_effector_link": 10,
                "right_upper_arm_roll_link": 10,
                "right_lower_arm_pitch_link": 10,
                "right_end_effector_link": 10,
            },
            name="target",
        ),
        PositionLimit(weight=30.0, residual_mode="exp_barrier", frame=HumanoidContext.FRAME, name="position_limit"),
        ConfigurationTask(
            rest_q="midrange",
            weight=[
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
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
            ],
            name="rest",
        ),
        SmoothnessTask(
            prev_var=HumanoidContext.PREVIOUS_STATE,
            weight=[
                100.0,
                100.0,
                100.0,
                50.0,
                50.0,
                25.0,
                15.0,
                5.0,
                13.0,
                5.0,
                5.0,
                5.0,
                25.0,
                15.0,
                5.0,
                13.0,
                5.0,
                5.0,
                5.0,
                15.0,
                20.0,
                15.0,
                45.0,
                10.0,
                10.0,
                15.0,
                20.0,
                15.0,
                45.0,
                10.0,
                10.0,
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
