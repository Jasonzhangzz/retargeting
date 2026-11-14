"""Booster K1 retargeting preset."""

import numpy as np

from robokit.assets.robots.humanoids import booster_k1
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


k1 = HumanoidRetargeterOnlineConfig(
    urdf_path=str(booster_k1.URDF_PATH),
    human_root_name="pelvis",
    human_height_assumption=1.8,
    velocity_limit_dt=0.03333333,
    velocity_clamp_scale=0.8,
    solver=PopulationSolverConfig(
        stages=[Stage(6, 4, LMOptimizerConfig(lm_lambda=1.5)), Stage(3, 1)],
        cuda_graph_mode="full",
    ),
    scale_table={
        "pelvis": 0.546,
        "left_hip": 0.546,
        "left_knee": 0.546,
        "left_foot": 0.546,
        "left_elbow": 0.546,
        "left_wrist": 0.546,
        "right_hip": 0.546,
        "right_knee": 0.546,
        "right_foot": 0.546,
        "right_elbow": 0.546,
        "right_wrist": 0.546,
    },
    link_mapping={
        "Trunk": LinkMapping(
            "pelvis", np.zeros(3, dtype=np.float32), np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float32)
        ),
        "Left_Hip_Roll": LinkMapping(
            "left_hip", np.zeros(3, dtype=np.float32), np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float32)
        ),
        "Left_Shank": LinkMapping(
            "left_knee", np.zeros(3, dtype=np.float32), np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float32)
        ),
        "left_foot_link": LinkMapping(
            "left_foot", np.zeros(3, dtype=np.float32), np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float32)
        ),
        "left_hand_link": LinkMapping(
            "left_elbow", np.zeros(3, dtype=np.float32), np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float32)
        ),
        "left_hand_end_ball": LinkMapping(
            "left_wrist", np.zeros(3, dtype=np.float32), np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float32)
        ),
        "Right_Hip_Roll": LinkMapping(
            "right_hip", np.zeros(3, dtype=np.float32), np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float32)
        ),
        "Right_Shank": LinkMapping(
            "right_knee", np.zeros(3, dtype=np.float32), np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float32)
        ),
        "right_foot_link": LinkMapping(
            "right_foot", np.zeros(3, dtype=np.float32), np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float32)
        ),
        "right_hand_link": LinkMapping(
            "right_elbow", np.zeros(3, dtype=np.float32), np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float32)
        ),
        "right_hand_end_ball": LinkMapping(
            "right_wrist", np.zeros(3, dtype=np.float32), np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float32)
        ),
    },
    terms=[
        FrameTask(
            frame_index=HumanoidContext.TARGET_LINKS,
            T_world_target=HumanoidContext.TARGETS,
            position_weight={
                "Trunk": 100,
                "Left_Hip_Roll": 10,
                "Left_Shank": 10,
                "left_foot_link": 100,
                "left_hand_link": 30,
                "left_hand_end_ball": 50,
                "Right_Hip_Roll": 10,
                "Right_Shank": 10,
                "right_foot_link": 100,
                "right_hand_link": 30,
                "right_hand_end_ball": 50,
            },
            orientation_weight={
                "Trunk": 60,
                "Left_Hip_Roll": 1,
                "Left_Shank": 1,
                "left_foot_link": 10,
                "left_hand_link": 10,
                "left_hand_end_ball": 10,
                "Right_Hip_Roll": 1,
                "Right_Shank": 1,
                "right_foot_link": 10,
                "right_hand_link": 10,
                "right_hand_end_ball": 10,
            },
            name="target",
        ),
        PositionLimit(weight=30.0, residual_mode="exp_barrier", frame=HumanoidContext.FRAME, name="position_limit"),
        ConfigurationTask(rest_q="midrange", weight=0.1, name="rest"),
        SmoothnessTask(prev_var=HumanoidContext.PREVIOUS_STATE, weight=15.0, base_weight=1.5, name="smoothness"),
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
