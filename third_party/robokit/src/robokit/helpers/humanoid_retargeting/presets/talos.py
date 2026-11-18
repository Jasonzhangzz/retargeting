"""PAL Talos retargeting preset."""

import numpy as np

from robokit.assets.robots.humanoids import pal_talos
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


talos = HumanoidRetargeterOnlineConfig(
    urdf_path=str(pal_talos.URDF_PATH),
    human_root_name="pelvis",
    human_height_assumption=1.8,
    velocity_limit_dt=0.03333333,
    velocity_clamp_scale=0.8,
    solver=PopulationSolverConfig(
        stages=[Stage(6, 4, LMOptimizerConfig(lm_lambda=1.5)), Stage(3, 1)],
        cuda_graph_mode="full",
    ),
    scale_table={
        "pelvis": 1.036,
        "spine3": 1.036,
        "left_hip": 1.036,
        "left_knee": 1.036,
        "left_foot": 1.036,
        "left_elbow": 1.036,
        "left_wrist": 1.036,
        "right_hip": 1.036,
        "right_knee": 1.036,
        "right_foot": 1.036,
        "right_elbow": 1.036,
        "right_wrist": 1.036,
    },
    link_mapping={
        "base_link": LinkMapping(
            "pelvis", np.zeros(3, dtype=np.float32), np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float32)
        ),
        "torso_2_link": LinkMapping(
            "spine3", np.zeros(3, dtype=np.float32), np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float32)
        ),
        "leg_left_2_link": LinkMapping(
            "left_hip", np.zeros(3, dtype=np.float32), np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float32)
        ),
        "leg_left_4_link": LinkMapping(
            "left_knee", np.zeros(3, dtype=np.float32), np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float32)
        ),
        "leg_left_6_link": LinkMapping(
            "left_foot", np.zeros(3, dtype=np.float32), np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float32)
        ),
        "arm_left_4_link": LinkMapping(
            "left_elbow", np.zeros(3, dtype=np.float32), np.array([0.5002, -0.5002, -0.4998, 0.4998], dtype=np.float32)
        ),
        "arm_left_7_link": LinkMapping(
            "left_wrist", np.zeros(3, dtype=np.float32), np.array([0.5002, -0.5002, -0.4998, 0.4998], dtype=np.float32)
        ),
        "leg_right_2_link": LinkMapping(
            "right_hip", np.zeros(3, dtype=np.float32), np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float32)
        ),
        "leg_right_4_link": LinkMapping(
            "right_knee", np.zeros(3, dtype=np.float32), np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float32)
        ),
        "leg_right_6_link": LinkMapping(
            "right_foot", np.zeros(3, dtype=np.float32), np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float32)
        ),
        "arm_right_4_link": LinkMapping(
            "right_elbow", np.zeros(3, dtype=np.float32), np.array([0.5002, -0.5002, 0.4998, -0.4998], dtype=np.float32)
        ),
        "arm_right_7_link": LinkMapping(
            "right_wrist", np.zeros(3, dtype=np.float32), np.array([0.5002, -0.5002, 0.4998, -0.4998], dtype=np.float32)
        ),
    },
    terms=[
        FrameTask(
            frame_index=HumanoidContext.TARGET_LINKS,
            T_world_target=HumanoidContext.TARGETS,
            position_weight={
                "base_link": 100,
                "torso_2_link": 0,
                "leg_left_2_link": 10,
                "leg_left_4_link": 10,
                "leg_left_6_link": 100,
                "arm_left_4_link": 30,
                "arm_left_7_link": 50,
                "leg_right_2_link": 10,
                "leg_right_4_link": 10,
                "leg_right_6_link": 100,
                "arm_right_4_link": 30,
                "arm_right_7_link": 50,
            },
            orientation_weight={
                "base_link": 60,
                "torso_2_link": 10,
                "leg_left_2_link": 1,
                "leg_left_4_link": 1,
                "leg_left_6_link": 10,
                "arm_left_4_link": 10,
                "arm_left_7_link": 10,
                "leg_right_2_link": 1,
                "leg_right_4_link": 1,
                "leg_right_6_link": 10,
                "arm_right_4_link": 10,
                "arm_right_7_link": 10,
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
