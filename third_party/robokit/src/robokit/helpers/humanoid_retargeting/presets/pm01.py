"""EngineAI PM01 retargeting preset."""

import numpy as np

from robokit.assets.robots.humanoids import engineai_pm01
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


pm01 = HumanoidRetargeterOnlineConfig(
    urdf_path=str(engineai_pm01.URDF_PATH),
    human_root_name="pelvis",
    human_height_assumption=1.8,
    velocity_limit_dt=0.03333333,
    velocity_clamp_scale=0.8,
    solver=PopulationSolverConfig(
        stages=[Stage(6, 4, LMOptimizerConfig(lm_lambda=1.5)), Stage(3, 1)],
        cuda_graph_mode="full",
    ),
    scale_table={
        "pelvis": 0.824,
        "spine3": 0.824,
        "left_hip": 0.824,
        "left_knee": 0.824,
        "left_foot": 0.824,
        "left_elbow": 0.824,
        "left_wrist": 0.824,
        "right_hip": 0.824,
        "right_knee": 0.824,
        "right_foot": 0.824,
        "right_elbow": 0.824,
        "right_wrist": 0.824,
    },
    link_mapping={
        "LINK_BASE": LinkMapping(
            "pelvis", np.zeros(3, dtype=np.float32), np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float32)
        ),
        "LINK_TORSO_YAW": LinkMapping(
            "spine3", np.zeros(3, dtype=np.float32), np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float32)
        ),
        "LINK_HIP_ROLL_L": LinkMapping(
            "left_hip", np.zeros(3, dtype=np.float32), np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float32)
        ),
        "LINK_KNEE_PITCH_L": LinkMapping(
            "left_knee", np.zeros(3, dtype=np.float32), np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float32)
        ),
        "LINK_ANKLE_ROLL_L": LinkMapping(
            "left_foot", np.zeros(3, dtype=np.float32), np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float32)
        ),
        "LINK_ELBOW_PITCH_L": LinkMapping(
            "left_elbow", np.zeros(3, dtype=np.float32), np.array([0.5002, -0.5002, -0.4998, 0.4998], dtype=np.float32)
        ),
        "LINK_ELBOW_END_L": LinkMapping(
            "left_wrist", np.zeros(3, dtype=np.float32), np.array([0.5002, -0.5002, -0.4998, 0.4998], dtype=np.float32)
        ),
        "LINK_HIP_ROLL_R": LinkMapping(
            "right_hip", np.zeros(3, dtype=np.float32), np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float32)
        ),
        "LINK_KNEE_PITCH_R": LinkMapping(
            "right_knee", np.zeros(3, dtype=np.float32), np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float32)
        ),
        "LINK_ANKLE_ROLL_R": LinkMapping(
            "right_foot", np.zeros(3, dtype=np.float32), np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float32)
        ),
        "LINK_ELBOW_PITCH_R": LinkMapping(
            "right_elbow", np.zeros(3, dtype=np.float32), np.array([0.5002, -0.5002, 0.4998, -0.4998], dtype=np.float32)
        ),
        "LINK_ELBOW_END_R": LinkMapping(
            "right_wrist", np.zeros(3, dtype=np.float32), np.array([0.5002, -0.5002, 0.4998, -0.4998], dtype=np.float32)
        ),
    },
    terms=[
        FrameTask(
            frame_index=HumanoidContext.TARGET_LINKS,
            T_world_target=HumanoidContext.TARGETS,
            position_weight={
                "LINK_BASE": 100,
                "LINK_TORSO_YAW": 0,
                "LINK_HIP_ROLL_L": 10,
                "LINK_KNEE_PITCH_L": 10,
                "LINK_ANKLE_ROLL_L": 100,
                "LINK_ELBOW_PITCH_L": 30,
                "LINK_ELBOW_END_L": 50,
                "LINK_HIP_ROLL_R": 10,
                "LINK_KNEE_PITCH_R": 10,
                "LINK_ANKLE_ROLL_R": 100,
                "LINK_ELBOW_PITCH_R": 30,
                "LINK_ELBOW_END_R": 50,
            },
            orientation_weight={
                "LINK_BASE": 60,
                "LINK_TORSO_YAW": 10,
                "LINK_HIP_ROLL_L": 1,
                "LINK_KNEE_PITCH_L": 1,
                "LINK_ANKLE_ROLL_L": 10,
                "LINK_ELBOW_PITCH_L": 10,
                "LINK_ELBOW_END_L": 10,
                "LINK_HIP_ROLL_R": 1,
                "LINK_KNEE_PITCH_R": 1,
                "LINK_ANKLE_ROLL_R": 10,
                "LINK_ELBOW_PITCH_R": 10,
                "LINK_ELBOW_END_R": 10,
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
