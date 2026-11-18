"""PND Adam Lite retargeting preset."""

import numpy as np

from robokit.assets.robots.humanoids import pnd_adam_lite
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


adam_lite = HumanoidRetargeterOnlineConfig(
    urdf_path=str(pnd_adam_lite.URDF_PATH),
    human_root_name="pelvis",
    human_height_assumption=1.8,
    velocity_limit_dt=0.03333333,
    velocity_clamp_scale=0.8,
    solver=PopulationSolverConfig(
        stages=[Stage(6, 4, LMOptimizerConfig(lm_lambda=1.5)), Stage(3, 1)],
        cuda_graph_mode="full",
    ),
    scale_table={
        "pelvis": 0.982,
        "spine3": 0.982,
        "left_hip": 0.982,
        "left_knee": 0.982,
        "left_foot": 0.982,
        "left_elbow": 0.982,
        "left_wrist": 0.982,
        "right_hip": 0.982,
        "right_knee": 0.982,
        "right_foot": 0.982,
        "right_elbow": 0.982,
        "right_wrist": 0.982,
    },
    link_mapping={
        "pelvis": LinkMapping(
            "pelvis", np.zeros(3, dtype=np.float32), np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float32)
        ),
        "torso": LinkMapping(
            "spine3", np.zeros(3, dtype=np.float32), np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float32)
        ),
        "hipRollLeft": LinkMapping(
            "left_hip", np.zeros(3, dtype=np.float32), np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float32)
        ),
        "shinLeft": LinkMapping(
            "left_knee", np.zeros(3, dtype=np.float32), np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float32)
        ),
        "toeLeft": LinkMapping(
            "left_foot", np.zeros(3, dtype=np.float32), np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float32)
        ),
        "elbowLeft": LinkMapping(
            "left_elbow", np.zeros(3, dtype=np.float32), np.array([0.5002, -0.5002, -0.4998, 0.4998], dtype=np.float32)
        ),
        "wristYawLeft": LinkMapping(
            "left_wrist", np.zeros(3, dtype=np.float32), np.array([0.5002, -0.5002, -0.4998, 0.4998], dtype=np.float32)
        ),
        "hipRollRight": LinkMapping(
            "right_hip", np.zeros(3, dtype=np.float32), np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float32)
        ),
        "shinRight": LinkMapping(
            "right_knee", np.zeros(3, dtype=np.float32), np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float32)
        ),
        "toeRight": LinkMapping(
            "right_foot", np.zeros(3, dtype=np.float32), np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float32)
        ),
        "elbowRight": LinkMapping(
            "right_elbow", np.zeros(3, dtype=np.float32), np.array([0.5002, -0.5002, 0.4998, -0.4998], dtype=np.float32)
        ),
        "wristYawRight": LinkMapping(
            "right_wrist", np.zeros(3, dtype=np.float32), np.array([0.5002, -0.5002, 0.4998, -0.4998], dtype=np.float32)
        ),
    },
    terms=[
        FrameTask(
            frame_index=HumanoidContext.TARGET_LINKS,
            T_world_target=HumanoidContext.TARGETS,
            position_weight={
                "pelvis": 100,
                "torso": 0,
                "hipRollLeft": 10,
                "shinLeft": 10,
                "toeLeft": 100,
                "elbowLeft": 30,
                "wristYawLeft": 50,
                "hipRollRight": 10,
                "shinRight": 10,
                "toeRight": 100,
                "elbowRight": 30,
                "wristYawRight": 50,
            },
            orientation_weight={
                "pelvis": 60,
                "torso": 10,
                "hipRollLeft": 1,
                "shinLeft": 1,
                "toeLeft": 10,
                "elbowLeft": 10,
                "wristYawLeft": 10,
                "hipRollRight": 1,
                "shinRight": 1,
                "toeRight": 10,
                "elbowRight": 10,
                "wristYawRight": 10,
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
