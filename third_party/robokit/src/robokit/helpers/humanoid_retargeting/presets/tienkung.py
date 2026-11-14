"""Tienkung Lite retargeting preset."""

import numpy as np

from robokit.assets.robots.humanoids import tienkung_lite
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


tienkung = HumanoidRetargeterOnlineConfig(
    urdf_path=str(tienkung_lite.URDF_PATH),
    human_root_name="pelvis",
    human_height_assumption=1.8,
    velocity_limit_dt=0.03333333,
    velocity_clamp_scale=0.8,
    solver=PopulationSolverConfig(
        stages=[Stage(6, 4, LMOptimizerConfig(lm_lambda=1.5)), Stage(3, 1)],
        cuda_graph_mode="full",
    ),
    scale_table={
        "pelvis": 0.957,
        "spine3": 0.957,
        "left_hip": 0.957,
        "left_knee": 0.957,
        "left_foot": 0.957,
        "left_elbow": 0.957,
        "right_hip": 0.957,
        "right_knee": 0.957,
        "right_foot": 0.957,
        "right_elbow": 0.957,
    },
    link_mapping={
        "pelvis": LinkMapping(
            "pelvis", np.zeros(3, dtype=np.float32), np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float32)
        ),
        "waist_link": LinkMapping(
            "spine3", np.zeros(3, dtype=np.float32), np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float32)
        ),
        "hip_roll_l_link": LinkMapping(
            "left_hip", np.zeros(3, dtype=np.float32), np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float32)
        ),
        "knee_pitch_l_link": LinkMapping(
            "left_knee", np.zeros(3, dtype=np.float32), np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float32)
        ),
        "ankle_roll_l_link": LinkMapping(
            "left_foot", np.zeros(3, dtype=np.float32), np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float32)
        ),
        "elbow_l_link": LinkMapping(
            "left_elbow", np.zeros(3, dtype=np.float32), np.array([0.5612, -0.4306, -0.4303, 0.5608], dtype=np.float32)
        ),
        "hip_roll_r_link": LinkMapping(
            "right_hip", np.zeros(3, dtype=np.float32), np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float32)
        ),
        "knee_pitch_r_link": LinkMapping(
            "right_knee", np.zeros(3, dtype=np.float32), np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float32)
        ),
        "ankle_roll_r_link": LinkMapping(
            "right_foot", np.zeros(3, dtype=np.float32), np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float32)
        ),
        "elbow_r_link": LinkMapping(
            "right_elbow", np.zeros(3, dtype=np.float32), np.array([0.4306, -0.5612, 0.5608, -0.4303], dtype=np.float32)
        ),
    },
    terms=[
        FrameTask(
            frame_index=HumanoidContext.TARGET_LINKS,
            T_world_target=HumanoidContext.TARGETS,
            position_weight={
                "pelvis": 100,
                "waist_link": 0,
                "hip_roll_l_link": 10,
                "knee_pitch_l_link": 10,
                "ankle_roll_l_link": 100,
                "elbow_l_link": 30,
                "hip_roll_r_link": 10,
                "knee_pitch_r_link": 10,
                "ankle_roll_r_link": 100,
                "elbow_r_link": 30,
            },
            orientation_weight={
                "pelvis": 60,
                "waist_link": 10,
                "hip_roll_l_link": 1,
                "knee_pitch_l_link": 1,
                "ankle_roll_l_link": 10,
                "elbow_l_link": 10,
                "hip_roll_r_link": 1,
                "knee_pitch_r_link": 1,
                "ankle_roll_r_link": 10,
                "elbow_r_link": 10,
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
