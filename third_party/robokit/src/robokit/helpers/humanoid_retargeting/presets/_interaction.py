"""Shared construction for source-specific interaction retargeting presets."""

from typing import Dict

import numpy as np

from robokit.helpers.humanoid_retargeting.config import HumanoidContext, HumanoidRetargeterOnlineConfig, LinkMapping
from robokit.opt.lm_optimizer import LMOptimizerConfig
from robokit.opt.population_solver import PopulationSolverConfig, Stage
from robokit.terms.dense.configuration_task import ConfigurationTask
from robokit.terms.dense.interaction_mesh_task import InteractionMeshTask
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.scene_collision_task import SceneCollisionTask
from robokit.terms.dense.self_collision_task import SelfCollisionTask
from robokit.terms.dense.smoothness_task import SmoothnessTask


def _interaction_preset(
    link_map: Dict[str, str], collision_spheres_path: str, urdf_path: str
) -> HumanoidRetargeterOnlineConfig:
    """Convert an ordered source-joint map into the shared online configuration."""
    human_root_name = next(iter(link_map))
    return HumanoidRetargeterOnlineConfig(
        urdf_path=urdf_path,
        human_root_name=human_root_name,
        collision_spheres_path=collision_spheres_path,
        link_mapping={
            robot_link: LinkMapping(
                human_joint, np.zeros(3, dtype=np.float32), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            )
            for human_joint, robot_link in link_map.items()
        },
        interaction_terms=[
            InteractionMeshTask(robot_link_indices=HumanoidContext.TARGET_LINKS, weight=100.0, name="interaction")
        ],
        velocity_clamp_scale=0.0,
        solver=PopulationSolverConfig(
            stages=[Stage(30, optimizer=LMOptimizerConfig(lm_lambda=1.0))], cuda_graph_mode="none"
        ),
        terms=[
            PositionLimit(weight=30.0, residual_mode="abs", frame=HumanoidContext.FRAME, name="position_limit"),
            ConfigurationTask(rest_q="midrange", weight=0.0, name="rest"),
            SmoothnessTask(prev_var=HumanoidContext.PREVIOUS_STATE, weight=20.0, base_weight=1.0, name="smoothness"),
            SceneCollisionTask(scene=HumanoidContext.SCENE, weight=200.0, margin=0.012, name="collision"),
            SelfCollisionTask(weight=0.0, margin=0.01, name="self_collision"),
        ],
    )
