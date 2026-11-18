"""Data types and the robot-free tuning recipe for full-body retargeting."""

from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Optional, Tuple

import numpy as np

from robokit.helpers.hand_retargeting.config import HandRetargeterOfflineConfig, HandSpec, refine_config
from robokit.helpers.hand_retargeting.presets import inspire as inspire_preset
from robokit.helpers.humanoid_retargeting.config import HumanoidRetargeterOfflineConfig
from robokit.helpers.ik.config import IKContext, IKSolverConfig
from robokit.opt.lm_optimizer import LMOptimizerConfig
from robokit.opt.population_solver import PopulationSolverConfig, Stage
from robokit.opt.sampler import RobertsSamplerConfig
from robokit.robo import Robot
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.position_task import PositionTask


_HAND_BASE_LINK = "hand_base_link"
_ORIENTATION_LINKS = ("thumb_proximal_base", "index_proximal", "pinky_proximal")


def _hand_pass(root_position: float, positions: Optional[float] = None) -> HandRetargeterOfflineConfig:
    """Captured MANO keypoints onto the standalone hand, from the Inspire preset."""
    config = replace(inspire_preset.offline, terms=deepcopy(inspire_preset.offline.terms))
    config.task("root_position").weight = root_position
    if positions is not None:
        config.task("positions").weight = positions
    return config


def _default_body() -> HumanoidRetargeterOfflineConfig:
    """Whole-body tracking with each grasp wrist injected as a high-weight target track."""
    from robokit.helpers.humanoid_retargeting.presets.g1_offline import g1_offline  # resolves the G1 assets

    config = replace(g1_offline, terms=deepcopy(g1_offline.terms))
    positions, orientations = config.task("positions"), config.task("orientations").dense_task
    rest = config.task("rest").dense_task.weight
    for side in ("left", "right"):
        positions.weight[f"{side}_wrist_yaw_link"] = 500.0
        orientations.weight[f"{side}_wrist_yaw_link"] = 150.0
        positions.weight[f"{side}_elbow_link"] = 60.0
    for start in (15, 22):  # G1's left and right shoulder pitch, roll and yaw, in its joint order
        rest[start : start + 3] = [1.0, 0.5, 0.1]
    return config


def _default_contact_refine() -> HandRetargeterOfflineConfig:
    """Captured trajectory anchored against contact and collision only."""
    config = refine_config(replace(inspire_preset.offline, locked_prefix_frames=2))
    config.task("velocity").weight = 10.0
    config.task("acceleration").weight = 20.0
    config.task("root_smoothness").base_weight[0:3] = 100.0
    config.task("root_smoothness").base_weight[3:6] = 50.0
    config.task("contact").weight = 300.0
    config.task("contact").contact_margin = 0.0
    config.task("collision").weight = 100.0
    config.task("collision").margin = 0.0
    config.task("anchor").dense_task.weight = 10.0
    return config


@dataclass
class FullBodyRetargetingConfig:
    """Robot-free recipe for the full-body pipeline; each stage solves its own config's terms."""

    hand_pass: HandRetargeterOfflineConfig = field(default_factory=lambda: _hand_pass(30.0, 10.0))
    """Captured MANO keypoints onto each standalone hand."""
    grasp_hand_pass: HandRetargeterOfflineConfig = field(default_factory=lambda: _hand_pass(10.0))
    """The hand pass of grasp refinement."""
    body: HumanoidRetargeterOfflineConfig = field(default_factory=_default_body)
    """Whole-body trajectory pass, with the grasp wrists as high-weight targets."""
    arm_snap: IKSolverConfig = field(
        default_factory=lambda: (
            IKSolverConfig(
                solver=PopulationSolverConfig(
                    stages=[Stage(15, optimizer=LMOptimizerConfig(lm_lambda=1.0))], cuda_graph_mode="full"
                ),
                sampler=RobertsSamplerConfig(keep_first=True, scale=0.0),
            )
            .add(PositionTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=50.0))
            .add(PositionLimit(weight=50.0))
        )
    )
    """Arm-only IK that puts each palm at its hand-pass pose."""
    contact_refine: HandRetargeterOfflineConfig = field(default_factory=_default_contact_refine)
    """Hand trajectory refinement against contact and collision."""
    hand_span_clamp: Tuple[float, float] = (0.9, 1.4)
    """Range of the robot-to-human hand span ratio that scales the captured hand."""
    arm_snap_trajectory_iters: int = 40
    """Arm-snap iterations for externally optimized hand trajectories, replacing `arm_snap`'s."""
    smooth_window: int = 3
    """Moving-average window, in frames, of the joint, base and object-frame hand base tracks."""
    object_frame_smooth_window: int = 5
    """Moving-average window, in frames, of the refined hand base in the object frame, fed back to the body."""
    tip_sphere_radius: float = 0.005
    """Radius of the fingertip spheres added to each hand."""


@dataclass
class HandTrajectory:
    """Externally optimized standalone-hand trajectory: `q (T, D)`, base pose `(T, 7)`."""

    q: np.ndarray
    T_world_base: np.ndarray


@dataclass
class _HandModel:
    """One loaded standalone hand and its frames relative to the assembly."""

    robot: Robot
    spec: HandSpec
    T_base_wrist: np.ndarray
    T_handbase_wristyaw: np.ndarray
    span: float


__all__ = ["FullBodyRetargetingConfig", "HandTrajectory"]
