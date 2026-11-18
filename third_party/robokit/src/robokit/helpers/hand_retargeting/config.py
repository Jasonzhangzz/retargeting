"""Robot specifications and solver settings for hand retargeting."""

from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

from robokit.opt.lm_optimizer import LMOptimizerConfig
from robokit.opt.population_solver import PopulationSolverConfig, Stage
from robokit.opt.sampler import RobertsSamplerConfig
from robokit.terms.dense.configuration_task import ConfigurationTask
from robokit.terms.dense.frame_vector_task import FrameVectorTask
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.rotation_task import RotationTask
from robokit.terms.dense.self_collision_task import SelfCollisionTask
from robokit.terms.dense.smoothness_task import SmoothnessTask
from robokit.terms.sparse.trajectory_collision_task import TrajectoryCollisionTask
from robokit.terms.sparse.trajectory_position_task import TrajectoryContactTask, TrajectoryPositionTask
from robokit.terms.sparse.trajectory_retargeting_task import TrajectoryRetargetingTask
from robokit.terms.sparse.trajectory_smoothness_task import TrajectorySmoothnessTask
from robokit.terms.sparse.trajectory_surface_contact_task import TrajectorySurfaceContactTask
from robokit.terms.task import Context, ResidualTask
from robokit.terms.trajectory_task import TrajectoryTask
from robokit.utils.ee_coord_utils import MANOPTH_HAND_COORD_SPEC, HandCoordinateSpec


_Correspondence = Tuple[str, str, str, str]  # origin link, task link, origin target, task target


def _online_solver(
    iters: Tuple[int, int, int] = (6, 10, 0), widths: Tuple[int, int, int] = (16, 4, 1)
) -> PopulationSolverConfig:
    damping = (1e-1, 1e-2, 1e-2)
    return PopulationSolverConfig(
        stages=[Stage(n, w, LMOptimizerConfig(lm_lambda=d)) for n, w, d in zip(iters, widths, damping)],
        cuda_graph_mode="full",
    )


def _offline_solver() -> LMOptimizerConfig:
    return LMOptimizerConfig(
        linear_solver="cg",
        lm_lambda=0.5,
        max_iter=25,
        lambda_factor=2.0,
        lambda_min=1e-6,
        lambda_max=1e6,
        cuda_graph_mode="full",
    )


class HandContext(Context):
    """Hand task-template values as per-solve buffers: `(batch, ...)` online, `(batch, frames, ...)` offline."""

    POINTS = 0
    """Point targets in `spec.target_names` order."""
    VECTORS = 1
    """Online vector targets `(batch, targets * targets, 3)`, one row per ordered target pair."""
    DIRECTIONS = 2
    """Online direction targets `(batch, targets * targets, 3)`, one row per ordered target pair."""
    PREVIOUS_STATE = 3
    """The previous frame's solution (online)."""
    REST_Q = 4
    """The spec's rest configuration."""
    REST_MASK = 5
    """Joints a rest task pulls: the hand's own joints on a fixed-base arm, every joint otherwise (offline)."""
    TARGET_LINKS = 6
    """Links of the mapped targets (offline)."""
    TARGET_POINTS = 7
    """Point indices of the mapped targets (offline)."""
    PAIRS = 8
    """Retargeting pairs over the mapped targets: adjacent chain targets and the root (offline)."""
    TARGET_SCALES = 9
    """Pairwise target scale of each trajectory (offline)."""
    CONTACT_LINKS = 10
    """Links of the contact targets (offline)."""
    CONTACT_TARGETS = 11
    """Point indices of the contact targets."""
    ROOT_LINK = 12
    """The link of the root target (offline)."""
    ROOT_TARGET = 13
    """Point index of the root target (offline)."""
    ROOT_POSE = 14
    """Root target poses `(batch * frames, 1)`; `solve` needs `root_quat_wxyz` when a template reads them."""
    REFERENCE_Q = 15
    """The trajectory velocity and acceleration smoothness follow: the anchor when anchored, zero otherwise."""
    CONTACT_POINTS = 16
    """World contact points of the contact targets (offline)."""
    CONTACT_MASK = 17
    """Active contacts (offline)."""
    LOCAL_CONTACT_POINTS = 18
    """Contact points in each contact link's frame; zero places them at the link origins (offline)."""
    SURFACE_CONTACT_MASK = 19
    """Active sphere-surface contacts (offline)."""
    CONTACT_SPHERES = 20
    """`spec.contact_sphere_indices`: one collision sphere per contact target (offline)."""
    COLLISION_SPHERES = 21
    """Collision spheres of the contact fingers' branches; empty without a scene (offline)."""
    SCENE = 22
    """The caller's collision scene, or None (offline)."""
    SCENE_INDICES = 23
    """The caller's scene mapping, or None (offline)."""
    VALID_FRAMES = 24
    """Real, unpadded frames `(batch, frames, 1)` (offline)."""
    ANCHOR_Q = 25
    """Joint trajectory `(batch * frames, dofs)` a refinement stays near: the init state of `solve_with_contact`."""
    ANCHOR_BASE = 26
    """Base trajectory `(batch * frames,)` a refinement stays near (offline)."""
    TARGET_NAMES = 27
    """`spec.target_names`: the order of target rows (online)."""
    TARGET_LINK_NAMES = 28
    """`spec.target_link_names`: the link that tracks each target (online)."""
    FRAME = 29
    """Frames solved since the last reset, a `(1,)` int32 buffer (online)."""


def _default_online_terms() -> List[ResidualTask]:
    """Track points, link-pair vectors and directions; weights are keyed by target or by correspondence."""
    ctx = HandContext
    return [
        FrameVectorTask(
            targets=ctx.POINTS,
            target_names=ctx.TARGET_NAMES,
            target_links=ctx.TARGET_LINK_NAMES,
            weight={},
            weight_mode="normalized",
            huber_delta=0.02,
            name="positions",
        ),
        FrameVectorTask(
            targets=ctx.VECTORS,
            target_names=ctx.TARGET_NAMES,
            soft_gate_targets=ctx.CONTACT_TARGETS,
            weight={},
            weight_mode="cost",
            huber_on_norm=True,
            name="vectors",
        ),
        FrameVectorTask(
            targets=ctx.DIRECTIONS,
            target_names=ctx.TARGET_NAMES,
            weight={},
            weight_mode="normalized",
            direction_only=True,
            name="directions",
        ),
        # smoothness starts with the second frame of a stream
        SmoothnessTask(prev_var=ctx.PREVIOUS_STATE, warmup_frames=1, frame=ctx.FRAME, weight=0.0, name="smoothness"),
        ConfigurationTask(rest_q=ctx.REST_Q, weight=0.0, name="rest"),
        PositionLimit(weight=1.0, name="position_limit"),
        SelfCollisionTask(weight=0.0, margin=0.005, name="self_collision"),
    ]


def _default_offline_terms() -> List[ResidualTask]:
    """Track the target points: contact and root positions, root rotation, pairwise shape, and regularizers."""
    ctx = HandContext
    return [
        TrajectoryPositionTask(
            frame_index=ctx.CONTACT_LINKS,
            target_positions=ctx.POINTS,
            target_point_indices=ctx.CONTACT_TARGETS,
            weight=20.0,
            name="positions",
        ),
        TrajectoryPositionTask(
            frame_index=ctx.ROOT_LINK,
            target_positions=ctx.POINTS,
            target_point_indices=ctx.ROOT_TARGET,
            weight=1.0,
            name="root_position",
        ),
        TrajectoryTask(
            RotationTask(frame_index=ctx.ROOT_LINK, T_world_target=ctx.ROOT_POSE, weight=0.2), name="root_rotation"
        ),
        TrajectoryRetargetingTask(
            target_keypoints=ctx.POINTS,
            robot_link_indices=ctx.TARGET_LINKS,
            target_joint_indices=ctx.TARGET_POINTS,
            pair_indices=ctx.PAIRS,
            target_scale=ctx.TARGET_SCALES,
            position_weight=0.5,
            angle_weight=10.0,
            name="retargeting",
        ),
        TrajectorySmoothnessTask(reference_q=ctx.REFERENCE_Q, weight=9.0, name="velocity"),
        TrajectorySmoothnessTask(reference_q=ctx.REFERENCE_Q, weight=2.0, order=2, name="acceleration"),
        TrajectorySmoothnessTask(weight=0.0, base_weight=np.array([10.0] * 3 + [5.0] * 3), name="root_smoothness"),
        TrajectoryTask(PositionLimit(weight=100.0), name="position_limit"),
        TrajectoryTask(ConfigurationTask(rest_q=ctx.REST_Q, weight=0.05, mask=ctx.REST_MASK), name="rest"),
    ]


def _contact_tasks() -> List[ResidualTask]:
    ctx = HandContext
    return [
        TrajectoryContactTask(
            contact_points=ctx.CONTACT_POINTS,
            contact_link_indices=ctx.CONTACT_LINKS,
            contact_mask=ctx.CONTACT_MASK,
            local_contact_points=ctx.LOCAL_CONTACT_POINTS,
            weight=100.0,
            contact_margin=0.01,
            name="contact",
        ),
        TrajectorySurfaceContactTask(
            scene=ctx.SCENE,
            sphere_indices=ctx.CONTACT_SPHERES,
            contact_mask=ctx.SURFACE_CONTACT_MASK,
            scene_indices=ctx.SCENE_INDICES,
            weight=0.0,
            name="surface_contact",
        ),
        TrajectoryCollisionTask(
            scene=ctx.SCENE,
            sphere_indices=ctx.COLLISION_SPHERES,
            scene_indices=ctx.SCENE_INDICES,
            frame_sphere_mask=ctx.VALID_FRAMES,
            weight=20.0,
            margin=0.005,
            name="collision",
        ),
    ]


@dataclass(frozen=True)
class HandSpec:
    """Input topology and reference joint states for one robot hand."""

    floating_base: bool
    """Whether the hand base moves freely (six base DOFs) rather than staying fixed."""
    target_names: Tuple[str, ...]
    """Names of the input points, in input row order."""
    target_link_names: Dict[str, str]
    """The robot link that tracks each mapped target."""
    target_chains: Tuple[Tuple[str, ...], ...]
    """Finger chains of targets, base to tip; offline retargeting pairs adjacent targets."""
    root_target_name: str
    """The target at the hand root (the wrist)."""
    contact_target_names: Tuple[str, ...]
    """Targets that touch objects, usually the fingertips."""
    root_link_coord_spec: HandCoordinateSpec
    """Hand axes in the root link's frame."""
    target_coord_spec: HandCoordinateSpec = MANOPTH_HAND_COORD_SPEC
    """Hand axes of the input points."""
    init_q_by_name: Dict[str, float] = field(default_factory=dict)
    """Initial-guess joint values by joint name."""
    rest_q_by_name: Dict[str, float] = field(default_factory=dict)
    """Rest joint values by joint name; other joints rest at zero."""
    contact_sphere_indices: Tuple[int, ...] = ()
    """One collision sphere per contact target, for sphere-surface contact (offline)."""


@dataclass
class HandRetargeterOnlineConfig:
    """Frame-level hand-retargeting configuration."""

    terms: List[ResidualTask] = field(default_factory=_default_online_terms)
    """Task templates built at warmup; `HandContext` lists their inputs."""
    pinch_correspondences: Tuple[_Correspondence, ...] = ()
    """Vector rows whose target shrinks to near zero while their two targets pinch."""
    target_scale: float = 1.0
    """Scale from human to robot hand size applied to vector targets; pinched rows keep their own length."""
    target_ema_alpha: float = 1.0
    """EMA weight of the newest vector and direction targets, in (0, 1]; 1 disables smoothing."""
    output_ema_alpha: float = 1.0
    """EMA weight of the newest output joints, in (0, 1]; 1 disables smoothing."""
    sampler: RobertsSamplerConfig = field(default_factory=lambda: RobertsSamplerConfig(keep_first=True))
    """Joint noise around the previous solution; `scale` is a fraction of each joint's range."""
    base_init_sample_range: Tuple[float, float, float] = (0.1, 0.1, 0.0)
    """Per-axis base translation noise (m) around the previous solution of a floating hand."""
    solver: PopulationSolverConfig = field(default_factory=_online_solver)
    """Candidate schedule and LM settings of each frame's solve."""
    pinch_threshold: float = 0.03
    """Target distance (m) below which a pinch pair latches."""
    pinch_release_threshold: float = 0.05
    """Target distance (m) above which a latched pinch pair releases."""
    reference_origin: Literal["world", "wrist"] = "world"
    """Solve floating hands relative to the target wrist position ("wrist") or in world; I/O frames are unchanged."""

    def add(self, term: ResidualTask, *, name: Optional[str] = None) -> "HandRetargeterOnlineConfig":
        """Append a task, optionally assigning its input role.

        >>> from robokit.terms.dense.configuration_task import ConfigurationTask
        >>> config = HandRetargeterOnlineConfig(terms=[]).add(ConfigurationTask(weight=0.1), name="rest")
        >>> config.task("rest").weight
        0.1
        """
        if name is not None:
            term.name = name
        if term.name and any(t.name == term.name for t in self.terms):
            raise ValueError(f"Duplicate task name {term.name!r}.")
        self.terms.append(term)
        return self

    def task(self, name: str) -> ResidualTask:
        """Return a named task; raise KeyError when absent.

        >>> from robokit.terms.dense.configuration_task import ConfigurationTask
        >>> config = HandRetargeterOnlineConfig(terms=[]).add(ConfigurationTask(weight=0.2), name="rest")
        >>> config.task("rest").weight
        0.2
        """
        for term in self.terms:
            if term.name == name:
                return term
        raise KeyError(f"No task named {name!r}.")


@dataclass
class HandRetargeterOfflineConfig:
    """One whole-trajectory problem; `contact_config` and `refine_config` derive the contact ones from tracking."""

    terms: List[ResidualTask] = field(default_factory=_default_offline_terms)
    """Task templates of the tracking solve; `HandContext` lists their inputs."""
    target_scale: float = 1.0
    """Scale from human to robot hand size of each trajectory's pairwise targets until `set_target_scales`."""
    optimize_base: bool = True
    """Optimize a floating hand's base trajectory; False keeps the initial one."""
    active_joint_names: Optional[Tuple[str, ...]] = None
    """Joints the solve moves; None moves every joint."""
    locked_prefix_frames: int = 0
    """Leading frames whose joints keep their initial values; base activity follows `optimize_base`."""
    solver: LMOptimizerConfig = field(default_factory=_offline_solver)
    """LM settings of the whole-trajectory solve."""

    def add(self, term: ResidualTask, *, name: Optional[str] = None) -> "HandRetargeterOfflineConfig":
        """Append a task, optionally assigning its input role.

        >>> from robokit.terms.dense.configuration_task import ConfigurationTask
        >>> config = HandRetargeterOfflineConfig(terms=[]).add(ConfigurationTask(weight=0.1), name="rest")
        >>> config.task("rest").weight
        0.1
        """
        if name is not None:
            term.name = name
        if term.name and any(t.name == term.name for t in self.terms):
            raise ValueError(f"Duplicate task name {term.name!r}.")
        self.terms.append(term)
        return self

    def task(self, name: str) -> ResidualTask:
        """Return a named task; raise KeyError when absent.

        >>> from robokit.terms.dense.configuration_task import ConfigurationTask
        >>> config = HandRetargeterOfflineConfig(terms=[]).add(ConfigurationTask(weight=0.2), name="rest")
        >>> config.task("rest").weight
        0.2
        """
        for term in self.terms:
            if term.name == name:
                return term
        raise KeyError(f"No task named {name!r}.")


def contact_config(config: HandRetargeterOfflineConfig) -> HandRetargeterOfflineConfig:
    """Derive contact tracking from a tracking config: pairwise shape, every mapped point, regularizers, contacts.

    >>> [term.name for term in contact_config(HandRetargeterOfflineConfig()).terms][:3]
    ['retargeting', 'positions', 'velocity']
    """
    ctx, terms = HandContext, {term.name: deepcopy(term) for term in config.terms}
    positions = TrajectoryPositionTask(
        frame_index=ctx.TARGET_LINKS,
        target_positions=ctx.POINTS,
        target_point_indices=ctx.TARGET_POINTS,
        weight=terms["positions"].weight,
        name="positions",
    )
    kept = [terms[name] for name in ("velocity", "acceleration", "root_smoothness", "position_limit", "rest")]
    return replace(config, terms=[terms["retargeting"], positions, *kept, *_contact_tasks()])


def refine_config(config: HandRetargeterOfflineConfig) -> HandRetargeterOfflineConfig:
    """Derive anchored refinement from a tracking config: contacts near the init trajectory, its smoothness and limits.

    >>> [term.name for term in refine_config(HandRetargeterOfflineConfig()).terms][:2]
    ['contact', 'anchor']
    """
    ctx, terms = HandContext, {term.name: deepcopy(term) for term in config.terms}
    contact, surface_contact, collision = _contact_tasks()
    anchor = TrajectoryTask(ConfigurationTask(rest_q=ctx.ANCHOR_Q, weight=1.0), name="anchor")
    base_anchor = TrajectoryTask(
        ConfigurationTask(T_world_base_rest=ctx.ANCHOR_BASE, weight=0.0, base_weight=20.0, include_joints=False),
        name="base_anchor",
    )
    kept = [terms[name] for name in ("velocity", "acceleration", "root_smoothness", "position_limit")]
    return replace(config, terms=[contact, anchor, surface_contact, collision, *kept, base_anchor])


__all__ = [
    "HandContext",
    "HandRetargeterOfflineConfig",
    "HandRetargeterOnlineConfig",
    "HandSpec",
    "contact_config",
    "refine_config",
]
