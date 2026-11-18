"""Shared humanoid retargeting configuration dataclasses for online mappings and offline optimizer settings."""

import inspect
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import yaml

from robokit.helpers.ik.config import IKContext, IKSolverConfig
from robokit.opt.gd_optimizer import GDOptimizerConfig
from robokit.opt.lbfgs_optimizer import LBFGSOptimizerConfig
from robokit.opt.lm_optimizer import LMOptimizerConfig
from robokit.opt.population_solver import PopulationSolverConfig, Stage
from robokit.opt.sampler import GaussianSamplerConfig, RobertsSamplerConfig
from robokit.opt.var_values import VarKey
from robokit.terms.dense.configuration_task import ConfigurationTask
from robokit.terms.dense.frame_task import FrameTask
from robokit.terms.dense.frame_vector_task import FrameVectorTask
from robokit.terms.dense.interaction_mesh_task import InteractionMeshTask
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.position_task import PositionTask
from robokit.terms.dense.rotation_task import RotationTask
from robokit.terms.dense.scene_collision_task import SceneCollisionTask
from robokit.terms.dense.self_collision_task import SelfCollisionTask
from robokit.terms.dense.smoothness_task import SmoothnessTask
from robokit.terms.dense.velocity_limit_task import VelocityLimitTask
from robokit.terms.sparse.trajectory_limit_task import TrajectoryLimitTask
from robokit.terms.sparse.trajectory_position_task import TrajectoryPositionTask
from robokit.terms.sparse.trajectory_self_collision_task import TrajectorySelfCollisionTask
from robokit.terms.sparse.trajectory_smoothness_task import TrajectorySmoothnessTask
from robokit.terms.task import Context, ResidualTask, Task
from robokit.terms.trajectory_task import TrajectoryTask


if TYPE_CHECKING:
    from robokit.robo import Robot


def _tasks_to_dict(terms: Sequence[Task]) -> List[dict]:
    """Encode task templates as their class and constructor arguments, read back from the attributes keeping them."""

    def plain(value):
        if isinstance(value, Task):
            return _tasks_to_dict([value])[0]
        if isinstance(value, Enum):
            return _yaml_enum(value)
        if isinstance(value, np.ndarray):
            return {"array": value.tolist(), "dtype": str(value.dtype)}
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {key: plain(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(item) for item in value]
        return value

    result = []
    for term in terms:
        if term.robot is not None:
            raise ValueError("Serialize task templates before building them for a robot.")
        stored, parameters = vars(term), {}
        for key in inspect.signature(type(term)).parameters:
            attr = next((a for a in (f"{key}_arg", f"_{key}_arg", f"{key}_input", key, f"_{key}") if a in stored), None)
            if attr is not None and key not in ("robot", "name"):
                parameters[key] = stored[attr]
        result.append({"type": type(term).__name__, "name": term.name, "parameters": plain(parameters)})
    return result


_ENUMS: Dict[str, Any] = {VarKey.__name__: VarKey}  # HumanoidContext joins below


def _yaml_enum(value: Any) -> Any:
    """Encode an enum as its `Enum.MEMBER` string; decode such a string back to the enum."""
    if isinstance(value, Enum):
        return f"{type(value).__name__}.{value.name}"
    kind, _, member = value.partition(".") if isinstance(value, str) else ("", "", "")
    return _ENUMS[kind][member] if kind in _ENUMS else value


def _tasks_from_dict(entries: List[dict]) -> List[ResidualTask]:
    types, pending = {}, [ResidualTask]
    while pending:  # every task class, by name
        cls = pending.pop()
        types[cls.__name__] = cls
        pending.extend(cls.__subclasses__())

    def value(item):
        if isinstance(item, dict) and set(item) == {"type", "name", "parameters"}:
            return _tasks_from_dict([item])[0]
        if isinstance(item, dict) and set(item) == {"array", "dtype"}:
            return np.asarray(item["array"], dtype=item["dtype"])
        if isinstance(item, dict):
            return {key: value(entry) for key, entry in item.items()}
        if isinstance(item, list):
            return [value(entry) for entry in item]
        return _yaml_enum(item)

    return [types[entry["type"]](**value(entry.get("parameters", {})), name=entry["name"]) for entry in entries]


class HumanoidContext(Context):
    """Values the humanoid retargeters give their task templates; arrays are buffers refilled for each solve."""

    TARGET_LINKS = 0
    """Robot links of `link_mapping`, in its order; a template keys named weights by link."""
    TARGETS = 1
    """Target poses of those links: `(batch, links)` online; offline `(batch * frames, links)` before overrides."""
    TARGET_POSITIONS = 2
    """Target positions `(batch, frames, links, 3)`, after `target_overrides` (offline)."""
    VELOCITY_LIMITS = 3
    """Joint velocity limits: `velocity_limit_override`, else the robot's."""
    STEP = 4
    """`velocity_limit_dt`, the time between frames."""
    PREVIOUS_STATE = 5
    """The previous frame's solution (online)."""
    FRAME = 6
    """Frames solved since the last reset, a `(1,)` int32 buffer (online)."""
    CORRESPONDENCE_ORIGINS = 7
    """Origin links of `correspondence_edges` (online)."""
    CORRESPONDENCE_LINKS = 8
    """Task links of `correspondence_edges` (online)."""
    VECTORS = 9
    """Human vectors `(batch, edges, 3)` of `correspondence_edges` (online)."""
    SCENE = 10
    """The caller's scene, or None (online)."""
    OVERRIDDEN_TARGETS = 11
    """Target poses `(batch * frames, links)` after `target_overrides` (offline); `TARGETS` holds them before."""


def _default_online_terms() -> List[ResidualTask]:
    ctx = HumanoidContext
    return [
        FrameTask(
            frame_index=ctx.TARGET_LINKS,
            T_world_target=ctx.TARGETS,
            position_weight={},
            orientation_weight={},
            name="target",
        ),
        PositionLimit(weight=30.0, residual_mode="exp_barrier", frame=ctx.FRAME, name="position_limit"),
        ConfigurationTask(rest_q="midrange", weight=0.0, name="rest"),
        SmoothnessTask(prev_var=ctx.PREVIOUS_STATE, weight=1.0, base_weight=1.5, name="smoothness"),
        VelocityLimitTask(
            prev_state_var=ctx.PREVIOUS_STATE,
            dt=ctx.STEP,
            velocity_limits=ctx.VELOCITY_LIMITS,
            weight=3.0,
            name="velocity_limit",
        ),
        FrameVectorTask(
            origin_link_indices=ctx.CORRESPONDENCE_ORIGINS,
            task_link_indices=ctx.CORRESPONDENCE_LINKS,
            targets=ctx.VECTORS,
            weight=0.0,
            weight_mode="cost",
            name="correspondence",
        ),
        SceneCollisionTask(scene=ctx.SCENE, weight=0.0, margin=0.02, name="collision"),
        # self-collision penalty over the collision spheres; weight > 0 needs collision_spheres_path
        SelfCollisionTask(weight=0.0, margin=0.01, name="self_collision"),
    ]


_ENUMS[HumanoidContext.__name__] = HumanoidContext
_IDENTITY_POSE = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def _default_offline_terms() -> List[ResidualTask]:
    """Track the mapped links, regularized by smoothness, limits and rest; set the tracking weights per link."""
    ctx = HumanoidContext
    return [
        TrajectoryPositionTask(
            frame_index=ctx.TARGET_LINKS, target_positions=ctx.TARGET_POSITIONS, weight={}, name="positions"
        ),
        TrajectoryTask(
            RotationTask(frame_index=ctx.TARGET_LINKS, T_world_target=ctx.TARGETS, weight={}), name="orientations"
        ),
        TrajectorySmoothnessTask(weight=20.0, base_weight=np.array([10.0] * 3 + [5.0] * 3), name="smoothness"),
        TrajectoryTask(PositionLimit(weight=100.0), name="position_limit"),
        TrajectoryTask(ConfigurationTask(rest_q="midrange", weight=1.0), name="rest"),
        TrajectoryTask(
            ConfigurationTask(T_world_base_rest=_IDENTITY_POSE, weight=0.0, base_weight=0.0, include_joints=False),
            name="base_rest",
        ),
        TrajectoryLimitTask(
            order=1, upper=ctx.VELOCITY_LIMITS, dt=ctx.STEP, stencil="forward", weight=0.0, name="velocity_limit"
        ),
        TrajectorySelfCollisionTask(weight=0.0, margin=0.01, name="self_collision"),
    ]


@dataclass
class LinkMapping:
    """The human joint a robot link tracks, and the link target's offsets from it."""

    human_joint: str
    """Human joint whose pose drives the link's target."""
    position_offset: np.ndarray
    """Position offset `(3,)` of the target, in the target frame."""
    rotation_offset: np.ndarray
    """Rotation `(4,)` wxyz from the human joint frame to the target frame; zero means identity."""


@dataclass
class CorrespondenceEdge:
    """Drive the robot link vector toward the corresponding human joint vector each frame."""

    origin_link: str
    """Robot link at the vector's origin."""
    task_link: str
    """Robot link at the vector's tip."""
    origin_joint: str
    """Human joint at the target vector's origin."""
    task_joint: str
    """Human joint at the target vector's tip."""


@dataclass
class TargetArrays:
    """Per-target inputs to `compute_targets_kernel`, resolved from a config's `link_mapping`."""

    joint_indices: np.ndarray
    """Index `(N,)` of each target's joint in the caller's human joint order."""
    scales: np.ndarray
    """Scale `(N,)` of each target, from `scale_table`."""
    rotation_offsets: np.ndarray
    """Normalized wxyz rotation offsets `(N, 4)`."""
    position_offsets: np.ndarray
    """Position offsets `(N, 3)`."""
    root_joint_index: int
    """Index of the human root joint."""
    root_target_index: int
    """Target index of the root; -1 when the root is not a mapped target."""
    root_scale: float
    """Scale of the root joint."""


def _default_retargeting_solver() -> PopulationSolverConfig:
    return PopulationSolverConfig(
        stages=[
            Stage(4, 16, LMOptimizerConfig(lm_lambda=10.0)),
            Stage(6, 4, LMOptimizerConfig(lm_lambda=1.0)),
            Stage(0, 1),
        ],
        cuda_graph_mode="full",
    )


@dataclass
class HumanoidRetargeterOnlineConfig:
    """A robot's human-to-robot mapping, task templates and per-frame solver; offline retargeting reuses the mapping."""

    terms: List[ResidualTask] = field(default_factory=_default_online_terms)
    """Task templates; each names its inputs with `HumanoidContext` members."""
    urdf_path: Optional[str] = None
    """Robot URDF, loaded when the retargeter gets no robot."""
    human_root_name: str = "pelvis"
    """Human joint that places the robot base."""
    human_height_assumption: float = 1.8
    """Human height in meters when a solve gets no `human_heights`."""
    ground_height: float = 0.0
    """Ground height of the human motion, subtracted from every target height."""
    collision_spheres_path: Optional[Union[str, Path]] = None
    """Robot collision spheres, needed by a weighted self-collision task."""
    scale_table: Dict[str, float] = field(default_factory=dict)
    """Target scale per human joint; absent joints use 1.0."""
    link_mapping: Dict[str, LinkMapping] = field(default_factory=dict)
    """Robot link to the human joint it tracks, in target order."""
    correspondence_edges: Optional[List[CorrespondenceEdge]] = None
    """Link vectors the `correspondence` task drives toward human joint vectors; None is off (online only)."""
    interaction_terms: List[ResidualTask] = field(default_factory=list)
    """Interaction-mesh task templates, built at construction so the first frame is set before warmup; empty is off."""
    velocity_limit_dt: float = 1.0 / 30.0
    """Time between frames, for the velocity limits and the output clamp."""
    velocity_clamp_scale: float = 0.6
    """Clamp each output step at this fraction of the velocity limits; 0 is off."""
    velocity_limit_override: Optional[Sequence[Union[int, float]]] = None
    """Joint velocity limits in rad/s, replacing the URDF's; None uses the URDF's."""
    solver: PopulationSolverConfig = field(default_factory=_default_retargeting_solver)
    """Candidate schedule and optimizers of each frame's solve."""

    def add(self, term: ResidualTask, *, name: Optional[str] = None) -> "HumanoidRetargeterOnlineConfig":
        """Append a task, optionally assigning its input role.

        >>> from robokit.terms.dense.configuration_task import ConfigurationTask
        >>> config = HumanoidRetargeterOnlineConfig(terms=[]).add(ConfigurationTask(weight=0.1), name="rest")
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
        >>> config = HumanoidRetargeterOnlineConfig(terms=[]).add(ConfigurationTask(weight=0.2), name="rest")
        >>> config.task("rest").weight
        0.2
        """
        for term in self.terms:
            if term.name == name:
                return term
        raise KeyError(f"No task named {name!r}.")

    @property
    def human_joint_names(self) -> Tuple[str, ...]:
        """Return the ordered human joints expected by the solve APIs.

        Example:
            >>> import numpy as np
            >>> config = HumanoidRetargeterOnlineConfig(
            ...     link_mapping={"torso": LinkMapping("spine", np.zeros(3), np.zeros(4))},
            ... )
            >>> config.human_joint_names  # mapped joints first, then the root
            ('spine', 'pelvis')
        """
        mapping_joints = [entry.human_joint for entry in self.link_mapping.values()]
        edge_joints = [
            joint for edge in self.correspondence_edges or [] for joint in (edge.origin_joint, edge.task_joint)
        ]
        return tuple(dict.fromkeys([*mapping_joints, self.human_root_name, *edge_joints]))

    def target_arrays(self, human_joint_names: Sequence[str]) -> "TargetArrays":
        """Resolve `link_mapping` in input `human_joint_names` order for both solvers' `compute_targets_kernel` arrays.

        Example:
            >>> import numpy as np
            >>> config = HumanoidRetargeterOnlineConfig(
            ...     link_mapping={
            ...         "torso": LinkMapping("spine", np.zeros(3), np.zeros(4)),
            ...         "hip": LinkMapping("pelvis", np.zeros(3), np.zeros(4)),
            ...     },
            ...     scale_table={"spine": 0.9},
            ... )
            >>> arrays = config.target_arrays(config.human_joint_names)
            >>> arrays.joint_indices.tolist(), arrays.scales.tolist()
            ([0, 1], [0.8999999761581421, 1.0])
            >>> arrays.root_joint_index, arrays.root_target_index
            (1, 1)
            >>> arrays.rotation_offsets[0].tolist()  # zero offset normalizes to identity
            [1.0, 0.0, 0.0, 0.0]
        """
        mapping_joints = [entry.human_joint for entry in self.link_mapping.values()]
        joint_index = {name: i for i, name in enumerate(human_joint_names)}
        rotation_offsets = np.zeros((len(mapping_joints), 4), dtype=np.float32)
        rotation_offsets[:, 0] = 1.0
        position_offsets = np.zeros((len(mapping_joints), 3), dtype=np.float32)
        for i, entry in enumerate(self.link_mapping.values()):
            norm = float(np.linalg.norm(entry.rotation_offset))
            if norm > 1e-6:
                rotation_offsets[i] = entry.rotation_offset / norm
            if np.linalg.norm(entry.position_offset) > 1e-6:
                position_offsets[i] = entry.position_offset
        return TargetArrays(
            joint_indices=np.array([joint_index[name] for name in mapping_joints], dtype=np.int32),
            scales=np.array([self.scale_table.get(name, 1.0) for name in mapping_joints], dtype=np.float32),
            rotation_offsets=rotation_offsets,
            position_offsets=position_offsets,
            root_joint_index=joint_index[self.human_root_name],
            root_target_index=(
                mapping_joints.index(self.human_root_name) if self.human_root_name in mapping_joints else -1
            ),
            root_scale=float(self.scale_table.get(self.human_root_name, 1.0)),
        )

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "HumanoidRetargeterOnlineConfig":
        yaml_path = Path(yaml_path)
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        urdf_path: Optional[str] = None
        if "urdf_path" in data:
            from robokit.assets import fetch

            urdf_path = str(fetch(["robots/**"]) / "robots" / data["urdf_path"])

        config = cls(
            urdf_path=urdf_path,
            human_root_name=data.get("human_root_name", "pelvis"),
            human_height_assumption=float(data.get("human_height_assumption", 1.8)),
            ground_height=float(data.get("ground_height", 0.0)),
        )

        config.scale_table = {k: float(v) for k, v in data.get("scale_table", {}).items()}

        for robot_link, entry in data.get("link_mapping", {}).items():
            config.link_mapping[robot_link] = LinkMapping(
                human_joint=entry["human_joint"],
                position_offset=np.array(entry["position_offset"], dtype=np.float32),
                rotation_offset=np.array(entry["rotation_offset"], dtype=np.float32),
            )

        if "terms" in data:
            config.terms = _tasks_from_dict(data["terms"])
        else:
            config.task("target").position_weight = {
                name: entry.get("position_weight", 0.0) for name, entry in data.get("link_mapping", {}).items()
            }
            config.task("target").orientation_weight = {
                name: entry.get("orientation_weight", 0.0) for name, entry in data.get("link_mapping", {}).items()
            }
            if "position_limit_weight" in data:
                config.task("position_limit").weight = data["position_limit_weight"]
            if "velocity_limit_weight" in data:
                config.task("velocity_limit").weight = data["velocity_limit_weight"]
            if "scene_collision_weight" in data:
                config.task("collision").weight = data["scene_collision_weight"]
            if "scene_collision_margin" in data:
                config.task("collision").margin = data["scene_collision_margin"]
            if "self_collision_weight" in data:
                config.task("self_collision").weight = data["self_collision_weight"]
            if "self_collision_margin" in data:
                config.task("self_collision").margin = data["self_collision_margin"]
            # scalar or per-DOF list
            if "smoothness_weight" in data:
                config.task("smoothness").weight = data["smoothness_weight"]
            if "base_smoothness_weight" in data:
                config.task("smoothness").base_weight = data["base_smoothness_weight"]
            if "rest_weight" in data:
                config.task("rest").weight = data["rest_weight"]
        config.velocity_limit_dt = float(data.get("velocity_limit_dt", config.velocity_limit_dt))
        config.velocity_clamp_scale = float(data.get("velocity_clamp_scale", config.velocity_clamp_scale))
        if "collision_spheres_path" in data:
            config.collision_spheres_path = data["collision_spheres_path"]
        if "self_collision_max_active_pairs" in data:
            config.task("self_collision").max_active_pairs = int(data["self_collision_max_active_pairs"])
        if "velocity_limit_override" in data:
            config.velocity_limit_override = data["velocity_limit_override"]
        # joint-limit residual mode; per-DOF mask weights in [0, 1]
        if "position_limit_mode" in data:
            config.task("position_limit").residual_mode = str(data["position_limit_mode"])
        if "per_dof_limit_mask" in data:
            config.task("position_limit").mask = {k: float(v) for k, v in data["per_dof_limit_mask"].items()}
        # ramp the limit penalty 0 -> full over this many frames (0 = disabled)
        if "limit_warmup_frames" in data:
            config.task("position_limit").warmup_frames = int(data["limit_warmup_frames"])
        if "correspondence_edges" in data:
            config.correspondence_edges = [
                CorrespondenceEdge(
                    origin_link=e["origin_link"],
                    task_link=e["task_link"],
                    origin_joint=e["origin_joint"],
                    task_joint=e["task_joint"],
                )
                for e in data["correspondence_edges"]
            ]
            if "terms" not in data:
                config.task("correspondence").weight = [e.get("weight", 1.0) for e in data["correspondence_edges"]]
        if "interaction_terms" in data:
            config.interaction_terms = _tasks_from_dict(data["interaction_terms"])
        elif "interaction" in data:  # legacy YAML
            config.interaction_terms = [
                InteractionMeshTask(
                    robot_link_indices=HumanoidContext.TARGET_LINKS,
                    weight=data["interaction"]["laplacian_weight"],
                    name="interaction",
                )
            ]
        if "solver" in data:
            settings = dict(data["solver"])
            if "stages" not in settings:  # legacy YAML: segment ends, widths and one LM damping value per segment
                ends = [*settings.pop("select_at", []), settings.pop("num_iters", 10)]
                damping = settings.pop("lm_lambda", [10.0] * len(ends))
                settings["stages"] = [
                    {
                        "iters": end - start,
                        "num_candidates": width,
                        "optimizer": {"type": "LMOptimizerConfig", "lm_lambda": float(value)},
                    }
                    for start, end, width, value in zip([0, *ends], ends, settings.pop("num_candidates", [1]), damping)
                ]
            settings.setdefault("cuda_graph_mode", data.get("cuda_graph_mode", "full"))
            config_types = {
                c.__name__: c
                for c in (
                    LMOptimizerConfig,
                    GDOptimizerConfig,
                    LBFGSOptimizerConfig,
                    RobertsSamplerConfig,
                    GaussianSamplerConfig,
                )
            }
            stages = settings["stages"]
            restored = []
            for entry in [stage.get("optimizer") for stage in stages] + list(settings.get("samplers", [])):
                if entry is None:
                    restored.append(None)
                    continue
                options = dict(entry)
                name = options.pop("type")
                if name not in config_types:
                    raise ValueError(f"Unsupported online config type: {name}")
                if name == "LBFGSOptimizerConfig" and "line_search_alphas" in options:
                    options["line_search_alphas"] = tuple(options["line_search_alphas"])
                restored.append(config_types[name](**{key: _yaml_enum(value) for key, value in options.items()}))
            settings["stages"] = [Stage(**{**stage, "optimizer": c}) for stage, c in zip(stages, restored)]
            settings["samplers"] = restored[len(stages) :]
            config.solver = PopulationSolverConfig(**settings)
        elif "cuda_graph_mode" in data:
            config.solver = replace(config.solver, cuda_graph_mode=data["cuda_graph_mode"])

        return config

    def _weight_to_yaml(self, val: Union[float, int, Sequence[float], np.ndarray]) -> Union[float, List[float]]:
        if isinstance(val, np.ndarray):
            return val.tolist()
        if isinstance(val, (list, tuple)):
            return list(val)
        return float(val)  # type: ignore[arg-type]

    def to_dict(self) -> dict:
        solver = asdict(self.solver)
        stages = self.solver.stages
        encoded = []
        for config in [stage.optimizer for stage in stages] + list(self.solver.samplers):
            if config is None:
                encoded.append(None)
                continue
            if not is_dataclass(config) or type(config) not in (
                LMOptimizerConfig,
                GDOptimizerConfig,
                LBFGSOptimizerConfig,
                RobertsSamplerConfig,
                GaussianSamplerConfig,
            ):
                raise ValueError(f"Unsupported online config type: {type(config).__name__}")
            options = {key: _yaml_enum(value) for key, value in asdict(config).items()}
            if any(callable(value) for value in options.values()):
                raise ValueError("Callable optimizer schedules cannot be serialized.")
            encoded.append({"type": type(config).__name__, **options})
        solver["stages"] = [{**stage, "optimizer": c} for stage, c in zip(solver["stages"], encoded)]
        solver["samplers"] = encoded[len(stages) :]
        data: dict = {}
        if self.urdf_path is not None:
            data["urdf_path"] = self.urdf_path
        if self.collision_spheres_path is not None:
            data["collision_spheres_path"] = str(self.collision_spheres_path)
        data.update(
            {
                "human_root_name": self.human_root_name,
                "human_height_assumption": self.human_height_assumption,
                "ground_height": self.ground_height,
                "scale_table": self.scale_table,
                "link_mapping": {
                    robot_link: {
                        "human_joint": m.human_joint,
                        "position_offset": m.position_offset.tolist(),
                        "rotation_offset": m.rotation_offset.tolist(),
                    }
                    for robot_link, m in self.link_mapping.items()
                },
                "velocity_limit_dt": self.velocity_limit_dt,
                "velocity_clamp_scale": self.velocity_clamp_scale,
                "solver": solver,
            }
        )
        if self.velocity_limit_override is not None:
            data["velocity_limit_override"] = self._weight_to_yaml(
                np.asarray(self.velocity_limit_override)
                if not isinstance(self.velocity_limit_override, (float, int))
                else self.velocity_limit_override
            )
        if self.correspondence_edges:
            data["correspondence_edges"] = [
                {
                    "origin_link": e.origin_link,
                    "task_link": e.task_link,
                    "origin_joint": e.origin_joint,
                    "task_joint": e.task_joint,
                }
                for e in self.correspondence_edges
            ]
        if self.interaction_terms:
            data["interaction_terms"] = _tasks_to_dict(self.interaction_terms)
        data["terms"] = _tasks_to_dict(self.terms)
        return data

    def to_yaml(self, yaml_path: Union[str, Path]):
        data = self.to_dict()
        with open(yaml_path, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

    def load_robot(self, load_meshes: bool = False) -> "Robot":
        """Load the config's robot, with its collision spheres when the config names them."""
        from robokit.robo import Robot

        assert self.urdf_path is not None
        return Robot.load(
            str(self.urdf_path),
            load_meshes=load_meshes,
            load_collision_spheres=self.collision_spheres_path is not None,
            collision_spheres_path=self.collision_spheres_path,
        )


@dataclass
class HumanoidRetargeterOfflineConfig:
    """Whole-clip optimizer settings; the paired online config supplies the mapping and velocity limits."""

    terms: List[ResidualTask] = field(default_factory=_default_offline_terms)
    """Trajectory task templates; tracking weights are per link, and self-collision costs O(pairs) per frame."""
    warm_start: IKSolverConfig = field(
        default_factory=lambda: (
            IKSolverConfig(
                enable_T_world_base=True,
                solver=PopulationSolverConfig(stages=[Stage(6, 4, LMOptimizerConfig(lm_lambda=1.5)), Stage(3, 1)]),
            )
            .add(PositionTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=10.0))
            .add(RotationTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=5.0))
            .add(PositionLimit(weight=30.0))
        )
    )
    """Per-frame IK that seeds the trajectory."""
    solver: LMOptimizerConfig = field(
        default_factory=lambda: LMOptimizerConfig(
            linear_solver="cg",
            lm_lambda=0.5,
            max_iter=25,
            lambda_min=1e-6,
            lambda_max=1e6,
            rho_min=1e-4,
            cuda_graph_mode="full",
        )
    )
    """Trajectory solver."""

    def add(self, term: ResidualTask, *, name: Optional[str] = None) -> "HumanoidRetargeterOfflineConfig":
        """Append a task, optionally assigning its input role.

        >>> from robokit.terms.dense.configuration_task import ConfigurationTask
        >>> config = HumanoidRetargeterOfflineConfig(terms=[]).add(ConfigurationTask(weight=0.1), name="rest")
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
        >>> config = HumanoidRetargeterOfflineConfig(terms=[]).add(ConfigurationTask(weight=0.2), name="rest")
        >>> config.task("rest").weight
        0.2
        """
        for term in self.terms:
            if term.name == name:
                return term
        raise KeyError(f"No task named {name!r}.")

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "HumanoidRetargeterOfflineConfig":
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        settings = dict(data.pop("solver", {}))
        for key in ("lm_lambda", "max_iter", "lambda_factor", "lambda_min", "lambda_max", "rho_min", "cuda_graph_mode"):
            if key in data:
                value = data.pop(key)
                if key == "cuda_graph_mode" and value == "iter":
                    value = "full"  # legacy offline execution treated iter as full
                if key in settings:
                    raise ValueError(f"Duplicate solver setting: {key}")
                settings[key] = value
        defaults = cls()
        if "terms" in data:
            data["terms"] = _tasks_from_dict(data["terms"])
        else:
            if "self_collision_weight" in data:
                defaults.task("self_collision").weight = data.pop("self_collision_weight")
            if "self_collision_margin" in data:
                defaults.task("self_collision").margin = data.pop("self_collision_margin")
            if "joint_smoothness_weight" in data:
                defaults.task("smoothness").weight = data.pop("joint_smoothness_weight")
            if "root_position_smoothness_weight" in data:
                defaults.task("smoothness").base_weight[:3] = data.pop("root_position_smoothness_weight")
            if "root_orientation_smoothness_weight" in data:
                defaults.task("smoothness").base_weight[3:] = data.pop("root_orientation_smoothness_weight")
            if "limit_weight" in data:
                defaults.task("position_limit").dense_task.weight = data.pop("limit_weight")
            if "rest_weight" in data:
                defaults.task("rest").dense_task.weight = data.pop("rest_weight")
            if "base_rest_weight" in data:
                defaults.task("base_rest").dense_task.base_weight = data.pop("base_rest_weight")
            if "velocity_limit_weight" in data:
                defaults.task("velocity_limit").weight = data.pop("velocity_limit_weight")
            data["terms"] = defaults.terms
        return cls(**data, solver=replace(defaults.solver, **settings))

    def to_yaml(self, yaml_path: Union[str, Path]):
        with open(yaml_path, "w") as f:
            data = {"terms": _tasks_to_dict(self.terms), "solver": asdict(self.solver)}
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


__all__ = [
    "CorrespondenceEdge",
    "HumanoidContext",
    "HumanoidRetargeterOfflineConfig",
    "HumanoidRetargeterOnlineConfig",
    "LinkMapping",
    "TargetArrays",
]
