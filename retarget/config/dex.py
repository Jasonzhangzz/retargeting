"""Validated prepared-reference and ComFree prediction parameters."""

from __future__ import annotations
import json
from dataclasses import dataclass, fields
from pathlib import Path
import numpy as np
import yaml
from retarget.env.dex.time_grid import TimeGrid

MIN_TRAJECTORY_FRAMES = 2
OBJECT_POSE_DIM = 7


@dataclass(frozen=True)
class LossWeights:
    obj_pos: float
    obj_rot: float
    terminal: float
    reference_pose_scale: float
    init: float
    action_ref_base: float
    action_ref_base_rot: float
    action_ref_finger: float
    contact_robustness: float
    marked_query_pos: float
    grasp_closure: float
    grasp_wrench: float
    soft_penetration: float
    penetration: float
    finger_joint_collision: float
    obj_vel: float = 0.1
    obj_ang_vel: float = 0.03


@dataclass(frozen=True)
class SmoothingConfig:
    enabled: bool
    control_velocity_weight: float
    control_acceleration_weight: float
    hand_state_weight: float
    control_velocity_base_position_scale: float
    control_velocity_base_rotation_scale: float
    control_velocity_finger_scale: float
    control_acceleration_base_position_scale: float
    control_acceleration_base_rotation_scale: float
    control_acceleration_finger_scale: float


@dataclass(frozen=True)
class BoundsConfig:
    init_base_pos: float
    init_base_rot: float
    init_finger: float


@dataclass(frozen=True)
class ContactConfig:
    target_phi: float
    penetration_limit: float
    marked_query_dist_mm: float
    marked_query_temp_mm: float
    age_ramp_frames: int
    finger_joint_surface_margin_mm: float

    @property
    def soft_penetration_limit(self) -> float:
        """Shared penetration depth for contact encouragement and its soft penalty."""
        return -self.target_phi


@dataclass(frozen=True)
class SeqConfig:
    seq_dir: Path
    hand: str
    xml: Path
    contact_guidance_npz: Path
    start: int
    end: int
    hold_frame: int
    hold_seconds: float


@dataclass(frozen=True)
class VideoConfig:
    fps: float


@dataclass(frozen=True)
class PredictionConfig:
    horizon: int
    warm_shift_scale: float


@dataclass(frozen=True)
class SimConfig:
    time_grid: TimeGrid
    exec_backend: str
    stiffness: float
    damping: float


@dataclass(frozen=True)
class RetargetConfig:
    loss: LossWeights
    smoothing: SmoothingConfig
    bounds: BoundsConfig
    prediction: PredictionConfig
    contact: ContactConfig
    simulator: SimConfig
    sequence: SeqConfig
    video: VideoConfig


def config_value(config: dict, path: str):
    value = config
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(f"Missing config key: {path}")
        value = value[key]
    if value is None:
        raise ValueError(f"Config key must not be null: {path}")
    return value


def cfg_str(config: dict, path: str) -> str:
    value = config_value(config, path)
    if not isinstance(value, str) or not value:
        raise TypeError(f"Config key must be a non-empty string: {path}")
    return value


def cfg_int(config: dict, path: str) -> int:
    value = config_value(config, path)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Config key must be an integer: {path}")
    return value


def cfg_float(config: dict, path: str) -> float:
    value = config_value(config, path)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Config key must be numeric: {path}")
    return float(value)


def cfg_bool(config: dict, path: str) -> bool:
    value = config_value(config, path)
    if not isinstance(value, bool):
        raise TypeError(f"Config key must be boolean: {path}")
    return value


def object_xml_stem(seq_dir: Path) -> str:
    meta = json.loads((seq_dir / "mano_raw/meta.json").read_text(encoding="utf-8"))
    object_name = str(meta["object"])
    prefix, _, rest = object_name.partition("_")
    return rest if prefix.isdigit() and rest else object_name


def contactaware_last_frame(contact_guidance_npz: Path, hand: str) -> int:
    output_dir = contact_guidance_npz.parent
    hand_path = output_dir / f"{hand}_qpos.npy"
    object_path = output_dir / "object_pose_7.npy"
    for label, path in (
        ("hand trajectory", hand_path),
        ("object trajectory", object_path),
        ("contact guidance", contact_guidance_npz),
    ):
        if path.exists():
            continue
        raise FileNotFoundError(f"Missing ContactAware {label}: {path}")

    hand_trajectory = np.load(hand_path, mmap_mode="r", allow_pickle=False)
    object_trajectory = np.load(object_path, mmap_mode="r", allow_pickle=False)
    with np.load(contact_guidance_npz, allow_pickle=False) as guidance:
        if "contact_mask" not in guidance:
            raise KeyError(
                f"Contact guidance has no contact_mask: {contact_guidance_npz}"
            )
        contact_mask = guidance["contact_mask"]

    if hand_trajectory.ndim != 2:
        raise ValueError(
            f"ContactAware hand trajectory must be rank-2: {hand_path} "
            f"shape={hand_trajectory.shape}"
        )
    frame_count = int(hand_trajectory.shape[0])
    if frame_count < MIN_TRAJECTORY_FRAMES:
        raise ValueError(
            "ContactAware trajectory must contain at least two frames: "
            f"{hand_path} shape={hand_trajectory.shape}"
        )
    if object_trajectory.shape != (frame_count, OBJECT_POSE_DIM):
        raise ValueError(
            "ContactAware hand/object frame counts must match: "
            f"hand={hand_trajectory.shape}, object={object_trajectory.shape}"
        )
    if contact_mask.ndim != 2 or contact_mask.shape[0] != frame_count:
        raise ValueError(
            "ContactAware hand/contact frame counts must match: "
            f"hand={hand_trajectory.shape}, contact={contact_mask.shape}"
        )
    return frame_count - 1


def _read_sequence(
    config: dict,
    sequence_dir: Path,
    contactaware_dir: Path | None = None,
) -> SeqConfig:
    seq_dir = sequence_dir.expanduser().resolve()
    if not seq_dir.is_dir():
        raise FileNotFoundError(f"Missing sequence directory: {seq_dir}")
    hand = cfg_str(config, "sequence.hand")
    contactaware_dir = (
        seq_dir / "retarget" / hand / "contactaware"
        if contactaware_dir is None
        else Path(contactaware_dir).expanduser().resolve()
    )
    contact_npz = contactaware_dir / "contact_guidance.npz"
    scene = cfg_str(config, "sequence.scene")
    xml = seq_dir / "scene" / hand / f"{object_xml_stem(seq_dir)}_{scene}.xml"
    start = cfg_int(config, "sequence.start")
    last_frame = contactaware_last_frame(contact_npz, hand)
    hold_frame = cfg_int(config, "sequence.hold_frame")
    hold_seconds = cfg_float(config, "sequence.hold_seconds")
    if hold_frame > last_frame:
        raise ValueError(
            "sequence.hold_frame exceeds ContactAware last frame: "
            f"hold_frame={hold_frame}, last_frame={last_frame}"
        )
    return SeqConfig(
        seq_dir=seq_dir,
        hand=hand,
        xml=xml,
        contact_guidance_npz=contact_npz,
        start=start,
        end=last_frame,
        hold_frame=hold_frame,
        hold_seconds=hold_seconds,
    )


def _require_finite_fields(section: str, value: object) -> None:
    for field in fields(value):
        scalar = getattr(value, field.name)
        if isinstance(scalar, (int, float)) and not np.isfinite(scalar):
            raise ValueError(f"{section}.{field.name} must be finite, got {scalar}")


def _require_nonnegative_fields(section: str, value: object) -> None:
    _require_finite_fields(section, value)
    for field in fields(value):
        scalar = getattr(value, field.name)
        if isinstance(scalar, (int, float)) and scalar < 0:
            raise ValueError(f"{section}.{field.name} must be >= 0, got {scalar}")


def _validate_bounds(bounds: BoundsConfig) -> None:
    _require_finite_fields("bounds", bounds)
    for field in fields(bounds):
        value = getattr(bounds, field.name)
        if value <= 0.0:
            raise ValueError(f"bounds.{field.name} must be positive")


def _validate_contact(contact: ContactConfig) -> None:
    _require_finite_fields("contact", contact)
    limits = (
        contact.soft_penetration_limit,
        contact.penetration_limit,
    )
    if min(limits) < 0.0 or limits != tuple(sorted(limits)):
        raise ValueError("contact penetration limits must be nonnegative and ordered")
    nonnegative = (
        contact.marked_query_dist_mm,
        contact.finger_joint_surface_margin_mm,
    )
    if min(nonnegative) < 0.0:
        raise ValueError("contact distance thresholds must be nonnegative")
    if contact.marked_query_temp_mm <= 0.0:
        raise ValueError("contact temperatures must be positive")
    if contact.age_ramp_frames < 0:
        raise ValueError("contact.age_ramp_frames must be >= 0")


def _validate_smoothing(smoothing: SmoothingConfig) -> None:
    _require_nonnegative_fields("smoothing", smoothing)
    scales = (
        smoothing.control_velocity_base_position_scale,
        smoothing.control_velocity_base_rotation_scale,
        smoothing.control_velocity_finger_scale,
        smoothing.control_acceleration_base_position_scale,
        smoothing.control_acceleration_base_rotation_scale,
        smoothing.control_acceleration_finger_scale,
    )
    if min(scales) <= 0.0:
        raise ValueError("smoothing normalization scales must be positive")


def load_retarget_config(sequence_dir, config_path, *, contactaware_dir=None):
    values = yaml.safe_load(Path(config_path).read_text())
    sections = {
        "sequence",
        "simulator",
        "contact_physics",
        "prediction",
        "loss",
        "smoothing",
        "bounds",
        "contact",
        "video",
    }
    if set(values) != sections:
        raise ValueError(f"Model YAML sections must be {sorted(sections)}")
    seq = _read_sequence(values, Path(sequence_dir), contactaware_dir)
    grid = TimeGrid.create(
        **{k: v for k, v in values["simulator"].items() if k != "exec_backend"}
    )
    sim = SimConfig(
        grid, values["simulator"]["exec_backend"], **values["contact_physics"]
    )
    if sim.exec_backend != "comfree":
        raise ValueError("Prepared dex/grasp model requires ComFree")
    cfg = RetargetConfig(
        LossWeights(**values["loss"]),
        SmoothingConfig(**values["smoothing"]),
        BoundsConfig(**values["bounds"]),
        PredictionConfig(**values["prediction"]),
        ContactConfig(**values["contact"]),
        sim,
        seq,
        VideoConfig(**values["video"]),
    )
    _require_nonnegative_fields("loss", cfg.loss)
    _validate_smoothing(cfg.smoothing)
    _validate_bounds(cfg.bounds)
    _validate_contact(cfg.contact)
    _require_nonnegative_fields("prediction", cfg.prediction)
    _require_nonnegative_fields("contact_physics", cfg.simulator)
    if cfg.prediction.horizon < 1:
        raise ValueError("Invalid prediction parameters")
    if not cfg.sequence.xml.is_file():
        raise FileNotFoundError(f"Missing scene: {cfg.sequence.xml}")
    return cfg
