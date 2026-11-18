"""Canonical contact CEM parameters, importable by the isolated CPU4 worker."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Parameters:
    top_k: int = 4
    max_events: int = 3
    elite_fraction: float = .2
    refit: float = .5
    min_sigma_fraction: float = .05
    cpu_candidates: int = 8
    cpu_workers: int = 9
    lm_iterations: int = 5
    lm_damping: float = .02
    position_sigma_m: float = .005
    time_sigma_fraction: float = .12
    normal_speed_sigma: float = .05
    tangent_speed_sigma: float = .05
    humanoid_force_peak_weight: float = .01
    tracking_root_position_weight: tuple = (1., 1., 20.)
    tracking_root_rotation_weight: float = 5.
    tracking_joint_position_weight: float = 1.
    tracking_root_velocity_weight: float = .1
    tracking_joint_velocity_weight: float = .01


DEFAULTS = Parameters()
