"""Minimal configuration for the retained SPIDER humanoid implementation."""

from dataclasses import dataclass, field


@dataclass
class PhysicsConfig:
    embodiment_type: str = "bimanual"
    seed: int = 0
    sanity_check_seconds: float = 3.0
    sanity_check_pos_thresh: float = 0.01
    sanity_check_rot_thresh: float = 0.3
    model_path: str = ""
    device: str = "cuda:0"
    sim_dt: float = 0.01
    ctrl_dt: float = 0.4
    ref_dt: float = 0.02
    horizon: float = 1.6
    knot_dt: float = 0.4
    nconmax_per_env: int = 100
    njmax_per_env: int = 350
    num_dyn: int = (
        1  # number of environments for annealing, used for virtual contact constraint
    )
    num_dr: int = (
        1  # number of domain randomization groups, used for domain randomization
    )
    pair_margin_range: tuple[float, float] = (-0.005, 0.005)
    xy_offset_range: tuple[float, float] = (-0.005, 0.005)
    perturb_force: float = 0.0
    perturb_torque: float = 0.0
    object_actuator_ids: list[int] = field(default_factory=list)
    num_samples: int = 2048
    terminate_resample: bool = False
    object_pos_threshold: float = 0.1
    object_rot_threshold: float = 0.3
    base_pos_threshold: float = 0.5
    base_rot_threshold: float = 0.4
    first_ctrl_noise_scale: float = 0.5
    last_ctrl_noise_scale: float = 1.0
    joint_noise_scale: float = 0.15
    base_pos_rew_scale: float = 1.0
    base_rot_rew_scale: float = 0.3
    joint_rew_scale: float = 0.003
    pos_rew_scale: float = 1.0
    rot_rew_scale: float = 0.3
    vel_rew_scale: float = 0.0001
    terminal_rew_scale: float = 1.0
    contact_rew_scale: float = 0.0
    save_video: bool = True
    trace_site_ids: list = field(default_factory=list)
    horizon_steps: int = -1
    knot_steps: int = -1
    ref_steps: int = -1
    ctrl_steps: int = -1
    nq_obj: int = -1
    nv: int = -1
    nu: int = -1
    npair: int = -1

    nq: int = -1
    contact_site_ids: list = field(default_factory=list)
