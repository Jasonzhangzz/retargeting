"""Image pyramids, candidate sampling and coupled trajectory calibration settings."""

from dataclasses import dataclass, field, replace
from typing import List, Literal, Optional, Sequence

from robokit.helpers.hand_eye_calibration.mask_alignment_task import MaskAlignmentNormalEqTask
from robokit.opt.lm_optimizer import LMOptimizerConfig
from robokit.opt.population_solver import PopulationSolverConfig, Stage
from robokit.terms.task import ResidualTask


@dataclass
class PyramidLevel:
    """One coarse-to-fine level: the images and views one solver segment fits."""

    downscale: int
    """Image downscale factor, capped so the short side keeps at least 32 pixels."""
    blur_sigma: float
    """Blur (pixels) of the rendered and target masks; coarse levels blur more for a wider basin."""
    patience: int
    """Iterations without improvement before the level stops."""
    obs_subset_size: Optional[int] = None
    """Use only this many views, those with the largest masks; None uses every view."""
    use_depth: bool = False
    """Add the depth residual; useful on fine levels only, since coarse levels search for the basin."""


def _default_levels() -> List[PyramidLevel]:
    # coarse-level obs_subset_size bounds per-iter mesh-transform work (~S*N*V); final polish uses all views
    return [
        PyramidLevel(downscale=16, blur_sigma=2.5, patience=3, obs_subset_size=5),
        PyramidLevel(downscale=8, blur_sigma=1.8, patience=2, obs_subset_size=5),
        PyramidLevel(downscale=4, blur_sigma=1.2, patience=2, obs_subset_size=8),
        PyramidLevel(downscale=2, blur_sigma=0.6, patience=2),
        PyramidLevel(downscale=1, blur_sigma=0.0, patience=4),
    ]


def _default_solver() -> PopulationSolverConfig:
    lm = LMOptimizerConfig(lm_lambda=1e-3, rho_min=1e-3, early_stopping_interval=2)
    return PopulationSolverConfig(
        stages=[
            Stage(15, 32, lm),
            Stage(10, 3),
            Stage(12, 2),
            Stage(10, 1, replace(lm, early_stopping_interval=1)),
            Stage(30),
        ]
    )


@dataclass
class HandEyeCalibratorConfig:
    """Eager fitting with pyramid levels, cost terms and candidate sampling.

    Example:
        >>> # xdoctest: +REQUIRES(module:nvdiffrast)
        >>> config = HandEyeCalibratorConfig(optimize_q=True, optimize_camera=False)
        >>> config.optimize_q, config.optimize_camera, config.candidate_noise_q
        (True, False, None)
    """

    levels: Sequence[PyramidLevel] = field(default_factory=_default_levels)
    """Coarse-to-fine levels, one per solver segment."""
    terms: Sequence[ResidualTask] = field(default_factory=lambda: [MaskAlignmentNormalEqTask()])
    """Cost terms; each walks `levels` itself. `MaskAlignmentOffsetTask` also calibrates the kinematic model."""
    solver: PopulationSolverConfig = field(default_factory=_default_solver)
    """One segment per level; native final ranking returns the best candidate."""
    candidate_noise_t: float = 0.35
    """Largest translation (m) of the initial candidates from the initial guess; each draws 0.5-1x of it."""
    candidate_noise_r: float = 0.7
    """Largest rotation (rad) of the initial candidates from the initial guess; each draws 0.5-1x of it."""
    restart_mismatch: float = 0.5
    """Re-solve with fresh candidates while the final mask mismatch (2 * cost / target mask area) exceeds this."""
    max_restarts: int = 4
    """Re-solves allowed after the first; 0 turns restarts off."""
    seed: int = 0
    """Seed of the candidate noise."""
    optimize_q: bool = False
    """Fit one unknown q vector shared across views; `solve(q=...)` takes an initial `[num_dofs]` guess."""
    optimize_camera: bool = True
    """Optimize the camera extrinsic; False keeps the supplied extrinsic fixed."""
    candidate_noise_q: Optional[float] = None
    """Joint sampling radius in radians/meters; None samples the full finite joint limits. Candidate 0 keeps initial q."""
    keypoint_weight: float = 0.03
    """Relative weight of optional 2D link-origin observations."""
    keypoint_sigma: float = 8.0
    """Keypoint residual scale in input-image pixels."""
    keypoint_huber_delta: float = 12.0
    """Robust-loss transition in input-image pixels."""
    keypoint_warmup_iters: int = 24
    """Keypoint-only initialization steps; half the broad samples remain unmodified."""
    candidate_metric: Literal["objective", "iou"] = "objective"
    """Select by cost or soft IoU; IoU mode uses union-normalized coarse fitting."""

    def __post_init__(self):
        self.solver = replace(self.solver, cuda_graph_mode="none")
        if self.candidate_metric not in ("objective", "iou"):
            raise ValueError("candidate_metric must be objective or iou")
        if self.candidate_noise_q is not None and self.candidate_noise_q < 0:
            raise ValueError("candidate_noise_q must be nonnegative or None")
        if len(self.levels) != len(self.solver.num_candidates) or any(n < 1 for n in self.solver.num_candidates):
            raise ValueError("solver needs one positive candidate count per level")
        if (
            self.keypoint_weight < 0
            or min(self.keypoint_sigma, self.keypoint_huber_delta) <= 0
            or self.keypoint_warmup_iters < 0
        ):
            raise ValueError("invalid keypoint loss or initialization settings")


@dataclass
class QposTrajectoryCalibratorConfig:
    """Eager frame fitting followed by coupled mask and temporal refinement.

    >>> # xdoctest: +REQUIRES(module:nvdiffrast)
    >>> config = QposTrajectoryCalibratorConfig(velocity_weight=0.01)
    >>> config.optimize_camera, config.velocity_weight
    (False, 0.01)
    """

    frame: HandEyeCalibratorConfig = field(
        default_factory=lambda: HandEyeCalibratorConfig(optimize_q=True, optimize_camera=False)
    )
    """Unknown-qpos frame initialization; cameras stay fixed during initialization."""
    initialization_frames: Optional[int] = None
    """Limit broad frame searches to this many visible frames; interpolate before fitting every frame. None searches all."""
    initialization_paths: int = 1
    """Alternative initial pose branches to rank against every source mask before refinement."""
    initialization_repairs: int = 0
    """Additional frame searches at the worst fitted source times; 0 disables repair."""
    initialization_seconds: Optional[float] = None
    """Stop starting repairs after this initialization time; the initial searches and ranking still complete."""
    levels: Sequence[PyramidLevel] = field(default_factory=lambda: [PyramidLevel(4, 1.0, 5), PyramidLevel(1, 0.0, 5)])
    solver: PopulationSolverConfig = field(
        default_factory=lambda: PopulationSolverConfig(
            stages=[Stage(20, 1, LMOptimizerConfig(lm_lambda=1e-3, rho_min=1e-3, early_stopping_interval=2)), Stage(20)]
        )
    )
    path_weight: float = 0.08
    velocity_weight: float = 0.015
    acceleration_weight: float = 0.02
    optimize_camera: bool = False
    """Refine one static extrinsic per view, shared across all frames."""
    camera_prior_weight: float = 0.0
    """Prior cost relative to the supplied calibration; 0 disables it."""
    camera_prior_t: float = 0.03
    camera_prior_r: float = 0.06
    """Translation (m) and rotation (rad) scales of shared camera corrections."""

    def __post_init__(self):
        self.solver = replace(self.solver, cuda_graph_mode="none")
        if self.initialization_frames is not None and self.initialization_frames < 1:
            raise ValueError("initialization_frames must be positive or None")
        if self.initialization_paths < 1 or self.initialization_repairs < 0:
            raise ValueError("initialization paths must be positive and repairs nonnegative")
        if self.initialization_seconds is not None and self.initialization_seconds <= 0:
            raise ValueError("initialization_seconds must be positive or None")
        if min(self.path_weight, self.velocity_weight, self.acceleration_weight) < 0:
            raise ValueError("temporal weights must be nonnegative")
        if self.camera_prior_weight < 0 or min(self.camera_prior_t, self.camera_prior_r) <= 0:
            raise ValueError("camera prior weight must be nonnegative and scales positive")
        if len(self.levels) != len(self.solver.num_candidates) or any(n != 1 for n in self.solver.num_candidates):
            raise ValueError("trajectory solver needs one candidate per level")
        if any(level.use_depth or level.obs_subset_size is not None for level in self.levels):
            raise ValueError("trajectory levels use all mask observations; depth and view subsets are unsupported")
