"""Module-level HEC preset constants."""

from dataclasses import replace
from typing import cast

from robokit.helpers.hand_eye_calibration.config import (
    HandEyeCalibratorConfig,
    PyramidLevel,
    QposTrajectoryCalibratorConfig,
    _default_levels,
)
from robokit.opt.lm_optimizer import LMOptimizerConfig
from robokit.opt.population_solver import PopulationSolverConfig, Stage


# --- camera calibration -----------------------------------------------------
_levels = _default_levels()

rgb = HandEyeCalibratorConfig()
rgbd = HandEyeCalibratorConfig(levels=_levels[:-2] + [replace(level, use_depth=True) for level in _levels[-2:]])
depth = HandEyeCalibratorConfig(levels=[replace(level, use_depth=True) for level in _levels])
rgb_single_level = replace(  # the full-resolution level alone, with the whole pyramid's iteration budget
    rgb,
    levels=[rgb.levels[-1]],
    solver=replace(rgb.solver, stages=[Stage(rgb.solver.num_iters, 1, cast(list, rgb.solver.optimizer)[-1])]),
)

# --- unknown frame qpos ------------------------------------------------------
_lm = LMOptimizerConfig(lm_lambda=1e-3, rho_min=1e-3, early_stopping_interval=2)
# Screen broad seeds once, then refine a smaller beam until image convergence.
rgb_qpos = replace(
    rgb,
    levels=[
        PyramidLevel(8, 2.0, 5),
        PyramidLevel(4, 1.0, 4),
        PyramidLevel(1, 0.0, 5),
    ],
    solver=PopulationSolverConfig(stages=[Stage(1, 512, _lm), Stage(39, 64), Stage(40, 8)]),
    optimize_q=True,
    optimize_camera=False,
    candidate_noise_q=None,
    restart_mismatch=0.02,
    max_restarts=2,
)
rgb_qpos_camera = replace(rgb_qpos, optimize_camera=True)

# --- coupled trajectories ---------------------------------------------------
rgb_qpos_trajectory = QposTrajectoryCalibratorConfig(
    frame=replace(
        rgb_qpos,
        solver=replace(
            rgb_qpos.solver,
            stages=[replace(s, num_candidates=n) for s, n in zip(rgb_qpos.solver.stages, (1024, 32, 8))],
        ),
        max_restarts=0,
        candidate_metric="iou",
        keypoint_sigma=16.0,
        keypoint_huber_delta=24.0,
    ),
    initialization_frames=4,
    initialization_paths=rgb_qpos.solver.num_candidates[-1],
    initialization_repairs=1,
    initialization_seconds=18.0,
    levels=[PyramidLevel(4, 1.0, 5), PyramidLevel(2, 0.6, 5), PyramidLevel(1, 0.0, 5)],
    solver=PopulationSolverConfig(
        stages=[Stage(16, 1, replace(_lm, lambda_min=1e-4, linear_solver="torch_cholesky")), Stage(16), Stage(16)]
    ),
    velocity_weight=0.015,
    acceleration_weight=0.05,
)
rgb_qpos_trajectory_camera = replace(rgb_qpos_trajectory, optimize_camera=True, camera_prior_weight=0.5)

__all__ = [
    "depth",
    "rgb",
    "rgb_single_level",
    "rgb_qpos",
    "rgb_qpos_camera",
    "rgb_qpos_trajectory",
    "rgb_qpos_trajectory_camera",
    "rgbd",
]
