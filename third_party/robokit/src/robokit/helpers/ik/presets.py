"""Robot-free IK preset configs build with `IKSolver(presets.basic, robot, target_links)`."""

from robokit.helpers.ik.config import IKContext, IKSolverConfig
from robokit.opt.lm_optimizer import LMOptimizerConfig
from robokit.opt.population_solver import PopulationSolverConfig, Stage
from robokit.opt.sampler import RobertsSamplerConfig
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.position_task import PositionTask
from robokit.terms.dense.rotation_task import RotationTask
from robokit.terms.dense.smoothness_task import SmoothnessTask


basic = (
    IKSolverConfig(sampler=RobertsSamplerConfig(keep_first=True, scale=1.0))
    .add(PositionTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=20.0))
    .add(RotationTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=10.0))
    .add(PositionLimit(weight=50.0))
)

smooth = (
    IKSolverConfig(sampler=RobertsSamplerConfig(keep_first=True, scale=1.0))
    .add(PositionTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=20.0))
    .add(RotationTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=10.0))
    .add(PositionLimit(weight=50.0))
    .add(SmoothnessTask(prev_var=IKContext.PREVIOUS_STATE, weight=0.1))
)

single_candidate = (
    IKSolverConfig(
        solver=PopulationSolverConfig(stages=[Stage(30, 1, LMOptimizerConfig(lm_lambda=1.0))], cuda_graph_mode="full"),
    )
    .add(PositionTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=20.0))
    .add(RotationTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=10.0))
    .add(PositionLimit(weight=50.0))
    .add(SmoothnessTask(prev_var=IKContext.PREVIOUS_STATE, weight=0.1))
)


__all__ = ["basic", "single_candidate", "smooth"]
