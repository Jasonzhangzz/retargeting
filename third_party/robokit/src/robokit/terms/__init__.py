from robokit.terms.composite_task import CompositeTask
from robokit.terms.dense.com_position_task import ComPositionTask
from robokit.terms.dense.configuration_task import ConfigurationTask
from robokit.terms.dense.force_closure_task import ForceClosureTask
from robokit.terms.dense.frame_task import FrameTask
from robokit.terms.dense.frame_vector_task import FrameVectorTask
from robokit.terms.dense.interaction_mesh_task import InteractionMeshTask
from robokit.terms.dense.manipulability_task import ManipulabilityTask
from robokit.terms.dense.pixel_task import PixelTask
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.position_task import PositionTask
from robokit.terms.dense.rotation_task import AxisLimitTask, RotationTask
from robokit.terms.dense.scene_collision_task import SceneCollisionTask
from robokit.terms.dense.scene_distance_task import SceneDistanceTask
from robokit.terms.dense.self_collision_task import SelfCollisionTask
from robokit.terms.dense.smoothness_task import SmoothnessTask
from robokit.terms.dense.velocity_limit_task import VelocityLimitTask
from robokit.terms.robot_task import RobotTask
from robokit.terms.simulation import (
    AlignTask,
    ContactTask,
    ControlTask,
    GraspTask,
    PinchTask,
    SensorTask,
    SimContext,
    StateTask,
    SupportTask,
)
from robokit.terms.sparse.trajectory_collision_task import TrajectoryCollisionTask
from robokit.terms.sparse.trajectory_limit_task import TrajectoryLimitTask
from robokit.terms.sparse.trajectory_position_task import TrajectoryContactTask, TrajectoryPositionTask
from robokit.terms.sparse.trajectory_retargeting_task import TrajectoryRetargetingTask
from robokit.terms.sparse.trajectory_self_collision_task import TrajectorySelfCollisionTask
from robokit.terms.sparse.trajectory_smoothness_task import TrajectorySmoothnessTask
from robokit.terms.task import EagerTask, GradientTask, ResidualTask, SparseTask, Task
from robokit.terms.trajectory_task import TrajectoryTask


__all__ = [
    # protocols
    "Task",
    "ResidualTask",
    "GradientTask",
    "SparseTask",
    "EagerTask",
    "RobotTask",
    "TrajectoryTask",
    # dense tasks
    "AxisLimitTask",
    "ComPositionTask",
    "CompositeTask",
    "ForceClosureTask",
    "FrameTask",
    "FrameVectorTask",
    "InteractionMeshTask",
    "ManipulabilityTask",
    "PositionLimit",
    "PixelTask",
    "PositionTask",
    "ConfigurationTask",
    "RotationTask",
    "SceneCollisionTask",
    "SceneDistanceTask",
    "SelfCollisionTask",
    "SmoothnessTask",
    "VelocityLimitTask",
    # trajectory (sparse) tasks
    "TrajectoryCollisionTask",
    "TrajectoryContactTask",
    "TrajectoryPositionTask",
    "TrajectoryRetargetingTask",
    "TrajectorySelfCollisionTask",
    "TrajectorySmoothnessTask",
    "TrajectoryLimitTask",
    # simulation tasks
    "SimContext",
    "AlignTask",
    "ContactTask",
    "ControlTask",
    "GraspTask",
    "PinchTask",
    "SensorTask",
    "StateTask",
    "SupportTask",
]
