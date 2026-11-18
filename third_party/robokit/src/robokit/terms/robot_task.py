from enum import Enum
from typing import TYPE_CHECKING, Any, Mapping, Optional, cast

from robokit.opt.var_values import VarValues
from robokit.robo.robot_state import RobotState
from robokit.terms.task import Context, Task


if TYPE_CHECKING:
    from robokit.robo.robot import Robot


class RobotTask(Task):
    """Serial preparation shared by terms that consume a robot state."""

    robot: Optional["Robot"]
    var_key: Enum
    precompute_collision_geometry: str = ""

    def resolve(self, var_values: VarValues, context: Mapping[Context, Any]):
        """Swap placeholders, then take the robot of the state this task reads."""
        super().resolve(var_values, context)
        self.set_robot(cast(RobotState, var_values.get(self.var_key)).robot)

    def set_robot(self, robot: "Robot"):
        self.robot = robot

    def precompute(self, var_values: VarValues, need_gradient: bool = False):
        """Prepare kinematics and geometry required by the task."""
        robot = cast("Robot", self.robot)
        state = cast(RobotState, var_values.get(self.var_key))
        if not state.is_fk_computed:
            robot.forward_kinematics(state)
        if self.precompute_collision_geometry == "sphere" and not state.is_collision_spheres_computed:
            robot.transform_collision_spheres(state)
        if self.precompute_collision_geometry == "capsule" and not state.is_capsules_computed:
            robot.transform_collision_capsules(state)
        if need_gradient and not state.is_motion_subspace_computed:
            robot.compute_motion_subspace(state)
