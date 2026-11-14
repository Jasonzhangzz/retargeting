from dataclasses import dataclass
from typing import Optional

import numpy as np
import warp as wp

from robokit.terms.task import Context


@dataclass
class PlanGoal:
    """What every planning step solves towards; one row per query unless noted."""

    start_q: wp.array  # (batch, dofs)
    T_world_target: Optional[wp.array]  # (batch,) pose goals, or None for joint goals
    target_q: Optional[wp.array]  # (batch, dofs) joint goals, or None
    target_q_per_candidate: wp.array  # (batch * candidates, dofs): each candidate's joint goal
    scene_indices: wp.array  # (batch,)
    q_history: Optional[wp.array] = None  # (batch, 3, dofs): the last executed configurations, online only
    history_count: int = 0
    active_joint_mask: Optional[np.ndarray] = None  # (dofs,): zero pins a joint; None moves all
    start_T_world_base: Optional[wp.array] = None  # (batch,) a floating base's start poses


class MotionPlanContext(Context):
    """Values a trajectory stage gives its task templates; arrays are buffers refilled for each query."""

    TARGET_LINKS = 0
    """Indices of the planned links."""
    TARGET = 1
    """Pose goals of the planned links, `(batch * candidates, links)` vec7."""
    START_Q = 2
    """Start configurations, `(batch * candidates, dofs)`."""
    GOAL_Q = 3
    """Each candidate's joint goal, `(batch * candidates, dofs)`."""
    STEP = 4
    """The stage's timestep."""
    QUERY_STEP = 5
    """Each query's step `(batch,)`: the planner's, scaled between stages; gradient stages only."""
    HISTORY = 6
    """Executed-state prefix `(q_history, history_valid)` of online plans; gradient stages only."""
    VELOCITY_LIMITS = 7
    """Joint velocity limits, `(dofs,)`."""
    ACCELERATION_LIMIT = 8
    """The plan's joint acceleration limit (rad/s²); gradient stages only."""
    JERK_LIMIT = 9
    """The plan's joint jerk limit (rad/s³); gradient stages only."""
    SCENE = 10
    """The planner's collision scene."""
    SCENE_INDICES = 11
    """Each candidate's scene, `(batch * candidates,)`."""
    EE_LINK = 12
    """The first planned link, which MPPI settles; MPPI only."""
    LINK_TARGETS = 13
    """Goal positions of the first planned link for every draw, `(rows, frames, 3)`; MPPI only."""
    GOAL_FRAMES = 14
    """Frames a pose goal is pursued at: the last, or none when the stage locks its endpoints; gradient stages only."""
    GOAL_DISTANCE = 15
    """Each candidate's start-to-goal joint distance `(batch * candidates,)`; gradient stages only."""
    CANDIDATES = 16
    """Candidates per query."""
    CANDIDATE_COSTS = 17
    """A callable returning each candidate's cost after a solve, for the score."""
    COLLISION_TASKS = 18
    """The built tasks that report penetration, for the score."""
    RETIME = 19
    """The planner's `RetimeConfig` for this stage, or None; the score samples plans at the step it gives."""
    DRAWS = 20
    """Draws each candidate holds in a ranked population; MPPI only."""
    SPHERE_MASK = 21
    """Each frame's checked spheres `(frames, spheres)` int32 (0 drops a sphere there, e.g. an attached object), or None."""


__all__ = ["MotionPlanContext", "PlanGoal"]
