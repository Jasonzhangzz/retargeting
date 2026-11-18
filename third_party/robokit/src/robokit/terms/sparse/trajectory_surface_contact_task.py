"""Masked trajectory contact between selected robot spheres and a scene surface."""

from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, Union

import numpy as np
import warp as wp

from robokit.terms.sparse.trajectory_collision_task import TrajectoryCollisionTask
from robokit.terms.task import Context


if TYPE_CHECKING:
    from robokit.geom import BaseGeom, WarpScene
    from robokit.opt.var_values import VarValues
    from robokit.robo import Robot


class TrajectorySurfaceContactTask(TrajectoryCollisionTask):
    """Pull selected spheres to the surface with one residual per frame/sphere and a fixed `contact_mask` pattern.

    Scene mappings:
    - Per batch item: one static scene.
    - Per `(batch, frame)`: a moving scene.
    """

    def __init__(
        self,
        robot: Optional["Robot"] = None,  # noqa: F821
        scene: Optional[Union["WarpScene", Context]] = None,  # noqa: F821
        num_frames: Optional[int] = None,
        sphere_indices: Optional[Union[Sequence[int], np.ndarray, Context]] = None,
        weight: Union[float, Sequence[float]] = 100.0,
        contact_mask: Optional[Union[np.ndarray, wp.array, Context]] = None,
        scene_indices: Optional[Union[wp.array, Context]] = None,
        active_actuated_indices: Optional[Union[Sequence[int], np.ndarray]] = None,
        geometry: Optional["BaseGeom"] = None,  # noqa: F821
        name: str = "",
    ) -> None:
        super().__init__(
            robot=robot,
            scene=scene,
            num_frames=num_frames,
            weight=weight,
            margin=0.0,
            sphere_indices=sphere_indices,
            scene_indices=scene_indices,
            active_actuated_indices=active_actuated_indices,
            penalty="surface_distance",
            geometry=geometry,
            name=name,
            frame_sphere_mask=contact_mask,
        )

    def resolve(self, var_values: "VarValues", context: Mapping[Context, Any]):  # noqa: F821
        """Build like a collision task; a weighted surface contact needs a scene."""
        super().resolve(var_values, context)
        if np.any(self.weight) and self._scene is None:
            raise ValueError("surface contacts require a scene.")

    def set_contact_mask(self, contact_mask: wp.array) -> None:
        """Set active contacts shaped `(batch, frames, selected spheres)`."""
        self.set_frame_sphere_mask(contact_mask)


__all__ = ["TrajectorySurfaceContactTask"]
