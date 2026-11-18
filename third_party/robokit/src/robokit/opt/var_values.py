# pyright: reportOperatorIssue=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportArgumentType=false
# pyright: reportCallIssue=false
"""Optimization variables: `Var` defines the protocol and `VarValues` combines leaves keyed by an Enum such as `VarKey`.

Leaf types:
- `VectorVar`: Euclidean vectors.
- `RobotState` and `SE3Var`: manifolds, defined with their geometry.
- `ContactVar`: discrete choices redrawn by application-specific proposals.
"""

import abc
from enum import Enum
from typing import Dict, Mapping, Optional

import warp as wp
from typing_extensions import Self

from robokit.types import ArrayLike
from robokit.utils.warp_utils import gather, masked_copy, wp_device_type


class Var(abc.ABC):
    @property
    @abc.abstractmethod
    def tangent_dim(self) -> int: ...

    @abc.abstractmethod
    def clone(self) -> Self: ...

    @abc.abstractmethod
    def view(self, batch_size: int, offset: int = 0) -> Self:
        """Elements `offset` to `offset + batch_size`, sharing storage."""
        ...

    @abc.abstractmethod
    def integrate(
        self,
        velocity: ArrayLike,
        out: Optional[Self] = None,
        tangent_mask: Optional[wp.array] = None,
        weight_decay: float = 0.0,
        out_velocity: Optional[ArrayLike] = None,
    ) -> Self:
        """Move by velocity; only shortened steps write `out_velocity`, which may alias the input velocity."""
        ...

    def difference(self, other: Self, out: Optional[ArrayLike] = None) -> ArrayLike:
        """`(batch, tangent_dim)` velocities that `integrate` moves each element by to reach `other`'s."""
        raise NotImplementedError(f"{type(self).__name__} has no difference")

    def invalidate(self):
        pass

    @property
    @abc.abstractmethod
    def batch_size(self) -> int: ...

    @property
    @abc.abstractmethod
    def device(self) -> wp_device_type: ...

    @abc.abstractmethod
    def gather(self, indices: ArrayLike, out: Optional[Self] = None) -> Self: ...

    @abc.abstractmethod
    def accept(self, accept_mask: ArrayLike, proposed: Self) -> Self:
        """Accept proposed values and derived state where accept_mask[i] == 1; leave other rows unchanged."""
        ...


class VarKey(Enum):
    """Keys of the variables robokit's tasks and helpers read; a problem with its own variables defines its own enum."""

    ROBOT = 0
    """Robot joint configuration and optional floating base (or a camera pose for hand-eye calibration)."""
    TIMING = 1
    """Shared trajectory timestep of a motion-planning population."""
    ACTION = 2
    """Control actions that MPPI samples, or that a dynamics model steps under (iLQR)."""
    CONTACTS = 3
    """Discrete contact choices of grasp optimization."""
    QPOS_OFFSET = 4
    """Joint-encoder offsets that hand-eye calibration estimates."""
    MOUNT_OFFSET = 5
    """Mount-pose offset that hand-eye calibration estimates."""
    STATE = 6
    """States a dynamics model steps from (`ILQROptimizer`'s residual terms read them beside `ACTION`)."""
    SEGMENT_STEP = 7
    """Log frame step (s) of each waypoint segment of a motion-planning candidate whose durations are optimized."""


class VarValues(Var):
    """Combine Vars into tangent-column blocks; preserve subclass step rules across copies and views."""

    def __init__(self, named_vars: Mapping[Enum, Var]):
        if not named_vars:
            raise ValueError("VarValues requires at least one variable")

        self._vars: Dict[Enum, Var] = dict(named_vars)
        self._offsets: Dict[Enum, int] = {}
        offset = 0
        for name, var in self._vars.items():
            self._offsets[name] = offset
            offset += var.tangent_dim
        self._tangent_dim = offset

        first_var = next(iter(self._vars.values()))
        self._batch_size = first_var.batch_size
        self._device = first_var.device

    @classmethod
    def from_var(cls, var: Var, key: Enum) -> "VarValues":
        return cls({key: var})

    def get(self, key: Enum) -> Var:
        return self._vars[key]

    def __contains__(self, key: Enum) -> bool:
        return key in self._vars

    def values(self):
        return self._vars.values()

    def tangent_offset(self, key: Enum) -> int:
        return self._offsets[key]

    @property
    def tangent_dim(self) -> int:
        return self._tangent_dim

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self) -> wp_device_type:
        return self._device

    def clone(self) -> "VarValues":
        cloned_vars = {name: var.clone() for name, var in self._vars.items()}
        return type(self)(cloned_vars)

    def view(self, batch_size: int, offset: int = 0) -> "VarValues":
        return type(self)({name: var.view(batch_size, offset) for name, var in self._vars.items()})

    def _get_velocity(self, name: Enum, velocity: ArrayLike):
        offset = self._offsets[name]
        dim = self._vars[name].tangent_dim
        if dim == 0:  # a zero-tangent var ignores the velocity; warp cannot slice a zero-width range
            return velocity
        if offset == 0 and dim == velocity.shape[1]:
            return velocity
        return velocity[:, offset : offset + dim]

    def _get_tangent_mask(self, name: Enum, tangent_mask: Optional[wp.array]):
        if tangent_mask is None:
            return None
        offset = self._offsets[name]
        dim = self._vars[name].tangent_dim
        if dim == 0:
            return None
        return tangent_mask[offset : offset + dim]

    def integrate(
        self,
        velocity: ArrayLike,
        out: Optional["VarValues"] = None,
        tangent_mask: Optional[wp.array] = None,
        weight_decay: float = 0.0,
        out_velocity: Optional[ArrayLike] = None,
    ) -> "VarValues":
        if out is not None:
            for name, var in self._vars.items():
                var.integrate(
                    self._get_velocity(name, velocity),
                    out=out.get(name),
                    tangent_mask=self._get_tangent_mask(name, tangent_mask),
                    weight_decay=weight_decay,
                    out_velocity=None if out_velocity is None else self._get_velocity(name, out_velocity),
                )
            return out
        integrated_vars = {}
        for name, var in self._vars.items():
            integrated_vars[name] = var.integrate(
                self._get_velocity(name, velocity),
                tangent_mask=self._get_tangent_mask(name, tangent_mask),
                weight_decay=weight_decay,
                out_velocity=None if out_velocity is None else self._get_velocity(name, out_velocity),
            )
        return type(self)(integrated_vars)

    def difference(self, other: "VarValues", out: Optional[ArrayLike] = None) -> ArrayLike:
        if out is None:
            out = wp.empty((self.batch_size, self.tangent_dim), dtype=wp.float32, device=self.device)
        for name, var in self._vars.items():
            if var.tangent_dim:
                var.difference(other.get(name), out=self._get_velocity(name, out))
        return out

    def gather(self, indices: ArrayLike, out: Optional["VarValues"] = None) -> "VarValues":
        if out is None:
            return type(self)({name: var.gather(indices) for name, var in self._vars.items()})
        for name, var in self._vars.items():
            var.gather(indices, out=out.get(name))
        return out

    def accept(self, accept_mask: ArrayLike, proposed: "VarValues") -> "VarValues":
        for name, var in self._vars.items():
            var.accept(accept_mask, proposed.get(name))
        return self

    def invalidate(self):
        for var in self._vars.values():
            var.invalidate()


@wp.kernel
def _integrate_vector_kernel(
    x: wp.array2d(dtype=wp.float32),
    velocity: wp.array2d(dtype=wp.float32),
    tangent_mask: wp.array1d(dtype=wp.float32),
    has_tangent_mask: wp.bool,
    weight_decay: float,
    write_applied: wp.bool,
    unit: float,
    max_step: float,
    out: wp.array2d(dtype=wp.float32),
    out_velocity: wp.array2d(dtype=wp.float32),
):
    batch_idx, dim_idx = wp.tid()  # pyright: ignore[reportAssignmentType]
    mask = wp.float32(1.0)
    if has_tangent_mask:
        mask = tangent_mask[dim_idx]
    move = mask * (velocity[batch_idx, dim_idx] - weight_decay * x[batch_idx, dim_idx])
    move = wp.where(max_step > 0.0, wp.clamp(unit * move, -max_step, max_step), unit * move)
    if write_applied:
        out_velocity[batch_idx, dim_idx] = move / unit
    out[batch_idx, dim_idx] = x[batch_idx, dim_idx] + move


@wp.kernel
def _difference_vector_kernel(
    x: wp.array2d(dtype=wp.float32), y: wp.array2d(dtype=wp.float32), out: wp.array2d(dtype=wp.float32)
):
    batch_idx, dim_idx = wp.tid()  # pyright: ignore[reportAssignmentType]
    out[batch_idx, dim_idx] = y[batch_idx, dim_idx] - x[batch_idx, dim_idx]


class VectorVar(Var):
    """A batched vector optimization variable on Euclidean space.

    Example:
        >>> from enum import Enum
        >>> from robokit.opt import VarValues
        >>> class Key(Enum):
        ...     X = 0
        >>> var = VectorVar(wp.array([[1.0, 2.0]], dtype=wp.float32))
        >>> values = VarValues.from_var(var, Key.X)
        >>> result = values.integrate(wp.array([[0.5, -1.0]], dtype=wp.float32))
        >>> result.get(Key.X).x.numpy().tolist()
        [[1.5, 1.0]]
        >>> capped = VectorVar(wp.zeros((1, 2), dtype=wp.float32), unit=10.0, max_step=1.0)  # moves 10x, 1 at most
        >>> capped.integrate(wp.array([[0.0625, 1.0]], dtype=wp.float32)).x.numpy().tolist()
        [[0.625, 1.0]]
    """

    def __init__(self, x: wp.array, unit: float = 1.0, max_step: float = 0.0):
        if x.ndim != 2 or x.dtype != wp.float32:
            raise ValueError("VectorVar requires a 2D float32 Warp array")
        self._x = x.contiguous()
        # a move is `unit` times the velocity (tasks scale their Jacobians by it), at most `max_step` when > 0
        self.unit, self.max_step = unit, max_step

    @property
    def x(self) -> wp.array:
        return self._x

    @x.setter
    def x(self, value: wp.array):
        if value.ndim != 2 or value.dtype != wp.float32:
            raise ValueError("VectorVar requires a 2D float32 Warp array")
        self._x = value.contiguous()

    @property
    def tangent_dim(self) -> int:
        return int(self._x.shape[1])

    @property
    def batch_size(self) -> int:
        return int(self._x.shape[0])

    @property
    def device(self) -> wp_device_type:
        return self._x.device

    def clone(self) -> "VectorVar":
        return VectorVar(wp.clone(self._x), self.unit, self.max_step)

    def view(self, batch_size: int, offset: int = 0) -> "VectorVar":
        return VectorVar(self._x[offset : offset + batch_size], self.unit, self.max_step)

    def integrate(
        self,
        velocity: ArrayLike,
        out: Optional["VectorVar"] = None,
        tangent_mask: Optional[wp.array] = None,
        weight_decay: float = 0.0,
        out_velocity: Optional[ArrayLike] = None,
    ) -> "VectorVar":
        if velocity.shape != self._x.shape:
            raise ValueError(f"Expected velocity shape {self._x.shape}, but got {velocity.shape}")
        if tangent_mask is not None and tangent_mask.shape != (self.tangent_dim,):
            raise ValueError(f"Expected tangent_mask shape {(self.tangent_dim,)}, but got {tangent_mask.shape}")
        if out is None:
            out = VectorVar(wp.empty_like(self._x, requires_grad=velocity.requires_grad or self._x.requires_grad))
            out.unit, out.max_step = self.unit, self.max_step
        wp.launch(
            _integrate_vector_kernel,
            dim=self._x.shape,
            inputs=[self._x, velocity, tangent_mask, tangent_mask is not None, weight_decay, out_velocity is not None]
            + [self.unit, self.max_step],
            outputs=[out._x, out_velocity],
            device=self.device,
        )
        return out

    def difference(self, other: "VectorVar", out: Optional[ArrayLike] = None) -> ArrayLike:
        if out is None:
            out = wp.empty_like(self._x)
        wp.launch(
            _difference_vector_kernel, dim=self._x.shape, inputs=[self._x, other._x], outputs=[out], device=self.device
        )
        return out

    def gather(self, indices: ArrayLike, out: Optional["VectorVar"] = None) -> "VectorVar":
        return VectorVar(gather(self._x, indices, out._x if out is not None else None), self.unit, self.max_step)

    def accept(self, accept_mask: ArrayLike, proposed: "VectorVar") -> "VectorVar":
        masked_copy(accept_mask, proposed._x, self._x)
        return self


__all__ = ["Var", "VarKey", "VarValues", "VectorVar"]
