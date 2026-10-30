"""Closed-form contact forces from already assembled constraint rows."""

import math

import torch
import warp as wp

from .kernels import _advance_vel, _compute_qfrc_constraint, _compute_qfrc_total


class ContactResponse:
    """No collision detection, model loading, mass solve, or integration here.

    Inputs use dense constraint rows (padding beyond nv is ignored). Parameters
    follow upstream defaults. ``mass`` is the row's impedance-weighted mass,
    not the rigid-body mass matrix. Friction rays are assembled by the engine.
    """

    def __init__(self, worlds, nv, *, device, stiffness=0.2, damping=0.001):
        if any(not math.isfinite(x) or x < 0 for x in (stiffness, damping)):
            raise ValueError("ComFree stiffness/damping must be finite and nonnegative")
        self.worlds, self.nv = worlds, nv
        self.device = device
        self.stiffness = wp.array([stiffness], dtype=float, device=device)
        self.damping = wp.array([damping], dtype=float, device=device)
        self.predicted_velocity = wp.zeros((worlds, nv), device=device)
        self.total_force = wp.zeros((worlds, nv), device=device)

    def compute(
        self,
        timestep,
        qvel,
        qacc_smooth,
        jacobian,
        distance,
        mass,
        count,
        smooth_force,
        row_force,
        constraint_force,
    ):
        """Write row/constraint forces and return total generalized force."""
        if jacobian.shape[:2] != distance.shape or mass.shape != distance.shape:
            raise ValueError("ComFree row shapes do not match")
        if jacobian.shape[0] != self.worlds or jacobian.shape[2] < self.nv:
            raise ValueError(
                "ComFree requires dense Jacobian rows with at least nv columns"
            )
        row_force.zero_()  # Inactive rows must never retain a previous response.
        wp.launch(
            _advance_vel,
            (self.worlds, self.nv),
            inputs=[timestep, qvel, qacc_smooth],
            outputs=[self.predicted_velocity, constraint_force],
            device=self.device,
        )
        wp.launch(
            _compute_qfrc_constraint,
            distance.shape,
            inputs=[
                timestep,
                self.stiffness,
                self.damping,
                jacobian,
                distance,
                mass,
                self.predicted_velocity,
                self.nv,
                count,
            ],
            outputs=[row_force, constraint_force],
            device=self.device,
        )
        wp.launch(
            _compute_qfrc_total,
            (self.worlds, self.nv),
            inputs=[smooth_force, constraint_force],
            outputs=[self.total_force],
            device=self.device,
        )
        return self.total_force


def contact_response_torch(
    timestep,
    qvel,
    qacc_smooth,
    jacobian,
    distance,
    mass,
    count,
    smooth_force,
    *,
    stiffness=0.2,
    damping=0.001,
):
    """Differentiable equivalent of :meth:`ContactResponse.compute`.

    State/force tensors have shape ``(worlds, nv)``; dense Jacobians have
    shape ``(worlds, rows, >=nv)``. ``distance`` and impedance-weighted
    ``mass`` have shape ``(worlds, rows)``, and integer ``count`` contains
    the active row count for each world. Inactive rows and Jacobian padding
    may contain stale values, including NaNs, and are ignored.

    ``timestep``, ``stiffness`` and ``damping`` accept scalars or nonempty
    one-dimensional tensors, indexed periodically by world as in the Warp
    kernels. All continuous inputs preserve their Torch autograd graph.
    The returned tuple is ``(total_force, row_force, constraint_force)``.
    Contact discovery and the active row counts remain discrete inputs.
    """
    continuous = (qvel, qacc_smooth, jacobian, distance, mass, smooth_force)
    if any(not isinstance(value, torch.Tensor) for value in continuous):
        raise TypeError("ComFree continuous inputs must be Torch tensors")
    if qvel.ndim != 2 or qvel.shape[0] == 0 or qvel.shape[1] == 0:
        raise ValueError("ComFree qvel must have shape (worlds, nv) with positive sizes")
    if not qvel.is_floating_point():
        raise TypeError("ComFree continuous inputs must have a floating-point dtype")
    if any(value.dtype != qvel.dtype or value.device != qvel.device for value in continuous):
        raise ValueError("ComFree continuous inputs must share a dtype and device")
    if qacc_smooth.shape != qvel.shape or smooth_force.shape != qvel.shape:
        raise ValueError("ComFree velocity, acceleration, and force shapes do not match")
    worlds, nv = qvel.shape
    if (
        jacobian.ndim != 3
        or jacobian.shape[0] != worlds
        or jacobian.shape[2] < nv
        or distance.shape != jacobian.shape[:2]
        or mass.shape != distance.shape
    ):
        raise ValueError("ComFree requires matching dense row shapes with at least nv columns")
    if not isinstance(count, torch.Tensor) or count.shape != (worlds,):
        raise ValueError("ComFree count must have shape (worlds,)")
    if count.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError("ComFree count must have an integer dtype")
    rows = distance.shape[1]
    if bool(torch.any((count < 0) | (count > rows))):
        raise ValueError("ComFree count must be between zero and the number of rows")

    def world_parameter(value, name, *, positive=False):
        value = torch.as_tensor(value, dtype=qvel.dtype, device=qvel.device)
        if value.ndim > 1 or value.numel() == 0:
            raise ValueError(f"ComFree {name} must be a scalar or nonempty one-dimensional tensor")
        valid = torch.isfinite(value) & ((value > 0) if positive else (value >= 0))
        if not bool(torch.all(valid)):
            condition = "positive" if positive else "nonnegative"
            raise ValueError(f"ComFree {name} must be finite and {condition}")
        indices = torch.arange(worlds, device=qvel.device) % value.numel()
        return value.reshape(-1)[indices, None]

    dt = world_parameter(timestep, "timestep", positive=True)
    k = world_parameter(stiffness, "stiffness") / dt
    d = world_parameter(damping, "damping") / dt
    active = torch.arange(rows, device=qvel.device)[None, :] < count.to(qvel.device)[:, None]
    # Mask before multiplication: zeroing a completed response would leave
    # inactive NaNs in both the forward reduction and its backward pass.
    J = torch.where(active[..., None], jacobian[..., :nv], 0.0)
    row_distance = torch.where(active, distance, 0.0)
    row_mass = torch.where(active, mass, 0.0)
    predicted_velocity = qvel + qacc_smooth * dt
    row_velocity = (J * predicted_velocity[:, None, :]).sum(dim=-1)
    penetration = row_velocity * dt + row_distance
    row_force = torch.clamp_min(row_mass * (-d * row_velocity - k * penetration), 0.0)
    constraint_force = (J * row_force[..., None]).sum(dim=1)
    return smooth_force + constraint_force, row_force, constraint_force
