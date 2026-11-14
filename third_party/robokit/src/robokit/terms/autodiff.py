"""Ground-truth Jacobians from Warp reverse-mode autodiff."""

import weakref
from typing import Any, Optional

import warp as wp

from robokit.opt.var_values import VarValues
from robokit.terms.task import ResidualTask


# --- kernels ----------------------------------------------------------------
@wp.kernel
def _set_seed_kernel(seed_array: wp.array2d(dtype=wp.float32), col_idx: int, value: float):
    i = wp.tid()
    seed_array[i, col_idx] = value


@wp.kernel
def _set_jacobian_row_kernel(
    grad: wp.array2d(dtype=wp.float32),
    jacobian: wp.array3d(dtype=wp.float32),
    row_idx: int,
    row_offset: int,
    col_offset: int,
):
    i, j = wp.tid()  # type: ignore[misc]
    jacobian[i, row_offset + row_idx, col_offset + j] = grad[i, j]


@wp.kernel
def _set_diagonal_jacobian_kernel(
    grad: wp.array2d(dtype=wp.float32),
    jacobian: wp.array3d(dtype=wp.float32),
    residual_dim: int,
    row_offset: int,
    col_offset: int,
):
    batch_idx, dof_idx = wp.tid()  # pyright: ignore
    if dof_idx < residual_dim:
        jacobian[batch_idx, row_offset + dof_idx, col_offset + dof_idx] = grad[batch_idx, dof_idx]


# --- public API -------------------------------------------------------------
# key by `id(task)` because task objects may be unhashable; the finalizer releases cached GPU buffers
_STATE: dict = {}


def autodiff_weighted_jacobian(
    task: ResidualTask,
    var_values: VarValues,
    *args: Any,
    out_jacobian: Optional[wp.array] = None,
    row_offset: int = 0,
    col_offset: int = 0,
    diagonal: bool = False,
    **kwargs: Any,
) -> wp.array:
    """Compute a weighted Jacobian with reverse-mode autodiff.

    Lifecycle:
        1. Build and cache the tape, perturbation, residual, and seed buffers.
        2. Record the task residual under a tangent perturbation.
        3. Run one backward pass per row, or one pass for a diagonal Jacobian.

    Args:
        task: Residual task to differentiate.
        var_values: Values at which to evaluate the Jacobian.
        *args: Positional arguments forwarded to the residual method.
        out_jacobian: Optional output buffer.
        row_offset: First output row.
        col_offset: First output column.
        diagonal: Whether the residual Jacobian is diagonal.
        **kwargs: Keyword arguments forwarded to the residual method.

    Returns:
        The weighted Jacobian.
    """
    # --- build buffers ---
    state = _STATE.get(id(task))
    if state is None:
        batch_size = var_values.batch_size
        tangent_dim = var_values.tangent_dim
        device = var_values.device
        velocity = wp.zeros((batch_size, tangent_dim), dtype=wp.float32, device=device, requires_grad=True)
        state = {
            "tape": wp.Tape(),
            "velocity": velocity,
            "proposed_var": var_values.integrate(velocity),
            "out_jacobian": wp.zeros((batch_size, task.residual_dim, tangent_dim), dtype=wp.float32, device=device),
        }
        if diagonal:
            state["out_residual"] = wp.zeros((batch_size, task.residual_dim), dtype=wp.float32, device=device)
            state["diag_seed"] = wp.ones((batch_size, task.residual_dim), dtype=wp.float32, device=device)
        else:
            state["out_residual"] = wp.zeros(
                (batch_size, task.residual_dim), dtype=wp.float32, device=device, requires_grad=True
            )
            seeds = []
            for k in range(task.residual_dim):
                seed = wp.zeros((batch_size, task.residual_dim), dtype=wp.float32, device=device)
                wp.launch(kernel=_set_seed_kernel, dim=batch_size, inputs=[seed, k, 1.0], device=device)
                seeds.append(seed)
            state["seeds"] = seeds
        _STATE[id(task)] = state
        weakref.finalize(task, _STATE.pop, id(task), None)

    tape = state["tape"]
    velocity = state["velocity"]
    if out_jacobian is None:
        out_jacobian = state["out_jacobian"]
    assert out_jacobian is not None

    batch_size = var_values.batch_size
    tangent_dim = var_values.tangent_dim
    device = var_values.device

    # --- record residual ---
    tape.reset()
    tape.gradients = {}
    with tape:
        new_var = var_values.integrate(velocity, out=state["proposed_var"])
        residual = task.compute_weighted_residual(new_var, *args, out_residual=state["out_residual"], **kwargs)

    # --- run backward passes ---
    if diagonal:
        tape.backward(grads={residual: state["diag_seed"]})
        grad = velocity.grad
        wp.launch(
            kernel=_set_diagonal_jacobian_kernel,
            dim=(batch_size, tangent_dim),
            inputs=[grad, out_jacobian, task.residual_dim, row_offset, col_offset],
            device=device,
        )
        return out_jacobian

    for k in range(task.residual_dim):
        residual.grad.assign(state["seeds"][k])
        tape.backward()
        grad = velocity.grad
        wp.launch(
            kernel=_set_jacobian_row_kernel,
            dim=(batch_size, tangent_dim),
            inputs=[grad, out_jacobian, k, row_offset, col_offset],
            device=device,
        )
        tape.zero()
    return out_jacobian
