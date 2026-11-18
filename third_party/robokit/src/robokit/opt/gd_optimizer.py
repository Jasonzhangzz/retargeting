# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportCallIssue=false
# pyright: reportOperatorIssue=false
# pyright: reportGeneralTypeIssues=false
import math
from dataclasses import dataclass
from typing import Callable, Literal, Optional, Sequence

import warp as wp

from robokit.opt.optimizer import Optimizer, OptimizerConfig, aggregate_residuals_to_costs_kernel
from robokit.opt.var_values import VarValues
from robokit.terms.task import GradientTask, Task
from robokit.utils.warp_utils import seed_rand_states_kernel, wp_device_type


@wp.kernel
def _gd_update_kernel(
    gradient: wp.array2d(dtype=wp.float32),
    learning_rate: wp.array1d(dtype=wp.float32),
    gradient_clip: float,
    velocity: wp.array2d(dtype=wp.float32),
):
    batch_idx, dof_idx = wp.tid()  # pyright: ignore[reportAssignmentType]
    velocity[batch_idx, dof_idx] = -learning_rate[0] * wp.clamp(
        gradient[batch_idx, dof_idx], -gradient_clip, gradient_clip
    )


@wp.kernel
def _rmsprop_velocity_step_kernel(
    gradient: wp.array2d(dtype=wp.float32),
    ema: wp.array2d(dtype=wp.float32),
    velocity: wp.array2d(dtype=wp.float32),
    step_buffer: wp.array1d(dtype=wp.int32),
    temperature_decay: wp.float32,
    stepsize_period: wp.int32,
    mu: wp.float32,
    base_step_size: wp.float32,
    eps: wp.float32,
):
    batch_idx, d = wp.tid()  # pyright: ignore[reportAssignmentType]
    step = step_buffer[0]
    decay_power = step // stepsize_period
    step_size = base_step_size * wp.pow(temperature_decay, wp.float32(decay_power))
    g = gradient[batch_idx, d]
    new_ema = mu * g * g + (wp.float32(1.0) - mu) * ema[batch_idx, d]
    ema[batch_idx, d] = new_ema
    velocity[batch_idx, d] = -step_size * g / (wp.sqrt(new_ema) + eps)


@wp.kernel
def _metropolis_accept_step_kernel(
    costs: wp.array1d(dtype=wp.float32),
    proposed_costs: wp.array1d(dtype=wp.float32),
    step_buffer: wp.array1d(dtype=wp.int32),
    starting_temperature: wp.float32,
    temperature_decay: wp.float32,
    annealing_period: wp.int32,
    rand_states: wp.array1d(dtype=wp.uint32),
    accept_mask: wp.array1d(dtype=wp.int32),
):
    tid = wp.tid()
    step = step_buffer[0]
    old_cost = costs[tid]
    new_cost = proposed_costs[tid]
    state = rand_states[tid]
    rand_val = wp.randf(state)
    rand_states[tid] = state
    decay_power = step // annealing_period
    temperature = starting_temperature * wp.pow(temperature_decay, wp.float32(decay_power))
    delta = old_cost - new_cost
    accept = wp.int32(0)
    if delta >= wp.float32(0.0):
        accept = wp.int32(1)
    else:
        threshold = wp.exp(delta / temperature)
        if rand_val < threshold:
            accept = wp.int32(1)
    accept_mask[tid] = accept
    if accept == wp.int32(1):
        costs[tid] = new_cost


@wp.kernel
def _accumulate_jtr_gradient_kernel(
    jacobian: wp.array3d(dtype=wp.float32),
    residuals: wp.array2d(dtype=wp.float32),
    row_offset: int,
    gradient: wp.array2d(dtype=wp.float32),
):
    """Add one term's `J^T r` into the gradient, for kernels the tape cannot differentiate."""
    instance, dof = wp.tid()  # pyright: ignore[reportAssignmentType]
    acc = wp.float32(0.0)
    for k in range(jacobian.shape[1]):
        acc += jacobian[instance, k, dof] * residuals[instance, row_offset + k]
    gradient[instance, dof] = gradient[instance, dof] + acc


@wp.kernel
def _increment_step_kernel(step_buffer: wp.array1d(dtype=wp.int32)):
    step_buffer[0] = step_buffer[0] + 1


@wp.kernel
def _accept_gradient_kernel(
    accept_mask: wp.array1d(dtype=wp.int32),
    accepted_gradient: wp.array2d(dtype=wp.float32),
    step_buffer: wp.array1d(dtype=wp.int32),
    gradient: wp.array2d(dtype=wp.float32),
):
    tid = wp.tid()
    if accept_mask[tid] == 0:  # a rejected element keeps the gradient of the point it stays at
        for d in range(gradient.shape[1]):
            gradient[tid, d] = accepted_gradient[tid, d]
    if tid == 0:
        step_buffer[0] = step_buffer[0] + 1


@wp.kernel
def _adam_update_kernel(
    gradient: wp.array2d(dtype=wp.float32),
    m: wp.array2d(dtype=wp.float32),
    v: wp.array2d(dtype=wp.float32),
    velocity: wp.array2d(dtype=wp.float32),
    step_buffer: wp.array1d(dtype=wp.int32),
    lr: wp.array1d(dtype=wp.float32),
    gradient_clip: float,
    beta1: float,
    beta2: float,
    epsilon: float,
):
    # The step count lives on device: a captured iteration replays without the host, so it cannot be counted there.
    batch_idx, dof_idx = wp.tid()  # pyright: ignore[reportAssignmentType]
    step = wp.float32(step_buffer[0] + 1)
    bias_correction1 = wp.float32(1.0) - wp.pow(beta1, step)
    bias_correction2 = wp.float32(1.0) - wp.pow(beta2, step)
    g = wp.clamp(gradient[batch_idx, dof_idx], -gradient_clip, gradient_clip)
    m_new = beta1 * m[batch_idx, dof_idx] + (1.0 - beta1) * g
    v_new = beta2 * v[batch_idx, dof_idx] + (1.0 - beta2) * g * g
    m[batch_idx, dof_idx] = m_new
    v[batch_idx, dof_idx] = v_new
    m_hat = m_new / bias_correction1
    v_hat = v_new / bias_correction2
    velocity[batch_idx, dof_idx] = -lr[0] * m_hat / (wp.sqrt(v_hat) + epsilon)


@dataclass(frozen=True)
class GDOptimizerConfig(OptimizerConfig):
    learning_rate: float = 0.01
    """Step size."""
    lr_schedule: Optional[Callable[[int, float], float]] = None
    """Learning rate per iteration from (zero-based iteration, `learning_rate`); None keeps it constant."""
    gradient_mode: Literal["analytic_jacobian", "autodiff", "analytic_gradient"] = "analytic_jacobian"
    """analytic_jacobian: terms fill a shared sparse/dense J, then one Jᵀr reduction; autodiff: tape gradients;
    analytic_gradient: term kernels accumulate cost and gradient directly."""
    optimizer_type: Literal["gd", "adam"] = "gd"
    """Update rule: plain gradient descent or Adam."""
    gradient_clip: float = math.inf
    """Clamp each gradient entry to ±this before the update, so a stiff cost cannot overshoot."""
    beta1: float = 0.9
    """Adam's first-moment decay."""
    beta2: float = 0.999
    """Adam's second-moment decay."""
    epsilon: float = 1e-8
    """Adam's denominator stabilizer."""
    weight_decay: float = 0.0
    """Decoupled decay: each step also moves x by -learning_rate * weight_decay * x (AdamW under Adam)."""

    def build(
        self, terms: Sequence[Task], device: wp_device_type, active_dof_mask: Optional[wp.array] = None
    ) -> "GDOptimizer":
        return GDOptimizer(terms, self, device, active_dof_mask)


class GDOptimizer(Optimizer):
    """First-order optimizer supporting GD and Adam, with optional decoupled weight decay (AdamW)."""

    _config_type = GDOptimizerConfig

    def _build(self, var: VarValues):
        self._build_gradient(var)
        config, shape, device = self.config, (self.max_batch_size, self.total_tangent_dim), self.device
        self._optimizer_type = config.optimizer_type
        if config.optimizer_type == "adam":
            self.m = wp.zeros(shape, dtype=wp.float32, device=device)
            self.v = wp.zeros(shape, dtype=wp.float32, device=device)
            self._adam_step = wp.zeros(1, dtype=wp.int32, device=device)
        self._lr_iter, self._weight_decay = 0, 0.0
        # The learning rate lives on device: `step` writes the schedule's value before a captured iteration replays.
        self._learning_rate = wp.full((1,), config.learning_rate, dtype=wp.float32, device=device)
        if config.lr_schedule is not None and config.cuda_graph_mode == "full":
            raise ValueError("lr_schedule needs cuda_graph_mode 'iter' or 'none': a whole-solve graph freezes it")

    def _build_gradient(self, var: VarValues):
        """Cost, gradient and step buffers for `gradient_mode`, shared with `AnnealedGDOptimizer`."""
        config = self.config
        terms = self.terms
        device = self.device
        self._build_terms(var, residuals_require_grad=(config.gradient_mode == "autodiff"))

        batch_size = self.max_batch_size
        total_tangent_dim = self.total_tangent_dim
        self.gradient = wp.zeros((batch_size, total_tangent_dim), dtype=wp.float32, device=device)
        self.velocity = wp.zeros((batch_size, total_tangent_dim), dtype=wp.float32, device=device)
        if config.gradient_mode == "autodiff":
            self._autodiff_tape = wp.Tape()
            self._autodiff_velocity = wp.zeros(
                (batch_size, total_tangent_dim), dtype=wp.float32, device=device, requires_grad=True
            )
            self._autodiff_proposed_var = var.integrate(self._autodiff_velocity)
            # terms without a tape adjoint contribute their analytic `J^T r` instead
            self._analytic_terms = [i for i, t in enumerate(terms) if not t.SUPPORTS_AUTODIFF]
            self._analytic_jacobians = [
                wp.zeros((batch_size, terms[i].residual_dim, total_tangent_dim), dtype=wp.float32, device=device)
                for i in self._analytic_terms
            ]
        elif config.gradient_mode == "analytic_jacobian":
            # GD has no expanded line-search batch, so the Jacobian batch is batch_size
            self._build_jacobian(var, self.max_batch_size)
        else:  # analytic_gradient; the closures capture buffers, not `self` (see _compute_cost_and_gradient)
            costs, gradient = self.costs, self.gradient
            self._cost_grad_fns: list = []
            for t in terms:
                assert isinstance(t, GradientTask)
                fn = t.compute_weighted_cost_and_gradient
                self._cost_grad_fns.append(lambda offset, var, _fn=fn: _fn(var, out_cost=costs, out_gradient=gradient))

    def _compute_cost_and_gradient(self, var: VarValues) -> None:
        """Dispatch `gradient_mode` to fill costs and gradients without storing `self` methods or creating GC cycles."""
        if self.config.gradient_mode == "autodiff":
            self._compute_cost_and_gradient_autodiff(var)
        elif self.config.gradient_mode == "analytic_jacobian":
            self._compute_cost_and_gradient_jacobian(var)
        else:
            self._compute_cost_and_gradient_direct(var)

    def _compute_cost_and_gradient_direct(self, var: VarValues):
        """analytic_gradient mode: each term's kernel atomic-adds cost+gradient into shared buffers."""
        self._precompute_terms(var, need_gradient=True)
        self.gradient.zero_()
        self.costs.zero_()
        self._parallel_for_objectives(self._cost_grad_fns, var)
        # Mask the gradient; m/v starting at zero keeps Adam/AdamW momentum zero on locked dims too.
        self._mask_gradient(self.gradient)

    def _compute_cost_and_gradient_autodiff(self, var: VarValues):
        """Autodiff mode: tape forward -> backward to get gradient directly (no Jacobian)."""
        self._autodiff_tape.reset()
        self._autodiff_tape.gradients = {}
        self._autodiff_velocity.zero_()

        with self._autodiff_tape:
            proposed = var.integrate(self._autodiff_velocity, out=self._autodiff_proposed_var)
            self._precompute_terms(proposed, need_gradient=False)
            for offset, fn in zip(self.residual_offsets, self._residual_fns):
                fn(offset, proposed, self.residuals)

        wp.launch(
            kernel=aggregate_residuals_to_costs_kernel,
            dim=self.batch_size,
            inputs=[self.residuals, self.costs],
            device=self.device,
        )
        self._autodiff_tape.backward(grads={self.residuals: self.residuals})
        wp.copy(self.gradient, self._autodiff_velocity.grad)
        self._autodiff_tape.zero()
        for term_idx, jacobian in zip(self._analytic_terms, self._analytic_jacobians):
            jacobian = jacobian[: self.batch_size]
            jacobian.zero_()
            self.terms[term_idx].compute_weighted_jacobian_analytic(proposed, out_jacobian=jacobian)
            wp.launch(
                _accumulate_jtr_gradient_kernel,
                dim=(self.batch_size, self.total_tangent_dim),
                inputs=[jacobian, self.residuals, self.residual_offsets[term_idx]],
                outputs=[self.gradient],
                device=self.device,
            )

    def step(self, var: VarValues) -> None:
        self._set_learning_rate()
        super().step(var)

    def _set_learning_rate(self) -> None:
        cfg = self.config
        lr = cfg.lr_schedule(self._lr_iter, cfg.learning_rate) if cfg.lr_schedule is not None else cfg.learning_rate
        self._lr_iter += 1
        if cfg.lr_schedule is not None:
            self._learning_rate.fill_(lr)
        self._weight_decay = lr * cfg.weight_decay

    def _compute_velocity(self):
        cfg = self.config
        dim = (self.batch_size, self.total_tangent_dim)

        if self._optimizer_type == "gd":
            wp.launch(
                kernel=_gd_update_kernel,
                dim=dim,
                inputs=[self.gradient, self._learning_rate, cfg.gradient_clip],
                outputs=[self.velocity],
                device=self.device,
            )
        else:
            wp.launch(
                kernel=_adam_update_kernel,
                dim=dim,
                inputs=[
                    self.gradient,
                    self.m,
                    self.v,
                    self.velocity,
                    self._adam_step,
                    self._learning_rate,
                    cfg.gradient_clip,
                    cfg.beta1,
                    cfg.beta2,
                    cfg.epsilon,
                ],
                device=self.device,
            )
            wp.launch(_increment_step_kernel, dim=1, inputs=[self._adam_step], device=self.device)

    _BATCH_BUFFERS = ("residuals", "costs", "gradient", "velocity", "m", "v", "_autodiff_velocity")

    def _start(self, var: VarValues) -> None:
        # Schedule and Adam state restart per solve, else a captured graph bakes in the tail schedule.
        self._lr_iter = 0
        if self._optimizer_type == "adam":
            self._adam_step.zero_()
            self.m.zero_()
            self.v.zero_()
        self._compute_cost_and_gradient(var)

    def _step_body(self, var: VarValues) -> None:
        """One iteration: step along the current gradient, then score the new point."""
        self._compute_velocity()
        # in-place: every integrate kernel is elementwise on the same index, so out=var aliases safely
        var.integrate(self.velocity, out=var, tangent_mask=self._active_dof_mask, weight_decay=self._weight_decay)
        var.invalidate()
        self._compute_cost_and_gradient(var)


# --- simulated annealing --------------------------------------------------------
@dataclass(frozen=True)
class AnnealedGDConfig(OptimizerConfig):
    """`learning_rate` and the Metropolis temperature both decay by `temperature_decay`, on their own periods."""

    learning_rate: float = 0.01
    """Initial RMSProp step size."""
    gradient_mode: Literal["analytic_jacobian", "autodiff", "analytic_gradient"] = "analytic_jacobian"
    """How the gradient is computed, as in `GDOptimizerConfig`."""
    mu: float = 0.98
    """RMSProp's weight on the newest squared gradient."""
    stepsize_period: int = 50
    """Iterations between step-size decays."""
    starting_temperature: float = 18.0
    """Initial Metropolis temperature."""
    temperature_decay: float = 0.95
    """Factor applied to the step size and the temperature at each of their periods."""
    annealing_period: int = 30
    """Iterations between temperature decays."""
    grad_eps: float = 1e-6
    """RMSProp's denominator stabilizer."""
    seed: int = 0
    """Seed of the Metropolis acceptance draws."""


class AnnealedGDOptimizer(GDOptimizer):
    """GD with RMSProp proposals and Metropolis acceptance, supporting every base-class gradient mode.

    Acceptance outcomes:
    - Accept: keep the proposed point and gradient.
    - Reject: restore the previous point and gradient, continuing the chain there.
    """

    _config_type = AnnealedGDConfig

    def _build(self, var: VarValues) -> None:
        self._build_gradient(var)
        batch_size, tangent_dim, device = self.max_batch_size, self.total_tangent_dim, self.device
        self.sq_grad_ema = wp.zeros((batch_size, tangent_dim), dtype=wp.float32, device=device)
        self._accepted_gradient = wp.zeros_like(self.gradient)
        self._proposed = var.clone()
        self._accepted_costs = wp.zeros((batch_size,), dtype=wp.float32, device=device)
        self._accept_mask = wp.zeros((batch_size,), dtype=wp.int32, device=device)
        self._accept_rand_states = wp.zeros((batch_size,), dtype=wp.uint32, device=device)
        self._step_buffer = wp.zeros((1,), dtype=wp.int32, device=device)

    _BATCH_BUFFERS = GDOptimizer._BATCH_BUFFERS + (
        "sq_grad_ema",
        "_accepted_gradient",
        "_proposed",
        "_accepted_costs",
        "_accept_mask",
        "_accept_rand_states",
    )

    def _compute_velocity(self) -> None:
        """RMSProp, with the step size decayed on `stepsize_period` like the temperature is on its own period."""
        cfg = self.config
        wp.launch(
            _rmsprop_velocity_step_kernel,
            dim=(self.batch_size, self.total_tangent_dim),
            inputs=[
                self.gradient,
                self.sq_grad_ema,
                self.velocity,
                self._step_buffer,
                wp.float32(cfg.temperature_decay),
                wp.int32(cfg.stepsize_period),
                wp.float32(cfg.mu),
                wp.float32(cfg.learning_rate),
                wp.float32(cfg.grad_eps),
            ],
            device=self.device,
        )

    def step(self, var: VarValues) -> None:
        Optimizer.step(self, var)  # no learning-rate schedule: the RMSProp kernel decays its own step size

    def _start(self, var: VarValues) -> None:
        self.sq_grad_ema.zero_()
        self.gradient.zero_()
        self._step_buffer.zero_()
        wp.launch(
            seed_rand_states_kernel,
            dim=self.batch_size,
            inputs=[self.config.seed, self._accept_rand_states],
            device=self.device,
        )
        self._compute_cost_and_gradient(var)

    def _step_body(self, var: VarValues) -> None:
        """Propose an RMSProp step on a copy, then keep or revert it (and its gradient) by Metropolis."""
        cfg = self.config
        wp.copy(self._accepted_costs, self.costs)  # `self.costs` holds the accepted point's cost on entry
        self._compute_velocity()
        wp.copy(self._accepted_gradient, self.gradient)
        var.integrate(self.velocity, out=self._proposed, tangent_mask=self._active_dof_mask)
        self._proposed.invalidate()
        # `_compute_cost_and_gradient` writes `self.costs`, so it now holds the *proposed* cost.
        self._compute_cost_and_gradient(self._proposed)
        wp.launch(
            _metropolis_accept_step_kernel,
            dim=self.batch_size,
            inputs=[
                self._accepted_costs,
                self.costs,
                self._step_buffer,
                wp.float32(cfg.starting_temperature),
                wp.float32(cfg.temperature_decay),
                wp.int32(cfg.annealing_period),
                self._accept_rand_states,
                self._accept_mask,
            ],
            device=self.device,
        )
        var.accept(self._accept_mask, self._proposed)
        wp.launch(
            _accept_gradient_kernel,
            dim=self.batch_size,
            inputs=[self._accept_mask, self._accepted_gradient, self._step_buffer, self.gradient],
            device=self.device,
        )
        wp.copy(self.costs, self._accepted_costs)
