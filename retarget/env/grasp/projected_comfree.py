"""Fixed-projection ComFree closure with one shared preload per contact.

The default static response uses a derived six-dimensional velocity equilibrium
to find friction-cone force witnesses. The previous finite-horizon response is
available explicitly for reproduction; it is not a static closure criterion.
Friction directions are fixed rays and are never optimization variables. The
impedance-weighted ray masses approximate, rather than reproduce, MJWarp's
``efc.D`` assembly. They never depend on the optimized preload.
"""

from dataclasses import asdict, dataclass, field, fields
import math
import time

import torch

from retarget.dynamics.comfree.response import _contact_response_unpadded


@dataclass(frozen=True)
class ClosureSettings:
    equilibrium_backend: str = "torch"
    response_mode: str = "equilibrium"
    friction: float = .6
    torsional_friction_m: float = .02
    max_preload_n: float = 20.
    initial_preload_n: float = 2.
    test_force_n: float | None = None
    gn_iterations: int = 8
    initial_damping: float = .01
    timestep: float = .01
    horizon: int = 64
    tail_steps: int = 16
    stiffness: float = .2
    damping: float = .001
    impedance: float = .9
    certificate_tolerance: float = .05
    velocity_regularization: float = 1.e-4
    displacement_regularization: float = 1.e-4
    preload_regularization: float = 1.e-5
    gn_tolerance: float = 1.e-5
    invalid_cost: float = 1.e6
    cuda_graph: bool = True
    equilibrium_iterations: int = 16
    equilibrium_line_search_steps: int = 8
    equilibrium_damping: float = 1.e-10
    equilibrium_tolerance: float = 1.e-6
    force_effort_weight: float = 1.e-4

    def __post_init__(self):
        if self.equilibrium_backend not in ("torch", "cuda_kernel"):
            raise ValueError("closure.equilibrium_backend must be torch or cuda_kernel")
        positive = ("max_preload_n", "initial_damping", "timestep", "stiffness",
                    "certificate_tolerance", "gn_tolerance", "invalid_cost", "equilibrium_damping", "equilibrium_tolerance")
        nonnegative = ("friction", "torsional_friction_m", "initial_preload_n", "damping",
                       "velocity_regularization", "displacement_regularization",
                       "preload_regularization", "force_effort_weight")
        for name in positive + nonnegative:
            value = getattr(self, name)
            if not math.isfinite(value) or (value <= 0 if name in positive else value < 0):
                raise ValueError(f"Invalid closure.{name}")
        if not 0 < self.impedance < 1:
            raise ValueError("closure.impedance must be strictly between zero and one")
        if self.test_force_n is not None and (not math.isfinite(self.test_force_n) or self.test_force_n <= 0):
            raise ValueError("closure.test_force_n must be positive or None")
        for name in ("gn_iterations", "horizon", "tail_steps", "equilibrium_iterations", "equilibrium_line_search_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == "gn_iterations" else 1):
                raise ValueError(f"Invalid closure.{name}")
        if self.tail_steps > self.horizon:
            raise ValueError("closure.tail_steps exceeds horizon")
        if not isinstance(self.cuda_graph, bool):
            raise ValueError("closure.cuda_graph must be a Boolean")
        if self.response_mode not in ("equilibrium", "finite_horizon"):
            raise ValueError("closure.response_mode must be equilibrium or finite_horizon")


@dataclass
class ClosureResult:
    cost: torch.Tensor
    preloads: torch.Tensor
    row_forces: torch.Tensor
    scenario_residuals: torch.Tensor
    scenario_rms: torch.Tensor
    converged: torch.Tensor
    valid: torch.Tensor
    certified: torch.Tensor
    certified_margin: torch.Tensor
    objective_history: torch.Tensor | None = None
    metadata: dict = field(default_factory=dict)


class ProjectedComFreeClosure:
    """Batched bounded LM/GN; only the n nominal preloads are optimized.

    ``center``, ``points`` and the COM ``object_inertia`` use the same Cartesian
    frame. Rotations are scaled by ``radius`` so the probe state, velocity and
    wrench all use consistent six-dimensional coordinates. Actual reaction
    forces can exceed ``max_preload_n``; that bound applies to nominal preload.
    """

    def __init__(self, settings=None, *, center, radius, object_mass, object_inertia):
        self.settings = settings or ClosureSettings()
        self.center = torch.as_tensor(center).detach().to(device="cpu", dtype=torch.float64)
        self.radius, self.object_mass = float(radius), float(object_mass)
        inertia = torch.as_tensor(object_inertia).detach().to(device="cpu", dtype=torch.float64)
        if inertia.shape == (3,):
            inertia = torch.diag(inertia)
        if self.center.shape != (3,) or not bool(torch.isfinite(self.center).all()):
            raise ValueError("Object center must be a finite three-vector")
        if not all(math.isfinite(v) and v > 0 for v in (self.radius, self.object_mass)):
            raise ValueError("Object radius and mass must be finite and positive")
        if inertia.shape != (3, 3) or not bool(torch.isfinite(inertia).all()):
            raise ValueError("Object inertia must be a finite diagonal or 3 by 3 tensor")
        if not torch.allclose(inertia, inertia.T) or bool((torch.linalg.eigvalsh(inertia) <= 0).any()):
            raise ValueError("Object inertia must be symmetric positive definite")
        mass = torch.zeros((6, 6), dtype=torch.float64)
        mass[:3, :3] = torch.eye(3, dtype=torch.float64) * self.object_mass
        mass[3:, 3:] = inertia / self.radius ** 2
        self.inverse_mass = torch.linalg.inv(mass)
        self.test_force_n = self.settings.test_force_n or max(self.object_mass * 9.81, 1.)
        self.ray_count = 6 if self.settings.torsional_friction_m > 0 else 4
        soft_finger = self.ray_count == 6
        self.metadata = dict(
            model=(("fixed_projection_soft_finger_comfree_static_equilibrium_v3" if soft_finger else
                    "fixed_projection_comfree_static_equilibrium_v2") if self.settings.response_mode == "equilibrium"
                   else ("fixed_projection_soft_finger_comfree_finite_horizon_v3" if soft_finger else
                         "fixed_projection_native_comfree_finite_horizon_v1")),
            response_version=(3 if soft_finger else 2 if self.settings.response_mode == "equilibrium" else 1),
            contact_model=("coupled sliding and torsional friction pyramid" if soft_finger else "point sliding friction pyramid"),
            ray_count_per_contact=self.ray_count,
            ray_order=["+tangent_1", "-tangent_1", "+tangent_2", "-tangent_2"] +
                (["+normal_spin", "-normal_spin"] if soft_finger else []),
            torsional_coefficient_units="m; maximum intrinsic normal torque in Nm is coefficient times normal force in N",
            torsional_budget="sliding and intrinsic torsion share the nonnegative-ray normal-force budget",
            equilibrium_state_selection=("smallest finite raw wrench residual among semismooth Newton iterates including initial state; force witness always audited"
                                         if self.settings.response_mode == "equilibrium" else "not applied; finite horizon uses the tail mean"),
            row_force_statistic=("static equilibrium response" if self.settings.response_mode == "equilibrium"
                                 else "mean over final tail_steps of finite horizon"),
            scenario_rms_statistic=("static equilibrium wrench norm" if self.settings.response_mode == "equilibrium"
                                    else "root mean squared wrench over final tail_steps"),
            force_effort_normalization="mean over contacts of squared normal force / test_force_n squared",
            row_mass="effective_ray_mass_approximation",
            optimized_variables="one nominal normal preload per fixed template contact",
            scenarios="zero external wrench, then positive and negative six-dimensional axes",
            force_bound="nominal preload only; actual normal force is not bounded by it",
            center=self.center.tolist(), radius_m=self.radius,
            object_mass_kg=self.object_mass, object_inertia_kg_m2=inertia.tolist(),
            scenario_names=["zero"] + [f"{sign}{kind}_{axis}" for sign in ("+", "-")
                for kind in ("force", "torque") for axis in ("x", "y", "z")],
            wrench_residual_units="force / test_force_n; torque / (radius_m * test_force_n)",
            **asdict(self.settings),
        )
        self.metadata["test_force_n"] = self.test_force_n
        self._constant_cache = {}
        self._graph_cache = {}

    @staticmethod
    def _contact_frame(outward_normals):
        normal = -torch.nn.functional.normalize(outward_normals, dim=-1)
        axis = torch.nn.functional.one_hot(normal.abs().argmin(-1), 3).to(normal.dtype)
        tangent = torch.nn.functional.normalize(torch.linalg.cross(normal, axis), dim=-1)
        return normal, tangent, torch.linalg.cross(normal, tangent)

    def contact_wrenches(self, row_forces, outward_normals):
        """Decode object force (N) and intrinsic contact torque (Nm).

        Weights have shape ``(B,n,R)`` or ``(B,S,n,R)`` and normals ``(B,n,3)``.
        The intrinsic torque excludes the point's ``p cross force`` moment.
        Six-ray contacts obey both Coulomb bounds and share their sliding and
        spinning budget; ``torsional_friction_m`` is already a length coefficient
        and is not multiplied by the sliding friction coefficient.
        """
        if (not isinstance(row_forces, torch.Tensor) or not isinstance(outward_normals, torch.Tensor) or
                row_forces.ndim not in (3, 4) or outward_normals.ndim != 3 or
                outward_normals.shape[-1] != 3 or row_forces.shape[0] != outward_normals.shape[0] or
                row_forces.shape[-2] != outward_normals.shape[-2] or row_forces.shape[-1] != self.ray_count or
                row_forces.dtype != outward_normals.dtype or row_forces.device != outward_normals.device):
            raise ValueError("Contact weights and normals require matching (B,[S,]n,R) and (B,n,3) tensors")
        normal, tangent, second = self._contact_frame(outward_normals)
        directions = normal[:, :, None] + self.settings.friction * torch.stack((tangent, -tangent, second, -second), dim=-2)
        if self.ray_count == 6:
            directions = torch.cat((directions, normal[:, :, None].expand(-1, -1, 2, -1)), dim=-2)
        if row_forces.ndim == 4:
            directions, normal = directions[:, None], normal[:, None]
        force = (row_forces[..., None] * directions).sum(-2)
        intrinsic = (self.settings.torsional_friction_m * (row_forces[..., 4] - row_forces[..., 5]))[..., None] * normal if self.ray_count == 6 else torch.zeros_like(force)
        return force, intrinsic

    def _constants(self, points):
        """Allocate immutable object constants once, outside CUDA capture."""
        key = (points.device, points.dtype)
        cached = self._constant_cache.get(key)
        if cached is None:
            cached = (
                self.center.to(points), self.inverse_mass.to(points),
                points.new_tensor([1., 0., 0.]),
                torch.cat((points.new_zeros(1, 6), torch.eye(6, device=points.device, dtype=points.dtype),
                           -torch.eye(6, device=points.device, dtype=points.dtype))) * self.test_force_n,
            )
            self._constant_cache[key] = cached
        return cached

    def _geometry(self, points, outward_normals):
        if not isinstance(points, torch.Tensor) or not isinstance(outward_normals, torch.Tensor):
            raise TypeError("Closure points and normals must be Torch tensors")
        if points.ndim != 3 or points.shape[-1] != 3 or min(points.shape[:2]) == 0 or outward_normals.shape != points.shape:
            raise ValueError("Closure points and normals must have matching nonempty (batch, contacts, 3) shapes")
        if points.dtype not in (torch.float32, torch.float64) or points.dtype != outward_normals.dtype or points.device != outward_normals.device:
            raise ValueError("Closure points and normals must share a float32/float64 dtype and device")
        valid = (torch.isfinite(points).all((1, 2)) & torch.isfinite(outward_normals).all((1, 2)) &
                 (outward_normals.square().sum(-1) > 1.e-20).all(1))
        center, inverse, fallback, loads = self._constants(points)
        clean_points = torch.where(valid[:, None, None], points, 0.)
        normal, tangent, second = self._contact_frame(torch.where(valid[:, None, None], outward_normals, fallback))
        directions = normal[:, :, None] + self.settings.friction * torch.stack((tangent, -tangent, second, -second), dim=-2)
        arms = (clean_points - center) / self.radius
        sliding = torch.cat((directions, torch.linalg.cross(arms[:, :, None].expand_as(directions), directions)), dim=-1)
        normal_moment = torch.linalg.cross(arms, normal)
        if self.ray_count == 6:
            spin = self.settings.torsional_friction_m / self.radius * normal
            moments = torch.stack((normal_moment + spin, normal_moment - spin), dim=-2)
            torsional = torch.cat((normal[:, :, None].expand(-1, -1, 2, -1), moments), dim=-1)
            sliding = torch.cat((sliding, torsional), dim=-2)
        rays = sliding.flatten(1, 2)
        normal_rows = torch.cat((normal, normal_moment), dim=-1).repeat_interleave(self.ray_count, dim=1)
        inverse_weight = ((rays @ inverse) * rays).sum(-1).clamp_min(torch.finfo(points.dtype).eps)
        s = self.settings
        row_mass = (s.impedance / (1. - s.impedance)) / inverse_weight
        kappa = row_mass * (s.stiffness / s.timestep)
        drag = row_mass * (s.damping / s.timestep + s.stiffness)
        return rays, normal_rows, inverse, kappa, drag, loads, valid

    def _rollout(self, preloads, geometry, *, jacobian=False):
        if self.settings.response_mode == "equilibrium":
            return self._equilibrium_rollout(preloads, geometry, jacobian=jacobian)
        return self._finite_horizon_rollout(preloads, geometry, jacobian=jacobian)

    def _equilibrium_rollout(self, preloads, geometry, *, jacobian=False):
        from .comfree_equilibrium import equilibrium_response

        if self.settings.equilibrium_backend == "cuda_kernel" and preloads.is_cuda and not preloads.requires_grad:
            from .comfree_cuda import equilibrium_response_cuda
            equilibrium_response = equilibrium_response_cuda

        A, _, _, _, drag, loads, geometry_valid = geometry
        batch, contacts = preloads.shape
        s = self.settings
        response = equilibrium_response(
            preloads, A, drag, loads, iterations=s.equilibrium_iterations,
            line_search_steps=s.equilibrium_line_search_steps,
            damping=s.equilibrium_damping, tolerance=s.equilibrium_tolerance,
            jacobian=jacobian,
        )
        force = response.balance / self.test_force_n
        normal_force = response.row_forces.reshape(batch, 13, contacts, self.ray_count).sum(-1)
        # Force effort prevents a zero balance residual from making every
        # feasible grasp and every shared preload equally attractive.
        effort = normal_force / (self.test_force_n * math.sqrt(contacts))
        force_worst = force.square().sum(-1).argmax(1)
        effort_worst = effort.square().sum(-1).argmax(1)
        batch_index = torch.arange(batch, device=preloads.device)
        mean_weight = math.sqrt(.25 / 13.)
        effort_weight = math.sqrt(s.force_effort_weight)
        preload_weight = math.sqrt(s.preload_regularization / contacts) / s.max_preload_n
        residual = torch.cat((
            (mean_weight * force).flatten(1), force[batch_index, force_worst],
            (effort_weight * mean_weight * effort).flatten(1),
            effort_weight * effort[batch_index, effort_worst], preload_weight * preloads,
        ), dim=1)
        derivative = None
        if jacobian:
            force_jacobian = response.balance_jacobian / self.test_force_n
            effort_jacobian = response.row_force_jacobian.reshape(batch, 13, contacts, self.ray_count, contacts).sum(3)
            effort_jacobian = effort_jacobian / (self.test_force_n * math.sqrt(contacts))
            identity = torch.eye(contacts, dtype=preloads.dtype, device=preloads.device).expand(batch, -1, -1)
            derivative = torch.cat((
                (mean_weight * force_jacobian).flatten(1, 2), force_jacobian[batch_index, force_worst],
                (effort_weight * mean_weight * effort_jacobian).flatten(1, 2),
                effort_weight * effort_jacobian[batch_index, effort_worst], preload_weight * identity,
            ), dim=1)
        valid = geometry_valid & response.valid & torch.isfinite(residual).all(1)
        if jacobian:
            valid = valid & torch.isfinite(derivative).all((1, 2))
        return residual, derivative, force[:, :, None], response.row_forces.reshape(batch, 13, contacts, self.ray_count), valid

    def _finite_horizon_rollout(self, preloads, geometry, *, jacobian=False):
        A, N, inverse, kappa, drag, loads, geometry_valid = geometry
        batch, contacts = preloads.shape
        s = self.settings
        velocity = preloads.new_zeros(batch, 13, 6)
        displacement = torch.zeros_like(velocity)
        if jacobian:
            V = preloads.new_zeros(batch, 13, 6, contacts)
            Z = torch.zeros_like(V)
        bias = preloads.repeat_interleave(self.ray_count, dim=-1)[:, None] / self.ray_count
        smooth_acceleration = loads @ inverse.T
        row_mass = kappa[:, None] * (s.timestep / s.stiffness)
        force_tail, rows_tail, jac_tail = [], [], []
        finite = geometry_valid & torch.isfinite(preloads).all(1)
        for step in range(s.horizon):
            normal_gap = torch.einsum("brj,bsj->bsr", N, displacement) - bias / kappa[:, None]
            balance, row_force, _ = _contact_response_unpadded(
                s.timestep, velocity, smooth_acceleration, A[:, None], normal_gap,
                row_mass, loads, stiffness=s.stiffness, damping=s.damping,
            )
            if jacobian:
                active = (row_force > 0).to(preloads.dtype)
                direct = torch.einsum("bsnc,bncj->bsjn", active.reshape(batch, 13, contacts, self.ray_count), A.reshape(batch, contacts, self.ray_count, 6)) / self.ray_count
                position_matrix = torch.einsum("brj,bsr,brk->bsjk", A, active * kappa[:, None], N)
                velocity_matrix = torch.einsum("brj,bsr,brk->bsjk", A, active * drag[:, None], A)
                derivative = direct - position_matrix @ Z - velocity_matrix @ V
            if step >= s.horizon - s.tail_steps:
                force_tail.append(balance / self.test_force_n)
                rows_tail.append(row_force)
                if jacobian:
                    jac_tail.append(derivative / self.test_force_n)
            velocity = velocity + s.timestep * (balance @ inverse.T)
            displacement = displacement + s.timestep * velocity
            if jacobian:
                V = V + s.timestep * torch.einsum("jk,bskn->bsjn", inverse, derivative)
                Z = Z + s.timestep * V
            finite = finite & torch.isfinite(velocity).all((1, 2)) & torch.isfinite(displacement).all((1, 2))
        force = torch.stack(force_tail, dim=2)
        rows = torch.stack(rows_tail, dim=2).mean(2).reshape(batch, 13, contacts, self.ray_count)
        # Worst-direction support is decisive; the mean also supplies guidance
        # on the other scenarios. The selected worst branch is piecewise smooth.
        force_flat = force.flatten(2) / math.sqrt(s.tail_steps)
        scenario_square = force_flat.square().sum(-1)
        worst = scenario_square.argmax(1)
        batch_index = torch.arange(batch, device=preloads.device)
        mean_weight = math.sqrt(.25 / 13.)
        residual_parts = [(mean_weight * force_flat).flatten(1), force_flat[batch_index, worst]]
        duration = s.horizon * s.timestep
        vscale = self.test_force_n * duration / self.object_mass
        zscale = self.test_force_n * duration ** 2 / self.object_mass
        vweight = math.sqrt(s.velocity_regularization / 13.) / vscale
        zweight = math.sqrt(s.displacement_regularization / 13.) / zscale
        pweight = math.sqrt(s.preload_regularization / contacts) / s.max_preload_n
        residual_parts.extend(((vweight * velocity).flatten(1), (zweight * displacement).flatten(1), pweight * preloads))
        residual = torch.cat(residual_parts, dim=1)
        derivative_residual = None
        if jacobian:
            Jforce = torch.stack(jac_tail, dim=2).flatten(2, 3) / math.sqrt(s.tail_steps)
            identity = torch.eye(contacts, dtype=preloads.dtype, device=preloads.device).expand(batch, -1, -1)
            derivative_residual = torch.cat(((mean_weight * Jforce).flatten(1, 2), Jforce[batch_index, worst], (vweight * V).flatten(1, 2),
                                              (zweight * Z).flatten(1, 2), pweight * identity), dim=1)
            finite = finite & torch.isfinite(derivative_residual).all((1, 2))
        finite = finite & torch.isfinite(residual).all(1) & torch.isfinite(rows).all((1, 2, 3))
        return residual, derivative_residual, force, rows, finite

    def _result(self, preloads, rollout, *, converged=None, history=None):
        residual, _, force, rows, valid = rollout
        cost = torch.where(valid, residual.square().sum(1), self.settings.invalid_cost)
        # Averaging nonnegative ray forces preserves the same friction cone.
        mean_force = force.mean(2)
        error = torch.linalg.vector_norm(mean_force[:, 1:], dim=-1).amax(1)
        margin = self.test_force_n * (1. / math.sqrt(6.) - error)
        rms = force.square().sum(-1).mean(2).sqrt()
        certified = valid & (error <= self.settings.certificate_tolerance) & (margin > 0.)
        if converged is None:
            converged = torch.zeros_like(valid)
        return ClosureResult(
            cost=cost, preloads=preloads, row_forces=rows,
            scenario_residuals=mean_force, scenario_rms=rms,
            converged=converged & valid, valid=valid, certified=certified,
            certified_margin=torch.where(valid, margin, torch.full_like(margin, -torch.inf)),
            objective_history=history, metadata=self.metadata.copy(),
        )

    def probe(self, points, outward_normals, preloads):
        """Differentiable fixed-preload probe, without an optimization step."""
        geometry = self._geometry(points, outward_normals)
        if not isinstance(preloads, torch.Tensor) or preloads.shape != points.shape[:2] or preloads.dtype != points.dtype or preloads.device != points.device:
            raise ValueError("Preloads must match the points' batch/contact shape, dtype and device")
        return self._result(preloads, self._rollout(preloads, geometry))

    @torch.no_grad()
    def solve(self, points, outward_normals):
        """Solve with cached CUDA graphs on GPU and the eager reference on CPU.

        The first call for a new batch/contact shape warms up and captures the
        complete, unchanged optimizer. Returned tensors own their storage and
        remain valid after subsequent replays. Set ``cuda_graph=False`` to use
        the eager reference explicitly. Capture failures are reported rather
        than silently switching to a slower algorithm.
        """
        if isinstance(points, torch.Tensor) and points.is_cuda and self.settings.cuda_graph:
            return self._solve_graph(points, outward_normals)
        return self.solve_eager(points, outward_normals)

    @torch.no_grad()
    def solve_eager(self, points, outward_normals):
        """Uncaptured reference implementation of the same bounded LM/GN."""
        geometry = self._geometry(points, outward_normals)
        s = self.settings
        batch, contacts = points.shape[:2]
        preloads = points.new_full((batch, contacts), min(s.initial_preload_n, s.max_preload_n))
        damping = points.new_full((batch,), s.initial_damping)
        converged = torch.zeros(batch, dtype=torch.bool, device=points.device)
        current = self._rollout(preloads, geometry, jacobian=True)
        history = [self._result(preloads, current).cost]
        identity = torch.eye(contacts, dtype=points.dtype, device=points.device)
        for _ in range(s.gn_iterations):
            residual, jacobian, _, _, valid = current
            safe_jacobian = torch.where(valid[:, None, None], jacobian, 0.)
            safe_residual = torch.where(valid[:, None], residual, 0.)
            jt = safe_jacobian.transpose(-1, -2)
            normal = jt @ safe_jacobian
            gradient = (jt @ safe_residual[..., None]).squeeze(-1)
            scale = normal.diagonal(dim1=-2, dim2=-1).clamp_min(1.e-6)
            system = normal + torch.diag_embed(damping[:, None] * scale) + 1.e-10 * identity
            delta, status = torch.linalg.solve_ex(system, -gradient[..., None], check_errors=False)
            delta = delta[..., 0]
            trial = (preloads + delta).clamp(0., s.max_preload_n)
            candidate = self._rollout(trial, geometry, jacobian=True)
            old_cost = history[-1]
            trial_cost = self._result(trial, candidate).cost
            accepted = valid & candidate[-1] & (status == 0) & torch.isfinite(delta).all(1) & (trial_cost <= old_cost)
            step_small = (trial - preloads).abs().amax(1) <= s.gn_tolerance * (1. + preloads.abs().amax(1))
            projected_gradient = preloads - (preloads - gradient).clamp(0., s.max_preload_n)
            converged |= valid & ((projected_gradient.abs().amax(1) <= s.gn_tolerance) | (accepted & step_small))
            preloads = torch.where(accepted[:, None], trial, preloads)
            current = tuple(torch.where(accepted.reshape((batch,) + (1,) * (a.ndim - 1)), b, a)
                            for a, b in zip(current, candidate))
            history.append(torch.where(accepted, trial_cost, old_cost))
            damping = torch.where(accepted, damping * .5, damping * 4.).clamp(1.e-8, 1.e8)
        return self._result(preloads, current, converged=converged, history=torch.stack(history, dim=1))

    def _capture_graph(self, points, outward_normals):
        # Validate shape/type and populate the resident constant cache before
        # capture. No CPU tensor is copied to CUDA inside the captured region.
        self._geometry(points, outward_normals)
        stream = torch.cuda.Stream(device=points.device)
        stream.wait_stream(torch.cuda.current_stream(points.device))
        static_points = torch.empty_like(points, memory_format=torch.contiguous_format)
        static_normals = torch.empty_like(outward_normals, memory_format=torch.contiguous_format)
        start = time.perf_counter()
        with torch.cuda.stream(stream):
            static_points.copy_(points)
            static_normals.copy_(outward_normals)
            for _ in range(2):
                self.solve_eager(static_points, static_normals)
        stream.synchronize()
        warmup_seconds = time.perf_counter() - start
        graph = torch.cuda.CUDAGraph()
        start = time.perf_counter()
        with torch.cuda.graph(graph, stream=stream):
            captured = self.solve_eager(static_points, static_normals)
        # Instantiate and execute once as setup, outside the timed replay path.
        with torch.cuda.stream(stream):
            graph.replay()
        stream.synchronize()
        capture_seconds = time.perf_counter() - start
        torch.cuda.current_stream(points.device).wait_stream(stream)
        return dict(graph=graph, points=static_points, normals=static_normals, result=captured,
                    warmup_seconds=warmup_seconds, capture_seconds=capture_seconds)

    def _solve_graph(self, points, outward_normals):
        # The eager validation supplies the same errors for malformed inputs;
        # avoiding any content-dependent CPU decisions keeps replay asynchronous.
        if (not isinstance(outward_normals, torch.Tensor) or points.ndim != 3 or
                points.shape[-1] != 3 or min(points.shape[:2]) == 0 or
                outward_normals.shape != points.shape or points.dtype not in (torch.float32, torch.float64) or
                points.dtype != outward_normals.dtype or points.device != outward_normals.device):
            return self.solve_eager(points, outward_normals)
        key = (self.settings.response_mode, self.settings, points.shape[0], points.shape[1], points.dtype, points.device)
        entry = self._graph_cache.get(key)
        if entry is None:
            try:
                entry = self._capture_graph(points, outward_normals)
            except RuntimeError as error:
                raise RuntimeError(
                    f"Projected ComFree CUDA graph capture failed for {key}; "
                    "use ClosureSettings(cuda_graph=False) to select the eager reference"
                ) from error
            self._graph_cache[key] = entry
        entry["points"].copy_(points)
        entry["normals"].copy_(outward_normals)
        entry["graph"].replay()
        values = {}
        for item in fields(ClosureResult):
            value = getattr(entry["result"], item.name)
            values[item.name] = value.clone() if isinstance(value, torch.Tensor) else value.copy() if isinstance(value, dict) else value
        values["metadata"].update(
            execution="cuda_graph", cuda_graph_warmup_seconds=entry["warmup_seconds"],
            cuda_graph_capture_seconds=entry["capture_seconds"],
        )
        return ClosureResult(**values)

    def clear_cuda_graph_cache(self):
        """Release captured shape-specific workspaces; keep object constants."""
        self._graph_cache.clear()
