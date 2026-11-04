"""Torch/batch adapter of historical contact_solvers.wrench_qp (4cc156c3^).

The force QP and its 96 FISTA updates are unchanged. Twelve axis targets
exercise the projected friction cone; returned witnesses are audited separately.
"""

import math
import torch
from .closure_types import ClosureResult, ClosureSettings
from .contact_margin import geom_contact_margins


def capped_simplex(value, cap):
    positive = value.clamp_min(0)
    ordered = value.sort(dim=-1, descending=True).values
    threshold = (ordered.cumsum(-1) - cap[..., None]) / torch.arange(
        1, value.shape[-1] + 1, device=value.device, dtype=value.dtype
    )
    count = (ordered > threshold).sum(-1).clamp_min(1)
    theta = threshold.gather(-1, (count - 1)[..., None]).squeeze(-1)
    projected = (value - theta[..., None]).clamp_min(0)
    return torch.where((positive.sum(-1) <= cap)[..., None], positive, projected)


def project_pressures(value, caps, minimum, contact_mask=None):
    if value.is_cuda and value.dtype == torch.float32 and value.ndim == 4:
        from .qp_projection import project_cuda

        return project_cuda(value.contiguous(), contact_mask)
    projected = capped_simplex(value, caps)
    missing = projected.sum((-2, -1)) < minimum
    low = torch.zeros_like(minimum)
    high = value.abs().amax((-2, -1)) + caps.amax(-1) + minimum + 1
    for _ in range(32):
        mid = 0.5 * (low + high)
        mass = capped_simplex(value + mid[..., None, None], caps).sum((-2, -1))
        low = torch.where(mass < minimum, mid, low)
        high = torch.where(mass >= minimum, mid, high)
    filled = capped_simplex(value + high[..., None, None], caps)
    return torch.where(missing[..., None, None], filled, projected)


def wrench_qp(matrix, target, contacts, iterations=96, contact_mask=None):
    # Same pressure bounds, initialization, regularization and FISTA as reference.
    shape = target.shape[:-1]
    caps = target.new_full((*shape, contacts), 10.0)
    minimum = target.new_ones(shape)
    alpha = target.new_full((*shape, contacts, 4), 1 / (4 * contacts))
    if contact_mask is not None:
        selected = contact_mask[:, None, :].expand(*shape, contacts)
        caps = caps * selected
        alpha = selected[..., None].expand_as(alpha).to(alpha) / (
            4 * selected.sum(-1).clamp_min(1)[..., None, None]
        )
    y = alpha.clone()
    moment = 1.0
    regularization = 1.0e-7
    lipschitz = matrix.square().sum((-2, -1)) + regularization
    for _ in range(iterations):
        residual = (matrix @ y.flatten(-2)[..., None]).squeeze(-1) - target
        gradient = (matrix.transpose(-1, -2) @ residual[..., None]).squeeze(
            -1
        ).reshape_as(y) + regularization * y
        proposal = project_pressures(
            y - gradient / lipschitz[..., None, None], caps, minimum, contact_mask
        )
        next_moment = 0.5 * (1 + math.sqrt(1 + 4 * moment * moment))
        y = proposal + (moment - 1) / next_moment * (proposal - alpha)
        alpha, moment = proposal, next_moment
    residual = (matrix @ alpha.flatten(-2)[..., None]).squeeze(-1) - target
    cost = 0.5 * residual.square().sum(-1) + 0.5 * regularization * alpha.square().sum(
        (-2, -1)
    )
    return alpha, residual, cost


class ProjectedQPClosure:
    @staticmethod
    def _contact_frame(outward_normals):
        normal = -torch.nn.functional.normalize(outward_normals, dim=-1)
        axis = torch.nn.functional.one_hot(normal.abs().argmin(-1), 3).to(normal.dtype)
        tangent = torch.nn.functional.normalize(
            torch.linalg.cross(normal, axis), dim=-1
        )
        return normal, tangent, torch.linalg.cross(normal, tangent)

    def __init__(
        self, settings=None, *, center, radius, object_mass, object_inertia=None
    ):
        self.settings = settings or ClosureSettings()
        self.center = torch.as_tensor(center).detach()
        self.radius = radius
        self.object_mass = object_mass
        self.inverse_mass = torch.eye(
            6, device=self.center.device, dtype=self.center.dtype
        )
        self.test_force_n = self.settings.test_force_n or max(object_mass * 9.81, 1.0)
        self.ray_count = 4
        self.metadata = dict(
            model="historical_wrench_qp_axis_witnesses",
            source="4cc156c3^:retarget/contact_solvers.py:wrench_qp",
            friction=self.settings.friction,
            torsional_friction_m=0.0,
            iterations=96,
            force_limit_n=10.0,
            minimum_pressure_n=1.0,
            criterion="12 axis witnesses, audited nonnegative pressure/caps and normalized wrench residual; numerical polyhedral force closure",
            comfree=False,
        )
        self.metadata.update(
            object_mass_kg=object_mass,
            center=self.center.tolist(),
            radius_m=float(radius),
            test_force_n=self.test_force_n,
            certificate_tolerance=self.settings.certificate_tolerance,
        )
        self.batch_centers = None
        self.graphs = {}

    def contact_wrenches(self, row_forces, outward_normals):
        normal, tangent, second = self._contact_frame(outward_normals)
        directions = normal[:, :, None] + self.settings.friction * torch.stack(
            (tangent, -tangent, second, -second), dim=-2
        )
        if row_forces.ndim == 4:
            directions = directions[:, None]
        force = (row_forces[..., None] * directions).sum(-2)
        return force, torch.zeros_like(force)

    def set_object_batch(self, centers, radii, inverses=None):
        self.batch_centers = centers
        self.batch_radii = radii

    def _compute(self, points, normals, centers, radii, contact_mask=None):
        finite = (
            torch.isfinite(points).all(-1)
            & torch.isfinite(normals).all(-1)
            & (normals.square().sum(-1) > 1.0e-20)
        )
        if contact_mask is None:
            valid = finite.all(-1)
        else:
            valid = (finite | ~contact_mask).all(-1) & contact_mask.any(-1)
            safe = finite & contact_mask
            points = torch.where(safe[..., None], points, 0.0)
            fallback = torch.zeros_like(normals)
            fallback[..., 0] = 1.0
            normals = torch.where(safe[..., None], normals, fallback)
        normal, t1, t2 = self._contact_frame(normals)
        directions = normal[:, :, None] + self.settings.friction * torch.stack(
            (t1, -t1, t2, -t2), -2
        )
        arms = (points - centers[:, None]) / radii[:, None, None]
        rays = torch.cat(
            (
                directions,
                torch.linalg.cross(arms[:, :, None].expand_as(directions), directions),
            ),
            -1,
        )
        if contact_mask is not None:
            rays = rays * contact_mask[:, :, None, None]
        matrix = rays.flatten(1, 2).transpose(-1, -2)[:, None]
        target = (
            torch.cat(
                (
                    torch.zeros((1, 6), device=points.device, dtype=points.dtype),
                    torch.eye(6, device=points.device, dtype=points.dtype),
                    -torch.eye(6, device=points.device, dtype=points.dtype),
                )
            )[None].expand(len(points), -1, -1)
            * self.test_force_n
        )
        alpha, residual, cost = wrench_qp(
            matrix, target, points.shape[1], contact_mask=contact_mask
        )
        # Do not certify an unconverged optimizer output by residual alone.
        constraints = (
            (alpha.amin((1, 2, 3)) >= -1.0e-6)
            & (alpha.sum(-1).amax((1, 2)) <= 10.0 + 1.0e-5)
            & (alpha.sum((-2, -1)).amin(1) >= 1.0 - 1.0e-5)
        )
        error = (residual[:, 1:] / self.test_force_n).norm(dim=-1).amax(-1)
        valid = valid & constraints & torch.isfinite(cost).all(-1)
        certified = (
            valid
            & (error <= self.settings.certificate_tolerance)
            & (error < 1 / math.sqrt(6))
        )
        margin = self.test_force_n * (1 / math.sqrt(6) - error)
        return (
            cost.mean(-1),
            alpha,
            residual / self.test_force_n,
            valid,
            certified,
            margin,
        )

    @torch.no_grad()
    def solve(self, points, outward_normals, *, contact_mask=None):
        if (
            points.ndim != 3
            or points.shape[-1] != 3
            or outward_normals.shape != points.shape
            or not points.shape[1]
        ):
            raise ValueError("Projected contacts must have shape (B,N,3), N > 0")
        if contact_mask is not None:
            contact_mask = torch.as_tensor(
                contact_mask, device=points.device, dtype=torch.bool
            )
            if contact_mask.shape != points.shape[:2]:
                raise ValueError("Contact mask must have shape (B,N)")
        centers = (
            self.center.to(points)[None].expand(len(points), -1)
            if self.batch_centers is None
            else self.batch_centers.to(points)
        )
        radii = (
            torch.as_tensor(
                self.radius, device=points.device, dtype=points.dtype
            ).expand(len(points))
            if self.batch_centers is None
            else self.batch_radii.to(points)
        )
        if points.is_cuda and self.settings.cuda_graph:
            key = (
                tuple(points.shape),
                points.dtype,
                points.device,
                contact_mask is not None,
            )
            if key not in self.graphs:
                p = points.clone()
                n = outward_normals.clone()
                c = centers.clone()
                r = radii.clone()
                m = None if contact_mask is None else contact_mask.clone()
                stream = torch.cuda.Stream(device=points.device)
                stream.wait_stream(torch.cuda.current_stream(points.device))
                with torch.cuda.stream(stream):
                    self._compute(p, n, c, r, m)
                torch.cuda.current_stream(points.device).wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    out = self._compute(p, n, c, r, m)
                self.graphs[key] = (p, n, c, r, m, graph, out)
            p, n, c, r, m, graph, out = self.graphs[key]
            p.copy_(points)
            n.copy_(outward_normals)
            c.copy_(centers)
            r.copy_(radii)
            if m is not None:
                m.copy_(contact_mask)
            graph.replay()
            cost, alpha, residual, valid, certified, margin = [x.clone() for x in out]
        else:
            cost, alpha, residual, valid, certified, margin = self._compute(
                points, outward_normals, centers, radii, contact_mask
            )
        return ClosureResult(
            cost=cost,
            preloads=alpha[:, 0].sum(-1),
            row_forces=alpha,
            scenario_residuals=residual,
            scenario_rms=residual.norm(dim=-1),
            converged=certified,
            valid=valid,
            certified=certified,
            certified_margin=margin,
            metadata=self.metadata,
        )

    def solve_with_geom_margins(
        self,
        points,
        outward_normals,
        *,
        signed_distance,
        hand_margin,
        object_margin,
        hand_gap=0.0,
        object_gap=0.0,
        pair_margin=None,
        pair_gap=None,
    ):
        """Evaluate actual margin-eligible slots at their unchanged surface projections.

        Returns the QP result and the selection, separately from the all-slot
        hypothetical projection diagnostic returned by solve without a mask.
        """
        distance = torch.as_tensor(
            signed_distance, device=points.device, dtype=points.dtype
        )
        if distance.shape != points.shape[:2]:
            raise ValueError("Signed distances must have shape (B,N)")
        selection = geom_contact_margins(
            distance,
            hand_margin,
            object_margin,
            hand_gap=hand_gap,
            object_gap=object_gap,
            pair_margin=pair_margin,
            pair_gap=pair_gap,
        )
        if selection.active.shape != distance.shape:
            raise ValueError("Geom parameters must broadcast to (B,N)")
        result = self.solve(points, outward_normals, contact_mask=selection.active)
        valid_geometry = selection.valid.all(-1)
        result.valid = result.valid & valid_geometry
        result.certified = result.certified & valid_geometry
        result.converged = result.converged & valid_geometry
        result.certified_margin = torch.where(
            result.valid, result.certified_margin, -torch.inf
        )
        result.metadata = dict(
            result.metadata,
            contact_selection="MuJoCo 3.3.3 max geom margin/gap; distance < margin-gap",
            projection="unchanged object surface; margin is not a surface offset",
        )
        return result, selection
