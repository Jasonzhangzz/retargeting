"""Transport canonical object contacts and adapt all 26 hand coordinates on CUDA."""
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import warp as wp

from .audit import GraspAudit
from ..geometry import ObjectMeshSurface

POLICY = 'solved_canonical_contacts_cuda_lm_v2'


@dataclass(frozen=True)
class SizeIKSettings:
    enabled: bool = True
    iterations: int = 32
    starts: int = 4
    damping: float = .01
    bootstrap_canonical: bool = True
    canonical_grasp_directory: str | None = None

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError('size_ik.enabled must be boolean')
        if not isinstance(self.bootstrap_canonical, bool):
            raise ValueError('size_ik.bootstrap_canonical must be boolean')
        if self.canonical_grasp_directory is not None and not isinstance(self.canonical_grasp_directory, str):
            raise ValueError('size_ik.canonical_grasp_directory must be a path string')
        for name in ('iterations', 'starts'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'size_ik.{name} must be a positive integer')
        if isinstance(self.damping, bool) or not math.isfinite(self.damping) or self.damping <= 0:
            raise ValueError('size_ik.damping must be finite and positive')


def transport_contacts(points, canonical, primitive):
    points = np.asarray(points, float)
    if canonical is None or primitive is None:
        return points.copy()
    if primitive['kind'] != canonical['kind']:
        raise ValueError('Cannot transport contacts between different shape families')
    ratio = np.array([primitive['radius_m']/canonical['radius_m']]*2 +
                     [primitive['height_m']/canonical['height_m']])
    return points*ratio


def primitive_sdf(points, primitive):
    """Differentiable primitive SDF for whole-hand IK collision guidance."""
    kind = primitive['kind']
    radius, height = primitive['radius_m'], primitive['height_m']
    if kind in ('cylinder', 'disk'):
        radial = (points[..., :2].square().sum(-1)+1.e-16).sqrt()-radius
        axial = points[..., 2].abs()-height/2
        q = torch.stack((radial, axial), -1)
    elif kind in ('plate', 'cuboid'):
        q = points.abs()-points.new_tensor(primitive['dimensions_m'])/2
    elif kind == 'sphere':
        return (points.square().sum(-1)+1.e-16).sqrt()-radius
    else:
        raise ValueError(f'Unsupported IK primitive {kind}')
    outside = (q.clamp_min(0).square().sum(-1)+1.e-16).sqrt()
    return outside+q.amax(-1).clamp_max(0)


class CylinderSideProjection:
    """Keep specified finger slots on the actual lateral mesh triangles."""
    def __init__(self, surface, device, slots):
        triangles = surface.triangles[np.abs(surface.normals[:, 2]) < .25]
        if not len(triangles):
            raise ValueError('Cylinder side constraint has no lateral triangles')
        self.audit = GraspAudit(ObjectMeshSurface(triangles.reshape(-1, 3),
            np.arange(len(triangles)*3).reshape(-1, 3)), str(device))
        self.slots = torch.as_tensor(slots, device=device, dtype=torch.long)

    def apply(self, points, projected, normals, gap):
        p, n, _ = self.audit.query(points[:, self.slots].flatten(0, 1))
        p = p.reshape(len(points), len(self.slots), 3)
        n = n.reshape_as(p)
        distance = torch.linalg.vector_norm(points[:, self.slots]-p, dim=-1)
        signed = torch.where(gap[:, self.slots] < 0, -distance, distance)
        projected, normals, gap = projected.clone(), normals.clone(), gap.clone()
        projected[:, self.slots], normals[:, self.slots], gap[:, self.slots] = p, n, signed
        return projected, normals, gap


def bounded_lm(residual, seeds, low, high, settings, oracle=None):
    """Independent bounded LM acceptance for a batch of hand seeds."""
    q = seeds.clamp(low, high)
    evaluate = torch.vmap(residual)
    jacobian = torch.vmap(torch.func.jacfwd(residual))
    frozen = oracle(q) if oracle else None
    r = evaluate(q, frozen) if oracle else evaluate(q)
    cost = r.square().sum(-1)
    damping = q.new_full((len(q),), settings.damping)
    history = [cost.detach().clone()]
    eye = torch.eye(q.shape[-1], device=q.device, dtype=q.dtype)
    for _ in range(settings.iterations):
        j = jacobian(q, frozen) if oracle else jacobian(q)
        h = j.transpose(-1, -2)@j
        gradient = (j.transpose(-1, -2)@r[..., None])[..., 0]
        diagonal = h.diagonal(dim1=-2, dim2=-1).clamp_min(1.)
        matrix = h+torch.diag_embed(damping[:, None]*diagonal)+1.e-9*eye
        step, info = torch.linalg.solve_ex(matrix, -gradient)
        finite = (info == 0) & torch.isfinite(step).all(-1)
        candidate = (q+torch.where(finite[:, None], step, 0.)).clamp(low, high)
        candidate_frozen = oracle(candidate) if oracle else None
        candidate_r = evaluate(candidate, candidate_frozen) if oracle else evaluate(candidate)
        candidate_cost = candidate_r.square().sum(-1)
        accepted = finite & torch.isfinite(candidate_cost) & (candidate_cost < cost)
        q = torch.where(accepted[:, None], candidate, q)
        r = torch.where(accepted[:, None], candidate_r, r)
        cost = torch.where(accepted, candidate_cost, cost)
        if oracle:
            frozen = {key: torch.where(accepted.reshape(len(q), *([1]*(value.ndim-1))),
                       candidate_frozen[key], value) for key, value in frozen.items()}
        damping = torch.where(accepted, damping*.5, damping*4).clamp(1.e-8, 1.e8)
        history.append(cost.detach().clone())
    return q.detach(), cost.detach(), torch.stack(history).detach()


class NativeCollisionLinearization:
    """Exact active contact distances, differentiated through frozen attachments.

    Rebuild native CUDA collisions at both current and trial q. The deepest
    hand/object and self-contact supply two piecewise smooth distance rows.
    Autodiff uses the rigid hand-local contact positions and frozen normals.
    """
    def __init__(self, objective, frame_fk):
        from .hand import HAND_BODIES
        self.objective, self.fk = objective, frame_fk
        model = objective.cpu
        owner = []
        for geom_body in model.geom_bodyid:
            body = int(geom_body)
            while body and model.body(body).name not in HAND_BODIES:
                body = int(model.body_parentid[body])
            owner.append(HAND_BODIES.index(model.body(body).name) if body else len(HAND_BODIES))
        self.owner = torch.tensor(owner, device=objective.device)
        self.object_geom = torch.tensor(model.geom_bodyid == objective.object_id, device=objective.device)

    def frames(self, q):
        p, r = self.fk.frames(q)
        return torch.cat((p, p.new_zeros(1, 3))), torch.cat((r, torch.eye(3, device=r.device, dtype=r.dtype)[None]))

    @torch.no_grad()
    def __call__(self, q):
        o = self.objective
        o._geometry(q.float())
        contact = o.physics.data.contact
        count = int(wp.to_torch(o.physics.data.nacon)[0])
        if count > o.physics.data.naconmax:
            raise RuntimeError('Collision buffer overflow during size IK')
        if not count:
            return dict(owner=torch.full((len(q), 2, 2), len(self.fk.finger), device=q.device, dtype=torch.long),
                local=q.new_zeros(len(q), 2, 2, 3), normal=q.new_zeros(len(q), 2, 3), distance=q.new_ones(len(q), 2))
        geoms = wp.to_torch(contact.geom)[:count].long()
        worlds = wp.to_torch(contact.worldid)[:count].long()
        distance = wp.to_torch(contact.dist)[:count]
        is_object = self.object_geom[geoms]
        kind = (~is_object.any(-1)).long()
        valid = ~is_object.all(-1)
        mask = ((worlds[None, None] == torch.arange(len(q), device=q.device)[:, None, None])
            & (kind[None, None] == torch.arange(2, device=q.device)[None, :, None]) & valid[None, None])
        scores = torch.where(mask, distance[None, None], torch.inf)
        best, ids = scores.min(-1)
        active = torch.isfinite(best)
        selected_geoms = geoms[ids]
        owner = self.owner[selected_geoms]
        pos = wp.to_torch(contact.pos)[:count][ids].to(q.dtype)
        normal = wp.to_torch(contact.frame).reshape(-1, 3, 3)[:count][ids, 0].to(q.dtype)
        p, r = torch.vmap(self.frames)(q)
        rows = torch.arange(len(q), device=q.device)[:, None, None]
        local = torch.einsum('bkpji,bkpj->bkpi', r[rows, owner], pos[:, :, None]-p[rows, owner])
        return dict(owner=owner, local=local, normal=torch.where(active[..., None], normal, 0.),
                    distance=torch.where(active, best.to(q.dtype), 1.))

    def residual(self, q, frozen):
        p, r = self.frames(q)
        owner = frozen['owner']
        points = p[owner]+torch.einsum('kpij,kpj->kpi', r[owner], frozen['local'])
        distance = frozen['distance']+((points[:, 1]-points[:, 0])*frozen['normal']).sum(-1)
        allowance = q.new_tensor([self.objective.settings.hand_object_allowance_m, 0.])
        return (-distance-allowance).clamp_min(0)/self.objective.settings.penetration_scale_m*math.sqrt(self.objective.settings.penetration_weight)


def initialize_size(objective, reference_path, primitive, reference_q, settings, seed):
    from .template import WujiContactKinematics
    if objective.device.type != 'cuda':
        raise ValueError('Production size IK requires CUDA FK')
    started = time.perf_counter()
    reference = json.loads(Path(reference_path).read_text())
    seed_q = reference.get('canonical_grasp_qpos', reference_q)
    objective._search_bounds(seed_q)
    points = transport_contacts(reference['object_points'], reference.get('canonical_primitive'), primitive)
    if objective.side_projection is not None and primitive:
        # Avoid cap rims when converting a cap correspondence to a side target.
        slots = objective.side_projection.slots.cpu().numpy()
        rim_margin = min(.001, primitive['height_m']*.2)
        points[slots, 2] = np.clip(points[slots, 2], -primitive['height_m']/2+rim_margin,
                                 primitive['height_m']/2-rim_margin)
    target = torch.as_tensor(points, device=objective.device, dtype=torch.float32)
    projected, outward, gap = objective.audit.query(target)
    if objective.side_projection is not None:
        projected, outward, _ = objective.side_projection.apply(target[None], projected[None], outward[None], gap[None])
        projected, outward = projected[0], outward[0]
    fk = WujiContactKinematics(objective.cpu, objective.bodies.cpu().numpy(), objective.device)
    from .hand import HAND_BODIES
    frame_fk = WujiContactKinematics(objective.cpu, [objective.cpu.body(n).id for n in HAND_BODIES], objective.device)
    oracle = NativeCollisionLinearization(objective, frame_fk)
    local, normals = fk.tensor(objective.template.local_points), fk.tensor(objective.template.local_normals)
    collision_fk = None
    if primitive:
        from .hand import HAND_BODIES, query_path
        collision_bodies, collision_local = [], []
        for name in HAND_BODIES:
            path = query_path(objective.cpu, name)
            if not path.is_file():
                continue
            with np.load(path) as pool:
                points_local = pool['local_pos']
                indices = np.linspace(0, len(points_local)-1, min(32, len(points_local)), dtype=int)
                collision_local.extend(points_local[indices])
                collision_bodies.extend([objective.cpu.body(name).id]*len(indices))
        collision_fk = WujiContactKinematics(objective.cpu, collision_bodies, objective.device)
        collision_local = fk.tensor(collision_local)
        collision_zero = torch.zeros_like(collision_local)
    target_points, target_normals = projected.to(fk.dtype), outward.to(fk.dtype)
    normal_scale = math.sqrt(2*(1-math.cos(math.radians(objective.settings.normal_tolerance_deg))))
    def residual(q, frozen):
        p, n = fk.contacts(q, local, normals)
        terms = [(p-target_points).flatten()/objective.settings.contact_scale_m,
                 (n+target_normals).flatten()/normal_scale]
        if collision_fk is not None:
            skin, _ = collision_fk.contacts(q, collision_local, collision_zero)
            depth = -primitive_sdf(skin, primitive)
            terms.append((depth-objective.settings.hand_object_allowance_m).clamp_min(0)
                /objective.settings.penetration_scale_m * math.sqrt(objective.settings.penetration_weight/len(skin)))
        terms.append(oracle.residual(q, frozen))
        if objective.posture is not None:
            terms.append(objective.posture.single(q)[0])
        if callable(getattr(objective, "extra_ik_residual", None)):
            terms.append(objective.extra_ik_residual(q))
        return torch.cat(terms)
    generator = torch.Generator(device=objective.device).manual_seed(int(seed))
    base = fk.tensor(seed_q)
    seeds = base.expand(settings.starts, -1).clone()
    if settings.starts > 1:
        noise = fk.tensor([.003]*3+[.05]*3+[.12]*20)
        seeds[1:] += torch.randn(settings.starts-1, 26, generator=generator,
            device=objective.device, dtype=fk.dtype)*noise
    low, high = objective.qlo.to(fk.dtype), objective.qhi.to(fk.dtype)
    previous_worlds = objective.worlds
    objective.set_batch_size(settings.starts)
    q, losses, history = bounded_lm(residual, seeds, low, high, settings, oracle=oracle)
    # Native full-hand collision scoring selects among LM solutions and the
    # finite canonical fallback. No force solve or dynamics is run for IK.
    candidates = torch.cat((q, base[None])).float()
    scores = objective.evaluate_initialization(candidates[:, None])
    selected = int(scores.argmin())
    chosen = candidates[selected].cpu().numpy().astype(float)
    if not bool(torch.isfinite(scores[selected])):
        raise FloatingPointError('No finite size IK initialization')
    before = history[0, 0]
    chosen_frozen = oracle(fk.tensor(chosen)[None].expand(settings.starts, -1))
    final_loss = residual(fk.tensor(chosen), {k: v[0] for k, v in chosen_frozen.items()}).square().sum()
    selection_parts = {key: objective.parts[key].cpu().tolist() for key in
            ('distance', 'normal', 'required_group', 'penetration', 'soft_penetration')}
    objective.set_batch_size(previous_worlds)
    report = dict(policy=POLICY, enabled=True, backend='torch_cuda_batched_bounded_LM',
        variables=26, pose_prior_used=False, canonical_object_pose_reoptimized=False,
        collision_guide='native CUDA hand-object and self-contact linearizations plus analytic object SDF on hand surface samples; native trial acceptance',
        template_slots_preserved=True, canonical_finger_angles_used_as_seed_only=True,
        reference_initial_qpos=np.asarray(reference_q).tolist(), initial_qpos=chosen.tolist(),
        source_initial_qpos=np.asarray(seed_q).tolist(),
        canonical_grasp_seed_used='canonical_grasp_qpos' in reference,
        canonical_grasp_source=reference.get('canonical_grasp_reference'),
        canonical_size_scale=reference.get("canonical_size_scale", 1.), size_scale=primitive.get('size_scale') if primitive else 1.,
        canonical_primitive=reference.get('canonical_primitive'), target_primitive=primitive,
        transported_object_points=points.tolist(), target_object_points=projected.cpu().tolist(),
        target_object_outward_normals=outward.cpu().tolist(),
        side_contact_slots=objective.side_projection.slots.cpu().tolist() if objective.side_projection is not None else [],
        initial_ik_residual_squared=float(before), candidate_ik_residual_squared=losses.cpu().tolist(),
        lm_objective_history=history.cpu().tolist(), candidate_selection_costs=scores.cpu().tolist(),
        candidate_qpos=candidates.cpu().tolist(),
        candidate_selection_parts=selection_parts,
        selected=selected, canonical_fallback_selected=selected == len(q),
        final_ik_residual_squared=float(final_loss),
        seconds=time.perf_counter()-started)
    return chosen, report
