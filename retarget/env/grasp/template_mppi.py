"""Static q-only MPPI with immutable template contacts and ComFree preloads.

Projection and native collisions are rebuilt for every candidate. Only the n
preload magnitudes are differentiated by the inner solver; neither hand-local
anchors nor force directions are search variables.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, replace
import math
import time

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import trimesh
import warp as wp

from retarget.env.geometry import ObjectMeshSurface
from retarget.solver.spider_mppi import optimize as sample_mppi
from .audit import GraspAudit
from .graspadmm_mppi import _depths, refinement_model
from .projected_comfree import ClosureSettings, ProjectedComFreeClosure
from .spider_physics import BatchedPhysics

OBJECTIVE = "template_comfree_soft_contact_v4"
SUCCESS_CRITERION = "projected_soft_contact_closure_v2"


@dataclass(frozen=True)
class TemplateMPPISettings:
    stages: int = 2
    final_stage_noise: float = .25
    pos_noise: float = .005
    rot_noise: float = .05
    joint_noise: float = .1
    max_pos_delta: float = .1
    max_rot_delta: float = 1.
    contact_fraction: float = .8
    contact_scale_m: float = .002
    contact_tolerance_m: float = .002
    normal_tolerance_deg: float = 30.
    outlier_weight: float = .05
    distance_weight: float = 1.
    normal_weight: float = 1.
    required_group_weight: float = 1.
    force_contact_weight: float = 1.
    antipodal_weight: float = 0.
    type9_antipodal_weight: float = 1.
    type24_antipodal_weight: float = 1.
    antipodal_scale_m: float = .001
    closure_weight: float = 1.
    pose_weight: float = 0.
    prismatic_reverse_limit_deg: float = 0.
    finger_order_margin_m: float = .004
    finger_order_scale_m: float = .005
    finger_order_weight: float = 1000.
    pip_flexion_cap_deg: float = 100.
    joint_prior_scale_rad: float = .5
    penetration_scale_m: float = .001
    penetration_weight: float = 1.e4
    hand_object_allowance_m: float = .002
    soft_penetration_weight: float = .1
    collision_clearance_m: float = 0.
    collision_margin_m: float = 0.
    penetration_tolerance_m: float = 1.e-5
    nconmax: int = 256
    guard_mean: bool = True
    closure: ClosureSettings = field(default_factory=ClosureSettings)

    def __post_init__(self):
        if isinstance(self.closure, dict):
            object.__setattr__(self, "closure", ClosureSettings(**self.closure))
        if not isinstance(self.closure, ClosureSettings):
            raise ValueError("template_mppi.objective.closure must be a mapping")
        for key, value in asdict(self).items():
            if key in ("closure", "guard_mean"):
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"template_mppi.objective.{key} must be finite numeric")
            if value < 0 or (key not in ("outlier_weight", "pose_weight", "prismatic_reverse_limit_deg", "required_group_weight", "force_contact_weight", "antipodal_weight", "type9_antipodal_weight", "type24_antipodal_weight", "collision_clearance_m", "collision_margin_m", "penetration_tolerance_m", "hand_object_allowance_m", "soft_penetration_weight") and value == 0):
                raise ValueError(f"template_mppi.objective.{key} has invalid sign")
        if any(not isinstance(getattr(self, key), int) for key in ("stages", "nconmax")):
            raise ValueError("stages and nconmax must be positive integers")
        if not 0 < self.contact_fraction <= 1 or not 0 < self.final_stage_noise <= 1:
            raise ValueError("contact_fraction and final_stage_noise must be in (0,1]")
        if not 0 < self.normal_tolerance_deg < 180:
            raise ValueError("normal_tolerance_deg must be in (0,180)")
        if self.collision_margin_m < self.collision_clearance_m:
            raise ValueError("collision_margin_m must cover collision_clearance_m")
        if not isinstance(self.guard_mean, bool):
            raise ValueError("guard_mean must be boolean")


def contact_terms(distance, hand_normals, object_normals, settings):
    """Select the same immutable slots for distance and normal terms."""
    # Compression in the allowed soft layer is contact, not a separation error.
    distance_residual = distance.clamp_min(0.) + (distance + settings.hand_object_allowance_m).clamp_max(0.)
    distance_error = (distance_residual / settings.contact_scale_m).square()
    cosine = -(hand_normals * object_normals).sum(-1).clamp(-1., 1.)
    # Squared vector residual, normalized so the acceptance angle costs one.
    # The former (1-cos)^2 was fourth order near zero and cost only .018 at 30°.
    normal_error = (1. - cosine) / (1. - math.cos(math.radians(settings.normal_tolerance_deg)))
    joint_error = settings.distance_weight * distance_error + settings.normal_weight * normal_error
    count = max(1, math.ceil(settings.contact_fraction * distance.shape[-1]))
    ids = joint_error.topk(count, dim=-1, largest=False, sorted=False).indices
    distance_cost = distance_error.gather(1, ids).mean(1) + settings.outlier_weight * distance_error.mean(1)
    normal_cost = normal_error.gather(1, ids).mean(1) + settings.outlier_weight * normal_error.mean(1)
    fitted = (distance >= -settings.hand_object_allowance_m) & (distance <= settings.contact_tolerance_m) & (
        cosine >= math.cos(math.radians(settings.normal_tolerance_deg)))
    return dict(distance=distance_cost, normal=normal_cost, selected=ids, slot_error=joint_error,
                fitted=fitted, coverage=fitted.to(distance.dtype).mean(1), cosine=cosine)


def penetration_terms(depth, settings):
    """Full-hand clearance deficits [hand/object, hand/hand], in metres.

    Only hand/object compression gets a soft allowance. The independent self
    term still penalizes any overlap; touching at zero distance is allowed.
    """
    object_depth = (depth[:, 0] - settings.collision_clearance_m).clamp_min(0.)
    excess = (object_depth - settings.hand_object_allowance_m).clamp_min(0.)
    self_depth = depth[:, 1]
    strict = (excess / settings.penetration_scale_m).square() + (self_depth / settings.penetration_scale_m).square()
    soft_scale = max(settings.hand_object_allowance_m, settings.penetration_scale_m)
    soft = (object_depth / soft_scale).square()
    return strict, soft, excess


def acceptance(coverage, certified, valid, depth, settings):
    finite = bool(valid) and bool(np.isfinite(depth).all())
    object_in_soft_layer = finite and float(depth[0]) <= settings.hand_object_allowance_m + 1.e-8
    self_ok = finite and float(depth[1]) <= settings.penetration_tolerance_m
    soft_success = bool(finite and certified and self_ok)
    return dict(hand_object_within_soft_allowance=object_in_soft_layer, hand_hand_penetration_accepted=self_ok,
        soft_contact_success=soft_success,
        geometry_success=bool(soft_success and coverage + 1.e-7 >= settings.contact_fraction))


def force_contact_terms(slot_error, fitted, row_forces):
    """Couple projected wrench support to the same fixed hand-local slots.

    Far-away projections remain available for early guidance. If the solver
    relies on one of them, its gap/normal error cannot disappear in top-k.
    """
    normal_force = row_forces.sum(-1)
    total = normal_force.sum(-1, keepdim=True)
    share = torch.where(total > 1.e-8, normal_force / total.clamp_min(1.e-8),
                        torch.full_like(normal_force, 1. / normal_force.shape[-1]))
    support_error = (share * slot_error[:, None]).sum(-1)[:, 1:]
    return (support_error.amax(1) + .25 * support_error.mean(1),
            (share * fitted[:, None]).sum(-1))


def required_contact_terms(terms, groups):
    return (torch.stack([terms["slot_error"][:, ids].amin(1) for ids in groups], 1).mean(1),
            torch.stack([terms["fitted"][:, ids].any(1) for ids in groups], 1))


def antipodal_contact_term(projected, outward, scale):
    """Penalize tangential separation of two fixed opposing contact slots.

    On a long thin cylinder, gap and normal costs allow the contacts to slide
    apart along its axis. This q-only guide removes that offset; it optimizes
    neither contact identities nor force directions. Used only when enabled.
    """
    if projected.shape[1] != 2:
        raise ValueError("Antipodal guidance requires exactly two fixed contact slots")
    chord = projected[:, 1]-projected[:, 0]
    tangent = chord[:, None]-(chord[:, None]*outward).sum(-1, keepdim=True)*outward
    # Coincident projections on one face have zero tangential separation too.
    # They must not be rewarded as an opposed pinch.
    opposition = 10.*(1.+(outward[:,0]*outward[:,1]).sum(-1).clamp(-1.,1.))
    return tangent.square().sum(-1).mean(-1)/scale**2 + opposition


class TemplatePoseObjective:
    """Kinematic candidate scoring; the actual object is never integrated.

    CUDA uses MuJoCo-Warp batched FK/collisions. The CPU path is intended for
    independent geometry regression tests and small diagnostics.
    """

    def __init__(self, mesh_path, settings=None, device="cuda:0", object_mass=.1):
        self.settings = settings or TemplateMPPISettings()
        self.device = torch.device(device)
        if not math.isfinite(object_mass) or object_mass <= 0:
            raise ValueError("object_mass must be finite and positive")
        mesh = trimesh.load_mesh(mesh_path, process=False)
        if isinstance(mesh, trimesh.Trimesh):
            # MuJoCo exports cap/side seams with duplicate vertex indices. Weld
            # identical coordinates only; neither geometry nor units change.
            vertices, inverse = np.unique(mesh.vertices, axis=0, return_inverse=True)
            mesh = trimesh.Trimesh(vertices=vertices, faces=inverse[mesh.faces], process=False)
        if not isinstance(mesh, trimesh.Trimesh) or not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0:
            raise ValueError("Template MPPI requires a closed, outward-oriented object mesh")
        self.center = np.asarray(mesh.center_mass, dtype=float)
        self.radius = max(float(np.linalg.norm(mesh.vertices - self.center, axis=1).max()), .005)
        self.inertia = np.asarray(mesh.moment_inertia, float) * (object_mass / mesh.mass)
        self.closure_solver = ProjectedComFreeClosure(self.settings.closure, center=self.center,
            radius=self.radius, object_mass=object_mass, object_inertia=self.inertia)
        self.xml = refinement_model(mesh_path, self.settings)
        self.cpu = mujoco.MjModel.from_xml_string(self.xml)
        if (self.cpu.nq, self.cpu.nv) != (26, 26):
            raise ValueError("Template MPPI expects a fixed object and 26-coordinate WUJI2")
        surface = ObjectMeshSurface.from_model(self.cpu, self.cpu.body("grasp_object").id)
        self.audit = GraspAudit(surface, str(self.device))
        joints = [np.flatnonzero(self.cpu.jnt_qposadr == i)[0] for i in range(26)]
        self.physical_low = torch.tensor(self.cpu.jnt_range[joints, 0], device=self.device, dtype=torch.float32)
        self.physical_high = torch.tensor(self.cpu.jnt_range[joints, 1], device=self.device, dtype=torch.float32)
        self.low, self.high = self.physical_low.clone(), self.physical_high.clone()
        self.low[6:] += 1.e-6
        self.high[6:] -= 1.e-6
        self.object_id = self.cpu.body("grasp_object").id
        self.physics = None
        self.worlds = 1
        self.parts = {}
        self.template = None
        self.antipodal_weight = self.settings.antipodal_weight
        self.side_projection = None
        self.grasp_posture_bounds = {}
        self.posture = None
        self.trace = []

    def _allocate(self, worlds):
        if self.device.type != "cuda" or (self.physics is not None and self.worlds == worlds):
            return
        self.worlds = int(worlds)
        self.physics = BatchedPhysics(self.cpu, self.worlds, str(self.device),
            nconmax=self.settings.nconmax, capture=False)
        self.object_flags = wp.array((self.cpu.geom_bodyid == self.object_id).astype(np.int32),
                                    dtype=int, device=str(self.device))
        self.depth = wp.zeros((self.worlds, 2), dtype=float, device=str(self.device))
        self.zeros = torch.zeros((self.worlds, 26), device=self.device)

    def set_batch_size(self, worlds):
        if isinstance(worlds, bool) or not isinstance(worlds, int) or worlds < 1:
            raise ValueError("Candidate batch size must be a positive integer")
        self._allocate(worlds)

    def bind(self, template, initial_q):
        if self.settings.antipodal_weight and len(template.body_names) != 2:
            raise ValueError("Antipodal guidance requires exactly two fixed contact slots")
        self.template = template
        self.bound_initial_q = np.asarray(initial_q, dtype=float).copy()
        self.bodies = torch.tensor([self.cpu.body(name).id for name in template.body_names],
                                   dtype=torch.long, device=self.device)
        self.local = torch.tensor(np.asarray(template.local_points).copy(), device=self.device, dtype=torch.float32)
        self.local_normals = torch.tensor(np.asarray(template.local_normals).copy(), device=self.device, dtype=torch.float32)
        self.joint_reference = torch.tensor(np.asarray(template.finger_qpos).copy(), device=self.device, dtype=torch.float32)
        self.required_groups = [torch.tensor([i for i, name in enumerate(template.body_names) if name in group],
                                            device=self.device, dtype=torch.long)
                                for group in template.required_cbody]
        self._search_bounds(initial_q)
        self.parts = {}
        return self

    def _search_bounds(self, q):
        reference = torch.as_tensor(q, device=self.device, dtype=torch.float32)
        if reference.shape != (26,) or not bool(torch.isfinite(reference).all()):
            raise ValueError("Initial hand q must be a finite 26-vector")
        self.qlo, self.qhi = self.low.clone(), self.high.clone()
        radius = reference.new_tensor([self.settings.max_pos_delta] * 3 + [self.settings.max_rot_delta] * 3)
        self.qlo[:6] = torch.maximum(self.qlo[:6], reference[:6] - radius)
        self.qhi[:6] = torch.minimum(self.qhi[:6], reference[:6] + radius)
        if bool((self.qlo > self.qhi).any()):
            raise ValueError("Initial root is outside the physical search bounds")

    def project(self, q):
        return torch.maximum(torch.minimum(q, self.qhi), self.qlo)

    def configure_object_regions(self, name, primitive):
        from .template_mppi_size_ik import CylinderSideProjection
        from .template_mppi_posture import GraspPosture, POSTURE_TYPES
        self.side_projection = None
        self.grasp_posture_bounds = {}
        self.low = self.physical_low.clone()
        self.low[6:] += 1.e-6
        self.high = self.physical_high.clone()
        self.high[6:] -= 1.e-6
        number = int(name.partition('_')[0])
        self.posture = GraspPosture(self.cpu,self.template,number,self.device,self.settings) if number in POSTURE_TYPES else None
        if self.posture is not None:
            self.posture.bounds(self.low,self.high)
        if name.startswith(('6_', '7_')):
            # This is a grasp-family range, not a reference-pose prior. Every
            # flexion joint remains free to close/open across object sizes,
            # but the solver cannot trade a reversed PIP for a fully bent DIP.
            indices = [6+4*finger+joint for finger in range(5) for joint in (0,2,3)]
            reverse = math.radians(self.settings.prismatic_reverse_limit_deg)
            self.low[indices] = self.low[indices].clamp_min(-reverse)
            self.grasp_posture_bounds = dict(policy='prismatic_non_reversed_flexion_v1',
                joint_indices=indices, lower_limits_rad=self.low[indices].cpu().tolist(),
                reference_pose_used=False)
        self._search_bounds(self.bound_initial_q)
        self.antipodal_weight = self.settings.antipodal_weight
        if name.startswith(('9_', '24_')):
            self.antipodal_weight = self.settings.antipodal_weight or (self.settings.type9_antipodal_weight
                if name.startswith('9_') else self.settings.type24_antipodal_weight)
        if primitive and primitive['kind'] in ('cylinder', 'disk') and int(name.partition('_')[0]) in (6, 7, 9, 10):
            surface = ObjectMeshSurface.from_model(self.cpu, self.object_id)
            slots = [i for i, body in enumerate(self.template.body_names) if body.startswith('right_finger')]
            if name.startswith('10_'):
                # A wrapped finger may contact the rim with one phalanx and a
                # cap with the other. Select one side-normal-compatible slot
                # per finger at the canonical grasp, then keep those identities
                # at every size. Requiring both phalanges on a thin wall is an
                # unintended and often impossible extra constraint.
                if self.device.type == 'cuda':
                    from .template import WujiContactKinematics
                    fk = WujiContactKinematics(self.cpu, self.bodies.cpu().numpy(), self.device)
                    with torch.no_grad():
                        points, normals = fk.contacts(fk.tensor(self.bound_initial_q),
                            fk.tensor(self.template.local_points), fk.tensor(self.template.local_normals))
                        points, normals = points.cpu().numpy(), normals.cpu().numpy()
                else:
                    data = mujoco.MjData(self.cpu)
                    data.qpos[:] = self.bound_initial_q
                    mujoco.mj_kinematics(self.cpu, data)
                    ids = self.bodies.numpy()
                    rotations = data.xmat[ids].reshape(-1,3,3)
                    points = data.xpos[ids]+np.einsum('nij,nj->ni',rotations,self.template.local_points)
                    normals = np.einsum('nij,nj->ni',rotations,self.template.local_normals)
                radial = points.copy()
                radial[:,2] = 0.
                radial /= np.maximum(np.linalg.norm(radial,axis=-1,keepdims=True),1.e-12)
                alignment = -(radial*normals).sum(-1)
                chosen = []
                for finger in range(1,6):
                    group = [i for i in slots if self.template.body_names[i].startswith(f'right_finger{finger}_')]
                    if group:
                        chosen.append(max(group,key=lambda i:alignment[i]))
                slots = sorted(chosen)
            if slots:
                self.side_projection = CylinderSideProjection(surface, self.device, slots)

    def initialize_size(self, reference, primitive, q, settings, seed):
        from .template_mppi_size_ik import initialize_size
        return initialize_size(self, reference, primitive, q, settings, seed)

    def _geometry(self, q):
        if self.device.type == "cuda":
            self._allocate(self.worlds)
            self.physics.reset(q, self.zeros, control=q)
            mjw.kinematics(self.physics.model, self.physics.data)
            mjw.collision(self.physics.model, self.physics.data)
            self.depth.zero_()
            wp.launch(_depths, self.physics.data.naconmax, inputs=[self.physics.data.nacon,
                self.physics.data.contact.worldid, self.physics.data.contact.geom,
                self.physics.data.contact.dist, self.object_flags,
                self.settings.collision_clearance_m, self.depth], device=str(self.device))
            p, r = self.physics.xpos[:, self.bodies], self.physics.xmat[:, self.bodies]
            depths = wp.to_torch(self.depth).clone()
            overflow = wp.to_torch(self.physics.data.overflow) != 0
        else:
            positions, rotations, depths = [], [], []
            for value in q.detach().cpu().numpy():
                data = mujoco.MjData(self.cpu)
                data.qpos[:] = value
                mujoco.mj_kinematics(self.cpu, data)
                mujoco.mj_collision(self.cpu, data)
                positions.append(data.xpos[self.bodies.numpy()].copy())
                rotations.append(data.xmat[self.bodies.numpy()].reshape(-1, 3, 3).copy())
                depth = np.zeros(2)
                for contact in data.contact:
                    if np.all(self.cpu.geom_bodyid[contact.geom] == self.object_id):
                        continue
                    kind = 0 if self.object_id in self.cpu.geom_bodyid[contact.geom] else 1
                    depth[kind] = max(depth[kind], self.settings.collision_clearance_m - contact.dist)
                depths.append(depth)
            p, r, depths = [torch.as_tensor(np.asarray(x), device=self.device, dtype=q.dtype)
                             for x in (positions, rotations, depths)]
            overflow = torch.zeros(len(q), device=self.device, dtype=torch.bool)
        world = p + torch.einsum("bcij,cj->bci", r, self.local)
        normal_hand = torch.einsum("bcij,cj->bci", r, self.local_normals)
        projected, outward, distance = self.audit.query(world.flatten(0, 1))
        projected, outward, distance = projected.reshape_as(world), outward.reshape_as(world), distance.reshape(len(q), -1)
        if self.side_projection is not None:
            projected, outward, distance = self.side_projection.apply(world, projected, outward, distance)
        return world, normal_hand, projected, outward, distance, depths, overflow

    @torch.no_grad()
    def evaluate_initialization(self, controls):
        """Rank root-only matcher proposals before optimizing contact forces.

        Shared FK, projections and geometry scales are identical to the full
        objective. Closure and force-weighted contact costs are excluded because
        the template fingers have not yet been adapted by MPPI.
        """
        if self.template is None:
            raise RuntimeError("Bind an immutable template before scoring")
        if controls.ndim != 3 or controls.shape[1:] != (1, 26) or not len(controls):
            raise ValueError("Template MPPI candidates must have shape (B,1,26)")
        controls = controls.to(device=self.device, dtype=torch.float32)
        results, collected = [], {}
        stream = (wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(self.device)))
                  if self.device.type == "cuda" else nullcontext())
        with wp.ScopedDevice(str(self.device)), stream:
            for start in range(0, len(controls), self.worlds):
                original = controls[start:start + self.worlds, 0]
                count = len(original)
                valid_q = torch.isfinite(original).all(1) & (original >= self.low - 1.e-6).all(1) & (original <= self.high + 1.e-6).all(1)
                q = torch.where(valid_q[:, None], original, self.low)
                if self.device.type == "cuda" and count < self.worlds:
                    q = torch.cat((q, q[:1].expand(self.worlds - count, -1)))
                world, hand_normal, projected, outward, gap, depth, overflow = (
                    value[:count] for value in self._geometry(q))
                q = q[:count]
                terms = contact_terms(gap, hand_normal, outward, self.settings)
                required_group, required_fitted = required_contact_terms(terms, self.required_groups)
                valid_projection = torch.isfinite(projected).all((1, 2)) & torch.isfinite(outward).all((1, 2)) & torch.isfinite(gap).all(1)
                penetration, soft_penetration, _ = penetration_terms(depth, self.settings)
                pose = ((q[:, 6:] - self.joint_reference) / self.settings.joint_prior_scale_rad).square().mean(1)
                articulation, crossing, _ = self.posture.batch(q) if self.posture is not None else (q.new_zeros(count),q.new_zeros(count),None)
                cost = (self.settings.distance_weight * terms["distance"]
                        + self.settings.normal_weight * terms["normal"]
                        + self.settings.required_group_weight * required_group
                        + self.settings.penetration_weight * penetration
                        + self.settings.soft_penetration_weight * soft_penetration
                        + self.settings.pose_weight * pose + articulation)
                valid = valid_q & valid_projection & ~overflow & torch.isfinite(cost) & (crossing <= self.settings.penetration_tolerance_m)
                results.append(torch.where(valid, cost, torch.inf))
                batch = dict(distance=terms["distance"], normal=terms["normal"],
                             required_group=required_group, required_groups_fitted=required_fitted,
                             penetration=penetration, soft_penetration=soft_penetration,
                             pose=pose, articulation=articulation, crossing_m=crossing, coverage=terms["coverage"], valid=valid)
                for key, value in batch.items():
                    collected.setdefault(key, []).append(value.detach().clone())
        self.parts = {key: torch.cat(values) for key, values in collected.items()}
        return torch.cat(results)

    @torch.no_grad()
    def evaluate(self, controls):
        if self.template is None:
            raise RuntimeError("Bind an immutable template before scoring")
        if controls.ndim != 3 or controls.shape[1:] != (1, 26) or not len(controls):
            raise ValueError("Template MPPI candidates must have shape (B,1,26)")
        controls = controls.to(device=self.device, dtype=torch.float32)
        results, collected = [], {}
        stream = (wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(self.device)))
                  if self.device.type == "cuda" else nullcontext())
        with wp.ScopedDevice(str(self.device)), stream:
            for start in range(0, len(controls), self.worlds):
                original = controls[start:start + self.worlds, 0]
                count = len(original)
                valid_q = torch.isfinite(original).all(1) & (original >= self.low - 1.e-6).all(1) & (original <= self.high + 1.e-6).all(1)
                q = torch.where(valid_q[:, None], original, self.low)
                if self.device.type == "cuda" and count < self.worlds:
                    q = torch.cat((q, q[:1].expand(self.worlds - count, -1)))
                # Native FK/collision buffers retain their fixed world count.
                # The inner solve must only see real candidates, especially
                # for singleton mean guards and final diagnostics.
                world, hand_normal, projected, outward, gap, depth, overflow = (
                    value[:count] for value in self._geometry(q))
                q = q[:count]
                terms = contact_terms(gap, hand_normal, outward, self.settings)
                required_group, required_fitted = required_contact_terms(terms, self.required_groups)
                valid_projection = torch.isfinite(projected).all((1, 2)) & torch.isfinite(outward).all((1, 2)) & torch.isfinite(gap).all(1)
                # Keep invalid geometry out of the n-dimensional numerical solve.
                safe_projected = torch.where(valid_projection[:, None, None], projected, 0.)
                safe_outward = torch.where(valid_projection[:, None, None], outward,
                                           outward.new_tensor([1., 0., 0.]))
                closure = self.closure_solver.solve(safe_projected, safe_outward)
                antipodal = (antipodal_contact_term(safe_projected, safe_outward, self.settings.antipodal_scale_m)
                             if self.antipodal_weight else gap.new_zeros(count))
                force_contact, fitted_force_fraction = force_contact_terms(
                    terms["slot_error"], terms["fitted"], closure.row_forces)
                penetration, soft_penetration, excess = penetration_terms(depth, self.settings)
                pose = ((q[:, 6:] - self.joint_reference) / self.settings.joint_prior_scale_rad).square().mean(1)
                articulation, crossing, _ = self.posture.batch(q) if self.posture is not None else (q.new_zeros(count),q.new_zeros(count),None)
                cost = (self.settings.distance_weight * terms["distance"] + self.settings.normal_weight * terms["normal"]
                        + self.settings.closure_weight * closure.cost + self.settings.penetration_weight * penetration
                        + self.settings.soft_penetration_weight * soft_penetration
                        + self.settings.pose_weight * pose + self.settings.required_group_weight * required_group
                        + self.settings.force_contact_weight * force_contact
                        + self.antipodal_weight * antipodal + articulation)
                valid = valid_q & valid_projection & closure.valid & ~overflow & torch.isfinite(cost) & (crossing <= self.settings.penetration_tolerance_m)
                results.append(torch.where(valid, cost, torch.inf))
                batch = dict(distance=terms["distance"], normal=terms["normal"], required_group=required_group,
                    required_groups_fitted=required_fitted, force_contact=force_contact,
                    fitted_force_fraction=fitted_force_fraction, closure=closure.cost,
                    antipodal=antipodal, articulation=articulation, crossing_m=crossing,
                    penetration=penetration, soft_penetration=soft_penetration,
                    hand_object_excess_m=excess, pose=pose, coverage=terms["coverage"],
                    hand_points=world, hand_normals=hand_normal, projected_points=projected,
                    object_normals=outward, signed_distance_m=gap,
                    selected_slots=terms["selected"], fitted=terms["fitted"], normal_cosine=terms["cosine"],
                    clearance_deficit_m=depth,
                    penetration_depth_m=(depth - self.settings.collision_clearance_m).clamp_min(0.),
                    preloads=closure.preloads, response_row_forces=closure.row_forces,
                    scenario_residuals=closure.scenario_residuals, scenario_rms=closure.scenario_rms,
                    closure_converged=closure.converged, closure_certified=closure.certified,
                    closure_margin=closure.certified_margin)
                if closure.objective_history is not None:
                    batch["gn_objective_history"] = closure.objective_history
                batch["valid"] = valid
                for key, value in batch.items():
                    collected.setdefault(key, []).append(value.detach().clone())
        self.parts = {key: torch.cat(values) for key, values in collected.items()}
        return torch.cat(results)

    def diagnostics(self, q):
        value = torch.as_tensor(q, device=self.device, dtype=torch.float32)
        cost = self.evaluate(value.reshape(1, 1, 26))[0]
        parts = {key: item[0].cpu().numpy() for key, item in self.parts.items()}
        cost_parts = {key: float(parts[key]) for key in ("distance", "normal", "required_group", "force_contact", "closure", "penetration", "soft_penetration", "pose")}
        if self.antipodal_weight:
            cost_parts["antipodal"] = float(parts["antipodal"])
        if self.posture is not None:
            cost_parts['articulation'] = float(parts['articulation'])
        depth = parts["penetration_depth_m"]
        normal_force = parts["response_row_forces"].sum(-1)
        contact_force, intrinsic_torque = self.closure_solver.contact_wrenches(
            self.parts["response_row_forces"], self.parts["object_normals"])
        inward = -torch.nn.functional.normalize(self.parts["object_normals"], dim=-1)
        torsional_torque = (intrinsic_torque * inward[:, None]).sum(-1)[0].cpu().numpy()
        accepted = acceptance(parts["coverage"], parts["closure_certified"], parts["valid"], depth, self.settings)
        return dict(objective=OBJECTIVE, cost=float(cost), cost_parts=cost_parts,
            search_dimension=26, preload_dimension=len(self.local), contact_count=len(self.local),
            fk_backend="mujoco_warp_cuda_batched" if self.device.type == "cuda" else "mujoco_cpu_reference",
            side_contact_slots=self.side_projection.slots.cpu().tolist() if self.side_projection is not None else [],
            side_contact_policy='one_canonical_slot_per_finger' if self.side_projection is not None and len(self.local)>5 else 'all_specified_slots',
            effective_antipodal_weight=self.antipodal_weight,
            grasp_posture_bounds=self.grasp_posture_bounds,
            articulation_policy=self.posture.metadata() if self.posture is not None else None,
            max_finger_order_violation_m=float(parts['crossing_m']),
            optimization_joint_lower_limits=self.low.cpu().tolist(),
            optimization_joint_upper_limits=self.high.cpu().tolist(),
            **accepted, success_criterion=SUCCESS_CRITERION, coverage=float(parts["coverage"]),
            contact_model="ComFree compliant projected soft-finger response with softly penalized hand-object compression",
            hand_object_allowance_m=self.settings.hand_object_allowance_m,
            hand_hand_penetration_tolerance_m=self.settings.penetration_tolerance_m,
            max_hand_object_excess_m=float(parts["hand_object_excess_m"]),
            coverage_required_for_success=False,
            hand_object_depth_required_for_success=False,
            selected_slots=parts["selected_slots"].tolist(), fitted=parts["fitted"].tolist(),
            required_groups_fitted=parts["required_groups_fitted"].tolist(),
            fitted_force_fraction=parts["fitted_force_fraction"].tolist(),
            hand_points=parts["hand_points"].tolist(), hand_normals=parts["hand_normals"].tolist(),
            projected_points=parts["projected_points"].tolist(), object_normals=parts["object_normals"].tolist(),
            signed_distance_m=parts["signed_distance_m"].tolist(),
            normal_error_deg=np.rad2deg(np.arccos(np.clip(parts["normal_cosine"], -1., 1.))).tolist(),
            max_hand_object_penetration_m=float(depth[0]), max_hand_hand_penetration_m=float(depth[1]),
            nominal_preloads_n=parts["preloads"].tolist(), response_row_forces_n=parts["response_row_forces"].tolist(),
            response_normal_forces_n=normal_force.tolist(), max_response_normal_force_n=float(normal_force.max()),
            response_contact_forces_n=contact_force[0].cpu().tolist(),
            response_intrinsic_torques_nm=intrinsic_torque[0].cpu().tolist(),
            response_torsional_torques_nm=torsional_torque.tolist(),
            max_response_torsional_torque_nm=float(np.abs(torsional_torque).max()),
            torsional_friction_m=self.settings.closure.torsional_friction_m,
            intrinsic_torque_convention="free contact moment only; excludes moment arm cross contact force",
            scenario_residuals=parts["scenario_residuals"].tolist(), scenario_rms=parts["scenario_rms"].tolist(),
            closure_converged=bool(parts["closure_converged"]), projected_closure=bool(parts["closure_certified"]),
            certified_margin=float(parts["closure_margin"]), valid=bool(parts["valid"]),
            gn_objective_history=parts.get("gn_objective_history", np.empty(0)).tolist(),
            closure_model=getattr(self.closure_solver, "metadata", {}),
            settings=asdict(self.settings),
            actual_contact_required_for_projection=False,
            actual_object_integrated=False,
            acceptance="Projected friction closure and separate self-penetration limit; hand-object compression softly penalized; coverage diagnostic only; no native physical evaluation")

    def patch_diagnostics(self, q):
        """Report anatomical patch coverage after optimization, without forces."""
        from .template_mppi_patch_coverage import diagnose_patch_coverage
        return diagnose_patch_coverage(self.cpu, self.template, q, self.audit,
            self.settings, device=self.device,
            side_bodies=[self.template.body_names[i] for i in self.side_projection.slots.cpu().tolist()] if self.side_projection is not None else (),
            side_audit=self.side_projection.audit if self.side_projection is not None else None)

    def final_diagnostics(self, q):
        """Refine only final preloads, retaining any existing force witness."""
        fast = self.diagnostics(q)
        fast_parts = self.parts
        original = self.closure_solver
        if original.settings.gn_iterations >= 8:
            fast["preload_polish"] = dict(performed=False, q_changed=False)
            return fast
        settings = replace(original.settings, gn_iterations=8, cuda_graph=False)
        self.closure_solver = ProjectedComFreeClosure(settings, center=self.center,
            radius=self.radius, object_mass=original.object_mass, object_inertia=self.inertia)
        started = time.perf_counter()
        try:
            accurate = self.diagnostics(q)
        finally:
            self.closure_solver = original
        # A valid actual-force certificate is stronger evidence than an
        # unconverged larger-budget solve. Never discard that certificate.
        certificate_preserved = fast["projected_closure"] and not accurate["projected_closure"]
        preserve = fast["valid"] and (not accurate["valid"] or certificate_preserved
            or (fast["projected_closure"] == accurate["projected_closure"] and fast["cost"] < accurate["cost"]))
        chosen = fast if preserve else accurate
        if preserve:
            self.parts = fast_parts
        chosen["preload_polish"] = dict(performed=True, q_changed=False,
            optimization_gn_iterations=original.settings.gn_iterations, final_gn_iterations=8,
            seconds=time.perf_counter() - started, certificate_preserved=certificate_preserved,
            fast_result_retained=preserve,
            before_projected_closure=fast["projected_closure"],
            after_projected_closure=accurate["projected_closure"],
            before_cost=fast["cost"], after_cost=accurate["cost"])
        return chosen

    def optimize(self, initial_q, solver, seed):
        if solver.type != "mppi" or solver.gradient_steps:
            raise ValueError("Template MPPI requires MPPI with no outer gradient refinement")
        self._search_bounds(initial_q)
        self._allocate(solver.worlds)
        started = time.perf_counter()
        mean = self.project(torch.as_tensor(initial_q, device=self.device, dtype=torch.float32))[None]
        initial = self.diagnostics(mean[0])
        trajectory = [mean[0].cpu().numpy().copy()]
        noise = mean.new_tensor([self.settings.pos_noise] * 3 + [self.settings.rot_noise] * 3 + [self.settings.joint_noise] * 20)
        history = []
        # The sampler averages bounded absolute q values; projection also covers
        # each sampled candidate before it is used in the weighted update.
        def bounded_evaluate(candidates):
            candidates.copy_(self.project(candidates))
            return self.evaluate(candidates)
        with torch.no_grad():
            for stage in range(self.settings.stages):
                mean, detail = sample_mppi(mean, bounded_evaluate, solver, knots=1, knot_steps=1,
                    channel_noise=noise * self.settings.final_stage_noise ** stage,
                    seed=int(seed) + 104729 * stage, first_noise=1., last_noise=1., return_best=True,
                    guard_mean=self.settings.guard_mean)
                mean = self.project(mean)
                trajectory.append(mean[0].cpu().numpy().copy())
                history.append(dict(stage=stage, **detail))
        final = self.diagnostics(mean[0])
        if final["cost"] > initial["cost"] + 1.e-5:
            raise RuntimeError("MPPI increased its own best-candidate objective")
        self.trace = np.asarray(trajectory)
        return mean[0].cpu().numpy().astype(float), dict(objective=OBJECTIVE,
            initial_cost=initial["cost"], final_cost=final["cost"],
            initial_parts=initial["cost_parts"], final_parts=final["cost_parts"],
            initial_diagnostics=initial, final_diagnostics=final,
            optimization_seconds=time.perf_counter() - started, history=history,
            sampled=sum(item["sampled"] for item in history), seed=int(seed),
            search_dimension=26, preload_dimension=len(self.local), trajectory=self.trace,
            settings=asdict(self.settings), acceptance_during_optimization=False)
