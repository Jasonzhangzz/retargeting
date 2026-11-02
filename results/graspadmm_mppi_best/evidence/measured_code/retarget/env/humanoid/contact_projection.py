"""Reference-hand contact candidates, projected onto object geometry by Warp.

The candidates are stored in the object root body's frame. Reference forward
kinematics uses separate MJWarp data, so neither querying the dataset nor
sampling a new contact changes the live simulator state.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
import math

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from retarget.task.humanoid.cost_contact import _stream


def _torch_stream(device):
    if not wp.get_device(device).is_cuda:
        return nullcontext()
    current = torch.cuda.current_stream(device)
    warp_stream = wp.get_stream(device)
    if warp_stream.cuda_stream != current.cuda_stream:
        warp_stream = wp.stream_from_torch(current)
    return wp.ScopedStream(warp_stream)


@dataclass(frozen=True)
class ContactPointCandidates:
    """One nearest candidate plus neighbours for every reference hand and frame."""

    local_points: torch.Tensor  # (H, hands, candidates, 3)
    local_normals: torch.Tensor  # outward unit normals, same shape
    reference_hand_positions: torch.Tensor  # (H, hands, 3), physical hand surface, world frame
    hand_site_ids: tuple[int, ...]
    object_body_id: int
    object_qpos_adr: int
    object_dof_adr: int
    robot_local_points: torch.Tensor | None = None  # (H, hands, 3), in each site's BODY frame


@wp.kernel(enable_backward=False)
def _project_mesh(
    mesh: wp.uint64,
    queries: wp.array(dtype=wp.vec3),
    projected: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3),
    valid: wp.array(dtype=int),
):
    i = wp.tid()
    query = wp.mesh_query_point_no_sign(mesh, queries[i], 1.0e10)
    if query.result:
        projected[i] = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
        normals[i] = wp.mesh_eval_face_normal(mesh, query.face)
        valid[i] = 1


def _object_joint(cpu, body):
    free = np.flatnonzero(cpu.jnt_type == int(mujoco.mjtJoint.mjJNT_FREE))
    if body is None:
        if len(free) < 2:
            raise ValueError("SCSP requires separate robot and object free joints")
        joint = int(free[-1])
        body = int(cpu.jnt_bodyid[joint])
    else:
        if isinstance(body, bool) or not isinstance(body, (int, np.integer)) or not 0 < body < cpu.nbody:
            raise ValueError("object_body_id must identify a non-world body")
        joints = free[cpu.jnt_bodyid[free] == body]
        if len(joints) != 1:
            raise ValueError("object_body_id must identify a free-joint body")
        joint = int(joints[0])
    return int(body), int(cpu.jnt_qposadr[joint]), int(cpu.jnt_dofadr[joint])


def _surface_mesh(cpu, object_body):
    """Collect box/mesh collision geoms, including MuJoCo's explicit pairs.

    Transforms include fixed descendants and geom-local position/rotation. Mesh
    winding is repaired per geom before combining disconnected components, so
    normals remain defined even when a query lies exactly on the surface.
    """
    import trimesh

    data = mujoco.MjData(cpu)
    mujoco.mj_kinematics(cpu, data)
    root_rotation = data.xmat[object_body].reshape(3, 3)
    root_position = data.xpos[object_body]
    subtree = np.zeros(cpu.nbody, bool)
    subtree[object_body] = True
    for body in range(object_body + 1, cpu.nbody):
        subtree[body] = subtree[cpu.body_parentid[body]]
        if subtree[body] and cpu.body_jntnum[body]:
            raise ValueError("SCSP object geometry must be a rigid body subtree")
    paired = set(np.asarray(cpu.pair_geom1, int)) | set(np.asarray(cpu.pair_geom2, int))
    geoms = [
        geom for geom in range(cpu.ngeom)
        if subtree[cpu.geom_bodyid[geom]]
        and (cpu.geom_contype[geom] or cpu.geom_conaffinity[geom] or geom in paired)
    ]
    if not geoms:
        raise ValueError("SCSP object requires box or mesh collision geometry")
    vertices, faces = [], []
    offset = 0
    for geom in geoms:
        kind = int(cpu.geom_type[geom])
        if kind == int(mujoco.mjtGeom.mjGEOM_BOX):
            mesh = trimesh.creation.box(extents=2.0 * cpu.geom_size[geom])
        elif kind == int(mujoco.mjtGeom.mjGEOM_MESH):
            mesh_id = int(cpu.geom_dataid[geom])
            start, count = int(cpu.mesh_vertadr[mesh_id]), int(cpu.mesh_vertnum[mesh_id])
            points = cpu.mesh_vert[start : start + count].copy()
            start, count = int(cpu.mesh_faceadr[mesh_id]), int(cpu.mesh_facenum[mesh_id])
            triangles = cpu.mesh_face[start : start + count].copy()
            mesh = trimesh.Trimesh(vertices=points, faces=triangles, process=True)
            mesh.fix_normals(multibody=True)
        else:
            name = mujoco.mj_id2name(cpu, mujoco.mjtObj.mjOBJ_GEOM, geom)
            raise ValueError(f"SCSP object collision geom {name or geom!r} must be box or mesh")
        local = np.asarray(mesh.vertices)
        world = local @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        vertices.append((world - root_position) @ root_rotation)
        faces.append(np.asarray(mesh.faces) + offset)
        offset += len(local)
    return np.concatenate(vertices).astype(np.float32), np.concatenate(faces).astype(np.int32)


class ContactProjector:
    """GPU/CPU Warp surface queries for one rigid box or mesh object."""

    def __init__(self, cpu, object_body_id, *, device):
        vertices, faces = _surface_mesh(cpu, object_body_id)
        self._initialize_mesh(vertices, faces, device)

    def _initialize_mesh(self, vertices, faces, device):
        self.device = str(device)
        with wp.ScopedDevice(self.device), _torch_stream(self.device):
            self.mesh = wp.Mesh(
                points=wp.array(vertices, dtype=wp.vec3, device=self.device),
                indices=wp.array(faces.reshape(-1), dtype=int, device=self.device),
            )

    @classmethod
    def from_vertices(cls, vertices, faces, *, device):
        """A geom-local triangle query, used for robot mesh contact surfaces."""
        instance = cls.__new__(cls)
        instance._initialize_mesh(np.asarray(vertices, np.float32), np.asarray(faces, np.int32), device)
        return instance

    def project(self, local_queries):
        """Return nearest surface points and outward face normals, same shape."""
        if local_queries.shape[-1] != 3 or local_queries.numel() == 0:
            raise ValueError("projection queries must be nonempty 3D points")
        query = local_queries.detach().to(device=self.device, dtype=torch.float32).contiguous()
        if not bool(torch.isfinite(query).all()):
            raise ValueError("projection queries must be finite")
        flat = query.reshape(-1, 3)
        points = torch.empty_like(flat)
        normals = torch.empty_like(flat)
        valid = torch.zeros(len(flat), device=flat.device, dtype=torch.int32)
        with wp.ScopedDevice(self.device), _torch_stream(self.device):
            wp.launch(
                _project_mesh,
                dim=len(flat),
                inputs=[self.mesh.id, wp.from_torch(flat, dtype=wp.vec3)],
                outputs=[wp.from_torch(points, dtype=wp.vec3), wp.from_torch(normals, dtype=wp.vec3), wp.from_torch(valid)],
                device=self.device,
            )
        if not bool(valid.all()):
            raise ValueError("Warp could not project a hand query onto object geometry")
        return points.reshape(query.shape), normals.reshape(query.shape)

    def candidates(self, local_queries, *, radius=0.03, neighbors=10, seed=0, shared_over_time=False):
        """Nearest point, then tangent-disk projections with optional shared offsets.

        For a (H, K, 3) reference sequence, ``shared_over_time=True`` reuses each
        hand/candidate's polar offset over H. A fixed discrete candidate index
        then follows the same offset as its nearest reference projection moves.
        """
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("contact sampling radius must be finite and positive")
        if isinstance(neighbors, bool) or not isinstance(neighbors, int) or neighbors < 0:
            raise ValueError("contact neighbors must be a nonnegative integer")
        if not isinstance(shared_over_time, bool):
            raise ValueError("shared_over_time must be a boolean")
        if shared_over_time and local_queries.ndim < 2:
            raise ValueError("shared_over_time requires a leading time dimension")
        anchor, normal = self.project(local_queries)
        if neighbors == 0:
            return anchor.unsqueeze(-2), normal.unsqueeze(-2)
        axis = torch.zeros_like(normal)
        axis[..., 0] = 1.0
        axis = torch.where(normal[..., :1].abs() > 0.9, axis.roll(1, -1), axis)
        tangent1 = torch.linalg.cross(normal, axis, dim=-1)
        tangent1 = torch.nn.functional.normalize(tangent1, dim=-1)
        tangent2 = torch.linalg.cross(normal, tangent1, dim=-1)
        generator = torch.Generator(device=anchor.device).manual_seed(seed)
        sample_shape = ((1, *anchor.shape[1:-1]) if shared_over_time else anchor.shape[:-1])
        random = torch.rand((*sample_shape, neighbors, 2), generator=generator, device=anchor.device)
        distance = radius * torch.sqrt(random[..., :1])
        angle = 2.0 * torch.pi * random[..., 1:]
        offset = distance * (torch.cos(angle) * tangent1.unsqueeze(-2) + torch.sin(angle) * tangent2.unsqueeze(-2))
        extra, extra_normals = self.project(anchor.unsqueeze(-2) + offset)
        return torch.cat((anchor.unsqueeze(-2), extra), -2), torch.cat((normal.unsqueeze(-2), extra_normals), -2)


def _hand_collision_geoms(cpu, site, object_body):
    """Collision geoms rigidly attached to this hand, including explicit pairs."""
    hand_body = int(cpu.site_bodyid[site])
    weld = int(cpu.body_weldid[hand_body])
    paired = set(np.asarray(cpu.pair_geom1, int)) | set(np.asarray(cpu.pair_geom2, int))
    geoms = [geom for geom in range(cpu.ngeom)
             if int(cpu.body_weldid[cpu.geom_bodyid[geom]]) == weld
             and (cpu.geom_contype[geom] or cpu.geom_conaffinity[geom] or geom in paired)]
    if not geoms:
        name = mujoco.mj_id2name(cpu, mujoco.mjtObj.mjOBJ_SITE, site)
        raise ValueError(f"Hand site {name or site!r} has no rigidly attached collision geometry; set use_hand_surface=False for an explicit site contact")
    return geoms


def _reference_hand_surfaces(cpu, scratch, site_ids, object_body, object_position, object_rotation, projector, *, device):
    """Nearest hand/object surface pairs in reference poses.

    Sphere queries are exact, using the nearest object point to the sphere
    centre. Box and mesh pairs use three alternating nearest projections, with
    every rigid hand collision geom considered independently before taking the
    closest pair. Object/mesh projections run in Warp; sphere/box projections
    are batched tensor operations on the same device.
    """
    geom_position = wp.to_torch(scratch.geom_xpos)
    geom_rotation = wp.to_torch(scratch.geom_xmat)

    def object_project(world):
        local = torch.einsum("hji,hj->hi", object_rotation, world - object_position)
        point, _ = projector.project(local)
        return object_position + torch.einsum("hij,hj->hi", object_rotation, point)

    hands = []
    for site in site_ids:
        points, distances = [], []
        for geom in _hand_collision_geoms(cpu, site, object_body):
            position, rotation = geom_position[:, geom], geom_rotation[:, geom]
            kind = int(cpu.geom_type[geom])
            mesh_projector = None
            if kind == int(mujoco.mjtGeom.mjGEOM_MESH):
                mesh = int(cpu.geom_dataid[geom])
                start, count = int(cpu.mesh_vertadr[mesh]), int(cpu.mesh_vertnum[mesh])
                vertices = cpu.mesh_vert[start:start+count]
                start, count = int(cpu.mesh_faceadr[mesh]), int(cpu.mesh_facenum[mesh])
                mesh_projector = ContactProjector.from_vertices(vertices, cpu.mesh_face[start:start+count], device=device)
            elif kind not in (int(mujoco.mjtGeom.mjGEOM_SPHERE), int(mujoco.mjtGeom.mjGEOM_BOX)):
                name = mujoco.mj_id2name(cpu, mujoco.mjtObj.mjOBJ_GEOM, geom)
                raise ValueError(f"Hand collision geom {name or geom!r} must be sphere, box or mesh; set use_hand_surface=False for an explicit site contact")

            def hand_project(world):
                query = torch.einsum("hji,hj->hi", rotation, world-position)
                if kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                    length = query.norm(dim=-1, keepdim=True)
                    fallback = torch.zeros_like(query)
                    fallback[:, 0] = 1.
                    direction = torch.where(length > 1e-12, query/length.clamp_min(1e-12), fallback)
                    local = float(cpu.geom_size[geom, 0]) * direction
                elif kind == int(mujoco.mjtGeom.mjGEOM_BOX):
                    extent = torch.as_tensor(cpu.geom_size[geom].copy(), dtype=query.dtype, device=query.device)
                    local = torch.maximum(torch.minimum(query, extent), -extent)
                    # Queries inside a box need its closest face, rather than
                    # remaining at an interior clamped point.
                    face = (extent-query.abs()).argmin(-1, keepdim=True)
                    sign = torch.where(query.gather(-1, face) < 0, -1., 1.)
                    surface = local.scatter(-1, face, extent[None].expand_as(query).gather(-1, face)*sign)
                    local = torch.where((query.abs() <= extent).all(-1, keepdim=True), surface, local)
                else:
                    local, _ = mesh_projector.project(query)
                return position + torch.einsum("hij,hj->hi", rotation, local)

            target = object_project(position)
            hand = hand_project(target)
            # Projecting the centre gives the global nearest pair for a
            # disjoint sphere, so retain that exact radial surface point.
            if kind != int(mujoco.mjtGeom.mjGEOM_SPHERE):
                for _ in range(3):
                    target = object_project(hand)
                    hand = hand_project(target)
            target = object_project(hand)
            points.append(hand)
            distances.append((hand-target).square().sum(-1))
        candidates = torch.stack(points, 1)
        choice = torch.stack(distances, 1).argmin(-1)
        hands.append(candidates[torch.arange(len(candidates), device=candidates.device), choice])
    return torch.stack(hands, 1)


def build_contact_candidates(
    config,
    env,
    reference_qpos,
    hand_site_ids,
    *,
    object_body_id=None,
    radius=0.03,
    neighbors=10,
    seed=0,
    use_hand_surface=True,
):
    """Project dataset hand collision surfaces for every reference frame.

    ``reference_qpos`` has shape (H, nq); no frames are dropped. The returned
    (H, K, 1+neighbors, 3) arrays are detached, because discrete contact search
    uses them as fixed data. The robot optimization differentiates its current
    world contact positions through the cached contact Jacobians instead.
    ``robot_local_points`` stores the selected physical hand point in each
    configured site's body frame, for consistent downstream contact Jacobians.
    ``use_hand_surface=False`` explicitly treats the sites as contact points.
    """
    cpu = env.model_cpu
    if not isinstance(use_hand_surface, bool):
        raise ValueError("use_hand_surface must be a boolean")
    ids = tuple(int(i) for i in hand_site_ids)
    if not ids or len(set(ids)) != len(ids) or any(i < 0 or i >= cpu.nsite for i in ids):
        raise ValueError("hand_site_ids must contain distinct valid hand sites")
    body, qadr, vadr = _object_joint(cpu, object_body_id)
    qpos = torch.as_tensor(reference_qpos, device=config.device, dtype=torch.float32)
    if qpos.ndim != 2 or qpos.shape[1] != cpu.nq or len(qpos) == 0 or not bool(torch.isfinite(qpos).all()):
        raise ValueError("reference_qpos must be a finite nonempty (H, nq) tensor")
    with wp.ScopedDevice(config.device), _stream(env):
        scratch = mjw.make_data(cpu, nworld=len(qpos), nconmax=1, njmax=1)
        wp.to_torch(scratch.qpos).copy_(qpos)
        mjw.kinematics(env.model_wp, scratch)
        hands = wp.to_torch(scratch.site_xpos)[:, ids].clone()
        object_position = wp.to_torch(scratch.xpos)[:, body].clone()
        object_rotation = wp.to_torch(scratch.xmat)[:, body].clone()
        projector = ContactProjector(cpu, body, device=config.device)
        if use_hand_surface:
            hands = _reference_hand_surfaces(cpu, scratch, ids, body, object_position, object_rotation, projector, device=config.device)
        hand_bodies = torch.as_tensor(cpu.site_bodyid[list(ids)].copy(), device=qpos.device, dtype=torch.long)
        hand_position = wp.to_torch(scratch.xpos)[:, hand_bodies]
        hand_rotation = wp.to_torch(scratch.xmat)[:, hand_bodies]
        robot_local = torch.einsum("hkji,hkj->hki", hand_rotation, hands-hand_position)
        # Matrices transform object-local columns to world-space columns.
        local_hand = torch.einsum("hji,hkj->hki", object_rotation, hands - object_position[:, None])
        points, normals = projector.candidates(local_hand, radius=radius, neighbors=neighbors, seed=seed, shared_over_time=True)
    return ContactPointCandidates(points, normals, hands, ids, body, qadr, vadr, robot_local)
