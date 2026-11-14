"""GPU MANO decoding, Shadow initialization IK, immutable contacts and CEM bounds."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import collections, hashlib, math
import numpy as np
import torch
import warp as wp
from scipy.spatial.transform import Rotation
from scripts.calibrate_wuji_mano import PersonalizedMANO, rotmat, CHAINS, TIPS


from retarget.solver.cem import InitializationSettings as Settings,DEFAULTS


class ManoDecoder(torch.nn.Module):
    """Exact MANO with parallel finger chains and a five-tip subset for LM IK."""

    def __init__(self, path, device="cpu", components=DEFAULTS.initialization.pca_components):
        super().__init__()
        self.base = PersonalizedMANO(path).to(device=device, dtype=torch.float32)
        b = self.base
        self.register_buffer("rest", b.template.clone())
        self.register_buffer("joints_rest", b.calibrated_rest_joints.clone() if hasattr(b, "calibrated_rest_joints") else b.regressor @ b.template)
        self.register_buffer("basis", b.components[:components].clone())
        ids = sorted(
            set(
                np.linspace(0, 777, 112).astype(int).tolist()
                + list(TIPS)
                + [
                    int(v)
                    for f in range(5)
                    for v in getattr(b, "pad_" + str(f)).cpu().tolist()
                ]
            )
        )
        self.register_buffer("sample_ids", torch.tensor(ids, device=device))
        self.register_buffer(
            "tip_sample_ids", torch.tensor([ids.index(t) for t in TIPS], device=device)
        )
        self.edges = np.array(
            [(0, 1 + 4 * f) for f in range(5)]
            + [(1 + 4 * f + j, 2 + 4 * f + j) for f in range(5) for j in range(3)],
            int,
        )
        self.components = components
        parents = self.base.parents
        depth = [0] * 16
        for j in range(1, 16):
            depth[j] = depth[parents[j]] + 1
        self.levels = [
            [j for j in range(1, 16) if depth[j] == d] for d in range(1, max(depth) + 1)
        ]
        for index, level in enumerate(self.levels):
            self.register_buffer('level_indices_' + str(index), torch.tensor(level, device=device))
        self.register_buffer("ik_tip_ids", torch.tensor(TIPS, device=self.device))
        order = [0]
        for finger, chain in enumerate(CHAINS):
            order.extend([*chain, 16 + finger])
        self.register_buffer(
            "ordered_joint_ids", torch.tensor(order, device=self.device)
        )
        self.register_buffer(
            "relative_joints",
            torch.stack(
                [
                    self.joints_rest[j]
                    if p < 0
                    else self.joints_rest[j] - self.joints_rest[p]
                    for j, p in enumerate(parents)
                ]
            ),
        )

    def configure_contact_vertices(self, mappings):
        ids = set(self.sample_ids.cpu().tolist())
        for records in mappings.values():
            for record in records:
                face = int(record["mano_face_id"])
                if not 0 <= face < len(self.base.faces):
                    raise ValueError("Invalid fixed MANO face")
                ids.update(self.base.faces[face].tolist())
        ids = sorted(ids)
        self.sample_ids = torch.tensor(ids, device=self.device)
        self.tip_sample_ids = torch.tensor(
            [ids.index(t) for t in TIPS], device=self.device
        )

    @property
    def device(self):
        return self.rest.device

    def angles(self, variables):
        return self.base.pose_mean + variables[..., 6:] @ self.basis

    def _decode(self, variables, ids, tip_positions):
        pose = self.angles(variables).reshape(-1, 15, 3)
        b = len(pose)
        local = rotmat(torch.cat((torch.zeros_like(pose[:, :1]), pose), 1))
        rotations = [None] * 16
        positions = [None] * 16
        rotations[0] = local[:, 0]
        positions[0] = self.joints_rest[0].expand(b, -1)
        for level_index, level in enumerate(self.levels):
            level_ids = getattr(self, 'level_indices_' + str(level_index))
            parent_r = torch.stack([rotations[self.base.parents[j]] for j in level], 1)
            parent_p = torch.stack([positions[self.base.parents[j]] for j in level], 1)
            rr = parent_r @ local[:, level_ids]
            pp = parent_p + (
                parent_r @ self.relative_joints[level_ids][None, :, :, None]
            ).squeeze(-1)
            for k, j in enumerate(level):
                rotations[j] = rr[:, k]
                positions[j] = pp[:, k]
        rotations = torch.stack(rotations, 1)
        positions = torch.stack(positions, 1)
        feature = (
            local[:, 1:] - torch.eye(3, dtype=variables.dtype, device=self.device)
        ).reshape(b, 135)
        posed = self.rest[ids][None] + torch.einsum(
            "bp,vcp->bvc", feature, self.base.posedirs[ids]
        )
        offset = positions - (rotations @ self.joints_rest[None, :, :, None]).squeeze(
            -1
        )
        skin_r = torch.einsum("vj,bjkl->bvkl", self.base.weights[ids], rotations)
        skin_t = torch.einsum("vj,bjk->bvk", self.base.weights[ids], offset)
        v = (skin_r @ posed[:, :, :, None]).squeeze(-1) + skin_t
        root = rotmat(variables[:, 3:6])
        translation = variables[:, :3]
        v = (root[:, None] @ v[:, :, :, None]).squeeze(-1) + translation[:, None]
        joints = (root[:, None] @ positions[:, :, :, None]).squeeze(-1) + translation[
            :, None
        ]
        ordered = torch.cat((joints, v[:, tip_positions]), 1)[:, self.ordered_joint_ids]
        return v, ordered, root[:, None] @ rotations

    def forward(self, variables, full=False):
        ids = torch.arange(778, device=self.device) if full else self.sample_ids
        tip_positions = self.ik_tip_ids if full else self.tip_sample_ids
        return self._decode(variables, ids, tip_positions)

    def joints_only(self, variables):
        return self._decode(
            variables, self.ik_tip_ids, torch.arange(5, device=self.device)
        )[1]


class ShadowSkeleton:
    """Device native-model FK, including the source free-root transform."""

    def __init__(self, xml, device="cuda:0"):
        import mujoco
        from .pose_gpu import ModelFK

        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.device = device
        self.fk = ModelFK(self.model, device)
        self.ids = [self.model.body("rh_palm").id]
        for prefix in ("th", "ff", "mf", "rf", "lf"):
            self.ids.extend(
                [
                    self.model.body("rh_" + prefix + suffix).id
                    for suffix in ("proximal", "middle", "distal")
                ]
            )

    def frames(self, poses):
        from retarget.initialization.dexonomy.util.torch_util import (
            torch_quaternion_to_matrix,
        )

        q = torch.as_tensor(poses, device=self.device, dtype=torch.float32)
        p, r = self.fk.frames(q[:, 7:])
        root = torch_quaternion_to_matrix(q[:, 3:7])
        return q, p, r, root

    def __call__(self, poses):
        q, p, r, root = self.frames(poses)
        local = [p[:, self.ids[0]]]
        for finger in range(5):
            ids = self.ids[1 + 3 * finger : 4 + 3 * finger]
            local.extend(p[:, body] for body in ids)
            local.append(
                p[:, ids[-1]]
                + (r[:, ids[-1]] @ q.new_tensor([0, 0, 0.027])[..., None]).squeeze(-1)
            )
        return torch.stack(local, 1) @ root.transpose(-1, -2) + q[:, None, :3]

    def contacts(self, poses, records):
        q, p, r, root = self.frames(poses)
        bodies = [self.model.body(record["shadow_body"]).id for record in records]
        local = q.new_tensor([record["shadow_local_point"] for record in records])
        normals = q.new_tensor([record["shadow_local_normal"] for record in records])
        hand = p[:, bodies] + (r[:, bodies] @ local[..., None]).squeeze(-1)
        normal = (r[:, bodies] @ normals[..., None]).squeeze(-1)
        return hand @ root.transpose(-1, -2) + q[:, None, :3], normal @ root.transpose(
            -1, -2
        )


class FixedContacts:
    """The reviewed, immutable triangle/barycentric slots; no nearest skin reselection."""

    def __init__(self, decoder, records):
        if not records:
            raise ValueError("Fixed anatomical contacts are required")
        self.records = records
        lookup = {int(v): i for i, v in enumerate(decoder.sample_ids.cpu().tolist())}
        self.faces = np.array([int(r["mano_face_id"]) for r in records], np.int32)
        triangles = decoder.base.faces[self.faces]
        try:
            self.sample_triangles = torch.tensor(
                [[lookup[int(v)] for v in f] for f in triangles], device=decoder.device
            )
        except KeyError as e:
            raise ValueError(
                "Configure fixed contact vertices before optimizing"
            ) from e
        self.full_triangles = torch.tensor(triangles, device=decoder.device)
        bary = np.array([r["mano_barycentric"] for r in records])
        if (
            not np.isfinite(bary).all()
            or np.min(bary) < -1e-5
            or not np.allclose(bary.sum(-1), 1.0, atol=1e-5)
        ):
            raise ValueError("Invalid fixed contact barycentrics")
        self.bary = torch.tensor(bary, dtype=torch.float32, device=decoder.device)

    def evaluate(self, vertices, full=False):
        tri = vertices[:, self.full_triangles if full else self.sample_triangles]
        points = (tri * self.bary[None, :, :, None]).sum(-2)
        normal = torch.nn.functional.normalize(
            torch.linalg.cross(
                tri[:, :, 1] - tri[:, :, 0], tri[:, :, 2] - tri[:, :, 0]
            ),
            dim=-1,
        )
        return points, normal


@wp.kernel(enable_backward=False)
def mesh_query(
    meshes: wp.array(dtype=wp.uint64),
    rows: wp.array(dtype=int),
    points: wp.array(dtype=wp.vec3),
    projected: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3),
    distance: wp.array(dtype=float),
    faces: wp.array(dtype=int),
    bary: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    mesh = meshes[rows[i]]
    p = points[i]
    hit = wp.mesh_query_point(mesh, p, 1.0e6)
    if hit.result:
        q = wp.mesh_eval_position(mesh, hit.face, hit.u, hit.v)
        a = wp.mesh_get_point(mesh, 3 * hit.face)
        b = wp.mesh_get_point(mesh, 3 * hit.face + 1)
        c = wp.mesh_get_point(mesh, 3 * hit.face + 2)
        projected[i] = q
        normals[i] = wp.normalize(wp.cross(b - a, c - a))
        distance[i] = hit.sign * wp.length(p - q)
        faces[i] = hit.face
        bary[i] = wp.vec3(hit.u, hit.v, 1.0 - hit.u - hit.v)
    else:
        projected[i] = wp.vec3(wp.nan)
        normals[i] = wp.vec3(wp.nan)
        distance[i] = wp.nan
        faces[i] = -1


class MeshCache:
    """Device-owned LRU; batches hold references so eviction cannot invalidate queries."""

    def __init__(self, device, capacity=128):
        self.device = str(device)
        self.capacity = capacity
        self.entries = collections.OrderedDict()
        self.builds = 0
        self.hits = 0

    @staticmethod
    def key(geo):
        if "sha256" in geo:
            return (geo["sha256"], tuple(geo.get("base_scale", (1.0, 1.0, 1.0))))
        return (
            hashlib.sha256(np.ascontiguousarray(geo["vertices"]).tobytes()).hexdigest(),
            hashlib.sha256(np.ascontiguousarray(geo["faces"]).tobytes()).hexdigest(),
        )

    def get(self, geo):
        key = self.key(geo)
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return self.entries[key]
        vertices = wp.array(geo["vertices"], dtype=wp.vec3, device=self.device)
        faces = wp.array(geo["faces"].reshape(-1), dtype=wp.int32, device=self.device)
        entry = (vertices, faces, wp.Mesh(points=vertices, indices=faces))
        self.entries[key] = entry
        self.builds += 1
        if len(self.entries) > self.capacity:
            self.entries.popitem(last=False)
        return entry


class SurfaceBatch:
    def __init__(
        self,
        geometries,
        mesh_indices,
        object_pose,
        scene_scale,
        device,
        mesh_cache=None,
    ):
        self.device = torch.device(device)
        self.geometries = geometries
        self.geometry_indices = list(map(int, mesh_indices))
        self.object_keys = [
            (int(i), float(scale)) for i, scale in zip(mesh_indices, scene_scale)
        ]
        cache = mesh_cache or MeshCache(device)
        if cache.device != str(device):
            raise ValueError("Mesh cache device mismatch")
        # Build only meshes referenced by this batch, and reuse their uploaded BVHs.
        unique = sorted(set(map(int, mesh_indices)))
        self.storage = [cache.get(geometries[i]) for i in unique]
        self.meshes = [x[2] for x in self.storage]
        lookup = {key: self.meshes[i].id for i, key in enumerate(unique)}
        self.mesh_ids = wp.array(
            [lookup[int(i)] for i in mesh_indices], dtype=wp.uint64, device=str(device)
        )
        from retarget.initialization.dexonomy.util.torch_util import (
            torch_quaternion_to_matrix,
        )

        pose = torch.as_tensor(object_pose, device=device, dtype=torch.float32)
        self.rotation = torch_quaternion_to_matrix(pose[:, 3:])
        self.origin = pose[:, :3]
        self.scale = torch.tensor(scene_scale, dtype=torch.float32, device=device)
        self.cache = cache
        self.watertight = torch.tensor(
            [geometries[int(i)].get("watertight", True) for i in mesh_indices],
            dtype=torch.bool,
            device=device,
        )
        # Upload/build on Warp's stream, then establish a dependency without a host sync.
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).wait_stream(
                wp.stream_to_torch(wp.get_stream(str(device)))
            )

    def query(self, world, rows):
        n, k, _ = world.shape
        r = self.rotation[rows]
        s = self.scale[rows]
        local = (
            torch.einsum(
                "bij,bkj->bki", r.transpose(-1, -2), world - self.origin[rows, None]
            )
            / s[:, None, None]
        )
        source = wp.from_torch(local.contiguous().reshape(-1, 3), dtype=wp.vec3)
        indexes = rows.to(torch.int32).repeat_interleave(k).contiguous()
        shape = n * k
        q = wp.empty(shape, dtype=wp.vec3, device=str(self.device))
        normal = wp.empty_like(q)
        dist = wp.empty(shape, dtype=float, device=str(self.device))
        face = wp.empty(shape, dtype=int, device=str(self.device))
        bary = wp.empty_like(q)
        wp.launch(
            mesh_query,
            shape,
            inputs=[self.mesh_ids, wp.from_torch(indexes, dtype=wp.int32), source],
            outputs=[q, normal, dist, face, bary],
            device=str(self.device),
            stream=wp.stream_from_torch(self.device)
            if self.device.type == "cuda"
            else None,
        )
        projected = (
            torch.einsum(
                "bij,bkj->bki", r, wp.to_torch(q).reshape(n, k, 3) * s[:, None, None]
            )
            + self.origin[rows, None]
        )
        normals = torch.einsum("bij,bkj->bki", r, wp.to_torch(normal).reshape(n, k, 3))
        return (
            projected,
            normals,
            wp.to_torch(dist).reshape(n, k) * s[:, None],
            wp.to_torch(face).reshape(n, k),
            wp.to_torch(bary).reshape(n, k, 3),
        )


def matrix_to_rotvec(matrix):
    """Stable SO(3) logarithm including rotations close to pi."""
    # Quaternion from the largest diagonal element avoids division by sin(theta).
    m = matrix
    one = torch.ones(len(m), device=m.device, dtype=m.dtype)
    candidates = (
        torch.stack(
            (
                1 + m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2],
                1 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2],
                1 - m[:, 0, 0] + m[:, 1, 1] - m[:, 2, 2],
                1 - m[:, 0, 0] - m[:, 1, 1] + m[:, 2, 2],
            ),
            -1,
        )
        .clamp_min(0)
        .sqrt()
        * 0.5
    )
    q0, q1, q2, q3 = candidates.unbind(-1)
    den = (4 * candidates).clamp_min(1.0e-8)
    q = torch.stack(
        (
            torch.stack(
                (
                    q0,
                    (m[:, 2, 1] - m[:, 1, 2]) / den[:, 0],
                    (m[:, 0, 2] - m[:, 2, 0]) / den[:, 0],
                    (m[:, 1, 0] - m[:, 0, 1]) / den[:, 0],
                ),
                -1,
            ),
            torch.stack(
                (
                    (m[:, 2, 1] - m[:, 1, 2]) / den[:, 1],
                    q1,
                    (m[:, 0, 1] + m[:, 1, 0]) / den[:, 1],
                    (m[:, 0, 2] + m[:, 2, 0]) / den[:, 1],
                ),
                -1,
            ),
            torch.stack(
                (
                    (m[:, 0, 2] - m[:, 2, 0]) / den[:, 2],
                    (m[:, 0, 1] + m[:, 1, 0]) / den[:, 2],
                    q2,
                    (m[:, 1, 2] + m[:, 2, 1]) / den[:, 2],
                ),
                -1,
            ),
            torch.stack(
                (
                    (m[:, 1, 0] - m[:, 0, 1]) / den[:, 3],
                    (m[:, 0, 2] + m[:, 2, 0]) / den[:, 3],
                    (m[:, 1, 2] + m[:, 2, 1]) / den[:, 3],
                    q3,
                ),
                -1,
            ),
        ),
        1,
    )
    ids = candidates.argmax(-1)
    q = q[torch.arange(len(q), device=m.device), ids]
    q = q * torch.where(q[:, :1] < 0, -one[:, None], one[:, None])
    norm = q[:, 1:].norm(dim=-1, keepdim=True)
    angle = 2 * torch.atan2(norm, q[:, :1])
    return q[:, 1:] * torch.where(
        norm > 1.0e-7, angle / norm.clamp_min(1.0e-12), 2 * one[:, None]
    )


def initialize(decoder, target):
    """Same rigid palm SVD alignment, batched on the decoder device."""
    target = torch.as_tensor(target, device=decoder.device, dtype=torch.float32)
    variables = target.new_zeros((len(target), 6 + decoder.components))
    if not hasattr(decoder, "_neutral_palm_gpu"):
        with torch.no_grad():
            _, neutral, _ = decoder(variables[:1])
        decoder._neutral_palm_gpu = neutral[0, [0, 5, 9, 13, 17]].detach()
    source = decoder._neutral_palm_gpu
    dest = target[:, [0, 5, 9, 13, 17]]
    a = source - source.mean(0)
    b = dest - dest.mean(1, keepdim=True)
    u, _, vh = torch.linalg.svd(a.T @ b)
    r = vh.transpose(-1, -2) @ u.transpose(-1, -2)
    correction = torch.eye(3, device=target.device).expand(len(target), -1, -1).clone()
    correction[:, 2, 2] = torch.where(torch.linalg.det(r) < 0, -1.0, 1.0)
    r = vh.transpose(-1, -2) @ correction @ u.transpose(-1, -2)
    variables[:, :3] = dest.mean(1) - (r @ source.mean(0)[:, None]).squeeze(-1)
    variables[:, 3:6] = matrix_to_rotvec(r)
    return variables


def skeleton_loss(decoder, variables, target):
    v, joints, _ = decoder(variables)
    edges = decoder.edges
    a = joints[:, edges[:, 1]] - joints[:, edges[:, 0]]
    b = target[:, edges[:, 1]] - target[:, edges[:, 0]]
    a = torch.nn.functional.normalize(a, dim=-1)
    b = torch.nn.functional.normalize(b, dim=-1)
    direction = (a - b).square().sum(-1).mean(-1)
    # Mild position term accommodates unequal human/robot segment lengths.
    positions = (joints - target).square().sum(-1).mean(-1) / (0.04**2)
    prior = variables[:, 6:].square().mean(-1)
    return 6 * direction + 0.35 * positions + 0.015 * prior, v, joints


def vertex_normals(vertices, faces):
    v = np.asarray(vertices)
    f = np.asarray(faces)
    normals = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    out = np.zeros_like(v)
    for j in range(3):
        np.add.at(out, f[:, j], normals)
    return out / np.maximum(np.linalg.norm(out, axis=-1, keepdims=True), 1e-12)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def residuals(decoder, variables, target):
    joints = decoder.joints_only(variables)
    edges = decoder.edges
    a = torch.nn.functional.normalize(
        joints[:, edges[:, 1]] - joints[:, edges[:, 0]], dim=-1
    )
    b = torch.nn.functional.normalize(
        target[:, edges[:, 1]] - target[:, edges[:, 0]], dim=-1
    )
    return torch.cat(
        (
            (a - b).flatten(1) * math.sqrt(6 / 20),
            (joints - target).flatten(1) * (math.sqrt(0.35 / 21) / 0.04),
            variables[:, 6:] * math.sqrt(0.015 / decoder.components),
        ),
        1,
    )


def kinematic_loss(decoder, variables, target):
    return residuals(decoder, variables, target).square().sum(-1)


def project_ik(values, initial):
    return torch.cat(
        (
            torch.maximum(
                initial[:, :3] - 0.06,
                torch.minimum(initial[:, :3] + 0.06, values[:, :3]),
            ),
            values[:, 3:6],
            values[:, 6:].clamp(-3.0, 3.0),
        ),
        1,
    )


def lm_ik(decoder, initial, target, settings):
    """Batched forward-mode AD Jacobians and damped 30x30 systems, all on device."""
    n, d = initial.shape
    device = initial.device
    scale = torch.tensor(
        [0.01] * 3 + [0.1] * 3 + [0.3] * decoder.components,
        device=device,
        dtype=initial.dtype,
    )
    values = initial.detach().clone()
    iterations = torch.zeros(n, dtype=torch.int32, device=device)
    damping = torch.full((n,), settings.lm_damping, device=device)
    active = torch.ones(n, dtype=torch.bool, device=device)
    stale = torch.zeros(n, dtype=torch.int32, device=device)
    eye = torch.eye(d, device=device)

    def single(u, t):
        return residuals(decoder, (u * scale)[None], t[None])[0]

    jacobian = torch.func.vmap(torch.func.jacfwd(single, argnums=0))
    history = []
    with torch.no_grad():
        cost = kinematic_loss(decoder, values, target)
        initial_cost = cost.clone()
    ids = torch.arange(n, device=device)
    for step in range(settings.ik_steps):
        current = values[ids]
        desired = target[ids]
        with torch.enable_grad():
            j = jacobian(current / scale, desired).detach()
        with torch.no_grad():
            r = residuals(decoder, current, desired)
            jt = j.transpose(1, 2)
            h = jt @ j
            g = (jt @ r[:, :, None]).squeeze(-1)
            diag = h.diagonal(dim1=-2, dim2=-1).clamp_min(1.0e-4)
            system = (
                h + damping[ids, None, None] * torch.diag_embed(diag) + eye * 1.0e-7
            )
            delta, info = torch.linalg.solve_ex(system, -g)
            finite = torch.isfinite(delta).all(-1) & (info == 0)
            delta = torch.where(finite[:, None], delta, torch.zeros_like(delta))
            delta = (
                delta
                * (
                    settings.lm_max_step
                    / delta.norm(dim=-1).clamp_min(settings.lm_max_step)
                )[:, None]
            )
            proposal = project_ik(current + delta * scale, initial[ids])
            new_cost = kinematic_loss(decoder, proposal, desired)
            old = cost[ids]
            accept = finite & torch.isfinite(new_cost) & (new_cost < old)
            values[ids] = torch.where(accept[:, None], proposal, current)
            cost[ids] = torch.minimum(
                old, torch.where(torch.isfinite(new_cost), new_cost, old)
            )
            damping[ids] = torch.where(
                accept, damping[ids] * 0.5, damping[ids] * 4
            ).clamp(1.0e-6, 1.0e6)
            iterations[ids] += 1
            history.append(cost.clone())
            small = (
                old - cost[ids]
            ) <= settings.stop_absolute_tol + settings.lm_relative_tol * old.abs()
            stale[ids] = torch.where(
                small, stale[ids] + 1, torch.zeros_like(stale[ids])
            )
            if settings.early_stop and step >= 2:
                done = (stale[ids] >= 3) | (
                    (g.abs().max(-1).values < settings.lm_gradient_tol) & finite
                )
                active[ids] &= ~done
                ids = torch.nonzero(active, as_tuple=True)[0]
                if not len(ids):
                    break
    return (
        values.detach(),
        iterations,
        dict(initial_cost=initial_cost, final_cost=cost, history=history),
    )


def pose_problem(mean, adapter):
    """Original score and bounds for CEM."""
    from retarget.solver.base import Problem

    device = mean.device
    objective = adapter.settings
    low = mean.clone()
    high = mean.clone()
    low[:, :3] -= objective.max_pos_delta
    high[:, :3] += objective.max_pos_delta
    low[:, 3:6] -= objective.max_rot_delta
    high[:, 3:6] += objective.max_rot_delta
    low[:, 6:] = -3.0
    high[:, 6:] = 3.0
    sigma = mean.new_tensor(
        [objective.pos_noise] * 3
        + [objective.rot_noise] * 3
        + [objective.joint_noise] * (mean.shape[1] - 6)
    ).expand_as(mean)

    def project(values):
        return torch.maximum(low[:, None], torch.minimum(high[:, None], values))

    rows = torch.arange(len(mean), device=device)

    def evaluate(values):
        b, p, d = values.shape
        return adapter.evaluate(
            values.reshape(-1, d), rows.repeat_interleave(p)
        ).reshape(b, p)

    problem = Problem(mean, sigma, evaluate, project)
    return problem
