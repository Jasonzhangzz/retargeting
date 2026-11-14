"""Moving signed mesh witnesses and analytic separation constraints for SLSQP.

Witnesses are refreshed after every constrained solve. Both sides move with
their native rigid links; the normal derivative is included. Connected MCP
collars use exactly the exclusions of the independent dense mesh auditor.
"""
from __future__ import annotations
import numpy as np
import warp as wp


@wp.kernel(enable_backward=False)
def _witnesses(meshes: wp.array(dtype=wp.uint64), points: wp.array(dtype=wp.vec3),
        owners: wp.array(dtype=int), lower: wp.array(dtype=wp.vec3), upper: wp.array(dtype=wp.vec3),
        collar: wp.array(dtype=float), allowed: wp.array2d(dtype=int),
        positions: wp.array(dtype=wp.vec3), rotations: wp.array(dtype=wp.mat33), margin: float,
        gap: wp.array2d(dtype=float), hit: wp.array2d(dtype=wp.vec3), normal: wp.array2d(dtype=wp.vec3)):
    sample, target = wp.tid()
    source = owners[sample]
    if allowed[source, target] == 0:
        return
    world = positions[source]+rotations[source]*points[sample]
    local = wp.transpose(rotations[target])*(world-positions[target])
    if source == 0 and collar[target] > 0.0 and wp.length(local) <= collar[target]:
        return
    if target == 0 and collar[source] > 0.0 and wp.length(points[sample]) <= collar[source]:
        return
    lo = lower[target]; hi = upper[target]
    if (local[0] < lo[0]-margin or local[1] < lo[1]-margin or local[2] < lo[2]-margin
            or local[0] > hi[0]+margin or local[1] > hi[1]+margin or local[2] > hi[2]+margin):
        return
    query = wp.mesh_query_point_sign_winding_number(meshes[target], local, 1.0)
    if query.result:
        closest = wp.mesh_eval_position(meshes[target], query.face, query.u, query.v)
        difference = local-closest; distance = wp.length(difference)
        gap[sample, target] = query.sign*distance
        hit[sample, target] = closest
        if distance > 1.e-8:
            normal[sample, target] = query.sign*difference/distance
        else:
            normal[sample, target] = wp.mesh_eval_face_normal(meshes[target], query.face)


class SeparationConstraints:
    def __init__(self, fk, source_owners, source_local, target_owners, target_local, target_normals,
                 fixed_positions=None, fixed_rotations=None):
        self.fk = fk
        self.a = np.asarray(source_owners, int); self.la = np.asarray(source_local, float)
        self.b = np.asarray(target_owners, int); self.lb = np.asarray(target_local, float)
        self.nb = np.asarray(target_normals, float)
        self.fixed_positions = fixed_positions; self.fixed_rotations = fixed_rotations
        self.owners = np.unique(np.r_[self.a, self.b[self.b >= 0]])

    def evaluate(self, q):
        self.fk.forward(q)
        data = self.fk.data; ids = self.fk.body_ids
        p = data.xpos[ids]; r = data.xmat.reshape(-1, 3, 3)[ids]
        axis_jac = {}; origin_jac = {}
        import mujoco
        for owner in self.owners:
            linear = np.zeros((3, self.fk.model.nv)); angular = np.zeros_like(linear)
            mujoco.mj_jac(self.fk.model, data, linear, angular, p[owner], int(ids[owner]))
            origin_jac[int(owner)] = linear[:, 6:]; axis_jac[int(owner)] = angular[:, 6:]
        va = np.einsum('nij,nj->ni', r[self.a], self.la)
        pa = p[self.a]+va
        ja = np.stack([origin_jac[int(o)]+np.cross(axis_jac[int(o)].T, v).T for o, v in zip(self.a, va)])
        # A negative owner denotes a known object fixed in the robot-wrist frame.
        b = np.maximum(self.b, 0)
        vb = np.einsum('nij,nj->ni', r[b], self.lb); pb = p[b]+vb
        normals = np.einsum('nij,nj->ni', r[b], self.nb)
        jb = np.zeros_like(ja); jn = np.zeros_like(ja)
        for i, owner in enumerate(self.b):
            if owner >= 0:
                jb[i] = origin_jac[int(owner)]+np.cross(axis_jac[int(owner)].T, vb[i]).T
                jn[i] = np.cross(axis_jac[int(owner)].T, normals[i]).T
            else:
                pb[i] = self.fixed_positions[i]; normals[i] = self.fixed_rotations[i]
        delta = pa-pb
        gap = np.einsum('ni,ni->n', delta, normals)
        jac = np.einsum('ni,nik->nk', normals, ja-jb)+np.einsum('ni,nik->nk', delta, jn)
        return gap, jac


@wp.kernel(enable_backward=False)
def _pair_witness_keys(meshes: wp.array(dtype=wp.uint64), points: wp.array(dtype=wp.vec3),
        owners: wp.array(dtype=int), lower: wp.array(dtype=wp.vec3), upper: wp.array(dtype=wp.vec3),
        collar: wp.array(dtype=float), allowed: wp.array2d(dtype=int),
        positions: wp.array(dtype=wp.vec3), rotations: wp.array(dtype=wp.mat33), margin: float,
        keys: wp.array2d(dtype=wp.int64)):
    sample,target=wp.tid();source=owners[sample]
    if allowed[source,target]==0:return
    world=positions[source]+rotations[source]*points[sample]
    local=wp.transpose(rotations[target])*(world-positions[target])
    if source==0 and collar[target]>0.0 and wp.length(local)<=collar[target]:return
    if target==0 and collar[source]>0.0 and wp.length(points[sample])<=collar[source]:return
    lo=lower[target];hi=upper[target]
    if (local[0]<lo[0]-margin or local[1]<lo[1]-margin or local[2]<lo[2]-margin
            or local[0]>hi[0]+margin or local[1]>hi[1]+margin or local[2]>hi[2]+margin):return
    query=wp.mesh_query_point_sign_winding_number(meshes[target],local,1.0)
    if query.result:
        hit=wp.mesh_eval_position(meshes[target],query.face,query.u,query.v)
        gap=query.sign*wp.length(local-hit)
        if gap<=margin:
            # Metre gaps quantized to 10 nm only for deterministic argmin;
            # the winner's full Float32 geometry is recomputed below. Pack
            # distance and sample ID into one atomic key to avoid torn hits.
            distance=wp.int64(wp.clamp(gap,-1.0,1.0)*100000000.0)+wp.int64(100000000)
            key=(distance<<wp.int64(32))|wp.int64(sample)
            wp.atomic_min(keys,source,target,key)


@wp.kernel(enable_backward=False)
def _pair_witness_geometry(meshes: wp.array(dtype=wp.uint64), points: wp.array(dtype=wp.vec3),
        positions: wp.array(dtype=wp.vec3), rotations: wp.array(dtype=wp.mat33),
        keys: wp.array2d(dtype=wp.int64), sample_ids: wp.array2d(dtype=int),
        hit: wp.array2d(dtype=wp.vec3), normal: wp.array2d(dtype=wp.vec3)):
    source,target=wp.tid();key=keys[source,target]
    if key==wp.int64(9223372036854775807):return
    sample=wp.int32(key&wp.int64(4294967295))
    world=positions[source]+rotations[source]*points[sample]
    local=wp.transpose(rotations[target])*(world-positions[target])
    query=wp.mesh_query_point_sign_winding_number(meshes[target],local,1.0)
    if query.result:
        closest=wp.mesh_eval_position(meshes[target],query.face,query.u,query.v)
        difference=local-closest;distance=wp.length(difference)
        sample_ids[source,target]=sample;hit[source,target]=closest
        if distance>1.e-8:normal[source,target]=query.sign*difference/distance
        else:normal[source,target]=wp.mesh_eval_face_normal(meshes[target],query.face)


class NativeSeparation:
    def __init__(self, fk, auditor, object_sdf=None):
        self.fk = fk; self.auditor = auditor; self.device = auditor.device; self.object_sdf = object_sdf
        self.points, self.owners = auditor.samples[True]
        self.local = self.points.numpy(); self.owner_array = self.owners.numpy()
        self.meshes = list(auditor.meshes)
        allowed = auditor.allowed.numpy(); lower = auditor.lower.numpy(); upper = auditor.upper.numpy()
        collar = auditor.collar.numpy()
        if object_sdf is not None:
            vertices = np.asarray(object_sdf.vertices, np.float32); faces = np.asarray(object_sdf.faces, np.int32)
            self.meshes.append(wp.Mesh(points=wp.array(vertices, dtype=wp.vec3, device=self.device),
                indices=wp.array(faces.ravel(), dtype=int, device=self.device), support_winding_number=True))
            extended = np.zeros((22, 22), np.int32); extended[:21, :21] = allowed; extended[:21, 21] = 1
            allowed = extended; lower = np.vstack((lower, vertices.min(0))); upper = np.vstack((upper, vertices.max(0)))
            collar = np.r_[collar, 0.]
        self.mesh_ids = wp.array([m.id for m in self.meshes], dtype=wp.uint64, device=self.device)
        self.allowed = wp.array(allowed, dtype=int, device=self.device)
        self.lower = wp.array(lower, dtype=wp.vec3, device=self.device); self.upper = wp.array(upper, dtype=wp.vec3, device=self.device)
        self.collar = wp.array(collar, dtype=float, device=self.device)

    def refresh(self, q, rotation, origin, *, per_pair=4, margin_m=.002):
        self.fk.forward(q)
        p = self.fk.data.xpos[self.fk.body_ids].copy()
        r = self.fk.data.xmat.reshape(-1, 3, 3)[self.fk.body_ids].copy()
        if self.object_sdf is not None:
            tf = self.object_sdf.transform
            p = np.vstack((p, (tf[:3, 3]-origin) @ rotation))
            r = np.concatenate((r, (rotation.T @ tf[:3, :3])[None]))
        n = len(p); shape = (len(self.points), n)
        if per_pair==1:
            # Reduce on the GPU: read back only one witness per directed link
            # pair, rather than dense sample×link hit/normal arrays.
            keys=wp.full((n,n),wp.int64(9223372036854775807),dtype=wp.int64,device=self.device)
            samples=wp.full((n,n),-1,dtype=int,device=self.device)
            hit=wp.zeros((n,n),dtype=wp.vec3,device=self.device);normal=wp.zeros_like(hit)
            pp=wp.array(p.astype(np.float32),dtype=wp.vec3,device=self.device)
            rr=wp.array(r.astype(np.float32),dtype=wp.mat33,device=self.device)
            wp.launch(_pair_witness_keys,shape,inputs=[self.mesh_ids,self.points,self.owners,self.lower,self.upper,
                self.collar,self.allowed,pp,rr,margin_m,keys],device=self.device)
            wp.launch(_pair_witness_geometry,(n,n),inputs=[self.mesh_ids,self.points,pp,rr,keys,samples,hit,normal],device=self.device)
            ids=samples.numpy();hits=hit.numpy();ns=normal.numpy();aa,targets=np.nonzero(ids>=0)
            if not len(aa):return None
            sample_ids=ids[aa,targets];local_hit=hits[aa,targets];local_normal=ns[aa,targets]
            fixed_positions=p[targets]+np.einsum('nij,nj->ni',r[targets],local_hit)
            fixed_normals=np.einsum('nij,nj->ni',r[targets],local_normal)
            return SeparationConstraints(self.fk,self.owner_array[sample_ids],self.local[sample_ids],
                np.where(targets<21,targets,-1),local_hit,local_normal,fixed_positions,fixed_normals)
        gap = wp.full(shape, 1000., dtype=float, device=self.device)
        hit = wp.zeros(shape, dtype=wp.vec3, device=self.device); normals = wp.zeros_like(hit)
        wp.launch(_witnesses, shape, inputs=[self.mesh_ids, self.points, self.owners, self.lower, self.upper,
            self.collar, self.allowed, wp.array(p.astype(np.float32), dtype=wp.vec3, device=self.device),
            wp.array(r.astype(np.float32), dtype=wp.mat33, device=self.device), margin_m, gap, hit, normals], device=self.device)
        gaps = gap.numpy(); hits = hit.numpy(); ns = normals.numpy()
        selected = []
        for a in range(21):
            samples = np.flatnonzero(self.owner_array == a)
            for b in range(n):
                candidates = samples[gaps[samples, b] <= margin_m]
                if len(candidates):
                    order = np.argsort(gaps[candidates, b])[:per_pair]
                    selected.extend((int(i), b) for i in candidates[order])
        if not selected: return None
        sample_ids = np.array([i for i, _ in selected]); targets = np.array([j for _, j in selected])
        local_hit = hits[sample_ids, targets]; local_normal = ns[sample_ids, targets]
        fixed_positions = p[targets]+np.einsum('nij,nj->ni', r[targets], local_hit)
        fixed_normals = np.einsum('nij,nj->ni', r[targets], local_normal)
        return SeparationConstraints(self.fk, self.owner_array[sample_ids], self.local[sample_ids],
            np.where(targets < 21, targets, -1), local_hit, local_normal, fixed_positions, fixed_normals)
