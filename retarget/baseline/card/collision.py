"""Native WUJI2 surface penetration, independent of tactile associations."""
import numpy as np
import torch
import warp as wp


@wp.kernel(enable_backward=False)
def _native_penetration(meshes:wp.array(dtype=wp.uint64),
        points:wp.array(dtype=wp.vec3),owners:wp.array(dtype=int),
        lower:wp.array(dtype=wp.vec3),upper:wp.array(dtype=wp.vec3),collar:wp.array(dtype=float),allowed:wp.array2d(dtype=int),
        positions:wp.array2d(dtype=wp.vec3),rotations:wp.array2d(dtype=wp.mat33),
        depth:wp.array3d(dtype=float),invalid:wp.array(dtype=int)):
    batch,sample,target=wp.tid()
    source=owners[sample]
    if allowed[source,target]==0:
        return
    world=positions[batch,source]+rotations[batch,source]*points[sample]
    local=wp.transpose(rotations[batch,target])*(world-positions[batch,target])
    # A two-axis MCP has its flexion and abduction housings at one pivot.
    # Exclude only that measured mechanical collar, not the whole proximal.
    if source==0 and collar[target]>0.0 and wp.length(local)<=collar[target]:
        return
    if target==0 and collar[source]>0.0 and wp.length(points[sample])<=collar[source]:
        return
    lo=lower[target];hi=upper[target]
    if local[0]<lo[0] or local[1]<lo[1] or local[2]<lo[2] or local[0]>hi[0] or local[1]>hi[1] or local[2]>hi[2]:
        return
    q=wp.mesh_query_point_sign_winding_number(meshes[target],local,1.0)
    if not q.result:
        wp.atomic_add(invalid,batch,1)
    elif q.sign<0.0:
        hit=wp.mesh_eval_position(meshes[target],q.face,q.u,q.v)
        wp.atomic_max(depth,batch,source,target,wp.length(local-hit)*1000.0)


class WujiMeshCollision:
    """Both directions, all surfaces; dense audit uses vertices + face centres.

    Native mechanical connections (palm/base and neighbours in one finger)
    are excluded. All cross-finger pairs and nonconnected palm surfaces remain,
    including pairs associated by tactile contact. This is a sampled surface
    check, not a continuous triangle-intersection proof.
    """
    def __init__(self,surface,device):
        self.device=str(device);self.body_ids=surface.body_ids
        n=len(surface.meshes);allowed=np.zeros((n,n),np.int32)
        for a in range(n):
            for b in range(a+1,n):
                cross=a>0 and (a-1)//4!=(b-1)//4
                palm=a==0 and (b-1)%4>=1
                distant=a>0 and (a-1)//4==(b-1)//4 and b-a>1
                if cross or palm or distant:allowed[a,b]=allowed[b,a]=1
        # Resolve mechanical adjacency from the native body tree, including
        # intervening fixed bodies, rather than trusting link-number spacing.
        owners={body:i for i,body in enumerate(surface.body_ids)}
        self.parent_owners=np.full(n,-1,np.int32)
        for owner,body in enumerate(surface.body_ids):
            parent=int(surface.model.body_parentid[body])
            while parent and parent not in owners:parent=int(surface.model.body_parentid[parent])
            if parent in owners:
                neighbour=owners[parent];self.parent_owners[owner]=neighbour
                allowed[owner,neighbour]=allowed[neighbour,owner]=0
        self.allowed_mask=allowed.astype(bool)
        self.allowed=wp.array(allowed,dtype=int,device=self.device)
        collar=np.zeros(n,np.float32)
        for owner in (2,6,10,14,18):
            parent=surface.body_ids[owner-1];child=surface.body_ids[owner]
            data=surface.neutral
            world=np.asarray(surface.meshes[owner-1].vertices)@data.xmat[parent].reshape(3,3).T+data.xpos[parent]
            local=(world-data.xpos[child])@data.xmat[child].reshape(3,3)
            collar[owner]=np.linalg.norm(local,axis=1).max()+.0001
        self.collar_radii_m=collar;self.collar=wp.array(collar,dtype=float,device=self.device)
        self.meshes=[];coarse=[];dense=[];coarse_owners=[];dense_owners=[];low=[];high=[]
        rng=np.random.default_rng(8127)
        for owner,mesh in enumerate(surface.meshes):
            v=np.asarray(mesh.vertices,np.float32);f=np.asarray(mesh.faces,np.int32)
            self.meshes.append(wp.Mesh(points=wp.array(v,dtype=wp.vec3,device=self.device),indices=wp.array(f.ravel(),dtype=int,device=self.device),support_winding_number=True))
            low.append(v.min(0));high.append(v.max(0))
            tri=v[f];ids=rng.choice(len(f),48,p=mesh.area_faces/mesh.area)
            uv=rng.random((len(ids),2));flip=uv.sum(-1)>1;uv[flip]=1-uv[flip]
            sampled=(tri[ids]*np.c_[uv,1-uv.sum(-1)][...,None]).sum(1)
            extremes=v[np.r_[v.argmin(0),v.argmax(0)]]
            c=np.vstack((sampled,extremes));d=np.vstack((v,tri.mean(1),c))
            coarse.extend(c);dense.extend(d);coarse_owners.extend([owner]*len(c));dense_owners.extend([owner]*len(d))
        self.mesh_ids=wp.array([m.id for m in self.meshes],dtype=wp.uint64,device=self.device)
        self.lower=wp.array(np.asarray(low),dtype=wp.vec3,device=self.device);self.upper=wp.array(np.asarray(high),dtype=wp.vec3,device=self.device)
        self.samples={False:(wp.array(np.asarray(coarse),dtype=wp.vec3,device=self.device),wp.array(coarse_owners,dtype=int,device=self.device)),
                      True:(wp.array(np.asarray(dense),dtype=wp.vec3,device=self.device),wp.array(dense_owners,dtype=int,device=self.device))}
        self.names=['palm']+[f'{f}_link{k}' for f in ('thumb','index','middle','ring','pinky') for k in range(1,5)]

    def query(self,positions,rotations,dense=False):
        p=positions[:,self.body_ids].contiguous();r=rotations[:,self.body_ids].contiguous()
        if not torch.isfinite(p).all() or not torch.isfinite(r).all():raise ValueError('Nonfinite WUJI2 collision pose')
        points,owners=self.samples[dense];b=len(p)
        depth=wp.zeros((b,21,21),dtype=float,device=self.device);invalid=wp.zeros(b,dtype=int,device=self.device)
        wp.launch(_native_penetration,(b,len(points),21),inputs=[self.mesh_ids,points,owners,self.lower,self.upper,self.collar,self.allowed,
            wp.from_torch(p,dtype=wp.vec3),wp.from_torch(r,dtype=wp.mat33),depth,invalid],device=self.device)
        result=wp.to_torch(depth);valid=wp.to_torch(invalid)==0
        return result,valid

    def report(self,depth,valid):
        raw=depth.detach().cpu().numpy();pairs={}
        valid=bool(valid) and bool(np.isfinite(raw).all())
        # Never export a mechanical connection as a collision, even if an
        # upstream/legacy buffer contains a nonzero depth for an excluded pair.
        values=np.where(self.allowed_mask & np.isfinite(raw),raw,0.)
        for a in range(21):
            for b in range(a+1,21):
                value=float(max(values[a,b],values[b,a]))
                if value>1e-5:pairs[self.names[a]+'__'+self.names[b]]=value
        return dict(max_penetration_mm=float(values.max()),pair_depths_mm=pairs if valid else {},query_valid=valid,
            connection_exclusions_applied=True,display_threshold_mm=.5,
            method='native_mesh_winding_vertices_and_face_centres',continuous_intersection_proof=False,
            checked_scope='all_cross_finger_links_nonadjacent_same_finger_and_palm_except_connected_MCP_collars')
