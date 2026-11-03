"""Production Shadow-to-MANO solver: GPU LM IK and adaptive diagonal CEM.

One algorithm preset; immutable anatomical contact mapping, object/size-gated
warm starts, bounded mesh caches, per-row early exits and deferred CPU export.
Distances are metres. Geometry acceptance is not a force-closure certificate.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import collections,hashlib,math
import numpy as np
import torch
import warp as wp
from scipy.spatial.transform import Rotation
from scripts.calibrate_wuji_mano import PersonalizedMANO,rotmat,CHAINS,TIPS



@dataclass(frozen=True)
class Settings:
    """The production LM-15 / adaptive-CEM-128x24 preset."""
    batch_size:int=64
    population:int=128
    iterations:int=24
    ik_steps:int=15
    pca_components:int=24
    candidate_chunk:int=32768
    seed:int=7
    contact_tolerance_m:float=.002
    penetration_tolerance_m:float=.002
    pos_noise_m:float=.006
    rot_noise_rad:float=.08
    pca_noise:float=.18
    max_root_delta_m:float=.06
    max_root_delta_rad:float=.8
    memory_budget_gb:float=28.
    early_stop:bool=True
    search_min_iterations:int=16
    search_check_every:int=4
    search_patience:int=4
    stop_relative_tol:float=1.e-4
    stop_absolute_tol:float=1.e-5
    warm_start:bool=True
    warm_capacity:int=64
    warm_object_capacity:int=128
    warm_size_ratio:float=1.2
    lm_damping:float=.01
    lm_max_step:float=5.
    lm_relative_tol:float=1.e-4
    lm_gradient_tol:float=1.e-5
    cem_elite_fraction:float=.125
    cem_momentum:float=.2
    cem_sigma_floor:float=.02
    cem_reuse_fraction:float=.25
    cem_mean_step:float=2.
    cem_mean_step_limit:float=3.
    def __post_init__(self):
        for key,value in vars(self).items():
            if isinstance(value,(int,float)) and not math.isfinite(value):raise ValueError(key+' must be finite')
        for key in ('batch_size','population','iterations','pca_components','candidate_chunk',
                    'search_check_every','search_patience','warm_capacity','warm_object_capacity'):
            if getattr(self,key)<1:raise ValueError(key+' must be positive')
        if self.population<2 or not 1<=self.pca_components<=45 or self.ik_steps<0:raise ValueError('Invalid solver dimensions')
        if self.search_min_iterations<0 or self.stop_relative_tol<0 or self.stop_absolute_tol<0:raise ValueError('Invalid stopping settings')
        if self.warm_size_ratio<1:raise ValueError('Invalid warm-start size ratio')
        for key in ('contact_tolerance_m','penetration_tolerance_m','memory_budget_gb',
                    'pos_noise_m','rot_noise_rad','pca_noise','max_root_delta_m','max_root_delta_rad',
                    'lm_damping','lm_max_step','cem_sigma_floor','cem_mean_step','cem_mean_step_limit'):
            if getattr(self,key)<=0:raise ValueError(key+' must be positive')
        if self.lm_relative_tol<0 or self.lm_gradient_tol<0:raise ValueError('Invalid LM tolerances')
        if not 0<self.cem_elite_fraction<=1 or not 0<=self.cem_momentum<1 or not 0<=self.cem_reuse_fraction<1:raise ValueError('Invalid CEM fit')


class ManoDecoder(torch.nn.Module):
    """Exact MANO with parallel finger chains and a five-tip subset for LM IK."""
    def __init__(self,path,device='cpu',components=24):
        super().__init__()
        self.base=PersonalizedMANO(path).to(device=device,dtype=torch.float32)
        b=self.base
        self.register_buffer('rest',b.template.clone())
        self.register_buffer('joints_rest',b.regressor@b.template)
        self.register_buffer('basis',b.components[:components].clone())
        ids=sorted(set(np.linspace(0,777,112).astype(int).tolist()+list(TIPS)+
                       [int(v) for f in range(5) for v in getattr(b,'pad_'+str(f)).cpu().tolist()]))
        self.register_buffer('sample_ids',torch.tensor(ids,device=device))
        self.register_buffer('tip_sample_ids',torch.tensor([ids.index(t) for t in TIPS],device=device))
        self.edges=np.array([(0,1+4*f) for f in range(5)]+[(1+4*f+j,2+4*f+j) for f in range(5) for j in range(3)],int)
        self.components=components
        parents=self.base.parents;depth=[0]*16
        for j in range(1,16):depth[j]=depth[parents[j]]+1
        self.levels=[[j for j in range(1,16) if depth[j]==d] for d in range(1,max(depth)+1)]
        self.register_buffer('ik_tip_ids',torch.tensor(TIPS,device=self.device))
        order=[0]
        for finger,chain in enumerate(CHAINS):order.extend([*chain,16+finger])
        self.register_buffer('ordered_joint_ids',torch.tensor(order,device=self.device))
        self.register_buffer('relative_joints',torch.stack([self.joints_rest[j] if p<0 else self.joints_rest[j]-self.joints_rest[p] for j,p in enumerate(parents)]))
    def configure_contact_vertices(self,mappings):
        ids=set(self.sample_ids.cpu().tolist())
        for records in mappings.values():
            for record in records:
                face=int(record['mano_face_id'])
                if not 0<=face<len(self.base.faces):raise ValueError('Invalid fixed MANO face')
                ids.update(self.base.faces[face].tolist())
        ids=sorted(ids)
        self.sample_ids=torch.tensor(ids,device=self.device)
        self.tip_sample_ids=torch.tensor([ids.index(t) for t in TIPS],device=self.device)
    @property
    def device(self):return self.rest.device
    def angles(self,variables):return self.base.pose_mean+variables[...,6:]@self.basis
    def _decode(self,variables,ids,tip_positions):
        pose=self.angles(variables).reshape(-1,15,3);b=len(pose)
        local=rotmat(torch.cat((torch.zeros_like(pose[:,:1]),pose),1))
        rotations=[None]*16;positions=[None]*16
        rotations[0]=local[:,0];positions[0]=self.joints_rest[0].expand(b,-1)
        for level in self.levels:
            parent_r=torch.stack([rotations[self.base.parents[j]] for j in level],1)
            parent_p=torch.stack([positions[self.base.parents[j]] for j in level],1)
            rr=parent_r@local[:,level]
            pp=parent_p+(parent_r@self.relative_joints[level][None,:,:,None]).squeeze(-1)
            for k,j in enumerate(level):rotations[j]=rr[:,k];positions[j]=pp[:,k]
        rotations=torch.stack(rotations,1);positions=torch.stack(positions,1)
        feature=(local[:,1:]-torch.eye(3,dtype=variables.dtype,device=self.device)).reshape(b,135)
        posed=self.rest[ids][None]+torch.einsum('bp,vcp->bvc',feature,self.base.posedirs[ids])
        offset=positions-(rotations@self.joints_rest[None,:,:,None]).squeeze(-1)
        skin_r=torch.einsum('vj,bjkl->bvkl',self.base.weights[ids],rotations)
        skin_t=torch.einsum('vj,bjk->bvk',self.base.weights[ids],offset)
        v=(skin_r@posed[:,:,:,None]).squeeze(-1)+skin_t
        root=rotmat(variables[:,3:6]);translation=variables[:,:3]
        v=(root[:,None]@v[:,:,:,None]).squeeze(-1)+translation[:,None]
        joints=(root[:,None]@positions[:,:,:,None]).squeeze(-1)+translation[:,None]
        ordered=torch.cat((joints,v[:,tip_positions]),1)[:,self.ordered_joint_ids]
        return v,ordered,root[:,None]@rotations
    def forward(self,variables,full=False):
        ids=torch.arange(778,device=self.device) if full else self.sample_ids
        tip_positions=self.ik_tip_ids if full else self.tip_sample_ids
        return self._decode(variables,ids,tip_positions)
    def joints_only(self,variables):
        return self._decode(variables,self.ik_tip_ids,torch.arange(5,device=self.device))[1]


class ShadowSkeleton:
    """Native Shadow FK with source free-root transform; no collision/integration."""
    def __init__(self,xml):
        import mujoco
        self.mj=mujoco;self.model=mujoco.MjModel.from_xml_path(str(xml));self.data=mujoco.MjData(self.model)
        if self.model.nq!=22:raise ValueError('Expected source Shadow 22 joint model')
        self.ids=[self.model.body('rh_palm').id]
        for prefix in ('th','ff','mf','rf','lf'):
            self.ids.extend([self.model.body('rh_'+prefix+s).id for s in ('proximal','middle','distal')])
    def __call__(self,poses):
        output=[]
        for q in np.asarray(poses):
            self.data.qpos[:]=q[7:];self.mj.mj_kinematics(self.model,self.data)
            local=[self.data.xpos[self.ids[0]].copy()]
            for f in range(5):
                ids=self.ids[1+3*f:4+3*f]
                local.extend([self.data.xpos[i].copy() for i in ids])
                local.append(self.data.xpos[ids[-1]]+self.data.xmat[ids[-1]].reshape(3,3)@np.array([0,0,.027]))
            r=Rotation.from_quat(q[3:7],scalar_first=True).as_matrix()
            output.append(np.asarray(local)@r.T+q[:3])
        return np.asarray(output)

    def contacts(self,poses,records):
        points=[];normals=[]
        for q in np.asarray(poses):
            self.data.qpos[:]=q[7:];self.mj.mj_kinematics(self.model,self.data)
            root=Rotation.from_quat(q[3:7],scalar_first=True).as_matrix()
            pp=[];nn=[]
            for record in records:
                body=self.model.body(record['shadow_body']).id
                frame=self.data.xmat[body].reshape(3,3)
                pp.append((self.data.xpos[body]+frame@record['shadow_local_point'])@root.T+q[:3])
                nn.append((frame@record['shadow_local_normal'])@root.T)
            points.append(pp);normals.append(nn)
        return np.asarray(points),np.asarray(normals)


class FixedContacts:
    """The reviewed, immutable triangle/barycentric slots; no nearest skin reselection."""
    def __init__(self,decoder,records):
        if not records:raise ValueError('Fixed anatomical contacts are required')
        self.records=records
        lookup={int(v):i for i,v in enumerate(decoder.sample_ids.cpu().tolist())}
        self.faces=np.array([int(r['mano_face_id']) for r in records],np.int32)
        triangles=decoder.base.faces[self.faces]
        try:self.sample_triangles=torch.tensor([[lookup[int(v)] for v in f] for f in triangles],device=decoder.device)
        except KeyError as e:raise ValueError('Configure fixed contact vertices before optimizing') from e
        self.full_triangles=torch.tensor(triangles,device=decoder.device)
        bary=np.array([r['mano_barycentric'] for r in records])
        if not np.isfinite(bary).all() or np.min(bary)<-1e-5 or not np.allclose(bary.sum(-1),1.,atol=1e-5):raise ValueError('Invalid fixed contact barycentrics')
        self.bary=torch.tensor(bary,dtype=torch.float32,device=decoder.device)
    def evaluate(self,vertices,full=False):
        tri=vertices[:,self.full_triangles if full else self.sample_triangles]
        points=(tri*self.bary[None,:,:,None]).sum(-2)
        normal=torch.nn.functional.normalize(torch.linalg.cross(tri[:,:,1]-tri[:,:,0],tri[:,:,2]-tri[:,:,0]),dim=-1)
        return points,normal


@wp.kernel(enable_backward=False)
def mesh_query(meshes:wp.array(dtype=wp.uint64), rows:wp.array(dtype=int),
               points:wp.array(dtype=wp.vec3),projected:wp.array(dtype=wp.vec3),
               normals:wp.array(dtype=wp.vec3),distance:wp.array(dtype=float),
               faces:wp.array(dtype=int),bary:wp.array(dtype=wp.vec3)):
    i=wp.tid();mesh=meshes[rows[i]];p=points[i]
    hit=wp.mesh_query_point(mesh,p,1.e6)
    if hit.result:
        q=wp.mesh_eval_position(mesh,hit.face,hit.u,hit.v)
        a=wp.mesh_get_point(mesh,3*hit.face);b=wp.mesh_get_point(mesh,3*hit.face+1);c=wp.mesh_get_point(mesh,3*hit.face+2)
        projected[i]=q;normals[i]=wp.normalize(wp.cross(b-a,c-a))
        distance[i]=hit.sign*wp.length(p-q);faces[i]=hit.face;bary[i]=wp.vec3(hit.u,hit.v,1.-hit.u-hit.v)
    else:
        projected[i]=wp.vec3(wp.nan);normals[i]=wp.vec3(wp.nan);distance[i]=wp.nan;faces[i]=-1


class MeshCache:
    """Device-owned LRU; batches hold references so eviction cannot invalidate queries."""
    def __init__(self,device,capacity=128):
        self.device=str(device);self.capacity=capacity;self.entries=collections.OrderedDict()
        self.builds=0;self.hits=0
    @staticmethod
    def key(geo):
        if 'sha256' in geo:return (geo['sha256'],tuple(geo.get('base_scale',(1.,1.,1.))))
        return (hashlib.sha256(np.ascontiguousarray(geo['vertices']).tobytes()).hexdigest(),
                hashlib.sha256(np.ascontiguousarray(geo['faces']).tobytes()).hexdigest())
    def get(self,geo):
        key=self.key(geo)
        if key in self.entries:
            self.hits+=1;self.entries.move_to_end(key);return self.entries[key]
        vertices=wp.array(geo['vertices'],dtype=wp.vec3,device=self.device)
        faces=wp.array(geo['faces'].reshape(-1),dtype=wp.int32,device=self.device)
        entry=(vertices,faces,wp.Mesh(points=vertices,indices=faces))
        self.entries[key]=entry;self.builds+=1
        if len(self.entries)>self.capacity:self.entries.popitem(last=False)
        return entry


class SurfaceBatch:
    def __init__(self,geometries,mesh_indices,object_pose,scene_scale,device,mesh_cache=None):
        self.device=torch.device(device)
        cache=mesh_cache or MeshCache(device)
        if cache.device!=str(device):raise ValueError('Mesh cache device mismatch')
        # Build only meshes referenced by this batch, and reuse their uploaded BVHs.
        unique=sorted(set(map(int,mesh_indices)))
        self.storage=[cache.get(geometries[i]) for i in unique]
        self.meshes=[x[2] for x in self.storage]
        lookup={key:self.meshes[i].id for i,key in enumerate(unique)}
        self.mesh_ids=wp.array([lookup[int(i)] for i in mesh_indices],dtype=wp.uint64,device=str(device))
        self.rotation=torch.tensor(Rotation.from_quat(np.asarray(object_pose)[:,3:],scalar_first=True).as_matrix(),dtype=torch.float32,device=device)
        self.origin=torch.tensor(np.asarray(object_pose)[:,:3],dtype=torch.float32,device=device)
        self.scale=torch.tensor(scene_scale,dtype=torch.float32,device=device)
        self.cache=cache
        self.watertight=torch.tensor([geometries[int(i)].get('watertight',True) for i in mesh_indices],dtype=torch.bool,device=device)
        # Upload/build on Warp's stream, then establish a dependency without a host sync.
        if self.device.type=='cuda':
            torch.cuda.current_stream(self.device).wait_stream(wp.stream_to_torch(wp.get_stream(str(device))))
    def query(self,world,rows):
        n,k,_=world.shape;r=self.rotation[rows];s=self.scale[rows]
        local=torch.einsum('bij,bkj->bki',r.transpose(-1,-2),world-self.origin[rows,None])/s[:,None,None]
        source=wp.from_torch(local.contiguous().reshape(-1,3),dtype=wp.vec3)
        indexes=rows.to(torch.int32).repeat_interleave(k).contiguous();shape=n*k
        q=wp.empty(shape,dtype=wp.vec3,device=str(self.device));normal=wp.empty_like(q)
        dist=wp.empty(shape,dtype=float,device=str(self.device));face=wp.empty(shape,dtype=int,device=str(self.device));bary=wp.empty_like(q)
        wp.launch(mesh_query,shape,inputs=[self.mesh_ids,wp.from_torch(indexes,dtype=wp.int32),source],
                  outputs=[q,normal,dist,face,bary],device=str(self.device),
                  stream=wp.stream_from_torch(self.device) if self.device.type=='cuda' else None)
        projected=torch.einsum('bij,bkj->bki',r,wp.to_torch(q).reshape(n,k,3)*s[:,None,None])+self.origin[rows,None]
        normals=torch.einsum('bij,bkj->bki',r,wp.to_torch(normal).reshape(n,k,3))
        return projected,normals,wp.to_torch(dist).reshape(n,k)*s[:,None],wp.to_torch(face).reshape(n,k),wp.to_torch(bary).reshape(n,k,3)


class WarmStartBank:
    """Bounded GPU cache of solved poses in canonical object coordinates."""
    def __init__(self,device,settings):
        self.device=torch.device(device);self.settings=settings;self.entries=collections.OrderedDict()
    def lookup(self,keys,target,surfaces):
        target=torch.as_tensor(target,dtype=torch.float32,device=self.device)
        n=len(target);variables=torch.zeros((n,6+self.settings.pca_components),device=self.device)
        valid=torch.zeros(n,dtype=torch.bool,device=self.device)
        feature=((target-surfaces.origin[:,None])@surfaces.rotation)/surfaces.scale[:,None,None]
        for key in dict.fromkeys(keys):
            if key not in self.entries:continue
            self.entries.move_to_end(key);entry=self.entries[key]
            ids=torch.tensor([i for i,k in enumerate(keys) if k==key],device=self.device)
            distance=torch.cdist(feature[ids].flatten(1),entry['feature'].flatten(1))
            ratio=surfaces.scale[ids,None]/entry['scale'][None]
            eligible=(ratio>=1/self.settings.warm_size_ratio)&(ratio<=self.settings.warm_size_ratio)
            distance=distance.masked_fill(~eligible,torch.inf);minimum,nearest=distance.min(-1)
            v=entry['variables'][nearest].clone()
            v[:,:3]=torch.einsum('bij,bj->bi',surfaces.rotation[ids],v[:,:3]*surfaces.scale[ids,None])+surfaces.origin[ids]
            rotation=surfaces.rotation[ids]@entry['rotation'][nearest]
            # Small batched SO(3) log, no CPU transfer.
            v[:,3:6]=matrix_to_rotvec(rotation)
            variables[ids]=v;valid[ids]=torch.isfinite(minimum)
        return variables,valid,feature
    def add(self,keys,variables,feature,surfaces):
        local=variables.detach().clone()
        local[:,:3]=torch.einsum('bij,bj->bi',surfaces.rotation.transpose(-1,-2),local[:,:3]-surfaces.origin)/surfaces.scale[:,None]
        rotation=surfaces.rotation.transpose(-1,-2)@rotmat(variables[:,3:6])
        for key in dict.fromkeys(keys):
            ids=torch.tensor([i for i,k in enumerate(keys) if k==key],device=self.device)
            values=dict(variables=local[ids].clone(),rotation=rotation[ids].clone(),feature=feature[ids].clone(),scale=surfaces.scale[ids].clone())
            if key in self.entries:
                old=self.entries.pop(key)
                values={k:torch.cat((old[k],v),0)[-self.settings.warm_capacity:] for k,v in values.items()}
            else:values={k:v[-self.settings.warm_capacity:] for k,v in values.items()}
            self.entries[key]=values
            if len(self.entries)>self.settings.warm_object_capacity:self.entries.popitem(last=False)


def matrix_to_rotvec(matrix):
    """Stable SO(3) logarithm including rotations close to pi."""
    # Quaternion from the largest diagonal element avoids division by sin(theta).
    m=matrix;one=torch.ones(len(m),device=m.device,dtype=m.dtype)
    candidates=torch.stack((1+m[:,0,0]+m[:,1,1]+m[:,2,2],
        1+m[:,0,0]-m[:,1,1]-m[:,2,2],1-m[:,0,0]+m[:,1,1]-m[:,2,2],1-m[:,0,0]-m[:,1,1]+m[:,2,2]),-1).clamp_min(0).sqrt()*.5
    q0,q1,q2,q3=candidates.unbind(-1);den=(4*candidates).clamp_min(1.e-8)
    q=torch.stack((torch.stack((q0,(m[:,2,1]-m[:,1,2])/den[:,0],(m[:,0,2]-m[:,2,0])/den[:,0],(m[:,1,0]-m[:,0,1])/den[:,0]),-1),
        torch.stack(((m[:,2,1]-m[:,1,2])/den[:,1],q1,(m[:,0,1]+m[:,1,0])/den[:,1],(m[:,0,2]+m[:,2,0])/den[:,1]),-1),
        torch.stack(((m[:,0,2]-m[:,2,0])/den[:,2],(m[:,0,1]+m[:,1,0])/den[:,2],q2,(m[:,1,2]+m[:,2,1])/den[:,2]),-1),
        torch.stack(((m[:,1,0]-m[:,0,1])/den[:,3],(m[:,0,2]+m[:,2,0])/den[:,3],(m[:,1,2]+m[:,2,1])/den[:,3],q3),-1)),1)
    ids=candidates.argmax(-1);q=q[torch.arange(len(q),device=m.device),ids]
    q=q*torch.where(q[:,:1]<0,-one[:,None],one[:,None])
    norm=q[:,1:].norm(dim=-1,keepdim=True);angle=2*torch.atan2(norm,q[:,:1])
    return q[:,1:]*torch.where(norm>1.e-7,angle/norm.clamp_min(1.e-12),2*one[:,None])


def initialize(decoder,target):
    """Rigid palm alignment; shape remains beta=0 and unit scale."""
    device=decoder.device;n=len(target)
    variables=np.zeros((n,6+decoder.components),np.float32)
    if not hasattr(decoder,'_neutral_palm_cpu'):
        with torch.no_grad():_,neutral,_=decoder(torch.zeros((1,6+decoder.components),device=device))
        decoder._neutral_palm_cpu=neutral[0,[0,5,9,13,17]].cpu().numpy().copy()
    source=decoder._neutral_palm_cpu;target_np=np.asarray(target)
    for i in range(n):
        dest=target_np[i,[0,5,9,13,17]]
        a=source-source.mean(0);b=dest-dest.mean(0)
        u,_,vh=np.linalg.svd(a.T@b);r=vh.T@u.T
        if np.linalg.det(r)<0:vh[-1]*=-1;r=vh.T@u.T
        variables[i,:3]=dest.mean(0)-source.mean(0)@r.T
        variables[i,3:6]=Rotation.from_matrix(r).as_rotvec()
    return torch.as_tensor(variables,device=device)


def skeleton_loss(decoder,variables,target):
    v,joints,_=decoder(variables)
    edges=decoder.edges
    a=joints[:,edges[:,1]]-joints[:,edges[:,0]]
    b=target[:,edges[:,1]]-target[:,edges[:,0]]
    a=torch.nn.functional.normalize(a,dim=-1);b=torch.nn.functional.normalize(b,dim=-1)
    direction=(a-b).square().sum(-1).mean(-1)
    # Mild position term accommodates unequal human/robot segment lengths.
    positions=(joints-target).square().sum(-1).mean(-1)/(.04**2)
    prior=variables[:,6:].square().mean(-1)
    return 6*direction+.35*positions+.015*prior,v,joints


def vertex_normals(vertices,faces):
    v=np.asarray(vertices);f=np.asarray(faces)
    normals=np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]])
    out=np.zeros_like(v)
    for j in range(3):np.add.at(out,f[:,j],normals)
    return out/np.maximum(np.linalg.norm(out,axis=-1,keepdims=True),1e-12)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def residuals(decoder,variables,target):
    joints=decoder.joints_only(variables)
    edges=decoder.edges
    a=torch.nn.functional.normalize(joints[:,edges[:,1]]-joints[:,edges[:,0]],dim=-1)
    b=torch.nn.functional.normalize(target[:,edges[:,1]]-target[:,edges[:,0]],dim=-1)
    return torch.cat(((a-b).flatten(1)*math.sqrt(6/20),
        (joints-target).flatten(1)*(math.sqrt(.35/21)/.04),
        variables[:,6:]*math.sqrt(.015/decoder.components)),1)


def kinematic_loss(decoder,variables,target):
    return residuals(decoder,variables,target).square().sum(-1)


def project_ik(values,initial):
    return torch.cat((torch.maximum(initial[:,:3]-.06,torch.minimum(initial[:,:3]+.06,values[:,:3])),
        values[:,3:6],values[:,6:].clamp(-3.,3.)),1)


def lm_ik(decoder,initial,target,settings):
    """Batched forward-mode AD Jacobians and damped 30x30 systems, all on device."""
    n,d=initial.shape;device=initial.device
    scale=torch.tensor([.01]*3+[.1]*3+[.3]*decoder.components,device=device,dtype=initial.dtype)
    values=initial.detach().clone();iterations=torch.zeros(n,dtype=torch.int32,device=device)
    damping=torch.full((n,),settings.lm_damping,device=device);active=torch.ones(n,dtype=torch.bool,device=device)
    stale=torch.zeros(n,dtype=torch.int32,device=device);eye=torch.eye(d,device=device)
    def single(u,t):return residuals(decoder,(u*scale)[None],t[None])[0]
    jacobian=torch.func.vmap(torch.func.jacfwd(single,argnums=0))
    history=[]
    with torch.no_grad():cost=kinematic_loss(decoder,values,target);initial_cost=cost.clone()
    ids=torch.arange(n,device=device)
    for step in range(settings.ik_steps):
        current=values[ids];desired=target[ids]
        with torch.enable_grad():j=jacobian(current/scale,desired).detach()
        with torch.no_grad():
            r=residuals(decoder,current,desired);jt=j.transpose(1,2);h=jt@j;g=(jt@r[:,:,None]).squeeze(-1)
            diag=h.diagonal(dim1=-2,dim2=-1).clamp_min(1.e-4)
            system=h+damping[ids,None,None]*torch.diag_embed(diag)+eye*1.e-7
            delta,info=torch.linalg.solve_ex(system,-g)
            finite=torch.isfinite(delta).all(-1)&(info==0)
            delta=torch.where(finite[:,None],delta,torch.zeros_like(delta))
            delta=delta*(settings.lm_max_step/delta.norm(dim=-1).clamp_min(settings.lm_max_step))[:,None]
            proposal=project_ik(current+delta*scale,initial[ids])
            new_cost=kinematic_loss(decoder,proposal,desired);old=cost[ids]
            accept=finite&torch.isfinite(new_cost)&(new_cost<old)
            values[ids]=torch.where(accept[:,None],proposal,current)
            cost[ids]=torch.minimum(old,torch.where(torch.isfinite(new_cost),new_cost,old))
            damping[ids]=torch.where(accept,damping[ids]*.5,damping[ids]*4).clamp(1.e-6,1.e6)
            iterations[ids]+=1;history.append(cost.clone())
            small=(old-cost[ids])<=settings.stop_absolute_tol+settings.lm_relative_tol*old.abs()
            stale[ids]=torch.where(small,stale[ids]+1,torch.zeros_like(stale[ids]))
            if settings.early_stop and step>=2:
                done=(stale[ids]>=3)|((g.abs().max(-1).values<settings.lm_gradient_tol)&finite)
                active[ids]&=~done;ids=torch.nonzero(active,as_tuple=True)[0]
                if not len(ids):break
    return values.detach(),iterations,dict(initial_cost=initial_cost,final_cost=cost,history=history)


def cem_update(elites,old_mean,old_std,base_sigma,momentum,floor):
    fitted=elites.mean(1);variance=(elites-fitted[:,None]).square().mean(1)
    mean=momentum*old_mean+(1-momentum)*fitted
    std=(momentum*old_std.square()+(1-momentum)*variance).clamp_min(0).sqrt()
    return mean,torch.maximum(std,base_sigma*floor)


def optimize(decoder,target,surfaces,settings,seed,fixed_contacts,source_contacts,source_normals,
             warm_start=None,warm_available=None,on_solved=None):
    """Independent populations, per-row early exits and deferred CPU exports."""
    s=settings;device=decoder.device
    target_cpu=target.detach().cpu().numpy() if isinstance(target,torch.Tensor) else np.asarray(target)
    target_cpu=np.asarray(target_cpu,dtype=np.float32)
    initial=initialize(decoder,target_cpu)
    target=torch.as_tensor(target_cpu,dtype=torch.float32,device=device)
    n=len(target);rows=torch.arange(n,device=device)
    warm_input=torch.zeros_like(initial) if warm_start is None else warm_start.detach().clone()
    warm_used=torch.zeros(n,dtype=torch.bool,device=device)
    with torch.no_grad():
        if s.warm_start and warm_start is not None:
            warm=warm_start.clone()
            warm[:,:3]=torch.maximum(initial[:,:3]-.06,torch.minimum(initial[:,:3]+.06,warm[:,:3]))
            warm[:,6:].clamp_(-3.,3.)
            cold_cost=kinematic_loss(decoder,initial,target)
            warm_cost=kinematic_loss(decoder,warm,target)
            available=torch.ones(n,dtype=torch.bool,device=device) if warm_available is None else warm_available
            warm_used=available&torch.isfinite(warm_cost)&(warm_cost<cold_cost)
            initial_selected=torch.where(warm_used[:,None],warm,initial)
        else:initial_selected=initial
    mean,ik_iterations,_=lm_ik(decoder,initial_selected,target,s)
    anchor=mean.clone()
    if not isinstance(fixed_contacts,FixedContacts):raise TypeError('Reviewed fixed contacts required')
    source_contacts=torch.as_tensor(source_contacts,dtype=torch.float32,device=device)
    source_normals=torch.as_tensor(source_normals,dtype=torch.float32,device=device)
    with torch.no_grad():source_projection,_,_,_,_=surfaces.query(source_contacts,rows)
    low=anchor.clone();high=anchor.clone()
    low[:,:3]-=s.max_root_delta_m;high[:,:3]+=s.max_root_delta_m
    low[:,3:6]-=s.max_root_delta_rad;high[:,3:6]+=s.max_root_delta_rad
    low[:,6:]=-3;high[:,6:]=3
    sigma=torch.tensor([s.pos_noise_m]*3+[s.rot_noise_rad]*3+[s.pca_noise]*decoder.components,device=device)
    def project(x,idx):return torch.maximum(low[idx],torch.minimum(high[idx],x))
    def cost(x,idx):
        bone,v,_=skeleton_loss(decoder,x,target[idx]);mapped,hand_normal=fixed_contacts.evaluate(v)
        _,_,distance,_,_=surfaces.query(v,idx)
        projected,object_normal,contact_distance,_,_=surfaces.query(mapped,idx)
        contact=(contact_distance/.005).square().mean(-1)
        normal=(1+(hand_normal*object_normal).sum(-1).clamp(-1.,1.)).square().mean(-1)
        correspondence=((projected-source_projection[idx])/.01).square().sum(-1).mean(-1)
        pen=(torch.relu(-distance-s.penetration_tolerance_m)/.002).square()
        penetration=pen.mean(-1)+.25*pen.max(-1).values
        trust=((x[:,:3]-anchor[idx,:3])/.03).square().mean(-1)
        return bone+.3*contact+.15*normal+.1*correspondence+12*penetration+.1*trust
    def geometry_ok(x,idx):
        v,_,_=decoder(x,full=True);_,_,d,_,_=surfaces.query(v,idx)
        mapped,hn=fixed_contacts.evaluate(v,full=True);_,on,md,_,_=surfaces.query(mapped,idx)
        coverage=((md.abs()<=s.contact_tolerance_m)&((hn*on).sum(-1)<=-math.cos(math.pi/6))).float().mean(-1)
        return ((-d).clamp_min(0).max(-1).values<=s.penetration_tolerance_m)&((d.abs()<=s.contact_tolerance_m).sum(-1)>=2)&(coverage>=.8)
    generator=torch.Generator(device=device).manual_seed(seed)
    history=torch.empty((s.iterations,n),device=device)
    iterations=torch.zeros(n,dtype=torch.int32,device=device);stale=torch.zeros_like(iterations)
    reasons=torch.zeros(n,dtype=torch.int32,device=device) # 0 budget, 1 geometry, 2 plateau, 3 geometry recheck failed
    active_ids=rows
    std=sigma[None].expand(n,-1).clone()
    elite_count=max(2,math.ceil(s.population*s.cem_elite_fraction))
    elite_bank=mean[:,None].expand(-1,elite_count,-1).clone();elite_valid=0
    evaluations=torch.zeros(n,dtype=torch.int64,device=device)
    with torch.no_grad():
        best=mean.clone();best_cost=cost(best,rows);start_cost=best_cost.clone();checked=best_cost.clone()
        if s.early_stop:
            ready=geometry_ok(best,rows)
            if hasattr(surfaces,'watertight'):ready&=surfaces.watertight
            reasons=torch.where(ready,torch.ones_like(reasons),reasons)
            active_ids=rows[~ready]
            if not len(active_ids):history[:]=best_cost
        for iteration in range(s.iterations if len(active_ids) else 0):
            a=len(active_ids);population=s.population
            scale=std[active_ids,None]
            # Preserve each row's random stream when other rows exit early.
            noise=torch.randn((n,population,mean.shape[-1]),device=device,generator=generator)[active_ids]*scale
            idx=active_ids.repeat_interleave(population)
            values=project((mean[active_ids,None]+noise).reshape(-1,mean.shape[-1]),idx).reshape(a,population,-1)
            values[:,0]=mean[active_ids];values[:,1]=best[active_ids]
            if elite_valid:
                reuse=min(elite_valid,int(population*s.cem_reuse_fraction),population-2)
                if reuse:values[:,-reuse:]=elite_bank[active_ids,:reuse]
            flat=values.reshape(-1,mean.shape[-1]);evaluated=[]
            for start in range(0,len(flat),s.candidate_chunk):
                evaluated.append(cost(flat[start:start+s.candidate_chunk],idx[start:start+s.candidate_chunk]))
            costs=torch.cat(evaluated).reshape(a,population);costs=torch.where(torch.isfinite(costs),costs,torch.inf)
            arange=torch.arange(a,device=device);winners=costs.argmin(-1)
            candidate=values[arange,winners];minimum=costs[arange,winners]
            old=best_cost[active_ids];improve=minimum<old
            best[active_ids]=torch.where(improve[:,None],candidate,best[active_ids]);best_cost[active_ids]=torch.minimum(old,minimum)
            keep=max(2,math.ceil(population*s.cem_elite_fraction))
            top,indices=costs.topk(keep,largest=False)
            selected=values.gather(1,indices[:,:,None].expand(-1,-1,values.shape[-1]))
            proposed,new_std=cem_update(selected,mean[active_ids],std[active_ids],sigma,s.cem_momentum,s.cem_sigma_floor)
            delta=(proposed-mean[active_ids])*s.cem_mean_step
            bound=sigma*s.cem_mean_step_limit
            delta=torch.maximum(-bound,torch.minimum(bound,delta))
            proposed=project(mean[active_ids]+delta,active_ids)
            std[active_ids]=new_std;elite_bank[active_ids,:keep]=selected;elite_valid=keep
            proposed_cost=cost(proposed,active_ids);improve=proposed_cost<best_cost[active_ids]
            best[active_ids]=torch.where(improve[:,None],proposed,best[active_ids]);best_cost[active_ids]=torch.minimum(best_cost[active_ids],proposed_cost)
            mean[active_ids]=torch.where(torch.isfinite(proposed_cost)[:,None],proposed,best[active_ids])
            iterations[active_ids]+=1;evaluations[active_ids]+=population+1;history[iteration]=best_cost
            if s.early_stop and (iteration+1)%s.search_check_every==0:
                ok=geometry_ok(best[active_ids],active_ids)
                # A nonwatertight mesh can never pass geometric validation.
                if hasattr(surfaces,'watertight'):ok&=surfaces.watertight[active_ids]
                done=ok
                reasons[active_ids]=torch.where(ok,torch.ones_like(reasons[active_ids]),reasons[active_ids])
                threshold=s.stop_absolute_tol+s.stop_relative_tol*checked[active_ids].abs()
                stalled=(checked[active_ids]-best_cost[active_ids])<=threshold
                stale[active_ids]=torch.where(stalled,stale[active_ids]+1,torch.zeros_like(stale[active_ids]))
                checked[active_ids]=best_cost[active_ids]
                if iteration+1>=s.search_min_iterations:
                    plateau=(stale[active_ids]>=s.search_patience)&~ok
                    reasons[active_ids]=torch.where(plateau,torch.full_like(reasons[active_ids],2),reasons[active_ids]);done|=plateau
                active_ids=active_ids[~done]
                if not len(active_ids):
                    history[iteration+1:]=best_cost;break
        vertices,joints,frames=decoder(best,full=True);p,norm,d,face,bary=surfaces.query(vertices,rows)
        angles=decoder.angles(best);mapped,mapped_normals=fixed_contacts.evaluate(vertices,full=True)
        mapped_projected,mapped_object_normals,mapped_distance,mapped_face,mapped_bary=surfaces.query(mapped,rows)
        coverage=((mapped_distance.abs()<=s.contact_tolerance_m)&((mapped_normals*mapped_object_normals).sum(-1)<=-math.cos(math.pi/6))).float().mean(-1)
        final_geometry=(((-d).clamp_min(0).max(-1).values<=s.penetration_tolerance_m)&
                        ((d.abs()<=s.contact_tolerance_m).sum(-1)>=2)&(coverage>=.8))
        if hasattr(surfaces,'watertight'):final_geometry&=surfaces.watertight
        reasons=torch.where((reasons==1)&~final_geometry,torch.full_like(reasons,3),reasons)
    if not all(torch.isfinite(t).all() for t in (best,best_cost,vertices,d)):raise RuntimeError('Nonfinite MANO optimization')
    if on_solved is not None:
        with torch.no_grad():on_solved(best)
    # One export phase; no NumPy round-trip inside either optimization loop.
    tensors=dict(variables=best,angles=angles,vertices=vertices,joints=joints,frames=frames,
        projected=p,object_normals=norm,distance=d,object_face=face,object_bary=bary,
        mapped_hand_points=mapped,mapped_hand_normals=mapped_normals,mapped_object_points=mapped_projected,
        mapped_object_normals=mapped_object_normals,mapped_distance=mapped_distance,
        mapped_object_face_ids=mapped_face,mapped_object_barycentric=mapped_bary,
        source_annotation_points=source_contacts,source_annotation_normals=source_normals,source_annotation_object_projection=source_projection,
        initial_cost=start_cost,final_cost=best_cost,history=history,ik_iterations=ik_iterations,cem_iterations=iterations,
        stop_reason=reasons,warm_start_used=warm_used,warm_start_variables=warm_input,candidate_evaluations=evaluations)
    return {k:v.detach().cpu().numpy() for k,v in tensors.items()}
