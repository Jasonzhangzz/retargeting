"""Bounded contact-preserving Wuji2 -> native Moxian mesh IK.

Actual surface samples determine source contacts. Source objects remain fixed.
The LM variables are 25 native hinges plus a floating rigid base; analytical
skinning Jacobians keep independent batched solves practical on CUDA.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
import torch
import warp as wp
from scipy.spatial.transform import Rotation
from moxian_web.qnbot_surface import read_skin
from retarget.env.grasp.hand import palmar_axis
from .moxian_contact_kernels import witness_cost,clearance_query
from .triangle_penetration import depth_kernel

FINGERS = ('thumb', 'index', 'middle', 'ring', 'little')
REGIONS = ('palm',) + tuple(f'{f}_{p}' for f in FINGERS for p in
                          (('metacarpal','proximal','distal') if f=='thumb' else
                           ('proximal','intermediate','distal')))


@dataclass(frozen=True)
class Settings:
    iterations: int = 75
    starts: int = 3
    source_samples: int = 384
    region_samples: int = 256
    collision_samples: int = 1536
    source_gap_m: float = .003
    max_anchor_error_m: float = .005
    max_penetration_m: float = .002
    min_normal_dot: float = .5
    damping: float = .02
    normal_length_m: float = .008
    seed: int = 20261117


@wp.kernel(enable_backward=False)
def fk_kernel(q: wp.array2d[float], parents: wp.array[int], joint: wp.array[int],
              kinds: wp.array[int], axes: wp.array[wp.vec3], pivots: wp.array[wp.vec3],
              rest: wp.array[float], p0: wp.array[wp.vec3], r0: wp.array[wp.mat33],
              positions: wp.array2d[wp.vec3], rotations: wp.array2d[wp.mat33],
              world_axes: wp.array2d[wp.vec3], world_pivots: wp.array2d[wp.vec3], count: int):
    b = wp.tid()
    positions[b,0] = wp.vec3(0.)
    rotations[b,0] = wp.identity(n=3,dtype=float)
    for i in range(1,count):
        parent = parents[i]
        pr = rotations[b,parent]
        p = positions[b,parent] + pr @ p0[i]
        r = pr @ r0[i]
        j = joint[i]
        if j >= 0:
            x = q[b,j] - rest[j]
            if kinds[j] == 2:
                p = p + (r @ axes[j]) * x
            else:
                turn = wp.quat_to_matrix(wp.quat_from_axis_angle(axes[j],x))
                p = p + r @ (pivots[j] - turn @ pivots[j])
                r = r @ turn
            world_axes[b,j] = r @ axes[j]
            world_pivots[b,j] = p + r @ pivots[j]
        positions[b,i] = p
        rotations[b,i] = r


@wp.kernel(enable_backward=False)
def surface_kernel(meshes: wp.array[wp.uint64], points: wp.array2d[wp.vec3],
                   closest: wp.array2d[wp.vec3], normal: wp.array2d[wp.vec3], gap: wp.array2d[float]):
    b,k = wp.tid()
    mesh = meshes[b]; p = points[b,k]
    hit = wp.mesh_query_point(mesh,p,1.e6)
    if hit.result:
        x = wp.mesh_eval_position(mesh,hit.face,hit.u,hit.v)
        a = wp.mesh_get_point(mesh,3*hit.face)
        c = wp.mesh_get_point(mesh,3*hit.face+1)
        d = wp.mesh_get_point(mesh,3*hit.face+2)
        n = wp.normalize(wp.cross(c-a,d-a))
        distance = wp.length(p-x)
        if distance > 1.e-7:
            n = hit.sign * (p-x) / distance
        closest[b,k] = x; normal[b,k] = n; gap[b,k] = distance * hit.sign
    else:
        closest[b,k] = p; normal[b,k] = wp.vec3(1.,0.,0.); gap[b,k] = 1.e6


class Rig:
    def __init__(self, model, device):
        self.model, self.device = model, str(device)
        if model.body_jntnum.max()>1 or any(k not in (2,3) for k in model.jnt_type):
            raise ValueError('Requires native serial scalar joints')
        self.parents = model.body_parentid
        joint = np.where(model.body_jntnum>0,model.body_jntadr,-1).astype(np.int32)
        def w(a,dtype): return wp.array(np.asarray(a),dtype=dtype,device=device)
        self.args = [w(self.parents,wp.int32), w(joint,wp.int32),w(model.jnt_type,wp.int32),
                     w(model.jnt_axis,wp.vec3),w(model.jnt_pos,wp.vec3),w(model.qpos0,wp.float32),
                     w(model.body_pos,wp.vec3),w(Rotation.from_quat(model.body_quat[:,[1,2,3,0]]).as_matrix(),wp.mat33)]
        self.ancestor = np.zeros((model.nbody,model.nq),np.float32)
        for b in range(1,model.nbody):
            self.ancestor[b] = self.ancestor[self.parents[b]]
            if joint[b]>=0:self.ancestor[b,joint[b]]=1

    def frames(self,q):
        n = len(q); m = self.model
        p = q.new_empty((n,m.nbody,3));r = q.new_empty((n,m.nbody,3,3))
        axes = q.new_empty((n,m.nq,3));pivots = torch.empty_like(axes)
        stream = wp.stream_from_torch(torch.cuda.current_stream(q.device)) if q.is_cuda else None
        wp.launch(fk_kernel,n,inputs=[wp.from_torch(q.contiguous()),*self.args,
            wp.from_torch(p,dtype=wp.vec3),wp.from_torch(r,dtype=wp.mat33),
            wp.from_torch(axes,dtype=wp.vec3),wp.from_torch(pivots,dtype=wp.vec3),m.nbody],device=self.device,stream=stream)
        return p,r,axes,pivots


class Objects:
    def __init__(self,dataset,device):
        self.dataset,self.device=dataset,str(device); self.cache={}
    def get(self,record):
        name=record['mesh']
        if name not in self.cache:
            v,f=self.dataset.object_mesh(name)
            points=wp.array(v,dtype=wp.vec3,device=self.device)
            indices=wp.array(f.reshape(-1),dtype=wp.int32,device=self.device)
            self.cache[name]=wp.Mesh(points=points,indices=indices)
        return self.cache[name].id
    def batch(self,grid,ids):
        from visualization.viser_moxian_dataset import text
        mesh_ids=[self.get(self.dataset.objects['scenes'][text(grid['object_scene'][i])]) for i in ids]
        self.mesh_ids=wp.array(mesh_ids,dtype=wp.uint64,device=self.device)
        self.scale=torch.as_tensor(grid['object_scale'][ids],device=self.device,dtype=torch.float32)
        if not np.allclose(grid['object_scale'][ids,0],grid['object_scale'][ids,1:].T,rtol=1e-5):
            raise ValueError('Contact solver requires isotropic object scales')
        pose=grid['object_pose_world'][ids]
        self.origin=torch.as_tensor(pose[:,:3],device=self.device,dtype=torch.float32)
        self.rotation=torch.as_tensor(Rotation.from_quat(pose[:,[4,5,6,3]]).as_matrix(),device=self.device,dtype=torch.float32)
        bounds=[self.dataset.object_mesh(self.dataset.objects['scenes'][text(grid['object_scene'][i])]['mesh'])[0] for i in ids]
        self.low=torch.as_tensor(np.stack([v.min(0) for v in bounds]),device=self.device,dtype=torch.float32)
        self.high=torch.as_tensor(np.stack([v.max(0) for v in bounds]),device=self.device,dtype=torch.float32)
    def query(self,world):
        local=torch.einsum('nki,nij->nkj',world-self.origin[:,None],self.rotation)/self.scale[:,None]
        closest=torch.empty_like(local);normal=torch.empty_like(local);gap=world.new_empty(world.shape[:2])
        stream=wp.stream_from_torch(torch.cuda.current_stream(world.device)) if world.is_cuda else None
        wp.launch(surface_kernel,world.shape[:2],inputs=[self.mesh_ids,wp.from_torch(local.contiguous(),dtype=wp.vec3),
            wp.from_torch(closest,dtype=wp.vec3),wp.from_torch(normal,dtype=wp.vec3),wp.from_torch(gap)],device=self.device,stream=stream)
        return (torch.einsum('nki,nji->nkj',closest*self.scale[:,None],self.rotation)+self.origin[:,None],
                torch.einsum('nki,nji->nkj',normal,self.rotation),gap*self.scale[:,:1])

    def clearance(self,world):
        local=torch.einsum('nki,nij->nkj',world-self.origin[:,None],self.rotation)/self.scale[:,None]
        normal=torch.empty_like(local);gap=world.new_empty(world.shape[:2]);margin=.0015/self.scale[:,0]
        stream=wp.stream_from_torch(torch.cuda.current_stream(world.device)) if world.is_cuda else None
        wp.launch(clearance_query,world.shape[:2],inputs=[self.mesh_ids,wp.from_torch(local.contiguous(),dtype=wp.vec3),
            wp.from_torch(self.low,dtype=wp.vec3),wp.from_torch(self.high,dtype=wp.vec3),wp.from_torch(margin),
            wp.from_torch(normal,dtype=wp.vec3),wp.from_torch(gap)],device=self.device,stream=stream)
        return torch.einsum('nki,nji->nkj',normal,self.rotation),gap*self.scale[:,:1]

    def penetration(self,world):
        local=torch.einsum('nki,nij->nkj',world-self.origin[:,None],self.rotation)/self.scale[:,None]
        n=len(world);depth=world.new_zeros(n);invalid=torch.zeros(n,device=world.device,dtype=torch.int32)
        rows=torch.arange(n,device=world.device,dtype=torch.int32)
        stream=wp.stream_from_torch(torch.cuda.current_stream(world.device)) if world.is_cuda else None
        wp.launch(depth_kernel,world.shape[:2],inputs=[self.mesh_ids,wp.from_torch(rows),
            wp.from_torch(local.contiguous(),dtype=wp.vec3),wp.from_torch(self.low,dtype=wp.vec3),
            wp.from_torch(self.high,dtype=wp.vec3),wp.from_torch(self.scale[:,0].contiguous()),
            wp.from_torch(depth),wp.from_torch(invalid)],device=self.device,stream=stream)
        return depth,invalid==0


def cross_matrix(v):
    x,y,z=v.unbind(-1);o=torch.zeros_like(x)
    return torch.stack((o,-z,y,z,o,-x,-y,x,o),-1).reshape(*v.shape[:-1],3,3)


def turn_matrix(v):
    k=cross_matrix(v);theta=torch.linalg.vector_norm(v,dim=-1)
    a=torch.sinc(theta/torch.pi)[...,None,None]
    b=(.5*torch.sinc(theta/(2*torch.pi)).square())[...,None,None]
    return torch.eye(3,device=v.device)+a*k+b*(k@k)


def triangle_witness(tri,point):
    """Exact closest point and barycentric weights, including edges/vertices."""
    a,b,c=tri.unbind(-2);ab=b-a;ac=c-a;ap=point-a
    aa=(ab*ab).sum(-1);bb=(ac*ac).sum(-1);cc=(ab*ac).sum(-1)
    x=(ap*ab).sum(-1);y=(ap*ac).sum(-1);den=(aa*bb-cc*cc).clamp_min(1e-20)
    u=(bb*x-cc*y)/den;v=(aa*y-cc*x)/den
    interior=(u>=0)&(v>=0)&(u+v<=1)
    face=torch.stack((1-u-v,u,v),-1)
    def fraction(start,end):
        d=end-start
        return (((point-start)*d).sum(-1)/d.square().sum(-1).clamp_min(1e-20)).clamp(0,1)
    t=fraction(a,b);s=fraction(b,c);r=fraction(c,a);zero=torch.zeros_like(t)
    bary=torch.stack((face,torch.stack((1-t,t,zero),-1),
                      torch.stack((zero,1-s,s),-1),torch.stack((r,zero,1-r),-1)),-2)
    candidates=(bary[...,None]*tri[...,None,:,:]).sum(-2)
    cost=(candidates-point[...,None,:]).square().sum(-1)
    cost[...,0]=cost[...,0].masked_fill(~interior,1e6)
    best=cost.argmin(-1)
    weights=bary.gather(-2,best[...,None,None].expand(*best.shape,1,3)).squeeze(-2)
    return (weights[...,None]*tri).sum(-2),weights


class Skin:
    def __init__(self,root,device,settings):
        self.device=device;self.settings=settings
        asset=Path(root)/'hand_assets/moxian'
        self.model=mujoco.MjModel.from_xml_path(str(asset/'resources/mujoco/right.xml'))
        self.rig=Rig(self.model,device)
        self.raw=read_skin(asset/'resources/meshes/right/visual/right_hand_hd.skn')
        s=self.raw;v=s['vertices'];f=s['faces'];tri=v[f]
        self.bodies=[self.model.body(n).id for n in s['bone_names']]
        normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normals/=np.maximum(np.linalg.norm(normals,axis=-1,keepdims=True),1e-12)
        weights=s['weights'][f].mean(1);dominant=weights.argmax(1)
        names=np.array([n.removeprefix('right_') for n in s['bone_names']])
        palm_normal=np.mean([-s['bind_rotation'][i][:,1] for i,n in enumerate(names) if n.endswith('_distal') and not n.startswith('thumb')],0)
        palm_normal/=np.linalg.norm(palm_normal)
        rng=np.random.default_rng(settings.seed);pool=[];pool_regions=[]
        for name in REGIONS:
            if name=='palm':
                bones=[i for i,n in enumerate(names) if n=='wrist' or (n.endswith('metacarpal') and not n.startswith('thumb'))]
                mask=np.isin(dominant,bones)
                facing=normals@palm_normal>.1
            else:
                k=list(names).index(name);axis=-s['bind_rotation'][k][:,1]
                mask=dominant==k
                facing=normals@axis>-.15
            halves=[]
            for side,count in ((facing,settings.region_samples//2),(~facing,settings.region_samples-settings.region_samples//2)):
                choices=np.flatnonzero(mask&side)
                if not len(choices):choices=np.flatnonzero(mask)
                if not len(choices):raise ValueError('Empty skin region '+name)
                halves.extend(choices[np.linspace(0,len(choices)-1,count).astype(int)])
            chosen=np.array(halves)
            pool.extend(chosen);pool_regions.extend([REGIONS.index(name)]*len(chosen))
        self.contact_count=len(pool)
        # Collision witnesses include both skin vertices and face interiors.
        coll_faces=rng.choice(len(f),settings.collision_samples//2,replace=False)
        pool.extend(coll_faces);pool_regions.extend([-1]*len(coll_faces))
        faces=np.array(pool);vtri=v[f[faces]];wtri=s['weights'][f[faces]]
        w=wtri.mean(1)
        local=np.stack([((vtri-p)@r*wtri[:,:,i,None]).mean(1)
                        for i,(p,r) in enumerate(zip(s['bind_position'],s['bind_rotation']))],1)
        nlocal=np.einsum('pbi,bij->pbj',normals[faces,None]*w[:,:,None],s['bind_rotation'])
        vertex_ids=np.linspace(0,len(v)-1,settings.collision_samples//2).astype(int)
        vw=s['weights'][vertex_ids]
        vl=np.stack([(v[vertex_ids]-p)@r*vw[:,i,None] for i,(p,r) in enumerate(zip(s['bind_position'],s['bind_rotation']))],1)
        local=np.concatenate((local,vl));w=np.concatenate((w,vw));nlocal=np.concatenate((nlocal,np.zeros_like(vl)))
        def t(x):return torch.as_tensor(np.asarray(x),device=device,dtype=torch.float32)
        self.local,self.weights,self.normals=t(local),t(w),t(nlocal)
        self.contact_faces=f[faces[:self.contact_count]]
        self.tri_weights=t(wtri[:self.contact_count])
        self.tri_local=t(np.stack([((vtri[:self.contact_count]-p)@r)*wtri[:self.contact_count,:,i,None]
                         for i,(p,r) in enumerate(zip(s['bind_position'],s['bind_rotation']))],2))
        self.full_local=t(np.stack([(v-p)@r*s['weights'][:,i,None]
                          for i,(p,r) in enumerate(zip(s['bind_position'],s['bind_rotation']))],1))
        self.full_weights=t(s['weights'])
        self.anc=t(self.rig.ancestor[self.bodies]);self.limits=t(self.model.jnt_range)
        self.names=[self.model.joint(i).name for i in range(self.model.nq)]
        self.ids=torch.arange(self.contact_count,device=device).reshape(len(REGIONS),settings.region_samples)
        # Longitudinal material coordinates prevent a distal contact migrating to MCP.
        centers=tri[faces[:self.contact_count]].mean(1);coords=[];material=[]
        rest=mujoco.MjData(self.model);mujoco.mj_kinematics(self.model,rest)
        for region in REGIONS:
            block=centers[len(coords)*settings.region_samples:(len(coords)+1)*settings.region_samples]
            block_normal=normals[faces[len(coords)*settings.region_samples:(len(coords)+1)*settings.region_samples]]
            if region=='palm':
                start=rest.xpos[self.model.body('right_wrist').id]
                end=np.mean([rest.xpos[self.model.body('right_'+f+'_proximal').id] for f in FINGERS[1:]],0)
                pad=palm_normal;coord=np.full(len(block),.5)
            else:
                finger,part=region.split('_');bone=list(names).index(region)
                sequence=['metacarpal','proximal','distal','tip'] if finger=='thumb' else ['proximal','intermediate','distal','tip']
                start=s['bind_position'][bone]
                body=self.model.body('right_'+finger+'_'+sequence[sequence.index(part)+1]).id
                end=rest.xpos[body];axis=end-start;coord=np.clip((block-start)@axis/(axis@axis),0,1)
                pad=-s['bind_rotation'][bone][:,1]
            longitudinal=end-start;longitudinal/=np.linalg.norm(longitudinal)
            pad=pad-longitudinal*np.dot(pad,longitudinal);pad/=np.linalg.norm(pad)
            basis=np.stack((longitudinal,pad,np.cross(longitudinal,pad)),1)
            coords.append(coord);material.append(block_normal@basis)
        self.u=t(np.stack(coords));self.material_normal=t(np.stack(material))

    def state(self,q,R,t):
        p,r,axes,pivots=self.rig.frames(q)
        r=R[:,None]@r;p=torch.einsum('nbi,nji->nbj',p,R)+t[:,None]
        axes=torch.einsum('nbi,nji->nbj',axes,R)
        pivots=torch.einsum('nbi,nji->nbj',pivots,R)+t[:,None]
        br=r[:,self.bodies];bp=p[:,self.bodies]
        points=torch.einsum('nbij,pbj->npi',br,self.local)+torch.einsum('pb,nbi->npi',self.weights,bp)
        tri=(torch.einsum('nbij,pvbj->npvi',br,self.tri_local)
             +torch.einsum('pvb,nbi->npvi',self.tri_weights,bp))
        normal=torch.zeros_like(points)
        normal[:,:self.contact_count]=torch.nn.functional.normalize(
            torch.linalg.cross(tri[:,:,1]-tri[:,:,0],tri[:,:,2]-tri[:,:,0]),dim=-1)
        return dict(points=points,normals=normal,tri=tri,r=br,p=bp,axes=axes,pivots=pivots,t=t)

    def jacobian(self,state,ids,normal=False):
        if normal:
            return self.normal_jacobian(state,ids)
        local=(self.normals if normal else self.local)[ids];w=self.weights[ids]
        return self.point_jacobian(state,local,w)

    def point_jacobian(self,state,local,w):
        weighted=torch.einsum('nbij,nkbj->nkbi',state['r'],local)
        weighted+=w[...,None]*state['p'][:,None]
        affected=torch.einsum('nkbi,bj->nkji',weighted,self.anc)
        aw=torch.einsum('nkb,bj->nkj',w,self.anc)
        affected-=aw[...,None]*state['pivots'][:,None]
        j=torch.linalg.cross(state['axes'][:,None],affected,dim=-1).transpose(-1,-2)
        value=weighted.sum(2)
        root=-cross_matrix(value-state['t'][:,None]);trans=torch.eye(3,device=value.device).expand(*value.shape[:2],3,3)*.1
        return torch.cat((j,root,trans),-1)

    def normal_jacobian(self,state,ids):
        n,k=ids.shape
        local=self.tri_local[ids].flatten(1,2);weights=self.tri_weights[ids].flatten(1,2)
        j=self.point_jacobian(state,local,weights).reshape(n,k,3,3,31)
        tri=state['tri'].gather(1,ids[...,None,None].expand(-1,-1,3,3))
        a,b=tri[:,:,1]-tri[:,:,0],tri[:,:,2]-tri[:,:,0]
        da,db=j[:,:,1]-j[:,:,0],j[:,:,2]-j[:,:,0]
        dn=(torch.linalg.cross(da.transpose(-1,-2),b[:,:,None],dim=-1)
            +torch.linalg.cross(a[:,:,None],db.transpose(-1,-2),dim=-1)).transpose(-1,-2)
        raw=torch.linalg.cross(a,b);length=raw.norm(dim=-1).clamp_min(1e-12)
        normal=raw/length[...,None]
        return (dn-normal[...,None]*(normal[...,None]*dn).sum(-2,keepdim=True))/length[...,None,None]

    def full_vertices(self,state):
        return (torch.einsum('nbij,vbj->nvi',state['r'],self.full_local)
                +torch.einsum('vb,nbi->nvi',self.full_weights,state['p']))


def source_pool(model,settings):
    """Deterministic area samples on actual visual meshes, grouped anatomically."""
    state=mujoco.MjData(model);mujoco.mj_kinematics(model,state)
    rng=np.random.default_rng(settings.seed);groups={name:[] for name in REGIONS}
    for g in range(model.ngeom):
        if model.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH or model.geom_group[g]!=1:continue
        body=int(model.geom_bodyid[g]);owner=body
        while not model.body(owner).name.startswith('right_finger') and model.body(owner).name!='right_palm_link':owner=int(model.body_parentid[owner])
        name=model.body(owner).name
        if name=='right_palm_link':region='palm'
        else:
            finger=FINGERS[int(name[len('right_finger')])-1];part=int(name[-1])
            region=finger+'_'+(('metacarpal' if finger=='thumb' else 'proximal') if part<=2 else ('proximal' if finger=='thumb' else 'intermediate') if part==3 else 'distal')
        mesh=model.geom_dataid[g];a,n=model.mesh_vertadr[mesh],model.mesh_vertnum[mesh]
        v=model.mesh_vert[a:a+n]@Rotation.from_quat(model.geom_quat[g][[1,2,3,0]]).as_matrix().T+model.geom_pos[g]
        a,n=model.mesh_faceadr[mesh],model.mesh_facenum[mesh];tri=v[model.mesh_face[a:a+n]]
        groups[region].append((body,tri))
    points=[];normals=[];bodies=[];u=[];material=[]
    for region in REGIONS:
        records=groups[region];tri=np.concatenate([a[1] for a in records]);bs=np.concatenate([np.full(len(a[1]),a[0]) for a in records])
        cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);area=np.linalg.norm(cross,axis=-1)
        faces=rng.choice(len(tri),settings.source_samples,p=area/area.sum())
        uv=rng.random((len(faces),2));flip=uv.sum(1)>1;uv[flip]=1-uv[flip]
        selected=tri[faces];p=selected[:,0]+uv[:,:1]*(selected[:,1]-selected[:,0])+uv[:,1:]*(selected[:,2]-selected[:,0])
        nn=cross[faces]/area[faces,None];bid=bs[faces]
        world=np.einsum('kij,kj->ki',state.xmat[bid].reshape(-1,3,3),p)+state.xpos[bid]
        world_normal=np.einsum('kij,kj->ki',state.xmat[bid].reshape(-1,3,3),nn)
        if region=='palm':
            start=state.xpos[model.body('right_palm_link').id]
            end=np.mean([state.xpos[model.body(f'right_finger{i}_link2').id] for i in (2,3,4,5)],0)
            coord=np.full(len(faces),.5);pad=np.tile([0.,1.,0.],(len(faces),1))
        else:
            finger,part=region.split('_');fid=FINGERS.index(finger)+1
            begin=1 if part in ('metacarpal','proximal') and (finger!='thumb' or part=='metacarpal') else 3 if part in ('intermediate','proximal') else 4
            start=state.xpos[model.body(f'right_finger{fid}_link{begin}').id]
            end=state.site_xpos[model.site('right_'+('pinky' if finger=='little' else finger)+'_tip').id] if begin==4 else state.xpos[model.body(f'right_finger{fid}_link{3 if begin==1 else 4}').id]
            axis=end-start;coord=np.clip((world-start)@axis/(axis@axis),0,1)
            pad=[]
            for body in bid:
                owner=int(body)
                while not model.body(owner).name.startswith('right_finger'):owner=int(model.body_parentid[owner])
                rr=state.xmat[owner].reshape(3,3);endpoint=(end-state.xpos[owner])@rr
                pad.append(rr@palmar_axis(model,owner,endpoint))
            pad=np.array(pad)
        longitudinal=end-start;longitudinal/=np.linalg.norm(longitudinal)
        pad=pad-(pad@longitudinal)[:,None]*longitudinal;pad/=np.linalg.norm(pad,axis=-1,keepdims=True)
        sideways=np.cross(longitudinal,pad)
        material.extend(np.stack((world_normal@longitudinal,(world_normal*pad).sum(-1),(world_normal*sideways).sum(-1)),-1))
        points.extend(p);normals.extend(nn);bodies.extend(bid);u.extend(coord)
    return np.array(points),np.array(normals),np.array(bodies),np.array(u),np.array(material)


class Solver:
    def __init__(self,dataset,device='cuda:0',settings=Settings()):
        self.dataset,self.device,self.settings=dataset,device,settings
        self.skin=Skin(dataset.root,device,settings)
        self.source=Rig(dataset.wuji_model,device)
        self.objects=Objects(dataset,device)
        def t(x,dtype=torch.float32):return torch.as_tensor(np.asarray(x),device=device,dtype=dtype)
        p,n,b,u,material=source_pool(dataset.wuji_model,settings)
        self.source_p,self.source_n,self.source_b,self.source_u=t(p),t(n),t(b,torch.long),t(u)
        self.source_material=t(material)

    def contacts(self,grid,ids):
        s=self.settings;n=len(ids)
        q=torch.as_tensor(grid['wuji_qpos'][ids],device=self.device,dtype=torch.float32)
        p,r,_,_=self.source.frames(q);r=r[:,self.source_b];p=p[:,self.source_b]
        point=torch.einsum('nkij,kj->nki',r,self.source_p)+p
        normal=torch.einsum('nkij,kj->nki',r,self.source_n)
        anchor,on,gap=self.objects.query(point)
        facing=(normal*on).sum(-1)<-.1
        distance=gap.abs().reshape(n,len(REGIONS),-1)
        eligible=facing.reshape_as(distance)&(distance<=s.source_gap_m)
        score=distance.masked_fill(~eligible,1.e3)
        best=score.argmin(-1);base=torch.arange(len(REGIONS),device=self.device)*s.source_samples
        idx=best+base[None]
        active=eligible.any(-1)
        def take(a):return a.gather(1,idx[...,None].expand(-1,-1,3))
        return dict(anchor=take(anchor),normal=take(on),source_hand=take(point),active=active,
                    source_gap=gap.gather(1,idx),u=self.source_u[idx],source_count=eligible.sum(-1),material=self.source_material[idx])

    def witnesses(self,state,contacts):
        n=len(state['points']);s=self.settings
        cost=state['points'].new_empty((n,len(REGIONS),s.region_samples));material_ok=torch.empty_like(cost,dtype=torch.bool)
        def v(a):return wp.from_torch(a.contiguous(),dtype=wp.vec3)
        stream=wp.stream_from_torch(torch.cuda.current_stream(cost.device)) if cost.is_cuda else None
        wp.launch(witness_cost,(n,self.skin.contact_count),inputs=[v(state['tri']),v(state['normals']),v(contacts['anchor']),
            v(contacts['normal']),wp.from_torch(self.skin.u),wp.from_torch(contacts['u']),v(self.skin.material_normal),
            v(contacts['material']),s.region_samples,s.normal_length_m,wp.from_torch(cost),wp.from_torch(material_ok)],device=self.device,stream=stream)
        cost=cost.masked_fill(~material_ok&material_ok.any(-1,keepdim=True),1e3)
        idx=cost.argmin(-1)+torch.arange(len(REGIONS),device=self.device)[None]*s.region_samples
        return idx

    def residual(self,q,R,t,seed,contacts,jac=True,geometry=None):
        s=self.settings
        if geometry is None:
            state=self.skin.state(q,R,t);idx=self.witnesses(state,contacts)
            cn,cgap=self.objects.clearance(state['points'])
            state['collision_normals'],state['collision_gap']=cn,cgap
        else:
            state,idx=geometry;cn,cgap=state['collision_normals'],state['collision_gap']
        mask=contacts['active'].float()
        triangles=state['tri'].gather(1,idx[...,None,None].expand(-1,-1,3,3))
        point,bary=triangle_witness(triangles,contacts['anchor'])
        normal=state['normals'].gather(1,idx[...,None].expand(-1,-1,3))
        clearance=torch.full_like(cgap,-.0005)
        clearance[:,:self.skin.contact_count]=torch.where(
            contacts['active'][:,:,None].expand(-1,-1,s.region_samples).reshape(len(q),-1),-.0005,.001)
        # At most 48 deepest collision witnesses per row enter LM.
        count=min(48,cgap.shape[1]);depth,ci=(clearance-cgap).topk(count,dim=1)
        cids=ci;collision=depth.clamp_min(0)
        normals=cn.gather(1,ci[...,None].expand(-1,-1,3))
        qp=(q-seed[0])*.0008;tp=(t-seed[2])*.03
        rp=((R-seed[1]).flatten(1))*.00025
        residual=[((point-contacts['anchor'])*mask[...,None]).flatten(1),
                  ((normal+contacts['normal'])*mask[...,None]*s.normal_length_m).flatten(1),
                  collision*2,qp,tp,rp]
        result=torch.cat(residual,1)
        if not jac:return result,state,idx,cgap
        local=(self.skin.tri_local[idx]*bary[...,None,None]).sum(2)
        weights=(self.skin.tri_weights[idx]*bary[...,None]).sum(2)
        jp=self.skin.point_jacobian(state,local,weights);jn=self.skin.jacobian(state,idx,normal=True)
        jc=self.skin.jacobian(state,cids)
        jc=-(normals[...,None]*jc).sum(-2)*(collision>0)[...,None]*2
        n=len(q);eye=torch.eye(31,device=q.device).expand(n,31,31)
        jq=eye[:,:25]*.0008;jt=eye[:,28:31]*.003
        # Differential of a left-multiplied rotation matrix.
        axes=torch.eye(3,device=q.device)
        jr=torch.stack([(cross_matrix(a.expand(n,3))@R).flatten(1) for a in axes],-1)*.00025
        jr=torch.nn.functional.pad(jr,(25,3))
        J=torch.cat(((jp*mask[...,None,None]).flatten(1,2),
                     (jn*mask[...,None,None]*s.normal_length_m).flatten(1,2),jc,jq,jt,jr),1)
        return result,J,state,idx,cgap

    @torch.no_grad()
    def solve(self,grid,ids):
        original=np.asarray(ids)
        expanded=np.repeat(original,self.settings.starts)
        result=self._solve(grid,expanded)
        n=len(original);k=self.settings.starts
        active=result['contact_active'];error=np.where(active,result['anchor_error_m'],0).max(1)
        normal=np.where(active,result['normal_dot'],1).min(1)
        score=(error/self.settings.max_anchor_error_m+result['sampled_penetration_m']/self.settings.max_penetration_m
               +np.maximum(.5-normal,0)*4+result['unexpected_contact']*2
               +np.maximum(.5-np.where(active,result['material_side_dot'],1).min(1),0)*4)
        score-=result['contact_correspondence_valid']*1000
        best=score.reshape(n,k).argmin(1)+np.arange(n)*k
        return {key:value[best] for key,value in result.items()}

    def _solve(self,grid,ids):
        self.objects.batch(grid,ids);contacts=self.contacts(grid,ids)
        def t(x):return torch.as_tensor(x,device=self.device,dtype=torch.float32)
        q=t(grid['moxian_qpos'][ids,6:]).clamp(self.skin.limits[:,0],self.skin.limits[:,1])
        R=t(grid['wrist_rotation'][ids].transpose(0,2,1).copy());base=t(grid['wrist_translation'][ids])
        seed=(q.clone(),R.clone(),base.clone());n=len(q)
        initial_state=self.skin.state(*seed)
        initial_idx=self.witnesses(initial_state,contacts)
        initial_tri=initial_state['tri'].gather(1,initial_idx[...,None,None].expand(-1,-1,3,3))
        initial=(triangle_witness(initial_tri,contacts['anchor'])[0]-contacts['anchor']).norm(dim=-1)
        variants=torch.arange(n,device=self.device)%self.settings.starts
        alternative=q.clone()
        for j,name in enumerate(self.skin.names):
            if name.endswith('mcp_flexion'):alternative[:,j]=.55
            elif name.endswith('pip_flexion'):alternative[:,j]=.75
            elif name.endswith('dip_flexion'):alternative[:,j]=.4
            elif name.endswith('mcp_abduction'):alternative[:,j]=0.
        q=torch.where((variants==1)[:,None],alternative,q)
        index=t(np.asarray(ids,dtype=np.float32))[:,None];channels=torch.arange(25,device=self.device)[None]
        jitter=(torch.sin(index*12.9898+channels*78.233)*43758.5453).frac()*.35
        q=torch.where((variants>=2)[:,None],q+jitter,q).clamp(self.skin.limits[:,0],self.skin.limits[:,1])
        residual,J,state,idx,gap=self.residual(q,R,base,seed,contacts)
        cost=residual.square().sum(-1);damping=q.new_full((n,),self.settings.damping)
        eye=torch.eye(31,device=q.device)
        for _ in range(self.settings.iterations):
            H=J.transpose(-1,-2)@J;g=(J.transpose(-1,-2)@residual[...,None])[...,0]
            diagonal=H.diagonal(dim1=-2,dim2=-1).clamp_min(1e-7)
            step,info=torch.linalg.solve_ex(H+torch.diag_embed(damping[:,None]*diagonal)+eye*1e-9,-g)
            step=torch.nan_to_num(step).clamp(-.25,.25)
            cq=(q+step[:,:25]).clamp(self.skin.limits[:,0],self.skin.limits[:,1])
            cR=turn_matrix(step[:,25:28])@R;ct=base+.1*step[:,28:31]
            cr,cstate,cidx,_=self.residual(cq,cR,ct,seed,contacts,jac=False);cc=cr.square().sum(-1)
            accept=(cc<cost)&(info==0)&torch.isfinite(cc)
            q=torch.where(accept[:,None],cq,q);R=torch.where(accept[:,None,None],cR,R);base=torch.where(accept[:,None],ct,base)
            damping=torch.where(accept,damping*.65,damping*3).clamp(1e-6,1e5)
            state={key:torch.where(accept.reshape(n,*([1]*(value.ndim-1))),cstate[key],value) for key,value in state.items()}
            idx=torch.where(accept[:,None],cidx,idx)
            residual,J,state,idx,gap=self.residual(q,R,base,seed,contacts,geometry=(state,idx))
            cost=residual.square().sum(-1)
        triangles=state['tri'].gather(1,idx[...,None,None].expand(-1,-1,3,3))
        point,bary=triangle_witness(triangles,contacts['anchor'])
        # Independently audit the entire skin and actual deformed triangle normals.
        vertices=self.skin.full_vertices(state)
        face_ids=torch.as_tensor(self.skin.contact_faces,device=q.device)[idx]
        tri=vertices.gather(1,face_ids.flatten(1)[...,None].expand(-1,-1,3)).reshape(n,len(REGIONS),3,3)
        normal=torch.nn.functional.normalize(torch.linalg.cross(tri[:,:,1]-tri[:,:,0],tri[:,:,2]-tri[:,:,0]),dim=-1)
        np_faces=torch.as_tensor(self.skin.raw['faces'],device=q.device,dtype=torch.long)
        centers=vertices[:,np_faces].mean(2)
        penetration,query_valid=self.objects.penetration(torch.cat((vertices,centers),1))
        error=(point-contacts['anchor']).norm(dim=-1);ndot=-(normal*contacts['normal']).sum(-1)
        active=contacts['active'];count=active.sum(-1)
        maximum=error.masked_fill(~active,0).max(-1).values
        regional_gap=gap[:,:self.skin.contact_count].reshape(n,len(REGIONS),-1).min(-1).values
        unexpected=((regional_gap<.0005)&~active).any(-1)
        material_dot=(self.skin.material_normal.flatten(0,1)[idx]*contacts['material']).sum(-1)
        accepted=(count>0)&(maximum<=self.settings.max_anchor_error_m)&(penetration<=self.settings.max_penetration_m)&(ndot.masked_fill(~active,1).min(-1).values>=self.settings.min_normal_dot)&~unexpected&(material_dot.masked_fill(~active,1).min(-1).values>=.5)&query_valid
        def cpu(a):return a.cpu().numpy()
        output=dict(sample_indices=np.asarray(ids),target_qpos=cpu(q),target_base_position=cpu(base),target_base_rotation=cpu(R),
             contact_active=cpu(active),object_contact_points=cpu(contacts['anchor']),object_contact_normals=cpu(contacts['normal']),
             wuji_contact_points=cpu(contacts['source_hand']),source_contact_gap_m=cpu(contacts['source_gap']),
             glove_contact_points=cpu(point),glove_contact_normals=cpu(normal),anchor_error_m=cpu(error),
             initial_anchor_error_m=cpu(initial),normal_dot=cpu(ndot),sampled_penetration_m=cpu(penetration),
             contact_correspondence_valid=cpu(accepted),source_contact_count=cpu(count),solver_cost=cpu(cost),
             selected_material_sample=cpu(idx),glove_contact_faces=cpu(face_ids),
             unexpected_contact=cpu(unexpected),glove_region_min_gap_m=cpu(regional_gap),material_side_dot=cpu(material_dot),
             glove_contact_barycentric=cpu(bary),penetration_query_valid=cpu(query_valid))
        return output
