"""Batched native FK, signed triangle witnesses and bounded contact LM."""
import math
import numpy as np
import torch
import warp as wp
from retarget.env.grasp.pose_gpu import ModelFK
from retarget.baseline.card.collision import WujiMeshCollision


@wp.kernel(enable_backward=False)
def _surface(meshes:wp.array[wp.uint64], count:int, points:wp.array2d[wp.vec3],
             hitp:wp.array2d[wp.vec3], normal:wp.array2d[wp.vec3], gap:wp.array2d[float]):
    b,k=wp.tid();p=points[b,k];mesh=meshes[b%count]
    h=wp.mesh_query_point_sign_winding_number(mesh,p,1.e3)
    if h.result and wp.isfinite(p):
        x=wp.mesh_eval_position(mesh,h.face,h.u,h.v)
        a=wp.mesh_get_point(mesh,3*h.face);c=wp.mesh_get_point(mesh,3*h.face+1);d=wp.mesh_get_point(mesh,3*h.face+2)
        hitp[b,k]=x;normal[b,k]=wp.normalize(wp.cross(c-a,d-a));gap[b,k]=h.sign*wp.length(p-x)
    else:
        hitp[b,k]=p;normal[b,k]=wp.vec3(1.,0.,0.);gap[b,k]=1.e6


@wp.kernel(enable_backward=False)
def _self(meshes:wp.array[wp.uint64], points:wp.array[wp.vec3], owners:wp.array[int],
          allowed:wp.array2d[int],low:wp.array[wp.vec3],high:wp.array[wp.vec3],collar:wp.array[float],
          p:wp.array2d[wp.vec3],r:wp.array2d[wp.mat33],
          gap:wp.array2d[float],normal:wp.array2d[wp.vec3],targets:wp.array2d[int],hitp:wp.array2d[wp.vec3]):
    b,k=wp.tid();source=owners[k];x=p[b,source]+r[b,source]*points[k]
    best=float(0.);tn=wp.vec3(0.);ti=int(source);tx=x
    for target in range(21):
        if allowed[source,target]==0:continue
        local=wp.transpose(r[b,target])*(x-p[b,target])
        if source==0 and collar[target]>0. and wp.length(local)<=collar[target]:continue
        if target==0 and collar[source]>0. and wp.length(points[k])<=collar[source]:continue
        lo=low[target];hi=high[target]
        if local[0]<lo[0] or local[1]<lo[1] or local[2]<lo[2] or local[0]>hi[0] or local[1]>hi[1] or local[2]>hi[2]:continue
        h=wp.mesh_query_point_sign_winding_number(meshes[target],local,1.)
        if h.result and h.sign<0.:
            z=wp.mesh_eval_position(meshes[target],h.face,h.u,h.v)
            depth=wp.length(local-z)
            if -depth<best:
                best=-depth;ti=target;tx=p[b,target]+r[b,target]*z
                tn=r[b,target]*wp.normalize(z-local)
    gap[b,k]=best;normal[b,k]=tn;targets[b,k]=ti;hitp[b,k]=tx


@wp.kernel(enable_backward=False)
def _reverse_depth(meshes:wp.array[wp.uint64],points:wp.array2d[wp.vec3],count:int,
                   low:wp.array[wp.vec3],high:wp.array[wp.vec3],
                   p:wp.array2d[wp.vec3],r:wp.array2d[wp.mat33],depth:wp.array[float],invalid:wp.array[int]):
    b,k,target=wp.tid();x=wp.transpose(r[b,target])*(points[b%count,k]-p[b,target])
    lo=low[target];hi=high[target]
    if x[0]<lo[0] or x[1]<lo[1] or x[2]<lo[2] or x[0]>hi[0] or x[1]>hi[1] or x[2]>hi[2]:return
    h=wp.mesh_query_point_sign_winding_number(meshes[target],x,1.)
    if not h.result:wp.atomic_add(invalid,b,1)
    elif h.sign<0.:
        hit=wp.mesh_eval_position(meshes[target],h.face,h.u,h.v)
        wp.atomic_max(depth,b,wp.length(x-hit))


class ObjectQuery:
    def __init__(self, mesh, device):
        self.device=str(device);self.meshes=mesh if isinstance(mesh,list) else [mesh];self.mesh=self.meshes[0]
        self.warps=[wp.Mesh(points=wp.array(np.asarray(m.vertices,np.float32),dtype=wp.vec3,device=self.device),
                          indices=wp.array(np.asarray(m.faces,np.int32).ravel(),dtype=int,device=self.device),
                          support_winding_number=True) for m in self.meshes]
        self.ids=wp.array([m.id for m in self.warps],dtype=wp.uint64,device=self.device)

    def query(self, points):
        points=points.contiguous();hit=torch.empty_like(points);normal=torch.empty_like(points)
        if len(self.meshes) not in (1,len(points)):raise ValueError('Object batch and pose batch must align')
        gap=torch.empty(points.shape[:2],device=points.device,dtype=torch.float32)
        stream=wp.stream_from_torch(torch.cuda.current_stream(points.device)) if points.is_cuda else None
        wp.launch(_surface,points.shape[:2],inputs=[self.ids,len(self.meshes),wp.from_torch(points,dtype=wp.vec3),
            wp.from_torch(hit,dtype=wp.vec3),wp.from_torch(normal,dtype=wp.vec3),wp.from_torch(gap)],device=self.device,stream=stream)
        return hit,normal,gap

    def contains(self, points):
        x=torch.as_tensor(points,dtype=torch.float32,device=self.device)[None]
        return (self.query(x)[2][0]<0).cpu().numpy()


class Solver:
    def __init__(self, templates, device='cuda:0', iterations=50):
        self.templates=templates;self.surface=templates.surface;self.robot=templates.robot
        self.device=torch.device(device);self.iterations=iterations
        self.fk=ModelFK(self.robot.model,self.device,torch.float32)
        self.collision=WujiMeshCollision(self.surface,self.device)
        self.body_ids=self.surface.body_ids
        model=self.robot.model
        anc=np.zeros((model.nbody,20),np.float32)
        self.joint_body=[]
        for j in range(6,26):
            b=int(model.jnt_bodyid[j]);self.joint_body.append(b)
            for child in range(model.nbody):
                cur=child
                while cur>0:
                    if cur==b:anc[child,j-6]=1;break
                    cur=int(model.body_parentid[cur])
        self.ancestor=self.tensor(anc)
        self.joint_axis=self.tensor(model.jnt_axis[6:])
        self.joint_pivot=self.tensor(model.jnt_pos[6:])
        self.low=self.tensor(templates.lower);self.high=self.tensor(templates.upper)
        local=[];owners=[];dense=[];dense_owners=[]
        rng=np.random.default_rng(1927)
        for owner,mesh in enumerate(self.surface.meshes):
            ids=rng.choice(len(mesh.faces),48,p=mesh.area_faces/mesh.area)
            u=rng.random((len(ids),2));f=u.sum(1)>1;u[f]=1-u[f]
            v=(mesh.triangles[ids]*np.c_[1-u.sum(1),u][...,None]).sum(1)
            local.extend(v);owners.extend([owner]*len(v))
            d=np.vstack((mesh.vertices,mesh.triangles_center))
            dense.extend(d);dense_owners.extend([owner]*len(d))
        self.coarse=self.tensor(local);self.coarse_owners=np.array(owners)
        self.dense=self.tensor(dense);self.dense_owners=np.array(dense_owners)
        self.owners_wp=wp.array(self.coarse_owners.astype(np.int32),dtype=int,device=str(self.device))
        self.coarse_wp=wp.from_torch(self.coarse,dtype=wp.vec3)

    def tensor(self,value):
        return torch.as_tensor(np.asarray(value),device=self.device,dtype=torch.float32)

    def frames(self,q):
        p,r=self.fk.frames(q)
        axes=torch.einsum('bkij,kj->bki',r[:,self.joint_body],self.joint_axis)
        pivots=p[:,self.joint_body]+torch.einsum('bkij,kj->bki',r[:,self.joint_body],self.joint_pivot)
        return p,r,axes,pivots

    def points(self,p,r,local,owners):
        bodies=np.asarray(self.body_ids)[owners]
        return p[:,bodies]+torch.einsum('bkij,kj->bki',r[:,bodies],local)

    def jac(self,q,points,owners,frames,normal=False):
        p,r,axes,pivots=frames;b,k,_=points.shape
        if isinstance(owners,torch.Tensor):
            bodies=torch.as_tensor(self.body_ids,device=self.device)[owners]
            ancestor=self.ancestor[bodies]
        else:
            bodies=np.asarray(self.body_ids)[owners];ancestor=self.ancestor[bodies][None]
        delta=points[:,:,None,:] if normal else points[:,:,None,:]-pivots[:,None]
        finger=torch.linalg.cross(axes[:,None].expand(b,k,20,3),delta.expand(b,k,20,3),dim=-1)*ancestor[...,None]
        x,y=q[:,3],q[:,4]
        zeros=torch.zeros_like(x);ones=torch.ones_like(x)
        rootaxes=torch.stack((torch.stack((ones,zeros,zeros),-1),
                              torch.stack((zeros,x.cos(),x.sin()),-1),
                              torch.stack((y.sin(),-x.sin()*y.cos(),x.cos()*y.cos()),-1)),1)
        delta=points if normal else points-q[:,:3,None].transpose(1,2)
        rotation=torch.linalg.cross(rootaxes[:,None].expand(b,k,3,3),delta[:,:,None].expand(b,k,3,3),dim=-1)
        translation=torch.zeros((b,k,3,3),device=self.device) if normal else torch.eye(3,device=self.device).expand(b,k,3,3)
        return torch.cat((translation,rotation.transpose(-1,-2),finger.transpose(-1,-2)),-1)

    def self_query(self,frames):
        p,r,_,_=frames;b=len(p);k=len(self.coarse)
        pp=p[:,self.body_ids].contiguous();rr=r[:,self.body_ids].contiguous()
        gap=torch.zeros((b,k),device=self.device);normal=torch.empty((b,k,3),device=self.device)
        target=torch.empty((b,k),device=self.device,dtype=torch.int32);hit=torch.empty_like(normal)
        stream=wp.stream_from_torch(torch.cuda.current_stream(self.device)) if p.is_cuda else None
        c=self.collision
        wp.launch(_self,(b,k),inputs=[c.mesh_ids,self.coarse_wp,self.owners_wp,c.allowed,c.lower,c.upper,c.collar,
            wp.from_torch(pp,dtype=wp.vec3),wp.from_torch(rr,dtype=wp.mat33),wp.from_torch(gap),
            wp.from_torch(normal,dtype=wp.vec3),wp.from_torch(target),wp.from_torch(hit,dtype=wp.vec3)],device=str(self.device),stream=stream)
        return gap,normal,target.long(),hit

    def witnesses(self,q,template,obj,normal_length=.002):
        frames=self.frames(q);p,r,_,_=frames
        owners=np.repeat(template['owners'],self.templates.samples)
        local=self.tensor(np.concatenate([a['points'] for a in template['patches']]))
        nn=self.tensor(np.concatenate([a['normals'] for a in template['patches']]))
        world=self.points(p,r,local,owners)
        normals=torch.einsum('bkij,kj->bki',r[:,np.array(self.body_ids)[owners]],nn)
        hit,on,gap=obj.query(world)
        b=len(q);n=len(template['patches']);s=self.templates.samples
        dot=(normals*on).sum(-1)
        cost=(gap+.00002).square()+(normal_length*(1+dot)).square()
        chosen=cost.reshape(b,n,s).argmin(-1)
        ids=chosen+torch.arange(n,device=self.device)[None]*s
        batch=torch.arange(b,device=self.device)[:,None]
        return dict(frames=frames,world=world[batch,ids],normal=normals[batch,ids],
                    hit=hit[batch,ids],object_normal=on[batch,ids],gap=gap[batch,ids],
                    local_ids=chosen,owners=template['owners'])

    def system(self,q,seed,template,obj,slide=None):
        w=self.witnesses(q,template,obj,normal_length=.01 if slide is not None else .002);frames=w['frames'];p,r,_,_=frames
        n=len(template['patches']);needed=math.ceil(.8*n)
        costs=w['gap'].square()
        if slide is not None:costs=costs+(.01*(1+(w['normal']*w['object_normal']).sum(-1))).square()
        select=costs.argsort(-1)[:,:needed]
        mask=torch.zeros_like(costs).scatter_(1,select,1)
        j=self.jac(q,w['world'],template['owners'],frames)
        delta=w['world']-w['hit']+.00002*w['object_normal']
        contact_scale=.0005 if slide is not None else .003
        rows=[(delta*mask[...,None]/contact_scale).flatten(1)]
        jac=[(j*mask[...,None,None]/contact_scale).flatten(1,2)]
        nj=self.jac(q,w['normal'],template['owners'],frames,normal=True)
        nw=2. if slide is not None else .12
        rows.append(((w['normal']+w['object_normal'])*mask[...,None]*nw).flatten(1))
        jac.append((nj*mask[...,None,None]*nw).flatten(1,2))
        if slide is not None:
            rows.append(((w['world']-slide)*mask[...,None]/.004).flatten(1))
            jac.append((j*mask[...,None,None]/.004).flatten(1,2))
        # Coarse clearance guides optimization. Independent dense checks decide
        # acceptance, and never reuse this optimization sampling as a certificate.
        points=self.points(p,r,self.coarse,self.coarse_owners)
        hit,on,gap=obj.query(points)
        ids=gap.argsort(-1)[:,:48];batch=torch.arange(len(q),device=self.device)[:,None]
        chosen=points[batch,ids];normal=on[batch,ids];d=gap[batch,ids]
        owners=torch.as_tensor(self.coarse_owners,device=self.device)[ids]
        pj=self.jac(q,chosen,owners,frames)
        active=(d<-.0003).float()
        rows.append((d+.0003)*active/.002)
        jac.append((normal[...,None]*pj).sum(-2)*active[...,None]/.002)
        sg,sn,st,sh=self.self_query(frames)
        ids=sg.argsort(-1)[:,:32];d=sg[batch,ids];normal=sn[batch,ids]
        jp=self.jac(q,points[batch,ids],torch.as_tensor(self.coarse_owners,device=self.device)[ids],frames)
        jt=self.jac(q,sh[batch,ids],st[batch,ids],frames)
        active=(d<-.0005).float()
        rows.append((d+.0005)*active/.002)
        jac.append((normal[...,None]*(jp-jt)).sum(-2)*active[...,None]/.002)
        scale=self.tensor([.03]*3+[.5]*3+[1.]*20)
        rows.append(.025*(q-seed)/scale)
        jac.append((.025*torch.diag(1/scale))[None].expand(len(q),-1,-1))
        return torch.cat(rows,1),torch.cat(jac,1)

    @torch.no_grad()
    def solve(self, seeds, template, obj, slide=None, iterations=None):
        seed=self.tensor(seeds);q=seed.clone();q[:,6:]=q[:,6:].clamp(self.low,self.high)
        slide=None if slide is None else self.tensor(slide)
        damping=q.new_full((len(q),),.01);eye=torch.eye(26,device=self.device)
        residual,jac=self.system(q,seed,template,obj,slide)
        cost=residual.square().sum(-1)
        for _ in range(iterations or self.iterations):
            h=jac.transpose(1,2)@jac;g=(jac.transpose(1,2)@residual[...,None])[...,0]
            diagonal=h.diagonal(dim1=-2,dim2=-1).clamp_min(.01)
            step,info=torch.linalg.solve_ex(h+torch.diag_embed(damping[:,None]*diagonal)+1e-5*eye,-g)
            limit=q.new_tensor([.005]*3+[.15]*23)
            step=torch.maximum(torch.minimum(step,limit),-limit)
            candidate=q+step;candidate[:,6:]=candidate[:,6:].clamp(self.low,self.high)
            rr,jj=self.system(candidate,seed,template,obj,slide);cc=rr.square().sum(-1)
            accept=(info==0)&torch.isfinite(cc)&(cc<cost)
            q=torch.where(accept[:,None],candidate,q);cost=torch.where(accept,cc,cost)
            residual=torch.where(accept[:,None],rr,residual);jac=torch.where(accept[:,None,None],jj,jac)
            damping=torch.where(accept,damping*.6,damping*3).clamp(1e-6,1e6)
        return q.cpu().numpy()

    @torch.no_grad()
    def audit(self,poses,template,obj,sliding=False):
        q=self.tensor(poses);w=self.witnesses(q,template,obj)
        # Contact selection for audit is existential over ALL immutable patch
        # witnesses, not the optimizer's normal-weighted preferred point.
        p,r,_,_=w['frames'];n=len(template['patches']);s=self.templates.samples
        owners=np.repeat(template['owners'],s)
        local=self.tensor(np.concatenate([a['points'] for a in template['patches']]))
        nn=self.tensor(np.concatenate([a['normals'] for a in template['patches']]))
        points=self.points(p,r,local,owners);hit,on,gap=obj.query(points)
        normals=torch.einsum('bkij,kj->bki',r[:,np.array(self.body_ids)[owners]],nn)
        dot=(normals*on).sum(-1)
        good=(gap>-.005)&(gap<=0)&torch.isfinite(gap)
        if sliding:good &= (gap>=-.0001)&(dot<=-math.cos(math.pi/6))
        cost=torch.where(good,gap.abs(),1e3+gap.abs()).reshape(len(q),n,s)
        ids=cost.argmin(-1);flat=ids+torch.arange(n,device=self.device)[None]*s
        batch=torch.arange(len(q),device=self.device)[:,None]
        patch_hit=good.reshape(len(q),n,s).any(-1)
        dense=self.points(p,r,self.dense,self.dense_owners)
        object_depth=torch.zeros(len(q),device=self.device);object_valid=torch.ones(len(q),device=self.device,dtype=torch.bool)
        for start in range(0,dense.shape[1],4096):
            d=obj.query(dense[:,start:start+4096])[2]
            object_depth=torch.maximum(object_depth,(-d).clamp_min(0).amax(1))
            object_valid &= torch.isfinite(d).all(1)&(d.abs()<999).all(1)
        # Reverse direction detects objects buried in a hand component even if
        # no hand vertex lies inside the small object.
        raw_points=[np.vstack((m.vertices,m.triangles_center)) for m in obj.meshes]
        count=max(len(v) for v in raw_points)
        reverse_points=self.tensor(np.stack([np.resize(v,(count,3)) for v in raw_points]))
        reverse=torch.zeros(len(q),device=self.device);invalid=torch.zeros(len(q),device=self.device,dtype=torch.int32)
        pp=p[:,self.body_ids].contiguous();rr=r[:,self.body_ids].contiguous();c=self.collision
        stream=wp.stream_from_torch(torch.cuda.current_stream(self.device)) if q.is_cuda else None
        wp.launch(_reverse_depth,(len(q),count,21),inputs=[c.mesh_ids,
            wp.from_torch(reverse_points,dtype=wp.vec3),len(obj.meshes),c.lower,c.upper,wp.from_torch(pp,dtype=wp.vec3),
            wp.from_torch(rr,dtype=wp.mat33),wp.from_torch(reverse),wp.from_torch(invalid)],device=str(self.device),stream=stream)
        object_depth=torch.maximum(object_depth,reverse);object_valid &= invalid==0
        sd,sv=self.collision.query(p,r,dense=True);self_depth=sd.amax((1,2))*.001
        domain=torch.isfinite(q).all(1)&(q[:,6:]>=self.low).all(1)&(q[:,6:]<=self.high).all(1)
        query_valid=object_valid&sv&torch.isfinite(gap).all(1)&(gap.abs()<999).all(1)
        accepted=domain&query_valid&(object_depth<=.005)&(self_depth<=.005)&(patch_hit.sum(1)>=math.ceil(.8*n))
        def cpu(x):return x.cpu().numpy()
        return dict(accepted=cpu(accepted),patch_hit=cpu(patch_hit),witness_ids=cpu(ids),
                    contact_local=cpu(local[flat]),contact_object_local=cpu(points[batch,flat]),
                    contact_normal_object_local=cpu(normals[batch,flat]),
                    contact_gap_m=cpu(gap[batch,flat]),normal_dot=cpu(dot[batch,flat]),
                    object_penetration_m=cpu(object_depth),self_penetration_m=cpu(self_depth),
                    query_valid=cpu(query_valid),domain_valid=cpu(domain))
