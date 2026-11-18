"""Wuji two-stage full-hand costs plus differentiable observed-surface contact."""
from dataclasses import dataclass, asdict
import numpy as np
import torch
from torch.nn import functional as F
import warp as wp
from retarget.baseline.geort.adapter import RobotAdapter
from retarget.baseline.geort.pose_domain import extension_domain
from retarget.env.grasp.pose_gpu import ModelFK
from retarget.baseline.card.collision import WujiMeshCollision
from .wuji_surface import WujiSurface
from .cart_contacts import link_owner
from .cart_losses import _signed_witness
from .cart_contact_witness import contact_witness, DEEP_PENETRATION_M


@dataclass(frozen=True)
class ObjectiveSettings:
    contact: float = 20.
    normal: float = 5.
    penetration: float = 100.
    trajectory: float = 1.
    contact_keep: float = 1.
    acceleration: float = .001
    contact_scale_m: float = .001
    penetration_scale_m: float = .0005
    normal_deg: float = 30.
    contact_limit_m: float = .0005
    penetration_limit_m: float = .0005
    lambda_warm: float = 1.
    lambda_pose: float = .1
    lambda_lap: float = 150.
    lambda_smooth: float = .01
    lambda_thumb_rel: float = 5.
    lambda_antirev: float = 20.

    def payload(self): return asdict(self)


@wp.kernel(enable_backward=False)
def _nearest(mesh: wp.uint64, query: wp.array(dtype=wp.vec3),
             hits: wp.array(dtype=wp.vec3), normals: wp.array(dtype=wp.vec3), valid: wp.array(dtype=int)):
    i = wp.tid(); q = wp.mesh_query_point_no_sign(mesh, query[i], 2.)
    if q.result:
        hits[i] = wp.mesh_eval_position(mesh,q.face,q.u,q.v)
        normals[i] = wp.mesh_eval_face_normal(mesh,q.face); valid[i] = 1


@wp.kernel(enable_backward=False)
def _nearest_contact(skin: wp.uint64, solid: wp.uint64, query: wp.array(dtype=wp.vec3),
                     escape: wp.array(dtype=wp.vec3), threshold: float,
                     hits: wp.array(dtype=wp.vec3), normals: wp.array(dtype=wp.vec3),
                     valid: wp.array(dtype=int), retracted: wp.array(dtype=int)):
    i=wp.tid()
    witness=contact_witness(skin,solid,query[i],escape[i],threshold)
    hits[i]=witness.point;normals[i]=witness.normal
    valid[i]=witness.valid;retracted[i]=witness.retracted


class CartHand:
    def __init__(self, device='cpu'):
        self.device = str(device); self.robot = RobotAdapter(); self.surface = WujiSurface()
        self.domain = extension_domain(self.robot.lower, self.robot.upper)
        self.low = self.tensor(self.domain['lower']); self.high = self.tensor(self.domain['upper'])
        self.fk = ModelFK(self.robot.model, self.device); self.ids = self.robot.body_ids
        from .cart_kinematics import FixedWristFK
        self.fixed_fk=FixedWristFK(self.fk,self.ids)
        self.basis = self.tensor(self.robot.basis)
        self.collision = WujiMeshCollision(self.surface, self.device)
        self.full_meshes = self.collision.meshes; self.pad_meshes = {}; self.samples_cache = {}
        self.tip_owners = [self.ids.index(int(self.robot.model.site_bodyid[i])) for i in self.robot.tip_ids]
        self.tip_local = self.tensor(self.robot.model.site_pos[self.robot.tip_ids])
        self.centres=self.tensor([mesh.bounds.mean(0) for mesh in self.surface.meshes])

    def tensor(self, value): return torch.as_tensor(np.asarray(value), device=self.device, dtype=torch.float32)

    def frames(self, q):
        full_p, full_r = self.fixed_fk.frames(q)
        p, r = full_p[:,self.ids], full_r[:,self.ids]
        cp = (p-p[:,:1])@self.basis; cr = self.basis.T@r
        tips = cp[:,self.tip_owners]+(cr[:,self.tip_owners]@self.tip_local[...,None]).squeeze(-1)
        landmarks = [cp[:,0]]
        for f in range(5):
            landmarks.extend(cp[:,o] for o in (2+4*f,3+4*f,4+4*f)); landmarks.append(tips[:,f])
        return cp,cr,torch.stack(landmarks,1),full_p,full_r

    def nearest(self, query, owner, palmar=False, *, escape=None, return_retracted=False):
        if palmar and owner not in self.pad_meshes:
            tri = np.asarray(self.surface.contact_surface(owner)['triangles'], np.float32)
            if not len(tri): raise ValueError('Wuji2 缺少接触曲面')
            self.pad_meshes[owner] = wp.Mesh(points=wp.array(tri.reshape(-1,3),dtype=wp.vec3,device=self.device),
                indices=wp.array(np.arange(tri.size//3),dtype=int,device=self.device))
        mesh = self.pad_meshes[owner] if palmar else self.full_meshes[owner]
        shape = query.shape; flat = query.detach().reshape(-1,3).contiguous()
        hits = wp.zeros(len(flat),dtype=wp.vec3,device=self.device)
        normals = wp.zeros(len(flat),dtype=wp.vec3,device=self.device); valid = wp.zeros(len(flat),dtype=int,device=self.device)
        retracted=wp.zeros(len(flat),dtype=int,device=self.device)
        if escape is None:
            wp.launch(_nearest,len(flat),inputs=[mesh.id,wp.from_torch(flat,dtype=wp.vec3),hits,normals,valid],device=self.device)
        else:
            directions=escape.detach().reshape(-1,3).contiguous()
            wp.launch(_nearest_contact,len(flat),inputs=[mesh.id,self.full_meshes[owner].id,
                wp.from_torch(flat,dtype=wp.vec3),wp.from_torch(directions,dtype=wp.vec3),DEEP_PENETRATION_M,
                hits,normals,valid,retracted],device=self.device)
        values=(wp.to_torch(hits).reshape(shape),wp.to_torch(normals).reshape(shape),wp.to_torch(valid).bool().reshape(shape[:-1]))
        return values+(wp.to_torch(retracted).bool().reshape(shape[:-1]),) if return_retracted else values

    def samples(self, owner, dense):
        key = (owner,dense)
        if key not in self.samples_cache:
            mesh = self.surface.meshes[owner]
            p = np.vstack((mesh.vertices,mesh.triangles_center)); n = np.vstack((mesh.vertex_normals,mesh.face_normals))
            if not dense:
                ids = np.unique(np.r_[np.linspace(0,len(p)-1,64).astype(int),np.argmin(p,0),np.argmax(p,0)])
                p,n = p[ids],n[ids]
            self.samples_cache[key] = (self.tensor(p),self.tensor(n))
        return self.samples_cache[key]

    def pair(self,p,r,a,b,dense=False):
        choices = []
        for source,target in ((a,b),(b,a)):
            local,ln = self.samples(source,dense)
            points = (r[:,source,None]@local[None,:,:,None]).squeeze(-1)+p[:,source,None]
            normals = (r[:,source,None]@ln[None,:,:,None]).squeeze(-1)
            query = ((points-p[:,target,None])[...,None,:]@r[:,target,None]).squeeze(-2)
            escape=(-(normals[...,None,:]@r[:,target,None])).squeeze(-2)
            hit,hn,valid,retracted = self.nearest(query,target,escape=escape,return_retracted=True)
            centres=[]
            for owner in (source,target):
                centre=self.centres[owner] if hasattr(self,'centres') else self.tensor(self.surface.meshes[owner].bounds.mean(0))
                centres.append((r[:,owner]@centre[:,None]).squeeze(-1)+p[:,owner])
            facing=(normals*(centres[1]-centres[0])[:,None]).sum(-1)>1.e-7
            valid=valid&(~retracted|facing)
            other = (r[:,target,None]@hit[...,None]).squeeze(-1)+p[:,target,None]
            on = (r[:,target,None]@hn[...,None]).squeeze(-1)
            distance = (points-other).square().sum(-1)
            angle = (1+(normals*on).sum(-1).clamp(-1,1)).clamp_min(0)
            score = distance.detach()/.001**2 + angle.detach()/(1-np.cos(np.deg2rad(30)))
            score = torch.where(valid,score,torch.inf); index = score.argmin(-1); batch = torch.arange(len(p),device=p.device)
            one = tuple(v[batch,index] for v in (distance,angle,points,other,normals,on,valid))
            if source == b: one = (one[0],one[1],one[3],one[2],one[5],one[4],one[6])
            choices.append(one)
        use = choices[0][6] & (~choices[1][6] | (choices[0][0].detach() <= choices[1][0].detach()))
        return tuple(torch.where(use.reshape((-1,)+(1,)*(x.ndim-1)),x,y) for x,y in zip(*choices))

    def penetration(self,q,pfull,rfull,dense=False):
        if not q.requires_grad:
            depth,valid = self.collision.query(pfull,rfull,dense=dense)
            maximum = depth.amax((1,2))/1000
            return maximum,valid
        # Freeze nearest witnesses, then differentiate BOTH moving links.
        p,r = pfull[:,self.ids],rfull[:,self.ids]
        points,owners = self.collision.samples[dense]; source = wp.to_torch(points); owner = wp.to_torch(owners).long()
        shape = (len(q),len(source),21)
        hits = wp.zeros(shape,dtype=wp.vec3,device=self.device); normals = wp.zeros(shape,dtype=wp.vec3,device=self.device)
        valid = wp.zeros(shape,dtype=int,device=self.device)
        wp.launch(_signed_witness,shape,inputs=[self.collision.mesh_ids,points,owners,self.collision.allowed,self.collision.collar,
            wp.from_torch(p.detach().contiguous(),dtype=wp.vec3),wp.from_torch(r.detach().contiguous(),dtype=wp.mat33),hits,normals,valid],device=self.device)
        world = (r[:,owner]@source[...,None]).squeeze(-1)+p[:,owner]
        hit = (r[:,None]@wp.to_torch(hits)[...,None]).squeeze(-1)+p[:,None]
        normal = (r[:,None]@wp.to_torch(normals)[...,None]).squeeze(-1)
        depths = torch.where(wp.to_torch(valid).bool(),(-((world[:,:,None]-hit)*normal).sum(-1)).clamp_min(0),0.)
        return depths.amax((1,2)),torch.ones(len(q),device=q.device,dtype=torch.bool)


class CartObjective:
    def __init__(self,hand,settings=None,collision=None):
        self.hand=hand; self.settings=settings or ObjectiveSettings(); self.cache={};self.collision=collision
        self.vectorized=True;self.cache_rows=False;self.fast_rows={}

    def fast_constants(self,row):
        from .cart_fast import semantic_key
        key=id(row)
        if self.cache_rows and key in self.fast_rows:return self.fast_rows[key][1]
        _,_,_,edges,desired,_=self.targets(row);items=[];h=self.hand
        observed=row['contacts'];maximum=max((c['force'] for c in observed['object_contacts']),default=0.)
        for c in observed['self_pairs']:
            items.append(dict(key=semantic_key(c,True),owners=tuple(map(link_owner,c['regions'])),target=h.tensor(c['points_m']).mean(0)))
        for c in observed['object_contacts']:
            items.append(dict(key=semantic_key(c),owner=link_owner(c['region']),target=h.tensor(c['position_m']),normal=h.tensor(c['normal']),
                rho=1+.5*c['force']/maximum if maximum>0 else 1.))
        result=dict(edges=edges,edge_key=tuple(edges.detach().cpu().flatten().tolist()),desired=desired,contacts=items)
        if self.cache_rows:self.fast_rows[key]=(row,result)
        return result

    def targets(self,row):
        # Prepared constants are cached by content, never serialized into raw rows.
        from retarget.baseline.wuji_retargeting.reference import REFERENCE_ROOT  # noqa: F401
        from retarget_reference.preprocess import prepare_targets
        from retarget_reference.objectives import prepare_thumb_targets
        source = np.asarray(row['joints_m'],np.float32)
        key = source.tobytes()
        if key not in self.cache:
            target=prepare_targets(source,thumb_free=False)
            thumb=prepare_thumb_targets(source,[(1,2,3),(2,3,4)],self.settings.lambda_thumb_rel,.35,.8,1e-8)
            if len(self.cache)>=256:self.cache.pop(next(iter(self.cache)))
            self.cache[key]=(self.hand.tensor(source),self.hand.tensor(target['L']),self.hand.tensor(target['D']),
                torch.as_tensor(target['edges'],device=self.hand.device),self.hand.tensor(target['directions']),thumb)
        return self.cache[key]

    def __call__(self,q,row,reference=None,stage=2,dense=False,*,frames=None,penetration=None):
        s=self.settings; h=self.hand; p,r,j,pfull,rfull=h.frames(q) if frames is None else frames
        source,lap,differential,edges,desired,thumb=self.targets(row)
        dirs=F.normalize(j[:,edges[:,1]]-j[:,edges[:,0]],dim=-1,eps=1e-8)
        parts=dict(direction=.5*(s.lambda_warm if stage==1 else s.lambda_pose)*(dirs-desired).square().sum((1,2)))
        parts['laplacian']=.5*s.lambda_lap*(lap@j-differential).square().sum((1,2)) if stage==2 else q.new_zeros(len(q))
        parts['smooth']=.5*s.lambda_smooth*(q-reference).square().sum(-1) if reference is not None else q.new_zeros(len(q))
        parts['antireverse']=.5*s.lambda_antirev*(-q[:,[2,3,6,7,10,11,14,15,18,19]]).clamp_min(0).square().sum(-1)
        parts['thumb_relative']=q.new_zeros(len(q))
        for a,b,c,cosine,sine,normal,weight in thumb:
            u=F.normalize(j[:,b]-j[:,a],dim=-1,eps=1e-8); v=F.normalize(j[:,c]-j[:,b],dim=-1,eps=1e-8)
            parts['thumb_relative']=parts['thumb_relative']+.5*weight*((u*v).sum(-1)-cosine).square()
            parts['thumb_relative']=parts['thumb_relative']+.5*weight*((torch.linalg.cross(u,v)*h.tensor(normal)).sum(-1)-sine).square()
        direction_error=torch.rad2deg(torch.acos((dirs.detach()*desired).sum(-1).clamp(-1,1))).mean(-1)
        parts['contact']=q.new_zeros(len(q)); parts['normal']=q.new_zeros(len(q))
        groups={}; gaps=[]; angles=[]; contacts=[]; contact_valid=torch.ones(len(q),device=q.device,dtype=torch.bool)
        unresolved=torch.zeros(len(q),device=q.device,dtype=torch.long)
        observed=row['contacts']; force_max=max((c['force'] for c in observed['object_contacts']),default=0.)
        for pair in observed['self_pairs']:
            a,b=map(link_owner,pair['regions']); distance,angle,pa,pb,na,nb,witness_valid=h.pair(p,r,a,b,dense)
            target=h.tensor(pair['points_m']).mean(0)
            tracking=((pa+pb)*.5-target).square().sum(-1)
            pos=(distance+.02*tracking)/s.contact_scale_m**2
            norm=angle/(1-np.cos(np.deg2rad(s.normal_deg)))
            # Unresolved contacts must not pull towards a guessed back surface.
            pos=torch.where(witness_valid,pos,0.);norm=torch.where(witness_valid,norm,0.)
            contact_valid=contact_valid&witness_valid
            unresolved=unresolved+(~witness_valid).long()
            key=('self',*sorted(pair['regions'])); groups.setdefault(key,[]).append((pos,norm))
            gaps.append(torch.where(witness_valid,distance.detach().clamp_min(0).sqrt(),0.)); angles.append(torch.rad2deg(torch.acos((-na.detach()*nb.detach()).sum(-1).clamp(-1,1))))
            contacts.append((pa,pb))
        for c in observed['object_contacts']:
            owner=link_owner(c['region']); target=h.tensor(c['position_m']); normal=h.tensor(c['normal'])
            query=((target-p[:,owner])[:,None,:]@r[:,owner]).squeeze(1)
            escape=(normal[None,None,:]@r[:,owner]).squeeze(1)
            local,ln,valid=h.nearest(query,owner,palmar=True,escape=escape)
            pos=(r[:,owner]@local[...,None]).squeeze(-1)+p[:,owner]
            nn=(r[:,owner]@ln[...,None]).squeeze(-1)
            distance=(pos-target).square().sum(-1); dot=(nn*normal).sum(-1).clamp(-1,1)
            rho=1+.5*c['force']/force_max if force_max>0 else 1.
            cost=torch.where(valid,rho**2*distance/s.contact_scale_m**2,0.)
            norm=torch.where(valid,rho**2*(1-dot)/(1-np.cos(np.deg2rad(s.normal_deg))),0.)
            contact_valid=contact_valid&valid
            unresolved=unresolved+(~valid).long()
            groups.setdefault(('object',c['region']),[]).append((cost,norm))
            gaps.append(torch.where(valid,distance.detach().sqrt(),0.)); angles.append(torch.rad2deg(torch.acos(dot.detach())))
            contacts.append((pos,target.expand_as(pos)))
        if groups:
            parts['contact']=torch.stack([torch.stack([v[0] for v in values]).mean(0) for values in groups.values()]).mean(0)
            parts['normal']=torch.stack([torch.stack([v[1] for v in values]).mean(0) for values in groups.values()]).mean(0)
        depth,valid=h.penetration(q,pfull,rfull,dense) if penetration is None else penetration
        valid=valid&contact_valid
        parts['penetration']=(depth/s.penetration_scale_m).square()
        total=sum(parts[k] for k in ('direction','laplacian','smooth','antireverse','thumb_relative'))
        total=total+s.contact*parts['contact']+s.normal*parts['normal']+s.penetration*parts['penetration']
        max_gap=torch.stack(gaps).amax(0) if gaps else q.new_zeros(len(q))
        max_normal=torch.stack(angles).amax(0) if angles else q.new_zeros(len(q))
        feasible=(valid & torch.isfinite(total) & (depth<=s.penetration_limit_m) & (max_gap<=s.contact_limit_m)
                  & (max_normal<=s.normal_deg) & (q>=h.low-1e-6).all(-1) & (q<=h.high+1e-6).all(-1))
        metrics=dict(feasible=feasible,contact_count=len(gaps),max_gap_m=max_gap,max_normal_deg=max_normal,
            max_penetration_m=depth,bone_direction_deg=direction_error,points=contacts,query_valid=valid,groups=groups,
            contact_query_valid=contact_valid,unresolved_contact_count=unresolved)
        return total,parts,metrics

    def batch(self,q,rows,references=None,targets=None,acceleration=None,previous_rows=None,audit=False,normalizers=None,*,collect_metrics=True):
        s=self.settings
        if self.collision is not None and not audit:
            if targets is not None or any((s.lambda_lap,s.lambda_thumb_rel,s.lambda_antirev)):
                raise ValueError('可微策略必须使用连续接触场目标；旧 CARD／轨迹代价仅供历史审计')
            from .cart_fast import fast_objective
            return fast_objective(self,q,rows,references,acceleration,previous_rows,False,normalizers,collect_metrics=collect_metrics)
        if self.vectorized and self.collision is not None and targets is None and not any((s.lambda_lap,s.lambda_thumb_rel,s.lambda_antirev)):
            from .cart_fast import fast_objective
            return fast_objective(self,q,rows,references,acceleration,previous_rows,audit,normalizers,collect_metrics=collect_metrics)
        totals=[]; parts={}; metrics=[];groups={}
        frames=self.hand.frames(q)
        if self.collision is None:
            depth,valid=self.hand.penetration(q,frames[3],frames[4])
            self_depth=point_depth=None
        else:
            self_depth,point_depth=self.collision.penalties(q,rows,frames)
            depth=torch.maximum(self_depth,point_depth)
            valid=torch.ones(len(q),device=q.device,dtype=torch.bool)
            if audit:depth,valid=self.collision.audit(q,rows,frames)
        for i,row in enumerate(rows):
            total,values,quality=self(q[i:i+1],row,None if references is None else references[i:i+1],
                dense=audit,frames=tuple(v[i:i+1] for v in frames),penetration=(depth[i:i+1],valid[i:i+1]))
            totals.append(total[0]); metrics.append(quality)
            for k,v in values.items():parts.setdefault(k,[]).append(v[0])
            for key,items in quality['groups'].items():
                groups.setdefault(key,[]).append((torch.stack([v[0] for v in items]).mean(),torch.stack([v[1] for v in items]).mean()))
        parts={k:torch.stack(v).mean() for k,v in parts.items()}
        loss=torch.stack(totals).mean()
        if groups:
            contact=torch.stack([torch.stack([v[0] for v in items]).mean() for items in groups.values()]).mean()
            normal=torch.stack([torch.stack([v[1] for v in items]).mean() for items in groups.values()]).mean()
            loss=loss+self.settings.contact*(contact-parts['contact'])+self.settings.normal*(normal-parts['normal'])
            parts.update(contact=contact,normal=normal)
        if self.collision is not None:
            # Sum the two independent barriers instead of allowing one to hide
            # behind the maximum used by the physical feasibility report.
            barrier=((self_depth/self.settings.penetration_scale_m).square()+
                     (point_depth/self.settings.penetration_scale_m).square()).mean()
            if audit:barrier=(depth/self.settings.penetration_scale_m).square().mean()
            loss=loss+self.settings.penetration*(barrier-parts['penetration'])
            parts['penetration']=barrier
            parts['self_penetration']=(self_depth/self.settings.penetration_scale_m).square().mean()
            parts['tactile_penetration']=(point_depth/self.settings.penetration_scale_m).square().mean()
            keep,counts=self.collision.retention(q,rows,previous_rows or [None]*len(rows),frames,self.settings.contact_scale_m)
            parts['contact_keep']=keep.mean();loss=loss+self.settings.contact_keep*parts['contact_keep']
            for i,quality in enumerate(metrics):
                quality.update(contact_keep_count=counts[i],contact_keep_loss=keep[i:i+1].detach())
        if targets is not None:
            parts['trajectory']=((q-targets)/(self.hand.high-self.hand.low)).square().mean()
            loss=loss+self.settings.trajectory*parts['trajectory']
        if acceleration is not None:
            parts['acceleration']=acceleration.square().mean()
            loss=loss+self.settings.acceleration*parts['acceleration']
        return loss,parts,metrics


def policy_settings():
    """No offline teacher or unrelated pose priors in the v5 objective."""
    return ObjectiveSettings(trajectory=0.,acceleration=0.,lambda_lap=0.,lambda_thumb_rel=0.,lambda_antirev=0.)
