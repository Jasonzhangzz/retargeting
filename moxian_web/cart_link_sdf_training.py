"""Distance/first-derivative training and native validation for rigid link SDFs."""
import numpy as np
import torch
from torch.nn import functional as F
from .cart_link_sdf import VALIDATION
from .cart_collision import SDFLearner


class LinkSDFLearner:
    add=SDFLearner.add
    def __init__(self,collision,capacity=200000,batch_size=2048,optimizer_kind='muon_sgd'):
        self.collision=collision;self.model=collision.model;self.oracle=collision.oracle
        if not 0<batch_size<=capacity:raise ValueError('Expected 0 < batch_size <= replay capacity')
        self.capacity=capacity;self.batch_size=batch_size;self.cursor=0;self.size=0;self.steps=0
        self.points=torch.empty(capacity,3);self.links=torch.empty(capacity,dtype=torch.long)
        self.distances=torch.empty(capacity);self.normals=torch.empty(capacity,3)
        self.gradient_valid=torch.empty(capacity,dtype=torch.bool)
        if optimizer_kind!='muon_sgd':raise ValueError('This experiment uses Muon+SGD')
        from .cart_optim import MuonSGD
        self.optimizer=MuonSGD(self.model,lr=1.e-3,pretrained_lr=self.model.pretrained_lr,vector_lr_scale=self.model.vector_lr_scale)
        self.triangles=[];self.areas=[]
        for mesh in collision.hand.surface.meshes:
            tri=collision.hand.tensor(np.array(mesh.triangles,copy=True))
            self.triangles.append(tri);self.areas.append(torch.linalg.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]).norm(dim=-1))

    @torch.no_grad()
    def collect(self):
        # Balanced links, true surface-area sampling; no learned/FK quantity is
        # cached across policy states. These replay entries are rigid-local.
        n=self.batch_size;owners=torch.arange(n,device=self.oracle.device)%21
        x=torch.empty(n,3,device=self.oracle.device)
        for owner in range(21):
            ids=(owners==owner).nonzero().flatten();k=len(ids)
            tri=self.triangles[owner][torch.multinomial(self.areas[owner],k,replacement=True)]
            u=torch.rand(k,2,device=x.device);s=u[:,0].sqrt()
            bary=torch.stack((1-s,s*(1-u[:,1]),s*u[:,1]),-1)
            near=(tri*bary[...,None]).sum(1)
            sigma=torch.where(torch.rand(k,1,device=x.device)<.75,.003,.025)
            near=near+torch.randn_like(near)*sigma
            box=self.oracle.lower[owner]-.03+torch.rand_like(near)*(self.oracle.upper[owner]-self.oracle.lower[owner]+.06)
            x[ids]=torch.where(torch.rand(k,1,device=x.device)<.75,near,box)
        self.add(x,owners)
        # A scalar distance remains valid at a sharp feature; derivative labels
        # are scored separately and only locally stable derivatives are fitted.
        ids=(torch.arange(n)+self.cursor-n)%self.capacity
        p=self.points[ids].to(x.device);l=self.links[ids].to(x.device);normal=self.normals[ids].to(x.device)
        probe=F.normalize(torch.randn_like(p),dim=-1);fds=[]
        for eps in (2.e-5,1.e-5):
            fds.append((self.oracle.label(p+eps*probe,l)[0]-self.oracle.label(p-eps*probe,l)[0])/(2*eps))
        analytic=(normal*probe).sum(-1)
        valid=(fds[0]-fds[1]).abs()<.02+.02*fds[1].abs()
        valid &= (analytic-fds[1]).abs()<.02+.02*fds[1].abs()
        self.gradient_valid[ids]=valid.cpu()
        self.collision.collected=[];self.collision.contact_collected=[];self.collision.q_collected=[]

    def update(self):
        from .cart_optim import scale_learning_rates
        scale_learning_rates(self.optimizer,max(.0625,.5**(self.steps//5000)))
        ids=torch.randint(self.size,(min(self.batch_size,self.size),))
        x=self.points[ids].to(self.oracle.device).requires_grad_();links=self.links[ids].to(x.device)
        target=self.distances[ids].to(x.device);normal=self.normals[ids].to(x.device);valid=self.gradient_valid[ids].to(x.device)
        d=self.model(x,links);g=torch.autograd.grad(d.sum(),x,create_graph=True)[0]
        distance=F.smooth_l1_loss(d/.001,target/.001)
        derivative=F.smooth_l1_loss(g[valid],normal[valid]) if valid.any() else g.sum()*0
        alignment=(1-F.cosine_similarity(g[valid],normal[valid])).mean() if valid.any() else g.sum()*0
        eikonal=(g.norm(dim=-1)-1).square().mean()
        sign=F.softplus(-d*torch.sign(target)/.001).mean()
        loss=distance+.5*derivative+.5*alignment+.1*eikonal+.1*sign
        if not torch.isfinite(loss):raise ValueError('Link SDF training became nonfinite')
        self.optimizer.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(self.model.pretrained.parameters(),10.,error_if_nonfinite=True)
        self.optimizer.step();self.steps+=1
        return {k:float(v.detach()) for k,v in dict(total=loss,distance=distance,distance_jacobian=derivative,
            normal=alignment,eikonal=eikonal,sign=sign,derivative_label_valid_fraction=valid.float().mean(),
            gradient_norm_before_clip=norm,gradient_clip_multiplier=(10/(norm+1.e-6)).clamp(max=1)).items()}

    def state_dict(self):
        return dict(schema=VALIDATION,steps=self.steps,capacity=self.capacity,batch_size=self.batch_size,
            size=self.size,cursor=self.cursor,optimizer=self.optimizer.state_dict(),
            **{k:getattr(self,k)[:self.size].clone() for k in ('points','links','distances','normals','gradient_valid')})

    def load_state_dict(self,state,*,load_optimizer=True):
        if any(state[k]!=getattr(self,k) for k in ('capacity','batch_size')) or state.get('schema')!=VALIDATION:
            raise ValueError('Link SDF replay or supervision version changed')
        self.steps=state['steps'];self.size=state['size'];self.cursor=state['cursor']
        for key in ('points','links','distances','normals','gradient_valid'):getattr(self,key)[:self.size].copy_(state[key])
        if load_optimizer:self.optimizer.load_state_dict(state['optimizer'])

    def audit_labels(self,output):
        from .cart_geometry_audit import audit_geometry_labels
        # Independent winding/distance, two-scale SDF derivatives and exact FK
        # are rechecked. The extra native projection audit is teacher-only.
        return audit_geometry_labels(self.collision,output=output)

    @torch.no_grad()
    def validate(self,points,links):
        truth,n=self.oracle.label(points,links)
        with torch.enable_grad():
            x=points.detach().requires_grad_();d=self.model(x,links)
            g=torch.autograd.grad(d.sum(),x)[0].detach();d=d.detach()
        error=(d-truth).abs();inside=truth<-.0005
        probe=F.normalize(torch.sin(torch.arange(points.numel(),device=points.device).reshape_as(points)+.4),dim=-1)
        fd=[]
        for eps in (2.e-5,1.e-5):fd.append((self.oracle.label(points+eps*probe,links)[0]-self.oracle.label(points-eps*probe,links)[0])/(2*eps))
        stable=(fd[0]-fd[1]).abs()<.02+.02*fd[1].abs()
        stable &= ((n*probe).sum(-1)-fd[1]).abs()<.02+.02*fd[1].abs()
        report=dict(geometry_validation=VALIDATION,samples=len(points),distance_mean_mm=float(error.mean()*1000),
            distance_p95_mm=float(torch.quantile(error,.95)*1000),penetration_cases=int(inside.sum()),
            penetration_false_negative_rate=float((d[inside]>=0).float().mean()) if inside.any() else None,
            spatial_gradient_norm_mean=float(g[stable].norm(dim=-1).mean()),
            spatial_gradient_cosine_mean=float(F.cosine_similarity(g[stable],n[stable]).mean()),
            derivative_label_valid_fraction=float(stable.float().mean()),
            spatial_gradient_error_mean=float((g[stable]-n[stable]).norm(dim=-1).mean()),
            projection_required=False)
        for owner in range(21):
            mask=links==owner;regular=mask&stable
            report[f'link_{owner}_distance_p95_mm']=float(torch.quantile(error[mask],.95)*1000) if mask.any() else None
            report[f'link_{owner}_gradient_cosine_mean']=float(F.cosine_similarity(g[regular],n[regular]).mean()) if regular.any() else None
        # Point/FK Jacobian: a fixed observed point must see wrist and finger
        # motion. Compare AD to native SDF gradients composed with the same FK.
        devices=[torch.device(points.device)] if points.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(9716);h=self.collision.hand
            q=h.low+torch.rand(128,20,device=points.device)*(h.high-h.low)
        owner=links[:128];local=points[:128];ids=torch.arange(len(q),device=q.device)
        state=torch.cat((q,q.new_zeros(len(q),6)),-1).requires_grad_()
        with torch.enable_grad():
            p,r=h.frames(state)[:2];world=((r[ids,owner]@local[...,None]).squeeze(-1)+p[ids,owner]).detach()
            query=(r[ids,owner].transpose(-1,-2)@(world-p[ids,owner])[...,None]).squeeze(-1)
            native,ng=self.oracle.label(query,owner)
            native=native+((query-query.detach())*ng).sum(-1)
            predicted=self.model(query,owner)
            ga=torch.autograd.grad(predicted.sum(),state,retain_graph=True)[0]
            gt=torch.autograd.grad(native.sum(),state)[0]
        report['articulated_gradient_cosine_mean']=float(F.cosine_similarity(ga,gt).mean())
        # Physical collision energy and its actual gradient/escape, on the same
        # dense fixed guard set and native reference; no projection requirement.
        predicted=[];native=[];pg=[];ng=[]
        for batch in q.split(8):
            with torch.enable_grad():
                s=batch.detach().requires_grad_();a=self.model.self_terms(s);b=self.model.self_terms(s,oracle=self.oracle)
                ga=torch.autograd.grad(a[:,0].sum(),s,retain_graph=True)[0]
                gb=torch.autograd.grad(b[:,0].sum(),s,retain_graph=True)[0]
                ca=torch.autograd.grad(a[:,1].sum(),s)[0]
                cb=torch.autograd.grad(b[:,1].sum(),s)[0]
            predicted.append(a.detach());native.append(b.detach());pg.append(torch.stack((ga,ca),1));ng.append(torch.stack((gb,cb),1))
        a,b=torch.cat(predicted),torch.cat(native);ga,gb=torch.cat(pg),torch.cat(ng)
        collision=b[:,2]>.0005;cap=b[:,1]>1.e-6
        direction=F.normalize(ga[:,0],dim=-1);cap_direction=F.normalize(ga[:,1],dim=-1)
        # Native reference is recomputed after a small actual joint-space step.
        escaped=self.model.self_terms(q-.001*direction,oracle=self.oracle)
        cap_escaped=self.model.self_terms(q-.001*cap_direction,oracle=self.oracle)
        report.update(q_samples=len(q),q_collision_cases=int(collision.sum()),q_cap_cases=int(cap.sum()),
            q_false_negative_rate=float((a[collision,0]<=1.e-12).float().mean()) if collision.any() else None,
            q_gradient_cosine_mean=float(F.cosine_similarity(ga[collision,0],gb[collision,0]).mean()) if collision.any() else None,
            q_cap_gradient_cosine_mean=float(F.cosine_similarity(ga[cap,1],gb[cap,1]).mean()) if cap.any() else None,
            q_escape_cost_descent_rate=float((escaped[collision,0]<b[collision,0]).float().mean()) if collision.any() else None,
            q_combined_escape_cost_descent_rate=float((escaped[collision,0]<b[collision,0]).float().mean()) if collision.any() else None,
            q_combined_escape_cap_descent_rate=float((cap_escaped[cap,1]<b[cap,1]).float().mean()) if cap.any() else None,
            q_depth_mae_mm=float((a[:,2]-b[:,2]).abs().mean()*1000))
        return report
