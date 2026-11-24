"""Joint/point grouped replay, distance, spatial derivatives and native q VJPs."""
import numpy as np
import torch
from torch.nn import functional as F
from .cart_conditional_sdf import VALIDATION,observed_to_robot


class ConditionalSDFLearner:
    def __init__(self,collision,capacity=200000,batch_size=2048,optimizer_kind='muon_sgd'):
        from .cart_optim import MuonSGD
        if optimizer_kind!='muon_sgd':raise ValueError('Conditional run requires Muon+SGD')
        self.collision=collision;self.model=collision.model;self.oracle=collision.oracle
        self.capacity=capacity;self.batch_size=batch_size;self.points_per_pose=64
        self.pose_batch=max(1,batch_size//self.points_per_pose)
        self.pose_capacity=max(self.pose_batch,capacity//self.points_per_pose)
        self.steps=0;self.size=0;self.cursor=0
        shapes=dict(q=(20,),points=(64,3),links=(64,),distance=(64,21),normal=(64,3),valid=(64,),
                    clearance_gradient=(20,),guard_gradient=(20,),cost_valid=(2,))
        self.replay={k:torch.empty((self.pose_capacity,*shape),dtype=torch.long if k=='links' else torch.bool if k in ('valid','cost_valid') else torch.float32) for k,shape in shapes.items()}
        self.optimizer=MuonSGD(self.model,lr=1.e-3,pretrained_lr=self.model.pretrained_lr,vector_lr_scale=.1)
        self.triangles=[];self.areas=[]
        for mesh in collision.hand.surface.meshes:
            tri=collision.hand.tensor(np.array(mesh.triangles,copy=True));self.triangles.append(tri)
            self.areas.append(torch.linalg.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]).norm(dim=-1))

    @torch.no_grad()
    def collect(self):
        h=self.collision.hand;device=h.device;b=self.pose_batch;n=self.points_per_pose
        q=h.low+torch.rand(b,20,device=device)*(h.high-h.low)
        prior=getattr(self.collision,'geometry_seed_q',None)
        if prior is not None:
            seed=prior[torch.randint(len(prior),(b,),device=device)]+.04*torch.randn_like(q)
            q=torch.where(torch.rand(b,1,device=device)<.5,seed.clamp(h.low,h.high),q)
        links=(torch.arange(b*n,device=device)%21).reshape(b,n);local=torch.empty(b*n,3,device=device)
        for owner in range(21):
            ids=(links.flatten()==owner).nonzero().flatten();k=len(ids)
            if not k:continue
            tri=self.triangles[owner][torch.multinomial(self.areas[owner],k,replacement=True)]
            uv=torch.rand(k,2,device=device);s=uv[:,0].sqrt()
            bary=torch.stack((1-s,s*(1-uv[:,1]),s*uv[:,1]),-1)
            near=(tri*bary[...,None]).sum(1)+torch.randn(k,3,device=device)*torch.where(torch.rand(k,1,device=device)<.75,.003,.025)
            box=self.oracle.lower[owner]-.03+torch.rand_like(near)*(self.oracle.upper[owner]-self.oracle.lower[owner]+.06)
            local[ids]=torch.where(torch.rand(k,1,device=device)<.75,near,box)
        local=local.reshape(b,n,3);p,r=h.frames(q)[:2];ids=torch.arange(b,device=device)[:,None]
        world=(r[ids,links]@local[...,None]).squeeze(-1)+p[ids,links]
        distances,normals=self.model.native_all(q,world,self.oracle)
        normal=normals.gather(2,links[...,None,None].expand(-1,-1,1,3)).squeeze(2)
        probe=F.normalize(torch.randn_like(local),dim=-1);fds=[]
        for eps in (2.e-5,1.e-5):
            plus=self.oracle.label((local+eps*probe).reshape(-1,3),links.flatten())[0].reshape(b,n)
            minus=self.oracle.label((local-eps*probe).reshape(-1,3),links.flatten())[0].reshape(b,n)
            fds.append((plus-minus)/(2*eps))
        valid=(fds[0]-fds[1]).abs()<.02+.02*fds[1].abs()
        valid &= ((normal*probe).sum(-1)-fds[1]).abs()<.02+.02*fds[1].abs()
        # Audit joint motion labels at fixed world points before fitting q VJPs.
        direction=F.normalize(torch.randn_like(q),dim=-1)
        def local_fn(a):return self.model.local_selected(a,world,links)
        with torch.enable_grad():
            _,tangent=torch.autograd.functional.jvp(local_fn,q,direction)
        analytic=(normal*tangent).sum(-1);qfd=[]
        for eps in (5.e-4,2.5e-4):
            plus=self.model.native_selected(q+eps*direction,world,links,self.oracle)[0]
            minus=self.model.native_selected(q-eps*direction,world,links,self.oracle)[0]
            qfd.append((plus-minus)/(2*eps))
        tolerance=5.e-4+.03*qfd[1].abs()
        valid &= (qfd[0]-qfd[1]).abs()<tolerance
        valid &= (analytic-qfd[1]).abs()<tolerance
        # A small subset of grouped poses carries direct native COST-gradient
        # labels, not just distance Jacobians. Every term still comes from the
        # same differentiable scalar potential used by the policy.
        count=min(4,b);cg=torch.zeros_like(q);gg=torch.zeros_like(q)
        cost_valid=torch.zeros((b,2),device=device,dtype=torch.bool)
        with torch.enable_grad():
            state=q[:count].detach().requires_grad_()
            native=self.model.self_terms(state,oracle=self.oracle,details=True)
            gc=torch.autograd.grad(native[:,0].sum(),state,retain_graph=True)[0]
            gp=torch.autograd.grad(native[:,1].sum(),state)[0]
        cg[:count]=gc;gg[:count]=gp
        direction=F.normalize(gc/(gc.norm(dim=-1,keepdim=True).clamp_min(1.e-8))+
                              gp/(gp.norm(dim=-1,keepdim=True).clamp_min(1.e-8)),dim=-1)
        fd=[]
        for eps in (5.e-4,2.5e-4):
            plus=self.model.self_terms(q[:count]+eps*direction,oracle=self.oracle)
            minus=self.model.self_terms(q[:count]-eps*direction,oracle=self.oracle)
            fd.append((plus[:,:2]-minus[:,:2])/(2*eps))
        analytic=torch.stack(((gc*direction).sum(-1),(gp*direction).sum(-1)),-1)
        tolerance=.05+.05*torch.maximum(fd[0].abs(),fd[1].abs())
        cost_valid[:count]=((fd[0]-fd[1]).abs()<tolerance)&((analytic-fd[1]).abs()<tolerance)
        entries=dict(q=q,points=world,links=links,distance=distances,normal=normal,valid=valid,
                     clearance_gradient=cg,guard_gradient=gg,cost_valid=cost_valid)
        slot=(torch.arange(b)+self.cursor)%self.pose_capacity
        for name,value in entries.items():self.replay[name][slot]=value.cpu()
        self.cursor=(self.cursor+b)%self.pose_capacity;self.size=min(self.pose_capacity,self.size+b)
        self.collision.collected=[];self.collision.contact_collected=[];self.collision.q_collected=[]

    def update(self):
        from .cart_optim import scale_learning_rates
        scale_learning_rates(self.optimizer,max(.0625,.5**(self.steps//5000)))
        ids=torch.randint(self.size,(min(self.pose_batch,self.size),))
        data={k:v[ids].to(self.oracle.device) for k,v in self.replay.items()}
        q=data['q'].requires_grad_();x=data['points'].requires_grad_();links=data['links'];valid=data['valid']
        d=self.model.pretrained(q,x);selected=d.gather(-1,links[...,None]).squeeze(-1)
        spatial=torch.autograd.grad(selected.sum(),x,create_graph=True,retain_graph=True)[0]
        p,r=self.collision.hand.frames(q)[:2];batch=torch.arange(len(q),device=q.device)[:,None]
        target_spatial=(r[batch,links].detach()@data['normal'][...,None]).squeeze(-1)
        probe=torch.randint(2,links.shape,device=q.device).to(q.dtype).mul(2).sub(1)*valid
        predicted_q=torch.autograd.grad((selected*probe).sum(),q,create_graph=True,retain_graph=True)[0]/self.points_per_pose**.5
        local=self.model.local_selected(q,x.detach(),links)
        native_linear=((local-local.detach())*data['normal']).sum(-1)
        target_q=torch.autograd.grad((native_linear*probe).sum(),q,retain_graph=True)[0].detach()/self.points_per_pose**.5
        weights=1+4*(data['distance'].abs()<.008).to(d.dtype)
        distance=(F.smooth_l1_loss(d/.001,data['distance']/.001,reduction='none')*weights).sum()/weights.sum()
        derivative=F.smooth_l1_loss(spatial[valid],target_spatial[valid]) if valid.any() else spatial.sum()*0
        normal=(1-F.cosine_similarity(spatial[valid],target_spatial[valid])).mean() if valid.any() else spatial.sum()*0
        q_jacobian=F.smooth_l1_loss(predicted_q/.05,target_q/.05)
        q_normal=(1-F.cosine_similarity(predicted_q,target_q)).mean()
        eikonal=(spatial.norm(dim=-1)-1).square().mean()
        sign=F.softplus(-d*torch.sign(data['distance'])/.001).mean()
        # Supervise escape directions of the actual aggregate cost. The masks
        # derive only from native finite-difference label validity.
        selected=data['cost_valid'].any(-1).nonzero().flatten()
        cost_fit=d.sum()*0;cost_direction=d.sum()*0
        if len(selected):
            potential=self.model.self_terms(q[selected])
            for column,name in ((0,'clearance_gradient'),(1,'guard_gradient')):
                gp=torch.autograd.grad(potential[:,column].sum(),q,create_graph=True,retain_graph=True)[0][selected]
                gt=data[name][selected];mask=data['cost_valid'][selected,column]&(gt.norm(dim=-1)>1.e-3)
                if mask.any():
                    scale=gt[mask].norm(dim=-1,keepdim=True).clamp_min(1.)
                    cost_fit=cost_fit+F.smooth_l1_loss(gp[mask]/scale,gt[mask]/scale)
                    cost_direction=cost_direction+(1-F.cosine_similarity(gp[mask],gt[mask])).mean()
        loss=distance+.5*derivative+.5*normal+q_jacobian+.2*q_normal+.1*eikonal+.1*sign+cost_fit+.5*cost_direction
        if not torch.isfinite(loss):raise ValueError('Nonfinite conditional geometry loss')
        self.optimizer.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(self.model.pretrained.parameters(),10.,error_if_nonfinite=True)
        self.optimizer.step();self.steps+=1
        return {k:float(v.detach()) for k,v in dict(total=loss,distance=distance,distance_jacobian=derivative,
            normal=normal,q_jacobian=q_jacobian,q_direction=q_normal,eikonal=eikonal,sign=sign,
            self_cost_jacobian=cost_fit,self_escape_alignment=cost_direction,
            derivative_label_valid_fraction=valid.float().mean(),gradient_norm_before_clip=norm,
            gradient_clip_multiplier=(10/(norm+1.e-6)).clamp(max=1)).items()}

    def state_dict(self):
        return dict(schema=VALIDATION,steps=self.steps,size=self.size,cursor=self.cursor,capacity=self.capacity,
            batch_size=self.batch_size,optimizer=self.optimizer.state_dict(),replay={k:v[:self.size].clone() for k,v in self.replay.items()})

    def load_state_dict(self,state):
        if state['schema']!=VALIDATION or state['capacity']!=self.capacity or state['batch_size']!=self.batch_size:
            raise ValueError('Conditional geometry replay/configuration changed')
        for name in ('steps','size','cursor'):setattr(self,name,state[name])
        for k,v in state['replay'].items():self.replay[k][:self.size]=v
        self.optimizer.load_state_dict(state['optimizer'])

    def audit_labels(self,output):
        from pathlib import Path
        import json
        from .cart_geometry_audit import audit_geometry_labels,directional_check
        report=audit_geometry_labels(self.collision,output=output)
        h=self.collision.hand;device=h.device
        devices=[torch.device(device)] if str(device).startswith('cuda') else []
        with torch.random.fork_rng(devices=devices),torch.no_grad():
            torch.manual_seed(98371)
            q=h.low+torch.rand(16,20,device=device)*(h.high-h.low)
            local,links=self.oracle.random_queries(256);local=local.reshape(16,16,3);links=links.reshape(16,16)
            p,r=h.frames(q)[:2];ids=torch.arange(16,device=device)[:,None]
            world=(r[ids,links]@local[...,None]).squeeze(-1)+p[ids,links]
            d,n=self.model.native_selected(q,world,links,self.oracle)
            all_d,_=self.model.native_all(q,world,self.oracle)
            routing_error=float((d-all_d.gather(-1,links[...,None]).squeeze(-1)).abs().max())
            direction=F.normalize(torch.randn_like(q),dim=-1)
            with torch.enable_grad():
                _,tangent=torch.autograd.functional.jvp(lambda a:self.model.local_selected(a,world,links),q,direction)
            analytic=(n*tangent).sum(-1).flatten();fds=[]
            for eps in (5.e-4,2.5e-4):
                plus=self.model.native_selected(q+eps*direction,world,links,self.oracle)[0]
                minus=self.model.native_selected(q-eps*direction,world,links,self.oracle)[0]
                fds.append(((plus-minus)/(2*eps)).flatten())
            derivative,stable=directional_check(analytic,*fds,atol=5.e-4,rtol=.03)
            report['conditional_native_q_derivative']=derivative
            report['conditional_all_link_routing_max_error_m']=routing_error
            report['conditional_label_schema']=VALIDATION
            report['accepted']=report['accepted'] and routing_error<1.e-5 and derivative['stable_fraction']>=.9 and derivative['correct_on_stable_fraction']>=.98
            np.savez(Path(output)/'conditional_samples.npz',q=q.cpu().numpy(),world_points=world.cpu().numpy(),links=links.cpu().numpy(),
                direction=direction.cpu().numpy(),analytic=analytic.cpu().numpy(),fd_coarse=fds[0].cpu().numpy(),fd_fine=fds[1].cpu().numpy(),stable=stable.cpu().numpy())
        path=Path(output)/'report.json';temp=path.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2,allow_nan=False));temp.replace(path)
        return report

    @torch.no_grad()
    def validate(self,points,links):
        devices=[torch.device(points.device)] if points.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(9716);h=self.collision.hand
            q=h.low+torch.rand(128,20,device=points.device)*(h.high-h.low)
        count=len(points)//len(q);local=points[:len(q)*count].reshape(len(q),count,3)
        owner=links[:len(q)*count].reshape(len(q),count);ids=torch.arange(len(q),device=q.device)[:,None]
        p,r=h.frames(q)[:2];world=(r[ids,owner]@local[...,None]).squeeze(-1)+p[ids,owner]
        truth,n=self.oracle.label(local.reshape(-1,3),owner.flatten());truth=truth.reshape_as(owner);n=n.reshape_as(local)
        with torch.enable_grad():
            x=world.detach().requires_grad_();predicted=self.model.pretrained(q,x).gather(-1,owner[...,None]).squeeze(-1)
            gradient=torch.autograd.grad(predicted.sum(),x)[0]
        predicted=predicted.detach();gradient=gradient.detach();target_normal=(r[ids,owner]@n[...,None]).squeeze(-1)
        probe=F.normalize(torch.sin(torch.arange(local.numel(),device=q.device).reshape_as(local)+.4),dim=-1);fds=[]
        for eps in (2.e-5,1.e-5):
            plus=self.oracle.label((local+eps*probe).reshape(-1,3),owner.flatten())[0].reshape_as(owner)
            minus=self.oracle.label((local-eps*probe).reshape(-1,3),owner.flatten())[0].reshape_as(owner)
            fds.append((plus-minus)/(2*eps))
        stable=(fds[0]-fds[1]).abs()<.02+.02*fds[1].abs()
        stable &= ((n*probe).sum(-1)-fds[1]).abs()<.02+.02*fds[1].abs()
        error=(predicted-truth).abs();inside=truth<-.0005
        report=dict(geometry_validation=VALIDATION,projection_required=False,samples=owner.numel(),
            distance_mean_mm=float(error.mean()*1000),distance_p95_mm=float(torch.quantile(error,.95)*1000),
            penetration_cases=int(inside.sum()),penetration_false_negative_rate=float((predicted[inside]>=0).float().mean()),
            spatial_gradient_norm_mean=float(gradient[stable].norm(dim=-1).mean()),
            spatial_gradient_cosine_mean=float(F.cosine_similarity(gradient[stable],target_normal[stable]).mean()),
            derivative_label_valid_fraction=float(stable.float().mean()))
        for link in range(21):
            mask=owner==link;regular=mask&stable
            report[f'link_{link}_distance_p95_mm']=float(torch.quantile(error[mask],.95)*1000)
            report[f'link_{link}_gradient_cosine_mean']=float(F.cosine_similarity(gradient[regular],target_normal[regular]).mean())
        with torch.enable_grad():
            state=torch.cat((q,q.new_zeros(len(q),6)),-1).requires_grad_();obs=world[:,0].detach()
            canonical=observed_to_robot(obs,state,torch.arange(len(q),device=q.device))
            target,_=self.model.native_selected(state[:,:20],canonical[:,None],owner[:,:1],self.oracle,linearize=True)
            value=self.model.observed(state,obs,torch.arange(len(q),device=q.device)).gather(-1,owner[:,:1])
            ga=torch.autograd.grad(value.sum(),state)[0];gt=torch.autograd.grad(target.sum(),state)[0]
        report['articulated_gradient_cosine_mean']=float(F.cosine_similarity(ga,gt).mean())
        report.update(self.validate_collision(q))
        return report

    @torch.no_grad()
    def validate_collision(self,q):
        predicted=[];native=[];pg=[];ng=[]
        for batch in q.split(self.model.spec['pose_chunk']):
            with torch.enable_grad():
                s=batch.detach().requires_grad_();a=self.model.self_terms(s,details=True);b=self.model.self_terms(s,oracle=self.oracle,details=True)
                pa=[];na=[]
                for col in (0,3,1):
                    pa.append(torch.autograd.grad(a[:,col].sum(),s,retain_graph=True)[0])
                    na.append(torch.autograd.grad(b[:,col].sum(),s,retain_graph=True)[0])
            predicted.append(a.detach());native.append(b.detach());pg.append(torch.stack(pa,1));ng.append(torch.stack(na,1))
        a,b=torch.cat(predicted),torch.cat(native);ga,gb=torch.cat(pg),torch.cat(ng)
        collision=b[:,2]>.0005;cap=b[:,2]>.005;guard=b[:,1]>1.e-6
        escaped=self.model.self_terms(q-.001*F.normalize(ga[:,0],dim=-1),oracle=self.oracle)
        combined=1.5**2*ga[:,0]+40**2*ga[:,2]
        next_native=self.model.self_terms(q-.001*F.normalize(combined,dim=-1),oracle=self.oracle)
        total=lambda v:1.5**2*v[:,0]+40**2*v[:,1]
        return dict(q_samples=len(q),q_collision_cases=int(collision.sum()),q_cap_cases=int(cap.sum()),
            q_false_negative_rate=float((a[collision,0]<=1.e-12).float().mean()),
            q_gradient_cosine_mean=float(F.cosine_similarity(ga[collision,0],gb[collision,0]).mean()),
            q_cap_gradient_cosine_mean=float(F.cosine_similarity(ga[cap,1],gb[cap,1]).mean()),
            q_guard_gradient_cosine_mean=float(F.cosine_similarity(ga[guard,2],gb[guard,2]).mean()),
            q_escape_cost_descent_rate=float((escaped[collision,0]<b[collision,0]).float().mean()),
            q_combined_escape_cost_descent_rate=float((total(next_native)[collision]<total(b)[collision]).float().mean()),
            q_combined_escape_cap_descent_rate=float((next_native[cap,1]<b[cap,1]).float().mean()),
            q_depth_mae_mm=float((a[:,2]-b[:,2]).abs().mean()*1000))
