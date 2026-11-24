"""Configuration-only collision potential, with mesh-supervised Jacobians.

Native queries are confined to QCollisionTeacher. The policy evaluates an
ordinary smooth MLP of the twenty joint angles: no FK, triangle IDs, witnesses,
or straight-through/custom backward are involved. Escape is the negative
derivative of the *same* scalar potential used in the policy objective.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

VERSION='q_collision_sobolev_v3'
GROUPS=[(a,a) for a in range(5)]+[(a,b) for a in range(5) for b in range(a+1,5)]


def field_spec(hand,group_sobolev=False,independent_decoders=False,consistent_supervision=False,structured=False):
    from .cart_pretrained_sdf import geometry_spec
    spec=dict(kind=VERSION,geometry=geometry_spec(hand),width=256,margin_mm=6.,
        allowed=hand.collision.allowed_mask.tolist() if hasattr(hand.collision,'allowed_mask') else
            __import__('warp').to_torch(hand.collision.allowed).bool().cpu().tolist(),
        groups=[list(g) for g in GROUPS],teacher='native_8_material_samples_CARD_energy_and_54_samples_depth',
        cap='squared_5mm_hinge_of_L2_native_penetration_upper_bound_54_samples',
        derivative='analytic_FK_signed_distance_witness_Jacobian_supervision',
        escape='negative_gradient_of_predicted_CARD_clearance_energy_in_radians')
    if structured:
        from .cart_structured_geometry import Q_KIND
        spec.update(kind=Q_KIND,architecture='exact_FK_fixed_material_samples_neural_SDF',
            derivative='AD_through_exact_FK_and_shared_point_SDF',depth_aggregation='logsumexp_upper_bound',
            smooth_temperature_mm=.05,mechanical_collar_band_m=2.e-6,learned_q_cost_decoder=False)
        return spec
    if group_sobolev:spec.update(kind='q_collision_group_sobolev_v4',group_sobolev=True)
    if independent_decoders:
        if not group_sobolev:raise ValueError('Independent collision decoders require grouped derivative supervision')
        spec.update(kind='q_collision_independent_sobolev_v5',independent_decoders=True,width=128,
            depth_aggregation='logsumexp_upper_bound',smooth_temperature_mm=.05)
    if consistent_supervision:
        if not independent_decoders:raise ValueError('Verified supervision requires independent collision decoders')
        spec.update(kind='q_collision_verified_sobolev_v6',consistent_supervision=True,
            teacher_depth='same_logsumexp_as_student',sampling='safe_boundary_shallow_deep_balanced',
            group_probe='active_or_uniform_one_hot',gradient_quality='two_scale_directional_native_audit')
    return spec


class QCollisionField(nn.Module):
    def __init__(self,hand,spec):
        super().__init__()
        if spec!=field_spec(hand,spec.get('group_sobolev',False),spec.get('independent_decoders',False),spec.get('consistent_supervision',False)):raise ValueError('Q collision geometry/domain/exclusions differ')
        self.spec=spec
        self.register_buffer('lower',hand.low.clone());self.register_buffer('upper',hand.high.clone())
        ids=[];mask=[]
        for a,b in GROUPS:
            ids.append(list(range(a*4,a*4+4))+list(range(b*4,b*4+4)))
            mask.append([1.]*4+([1.]*4 if a!=b else [0.]*4))
        self.register_buffer('joint_ids',torch.tensor(ids,device=hand.device))
        self.register_buffer('mask',hand.tensor(mask))
        w=spec['width']
        if spec.get('independent_decoders',False):
            self.decoders=nn.ModuleList([nn.Sequential(nn.Linear(24,w),nn.Softplus(beta=5),
                nn.Linear(w,w),nn.Softplus(beta=5),nn.Linear(w,w),nn.Softplus(beta=5),nn.Linear(w,3)) for _ in GROUPS])
        else:
            self.embedding=nn.Embedding(15,8)
            self.network=nn.Sequential(nn.Linear(32,w),nn.Softplus(beta=5),
                nn.Linear(w,w),nn.Softplus(beta=5),nn.Linear(w,w),nn.Softplus(beta=5),nn.Linear(w,3))
        self.to(hand.device)

    def components(self,q):
        u=(2*(q-self.lower)/(self.upper-self.lower)-1)[:,self.joint_ids]*self.mask
        angle=q[:,self.joint_ids]
        features=torch.cat((u,angle.sin()*self.mask,angle.cos()*self.mask),-1)
        if self.spec.get('independent_decoders',False):
            # Stack independent Linear parameters for batched GEMMs. Each 2-D
            # matrix remains a separate Muon parameter; no weights are shared.
            raw=features.transpose(0,1)
            for k in (0,2,4,6):
                w=torch.stack([net[k].weight for net in self.decoders])
                b=torch.stack([net[k].bias for net in self.decoders])
                raw=torch.bmm(raw,w.transpose(1,2))+b[:,None]
                if k!=6:raw=F.softplus(raw,beta=5)
            raw=raw.transpose(0,1)
        else:
            embedding=self.embedding.weight[None].expand(len(q),-1,-1)
            raw=self.network(torch.cat((features,embedding),-1))
        # Positive smooth outputs, in millimetres. Separate depth from CARD's
        # 6 mm safety margin: clearance cost is nonzero before intersection.
        return torch.stack((20*F.softplus(raw[...,0]),5*F.softplus(raw[...,1]),5*F.softplus(raw[...,2])),dim=-1)

    def forward(self,q,contact_groups=None):
        values=[]
        for x in q.split(2048):
            recompute=x.is_cuda and torch.is_grad_enabled() and x.requires_grad and not any(p.requires_grad for p in self.parameters())
            values.append(checkpoint(self.components,x,use_reentrant=False,preserve_rng_state=False) if recompute else self.components(x))
        parts=torch.cat(values)
        energy=parts[...,0].square()
        if contact_groups is not None:
            # Intended finger pairs may touch: replace the 6mm clearance
            # energy by their learned penetration energy. Never remove depth
            # or the global 5mm cap, including for these contact groups.
            energy=torch.where(contact_groups,parts[...,2].square(),energy)
        energy=energy.sum(-1)
        depth=self.aggregate_depth(parts[...,1])/1000
        return energy,depth,parts[...,2].square().sum(-1)

    def aggregate_depth(self,depth):
        if self.spec.get('independent_decoders',False):return .05*torch.logsumexp(depth/.05,-1)
        return (torch.softmax(depth/.05,-1)*depth).sum(-1)

    def escape(self,q):
        """Unit direction in joint radians, plus collision depth/probability.

        Direction is generated by autograd, not a disconnected vector head.
        Project only commands that would leave the allowed joint domain.
        """
        with torch.enable_grad():
            x=q.detach().requires_grad_();energy,depth,cap=self(x)
            clearance=-torch.autograd.grad(energy.sum(),x,retain_graph=True)[0]
            cap_penalty=(cap+1.e-8).sqrt().sub(5).clamp_min(0).square()
            direction=-torch.autograd.grad((1.5**2*energy+40**2*cap_penalty).sum(),x)[0]
            direction=torch.where(((x<=self.lower+1.e-6)&(direction<0))|((x>=self.upper-1.e-6)&(direction>0)),0.,direction)
            direction=F.normalize(direction,dim=-1)
        return dict(energy_mm2=energy.detach(),depth_m=depth.detach(),penetration_norm_m=cap.detach().sqrt()/1000,
            cap_energy_mm2=cap_penalty.detach(),
            collision_probability=torch.sigmoid((depth.detach()-.0005)/.0001),direction_rad=direction.detach(),
            clearance_direction_rad=F.normalize(clearance,dim=-1).detach())


class QCollisionTeacher:
    """Offline/online-label teacher only; never used by the policy backward."""
    def __init__(self,collision):
        self.collision=collision;self.hand=collision.hand;self.oracle=collision.oracle
        import copy
        self.depth_oracle=copy.copy(self.oracle)
        self.depth_oracle.surface=self.oracle.points;self.depth_oracle.surface_owners=self.oracle.owners
        self.depth_oracle.query_sources,self.depth_oracle.query_targets=self.oracle.allowed[self.oracle.owners].nonzero(as_tuple=True)
        self.groups=self.group_indices(self.oracle);self.depth_groups=self.group_indices(self.depth_oracle)

    @staticmethod
    def group_indices(oracle):
        source=oracle.surface_owners[oracle.query_sources];target=oracle.query_targets
        a=((source-1)//4).clamp_min(0);b=((target-1)//4).clamp_min(0)
        # Palm pairs belong to the connected finger's own group.
        a=torch.where(source==0,b,a);b=torch.where(target==0,a,b)
        return torch.tensor([GROUPS.index(tuple(sorted((int(x),int(y))))) for x,y in zip(a,b)],device=source.device)

    @staticmethod
    def distances(oracle,frames):
        local,links,valid=oracle.self_queries(frames)
        with torch.no_grad():d,n=oracle.label(local,links);d=d.reshape(links.shape);n=n.reshape(local.shape)
        return d+((local-local.detach())*n).sum(-1),valid

    @staticmethod
    def smooth_group_depths(signed,valid,groups):
        raw=torch.where(valid,-signed*1000,-torch.inf)
        return torch.stack([.05*torch.logsumexp(torch.cat((raw.new_zeros(len(raw),1),raw[:,groups==g]),-1)/.05,-1) for g in range(15)],-1)

    def label(self,q,group_sobolev=False,consistent=False):
        with torch.enable_grad():
            x=q.detach().requires_grad_();frames=self.hand.frames(x)
            signed,valid=self.distances(self.oracle,frames)
            # Correct local derivative of signed distance on each selected
            # feature. Face switches are outside the learned policy function.
            clearance=torch.where(valid,(6-signed*1000).clamp_min(0).square(),0.)
            energy=x.new_zeros(len(x),15).scatter_add(1,self.groups[None].expand(len(x),-1),clearance)
            amplitude=(energy+1.e-8).sqrt()
            total_amp=(energy.sum(-1)+.01).sqrt()
            gradient=torch.autograd.grad(total_amp.sum(),x,retain_graph=True)[0]
            extra={}
            if group_sobolev:
                if consistent:
                    weights=(energy.detach()>1.e-6).to(x.dtype)+.05
                    group=torch.multinomial(weights,1).squeeze(-1)
                    uniform=torch.randint(15,(len(x),),device=x.device)
                    group=torch.where(torch.rand(len(x),device=x.device)<.75,group,uniform)
                    probe=F.one_hot(group,15).to(x.dtype)
                else:probe=torch.randint(2,(len(x),15),device=x.device).mul(2).sub(1).to(x.dtype)
                extra.update(group_probe=probe,group_gradient=torch.autograd.grad((amplitude*probe).sum(),x,retain_graph=True)[0].detach())
            # Coarser native audit has 54 material points/link rather than 8.
            # Its derivative includes BOTH moving links, not point-SDF error.
            signed,valid=self.distances(self.depth_oracle,frames)
            penetration=torch.where(valid,(-signed*1000).clamp_min(0),0.)
            depth_groups=x.new_zeros(len(x),15).scatter_reduce(1,self.depth_groups[None].expand(len(x),-1),penetration,reduce='amax',include_self=True)
            depth_mm=depth_groups.amax(-1)
            if consistent:
                extra['native_depth_mm']=depth_mm.detach()
                depth_groups=self.smooth_group_depths(signed,valid,self.depth_groups)
                depth_mm=.05*torch.logsumexp(depth_groups/.05,-1)
            depth_gradient=torch.autograd.grad(depth_mm.sum(),x,retain_graph=True)[0]
            # ||all positive penetrations||_2 bounds every sampled depth and
            # has a much more stable derivative than the deepest witness.
            # Its 5 mm guard is conservative when several points penetrate.
            excess=penetration.square()
            cap_energy=x.new_zeros(len(x),15).scatter_add(1,self.depth_groups[None].expand(len(x),-1),excess)
            cap_amplitude=(cap_energy+1.e-8).sqrt()
            cap_gradient=torch.autograd.grad((cap_energy.sum(-1)+.0001).sqrt().sum(),x,retain_graph=group_sobolev)[0]
            if group_sobolev:
                extra['cap_group_gradient']=torch.autograd.grad((cap_amplitude*probe).sum(),x)[0].detach()
        if consistent:
            extra.update(self.verify_gradients(x.detach(),gradient.detach(),cap_gradient.detach(),depth_gradient.detach(),
                probe,extra['group_gradient'],extra['cap_group_gradient']))
        return dict(q=x.detach(),amplitude=amplitude.detach(),depth_mm=depth_mm.detach(),
            depth_groups_mm=depth_groups.detach(),gradient=gradient.detach(),depth_gradient=depth_gradient.detach(),
            cap_amplitude=cap_amplitude.detach(),cap_gradient=cap_gradient.detach(),**extra)

    @torch.no_grad()
    def values(self,frames,contact_groups=None,*,return_groups=False):
        signed,valid=self.distances(self.oracle,frames)
        clearance=torch.where(valid,(6-signed*1000).clamp_min(0).square(),0.)
        energy=clearance.sum(-1)
        signed,valid=self.distances(self.depth_oracle,frames)
        depth=torch.where(valid,(-signed).clamp_min(0),0.).amax(-1)
        penetration=torch.where(valid,(-signed*1000).clamp_min(0).square(),0.)
        cap=penetration.sum(-1)
        if return_groups:
            energy_groups=clearance.new_zeros(len(clearance),15).scatter_add(1,self.groups[None].expand(len(clearance),-1),clearance)
            cap_groups=penetration.new_zeros(len(penetration),15).scatter_add(1,self.depth_groups[None].expand(len(penetration),-1),penetration)
            z=self.smooth_group_depths(signed,valid,self.depth_groups)
            return dict(amplitude=(energy_groups+1.e-8).sqrt(),cap_amplitude=(cap_groups+1.e-8).sqrt(),
                clearance_norm=(energy_groups.sum(-1)+.01).sqrt(),cap_norm=(cap_groups.sum(-1)+.0001).sqrt(),
                smooth_depth=.05*torch.logsumexp(z/.05,-1),native_depth_mm=depth*1000)
        if contact_groups is not None:
            energy_groups=clearance.new_zeros(len(clearance),len(GROUPS)).scatter_add(1,self.groups[None].expand(len(clearance),-1),clearance)
            cap_groups=penetration.new_zeros(len(penetration),len(GROUPS)).scatter_add(1,self.depth_groups[None].expand(len(penetration),-1),penetration)
            energy=torch.where(contact_groups,cap_groups,energy_groups).sum(-1)
        return energy,depth,cap

    @torch.no_grad()
    def verify_gradients(self,q,gradient,cap_gradient,depth_gradient,probe,group_gradient,cap_group_gradient):
        """Reject derivative labels at native branch switches, not value samples."""
        random_direction=F.normalize(torch.randn_like(q),dim=-1)
        normal_direction=F.normalize(torch.where(cap_gradient.norm(dim=-1,keepdim=True)>1.e-3,cap_gradient,gradient),dim=-1)
        labels=dict(gradient_valid=gradient,cap_gradient_valid=cap_gradient,depth_gradient_valid=depth_gradient,
            group_gradient_valid=group_gradient,cap_group_gradient_valid=cap_group_gradient)
        valid={k:torch.ones(len(q),device=q.device,dtype=torch.bool) for k in labels}
        for direction in (random_direction,normal_direction):
            for step in (5.e-4,2.5e-4):
                plus=self.values(self.hand.frames(q+step*direction),return_groups=True)
                minus=self.values(self.hand.frames(q-step*direction),return_groups=True)
                values=dict(gradient_valid=(plus['clearance_norm'],minus['clearance_norm']),
                    cap_gradient_valid=(plus['cap_norm'],minus['cap_norm']),
                    depth_gradient_valid=(plus['smooth_depth'],minus['smooth_depth']),
                    group_gradient_valid=((plus['amplitude']*probe).sum(-1),(minus['amplitude']*probe).sum(-1)),
                    cap_group_gradient_valid=((plus['cap_amplitude']*probe).sum(-1),(minus['cap_amplitude']*probe).sum(-1)))
                for name,(a,b) in values.items():
                    fd=(a-b)/(2*step);ad=(labels[name]*direction).sum(-1)
                    valid[name] &= torch.isfinite(fd)&torch.isfinite(ad)&((fd-ad).abs()<.05+.05*torch.maximum(fd.abs(),ad.abs()))
        return valid


class QCollisionLearner:
    def __init__(self,collision,capacity=50000,batch_size=256):
        from .cart_optim import MuonSGD
        self.collision=collision;self.model=collision.model.q_collision
        self.teacher=QCollisionTeacher(collision);self.capacity=capacity;self.batch_size=batch_size
        self.structured=getattr(self.model,'structured',False)
        if self.structured:
            self.group_sobolev=False;self.consistent=False;self.optimizer=None
            self.replay={};self.size=0;self.cursor=0;self.steps=0
            return
        self.group_sobolev=self.model.spec.get('group_sobolev',False)
        self.consistent=self.model.spec.get('consistent_supervision',False)
        self.optimizer=MuonSGD(self.model,lr=1.e-3 if self.group_sobolev else 5.e-4,weight_decay=0.,
            vector_lr_scale=.1 if self.group_sobolev else 1.)
        self.replay={k:torch.empty((capacity,n)) if n else torch.empty(capacity) for k,n in
            dict(q=20,amplitude=15,depth_mm=0,depth_groups_mm=15,gradient=20,depth_gradient=20,cap_amplitude=15,cap_gradient=20).items()}
        self.size=0;self.cursor=0;self.steps=0
        if self.group_sobolev:
            self.replay.update(group_probe=torch.empty(capacity,15),group_gradient=torch.empty(capacity,20),
                cap_group_gradient=torch.empty(capacity,20))
        if self.consistent:
            self.replay.update({k:torch.empty(capacity,dtype=torch.bool if k.endswith('_valid') else torch.float32) for k in ('native_depth_mm','gradient_valid','cap_gradient_valid',
                'depth_gradient_valid','group_gradient_valid','cap_group_gradient_valid')})

    def add(self,q):
        if self.structured:raise ValueError('Structured SDF/FK collision has no q-regression replay')
        data=self.teacher.label(q,group_sobolev=self.group_sobolev,consistent=self.consistent)
        if not all(torch.isfinite(v).all() for v in data.values()):raise ValueError('Nonfinite native collision supervision')
        n=min(len(q),self.capacity);ids=(torch.arange(n)+self.cursor)%self.capacity
        for k,v in data.items():self.replay[k][ids]=v[-n:].cpu()
        self.cursor=(self.cursor+n)%self.capacity;self.size=min(self.capacity,self.size+n)

    def collect(self):
        if self.structured:
            self.collision.q_collected=[]
            return
        h=self.collision.hand;states=self.collision.q_collected;self.collision.q_collected=[]
        if states:
            q=torch.cat(states);q=q[torch.randperm(len(q),device=q.device)[:64]]
            self.add(q);self.add((q+torch.randn_like(q)*.04).clamp(h.low,h.high))
        seeds=getattr(self.collision,'geometry_seed_q',None)
        if self.consistent and seeds is not None and len(seeds):
            ids=torch.randint(len(seeds),(64,),device=seeds.device)
            self.add((seeds[ids]+torch.randn(64,20,device=h.device)*.04).clamp(h.low,h.high))
        count=256 if self.consistent else 64
        q=h.low+torch.rand(count,20,device=h.device)*(h.high-h.low)
        if self.consistent:
            energy,depth,_=self.teacher.values(h.frames(q))
            ids=self.stratified_indices(depth*1000,energy,64)
            q=q[ids]
        self.add(q)

    @staticmethod
    def stratified_indices(depth,energy,count):
        """Training-only strata; no held-out queries enter replay."""
        masks=((depth<=.5)&(energy<1.e-4),(depth<=.5)&(energy>=1.e-4),(depth>.5)&(depth<=5),depth>5)
        ids=[]
        for i,mask in enumerate(masks):
            pool=mask.nonzero().flatten()
            if not len(pool):pool=torch.arange(len(depth),device=depth.device)
            n=count//4+int(i<count%4)
            ids.append(pool[torch.randint(len(pool),(n,),device=pool.device)])
        return torch.cat(ids)

    def update(self):
        if self.structured:return dict(q_structured_sdf_fk=1.)
        if not self.size:raise ValueError('Q collision replay is empty')
        if self.group_sobolev:
            from .cart_optim import scale_learning_rates
            scale_learning_rates(self.optimizer,max(.0625,.5**(self.steps//5000)))
        count=min(self.batch_size,self.size)
        if self.consistent:
            ids=self.stratified_indices(self.replay['native_depth_mm'][:self.size],
                self.replay['amplitude'][:self.size].square().sum(-1),count)
        else:ids=torch.randint(self.size,(count,))
        t={k:v[ids].to(self.model.lower.device) for k,v in self.replay.items()}
        x=t['q'].requires_grad_();parts=self.model.components(x)
        amplitude=parts[...,0];depth=self.model.aggregate_depth(parts[...,1])
        potential=(amplitude.square().sum(-1)+.01).sqrt()
        g=torch.autograd.grad(potential.sum(),x,create_graph=True,retain_graph=True)[0]
        dg=torch.autograd.grad(depth.sum(),x,create_graph=True,retain_graph=True)[0]
        cap=(parts[...,2].square().sum(-1)+.0001).sqrt()
        cg=torch.autograd.grad(cap.sum(),x,create_graph=True)[0]
        fit=F.smooth_l1_loss(amplitude/5,t['amplitude']/5)
        depth_fit=F.smooth_l1_loss(depth/.5,t['depth_mm']/.5)+F.smooth_l1_loss(parts[...,1]/.5,t['depth_groups_mm']/.5)
        grad_fit=F.smooth_l1_loss(g/10,t['gradient']/10)
        depth_grad_fit=F.smooth_l1_loss(dg/10,t['depth_gradient']/10)
        active=t['gradient'].norm(dim=-1)>1.e-3
        direction=(1-F.cosine_similarity(g[active],t['gradient'][active])).mean() if active.any() else fit*0
        penetrating=(t['depth_mm']>.2)&(t['depth_gradient'].norm(dim=-1)>1.e-3)
        depth_direction=(1-F.cosine_similarity(dg[penetrating],t['depth_gradient'][penetrating])).mean() if penetrating.any() else fit*0
        cap_fit=F.smooth_l1_loss(parts[...,2]/5,t['cap_amplitude']/5)
        cap_grad_fit=F.smooth_l1_loss(cg/10,t['cap_gradient']/10)
        cap_active=t['cap_gradient'].norm(dim=-1)>1.e-3
        cap_direction=(1-F.cosine_similarity(cg[cap_active],t['cap_gradient'][cap_active])).mean() if cap_active.any() else fit*0
        loss=fit+depth_fit+grad_fit+.2*depth_grad_fit+.2*direction+cap_fit+cap_grad_fit+.5*cap_direction
        if self.consistent:
            def discrepancy(a,b,mask,scale=10.,support=None):
                error=F.smooth_l1_loss(a/scale,b/scale,reduction='none')
                if support is not None:error=(error*support).sum(-1)/support.sum(-1).clamp_min(1)
                else:error=error.mean(-1)
                return error[mask].mean() if mask.any() else a.sum()*0
            active &= t['gradient_valid'].bool();penetrating &= t['depth_gradient_valid'].bool()
            cap_active &= t['cap_gradient_valid'].bool()
            grad_fit=discrepancy(g,t['gradient'],active)
            cap_grad_fit=discrepancy(cg,t['cap_gradient'],cap_active)
            depth_grad_fit=discrepancy(dg,t['depth_gradient'],penetrating)
            direction=(1-F.cosine_similarity(g[active],t['gradient'][active])).mean() if active.any() else fit*0
            cap_direction=(1-F.cosine_similarity(cg[cap_active],t['cap_gradient'][cap_active])).mean() if cap_active.any() else fit*0
            loss=fit+depth_fit+grad_fit+.2*depth_grad_fit+.2*direction+cap_fit+cap_grad_fit+.5*cap_direction
        group_fit=fit*0
        if self.group_sobolev:
            gg=torch.autograd.grad((amplitude*t['group_probe']).sum(),x,create_graph=True,retain_graph=True)[0]
            gcg=torch.autograd.grad((parts[...,2]*t['group_probe']).sum(),x,create_graph=True)[0]
            group_fit=F.smooth_l1_loss(gg/10,t['group_gradient']/10)+F.smooth_l1_loss(gcg/10,t['cap_group_gradient']/10)
            if self.consistent:
                group=t['group_probe'].argmax(-1)
                support=torch.zeros_like(x).scatter_add(1,self.model.joint_ids[group],self.model.mask[group]).clamp_max(1)
                ga=(t['group_gradient'].norm(dim=-1)>1.e-3)&t['group_gradient_valid'].bool()
                gp=(t['cap_group_gradient'].norm(dim=-1)>1.e-3)&t['cap_group_gradient_valid'].bool()
                group_fit=discrepancy(gg,t['group_gradient'],ga,support=support)+discrepancy(gcg,t['cap_group_gradient'],gp,support=support)
                if ga.any():group_fit=group_fit+.5*(1-F.cosine_similarity(gg[ga],t['group_gradient'][ga])).mean()
                if gp.any():group_fit=group_fit+(1-F.cosine_similarity(gcg[gp],t['cap_group_gradient'][gp])).mean()
            loss=loss+group_fit+.5*direction+cap_direction
        if not torch.isfinite(loss):raise ValueError('Nonfinite Q collision training loss')
        self.optimizer.zero_grad();loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),10.,error_if_nonfinite=True)
        self.optimizer.step();self.steps+=1
        report={k:float(v.detach()) for k,v in dict(q_total=loss,q_value=fit,q_depth=depth_fit,
            q_jacobian=grad_fit,q_depth_jacobian=depth_grad_fit,q_direction=direction,q_depth_direction=depth_direction,
            q_cap=cap_fit,q_cap_jacobian=cap_grad_fit,q_cap_direction=cap_direction,q_group_jacobian=group_fit,q_grad_norm=norm).items()}
        if self.consistent:
            report.update({f'q_label_{k}_fraction':float(t[k].float().mean()) for k in ('gradient_valid','cap_gradient_valid','depth_gradient_valid','group_gradient_valid','cap_group_gradient_valid')})
            report.update(q_batch_deep_fraction=float((t['native_depth_mm']>5).float().mean()),q_batch_boundary_fraction=float((t['native_depth_mm']<=.5).float().mean()))
        return report

    def state_dict(self):
        if self.structured:return dict(kind='structured_sdf_fk_no_optimizer_v1')
        return dict(replay={k:v[:self.size].clone() for k,v in self.replay.items()},size=self.size,
            cursor=self.cursor,steps=self.steps,capacity=self.capacity,batch_size=self.batch_size,optimizer=self.optimizer.state_dict())

    def load_state_dict(self,state,*,load_optimizer=True):
        if self.structured:
            if state!={'kind':'structured_sdf_fk_no_optimizer_v1'}:raise ValueError('Cannot resume a learned q decoder as a structured SDF/FK field')
            return
        if state['capacity']!=self.capacity or state['batch_size']!=self.batch_size:raise ValueError('Q collision replay layout differs')
        self.size=state['size'];self.cursor=state['cursor'];self.steps=state['steps']
        for k,v in state['replay'].items():self.replay[k][:self.size].copy_(v)
        if load_optimizer:self.optimizer.load_state_dict(state['optimizer'])

    def validate(self,q):
        with torch.enable_grad():
            truth=self.teacher.label(q)
            x=q.detach().requires_grad_();parts=self.model.components(x)
            amp=(parts[...,0].square().sum(-1)+.01).sqrt()
            depth=self.model.aggregate_depth(parts[...,1])
            g=torch.autograd.grad(amp.sum(),x,retain_graph=True)[0]
            dg=torch.autograd.grad(depth.sum(),x,retain_graph=True)[0]
            cap=(parts[...,2].square().sum(-1)+.0001).sqrt()
            cg=torch.autograd.grad(cap.sum(),x)[0]
            active=truth['gradient'].norm(dim=-1)>1.e-3
            inside=truth['depth_mm']>.5
            deep=truth['depth_mm']>5.
            cosine=F.cosine_similarity(g,truth['gradient'])
            dc=F.cosine_similarity(dg,truth['depth_gradient'])
            cc=F.cosine_similarity(cg,truth['cap_gradient'])
            escape=self.model.escape(q)
            moved=(q+.025*escape['clearance_direction_rad']).clamp(self.model.lower,self.model.upper)
            after=self.teacher.label(moved)
            baseline=(truth['amplitude'].square().sum(-1)+.01).sqrt()
            new=(after['amplitude'].square().sum(-1)+.01).sqrt()
            combined_moved=(q+.025*escape['direction_rad']).clamp(self.model.lower,self.model.upper)
            combined_after=self.teacher.label(combined_moved)
            def guard(t):return t['cap_amplitude'].square().sum(-1).add(1.e-8).sqrt().sub(5).clamp_min(0).square()
            true_total=1.5**2*truth['amplitude'].square().sum(-1)+40**2*guard(truth)
            next_total=1.5**2*combined_after['amplitude'].square().sum(-1)+40**2*guard(combined_after)
        return dict(samples=len(q),collision_cases=int(inside.sum()),
            clearance_amplitude_mae_mm=float((parts[...,0]-truth['amplitude']).detach().abs().mean()),
            depth_mae_mm=float((depth-truth['depth_mm']).detach().abs().mean()),
            depth_p95_mm=float(torch.quantile((depth-truth['depth_mm']).detach().abs(),.95)),
            false_negative_rate=float((depth[inside]<.5).float().mean()) if inside.any() else None,
            gradient_cosine_mean=float(cosine[active].mean()),gradient_opposed_rate=float((cosine[active]<0).float().mean()),
            depth_gradient_cosine_mean=float(dc[inside].mean()) if inside.any() else None,
            cap_cases=int(deep.sum()),cap_gradient_cosine_mean=float(cc[deep].mean()) if deep.any() else None,
            cap_amplitude_mae_mm=float((parts[...,2]-truth['cap_amplitude']).detach().abs().mean()),
            combined_escape_cost_descent_rate=float((next_total[active]<true_total[active]).float().mean()),
            combined_escape_cap_descent_rate=float((combined_after['cap_amplitude'][deep].square().sum(-1)<truth['cap_amplitude'][deep].square().sum(-1)).float().mean()) if deep.any() else None,
            combined_escape_depth_descent_rate=float((combined_after['depth_mm'][inside]<truth['depth_mm'][inside]-1.e-4).float().mean()) if inside.any() else None,
            escape_cost_descent_rate=float((new[active]<baseline[active]).float().mean()),
            escape_depth_descent_rate=float((after['depth_mm'][inside]<truth['depth_mm'][inside]-1.e-4).float().mean()) if inside.any() else None)


def attach_field(collision,spec,state=None):
    collision.model.q_collision=QCollisionField(collision.hand,spec)
    collision.q_collected=[]
    if state is not None:collision.model.q_collision.load_state_dict(state,strict=True)


def collision_cost(objective,q,rows,frames):
    """Q-only self collision; (q, wrist-point, link) SDF for active glove points."""
    c=objective.collision;s=objective.card
    contact_groups=objective.contact_collision_groups(rows) if hasattr(objective,'contact_collision_groups') else None
    if c.auditing:
        if not hasattr(c,'q_audit_teacher'):c.q_audit_teacher=QCollisionTeacher(c)
        energy,depth,cap=c.q_audit_teacher.values(frames,contact_groups)
    else:energy,depth,cap=c.model.q_collision(q,contact_groups)
    tactile,links,counts=c.oracle.flat_tactile_queries(frames,rows)
    ids=torch.arange(len(q),device=q.device).repeat_interleave(torch.as_tensor(counts,device=q.device))
    values=c.distance(tactile,links,configuration=q,pose_ids=ids[:,None].expand_as(links),frames=frames)
    penalties=s.wobj_pseudo**2*(-values*1000-s.allowance_mm).clamp_min(0).square().sum(-1)
    cost_object=q.new_zeros(len(q)).index_add(0,ids,penalties)
    temperature=getattr(c.model,'smooth_temperature_mm',0.)
    object_depth=q.new_zeros(len(q))
    if temperature and not c.auditing:
        # Upper smooth maximum over ALL queried links/taxels plus zero. Fixed
        # ragged lengths depend only on observed contacts, never learned values.
        pieces=torch.split(-values.reshape(-1)*1000,[n*values.shape[-1] for n in counts])
        padded=torch.nn.utils.rnn.pad_sequence([torch.cat((q.new_zeros(1),x)) for x in pieces],batch_first=True,padding_value=-torch.inf)
        object_depth=temperature*torch.logsumexp(padded/temperature,-1)/1000
        depth=temperature*torch.logsumexp(torch.stack((depth,object_depth),-1)*1000/temperature,-1)/1000
    else:
        if len(values):object_depth=object_depth.scatter_reduce(0,ids,(-values).clamp_min(0).amax(-1),reduce='amax',include_self=True)
        depth=torch.maximum(depth,object_depth)
    cap=cap.add(1.e-8).sqrt().sub(5).clamp_min(0).square()+((object_depth-.005)*1000).clamp_min(0).square()
    return s.wself**2*energy,cost_object,depth,objective.settings.penetration_cap_weight**2*cap
