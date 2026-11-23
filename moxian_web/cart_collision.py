"""Online mesh-supervised distance and directional contact fields with exact FK."""
from contextlib import contextmanager
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
import warp as wp
from .cart_contact_field import ContactTeacher


@wp.kernel(enable_backward=False)
def _distance(meshes: wp.array(dtype=wp.uint64), points: wp.array(dtype=wp.vec3),
              links: wp.array(dtype=int), distances: wp.array(dtype=float),
              normals: wp.array(dtype=wp.vec3), valid: wp.array(dtype=int)):
    i=wp.tid(); p=points[i]
    hit=wp.mesh_query_point_sign_winding_number(meshes[links[i]],p,2.)
    if hit.result:
        closest=wp.mesh_eval_position(meshes[links[i]],hit.face,hit.u,hit.v)
        delta=p-closest; length=wp.length(delta)
        distances[i]=hit.sign*length
        n=wp.mesh_eval_face_normal(meshes[links[i]],hit.face)
        if length>1.e-7:n=hit.sign*delta/length
        normals[i]=n;valid[i]=1


@contextmanager
def frozen(model):
    flags=[p.requires_grad for p in model.parameters()]
    try:
        for p in model.parameters():p.requires_grad_(False)
        yield model
    finally:
        for p,flag in zip(model.parameters(),flags):p.requires_grad_(flag)


class MeshOracle:
    def __init__(self,hand):
        self.hand=hand;self.device=hand.device
        self.mesh_ids=hand.collision.mesh_ids
        self.lower=wp.to_torch(hand.collision.lower)
        self.upper=wp.to_torch(hand.collision.upper)
        self.points,self.owners=(wp.to_torch(x) for x in hand.collision.samples[False])
        self.owners=self.owners.long()
        # Equal material samples per link; all mechanical exclusions stay exact.
        selected=[]
        for link in range(21):
            ids=torch.where(self.owners==link)[0]
            selected.extend(ids[torch.linspace(0,len(ids)-1,8,device=ids.device).long()].tolist())
        ids=torch.as_tensor(selected,device=self.device)
        self.surface=self.points[ids];self.surface_owners=self.owners[ids]
        self.surface_by_link=[self.surface[self.surface_owners==i] for i in range(21)]
        cpu_points=self.surface.cpu().tolist();cpu_owners=self.surface_owners.cpu().tolist()
        self.surface_by_link_cpu=[[point for point,owner in zip(cpu_points,cpu_owners) if owner==i] for i in range(21)]
        self.allowed=wp.to_torch(hand.collision.allowed).bool()
        self.collar=wp.to_torch(hand.collision.collar)
        self.query_sources,self.query_targets=self.allowed[self.surface_owners].nonzero(as_tuple=True)

    @torch.no_grad()
    def label(self,points,links):
        points=points.detach().to(device=self.device,dtype=torch.float32).reshape(-1,3).contiguous()
        links=links.detach().to(device=self.device,dtype=torch.int32).reshape(-1).contiguous()
        n=len(points)
        if not n:return points.new_empty(0),points.new_empty((0,3))
        d=wp.zeros(n,dtype=float,device=self.device)
        normals=wp.zeros(n,dtype=wp.vec3,device=self.device);valid=wp.zeros(n,dtype=int,device=self.device)
        wp.launch(_distance,n,inputs=[self.mesh_ids,wp.from_torch(points,dtype=wp.vec3),wp.from_torch(links),d,normals,valid],device=self.device)
        valid=wp.to_torch(valid).bool()
        if not valid.all():raise ValueError('真实 mesh SDF 查询失败，不能作为训练标签')
        return wp.to_torch(d).clone(),wp.to_torch(normals).clone()

    @torch.no_grad()
    def random_queries(self,n):
        links=torch.randint(21,(n,),device=self.device)
        lower,upper=self.lower[links]-.03,self.upper[links]+.03
        points=lower+torch.rand(n,3,device=self.device)*(upper-lower)
        ids=torch.randint(len(self.points),(n,),device=self.device)
        near=torch.rand(n,device=self.device)<.7
        points[near]=self.points[ids[near]]+torch.randn(int(near.sum()),3,device=self.device)*.003
        links[near]=self.owners[ids[near]]
        return points,links

    def self_queries(self,frames):
        p,r=frames[:2];owners=self.surface_owners
        world=(r[:,owners]@self.surface[...,None]).squeeze(-1)+p[:,owners]
        a,b=self.query_sources,self.query_targets
        source=owners[a];points=world[:,a]
        local=(r[:,b].transpose(-1,-2)@(points-p[:,b])[...,None]).squeeze(-1)
        collar=((source==0)&(self.collar[b]>0))[None]&(local.norm(dim=-1)<=self.collar[b][None])
        reverse=((b==0)&(self.collar[source]>0))[None]&(self.surface[a].norm(dim=-1)<=self.collar[source])[None]
        return local,b.expand(len(p),-1),~(collar|reverse)

    def active_clouds(self,frames,rows):
        p=frames[0];clouds=[];counts=[]
        for i,row in enumerate(rows):
            if '_cart_active_points' in row:points=row['_cart_active_points']
            else:
                positions=torch.as_tensor(row['tactile_positions_m'],device=p.device,dtype=p.dtype)
                mask=torch.as_tensor(row['tactile_valid'],device=p.device,dtype=torch.bool)&torch.as_tensor(row['tactile_state'],device=p.device,dtype=torch.bool)
                points=positions[mask]
            clouds.append(points);counts.append(len(points))
        return clouds,counts

    def tactile_queries(self,frames,rows):
        """Only confirmed active taxels; every target link including the palm."""
        local,links,counts=self.flat_tactile_queries(frames,rows)
        return list(zip(local.split(counts),links.split(counts)))

    def flat_tactile_queries(self,frames,rows):
        p,r=frames[:2];clouds,counts=self.active_clouds(frames,rows)
        points=torch.cat(clouds)
        # One FK gather/transform for all ragged point clouds, instead of a
        # sequence of tiny CUDA launches for every window and time step.
        ids=torch.arange(len(rows),device=p.device).repeat_interleave(torch.as_tensor(counts,device=p.device))
        if len(points):local=(r[ids].transpose(-1,-2)@(points[:,None]-p[ids])[...,None]).squeeze(-1)
        else:local=p.new_empty((0,21,3))
        links=torch.arange(21,device=p.device).expand(len(points),-1)
        return local,links,counts

    def sampled_queries(self,frames,rows,budget):
        """Uniform detached label candidates, sampled before dense expansion."""
        p,r=frames[:2];a,b=self.query_sources,self.query_targets
        flat=torch.randint(len(p)*len(a),(2*budget,),device=p.device)
        pose=flat//len(a);selected=flat%len(a);source=a[selected];target=b[selected]
        owners=self.surface_owners[source];material=self.surface[source]
        world=(r[pose,owners]@material[...,None]).squeeze(-1)+p[pose,owners]
        local=(r[pose,target].transpose(-1,-2)@(world-p[pose,target])[...,None]).squeeze(-1)
        collar=(owners==0)&(self.collar[target]>0)&(local.norm(dim=-1)<=self.collar[target])
        reverse=(target==0)&(self.collar[owners]>0)&(material.norm(dim=-1)<=self.collar[owners])
        valid=~(collar|reverse)
        result=[(local[valid][:budget],target[valid][:budget])]
        clouds,counts=self.active_clouds(frames,rows);points=torch.cat(clouds)
        if len(points):
            # Sampling a point and target independently is uniform over the
            # same all-link query set used by the penetration objective.
            ids=torch.randint(len(points),(budget,),device=p.device)
            poses=torch.searchsorted(torch.as_tensor(counts,device=p.device).cumsum(0),ids,right=True)
            links=torch.randint(21,(budget,),device=p.device)
            local=(r[poses,links].transpose(-1,-2)@(points[ids]-p[poses,links])[...,None]).squeeze(-1)
            result.append((local,links))
        return result


class CollisionModel:
    """Geometry loss adapter, replay collection, and independent true-mesh audit."""
    def __init__(self,hand,model=None):
        from .cart_pretrained_sdf import fresh_geometry_spec,restore_geometry
        self.hand=hand;self.model=model if model is not None else restore_geometry(hand,fresh_geometry_spec(hand))
        self.oracle=MeshOracle(hand);self.collected=[];self.contact_collected=[];self.collecting=False;self.auditing=False
        self.q_collected=[]

    def contact(self,local,direction,owner,palmar=False):
        """No native queries, detached witnesses or data-dependent surface IDs."""
        links=torch.full(local.shape[:-1],owner,device=local.device,dtype=torch.long)
        flags=torch.full_like(links,palmar,dtype=torch.bool)
        if local.is_cuda and torch.is_grad_enabled() and (local.requires_grad or direction.requires_grad) and not any(p.requires_grad for p in self.model.parameters()):
            return checkpoint(self.model.contact,local,direction,links,flags,use_reentrant=False,preserve_rng_state=False)
        return self.model.contact(local,direction,links,flags)

    def pair_queries(self,p,r,a,b):
        for source,target in ((a,b),(b,a)):
            material,normal=self.hand.samples(source,False)
            points=(r[:,source,None]@material[None,:,:,None]).squeeze(-1)+p[:,source,None]
            normals=(r[:,source,None]@normal[None,:,:,None]).squeeze(-1)
            local=(r[:,target,None].transpose(-1,-2)@(points-p[:,target,None])[...,None]).squeeze(-1)
            escape=-(r[:,target,None].transpose(-1,-2)@normals[...,None]).squeeze(-1)
            yield source,target,points,normals,local,escape

    def collect_contact(self,local,direction,links,palmar):
        n=local.numel()//3
        if n:
            ids=torch.randint(n,(512,),device=local.device) if n>512 else slice(None)
            self.contact_collected.append(tuple(v.detach().reshape(n,-1)[ids] if v.ndim==local.ndim else v.detach().reshape(n)[ids]
                for v in (local,direction,links,palmar)))

    def distance(self,local,links,valid=None,*,configuration=None,pose_ids=None,frames=None):
        if self.auditing:
            return self.oracle.label(local,links)[0].reshape(links.shape)
        shape=links.shape
        # Chunk the transient allocation while preserving the full autograd graph.
        flat=local.reshape(-1,3);ids=links.reshape(-1)
        chunk=65536 if local.is_cuda else 8192
        recompute=local.is_cuda and torch.is_grad_enabled() and local.requires_grad and not any(p.requires_grad for p in self.model.parameters())
        articulated=hasattr(self.model,'at_frames') and configuration is not None
        if articulated:
            if pose_ids is None or frames is None:raise ValueError('Articulated SDF queries require pose IDs and native frames')
            poses=pose_ids.reshape(-1)
            if poses.shape!=ids.shape:raise ValueError('SDF query/pose correspondence changed')
            def evaluate(x,l,b):
                args=(x,l,b,configuration,frames[3],frames[4])
                return checkpoint(self.model.at_frames,*args,use_reentrant=False,preserve_rng_state=False) if recompute else self.model.at_frames(*args)
            values=torch.cat([evaluate(x,l,b) for x,l,b in zip(flat.split(chunk),ids.split(chunk),poses.split(chunk))]).reshape(shape) if len(flat) else local.new_empty(shape)
        else:
            def evaluate(x,l):
                return checkpoint(self.model,x,l,use_reentrant=False,preserve_rng_state=False) if recompute else self.model(x,l)
            values=torch.cat([evaluate(x,l) for x,l in zip(flat.split(chunk),ids.split(chunk))]).reshape(shape) if len(flat) else local.new_empty(shape)
        if self.collecting:self.collect_local(local,links,valid)
        return values

    def collect_local(self,local,links,valid=None,max_samples=512):
        x=local.detach().reshape(-1,3);l=links.detach().reshape(-1)
        if valid is not None:x=x[valid.flatten()];l=l[valid.flatten()]
        if len(x):
            select=torch.randint(len(x),(max_samples,),device=x.device) if len(x)>max_samples else slice(None)
            self.collected.append((x[select],l[select]))

    def collect_queries(self,q,rows,frames,sample_budget=512):
        """Collect the same moving surfaces/taxels without evaluating any loss."""
        if hasattr(self.model,'q_collision'):
            ids=torch.randperm(len(q),device=q.device)[:128]
            self.q_collected.append(q[ids].detach())
        for points,links in self.oracle.sampled_queries(frames,rows,sample_budget):
            self.collect_local(points,links,max_samples=sample_budget)
        from .cart_contacts import link_owner
        p,r=frames[:2];objects={};pairs={}
        for i,row in enumerate(rows):
            for c in row['contacts']['object_contacts']:
                objects.setdefault(link_owner(c['region']),[]).append((i,c))
            for c in row['contacts']['self_pairs']:
                pairs.setdefault(tuple(map(link_owner,c['regions'])),[]).append(i)
        for owner,items in objects.items():
            if len(items)>512:items=[items[i] for i in torch.randint(len(items),(512,)).tolist()]
            ids=torch.as_tensor([i for i,_ in items],device=q.device)
            points=q.new_tensor([c['position_m'] for _,c in items]);normals=q.new_tensor([c['normal'] for _,c in items])
            inverse=r[ids,owner].transpose(-1,-2)
            local=(inverse@(points-p[ids,owner])[...,None]).squeeze(-1)
            direction=(inverse@normals[...,None]).squeeze(-1)
            links=torch.full((len(ids),),owner,device=q.device,dtype=torch.long)
            self.collect_contact(local,direction,links,torch.ones_like(links,dtype=torch.bool))
        for (a,b),indices in pairs.items():
            if len(indices)>128:indices=[indices[i] for i in torch.randint(len(indices),(128,)).tolist()]
            ids=torch.as_tensor(indices,device=q.device)
            for _,owner,_,_,local,direction in self.pair_queries(p[ids],r[ids],a,b):
                links=torch.full(local.shape[:-1],owner,device=q.device,dtype=torch.long)
                self.collect_contact(local,direction,links,torch.zeros_like(links,dtype=torch.bool))

    @torch.no_grad()
    def audit(self,q,rows,frames):
        depth,valid=self.hand.penetration(q.detach(),frames[3].detach(),frames[4].detach(),dense=True)
        external=[]
        for x,links in self.oracle.tactile_queries(frames,rows):
            if x.numel():d,_=self.oracle.label(x,links);external.append((-d).clamp_min(0).max())
            else:external.append(q.new_zeros(()))
        return torch.maximum(depth,torch.stack(external)),valid


class SDFLearner:
    def __init__(self,collision,capacity=200000,batch_size=2048,optimizer_kind='adamw'):
        self.collision=collision;self.model=collision.model;self.oracle=collision.oracle
        self.q_learner=None
        if hasattr(self.model,'q_collision'):
            from .cart_q_collision import QCollisionLearner
            self.q_learner=QCollisionLearner(collision)
            if hasattr(collision,'q_initial_learner'):
                self.q_learner.load_state_dict(collision.q_initial_learner)
        if optimizer_kind=='muon_sgd':
            from .cart_optim import MuonSGD
            self.optimizer=MuonSGD(self.model,lr=1.e-3,
                pretrained_lr=self.model.pretrained_lr if hasattr(self.model,'pretrained') else None,
                contact_lr=1.e-3,vector_lr_scale=.01 if getattr(self.model,'intrinsic',False) else 1.)
        elif optimizer_kind!='adamw':raise ValueError('Unknown geometry optimizer')
        elif hasattr(self.model,'pretrained'):
            self.optimizer=torch.optim.AdamW([
                dict(params=self.model.pretrained.parameters(),lr=self.model.pretrained_lr),
                dict(params=[*self.model.links.parameters(),*self.model.contact_field.parameters()],lr=1.e-3)])
        else:self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=1.e-3)
        self.capacity=capacity;self.batch_size=batch_size;self.cursor=0;self.size=0;self.steps=0
        self.points=torch.empty(capacity,3);self.links=torch.empty(capacity,dtype=torch.long)
        self.distances=torch.empty(capacity);self.normals=torch.empty(capacity,3)
        self.contact_size=0;self.contact_cursor=0;self._teacher=None
        self.contact_replay={k:torch.empty((capacity,3) if k in ('points','directions','hits','normals') else (capacity,),
            dtype=torch.long if k=='links' else torch.bool if k=='palmar' else torch.float32)
            for k in ('points','directions','links','palmar','hits','normals')}

    @property
    def teacher(self):
        if self._teacher is None:self._teacher=ContactTeacher(self.oracle.hand)
        return self._teacher

    @torch.no_grad()
    def add_contacts(self,points,directions,links,palmar):
        hit,normal,valid=self.teacher.label(points,directions,links,palmar)
        values=[v[valid][-self.capacity:].cpu() for v in (points,directions,links,palmar,hit,normal)]
        count=len(values[0])
        ids=(torch.arange(count)+self.contact_cursor)%self.capacity
        for key,value in zip(self.contact_replay,values):self.contact_replay[key][ids]=value
        self.contact_cursor=(self.contact_cursor+count)%self.capacity
        self.contact_size=min(self.capacity,self.contact_size+count)
        return count

    @torch.no_grad()
    def add(self,points,links):
        points=points.to(self.oracle.device);links=links.to(self.oracle.device)
        d,n=self.oracle.label(points,links)
        count=min(len(points),self.capacity);points=points[-count:];links=links[-count:];d=d[-count:];n=n[-count:]
        ids=(torch.arange(count)+self.cursor)%self.capacity
        self.points[ids]=points.cpu();self.links[ids]=links.cpu();self.distances[ids]=d.cpu();self.normals[ids]=n.cpu()
        self.cursor=(self.cursor+count)%self.capacity;self.size=min(self.capacity,self.size+count)

    def collect(self):
        if self.q_learner is not None:self.q_learner.collect()
        collected=self.collision.collected;self.collision.collected=[]
        if collected:
            x=torch.cat([p for p,_ in collected]);links=torch.cat([l for _,l in collected])
            ids=torch.randperm(len(x))[:self.batch_size]
            x=x[ids];links=links[ids]
            self.add(torch.cat((x,x+torch.randn_like(x)*.001)),links.repeat(2))
        self.add(*self.oracle.random_queries(self.batch_size))
        contacts=self.collision.contact_collected;self.collision.contact_collected=[]
        if contacts:
            values=[torch.cat(x) for x in zip(*contacts)]
            ids=torch.randperm(len(values[0]),device=values[0].device)[:self.batch_size]
            x,d,l,p=[v[ids] for v in values]
            self.add_contacts(torch.cat((x,x+torch.randn_like(x)*.001)),d.repeat(2,1),l.repeat(2),p.repeat(2))
        if not getattr(self.model,'intrinsic',False):self.add_contacts(*self.teacher.random_queries(self.oracle,self.batch_size))
        if getattr(self.collision,'card_nearest_surface',False):
            if getattr(self.model,'intrinsic',False):
                ids=torch.randint(len(self.teacher.points),(self.batch_size,),device=self.oracle.device)
                x=self.teacher.points[ids];l=self.teacher.owners[ids];p=torch.ones_like(l,dtype=torch.bool)
                sigma=torch.where(torch.rand(len(x),1,device=x.device)<.75,.006,.025)
                x=x+torch.randn_like(x)*sigma
            else:
                x,_,l,p=self.teacher.random_queries(self.oracle,self.batch_size)
                x=x+torch.randn_like(x)*.025
            self.add_contacts(x,torch.zeros_like(x),l,p)

    def update(self):
        if not self.size:raise ValueError('SDF replay is empty')
        if getattr(self.model,'intrinsic',False):
            from .cart_optim import scale_learning_rates
            scale_learning_rates(self.optimizer,max(.0625,.5**(self.steps//5000)))
        ids=torch.randint(self.size,(min(self.size,self.batch_size),))
        x=self.points[ids].to(self.oracle.device).requires_grad_();links=self.links[ids].to(self.oracle.device)
        target=self.distances[ids].to(self.oracle.device);normal=self.normals[ids].to(self.oracle.device)
        prediction=self.model.training_distance(x,links) if hasattr(self.model,'training_distance') else self.model(x,links)
        gradient=torch.autograd.grad(prediction.sum(),x,create_graph=True)[0]
        weights=1/(1+(target/.01).square())
        fit=(weights*F.smooth_l1_loss(prediction/.001,target/.001,reduction='none')).mean()
        sign=F.softplus(-torch.sign(target)*prediction/.001)[target.abs()>.0001].mean() if (target.abs()>.0001).any() else fit*0
        near=target.abs()<.005
        direction=(1-F.cosine_similarity(gradient[near],normal[near])).mean() if near.any() else fit*0
        derivative=F.smooth_l1_loss(gradient[near],normal[near]) if near.any() else fit*0
        eikonal=(gradient.norm(dim=-1)-1).square().mean()
        if not self.contact_size:raise ValueError('连续接触场缺少有效真实 mesh 标签')
        ids=torch.randint(self.contact_size,(min(self.contact_size,self.batch_size),))
        c={k:v[ids].to(self.oracle.device) for k,v in self.contact_replay.items()}
        hit,normal=self.model.contact(c['points'],c['directions'],c['links'],c['palmar'])
        contact_fit=F.smooth_l1_loss(hit/.001,c['hits']/.001)
        contact_normal=(1-F.cosine_similarity(normal,c['normals'])).mean()
        intrinsic=getattr(self.model,'intrinsic',False)
        contact_jacobian=fit*0
        if intrinsic and getattr(self.collision,'card_nearest_surface',False):
            # Supervise the projection derivative too: a small position error
            # alone does not establish that policy gradients point correctly.
            points=c['points'][:256];links_c=c['links'][:256];palmar=c['palmar'][:256]
            direction_c=torch.zeros_like(points);step=F.normalize(torch.randn_like(points),dim=-1)*.00005
            target_jacobian,valid=self.teacher.projection_derivative(points,links_c,palmar,step/.00005)
            ph,_=self.model.contact(points+step,direction_c,links_c,palmar)
            mh,_=self.model.contact(points-step,direction_c,links_c,palmar)
            if valid.any():contact_jacobian=F.smooth_l1_loss((ph-mh)[valid]/.0001,target_jacobian[valid])
        loss=fit+.1*sign+(.5 if intrinsic else .1)*direction+.1*eikonal+contact_fit+(.5 if intrinsic else .1)*contact_normal
        if intrinsic:loss=loss+.5*derivative+2.*contact_jacobian
        if not torch.isfinite(loss):raise ValueError('SDF loss 出现 NaN/Inf')
        self.optimizer.zero_grad(set_to_none=True);loss.backward()
        # Do not let the unrelated projection head suppress the SDF update.
        if intrinsic:
            dn=torch.nn.utils.clip_grad_norm_(self.model.pretrained.parameters(),10.,error_if_nonfinite=True)
            cn=torch.nn.utils.clip_grad_norm_([*self.model.links.parameters(),*self.model.contact_field.parameters()],10.,error_if_nonfinite=True)
            grad_norm=torch.maximum(dn,cn)
        else:
            parameters=[p for n,p in self.model.named_parameters() if not n.startswith('q_collision.')]
            grad_norm=torch.nn.utils.clip_grad_norm_(parameters,10.,error_if_nonfinite=True)
        self.optimizer.step();self.steps+=1
        result={k:float(v.detach()) for k,v in dict(total=loss,distance=fit,sign=sign,normal=direction,eikonal=eikonal,
            contact_position=contact_fit,contact_normal=contact_normal,contact_jacobian=contact_jacobian,
            distance_jacobian=derivative,gradient_norm_before_clip=grad_norm,
            gradient_clip_multiplier=(10./(grad_norm+1.e-6)).clamp(max=1.)).items()}
        if self.q_learner is not None:result.update(self.q_learner.update())
        return result

    def state_dict(self):
        result=dict(points=self.points[:self.size].clone(),links=self.links[:self.size].clone(),
            distances=self.distances[:self.size].clone(),normals=self.normals[:self.size].clone(),
            cursor=self.cursor,size=self.size,steps=self.steps,capacity=self.capacity,batch_size=self.batch_size,
            contact_replay={k:v[:self.contact_size].clone() for k,v in self.contact_replay.items()},
            contact_size=self.contact_size,contact_cursor=self.contact_cursor,
            optimizer=self.optimizer.state_dict())
        if self.q_learner is not None:result['q_collision']=self.q_learner.state_dict()
        return result

    def load_state_dict(self,state,*,load_optimizer=True):
        if state['capacity']!=self.capacity or state['batch_size']!=self.batch_size:raise ValueError('SDF replay configuration changed')
        self.size=state['size'];self.cursor=state['cursor'];self.steps=state['steps']
        for key in ('points','links','distances','normals'):getattr(self,key)[:self.size].copy_(state[key])
        self.contact_size=state['contact_size'];self.contact_cursor=state['contact_cursor']
        for key,value in state['contact_replay'].items():self.contact_replay[key][:self.contact_size].copy_(value)
        if load_optimizer:self.optimizer.load_state_dict(state['optimizer'])
        if self.q_learner is not None and 'q_collision' in state:
            self.q_learner.load_state_dict(state['q_collision'],load_optimizer=load_optimizer)

    @torch.no_grad()
    def validate(self,points,links):
        truth,true_normal=self.oracle.label(points,links)
        with torch.enable_grad():
            x=points.detach().requires_grad_()
            pred=self.model.training_distance(x,links,validation=True) if hasattr(self.model,'training_distance') else self.model(x,links)
            spatial=torch.autograd.grad(pred.sum(),x)[0].detach();pred=pred.detach()
        error=(pred-truth).abs();inside=truth<-.0005
        # Generate a fresh, reproducible holdout without consuming training RNG.
        with torch.random.fork_rng(devices=[torch.device(self.oracle.device)] if str(self.oracle.device).startswith('cuda') else []):
            torch.manual_seed(9714)
            x,d,l,p=self.teacher.random_queries(self.oracle,len(points))
        if getattr(self.collision,'card_nearest_surface',False):
            # CARD queries semantic pads only; full-mesh escape modes are not
            # part of this policy's nearest-point precision denominator.
            with torch.random.fork_rng(devices=[torch.device(self.oracle.device)] if str(self.oracle.device).startswith('cuda') else []):
                torch.manual_seed(9715)
                ids=torch.randint(len(self.teacher.points),(len(points),),device=self.oracle.device)
                x=self.teacher.points[ids]+torch.randn_like(self.teacher.points[ids])*.006
                l=self.teacher.owners[ids]
            d=torch.zeros_like(x);p=torch.ones_like(l,dtype=torch.bool)
        hit,normal,valid=self.teacher.label(x,d,l,p);pred_hit,pred_normal=self.model.contact(x,d,l,p)
        step=F.normalize(torch.sin(torch.arange(x.numel(),device=x.device).reshape_as(x)+.4),dim=-1)*.00005
        target_jacobian,stable=self.teacher.projection_derivative(x,l,p,step/.00005)
        ph,_=self.model.contact(x+step,d,l,p);mh,_=self.model.contact(x-step,d,l,p)
        jacobian_error=((ph-mh)/.0001-target_jacobian).norm(dim=-1)
        contact_error=(pred_hit[valid]-hit[valid]).norm(dim=-1)
        normal_error=torch.rad2deg(torch.acos(F.cosine_similarity(pred_normal[valid],normal[valid]).clamp(-1,1)))
        report=dict(distance_mean_mm=float(error.mean())*1000,distance_p95_mm=float(torch.quantile(error,.95))*1000,
            spatial_gradient_norm_mean=float(spatial.norm(dim=-1).mean()),
            spatial_gradient_cosine_mean=float(F.cosine_similarity(spatial,true_normal).mean()),
            penetration_false_negative_rate=float((pred[inside]>=0).float().mean()) if inside.any() else None,
            penetration_cases=int(inside.sum()),samples=len(points),contact_samples=int(valid.sum()),
            contact_position_p95_mm=float(torch.quantile(contact_error,.95))*1000 if valid.any() else None,
            contact_jacobian_error_mean=float(jacobian_error[stable].mean()) if stable.any() else None,
            contact_jacobian_samples=int(stable.sum()),
            contact_normal_p95_deg=float(torch.quantile(normal_error,.95)) if valid.any() else None)
        if getattr(self.collision,'card_nearest_surface',False):
            with torch.random.fork_rng(devices=[torch.device(self.oracle.device)] if str(self.oracle.device).startswith('cuda') else []):
                torch.manual_seed(9717)
                ids=torch.randint(len(self.teacher.points),(len(points),),device=self.oracle.device)
                far=self.teacher.points[ids]+torch.randn_like(self.teacher.points[ids])*.025
                owners=self.teacher.owners[ids];flags=torch.ones_like(owners,dtype=torch.bool)
            exact,_,valid=self.teacher.label(far,torch.zeros_like(far),owners,flags)
            predicted,_=self.model.contact(far,torch.zeros_like(far),owners,flags)
            error_far=(predicted[valid]-exact[valid]).norm(dim=-1)
            probe=F.normalize(torch.cos(torch.arange(far.numel(),device=far.device).reshape_as(far)+.7),dim=-1)
            target_j,ok=self.teacher.projection_derivative(far,owners,flags,probe)
            ph,_=self.model.contact(far+probe*.00005,torch.zeros_like(far),owners,flags)
            mh,_=self.model.contact(far-probe*.00005,torch.zeros_like(far),owners,flags)
            report.update(contact_far_samples=int(valid.sum()),
                contact_far_position_p95_mm=float(torch.quantile(error_far,.95))*1000 if valid.any() else None,
                contact_far_jacobian_error_mean=float(((ph-mh)/.0001-target_j)[ok].norm(dim=-1).mean()) if ok.any() else None)
        if self.q_learner is not None:
            with torch.random.fork_rng(devices=[torch.device(self.oracle.device)] if str(self.oracle.device).startswith('cuda') else []):
                torch.manual_seed(9716)
                h=self.oracle.hand;q=h.low+torch.rand(128,20,device=h.device)*(h.high-h.low)
                report.update({'q_'+k:v for k,v in self.q_learner.validate(q).items()})
        return report


def geometry_rejection_reasons(report):
    """Physical gates for a geometry surrogate, independent of policy loss."""
    reasons=[]
    def maximum(name,limit):
        value=report.get(name)
        if value is None or not torch.isfinite(torch.tensor(value)) or value>limit:reasons.append(name)
    def minimum(name,limit):
        value=report.get(name)
        if value is None or not torch.isfinite(torch.tensor(value)) or value<limit:reasons.append(name)
    maximum('distance_p95_mm',.5);minimum('penetration_cases',20)
    maximum('penetration_false_negative_rate',.01)
    minimum('spatial_gradient_norm_mean',.8);maximum('spatial_gradient_norm_mean',1.2)
    minimum('spatial_gradient_cosine_mean',.9)
    minimum('contact_samples',20);maximum('contact_position_p95_mm',.5)
    maximum('contact_normal_p95_deg',30.)
    minimum('contact_jacobian_samples',20);maximum('contact_jacobian_error_mean',.25)
    minimum('contact_far_samples',20);maximum('contact_far_position_p95_mm',1.)
    maximum('contact_far_jacobian_error_mean',.25)
    minimum('q_collision_cases',20);maximum('q_false_negative_rate',.01)
    minimum('q_gradient_cosine_mean',.8);minimum('q_escape_cost_descent_rate',.9)
    minimum('q_cap_cases',20);minimum('q_cap_gradient_cosine_mean',.8)
    minimum('q_combined_escape_cost_descent_rate',.9)
    minimum('q_combined_escape_cap_descent_rate',.9)
    return reasons
