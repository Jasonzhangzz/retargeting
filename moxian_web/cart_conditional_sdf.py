"""Shared q-conditioned multi-link distances, encoded once per configuration.

The network takes canonical fixed-robot-wrist points. It predicts all 21 signed
link distances without a min/search. FK supplies moving source points and native
teachers, while target-link motion is learned through the q condition.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .cart_wrist import so3_exp

KIND='shared_q_encoded_multilink_sdf_v7'
OBJECTIVE='card_conditional_sdf_v7_wrist'
VALIDATION='conditional_sdf_distance_gradient_v2'


class ConditionalDistances(nn.Module):
    def __init__(self,lower,upper,width=256,latent=128):
        super().__init__()
        self.register_buffer('lower',torch.as_tensor(lower).clone())
        self.register_buffer('upper',torch.as_tensor(upper).clone())
        self.q_encoder=nn.Sequential(nn.Linear(60,128),nn.Softplus(beta=10),nn.Linear(128,latent),nn.Softplus(beta=10))
        self.point=nn.Linear(3,width);self.condition=nn.Linear(latent,width,bias=False)
        self.trunk=nn.Sequential(nn.Softplus(beta=10),nn.Linear(width,width),nn.Softplus(beta=10),
                                 nn.Linear(width,width),nn.Softplus(beta=10),nn.Linear(width,21))

    def encode(self,q):
        q=q[...,:20]
        return self.condition(self.q_encoder(torch.cat((2*(q-self.lower)/(self.upper-self.lower)-1,q.sin(),q.cos()),-1)))

    def decode(self,points,condition):
        return .05*self.trunk(self.point(points/.1)+condition)

    def forward(self,q,points):
        z=self.encode(q)
        return self.decode(points,z[:,None] if points.ndim==3 else z)


def geometry_spec(hand):
    from .cart_pretrained_sdf import geometry_spec as physical_spec
    from .cart_q_collision import field_spec
    return dict(kind=KIND,geometry=physical_spec(hand),contact_surface_policy=hand.surface.contact_policy,
        learning_rate=1.e-3,vector_lr_scale=.1,width=256,latent=128,pose_chunk=32,point_chunk=65536,
        input='q20_once_per_pose_and_canonical_fixed_robot_wrist_xyz',output='21_signed_link_distances_m_no_min',
        q_collision=field_spec(hand,structured=True),initialization='random_no_checkpoint',
        contact_projection=False,nearest_search_in_policy=False,geometry_validation=VALIDATION,
        supervision='all_link_distances_spatial_and_q_VJPs_plus_native_clearance_and_guard_cost_gradients',
        self_guard='relu(L2_all_positive_sample_penetrations_mm-5)^2',clearance_mm=6.)


def observed_to_robot(points,state,ids):
    if state.shape[-1]==20:return points
    rotation=so3_exp(state[:,23:26])
    return (rotation[ids].transpose(-1,-2)@(points-state[ids,20:23])[...,None]).squeeze(-1)


def smooth_taxel_depth(values,counts,temperature_mm=.05):
    """Ragged all-link logsumexp with fixed routing and an explicit zero."""
    size=len(counts);maximum=max(counts,default=0)
    if not maximum:return values.new_zeros(size)
    device=values.device;lengths=torch.tensor(counts,device=device)
    ids=torch.arange(size,device=device).repeat_interleave(lengths)
    offsets=lengths.cumsum(0)-lengths
    slot=torch.arange(len(values),device=device)-offsets[ids]
    route=ids[:,None]*(maximum*21+1)+1+slot[:,None]*21+torch.arange(21,device=device)
    scores=values.new_full((size*(maximum*21+1),),-torch.inf)
    zero_ids=torch.arange(size,device=device)*(maximum*21+1)
    scores=scores.index_fill(0,zero_ids,0.).index_copy(0,route.flatten(),(-values*1000/temperature_mm).flatten())
    return temperature_mm*torch.logsumexp(scores.reshape(size,-1),-1)


def norm_penetration_guard(distance_m,weight):
    penetration=F.relu(-distance_m*1000)
    norm=(weight*penetration.square()).sum(-1).add(1.e-8).sqrt()
    return norm,F.relu(norm-5).square()


class ConditionalSDFGeometry(nn.Module):
    direct_link_sdf=True
    conditional_sdf=True
    def __init__(self,hand,spec):
        super().__init__()
        if spec!=geometry_spec(hand):raise ValueError('Conditional SDF architecture/geometry changed')
        self.hand=hand;self.spec=spec
        self.pretrained=ConditionalDistances(hand.low,hand.high,spec['width'],spec['latent']).to(hand.device)
        self.pretrained_lr=spec['learning_rate'];self.vector_lr_scale=spec['vector_lr_scale']
        from .cart_structured_geometry import StructuredQCollision
        # Only fixed samples/exclusions/group IDs are used, never its learned path.
        self.q_collision=StructuredQCollision(hand,self.pretrained,spec['q_collision'])

    def encode(self,q):return self.pretrained.encode(q[...,:20])
    def forward(self,*args):raise RuntimeError('Conditional SDF requires an explicit q/point correspondence')
    def contact(self,*args,**kwargs):raise RuntimeError('Conditional distance objective has no projection field')

    def distances(self,points,ids,encoding):
        values=[]
        for x,i in zip(points.split(self.spec['point_chunk']),ids.split(self.spec['point_chunk'])):
            z=encoding[i]
            if torch.is_grad_enabled() and (x.requires_grad or z.requires_grad):
                values.append(checkpoint(self.pretrained.decode,x,z,use_reentrant=False,preserve_rng_state=False))
            else:values.append(self.pretrained.decode(x,z))
        return torch.cat(values) if values else points.new_empty((0,21))

    def observed(self,state,points,ids,encoding=None):
        return self.distances(observed_to_robot(points,state,ids),ids,self.encode(state) if encoding is None else encoding)

    def local_selected(self,q,points,links):
        p,r=self.hand.frames(q[...,:20])[:2]
        ids=torch.arange(len(q),device=q.device)[:,None]
        return (r[ids,links].transpose(-1,-2)@(points-p[ids,links])[...,None]).squeeze(-1)

    def native_selected(self,q,points,links,oracle,linearize=False):
        local=self.local_selected(q,points,links)
        d,n=oracle.label(local.reshape(-1,3),links.reshape(-1))
        d=d.reshape(links.shape);n=n.reshape_as(local)
        if linearize:d=d+((local-local.detach())*n).sum(-1)
        return d,n

    def native_all(self,q,points,oracle):
        p,r=self.hand.frames(q[...,:20])[:2]
        local=(r[:,None].transpose(-1,-2)@(points[:,:,None]-p[:,None])[...,None]).squeeze(-1)
        owners=torch.arange(21,device=q.device).expand(local.shape[:-1])
        d,n=oracle.label(local.reshape(-1,3),owners.reshape(-1))
        return d.reshape(local.shape[:-1]),n.reshape_as(local)

    def self_terms(self,q,*,oracle=None,encoding=None,contact_pairs=None,details=False):
        if q.shape[-1]!=20:raise ValueError('Self collision requires q20 in the robot wrist frame')
        z=self.encode(q) if encoding is None else encoding
        pairs=q.new_zeros((len(q),21,21),dtype=torch.bool) if contact_pairs is None else contact_pairs
        result=[];chunk=self.spec['pose_chunk']
        for start in range(0,len(q),chunk):
            def evaluate(state,condition,contact):
                sampler=self.q_collision;p,r=self.hand.frames(state)[:2];outputs=[]
                for prefix in ('clear','cap'):
                    material=getattr(sampler,prefix+'_points');owner=getattr(sampler,prefix+'_owners')
                    world=(r[:,owner]@material[None,:,:,None]).squeeze(-1)+p[:,owner]
                    a=getattr(sampler,prefix+'_sources');b=getattr(sampler,prefix+'_targets');src=owner[a]
                    local=(r[:,b].transpose(-1,-2)@(world[:,a]-p[:,b])[...,None]).squeeze(-1)
                    if oracle is None:
                        ids=torch.arange(len(state),device=q.device).repeat_interleave(len(material))
                        all_d=self.distances(world.reshape(-1,3),ids,condition).reshape(len(state),len(material),21)
                        d=all_d[:,a,b]
                    else:
                        native,n=oracle.label(local.reshape(-1,3),b.expand(len(state),-1).reshape(-1))
                        d=native.reshape(local.shape[:-1])+((local-local.detach())*n.reshape_as(local)).sum(-1)
                    collar=(src==0)&(sampler.collar[b]>0)
                    radius=(local.square().sum(-1)+1.e-24).sqrt()
                    logw=torch.where(collar[None],F.logsigmoid((radius-sampler.collar[b][None])/2.e-6),0.)
                    reverse=(b==0)&(sampler.collar[src]>0)&(material[a].norm(dim=-1)<=sampler.collar[src])
                    logw=torch.where(reverse[None],-torch.inf,logw);weight=logw.exp()
                    penetration=F.relu(-d*1000)
                    if prefix=='clear':
                        allowance=torch.where(contact[:,src,b],0.,6.)
                        outputs.append((weight*F.relu(allowance-d*1000).square()).sum(-1))
                    else:
                        norm,guard=norm_penetration_guard(d,weight)
                        depth=.05*torch.logsumexp(torch.cat((-d*1000/.05+logw,d.new_zeros(len(state),1)),-1),-1)/1000
                        outputs.extend((guard,depth,norm))
                return torch.stack(outputs,-1)
            args=(q[start:start+chunk],z[start:start+chunk],pairs[start:start+chunk])
            if oracle is None and torch.is_grad_enabled() and (args[0].requires_grad or args[1].requires_grad):
                result.append(checkpoint(evaluate,*args,use_reentrant=False,preserve_rng_state=False))
            else:result.append(evaluate(*args))
        result=torch.cat(result) if result else q.new_empty((0,4))
        return result if details else result[:,:3]

    def escape(self,q,contact_pairs=None,*,create_graph=False):
        """Direction of the same scalar collision cost, never a vector head.

        Policy BPTT uses self_terms directly; this interface exposes the field
        for diagnostics/optimization. create_graph retains direction derivatives.
        """
        with torch.enable_grad():
            state=q if q.requires_grad else q.detach().requires_grad_()
            terms=self.self_terms(state,contact_pairs=contact_pairs)
            cost=1.5**2*terms[:,0]+40**2*terms[:,1]
            gradient=torch.autograd.grad(cost.sum(),state,create_graph=create_graph)[0]
        return dict(cost=cost,gradient=gradient,direction_rad=-F.normalize(gradient,dim=-1))


def rejection_reasons(report):
    from .cart_link_sdf import rejection_reasons as common
    reasons=common(report)
    value=report.get('q_guard_gradient_cosine_mean')
    if value is None or not __import__('math').isfinite(value) or value<.8:reasons.append('q_guard_gradient_cosine_mean')
    return reasons
