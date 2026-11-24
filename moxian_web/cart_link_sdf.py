"""Independent rigid-link SDFs; all articulation is exact differentiable FK.

No contact projection, nearest-link reduction, witness caching or custom
backward is used by the policy. Native mesh queries are detached teachers and
independent validation only. Link identities are fixed observation labels.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

KIND='fk_independent_link_sdf_distance_contacts_v6'
OBJECTIVE='card_style_link_sdf_contacts_v6_wrist'
VALIDATION='link_sdf_distance_gradient_v1'


class IndependentLinkDistance(nn.Module):
    def __init__(self,width=128,depth=3):
        super().__init__()
        self.decoders=nn.ModuleList()
        for _ in range(21):
            layers=[nn.Linear(3,width),nn.Softplus(beta=10)]
            for _ in range(depth-1):layers.extend((nn.Linear(width,width),nn.Softplus(beta=10)))
            layers.append(nn.Linear(width,1));self.decoders.append(nn.Sequential(*layers))

    def forward(self,points,links):
        shape=points.shape[:-1];x=points.reshape(-1,3);owners=links.expand(shape).reshape(-1)
        if not len(x):return x.new_empty(shape)
        # Grouped dense GEMMs evaluate independent parameter matrices together.
        # Integer sorting is solely by fixed supplied link IDs, not proximity.
        active,compact=torch.unique(owners,sorted=True,return_inverse=True)
        active_ids=active.tolist()
        if len(active_ids)==1:return (.05*self.decoders[active_ids[0]](x/.05).squeeze(-1)).reshape(shape)
        order=torch.argsort(compact,stable=True);sorted_owner=compact[order]
        counts=torch.bincount(compact,minlength=len(active_ids));width=int(counts.max())
        offset=counts.cumsum(0)-counts
        slot=torch.arange(len(x),device=x.device)-offset[sorted_owner]
        route=sorted_owner*width+slot
        value=x.new_zeros((len(active_ids)*width,3)).index_copy(0,route,x[order]/.05).reshape(len(active_ids),width,3)
        layers=len(self.decoders[0])
        for layer in range(0,layers,2):
            weight=torch.stack([self.decoders[i][layer].weight for i in active_ids])
            bias=torch.stack([self.decoders[i][layer].bias for i in active_ids])
            value=torch.bmm(value,weight.transpose(1,2))+bias[:,None]
            if layer!=layers-1:value=F.softplus(value,beta=10)
        result=x.new_zeros(len(x)).index_copy(0,order,.05*value.reshape(-1)[route])
        return result.reshape(shape)


def geometry_spec(hand):
    from .cart_pretrained_sdf import geometry_spec as physical_spec
    from .cart_q_collision import field_spec
    return dict(kind=KIND,geometry=physical_spec(hand),learning_rate=1.e-3,width=128,depth=3,
        vector_lr_scale=.1,contact_surface_policy=hand.surface.contact_policy,
        query='specified_link_inverse_exact_FK_then_independent_local_signed_distance',
        output='one_signed_distance_in_metres; Jacobians_are_actual_autograd_derivatives',
        initialization='random_no_checkpoint',q_collision=field_spec(hand,structured=True),
        contact_projection=False,nearest_search_in_policy=False,
        supervision='balanced_links_area_weighted_surface_and_box; distance_and_stable_spatial_derivative',
        geometry_validation=VALIDATION)


class LinkSDFGeometry(nn.Module):
    direct_link_sdf=True
    intrinsic=True
    def __init__(self,hand,spec):
        super().__init__()
        if spec!=geometry_spec(hand):raise ValueError('Link SDF geometry/architecture changed')
        self.hand=hand;self.spec=spec
        self.pretrained=IndependentLinkDistance(spec['width'],spec['depth']).to(hand.device)
        self.pretrained_lr=spec['learning_rate'];self.vector_lr_scale=spec['vector_lr_scale']
        from .cart_structured_geometry import StructuredQCollision
        # Supplies fixed material samples and mechanical exclusions only.
        self.q_collision=StructuredQCollision(hand,self.pretrained,spec['q_collision'])

    def forward(self,points,links):return self.pretrained(points,links)
    def training_distance(self,points,links,*,validation=False):return self(points,links)
    def at_frames(self,local,links,poses,q,native_p,native_r):return self(local,links)
    def contact(self,*args,**kwargs):raise RuntimeError('Distance objective must not call a projection field')

    def distances(self,local,links):
        shape=local.shape[:-1];x=local.reshape(-1,3);l=links.expand(shape).reshape(-1)
        values=[]
        for a,b in zip(x.split(262144),l.split(262144)):
            if torch.is_grad_enabled() and a.requires_grad:
                values.append(checkpoint(self.pretrained,a,b,use_reentrant=False,preserve_rng_state=False))
            else:values.append(self.pretrained(a,b))
        return torch.cat(values).reshape(shape) if values else local.new_empty(shape)

    def self_terms(self,q,*,oracle=None):
        """All fixed allowed link pairs, 54 native guard samples per link.

        Every pair resists penetration, including intended contacts. Intended
        attraction is added only at the observed pair's common contact anchor.
        No finger-wide collision exemption or nearest-pair winner is used.
        """
        results=[]
        for state in q.split(8):
            def evaluate(s):
                sampler=self.q_collision
                p,r=self.hand.frames(s)[:2];a=sampler.cap_sources;b=sampler.cap_targets
                src=sampler.cap_owners[a];material=sampler.cap_points[a]
                world=(r[:,src]@material[None,:,:,None]).squeeze(-1)+p[:,src]
                local=(r[:,b].transpose(-1,-2)@(world-p[:,b])[...,None]).squeeze(-1)
                if oracle is None:d=self.distances(local,b[None])
                else:
                    native,n=oracle.label(local.reshape(-1,3),b.expand(len(s),-1).reshape(-1))
                    # Only validation uses this detached teacher linearization.
                    d=native.reshape(local.shape[:-1])+((local-local.detach())*n.reshape_as(local)).sum(-1)
                collar=(src==0)&(sampler.collar[b]>0)
                radius=(local.square().sum(-1)+1.e-24).sqrt()
                weight=torch.where(collar[None],torch.sigmoid((radius-sampler.collar[b][None])/2.e-6),1.)
                reverse=(b==0)&(sampler.collar[src]>0)&(material.norm(dim=-1)<=sampler.collar[src])
                weight=torch.where(reverse[None],0.,weight)
                penetration=F.relu(-d*1000)
                repulsion=(weight*penetration.square()).sum(-1)
                cap=(weight*F.relu(penetration-5).square()).sum(-1)
                # Smooth maximum is an upper bound, with excluded points removed.
                logw=torch.where(reverse[None],-torch.inf,F.logsigmoid((radius-sampler.collar[b][None])/2.e-6))
                logw=torch.where(collar[None]|reverse[None],logw,0.)
                scores=-d*1000/.05+logw
                depth=.05*torch.logsumexp(torch.cat((scores,scores.new_zeros(len(s),1)),-1),-1)/1000
                return torch.stack((repulsion,cap,depth),-1)
            if oracle is None and torch.is_grad_enabled() and state.requires_grad:
                results.append(checkpoint(evaluate,state,use_reentrant=False,preserve_rng_state=False))
            else:results.append(evaluate(state))
        return torch.cat(results) if results else q.new_empty((0,3))


def rejection_reasons(report):
    reasons=[]
    minima={'penetration_cases':20,'spatial_gradient_norm_mean':.8,'spatial_gradient_cosine_mean':.9,
        'q_gradient_cosine_mean':.8,'q_escape_cost_descent_rate':.9,'q_cap_gradient_cosine_mean':.8,
        'q_combined_escape_cap_descent_rate':.9,'q_collision_cases':20,'q_cap_cases':20,
        'derivative_label_valid_fraction':.9,'articulated_gradient_cosine_mean':.9}
    maxima={'distance_p95_mm':.5,'penetration_false_negative_rate':.01,'spatial_gradient_norm_mean':1.2,
        'q_false_negative_rate':.01}
    for key,limit in minima.items():
        value=report.get(key)
        if value is None or not __import__('math').isfinite(value) or value<limit:reasons.append(key)
    for key,limit in maxima.items():
        value=report.get(key)
        if value is None or not __import__('math').isfinite(value) or value>limit:reasons.append(key)
    return reasons
