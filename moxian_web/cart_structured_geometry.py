"""Smooth fixed-shape contact features and SDF/FK self-collision geometry.

No closest-face/link query or winner selection appears in these forward paths.
Contact evaluates every fixed feature through a softmin. Self collision moves
fixed material samples by exact FK and differentiates the learned distance field.
"""
import weakref
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

KIND='fk_intrinsic_sdf_smooth_features_v5'
Q_KIND='q_collision_sdf_fk_v1'


class SmoothPadProjection(nn.Module):
    fixed_geometry_projection=True

    def __init__(self,hand,band_m=5.e-6,normal_temperature_m=5.e-5,chunk_size=16):
        super().__init__()
        from .cart_contact_field import PAD_OWNERS
        self.band_m=band_m;self.normal_temperature_m=normal_temperature_m;self.chunk_size=chunk_size
        for owner in PAD_OWNERS:
            tri=np.asarray(hand.surface.contact_surface(owner)['triangles'])
            self.register_buffer('triangles_'+str(owner),torch.as_tensor(tri,dtype=torch.float64,device=hand.device))

    def features(self,points,triangles):
        # Near-equal feature energies require float64: float32 rounding in a
        # micrometre softmin can otherwise create spurious projection gradients.
        x=points.double();tri=triangles.double();a,b,c=tri.unbind(1)
        cross=torch.linalg.cross(b-a,c-a);area2=cross.norm(dim=-1)
        normal=cross/area2.clamp_min(1.e-30)[:,None]
        plane=x[:,None]-((x[:,None]-a)*normal).sum(-1)[...,None]*normal
        edges=torch.stack((b-a,c-b,a-c),1);lengths=edges.norm(dim=-1).clamp_min(1.e-30)
        inward=torch.linalg.cross(normal[:,None].expand_as(edges),edges)/lengths[...,None]
        inside=((plane[:,:,None]-tri)*inward).sum(-1)
        altitude=area2[:,None]/lengths
        # Barycentric violation scaled by the longest edge also bounds skinny
        # triangles. Half-space distance alone incorrectly admits their remote
        # plane extensions as valid projection candidates.
        violation=inside/altitude.clamp_min(1.e-30)*lengths.amax(-1)[None,:,None]
        penalty=100*(F.softplus(-violation/self.band_m)*self.band_m).square().sum(-1)
        parameter=((x[:,None,None]-tri)*edges).sum(-1)/lengths.square()
        beta=self.band_m/lengths
        fraction=beta*(F.softplus(parameter/beta)-F.softplus((parameter-1)/beta))
        edge=tri+fraction[...,None]*edges
        vertex=tri[None].expand(len(x),-1,-1,-1)
        candidates=torch.cat((plane[:,:,None],edge,vertex),2)
        energy=(x[:,None,None]-candidates).square().sum(-1)
        energy=energy+torch.cat((penalty[:,:,None],torch.zeros_like(energy[:,:,1:])),2)
        weights=torch.softmax(-energy.flatten(1)/self.band_m**2,-1).reshape_as(energy)
        hit=(weights[...,None]*candidates).sum((1,2))
        # Normal transitions use a wider, separate distance temperature to
        # avoid sharpening the orientation cost to the projection's tiny scale.
        nw=torch.softmax(-(energy+1.e-24).sqrt().flatten(1)/self.normal_temperature_m,-1).reshape_as(energy)
        n=F.normalize((nw.sum(-1)[...,None]*normal).sum(1),dim=-1,eps=1.e-15)
        return hit.to(points.dtype),n.to(points.dtype)

    def forward(self,points,links):
        shape=points.shape;x=points.reshape(-1,3);owners=links.reshape(-1)
        hit=torch.zeros_like(x);normal=torch.zeros_like(x)
        for owner in owners.unique().tolist():
            tri=getattr(self,'triangles_'+str(owner))
            for ids in (owners==owner).nonzero().flatten().split(self.chunk_size):
                query=x[ids]
                if torch.is_grad_enabled() and query.requires_grad:
                    y,n=checkpoint(self.features,query,tri,use_reentrant=False,preserve_rng_state=False)
                else:y,n=self.features(query,tri)
                hit=hit.index_copy(0,ids,y);normal=normal.index_copy(0,ids,n)
        return hit.reshape(shape),normal.reshape(shape)


class StructuredQCollision(nn.Module):
    """q-only interface; geometric dependence is supplied by FK and point SDF."""
    structured=True

    def __init__(self,hand,distance_model,spec):
        super().__init__()
        from .cart_collision import MeshOracle
        from .cart_q_collision import QCollisionTeacher
        import warp as wp
        self.spec=spec;self.hand=hand
        # The parent owns the SDF parameters. Do not register a second copy or
        # create a module cycle in checkpoints/optimizers.
        object.__setattr__(self,'_distance_ref',weakref.ref(distance_model))
        self.register_buffer('lower',hand.low.clone());self.register_buffer('upper',hand.high.clone())
        self.register_buffer('collar',wp.to_torch(hand.collision.collar).clone())
        oracle=MeshOracle(hand)
        for name,dense in [('clear',False),('cap',True)]:
            if dense:
                oracle.surface=oracle.points;oracle.surface_owners=oracle.owners
                oracle.query_sources,oracle.query_targets=oracle.allowed[oracle.owners].nonzero(as_tuple=True)
            for key,value in dict(points=oracle.surface,owners=oracle.surface_owners,
                    sources=oracle.query_sources,targets=oracle.query_targets,
                    groups=QCollisionTeacher.group_indices(oracle)).items():
                self.register_buffer(name+'_'+key,value.clone())
        total=torch.cuda.get_device_properties(hand.device).total_memory if str(hand.device).startswith('cuda') else 0
        self.chunk_size=32 if total>40*1024**3 else 4

    def sampled_distances(self,frames,prefix):
        p,r=frames[:2];points=getattr(self,prefix+'_points');owners=getattr(self,prefix+'_owners')
        a=getattr(self,prefix+'_sources');b=getattr(self,prefix+'_targets');source=owners[a]
        material=points[a]
        world=(r[:,source]@material[None,:,:,None]).squeeze(-1)+p[:,source]
        local=(r[:,b].transpose(-1,-2)@(world-p[:,b])[...,None]).squeeze(-1)
        field=self._distance_ref()
        if field is None:raise RuntimeError('Structured collision SDF owner was released')
        distance=field(local.reshape(-1,3),b.expand(len(p),-1).reshape(-1)).reshape(len(p),-1)
        # The only moving mechanical exclusion is a tiny collar transition.
        # Smooth it instead of introducing a q-dependent boolean search mask.
        collar=(source==0)&(self.collar[b]>0)
        radius=(local.square().sum(-1)+1.e-24).sqrt()
        log_weight=torch.where(collar[None],F.logsigmoid((radius-self.collar[b][None])/2.e-6),0.)
        reverse=(b==0)&(self.collar[source]>0)&(material.norm(dim=-1)<=self.collar[source])
        log_weight=torch.where(reverse[None],-torch.inf,log_weight)
        return distance,log_weight.exp(),log_weight

    def _components(self,q):
        frames=self.hand.frames(q);d,w,_=self.sampled_distances(frames,'clear')
        energy=q.new_zeros(len(q),15).scatter_add(1,self.clear_groups[None].expand(len(q),-1),
            w*(6-d*1000).clamp_min(0).square())
        d,w,log_w=self.sampled_distances(frames,'cap')
        cap=q.new_zeros(len(q),15).scatter_add(1,self.cap_groups[None].expand(len(q),-1),
            w*(-d*1000).clamp_min(0).square())
        # Smooth group maxima, including zero; log(weight) continuously removes
        # excluded material points without an argmin/argmax over distances.
        scores=-d*1000/.05+log_w
        depth=torch.stack([.05*torch.logsumexp(torch.cat((q.new_zeros(len(q),1),scores[:,self.cap_groups==g]),-1),-1) for g in range(15)],-1)
        return torch.stack(((energy+1.e-8).sqrt(),depth,(cap+1.e-8).sqrt()),-1)

    def components(self,q):
        values=[]
        for batch in q.split(self.chunk_size):
            if torch.is_grad_enabled() and batch.requires_grad:
                values.append(checkpoint(self._components,batch,use_reentrant=False,preserve_rng_state=False))
            else:values.append(self._components(batch))
        return torch.cat(values)

    def aggregate_depth(self,depth):return .05*torch.logsumexp(depth/.05,-1)

    def forward(self,q,contact_groups=None):
        parts=self.components(q);energy=parts[...,0].square()
        if contact_groups is not None:energy=torch.where(contact_groups,parts[...,2].square(),energy)
        return energy.sum(-1),self.aggregate_depth(parts[...,1])/1000,parts[...,2].square().sum(-1)

    def escape(self,q):
        # Same cost-to-direction rule and joint-domain treatment as the learned
        # q-only field, now differentiating its structured geometry directly.
        from .cart_q_collision import QCollisionField
        return QCollisionField.escape(self,q)
