"""Load the existing articulated Neural SDF without changing its q input.

CART contact heads stay in link coordinates. Distance queries are transformed
back to the checkpoint's native fixed-wrist frame and use the actual rollout q.
"""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .cart_data import file_hash
from .cart_contact_field import ContactField,IndependentPadField,PadPotentialField,PAD_OWNERS
from .cart_structured_geometry import KIND as STRUCTURED_KIND

KIND='configuration_sdf_v1'
INTRINSIC_KIND='fk_intrinsic_sdf_v1'
INDEPENDENT_KIND='fk_intrinsic_sdf_independent_decoders_v2'
POTENTIAL_KIND='fk_intrinsic_sdf_pad_potential_v3'
CONSISTENT_KIND='fk_intrinsic_sdf_verified_pad_v4'


class IntrinsicDistance(nn.Module):
    """One rigid surface field per semantic link, articulated by exact FK.

    Learning an invariant local surface avoids asking an MLP to rediscover FK
    from independently resampled joint configurations at every update.
    """
    def __init__(self):
        super().__init__()
        self.links=nn.Embedding(21,16)
        self.network=nn.Sequential(nn.Linear(19,256),nn.Softplus(beta=10),
            nn.Linear(256,256),nn.Softplus(beta=10),nn.Linear(256,256),nn.Softplus(beta=10),nn.Linear(256,1))

    def forward(self,points,links):
        return .05*self.network(torch.cat((points/.05,self.links(links.long())),-1)).squeeze(-1)


class ConfigurationDistance(nn.Module):
    """Checkpoint-compatible q/x/link MLP; no analytic FK inside the MLP."""
    def __init__(self,lower,upper,joint_mask,width=256,depth=4,embedding=16):
        super().__init__()
        if width<8 or depth<2 or embedding!=16:raise ValueError('Unsupported pretrained SDF architecture')
        self.register_buffer('lower',torch.tensor(lower,dtype=torch.float32))
        self.register_buffer('upper',torch.tensor(upper,dtype=torch.float32))
        self.register_buffer('joint_mask',torch.tensor(joint_mask,dtype=torch.float32))
        if self.lower.shape!=(20,) or self.upper.shape!=(20,) or self.joint_mask.shape!=(21,20) or not (self.upper>self.lower).all():
            raise ValueError('Invalid pretrained SDF joint domain or dependency mask')
        self.links=nn.Embedding(21,embedding)
        layers=[nn.Linear(63+embedding,width),nn.Softplus(beta=10)]
        for _ in range(depth-1):layers.extend((nn.Linear(width,width),nn.Softplus(beta=10)))
        layers.append(nn.Linear(width,1));self.network=nn.Sequential(*layers)

    def forward(self,q,points,links):
        shape=torch.broadcast_shapes(q.shape[:-1],points.shape[:-1],links.shape)
        q=q.expand(*shape,20);points=points.expand(*shape,3);links=links.long().expand(shape)
        mask=self.joint_mask[links]
        pose=torch.cat(((2*(q-self.lower)/(self.upper-self.lower)-1)*mask,q.sin()*mask,q.cos()*mask),-1)
        return .1*self.network(torch.cat((points/.1,pose,self.links(links)),-1)).squeeze(-1)


def geometry_spec(hand):
    mask=torch.zeros(21,20)
    model=hand.robot.model
    for owner,body in enumerate(hand.ids):
        while body:
            for j in range(int(model.body_jntnum[body])):
                address=int(model.jnt_qposadr[model.body_jntadr[body]+j])-6
                if address>=0:mask[owner,address]=1
            body=int(model.body_parentid[body])
    return dict(asset_sha256=hand.surface.asset_hash,joint_names=hand.robot.joint_names,
        link_names=list(hand.collision.names),joint_mask=mask.tolist(),domain=hand.domain,
        frame='native_model_world_with_six_wrist_dofs_zero',units='m',q_units='rad',
        sign='negative_inside_positive_outside')


def load_pretrained(hand,path):
    path=Path(path);saved=torch.load(path,map_location='cpu',weights_only=True)
    expected=geometry_spec(hand);actual=saved['geometry'];config=saved['model_config']
    if any(actual.get(k)!=v for k,v in expected.items()):
        raise ValueError('Pretrained SDF mesh, joint/link order, frame, sign or units do not match CART')
    for key in ('lower','upper'):
        if config[key]!=hand.domain[key]:raise ValueError('Pretrained SDF normalization domain does not match CART')
    if config['joint_mask']!=expected['joint_mask']:raise ValueError('Pretrained SDF dependency mask does not match CART')
    spec=dict(kind=KIND,model_config=config,geometry=expected,source_sha256=file_hash(path),
        source_name=path.name,source_fingerprint=saved.get('fingerprint'),source_epoch=saved.get('epoch'),
        fine_tune_lr=1.e-5,
        query='actual_rollout_q_native_wrist_points',online_supervision='mesh_local_replay_random_configuration_augmentation')
    from .cart_q_collision import field_spec
    spec['q_collision']=field_spec(hand)
    model=restore_geometry(hand,spec)
    model.pretrained.load_state_dict(saved['model'],strict=True)
    for name,want in (('lower',hand.low),('upper',hand.high),('joint_mask',hand.tensor(expected['joint_mask']))):
        if not torch.equal(getattr(model.pretrained,name),want):raise ValueError('Pretrained SDF buffers disagree with its geometry metadata')
    if not all(torch.isfinite(p).all() for p in model.pretrained.parameters()):raise ValueError('Nonfinite pretrained SDF weights')
    return model,spec


class PretrainedSDF(nn.Module):
    def __init__(self,hand,spec):
        super().__init__()
        if spec['kind']!=KIND or spec['geometry']!=geometry_spec(hand):raise ValueError('Incompatible CART pretrained geometry specification')
        self.hand=hand;self.pretrained=ConfigurationDistance(**spec['model_config']).to(hand.device)
        self.pretrained_lr=spec.get('fine_tune_lr',1.e-5)
        # Keep the contact embedding separate: the old directional head was
        # trained against it, not against the pretrained SDF's embedding.
        self.links=nn.Embedding(21,16).to(hand.device);self.contact_field=ContactField().to(hand.device)
        self.register_buffer('body_ids',torch.as_tensor(hand.ids,device=hand.device,dtype=torch.long))
        self.register_buffer('reference_q',(hand.low+hand.high)*.5)
        with torch.no_grad():
            p,r=hand.fixed_fk.frames(self.reference_q[None])
        self.register_buffer('reference_p',p[0,hand.ids]);self.register_buffer('reference_r',r[0,hand.ids])

    def forward(self,local,links):
        """Intrinsic reference slice, for explicit local-only diagnostic calls."""
        world=(self.reference_r[links]@local[...,None]).squeeze(-1)+self.reference_p[links]
        return self.pretrained(self.reference_q,world,links)

    def at_frames(self,local,links,poses,q,native_p,native_r):
        bodies=self.body_ids[links]
        world=(native_r[poses,bodies]@local[...,None]).squeeze(-1)+native_p[poses,bodies]
        return self.pretrained(q[poses],world,links)

    def training_distance(self,local,links,*,validation=False):
        # A local mesh label is valid at every q. Augment online replay with
        # fresh configurations instead of fine-tuning only a fixed-q slice.
        generator=torch.Generator(device=local.device).manual_seed(41714) if validation else None
        q=self.pretrained.lower+torch.rand((len(local),20),device=local.device,generator=generator)*(self.pretrained.upper-self.pretrained.lower)
        with torch.no_grad():p,r=self.hand.fixed_fk.frames(q)
        return self.at_frames(local,links,torch.arange(len(local),device=local.device),q,p,r)

    def contact(self,points,directions,links,palmar):
        return self.contact_field(points,directions,palmar,self.links(links.long()))

    def load_contact_state(self,state):
        self.links.load_state_dict({'weight':state['links.weight']})
        self.contact_field.load_state_dict({k.removeprefix('contact_field.'):v for k,v in state.items() if k.startswith('contact_field.')},strict=True)

    def query(self,hand,q,points,links,frames=None):
        frames=hand.frames(q) if frames is None else frames
        native=points@hand.basis.T+frames[3][:,hand.ids[0],None]
        return self.pretrained(q[:,None],native,links.expand(points.shape[:-1]))


def fresh_geometry_spec(hand):
    from .cart_q_collision import field_spec
    from .wuji_surface import COMPLETE_PAD_POLICY
    consistent=hand.surface.contact_policy==COMPLETE_PAD_POLICY
    geometry=geometry_spec(hand)
    centres=[[0.,0.,0.] for _ in range(21)]
    for owner in PAD_OWNERS:
        centres[owner]=np.asarray(hand.surface.contact_surface(owner)['triangles']).reshape(-1,3).mean(0).tolist()
    spec=dict(kind=STRUCTURED_KIND if consistent else POTENTIAL_KIND,geometry=geometry,learning_rate=1.e-3,contact_width=128,contact_depth=4,
        contact_decoders='independent_scalar_pad_potentials_v3',contact_centres_m=centres,contact_scale_m=.01,
        projection='x_minus_autograd_potential_gradient',projection_supervision='potential_value_gradient_and_exact_Hessian_vector',
        smooth_aggregation_temperature_mm=.05,vector_lr_scale=.1,
        q_collision=field_spec(hand,group_sobolev=True,independent_decoders=True,consistent_supervision=consistent),query='exact_differentiable_inverse_FK_then_intrinsic_link_field',
        online_supervision='mesh_local_signed_distance_normal_and_contact_replay',
        initialization='random_no_checkpoint')
    if consistent:
        import hashlib
        spec.update(contact_surface_policy=COMPLETE_PAD_POLICY,
            contact_face_sha256={str(l):hashlib.sha256(np.asarray(hand.surface.contact_surface(l)['face_ids'],dtype='<i8').tobytes()).hexdigest() for l in PAD_OWNERS},
            contact_sampling='balanced_links_area_weighted_barycentric_75pct_6mm_25pct_25mm',
            derivative_labels='central_analytic_with_two_scale_native_finite_difference_audit')
        spec.update(contact_decoders='fixed_geometric_features_smooth_softmin_v1',
            projection='all_face_edge_vertex_features_no_winner_search',
            projection_band_m=5.e-6,normal_temperature_m=5.e-5,
            projection_supervision='native_independent_audit_no_Hessian_distillation',
            q_collision=field_spec(hand,structured=True))
    return spec


class IntrinsicSDF(nn.Module):
    intrinsic=True
    def __init__(self,hand,spec):
        super().__init__()
        if spec['geometry']!=geometry_spec(hand):raise ValueError('Incompatible intrinsic SDF geometry')
        self.hand=hand;self.pretrained=IntrinsicDistance().to(hand.device)
        # The attribute preserves optimizer grouping, but these weights are fresh.
        self.pretrained_lr=spec['learning_rate']
        self.vector_lr_scale=spec.get('vector_lr_scale',.01)
        self.independent_decoders=spec.get('kind') in (INDEPENDENT_KIND,POTENTIAL_KIND,CONSISTENT_KIND,STRUCTURED_KIND)
        self.verified_supervision=spec.get('kind') in (CONSISTENT_KIND,STRUCTURED_KIND)
        if self.verified_supervision:
            expected=fresh_geometry_spec(hand)
            if any(spec.get(k)!=expected[k] for k in ('contact_surface_policy','contact_face_sha256')):
                raise ValueError('Contact supervision mesh policy or face identities differ from checkpoint')
        self.smooth_temperature_mm=spec.get('smooth_aggregation_temperature_mm',0.)
        if self.independent_decoders:
            self.links=nn.Identity()
            if spec['kind']==STRUCTURED_KIND:
                from .cart_structured_geometry import SmoothPadProjection
                self.contact_field=SmoothPadProjection(hand,band_m=spec['projection_band_m'],normal_temperature_m=spec['normal_temperature_m'])
            elif spec['kind'] in (POTENTIAL_KIND,CONSISTENT_KIND):
                self.contact_field=PadPotentialField(spec['contact_centres_m'],width=spec['contact_width'],depth=spec['contact_depth'],scale=spec['contact_scale_m']).to(hand.device)
            else:self.contact_field=IndependentPadField(width=spec['contact_width'],depth=spec['contact_depth']).to(hand.device)
        else:
            self.links=nn.Embedding(21,16).to(hand.device)
            self.contact_field=ContactField(width=spec['contact_width'],depth=spec['contact_depth']).to(hand.device)

    def forward(self,local,links):return self.pretrained(local,links)

    def training_distance(self,local,links,*,validation=False):return self(local,links)

    def at_frames(self,local,links,poses,q,native_p,native_r):
        # `local` already contains the differentiable inverse of the floating
        # link pose. Applying native FK again would cancel the wrist derivative.
        return self(local,links)

    def contact(self,points,directions,links,palmar):
        if self.independent_decoders:
            return self.contact_field(points,links.long())
        return self.contact_field(points,directions,palmar,self.links(links.long()))

    def query(self,hand,q,points,links,frames=None):
        p,r=(hand.frames(q) if frames is None else frames)[:2]
        ids=torch.arange(len(q),device=q.device)[:,None]
        links=links.expand(points.shape[:-1])
        local=(r[ids,links].transpose(-1,-2)@(points-p[ids,links])[...,None]).squeeze(-1)
        return self(local,links)


def restore_geometry(hand,spec):
    from .cart_conditional_sdf import KIND as CONDITIONAL_KIND,ConditionalSDFGeometry
    if spec.get('kind')==CONDITIONAL_KIND:return ConditionalSDFGeometry(hand,spec)
    from .cart_link_sdf import KIND as LINK_KIND,LinkSDFGeometry
    if spec.get('kind')==LINK_KIND:return LinkSDFGeometry(hand,spec)
    if spec.get('kind') not in (KIND,INTRINSIC_KIND,INDEPENDENT_KIND,POTENTIAL_KIND,CONSISTENT_KIND,STRUCTURED_KIND) or not spec.get('q_collision'):
        raise ValueError('CART v9/wrist requires an articulated SDF and q-only self-collision field')
    model=PretrainedSDF(hand,spec) if spec['kind']==KIND else IntrinsicSDF(hand,spec)
    from .cart_q_collision import QCollisionField
    from .cart_structured_geometry import Q_KIND,StructuredQCollision
    model.q_collision=(StructuredQCollision(hand,model.pretrained,spec['q_collision']) if spec['q_collision']['kind']==Q_KIND else QCollisionField(hand,spec['q_collision']))
    return model
