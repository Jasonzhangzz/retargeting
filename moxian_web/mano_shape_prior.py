"""Canonical MANO trust region and differentiable shape preservation."""
import math
import numpy as np
import torch
from scripts.calibrate_wuji_mano import CHAINS,TIPS

REVISION='canonical_mano_small_update_ema_v1'
BETA_RADIUS=1.5
EMA_DECAY=.95
GEOMETRY_UPPER=np.array([math.log(1.15)]*15+[math.log(1.10)]*10+[.2]*12+[math.radians(5)]*6)
GEOMETRY_SIGMA=np.array([.05]*25+[.1]*12+[math.radians(3)]*6)
SOFT_SHAPE_WEIGHT=1000.
SOFT_SHAPE_MARGIN=.02
SOFT_SHAPE_SCALE=.02


def project(vector,lower,upper,shared_size,unbounded_shape=False):
    value=vector.clamp(lower,upper)
    if shared_size and not unbounded_shape:
        beta=value[:10];beta=beta*(BETA_RADIUS/beta.norm().clamp_min(BETA_RADIUS))
        value=torch.cat((beta,value[10:]))
    return value


class ShapePrior:
    def __init__(self,model):
        self.model=model
        with torch.no_grad():
            v,j=model.model.official_rest(torch.zeros(10,device=model.device),torch.zeros(43,device=model.device),apply_residual=False)
        self.reference=v.detach();self.joints=j.detach()
        edges=np.unique(np.sort(np.concatenate([model.faces[:,[0,1]],model.faces[:,[1,2]],model.faces[:,[2,0]]]),axis=1),axis=0)
        self.edges=torch.as_tensor(edges,device=model.device,dtype=torch.long)
        self.lengths=(v[self.edges[:,1]]-v[self.edges[:,0]]).norm(dim=-1)
        self.sections=[];self.axes=[]
        weights=model.model.backend.lbs_weights.detach().cpu().numpy()
        for f,chain in enumerate(CHAINS):
            for k,joint in enumerate(chain):
                end=j[chain[k+1]] if k<2 else v[TIPS[f]]
                axis=torch.nn.functional.normalize(end-j[joint],dim=0)
                normal=torch.as_tensor(model.anatomy.palmar_normal,device=model.device,dtype=v.dtype)
                side=torch.nn.functional.normalize(torch.linalg.cross(axis,normal),dim=0)
                depth=torch.linalg.cross(side,axis)
                ids=np.flatnonzero(weights[:,joint]>.35)
                if len(ids)<4:raise ValueError('MANO 关节截面缺失')
                self.sections.append(torch.as_tensor(ids,device=model.device));self.axes.append(torch.stack((side,depth),-1))
        self.reference_sections=self.sections_size(v).detach()

    def sections_size(self,v):
        return torch.stack([((v[ids]@axis).amax(0)-(v[ids]@axis).amin(0)) for ids,axis in zip(self.sections,self.axes)])

    def residuals(self,beta,geometry):
        v,_=self.model.model.official_rest(beta,geometry)
        edge=(v[self.edges[:,1]]-v[self.edges[:,0]]).norm(dim=-1).clamp_min(1e-8)
        sections=self.sections_size(v).clamp_min(1e-8)
        result=dict(rest_displacement=(v-self.reference)/.004,
                    edge_strain=torch.log(edge/self.lengths)/.10,
                    cross_section_prior=torch.log(sections/self.reference_sections)/.10)
        if getattr(self,'soft_constraints',False):
            smooth=torch.nn.functional.softplus((SOFT_SHAPE_MARGIN-self.constraint_slacks(v))/SOFT_SHAPE_SCALE)
            # Stable p=8 pooling: a single collapsing edge/finger must not be
            # diluted by averaging over thousands of healthy edges. Gradients
            # remain finite and point back into the anatomical region outside
            # its boundary as well as just inside it.
            result['shape_soft_constraints']=torch.exp(torch.logsumexp(8*torch.log(smooth.clamp_min(1e-12)),0)/8)[None]
        if getattr(self,'unbounded_shape',False):
            # Former parameter ranges are prior scales, never a projection.
            result['shape_parameter_soft_constraints']=torch.cat((
                torch.nn.functional.softplus((beta.norm()-BETA_RADIUS)/.5)[None],
                torch.nn.functional.softplus((geometry.abs()-geometry.new_tensor(GEOMETRY_UPPER))/geometry.new_tensor(GEOMETRY_SIGMA))))
        return result

    @torch.no_grad()
    def valid_surface(self,v):
        return torch.isfinite(v).all() & (self.constraint_slacks(v)>=0).all()

    def constraint_slacks(self,v):
        sections=self.sections_size(v)/self.reference_sections
        edges=(v[self.edges[:,1]]-v[self.edges[:,0]]).norm(dim=-1)/self.lengths
        return torch.cat((edges-.65,1.5-edges,(sections-.8).flatten(),(1.25-sections).flatten()))

    @torch.no_grad()
    def ema_update(self,average,proposal):
        updated=average.lerp(proposal,1-EMA_DECAY)
        geometry=updated[46:89] if len(updated)>=89 else None
        v,_=self.model.model.official_rest(updated[:10],geometry)
        valid=self.valid_surface(v)
        # Reject only the proposed geometry update; sensor registration keeps
        # learning. No held-out observation or current touch enters this gate.
        beta=torch.where(valid,updated[:10],average[:10])
        if len(updated)>=89:return torch.cat((beta,updated[10:46],torch.where(valid,updated[46:89],average[46:89]),updated[89:]))
        return torch.cat((beta,updated[10:]))

    @torch.no_grad()
    def report(self,beta,geometry):
        v,_=self.model.model.official_rest(beta,geometry)
        ratios=(self.sections_size(v)/self.reference_sections).cpu().numpy()
        edge=((v[self.edges[:,1]]-v[self.edges[:,0]]).norm(dim=-1)/self.lengths).cpu().numpy()
        return dict(revision=REVISION,accepted=bool(np.isfinite(ratios).all() and ratios.min()>=.8 and ratios.max()<=1.25 and edge.min()>=.65 and edge.max()<=1.5),
                    minimum_section_ratio=float(ratios.min()),maximum_section_ratio=float(ratios.max()),
                    minimum_edge_ratio=float(edge.min()),maximum_edge_ratio=float(edge.max()),
                    per_finger_section_ratios={name:ratios[i*3:i*3+3].tolist() for i,name in enumerate(('thumb','index','middle','ring','pinky'))},
                    beta_kl_to_unit_gaussian=float(.5*beta.square().sum()),beta_norm=float(beta.norm()),
                    reference='unmodified_MANO_zero_betas',thickness_floor_ratio=.8,ema_decay=EMA_DECAY)
