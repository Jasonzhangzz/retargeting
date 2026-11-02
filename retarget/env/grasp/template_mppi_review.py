"""Review corrections with immutable contact identities and adjacent trust bounds."""
import math
import numpy as np
import torch
from .template import WujiContactKinematics
from .template_mppi import TemplatePoseObjective
from .template_mppi_continuation import inverse_pose, pose_change


class ReviewObjective(TemplatePoseObjective):
    def __init__(self,*args,warm_q,number,parent_q=None,parent_points=None,parent_primitive=None,**kwargs):
        self.warm_q=np.asarray(warm_q,float);self.number=number
        self.parent_q=None if parent_q is None else np.asarray(parent_q,float)
        self.parent_points=parent_points;self.parent_primitive=parent_primitive
        super().__init__(*args,**kwargs)

    def _search_bounds(self,q):
        super()._search_bounds(q)
        if self.parent_q is not None:
            p=torch.as_tensor(self.parent_q,device=self.device,dtype=self.qlo.dtype)
            radius=p.new_tensor([.002]*3+[math.radians(2.)]*3+[math.radians(8.)]*20)
            self.qlo=torch.maximum(self.qlo,p-radius);self.qhi=torch.minimum(self.qhi,p+radius)
            if bool((self.qlo>self.qhi).any()):raise ValueError('Adjacent trust bounds conflict')

    def configure_object_regions(self,name,primitive):
        super().configure_object_regions(name,primitive)
        self.primitive=primitive
        if self.number==24:
            from .template_mppi_size_ik import CylinderSideProjection
            from retarget.env.geometry import ObjectMeshSurface
            self.center_fk=WujiContactKinematics(self.cpu,self.bodies.cpu().numpy(),self.device)
            self.finger_slots=[i for i,n in enumerate(self.template.body_names) if n.startswith('right_finger')]
            self.side_projection=CylinderSideProjection(ObjectMeshSurface.from_model(self.cpu,self.object_id),self.device,self.finger_slots)
        if self.number==26:
            # Keep the index wrapping the sphere, with free articulation above
            # these broad curl limits. No exact reference angles are imposed.
            self.low[10]=max(float(self.low[10]),math.radians(40.))
            self.low[13]=max(float(self.low[13]),math.radians(25.))
            self.index_slots=[i for i,n in enumerate(self.template.body_names) if n in ('right_finger2_link3','right_finger2_link4')]
        self._search_bounds(self.bound_initial_q)
        if self.parent_q is not None:
            target=np.asarray(self.parent_points,float)*np.array([primitive['radius_m']/self.parent_primitive['radius_m']]*2+[primitive['height_m']/self.parent_primitive['height_m']])
            self.transported=torch.as_tensor(target,device=self.device,dtype=torch.float32)

    def project(self,q):
        q=super().project(q).clone()
        if self.number==24:
            flat=q.reshape(-1,26);p,r=torch.vmap(self.center_fk.frames)(flat.double())
            points=p+torch.einsum('bcij,cj->bci',r,self.local.double())
            flat[:,2]-=(points[:,self.finger_slots,2].mean(-1)-float(self.center[2])).to(flat.dtype)
        return q

    def optimize(self,initial_q,solver,seed):
        warm=self.project(torch.as_tensor(self.warm_q,device=self.device,dtype=torch.float32))
        return super().optimize(warm.cpu().numpy(),solver,seed)

    def evaluate(self,controls):
        cost=super().evaluate(controls);q=controls[:,0].to(device=self.device,dtype=torch.float32)
        if self.parent_q is not None:
            parent=q.new_tensor(self.parent_q);t,r=inverse_pose(q);pt,pr=inverse_pose(parent)
            displacement=torch.linalg.vector_norm(t-pt,dim=-1);chord=torch.linalg.matrix_norm(r-pr,dim=(-2,-1))
            allowed=2*math.sqrt(2)*math.sin(math.radians(2)/2)
            residual=torch.linalg.vector_norm(self.parts['projected_points']-self.transported,dim=-1)
            valid=(displacement<=.0020002)&(chord<=allowed+2.e-6)&((q[:,6:]-parent[6:]).abs().amax(-1)<=math.radians(8)+2.e-6)&(residual.amax(-1)<=.0020001)
            penalty=(displacement/.002).square()+(chord/allowed).square()+(residual/.001).square().mean(-1)
            self.parts['adjacent_valid']=valid;self.parts['adjacent_penalty']=penalty;self.parts['valid'] &= valid
            cost=torch.where(valid,cost+penalty,torch.inf)
        if self.number==26:
            gap=self.parts['signed_distance_m'][:,self.index_slots]
            cosine=self.parts['normal_cosine'][:,self.index_slots]
            error=(gap/.002).square()+(1-cosine)/(1-math.cos(math.radians(30)))
            penalty=50*error.amin(-1)
            self.parts['index_contact_penalty']=penalty
            cost=cost+penalty
        return cost

    def diagnostics(self,q):
        d=super().diagnostics(q)
        if self.number==24:
            z=np.asarray(d['hand_points'])[self.finger_slots,2]
            d['axial_centering']=dict(mean_z_m=float(z.mean()),error_m=float(z.mean()-self.center[2]),
                max_abs_z_m=float(np.abs(z-self.center[2]).max()),finger_slots=self.finger_slots,passed=bool(abs(z.mean()-self.center[2])<1.e-6))
        if self.parent_q is not None:
            value=np.asarray(q.detach().cpu() if torch.is_tensor(q) else q)
            d['adjacent_motion']=pose_change(self.parent_q,value)
            d['adjacent_motion_valid']=bool(self.parts['adjacent_valid'][0])
            d['max_contact_transport_residual_m']=float(np.linalg.norm(np.asarray(d['projected_points'])-self.transported.cpu().numpy(),axis=-1).max())
            d['cost_parts']['adjacent']=float(self.parts['adjacent_penalty'][0])
        if self.number==26:
            d['index_contact']=dict(slots=self.index_slots,fitted=any(d['fitted'][i] for i in self.index_slots),
                gaps_m=[d['signed_distance_m'][i] for i in self.index_slots],normal_error_deg=[d['normal_error_deg'][i] for i in self.index_slots])
            d['cost_parts']['index_contact']=float(self.parts['index_contact_penalty'][0])
        return d
