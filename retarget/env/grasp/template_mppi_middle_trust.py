"""A bounded recovery trust region for large centered-middle sphere sizes."""
import math
import numpy as np
import torch
from .template_mppi import TemplatePoseObjective
from .template_mppi_anatomy_review import CenteredMiddleObjective
from .template_mppi_continuation import MotionLimits,inverse_pose


class RecoveryCenteredMiddleObjective(CenteredMiddleObjective):
    def __init__(self,*args,**kwargs):
        self.motion_limits=MotionLimits(.004,4.,12.)
        self.contact_limit=.004
        super().__init__(*args,**kwargs)

    def _search_bounds(self,q):
        TemplatePoseObjective._search_bounds(self,q)
        if self.parent_q is not None:
            p=torch.as_tensor(self.parent_q,device=self.device,dtype=self.qlo.dtype)
            limits=self.motion_limits
            radius=p.new_tensor([limits.translation_m]*3+[math.radians(limits.rotation_deg)]*3+[math.radians(limits.joint_deg)]*20)
            self.qlo=torch.maximum(self.qlo,p-radius);self.qhi=torch.minimum(self.qhi,p+radius)
            if bool((self.qlo>self.qhi).any()):raise ValueError('Recovery trust region does not intersect joint bounds')

    def evaluate(self,controls):
        cost=TemplatePoseObjective.evaluate(self,controls)
        q=controls[:,0].to(device=self.device,dtype=torch.float32);parent=q.new_tensor(self.parent_q)
        t,r=inverse_pose(q);pt,pr=inverse_pose(parent);limits=self.motion_limits
        displacement=torch.linalg.vector_norm(t-pt,dim=-1);chord=torch.linalg.matrix_norm(r-pr,dim=(-2,-1))
        allowed=2*math.sqrt(2)*math.sin(math.radians(limits.rotation_deg)/2)
        residual=torch.linalg.vector_norm(self.parts['projected_points']-self.transported,dim=-1)
        valid=(displacement<=limits.translation_m+2.e-7)&(chord<=allowed+2.e-6)&((q[:,6:]-parent[6:]).abs().amax(-1)<=math.radians(limits.joint_deg)+2.e-6)&(residual.amax(-1)<=self.contact_limit+1.e-7)
        valid &= self.parts['fitted'][:,self.index_slots].any(-1)
        penalty=(displacement/limits.translation_m).square()+(chord/allowed).square()+(residual/.001).square().mean(-1)
        gap=self.parts['signed_distance_m'][:,self.index_slots];cosine=self.parts['normal_cosine'][:,self.index_slots]
        index=50*((gap/.002).square()+(1-cosine)/(1-math.cos(math.radians(30)))).amin(-1)
        _,_,gaps=self.posture.batch(q)
        middle=20*((gaps[:,0]-gaps[:,1])/.006).square().mean(-1)+100*((.008-gaps[:,:2]).clamp_min(0)/.004).square().mean((-1,-2))
        self.parts.update(adjacent_valid=valid,adjacent_penalty=penalty,index_contact_penalty=index,middle_balance=middle)
        self.parts['valid'] &= valid
        return torch.where(valid,cost+penalty+index+middle,torch.inf)

    def evaluate_initialization(self,controls):return self.evaluate(controls)

    def optimize(self,initial_q,solver,seed):return TemplatePoseObjective.optimize(self,initial_q,solver,seed)

    def diagnostics(self,q):
        d=super().diagnostics(q)
        d['adjacent_motion_limits']=vars(self.motion_limits)
        d['contact_transport_limit_m']=self.contact_limit
        return d
