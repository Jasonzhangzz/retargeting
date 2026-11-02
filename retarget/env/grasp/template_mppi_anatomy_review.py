"""Anatomical constraints for the extension disk and a centered middle finger."""
import math
import numpy as np
import torch
from .template_mppi import TemplatePoseObjective
from .template_mppi_index_engaged import IndexEngagedObjective
from .template_mppi_posture import GraspPosture


class ExtensionDiskObjective(TemplatePoseObjective):
    def configure_object_regions(self,name,primitive):
        super().configure_object_regions(name,primitive)
        for finger in range(2,6):
            start=6+4*(finger-1)
            self.low[start]=max(float(self.low[start]),math.radians(-5))
            self.high[start]=min(float(self.high[start]),math.radians(20))
            self.low[start+1]=max(float(self.low[start+1]),math.radians(-10))
            self.high[start+1]=min(float(self.high[start+1]),math.radians(10))
            self.high[start+2:start+4]=self.high[start+2:start+4].clamp_max(math.radians(40))
        self._search_bounds(self.bound_initial_q)

    def optimize(self,initial_q,solver,seed):
        initial=self.project(torch.as_tensor(initial_q,device=self.device,dtype=torch.float32))
        return super().optimize(initial.cpu().numpy(),solver,seed)


class CenteredMiddleObjective(IndexEngagedObjective):
    def configure_object_regions(self,name,primitive):
        super().configure_object_regions(name,primitive)
        self.posture=GraspPosture(self.cpu,self.template,26,self.device,self.settings)
        self.posture.bounds(self.low,self.high)
        self.low[15]=max(float(self.low[15]),math.radians(-5))
        self.high[15]=min(float(self.high[15]),math.radians(5))
        self._search_bounds(self.bound_initial_q)

    def evaluate(self,controls):
        cost=super().evaluate(controls)
        _,_,gaps=self.posture.batch(controls[:,0].to(device=self.device,dtype=torch.float32))
        balance=(gaps[:,0]-gaps[:,1])/.006
        crowded=(.008-gaps[:,:2]).clamp_min(0)/.004
        penalty=20*balance.square().mean(-1)+100*crowded.square().mean((-1,-2))
        self.parts['middle_balance']=penalty
        return cost+penalty

    def diagnostics(self,q):
        d=super().diagnostics(q)
        value=torch.as_tensor(q,device=self.device,dtype=torch.float32).reshape(1,26)
        _,_,gaps=self.posture.batch(value)
        gap=gaps[0,:2].detach().cpu().numpy()
        d['middle_centering']=dict(abduction_deg=float(np.degrees(value[0,15].cpu())),
            index_middle_gaps_m=gap[0].tolist(),middle_ring_gaps_m=gap[1].tolist(),
            mean_balance_error_m=float(np.abs(gap[0]-gap[1]).mean()),min_gap_m=float(gap.min()),
            policy='middle_abduction_within_5deg_and_balanced_lateral_spacing')
        d['cost_parts']['middle_balance']=float(self.parts['middle_balance'][0])
        return d
