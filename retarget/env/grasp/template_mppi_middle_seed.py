"""Seed the next sphere size by restoring its already-established index contact."""
import math
import mujoco
import numpy as np
import torch
from .template_mppi_middle_trust import RecoveryCenteredMiddleObjective
from .template_mppi import TemplatePoseObjective


class SeededCenteredMiddleObjective(RecoveryCenteredMiddleObjective):
    def optimize(self,initial_q,solver,seed):
        parent=np.asarray(self.parent_q,float);data=mujoco.MjData(self.cpu);data.qpos[:]=parent
        mujoco.mj_kinematics(self.cpu,data)
        ids=self.bodies.cpu().numpy();rot=data.xmat[ids].reshape(-1,3,3)
        points=data.xpos[ids]+np.einsum('nij,nj->ni',rot,self.template.local_points)
        _,normals,gap=self.audit.query(torch.as_tensor(points,device=self.device,dtype=torch.float32))
        normals=normals.cpu().numpy();gap=gap.cpu().numpy();candidates=[parent.copy()]
        for slot in self.index_slots:
            for target in (-.001,-.0005,0.,.0005,.001):
                for joint in (None,10,12,13):
                    for angle in ((0.,) if joint is None else (-2.,-1.,1.,2.)):
                        q=parent.copy();q[:3]+=(target-float(gap[slot]))*normals[slot]
                        if joint is not None:q[joint]+=math.radians(angle)
                        candidates.append(q)
        proposals=self.project(torch.as_tensor(np.asarray(candidates),device=self.device,dtype=torch.float32))
        costs=self.evaluate(proposals[:,None]);finite=torch.isfinite(costs)
        if not bool(finite.any()):raise FloatingPointError('No feasible index-restoring nominal pose')
        chosen=proposals[int(costs.argmin())].cpu().numpy()
        print('INDEX_NOMINAL',int(finite.sum()),'/',len(proposals),flush=True)
        return TemplatePoseObjective.optimize(self,chosen,solver,seed)
