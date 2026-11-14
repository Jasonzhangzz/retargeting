"""Crocoddyl C++ adapter; no Python callback in calc/calcDiff/solver loops."""
import sys
import numpy as np
import torch
from .trajectory import TrajectoryResult


def fk_payload(fk):
    links=[fk.root]+[j['child'] for j in fk.ordered];ids={n:i for i,n in enumerate(links)}
    cpu=lambda x:x.detach().cpu().numpy()
    return dict(d=fk.nq,nlinks=len(links),root=0,root_r=cpu(fk.root_R),root_p=cpu(fk.root_p),masks=cpu(fk.mask),outputs=[ids[n] for n in fk.links],
        joints=[dict(parent=ids[j['parent']],child=ids[j['child']],qi=-1 if j['q'] is None else j['q'],xyz=cpu(j['xyz']),axis=cpu(j['axis']),rotation=cpu(j['rotation'])) for j in fk.ordered])


def spatial_payload(spatial,n):
    from .trajectory_pipeline import HumanSpatial
    cpu=lambda x:x.detach().cpu().numpy()
    if isinstance(spatial,HumanSpatial):
        base=spatial.base;p,r=cpu(base.position),cpu(base.rotation);weights,ref=cpu(base.posture_weights),cpu(base.reference)
        return [dict(human=True,stage=2,position=p[t],rotation=r[t].reshape(15,3),posture=weights,reference=ref) for t in range(n)]
    names={'source':'s','laplacian':'l','differential':'d','directions':'directions','normal':'normal','cosine':'cosine','sine':'sine','thumb_weight':'thumb_weight'}
    arrays={k:cpu(getattr(spatial,v)) for k,v in names.items()};edges=cpu(spatial.edges)
    return [dict(human=False,stage=spatial.stage,edges=edges,**{k:(float(value[t]) if value.ndim==1 else value[t]) for k,value in arrays.items()}) for t in range(n)]



class CrocoddylTrajectory:
    def __init__(self,objective,lower,upper,module_directory,threads=8):
        sys.path.insert(0,str(module_directory));import wuji_crocoddyl
        self.objective=objective
        raw=getattr(objective.spatial,'raw',objective.spatial)
        fk=raw.base.fk if hasattr(raw,'base') else raw.fk
        self.solver=wuji_crocoddyl.Solver(fk_payload(fk),spatial_payload(raw,objective.offsets[-1]),list(objective.offsets),objective.reference.cpu().numpy(),lower.cpu().numpy(),upper.cpu().numpy(),objective.weight,threads)
    def refresh(self):
        raw=getattr(self.objective.spatial,'raw',self.objective.spatial)
        self.solver.update(spatial_payload(raw,self.objective.offsets[-1]))
    def __call__(self,initial,iterations):
        self.refresh()
        result=self.solver.solve(initial.detach().cpu().numpy(),iterations)
        q=torch.as_tensor(result['q'],device=initial.device,dtype=initial.dtype)
        return TrajectoryResult(q,self.objective.evaluate(initial,False),self.objective.evaluate(q,False),history=[{k:v for k,v in result.items() if k!='q'}],backend='crocoddyl_3.2.1_cpp_boxfddp_cpu',function_evaluations=result['function_evaluations'],gradient_evaluations=result['gradient_evaluations'])
