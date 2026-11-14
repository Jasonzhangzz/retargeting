#!/usr/bin/env python3
"""Check joint human SO(3) costs/derivatives, including arbitrary-axis half turns."""
from pathlib import Path
import argparse,json,numpy as np,torch
from scipy.spatial.transform import Rotation
from retarget.learning.wuji_ik.trajectory_pipeline import EMFTrajectorySolver,HumanSpatial
from retarget.learning.wuji_ik.batched_lm import HumanPoseObjective
from retarget.learning.wuji_ik.trajectory import TrajectoryObjective
from retarget.learning.wuji_ik.trajectory_crocoddyl import CrocoddylTrajectory
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);root=p.parse_args().root;torch.set_num_threads(2);axis=np.array([.3,.5,.8]);axis/=np.linalg.norm(axis);rows=[]
for side in ['left','right']:
 for config in ['builtin','hl_urdf']:
  s=EMFTrajectorySolver(root/'reference',root/'assets'/f'{config}_{side}.urdf',side,device='cpu');q=s.seed[None];position,rotation,*_=s.hfk.forward(q)
  for angle in [1e-8,np.pi-1e-3,np.pi-1e-6,np.pi]:
   target=rotation@torch.tensor(Rotation.from_rotvec(axis*angle).as_matrix(),dtype=torch.float64);spatial=HumanSpatial(HumanPoseObjective(s.hfk,position+.001,target,s.seed,s.posture));res,jac=spatial(q);obj=TrajectoryObjective(spatial,[0,1],q,20.);engine=CrocoddylTrajectory(obj,s.hlo,s.hhi,root/'native',threads=1);cpp=engine.solver.spatial_check(0,q[0].numpy());cost_difference=abs(.5*np.square(cpp['residual']).sum()-float(.5*res.square().sum()));assert cost_difference<1e-8
   finite_difference=None
   if angle<np.pi:
    np.testing.assert_allclose(cpp['residual'],res[0].numpy(),atol=2e-8,rtol=1e-8);np.testing.assert_allclose(cpp['jacobian'],jac[0].numpy(),atol=2e-7,rtol=1e-7)
    columns=[]
    for i in range(21):
     step=torch.zeros_like(q);step[0,i]=1e-7;columns.append((spatial(q+step)[0]-spatial(q-step)[0])/(2e-7))
    finite_difference=float((torch.stack(columns,-1)-jac).abs().max());assert finite_difference<3e-5,(side,config,angle,finite_difference)
   else:assert np.isfinite(cpp['residual']).all() and np.isfinite(cpp['jacobian']).all()
   row=dict(side=side,config=config,angle=angle,finite_difference_max=finite_difference,cpp_cost_difference=cost_difference);rows.append(row);print('ROTATION_PASS',row,flush=True)
(root/'rotation_log_validation.json').write_text(json.dumps(dict(passed=True,checks=rows,note='At exactly pi the geodesic cost has a branch cut; check finite residual/cost only, not a nonexistent unique derivative. Nearby angles validate all21 columns and CPU C++/Torch agreement.',neural_optimizer_updates=0),indent=2))
