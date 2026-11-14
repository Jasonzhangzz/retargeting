#!/usr/bin/env python3
"""Compare frozen original spatial costs and analytic gradients in FP64."""
from pathlib import Path
import numpy as np,torch,json,sys
from retarget.learning.wuji_ik.trajectory_pipeline import EMFTrajectorySolver
from retarget.learning.wuji_ik.batched_lm import RobotRetargetObjective,prepare_robot_targets
import argparse
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root;sys.path.insert(0,str(r/'reference'))
from retarget_reference.objectives import FrameObjective
from retarget_reference.session import builtin_parameters
rows=[];torch.set_num_threads(2);torch.manual_seed(42)
for side in ['left','right']:
 for config in ['builtin','hl_urdf']:
  s=EMFTrajectorySolver(r/'reference',r/'assets'/f'{config}_{side}.urdf',side,device='cpu');h=s.seed[None]+torch.randn(3,21,dtype=torch.float64)*.03;target=prepare_robot_targets(s.sfk.forward(h)[0],s.alignment);reference=s.tensor(s.robot.bounds.mean(1));q=s.rfk.lower[None]+torch.rand(3,20,dtype=torch.float64)*(s.rfk.upper-s.rfk.lower)
  # An exactly straight thumb has no defined bend-plane and must be gated off.
  source,lap,diff,edges,dirs=target;source[1,3]=source[1,2]+source.new_tensor([.01,0,0]);source[1,4]=source[1,3]+source.new_tensor([.01,0,0]);diff[1]=lap[1]@source[1]
  for stage in [1,2]:
   obj=RobotRetargetObjective(s.rfk,*target,stage=stage);res,jac=obj(q);points,_,jp,_=s.rfk.forward(q)
   for t in range(3):
    source,lap,diff,edges,dirs=[x[t].numpy() if i!=3 else x.numpy() for i,x in enumerate(target)];weights=np.ones(21);weights[1:4]=0;ref=FrameObjective(source,edges,directions=dirs,laplacian=lap,differential=diff,weights=weights,thumb_triplets=[(2,3,4)],params=builtin_parameters('WujiHand2'),q_ref=reference.numpy(),antirev_indices=s.robot.antirev_indices)
    cost,g,terms=ref.evaluate(q[t].numpy(),points[t].numpy(),jp[t].numpy(),stage=stage);actual=float(.5*res[t].square().sum()+.005*(q[t]-reference).square().sum());gradient=(jac[t]*res[t,:,None]).sum(0)+.01*(q[t]-reference)
    cd=abs(cost-actual);gd=float(abs(gradient.numpy()-g).max());assert cd<1e-9 and gd<1e-9,(config,side,stage,cd,gd);rows.append(dict(config=config,side=side,stage=stage,cost_difference=cd,gradient_difference=gd));print('OBJECTIVE_PASS',rows[-1],flush=True)
(r/'original_objective_validation.json').write_text(json.dumps(dict(passed=True,checks=rows,scope='Includes exactly straight thumb/gate; GPU spatial formula evaluated in CPU FP64 against frozen original NumPy FrameObjective, both stages and sides/configurations, randomized physical joint configurations; includes original raw-history penalty',neural_optimizer_updates=0),indent=2))
