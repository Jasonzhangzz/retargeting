#!/usr/bin/env python3
"""Compare vectorized glove target preprocessing with the frozen original."""
from pathlib import Path
import argparse,json,numpy as np,torch
from retarget.learning.wuji_ik.trajectory_pipeline import EMFTrajectorySolver
from retarget.learning.wuji_ik.batched_lm import prepare_robot_targets
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root;torch.set_num_threads(2);torch.manual_seed(42);checks=[]
for config in ['builtin','hl_urdf']:
 for side in ['left','right']:
  solver=EMFTrajectorySolver(r/'reference',r/'assets'/f'{config}_{side}.urdf',side,device='cpu');q=solver.seed[None]+torch.randn(7,21,dtype=torch.float64)*.1;points=solver.sfk.forward(q)[0];points[-1,4]=points[-1,8]+points.new_tensor([.002,0,0]);points[-1,12]=points[-1,8]+points.new_tensor([0,.003,0]);target=prepare_robot_targets(points,solver.alignment)
  for t in range(len(q)):
   source=solver.pre.preprocess(points[t].numpy(),'WujiHand2',side);old=solver.pre.prepare_targets(source,thumb_free=True,thumb_rel_skip_proximal=2)
   diffs={key:float(abs(target[k][t].numpy()-old[key]).max()) for k,key in [(0,'S'),(1,'L'),(2,'D'),(4,'directions')]};assert max(diffs.values())<1e-12,(config,side,diffs);np.testing.assert_array_equal(target[3].numpy(),old['edges']);checks.append(dict(config=config,side=side,**diffs))
print('PREPROCESS_PASS',max(max(v[k] for k in ['S','L','D','directions']) for v in checks),flush=True)
(r/'preprocess_validation.json').write_text(json.dumps(dict(passed=True,checks=checks,scope='Both gloves/sides, randomized human poses, includes simultaneous thumb/index/middle pinch; palm alignment + float32 skeleton boundary + pinch + full graph + directions against frozen original preprocessing',neural_optimizer_updates=0),indent=2))
