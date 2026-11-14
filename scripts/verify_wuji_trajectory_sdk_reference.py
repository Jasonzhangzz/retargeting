#!/usr/bin/env python3
"""Check frozen reference geometry and causal SLSQP against SDK recordings."""
from pathlib import Path
import argparse,json,sys
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;sys.path.insert(0,str(r/'reference'))
from human_ik_reference.kinematics import HumanHandKinematics
from retarget_reference.session import RetargetSession
rows=[]
for config in ['builtin','hl_urdf']:
 for record in json.loads((r/'selection.json').read_text())['records']:
  for side in record['sides']:
   gt=np.load(r/'sdk'/config/f'trajectory_{record["index"]}_{side}.npz');geometry=HumanHandKinematics(side,r/'assets'/f'{config}_{side}.urdf',sdk_tx_rounding=True);maximum=0.
   for i in sorted({0,len(gt['human_q'])//2,len(gt['human_q'])-1}):maximum=max(maximum,float(abs(geometry.forward(gt['human_q'][i])['skeleton'][:,:3,3]-gt['skeleton'][i]).max()))
   if maximum>2e-6:raise ValueError('SDK glove geometry differs from pinned URDF '+str((config,record['index'],side,maximum)))
   session=RetargetSession.for_hand('WujiHand2',side);n=min(16,len(gt['skeleton']));outputs=[]
   for i in range(n):outputs.append(session.step(gt['skeleton'][i]))
   difference=np.asarray(outputs).astype(float)-gt['q_sdk_filtered'][:n].astype(float);row=dict(configuration=config,record=record['index'],side=side,frames=n,geometry_max_m=maximum,robot_reference_rmse_rad=float(np.sqrt(np.square(difference).mean())),robot_reference_max_error_rad=float(abs(difference).max()),within_control_precision=bool(abs(difference).max()<=1e-4));rows.append(row);print('REFERENCE_CHECK',row,flush=True)
report=dict(complete=True,geometry_passed=True,robot_within_control_precision=all(v['within_control_precision'] for v in rows),rows=rows,scope='All streams: SDK human q/URDF FK at first/middle/last; independent causal SLSQP on first16 SDK skeleton frames. SDK intermediates enter this explicit reference diagnostic only.',neural_optimizer_updates=0)
(r/'sdk_reference_validation.json').write_text(json.dumps(report,indent=2))
