#!/usr/bin/env python3
"""Frozen causal human source vs SDK on bounded raw EMF prefixes."""
import argparse,json,sys
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root;sys.path.insert(0,str(r/'reference'))
from human_ik_reference.solver import HumanIKSession
rows=[]
for config in ['builtin','hl_urdf']:
 for record in json.loads((r/'selection.json').read_text())['records']:
  for side in record['sides']:
   data=np.load(r/'inputs'/f'trajectory_{record["index"]}_{side}.npz');sdk=np.load(r/'sdk'/config/f'trajectory_{record["index"]}_{side}.npz');session=HumanIKSession(side,r/'assets'/f'{config}_{side}.urdf');n=min(16,len(data['position']));errors=[]
   for i in range(n):errors.append(session.step(data['position'][i],data['quaternion'][i])['q']-sdk['human_q'][i])
   error=np.array(errors);row=dict(configuration=config,record=record['index'],side=side,frames=n,rmse_rad=float(np.sqrt(np.square(error).mean())),max_rad=float(abs(error).max()));rows.append(row);print('HUMAN_SOURCE_SDK',row,flush=True)
report=dict(complete=True,passed=max(v['max_rad'] for v in rows)<=1e-4,rows=rows,maximum_error_rad=max(v['max_rad'] for v in rows),scope='Original unmodified human source on first16 raw EMF frames of every stream/configuration; SDK cached human q is diagnostic-only',neural_optimizer_updates=0);(r/'human_source_sdk_validation.json').write_text(json.dumps(report,indent=2));print('HUMAN_SOURCE_SDK_DONE',report['passed'],report['maximum_error_rad'],flush=True)
