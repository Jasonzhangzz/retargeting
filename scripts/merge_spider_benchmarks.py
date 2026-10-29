#!/usr/bin/env python3
"""Append extra solver budgets after checking identical checkpoint costs."""
import argparse
import json
from pathlib import Path
import numpy as np


def merge(cases,output):
    configs=[json.loads((c/'config.json').read_text()) for c in cases]
    rows=[json.loads((c/'results.json').read_text()) for c in cases]
    expected={(r['checkpoint'],r['action_noise_std_rad']):r['value_before'] for r in rows[0]}
    for cfg,group in zip(configs[1:],rows[1:]):
        assert cfg['weights']==configs[0]['weights']
        assert cfg['action_parameterization']==configs[0]['action_parameterization']
        for r in group:
            reference=expected[(r['checkpoint'],r['action_noise_std_rad'])]
            np.testing.assert_allclose(r['value_before'],reference,rtol=1.e-6,atol=1.e-4)
    result=[r for group in rows for r in group]
    keys=[(r['checkpoint'],r['action_noise_std_rad'],r['solver']) for r in result]
    if len(keys)!=len(set(keys)):raise ValueError('Duplicate paired solver cases')
    arrays=[]
    for case in cases:
        with np.load(case/'solutions.npz') as a:arrays.append({key:a[key] for key in a.files})
    if any(set(a)!=set(arrays[0]) for a in arrays):raise ValueError('Replay all component solutions before merging')
    output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'solutions.npz',**{key:np.concatenate([a[key] for a in arrays],axis=0) for key in arrays[0]})
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    cfg=configs[0];cfg['methods']=list(dict.fromkeys(r['solver'] for r in result))
    cfg['component_benchmarks']=[str(c) for c in cases]
    (output/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    validation=[json.loads((c/'state_replay_validation.json').read_text()) for c in cases]
    (output/'state_replay_validation.json').write_text(json.dumps(dict(
        rollouts=sum(r['rollouts'] for r in validation),physics_steps_per_rollout=48,
        independent_next_state_abs_max=max(r['independent_next_state_abs_max'] for r in validation),
        all_recorded_next_states_match=all(r['all_recorded_next_states_match'] for r in validation),
        component_benchmarks=[str(c) for c in cases]),indent=2)+'\n')
    print(output,'rows',len(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('cases',type=Path,nargs='+')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();merge(a.cases,a.output)
