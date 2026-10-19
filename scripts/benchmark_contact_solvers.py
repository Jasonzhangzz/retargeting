#!/usr/bin/env python3
"""Reproduce a short fixed-contact / sampled-LM / QP MPC timing comparison."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,default=ROOT/'data/example')
    parser.add_argument('--report-dir',type=Path,default=ROOT/'reports/contact_solver_benchmark')
    parser.add_argument('--start',type=int,default=23)
    parser.add_argument('--duration',type=float,default=.09)
    parser.add_argument('--multistarts',type=int,default=8)
    args=parser.parse_args()
    report=args.report_dir.resolve();report.mkdir(parents=True,exist_ok=True)
    experiment=args.experiment.resolve()
    output=ROOT/'outputs/contact_solver_benchmark'
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    result=dict(start_frame=args.start,duration_seconds=args.duration,multistarts=args.multistarts,
                contact_rounds=1,hand_steps_per_round=1,contact_samples=8,cases={})
    environment=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    with tempfile.TemporaryDirectory(prefix='retarget_contact_benchmark_') as temporary:
        prepared=Path(temporary)
        for name in ('sequence','contactaware','demonstration_visual.npz'):
            if (experiment/name).exists():
                (prepared/name).symlink_to(experiment/name)
        config=yaml.safe_load((experiment/'forceaware.yaml').read_text())
        config['sequence']['start']=args.start
        (prepared/'forceaware.yaml').write_text(yaml.safe_dump(config))
        for method in ('fixed','sampled-lm','qp'):
            case=method+'_'+stamp
            command=[sys.executable,str(ROOT/'run.py'),'--adamw','--no-viewer','--experiment',str(prepared),
                '--duration',str(args.duration),'--multistarts',str(args.multistarts),
                '--contact-rounds','1','--iterations','1','--contact-samples','8',
                '--output-base',str(output),'--case-name',case]
            command+=['--contact-radius-mm','0'] if method=='fixed' else ['--contact-solver',method]
            with (report/(method+'.log')).open('w') as log:
                subprocess.run(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
            directory=output/'comfree'/case
            rows=[json.loads(line) for line in (directory/'windows.jsonl').read_text().splitlines()]
            warm=rows[1:] or rows
            device_times=[v for r in warm for v in r['solver'].get('contact_device_solve_seconds',[])]
            with np.load(directory/'contacts.npz') as contacts:
                active=contacts['mask']>0
                shift=np.linalg.norm(contacts['z']-contacts['template'],axis=-1)
                maximum=float(shift[active].max(initial=0)*1000)
                assert maximum <= (0. if method=='fixed' else 5.)+1e-4
                np.testing.assert_array_equal(contacts['z'][~active],contacts['template'][~active])
            result['cases'][method]=dict(case=str(directory),windows=len(rows),
                warm_window_seconds=float(np.mean([r['seconds'] for r in warm])),
                warm_contact_seconds=float(np.mean([r['solver']['contact_seconds'] for r in warm])),
                warm_hand_seconds=float(np.mean([r['solver']['hand_seconds'] for r in warm])),
                warm_contact_device_ms_per_round=float(np.mean(device_times)*1000) if device_times else None,
                warm_physics_forward_calls=float(np.mean([r['solver']['forward_calls'] for r in warm])),
                warm_contact_geometry_seconds=float(np.mean([sum(r['solver'].get('contact_geometry_seconds',[])) for r in warm])),
                held_feasible_fraction=float(np.mean([d.get('held_feasible',False) for r in warm
                    for d in r['solver'].get('contact_surface_diagnostics',[])])) if method!='fixed' else None,
                max_contact_shift_mm=maximum,
                final_selection_cost=rows[-1]['selection_merit'],
                selected_primal_mean_mm=rows[-1]['primal_mean_mm'])
            print(method,json.dumps(result['cases'][method]),flush=True)
    result['scope']='Short contact-stage timing and numeric checks only; does not establish complete grasp success.'
    (report/'summary.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
