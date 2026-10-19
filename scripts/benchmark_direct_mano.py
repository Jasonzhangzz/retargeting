#!/usr/bin/env python3
"""Direct noisy-MANO IK references: matched AdamW/ADMM ablation and timings."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.benchmark_mano_sdf_perturbation import metrics
from scripts.analyze_mano_sdf_perturbation import operation_metrics
from retarget.contact_exploration import ObjectMeshSurface


def timing(case):
    rows=[json.loads(line) for line in (case/'windows.jsonl').read_text().splitlines()]
    values=np.array([row['seconds'] for row in rows])*1000
    warm=values[1:]
    return dict(window_count=len(values),first_window_ms=float(values[0]),
        all_windows_mean_ms=float(values.mean()),warm_mean_ms=float(warm.mean()),
        warm_p50_ms=float(np.median(warm)),warm_p95_ms=float(np.percentile(warm,95)),
        warm_max_ms=float(warm.max()),warm_under_30ms_fraction=float(np.mean(warm<=30)),
        solve_wall_seconds=float(json.loads((case/'timing.json').read_text())['wall_seconds']))


def main():
    import mujoco
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,default=ROOT/'reports/mano_direct_reference')
    parser.add_argument('--methods',nargs='+',choices=['sampled-lm','no-admm','qp'],default=['sampled-lm','no-admm'])
    args=parser.parse_args();r=args.report.resolve();r.mkdir(parents=True,exist_ok=True)
    if not (r/'environment.json').exists():
        import platform
        from importlib.metadata import version
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip()
        (r/'environment.json').write_text(json.dumps(dict(gpu=gpu,python=sys.version,platform=platform.platform(),
            numpy=np.__version__,mujoco=mujoco.__version__,warp=version('warp-lang'),viewer=False),indent=2)+'\n')
    if not (r/'manifest.json').exists():
        from types import SimpleNamespace
        from retarget.mano_noise import prepare_inputs,resolve_mano_resources
        models,manopth=resolve_mano_resources()
        preparation=SimpleNamespace(source=ROOT/'data/example',experiment=ROOT/'data/example',output=r,
            manopth=manopth,mano_models=models,duration=3.6,fingers=['thumb','index','all'],amplitudes=[.15,.35,.6])
        conditions=[('projected_zero',None,0.)]+[(f'{f}_{round(v*1000):03d}',f,v)
            for f in preparation.fingers for v in preparation.amplitudes]
        prepare_inputs(preparation,conditions=conditions,task_feasibility=False)
    manifest=json.loads((r/'manifest.json').read_text())
    (r/'logs').mkdir(exist_ok=True)
    model=mujoco.MjModel.from_xml_path(str((ROOT/'data/example/sequence/scene/wuji_right/sugar_box_mesh.xml').resolve()))
    surface=ObjectMeshSurface.from_model(model,model.body('obj').id)
    path=r/'results.json';results=json.loads(path.read_text()) if path.exists() else {}
    environment=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    for condition in manifest['conditions']:
        for method in args.methods:
            name=condition['name']+'__'+method;case=r/'runs/comfree'/name
            if name in results:continue
            command=[sys.executable,str(ROOT/'run.py'),'--adamw','--no-viewer',
                '--experiment',condition['input'],'--duration','3.6','--multistarts','8',
                '--iterations','1','--contact-rounds','1','--output-base',str(r/'runs'),'--case-name',name,'--seed','0']
            command+=['--no-admm'] if method=='no-admm' else ['--contact-solver',method]
            wall=None;code=0
            if not (case/'rollout.npz').exists():
                started=perf_counter()
                with (r/'logs'/(name+'.log')).open('w') as stream:
                    process=subprocess.run(command,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT)
                wall=perf_counter()-started;code=process.returncode
            result=dict(condition=condition['name'],method=method,process_wall_seconds=wall,
                returncode=code,complete=(case/'rollout.npz').exists())
            if not result['complete']:
                result.update(lift_success=False,lift_and_hold_success=False)
            else:
                result.update(metrics(case),**timing(case))
                with np.load(case/'rollout.npz') as a:
                    result.update(operation_metrics(surface,a))
                    assert len(a['ctrl_traj'])==360 and np.isfinite(a['ctrl_traj']).all()
                with np.load(case/'contacts.npz') as c:
                    if method=='no-admm':
                        np.testing.assert_array_equal(c['dual'],0.)
                        np.testing.assert_array_equal(c['c'],c['template'])
                result['lift_success']=bool(result['motion_endpoint_lift_mm']>=50 and result['motion_endpoint_clearance_mm']>=5)
                result['lift_and_hold_success']=bool(result['lift_success'] and result['minimum_hold_clearance_mm']>=5)
            results[name]=result;path.write_text(json.dumps(results,indent=2)+'\n')
            print(json.dumps(result),flush=True)


if __name__=='__main__':main()
