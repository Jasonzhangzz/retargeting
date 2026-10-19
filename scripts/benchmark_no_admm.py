#!/usr/bin/env python3
"""Run hand-only AdamW on exactly the previously perturbed MANO inputs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.benchmark_mano_sdf_perturbation import metrics
from scripts.analyze_mano_sdf_perturbation import operation_metrics
from retarget.contact_exploration import ObjectMeshSurface


def main():
    import mujoco
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,default=ROOT/'reports/mano_sdf_perturbation')
    parser.add_argument('--output',type=Path,default=ROOT/'reports/mano_no_admm')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    (out/'logs').mkdir(exist_ok=True)
    manifest=json.loads((args.reference/'manifest.json').read_text())
    model=mujoco.MjModel.from_xml_path(str((ROOT/'data/example/sequence/scene/wuji_right/sugar_box_mesh.xml').resolve()))
    surface=ObjectMeshSurface.from_model(model,model.body('obj').id)
    result_path=out/'results.json';results=json.loads(result_path.read_text()) if result_path.exists() else {}
    conditions=[c for c in manifest['conditions'] if c['pattern'] is not None]
    conditions.sort(key=lambda c:({'all_600':0,'thumb_600':1}.get(c['name'],2),c['name']))
    environment=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    for condition in conditions:
        name=condition['name'];case=out/'runs/comfree'/name
        if name in results:continue
        if not (case/'rollout.npz').exists():
            command=[sys.executable,str(ROOT/'run.py'),'--adamw','--no-admm','--no-viewer',
                '--experiment',condition['input'],'--duration','3.6','--multistarts','8',
                '--iterations','1','--output-base',str(out/'runs'),'--case-name',name,'--seed','0']
            with (out/'logs'/(name+'.log')).open('w') as stream:
                subprocess.run(command,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT,check=True)
        audit=json.loads((case/'audit.json').read_text())
        assert not audit['contact_admm_enabled'] and not audit['contact_exploration_enabled']
        assert audit['physical_wrench_loss_enabled']
        with np.load(case/'contacts.npz') as contact:
            np.testing.assert_array_equal(contact['c'],contact['template'])
            np.testing.assert_array_equal(contact['dual'],0.)
        row=dict(condition=name,method='no-admm',**metrics(case))
        with np.load(case/'rollout.npz') as archive:row.update(operation_metrics(surface,archive))
        row['tracking_pass']=bool(row['moving_position_rmse_mm']<=20 and row['moving_rotation_rmse_deg']<=15
            and row['trajectory_endpoint_position_error_mm']<=20 and row['trajectory_endpoint_rotation_error_deg']<=15)
        row['lift_and_motion_complete']=bool(row['tracking_pass'] and row['motion_endpoint_lift_mm']>=50)
        row['operation_pass']=bool(row['lift_and_motion_complete'] and row['actual_airborne_fraction_when_reference_airborne']>=.95)
        results[name]=row;result_path.write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps(row),flush=True)


if __name__=='__main__':main()
