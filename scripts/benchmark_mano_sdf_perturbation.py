#!/usr/bin/env python3
"""MANO finger perturbation -> nearest SDF templates -> closed-loop object tracking.

Preparation uses a licensed local MANO model and manopth. Runtime experiments
use only the prepared inputs and this repository's existing controller.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from retarget.mano_noise import FINGERS, nearest_sdf, prepare_inputs as prepare


def worker(args):
    import warp as wp
    from forceaware import mpc_loss as fa
    from retarget.fa import solver as module
    @wp.kernel(enable_backward=False)
    def freeze(job:fa.LossJob,points:module.PointState):
        world,slot=wp.tid();point=job.target.query_position_object[slot]
        points.c[world,slot]=point;points.z[world,slot]=point;points.last_c[world,slot]=point
    factory=module.optimizer_class
    def build(settings):
        cls=factory(settings)
        if args.worker!='fixed':return cls
        class FixedContacts(cls):
            def _contact_block(self,variables):
                b=self.backend
                if not np.array_equal(variables,getattr(self,'_forward_variables',None)):self._evaluate(variables,False)
                if not hasattr(self,'_contact_reduction'):self._contact_reduction=wp.zeros(3,dtype=float,device=b.device)
                for workspace,points in zip((*b.loss.immediate,b.loss.terminal),b.objective.points):
                    wp.launch(freeze,points.c.shape,inputs=[workspace.job,points],device=b.device)
                self._objective_dirty=True
                return []
        return FixedContacts
    module.optimizer_class=build
    import run
    run.main(['--adamw','--no-viewer','--experiment',str(args.experiment),'--duration',str(args.duration),'--multistarts','8',
        '--iterations','1','--contact-rounds','1','--contact-solver','qp' if args.worker=='qp' else 'sampled-lm',
        '--output-base',str(args.output/'runs'),'--case-name',args.case_name,'--seed','0'])
    audit_path=args.output/'runs/comfree'/args.case_name/'audit.json'
    audit=json.loads(audit_path.read_text())
    audit.update(experiment_contact_method=args.worker,
                 template_query_targets_optimized=args.worker!='fixed',
                 contact_update_disabled=args.worker=='fixed',
                 fixed_control_keeps_physical_wrench_and_consensus=True)
    audit_path.write_text(json.dumps(audit,indent=2)+'\n')


def metrics(directory):
    a=np.load(directory/'rollout.npz');moving=(a['target_frames']>=21)&(a['target_frames']<=71)&(np.arange(len(a['drift']))<237)
    held=np.arange(len(a['drift']))>=237
    rows=[json.loads(line) for line in (directory/'windows.jsonl').read_text().splitlines()]
    with np.load(directory/'contacts.npz') as c:
        shift=np.linalg.norm(c['z']-c['template'],axis=-1)[c['mask']>0]*1000
        maximum=float(shift.max(initial=0))
    assert maximum<=5.+1e-3,maximum
    end=min(236,len(a['drift'])-1)
    return dict(moving_position_rmse_mm=float(np.sqrt(np.mean((a['drift'][moving]*1000)**2))),
        moving_rotation_rmse_deg=float(np.sqrt(np.mean(a['rot_err_deg'][moving]**2))),
        trajectory_endpoint_position_error_mm=float(a['drift'][end]*1000),
        trajectory_endpoint_rotation_error_deg=float(a['rot_err_deg'][end]),
        hold_position_rmse_mm=float(np.sqrt(np.mean((a['drift'][held]*1000)**2))) if held.any() else None,
        hold_rotation_rmse_deg=float(np.sqrt(np.mean(a['rot_err_deg'][held]**2))) if held.any() else None,
        final_position_error_mm=float(a['drift'][-1]*1000),final_rotation_error_deg=float(a['rot_err_deg'][-1]),
        peak_object_height_m=float(a['qpos_traj'][:,28].max()),
        max_contact_shift_mm=maximum,finite_simulation=bool(np.isfinite(a['qpos_traj']).all()),
        solver_held_feasible_fraction=float(np.mean([d['held_feasible'] for r in rows for d in r['solver']['contact_surface_diagnostics']]))
            if any(r['solver']['contact_surface_diagnostics'] for r in rows) else None)


def execute(args,manifest):
    report_path=args.output/'results.json'
    results=json.loads(report_path.read_text()) if report_path.exists() else {}
    environment=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    for condition in manifest['conditions']:
        for method in args.methods:
            name=condition['name']+'__'+method;directory=args.output/'runs/comfree'/name
            if name in results:continue
            command=[sys.executable,__file__,'--worker',method,'--experiment',condition['input'],
                '--duration',str(args.duration),'--output',str(args.output),'--case-name',name]
            log=args.output/'logs'/(name+'.log');log.parent.mkdir(exist_ok=True)
            if not (directory/'rollout.npz').exists():
                with log.open('w') as stream:
                    subprocess.run(command,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT,check=True)
            result=dict(condition=condition['name'],method=method,**metrics(directory))
            result['tracking_pass']=bool(result['moving_position_rmse_mm']<=20 and result['moving_rotation_rmse_deg']<=15
                and result['trajectory_endpoint_position_error_mm']<=20 and result['trajectory_endpoint_rotation_error_deg']<=15)
            results[name]=result;report_path.write_text(json.dumps(results,indent=2)+'\n')
            print('completed',json.dumps(result),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'data/example')
    parser.add_argument('--experiment',type=Path,default=ROOT/'data/example')
    parser.add_argument('--output',type=Path,default=ROOT/'reports/mano_sdf_perturbation')
    parser.add_argument('--manopth',type=Path,default=Path('/home/zz/GraspSONIC/thirdparty/DexForge/contactaware/third_party/manopth'))
    parser.add_argument('--mano-models',type=Path,default=Path('/home/zz/GraspSONIC/thirdparty/DexForge/assets/mano'))
    parser.add_argument('--duration',type=float,default=3.6)
    parser.add_argument('--amplitudes',type=float,nargs='+',default=[.15,.35,.6])
    parser.add_argument('--fingers',nargs='+',choices=[*FINGERS,'all'],default=['thumb','index','all'])
    parser.add_argument('--methods',nargs='+',choices=['fixed','sampled-lm','qp'],default=['fixed','sampled-lm','qp'])
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--run-only',action='store_true')
    parser.add_argument('--worker',choices=['fixed','sampled-lm','qp']);parser.add_argument('--case-name')
    args=parser.parse_args();args.output=args.output.resolve();args.experiment=args.experiment.resolve()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.worker:worker(args);return
    path=args.output/'manifest.json'
    manifest=json.loads(path.read_text()) if args.run_only else prepare(args)
    if not args.prepare_only:execute(args,manifest)


if __name__=='__main__':main()
