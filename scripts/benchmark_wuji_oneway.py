#!/usr/bin/env python3
"""Finite causal/frozen-history GPU comparison, no neural parameter updates.

SDK human states are admitted only in the explicitly marked common-target
ablation. Full-pipeline cases reconstruct human history from EMF themselves.
Cached causal-network seeds are used; timing excludes neural inference and IO.
"""
import argparse,json,os,time,csv,subprocess,signal,threading,gc,hashlib
os.environ.setdefault('PYTORCH_ALLOC_CONF','garbage_collection_threshold:0.6')
from pathlib import Path
import numpy as np
import torch,warp as wp
from retarget.learning.wuji_ik.trajectory_pipeline import EMFTrajectorySolver,CompiledSpatial,quaternion_matrix
from retarget.learning.wuji_ik.trajectory_oneway import OneWayLM,FrozenHistoryRobot,CausalHumanRollout
from retarget.learning.wuji_ik.trajectory import TrajectoryObjective,segmented_iir,robot_diagnostics
from retarget.learning.wuji_ik.batched_lm import RobotRetargetObjective,prepare_robot_targets


def atomic(path,d):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False));tmp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--configuration',required=True);p.add_argument('--side',required=True);p.add_argument('--repeats',type=int,default=10);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    r,out=a.root,a.output;out.mkdir(parents=True,exist_ok=True)
    # Existing campaign must have released its GPU worker before this job starts.
    status=json.loads((r/'STATUS.json').read_text());assert status['status']!='running',status
    budget=dict(neural_optimizer_updates=0,robot_modes=['causal_prefix','frozen_parallel','frozen_history_previous','frozen_history_network'],budgets=[4,16,32,200],stage1_updates=4,online_first_updates=[50,80],online_later_updates=[1,10],warmups=2,repeats=a.repeats,scenarios=['sdk_human_common_targets_DIAGNOSTIC','source_causal_human_from_emf'],scope='5 complete development records, both hands/gloves; no heldout generalization claim',timing_scope='robot-only; causal human reconstruction separately measured; cached neural inference excluded')
    atomic(out/'BUDGET.json',budget);atomic(out/'STATUS.json',dict(status='running',pid=os.getpid(),neural_optimizer_updates=0))
    stop=threading.Event();peaks=[]
    def monitor():
        while not stop.wait(1):
            if not torch.cuda.is_initialized():continue
            q=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True)
            used=sum(int(v[1]) for v in csv.reader(q.stdout.splitlines()) if len(v)>1 and v[0].strip()==str(os.getpid()) and v[1].strip().isdigit());peaks.append(used)
            if used>8192:
                atomic(out/'RESOURCE_VIOLATION.json',dict(pid=os.getpid(),resident_mib=used));os.kill(os.getpid(),signal.SIGTERM)
    threading.Thread(target=monitor,daemon=True).start()
    torch.set_grad_enabled(False);torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(6*1024**3/torch.cuda.get_device_properties(0).total_memory);torch._inductor.config.compile_threads=2;wp.config.kernel_cache_dir='/dev/shm/wuji_lm_runtime_2207_20261111/anchor_warp_cache'
    model=EMFTrajectorySolver(r/'reference',r/'assets'/f'{a.configuration}_{a.side}.urdf',a.side,device='cuda:0');tensor=model.tensor
    records=[v for v in json.loads((r/'selection.json').read_text())['records'] if v['split']=='development' and a.side in v['sides']]
    positions=[];rotations=[];human=[];sdkraw=[];sdkfiltered=[];network=[];offsets=[0];record_offsets=[0]
    for record in records:
        i=record['index'];data=np.load(r/'inputs'/f'trajectory_{i}_{a.side}.npz');gt=np.load(r/'sdk'/a.configuration/f'trajectory_{i}_{a.side}.npz');main=np.load(r/'jobs'/f'development_banded_lm_{a.configuration}_{a.side}'/'budget_16'/f'trajectory_{i}_{a.side}.npz');n=8 if a.smoke else len(data['position'])
        positions.append(data['position'][:n]);rotations.append(data['quaternion'][:n]);human.append(gt['human_q'][:n]);sdkraw.append(gt['q_sdk_raw_recovered'][:n]);sdkfiltered.append(gt['q_sdk_filtered'][:n]);network.append(main['q_network'][:n]);local=np.array([0,n]) if a.smoke else data['segment_offsets'];offsets.extend((local[1:]+record_offsets[-1]).tolist());record_offsets.append(record_offsets[-1]+n)
        if a.smoke:break
    positions=tensor(np.concatenate(positions)).float().double();rotations=quaternion_matrix(tensor(np.concatenate(rotations)).float().double());sdk_h=tensor(np.concatenate(human));sdk_q=tensor(np.concatenate(sdkraw));sdk_f=tensor(np.concatenate(sdkfiltered));initial=tensor(np.concatenate(network));records=records[:len(record_offsets)-1]
    human_engine=CausalHumanRollout(model,positions,rotations,offsets);human_times=[];cold=time.perf_counter()
    for rep in range(2+a.repeats):
        torch.cuda.synchronize();t=time.perf_counter();own_h=human_engine(positions,rotations);torch.cuda.synchronize()
        if rep>=2:human_times.append(time.perf_counter()-t)
    source_h_diff=own_h-sdk_h;human_report=dict(rmse_rad=float(source_h_diff.square().mean().sqrt()),max_rad=float(source_h_diff.abs().max()),timings_s=human_times,cold_and_repeats_wall_s=time.perf_counter()-cold)
    # Independently test CUDA lower scan vs a small FP64 dense solve.
    from retarget.learning.wuji_ik.trajectory_oneway import lower_bidiagonal_solve
    from retarget.learning.wuji_ik.trajectory_oneway_cuda import native_lower_scan
    torch.manual_seed(42);aa=torch.randn(17,20,20,device='cuda',dtype=torch.float64);aa=aa@aa.transpose(-1,-2)+torch.eye(20,device='cuda')*2;ll=torch.eye(20,device='cuda',dtype=torch.float64).expand(17,-1,-1).clone()*-.1;ll[[0,7]]=0;ss=torch.tensor([0]*7+[1]*10,device='cuda');bb=torch.randn(17,20,device='cuda',dtype=torch.float64)
    linear_error=float((native_lower_scan(aa,ll,bb,ss,10)-lower_bidiagonal_solve(aa,ll,bb,ss)).abs().max());assert linear_error<2e-6,linear_error
    rows=[];baselines={};curve_arrays=dict(q_sdk_filtered=sdk_f.cpu().numpy(),q_sdk_raw_recovered=sdk_q.cpu().numpy(),q_network=initial.cpu().numpy(),segment_offsets=np.array(offsets),record_offsets=np.array(record_offsets),record_indices=np.array([v['index'] for v in records]))
    for scenario,hq in [('sdk_human_common_targets_DIAGNOSTIC',sdk_h),('source_causal_human_from_emf',own_h)]:
        targets=prepare_robot_targets(model.sfk.forward(hq)[0],model.alignment)
        objectives=[TrajectoryObjective(CompiledSpatial(RobotRetargetObjective(model.rfk,*targets,stage=s)),offsets,tensor(model.robot.bounds.mean(1)).expand(len(offsets)-1,-1),.01) for s in (1,2)]
        sr=objectives[1].spatial(sdk_q)[0];_,ssdk=robot_diagnostics(objectives[1].spatial,sdk_q,offsets,objectives[1].reference,causal=True,lower=model.rfk.lower,upper=model.rfk.upper)
        baselines[scenario]=dict(sdk_raw_spatial_cost=float(.5*sr.square().sum()),sdk_raw_joint_cost=float(objectives[1].evaluate(sdk_q,False).sum()),sdk_causal_projected_gradient_max=float(ssdk['projected_gradient_inf'].max()),sdk_causal_stationary_fraction=float(ssdk['stationary'].double().mean()))
        for mode in ['causal_prefix','frozen_parallel','frozen_history_previous','frozen_history_network']:
            if mode.startswith('frozen_history'):
                engines=[FrozenHistoryRobot(objectives,initial,model.rfk.lower,model.rfk.upper,seed_mode='network' if mode.endswith('network') else 'previous')];budgets=[10]
            else:engines=[OneWayLM(o,initial,model.rfk.lower,model.rfk.upper,mode=mode) for o in objectives];budgets=[4,16,32,200] if not a.smoke else [4]
            for iterations in budgets:
                times=[];drift=0.;cold=time.perf_counter()
                for rep in range(2+a.repeats):
                    torch.cuda.synchronize();t=time.perf_counter()
                    if mode.startswith('frozen_history'):q,qf=engines[0](initial)
                    else:q=engines[1](engines[0](initial,4).q,iterations).q;q=q.float().double();qf=segmented_iir(q,offsets)
                    torch.cuda.synchronize();elapsed=time.perf_counter()-t
                    assert torch.isfinite(q).all() and (q>=model.rfk.lower-2e-6).all() and (q<=model.rfk.upper+2e-6).all()
                    if rep==1:repeat_q=q.clone()
                    if rep>=2:times.append(elapsed);drift=max(drift,float((q-repeat_q).abs().max()))
                err=qf-sdk_f;rawerr=q-sdk_q;_,state=robot_diagnostics(objectives[1].spatial,q,offsets,objectives[1].reference,causal=True,lower=model.rfk.lower,upper=model.rfk.upper);predicted_p=model.rfk.forward(qf)[0];sdk_p=model.rfk.forward(sdk_f)[0];pos=(predicted_p-sdk_p).norm(dim=-1)
                de=torch.cat([torch.diff(err[s:e],dim=0) for s,e in zip(offsets,offsets[1:])]);inc=torch.cat([torch.diff(qf[s:e],dim=0) for s,e in zip(offsets,offsets[1:])]);res=objectives[1].spatial(q)[0]
                row=dict(scenario=scenario,mode=mode,updates=iterations,frames=len(q),rmse_rad=float(err.square().mean().sqrt()),max_rad=float(err.abs().max()),raw_rmse_rad=float(rawerr.square().mean().sqrt()),above_1e4_fraction=float((err.abs()>1e-4).double().mean()),delta_rmse_rad=float(de.square().mean().sqrt()),max_increment_rad=float(inc.abs().max()),per_joint_rmse_rad=err.square().mean(0).sqrt().cpu().tolist(),spatial_cost=float(.5*res.square().sum()),joint_cost=float(objectives[1].evaluate(q,False).sum()),projected_gradient_max=float(state['projected_gradient_inf'].max()),fk_sdk_rmse_m=float(pos.square().mean().sqrt()),fk_sdk_max_m=float(pos.max()),timings_s=times,median_s=float(np.median(times)),cold_and_repeats_wall_s=time.perf_counter()-cold,repeat_max_drift_rad=drift,record_metrics=[])
                for record,s,e in zip(records,record_offsets,record_offsets[1:]):
                    ee=err[s:e];row['record_metrics'].append(dict(index=record['index'],collection=record['collection'],frames=e-s,rmse_rad=float(ee.square().mean().sqrt()),max_rad=float(ee.abs().max())))
                key=f'{scenario}__{mode}__{iterations}';curve_arrays[key]=qf.cpu().numpy();curve_arrays[key+'__raw']=q.cpu().numpy();rows.append(row)
                np.savez_compressed(out/'curves.npz',**curve_arrays);atomic(out/'result.json',dict(complete=False,configuration=a.configuration,side=a.side,rows=rows,human_source_vs_sdk=human_report,cuda_linear_solve_max_error=linear_error,neural_optimizer_updates=0));print('ONEWAY_CASE',json.dumps({k:v for k,v in row.items() if k not in ['record_metrics','per_joint_rmse_rad','timings_s']}),flush=True)
            del engines;gc.collect();torch.cuda.empty_cache()
        del objectives;gc.collect();torch.cuda.empty_cache()
    stop.set();atomic(out/'result.json',dict(complete=True,configuration=a.configuration,side=a.side,rows=rows,sdk_baselines=baselines,human_source_vs_sdk=human_report,cuda_linear_solve_max_error=linear_error,peak_resident_mib=max(peaks,default=0),limit_resident_mib=8192,neural_optimizer_updates=0,budget=budget));atomic(out/'STATUS.json',dict(status='complete',complete=True,pid=os.getpid(),neural_optimizer_updates=0))

if __name__=='__main__':main()
