#!/usr/bin/env python3
"""Frozen-upstream native CUDA ablation of q-init priors and per-frame trust boxes."""
import argparse,json,time,sys,gc,hashlib,os
from pathlib import Path
import numpy as np
import torch

CASES=[dict(name='baseline',weight=0.,step=.35,radius=None)]+[dict(name=f'prior_{v:g}',weight=v,step=.35,radius=None) for v in [.01,.1,1.,10.]]+[dict(name=f'step_{v:g}',weight=0.,step=v,radius=None) for v in [.05,.01]]+[dict(name=f'box_{v:g}',weight=0.,step=.35,radius=v) for v in [.02,.05,.1]]+[dict(name='prior_0.1_box_0.05',weight=.1,step=.35,radius=.05)]


def metrics(q,gt,anchor,offsets):
    e=q-gt;dif=np.concatenate([np.diff(e[a:b],axis=0) for a,b in zip(offsets[:-1],offsets[1:])]);steps=np.concatenate([np.diff(q[a:b],axis=0) for a,b in zip(offsets[:-1],offsets[1:])])
    return dict(frames=len(q),rmse_rad=float(np.sqrt((e*e).mean())),mae_rad=float(abs(e).mean()),max_error_rad=float(abs(e).max()),error_p99_rad=float(np.quantile(abs(e),.99)),within_1e4_fraction=float((abs(e)<=1e-4).mean()),all_joints_within_1e4_fraction=float((abs(e).max(-1)<=1e-4).mean()),delta_rmse_rad=float(np.sqrt((dif*dif).mean())),max_frame_step_rad=float(abs(steps).max()),max_network_offset_rad=float(abs(q-anchor).max()),network_offset_rmse_rad=float(np.sqrt(((q-anchor)**2).mean())),per_joint_rmse_rad=np.sqrt((e*e).mean(0)).tolist())


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--repeats',type=int,default=3);a=p.parse_args();r=a.root;old=a.baseline;sys.path[:0]=[str(old),str(old/'reference')]
    from benchmark_wuji_parallel_lm import reference_math
    from human_ik_reference.kinematics import HumanHandKinematics
    from human_ik_reference.parameters import default_parameters
    from retarget.learning.wuji_ik.batched_lm import FastFingerURDF,RobotRetargetObjective,JointAnchorObjective,prepare_robot_targets,parallel_sdk_output_filter
    from retarget.learning.wuji_ik.robokit_lm import RobokitCUDALM
    import warp as wp
    wp.config.kernel_cache_dir=os.environ.get('WUJI_WARP_CACHE_DIR',str(old/'warp_cache'));torch.set_num_threads(2);device=torch.device('cuda:0');torch.cuda.set_device(device);torch.cuda.set_per_process_memory_fraction(6*1024**3/torch.cuda.get_device_properties(0).total_memory,device);torch._inductor.config.compile_threads=2;torch._dynamo.config.recompile_limit=128
    def tensor(x):return torch.as_tensor(x,device=device,dtype=torch.float64)
    # Verify native per-frame bounds and graph replay after changing the bounds.
    target=tensor([[.8,-.7],[-.6,.9]])
    class Quadratic:
        def __call__(self,q):return q-target,torch.eye(2,device=device,dtype=q.dtype).expand(len(q),-1,-1)
    engine=RobokitCUDALM(Quadratic(),iterations=30,compile=False,max_step=.02);initial=torch.zeros_like(target);lo=tensor([[-.05,-.06],[-.07,-.08]]);hi=-lo
    torch.testing.assert_close(engine(initial,lo,hi).q,target.clamp(lo,hi),atol=1e-6,rtol=0)
    lo.mul_(.5);hi.mul_(.5);torch.testing.assert_close(engine(initial,lo,hi).q,target.clamp(lo,hi),atol=1e-6,rtol=0)
    anchored=JointAnchorObjective(Quadratic(),initial,2.);unbounded=RobokitCUDALM(anchored,iterations=30,compile=False)
    torch.testing.assert_close(unbounded(initial,initial.new_full((2,),-1),initial.new_full((2,),1)).q,target/3,atol=1e-5,rtol=0)
    (r/'preflight.json').write_text(json.dumps(dict(passed=True,neural_optimizer_updates=0,native_per_frame_bounds=True,graph_replay_refreshes_bounds=True,anchor_closed_form=True),indent=2));del engine,unbounded
    rk,pre,_=reference_math(old/'reference');selection=json.loads((old/'selection.json').read_text())['trajectories'];rows=[]
    for config in ['builtin','hl_urdf']:
      for side in ['left','right']:
        if json.loads((r/'CONTROL.json').read_text())['stop_requested']:raise InterruptedError('Stop requested')
        print('GROUP',config,side,flush=True);robot=rk.HandKinematics('WujiHand2',side);fk=FastFingerURDF(robot.urdf_path,robot.joint_names,robot.landmark_names,device=device);human=HumanHandKinematics(side,old/'assets'/f'{config}_{side}.urdf',sdk_tx_rounding=True)
        seed=np.concatenate(default_parameters(side)['seed']);neutral=human.forward(seed)['skeleton'][:,:3,3].astype(np.float32).astype(np.float64);op=pre.OPERATOR2MANO_RIGHT if side=='right' else pre.OPERATOR2MANO_LEFT;rz=np.array([[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]]);alignment=tensor(pre.estimate_frame(neutral)@op@rz.T)
        preds=[dict(np.load(old/'parallel_robokit_compiled'/config/f'trajectory_{x["index"]}_{side}.npz')) for x in selection];gts=[dict(np.load(old/'gt'/config/f'trajectory_{x["index"]}_{side}.npz')) for x in selection];offsets=np.cumsum([0]+[len(x['q_network_rad']) for x in preds]);anchor=tensor(np.concatenate([x['q_network_rad'] for x in preds]));skeleton=tensor(np.concatenate([x['skeleton_lm_m'] for x in preds]));gt=np.concatenate([x['q_sdk_rad'] for x in gts]);target_args=prepare_robot_targets(skeleton,alignment);base1=RobotRetargetObjective(fk,*target_args,stage=1);base2=RobotRetargetObjective(fk,*target_args,stage=2);saved_raw=[];saved_filtered=[]
        for case in CASES:
          if json.loads((r/'CONTROL.json').read_text())['stop_requested']:raise InterruptedError('Stop requested')
          obj1=JointAnchorObjective(base1,anchor,case['weight']) if case['weight'] else base1;obj2=JointAnchorObjective(base2,anchor,case['weight']) if case['weight'] else base2
          lo,hi=fk.lower,fk.upper
          if case['radius'] is not None:lo=torch.maximum(lo,anchor-case['radius']);hi=torch.minimum(hi,anchor+case['radius'])
          first=RobokitCUDALM(obj1,iterations=25,compile=True,max_step=case['step']);second=RobokitCUDALM(obj2,iterations=40,compile=True,max_step=case['step'])
          @torch.no_grad()
          def run():
            candidate=first(anchor,lo,hi).q;seed_cost=.5*obj2(anchor)[0].square().sum(-1);trial_cost=.5*obj2(candidate)[0].square().sum(-1);start=torch.where((trial_cost<seed_cost)[:,None],candidate,anchor);q=second(start,lo,hi).q;filtered=torch.cat([parallel_sdk_output_filter(q[u:v]) for u,v in zip(offsets[:-1],offsets[1:])]);return q,filtered
          torch.cuda.synchronize();started=time.perf_counter();q,f=run();torch.cuda.synchronize();cold=time.perf_counter()-started;reference=(q.clone(),f.clone());timing=[]
          for repeat in range(a.repeats):
            torch.cuda.synchronize();started=time.perf_counter();q,f=run();torch.cuda.synchronize();timing.append(time.perf_counter()-started);torch.testing.assert_close(q,reference[0],atol=1e-7,rtol=1e-7);torch.testing.assert_close(f,reference[1],atol=1e-7,rtol=1e-7)
          if case['radius'] is not None:assert float((q-anchor).abs().max())<=case['radius']+2e-6
          assert float(torch.maximum(lo-q,q-hi).clamp_min(0).max())<2e-6
          raw=q.cpu().numpy();filtered=f.cpu().numpy();baseline_difference=None
          if case['name']=='baseline':
            previous=np.concatenate([x['q_lm_raw_rad'] for x in preds]);baseline_difference=float(abs(previous-raw).max());assert baseline_difference<1e-5,baseline_difference
          row=dict(configuration=config,side=side,case=case,raw=metrics(raw,gt,anchor.cpu().numpy(),offsets),filtered=metrics(filtered,gt,anchor.cpu().numpy(),offsets),network=metrics(anchor.cpu().numpy(),gt,anchor.cpu().numpy(),offsets),robot_and_filter_seconds_repeats=timing,robot_and_filter_seconds_median=float(np.median(timing)),cold_seconds=cold,baseline_reproduction_max_rad=baseline_difference,spatial_cost_mean=float((.5*base2(q)[0].square().sum(-1)).mean()),regularized_cost_mean=float(second.objective(q)[0].square().sum(-1).mean()*.5));rows.append(row);saved_raw.append(raw);saved_filtered.append(filtered)
          (r/'partial_results.json').write_text(json.dumps(dict(complete=False,neural_optimizer_updates=0,rows=rows),indent=2));print('CASE_COMPLETE',config,side,case['name'],row['raw']['rmse_rad'],row['filtered']['rmse_rad'],row['raw']['max_error_rad'],flush=True)
          del first,second,reference,q,f,run,obj1,obj2;gc.collect();torch.cuda.empty_cache()
        np.savez_compressed(r/f'{config}_{side}.npz',q_raw_rad=np.stack(saved_raw),q_filtered_rad=np.stack(saved_filtered),q_network_rad=anchor.cpu().numpy(),q_sdk_rad=gt,offsets=offsets,cases=np.array([x['name'] for x in CASES]));del fk,base1,base2,anchor,skeleton,target_args;gc.collect();torch.cuda.empty_cache()
    result=dict(complete=True,neural_optimizer_updates=0,rows=rows,cases=CASES,frames_per_configuration=5770,upstream_cached=True,timing_scope='Robot LM and parallel output filter only; fixed raw-EMF-derived network/human reconstruction caches; excludes upstream, disk, reports and metrics. DiT remains active on shared GPU.',selection_scope='Parameter ablation on previously inspected held-out episodes; no unseen-generalization claim for selected settings')
    (r/'results.json').write_text(json.dumps(result,indent=2));print('ANCHOR_ABLATION_COMPLETE',flush=True)


if __name__=='__main__':main()
