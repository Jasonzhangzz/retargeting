#!/usr/bin/env python3
"""Finite sparse-trajectory benchmark jobs and SDK/reference data generation."""
import argparse,json,sys,time,os,traceback,hashlib
from pathlib import Path
import numpy as np


def save(path,data):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False));tmp.replace(path)


def stop(root):
 if json.loads((root/'CONTROL.json').read_text()).get('stop_requested'):raise InterruptedError('Campaign stop requested')


def sdk_job(a):
 root=a.root;sys.path[:0]=[str(root/'sdk831'),str(root/'reference')];import wuji_sdk as sdk
 sdk.set_log_level('error');assert sdk.__version__=='2026.8.31';records=json.loads((root/'selection.json').read_text())['records'];names=('wrist','thumb_cmc','thumb_mcp','thumb_ip','thumb_tip')+tuple(f+'_'+j for f in ('index_finger','middle_finger','ring_finger','pinky') for j in ('mcp','pip','dip','tip'))
 for config in ['builtin','hl_urdf']:
  rows=[]
  for record in records:
   for side in record['sides']:
    stop(root);idx=record['index'];data=dict(np.load(root/'inputs'/f'trajectory_{idx}_{side}.npz'));p=data['position'];rot=data['quaternion'];n=len(p);offsets=data.get('segment_offsets',np.array([0,n]));sn=record.get('metadata',{}).get('devices',{}).get(side,{}).get('sn') or record.get('sn') or 'offline';out=root/'sdk'/config/f'trajectory_{idx}_{side}.npz';out.parent.mkdir(parents=True,exist_ok=True)
    commands=[];hq=[];sk=[];timings=[]
    for rep in range(2+a.repeats):
     elapsed=0.;human_sec=0.;robot_sec=0.;objects_sec=0.;initialize=0.;qs=[];hs=[];ss=[]
     for start,end in zip(offsets,offsets[1:]):
      begin=time.perf_counter();pipeline=sdk.WujiGlove.offline_pipeline(sn=sn,hand_side=side,urdf_path=str(root/'assets'/f'{config}_{side}.urdf') if config=='hl_urdf' else None);session=sdk.RetargetSession.for_hand(sdk.HandModel.WujiHand2,side=sdk.Handedness.Left if side=='left' else sdk.Handedness.Right);initialize+=time.perf_counter()-begin
      for i in range(int(start),int(end)):
       b=time.perf_counter();frame=sdk.EmfPoseArray(header=sdk.FrameHeader(seq=int(data['source_seq'][i]),timestamp_us=int(data['timestamp_ns'][i]//1000),frame_id=side[0]+'_hand_emf_tx'),poses=[sdk.EmfPose(pose=sdk.Pose(position=p[i,j].tolist(),orientation=sdk.Quaternion(*rot[i,j].tolist())),confidence=float(data['confidence'][i,j])) for j in range(5)]);hstart=time.perf_counter();skeleton=pipeline.hand_skeleton().compute(frame);by={v.name:v for v in skeleton.joints};points=np.array([by[k].pose.position for k in names],np.float32);rstart=time.perf_counter();q=session.step(points);rend=time.perf_counter();objects_sec+=hstart-b;human_sec+=rstart-hstart;robot_sec+=rend-rstart
       if rep==0:
        angles=pipeline.hand_joint_angles().compute(frame);hs.append(np.concatenate([np.asarray(v.angles)[:5 if f==0 else 4] for f,v in enumerate(angles.fingers)]));ss.append(points)
       qs.append(q)
     if rep==0:commands=np.asarray(qs);hq=np.asarray(hs);sk=np.asarray(ss)
     elif not np.array_equal(commands,np.asarray(qs)):raise AssertionError('SDK repeat changed q')
     if rep>=2:timings.append(dict(total=initialize+objects_sec+human_sec+robot_sec,human=human_sec,robot=robot_sec,objects=objects_sec,initialize=initialize))
    raw=commands.astype(np.float64).copy()
    for start,end in zip(offsets,offsets[1:]):raw[start+1:end]=(commands[start+1:end].astype(float)-commands[start:end-1].astype(float))/.2+commands[start:end-1]
    np.savez_compressed(out,q_sdk_filtered=commands,q_sdk_raw_recovered=raw,human_q=hq,skeleton=sk,timestamp_ns=data['timestamp_ns'],source_seq=data['source_seq'])
    row=dict(configuration=config,trajectory=idx,split=record['split'],side=side,frames=n,timings=timings);rows.append(row);save(root/f'sdk_{config}_timings.json',rows);print('SDK_COMPLETE',config,idx,side,n,flush=True)
 save(root/'SDK_COMPLETE.json',dict(complete=True,version=sdk.__version__,warmups=2,repeats=a.repeats,neural_optimizer_updates=0))


def metrics(q,gt,offsets):
 error=q-gt;increments=np.concatenate([np.diff(q[s:e],axis=0) for s,e in zip(offsets,offsets[1:]) if e-s>1]);derr=np.concatenate([np.diff(error[s:e],axis=0) for s,e in zip(offsets,offsets[1:]) if e-s>1]);return dict(frames=len(q),valid_increments=len(derr),mae_rad=float(abs(error).mean()),rmse_rad=float(np.sqrt(np.square(error).mean())),max_error_rad=float(abs(error).max()),above_1e4_fraction=float((abs(error)>1e-4).mean()),frames_above_1e4_fraction=float((abs(error).max(-1)>1e-4).mean()),per_joint_rmse_rad=np.sqrt(np.square(error).mean(0)).tolist(),delta_rmse_rad=float(np.sqrt(np.square(derr).mean())),max_frame_increment_rad=float(abs(increments).max()))


def run_job(a,shared=None):
 import torch,warp as wp
 from retarget.learning.wuji_ik.trajectory_pipeline import EMFTrajectorySolver
 from retarget.learning.wuji_ik.trajectory import TrajectoryObjective
 from retarget.learning.wuji_ik.training import load_inference
 root=a.root;torch.set_grad_enabled(False);wp.config.kernel_cache_dir='/dev/shm/wuji_lm_runtime_2207_20261111/anchor_warp_cache';torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(6*1024**3/torch.cuda.get_device_properties(0).total_memory);torch._inductor.config.compile_threads=2;torch._dynamo.config.recompile_limit=128
 records=[r for r in json.loads((root/'selection.json').read_text())['records'] if r['split']==a.split and a.side in r['sides']]
 inputs=[dict(np.load(root/'inputs'/f'trajectory_{r["index"]}_{a.side}.npz')) for r in records]
 if a.smoke:records=records[:1];inputs=inputs[:1];inputs=[{k:(v[:8] if k!='segment_offsets' else np.array([0,8])) for k,v in x.items()} for x in inputs]
 positions=np.concatenate([x['position'] for x in inputs]);rot=np.concatenate([x['quaternion'] for x in inputs]);offsets=[0];record_offsets=[0]
 for x in inputs:
  local=x.get('segment_offsets',np.array([0,len(x['position'])]));offsets.extend((local[1:]+record_offsets[-1]).tolist());record_offsets.append(record_offsets[-1]+len(x['position']))
 if shared is None:shared={}
 if 'solver' not in shared:
  model,normalizer,state=load_inference(a.checkpoint,torch.device('cuda:0'),backend='flex');shared['state']=state;shared['solver']=EMFTrajectorySolver(root/'reference',root/'assets'/f'{a.configuration}_{a.side}.urdf',a.side,device='cuda:0',model=model,normalizer=normalizer,crocoddyl_module=root/'native')
 solver,state=shared['solver'],shared['state']
 assert state['variant']=='temporal_causal' and state['model_config']['width']==1024 and state['model_config']['layers']==4
 input_quality=[dict(trajectory=record['index'],side=a.side,frames=len(x['position']),confidence_min=float(np.nanmin(x['confidence'])),confidence_mean=np.nanmean(x['confidence'],axis=0).tolist(),confidence_below_08_fraction=float((x['confidence']<.8).mean()),nonfinite_confidence_count=int((~np.isfinite(x['confidence'])).sum()),receiver_distance_max_m=float(np.linalg.norm(x['position'],axis=-1).max()),quaternion_norm_max_deviation=float(abs(np.linalg.norm(x['quaternion'],axis=-1)-1).max())) for record,x in zip(records,inputs)]
 out=root/'jobs'/a.job_id;out.mkdir(parents=True,exist_ok=True);timings=[];result=None;first_timing=None;repeat_reference=None;repeat_drift=0.;save(out/'STATUS.json',dict(status='running',pid=os.getpid(),backend=a.backend,frames=len(positions),neural_optimizer_updates=0))
 cold=time.perf_counter()
 for rep in range(2+a.repeats):
  stop(root)
  if a.backend in ('causal_prefix','frozen_parallel','frozen_history'):result=solver.solve_oneway(positions,rot,offsets,backend=a.backend,iterations=a.iterations,stage1_iterations=200 if a.reference else min(4,a.iterations))
  elif a.backend.startswith('causal_'):result=solver.solve_causal(positions,rot,offsets,backend=a.backend)
  else:result=solver.solve_joint(positions,rot,offsets,backend=a.backend,iterations=a.iterations,stage1_iterations=200 if a.reference else min(4,a.iterations))
  if not torch.isfinite(result.q_raw).all() or not torch.isfinite(result.human_q).all():raise FloatingPointError('Nonfinite prediction')
  if torch.any(result.q_raw<solver.rfk.lower-2e-6) or torch.any(result.q_raw>solver.rfk.upper+2e-6):raise ValueError('Joint limit violation')
  if rep==0:first_timing=result.timings
  if rep==1:repeat_reference=result.q_raw.clone()
  if rep>=2:repeat_drift=max(repeat_drift,float((result.q_raw-repeat_reference).abs().max()))
  if rep>=2:timings.append(result.timings)
  print('REPEAT',a.job_id,rep,json.dumps(result.timings),flush=True)
  save(out/'STATUS.json',dict(status='running',pid=os.getpid(),completed_repeats=rep+1,measured_repeats=len(timings),frames=len(positions),neural_optimizer_updates=0))
 qraw=result.q_raw.cpu().numpy();qfiltered=result.q_filtered.cpu().numpy();hq=result.human_q.cpu().numpy();rows=[];pose=[]
 with torch.no_grad():
  p,r,_,_=solver.hfk.forward(result.human_q);target_r=solver.tensor(rot);from retarget.learning.wuji_ik.trajectory_pipeline import quaternion_matrix
  from retarget.learning.wuji_ik.batched_lm import so3_log
  ep=(p-solver.tensor(positions)).norm(dim=-1);er=so3_log(r.transpose(-1,-2)@quaternion_matrix(target_r)).norm(dim=-1)
  pose=dict(position_rmse_m=float(ep.square().mean().sqrt()),position_max_m=float(ep.max()),rotation_rmse_rad=float(er.square().mean().sqrt()),rotation_max_rad=float(er.max()))
 gt_human=[];gt_robot=[];gt_raw=[]
 for record,s,e,x in zip(records,record_offsets,record_offsets[1:],inputs):
  idx=record['index'];gt=np.load(root/'sdk'/a.configuration/f'trajectory_{idx}_{a.side}.npz');n=e-s;gt_human.append(gt['human_q'][:n]);gt_robot.append(gt['q_sdk_filtered'][:n]);gt_raw.append(gt['q_sdk_raw_recovered'][:n]);local=x.get('segment_offsets',np.array([0,n]));np.savez_compressed(out/f'trajectory_{idx}_{a.side}.npz',q_raw=qraw[s:e],q_filtered=qfiltered[s:e],human_q=hq[s:e],q_sdk_filtered=gt['q_sdk_filtered'][:n],q_sdk_raw_recovered=gt['q_sdk_raw_recovered'][:n],timestamp_ns=x['timestamp_ns'],segment_offsets=local,**({'q_network':result.initial_q[s:e].cpu().numpy()} if result.initial_q is not None else {}))
  for method,pred,key in [('raw',qraw[s:e],'q_sdk_raw_recovered'),('filtered',qfiltered[s:e],'q_sdk_filtered')]:rows.append(dict(trajectory=idx,collection=record['collection'],side=a.side,method=method,**metrics(pred,gt[key][:n],local)))
 from retarget.learning.wuji_ik.batched_lm import RobotRetargetObjective,prepare_robot_targets
 skeleton=solver.sfk.forward(result.human_q)[0];prepared=prepare_robot_targets(skeleton,solver.alignment);spatial=RobotRetargetObjective(solver.rfk,*prepared);rr=spatial(result.q_raw)[0]
 terms={name:float(.5*rr[:,s:e].square().sum()) for name,s,e in [('laplacian',0,63),('direction',63,111),('thumb_relative',111,113),('antireverse',113,123)]}
 previous=result.q_raw[torch.arange(len(qraw),device='cuda').sub(1).clamp_min(0)].clone();previous[torch.tensor(offsets[:-1],device='cuda')]=solver.tensor(solver.robot.bounds.mean(1));terms['history']=float(.005*(result.q_raw-previous).square().sum())
 gt_h=torch.as_tensor(np.concatenate(gt_human),device='cuda',dtype=torch.float64);gt_q=torch.as_tensor(np.concatenate(gt_robot),device='cuda',dtype=torch.float64)
 predicted_p,predicted_r,_,_=solver.rfk.forward(result.q_filtered);baseline_p,baseline_r,_,_=solver.rfk.forward(gt_q)
 human_diff=result.human_q-gt_h;fk_distance=(predicted_p-baseline_p).norm(dim=-1)
 fk_report=dict(human_q_sdk_rmse_rad=float(human_diff.square().mean().sqrt()),human_q_sdk_max_error_rad=float(human_diff.abs().max()),robot_landmark_sdk_rmse_m=float(fk_distance.square().mean().sqrt()),robot_landmark_sdk_max_m=float(fk_distance.max()))
 # SDK intermediates are scoring-only and never supplied to either optimizer.
 common_target=prepare_robot_targets(solver.sfk.forward(gt_h)[0],solver.alignment);common_spatial=RobotRetargetObjective(solver.rfk,*common_target)
 common_obj=TrajectoryObjective(common_spatial,offsets,solver.tensor(solver.robot.bounds.mean(1)).expand(len(offsets)-1,-1),.01)
 common_raw=torch.as_tensor(np.concatenate(gt_raw),device='cuda',dtype=torch.float64)
 common_score=dict(predicted_joint_objective=float(common_obj.evaluate(result.q_raw,False).sum()),sdk_raw_recovered_joint_objective=float(common_obj.evaluate(common_raw,False).sum()),predicted_raw_spatial_cost=float(.5*common_spatial(result.q_raw)[0].square().sum()),sdk_filtered_spatial_cost=float(.5*common_spatial(gt_q)[0].square().sum()),scope='SDK-human targets used only after solving for common-target scoring')
 stage_profile=None
 if a.backend in ('causal_lm','causal_lbfgs') and not a.smoke:
  from retarget.learning.wuji_ik.trajectory_profile import profile_causal_stages
  stage_profile=profile_causal_stages(solver,positions,rot,offsets,a.backend,result,repeats=a.repeats)
 trace={}
 for key,value in result.solver_traces.items():
  if isinstance(value,(str,dict)):trace[key]=value
  else:
   values=value if isinstance(value,list) else [value];trace[key]=[dict(backend=v.backend,function_evaluations=v.function_evaluations,gradient_evaluations=v.gradient_evaluations,initial_cost=v.initial_cost.cpu().tolist(),final_cost=v.cost.cpu().tolist(),history=v.history) for v in values]
 report=dict(complete=True,candidate_signature=json.loads((root/'MANIFEST.json').read_text()).get('candidate_signature'),job_id=a.job_id,backend=a.backend,configuration=a.configuration,side=a.side,split=a.split,iterations=None if a.backend in ('causal_lm','causal_lbfgs','frozen_history') else a.iterations,causal_update_caps={'first':[50,80],'later':[1,10]} if a.backend in ('causal_lm','causal_lbfgs','frozen_history') else None,smoke=a.smoke,frames=len(positions),segments=len(offsets)-1,warmups=2,repeats=a.repeats,timings=timings,cold_first_solve=first_timing,wall_solver_phase_seconds=time.perf_counter()-cold,timing_scope='CPU-resident EMF; features, transfers, network, human, targets, robot, filter included; model/asset load, disk IO and scoring excluded',peak_memory_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),torch_allocation_cap_gib=6,resident_cap_gib=8,repeat_max_drift_rad=repeat_drift,repeat_stable_within_control_precision=repeat_drift<=1e-4,information=result.information,quality=rows,input_quality=input_quality,isolated_stage_profile=stage_profile,human_emf_error=pose,fk_vs_sdk=fk_report,common_sdk_target_score=common_score,total_robot_objective=result.total_cost.cpu().tolist(),robot_objective_terms=terms,convergence=dict(max_projected_gradient=float(result.convergence["projected_gradient_inf"].max()),stationary_frame_fraction=float(result.convergence["stationary"].double().mean()),criterion=result.convergence["criterion"],information=result.convergence["information"]),trace=trace,neural_optimizer_updates=0,gpu=torch.cuda.get_device_name(),model_step=state['step'],model_data_signature=state['data_signature'])
 save(out/'result.json',report);save(out/'STATUS.json',dict(status='complete',frames=len(positions),complete=True,neural_optimizer_updates=0));print('JOB_COMPLETE',a.job_id,flush=True)


def main():
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['sdk','run']);p.add_argument('--root',type=Path,required=True);p.add_argument('--backend',default='banded_lm');p.add_argument('--configuration',default='builtin');p.add_argument('--side',default='right');p.add_argument('--split',default='development');p.add_argument('--iterations',type=int,default=16);p.add_argument('--repeats',type=int,default=10);p.add_argument('--job-id',default='smoke');p.add_argument('--smoke',action='store_true');p.add_argument('--reference',action='store_true');p.add_argument('--budgets',nargs='+',type=int);p.add_argument('--checkpoint',type=Path,default=Path('/dev/shm/wuji_lm_runtime_2207_20261111/ik_distilled_715516_latest_lm.pt'));a=p.parse_args()
 if a.phase=='sdk':sdk_job(a)
 else:
  if a.budgets:
   shared={};base=a.job_id
   for budget in a.budgets:
    a.iterations=budget;a.job_id=base+'/budget_'+str(budget);existing=a.root/'jobs'/a.job_id/'result.json'
    if existing.exists() and json.loads(existing.read_text()).get('candidate_signature')==json.loads((a.root/'MANIFEST.json').read_text()).get('candidate_signature') and json.loads(existing.read_text()).get('complete'):continue
    run_job(a,shared)
   save(a.root/'jobs'/base/'UNIT.json',dict(complete=True,budgets=a.budgets,candidate_signature=json.loads((a.root/'MANIFEST.json').read_text()).get('candidate_signature')))
  else:run_job(a)


if __name__=='__main__':main()
