#!/usr/bin/env python3
"""Diagnostic-only SDK-human common targets; never a production initializer.

Compare final-stage local optimization from NN and SDK raw-reference seeds,
then repeat the production four-update warm stage. This separates target shift,
causal/joint stationarity, initialization, and weakly identified thumb motion.
This command reports quality only, never oracle-assisted production speed.
"""
import argparse,json,os,threading,subprocess,csv,time,signal,gc
os.environ.setdefault('PYTORCH_ALLOC_CONF','garbage_collection_threshold:0.6')
from pathlib import Path
import numpy as np
import torch,warp as wp
from retarget.learning.wuji_ik.trajectory_pipeline import EMFTrajectorySolver
from retarget.learning.wuji_ik.batched_lm import RobotRetargetObjective,prepare_robot_targets
from retarget.learning.wuji_ik.trajectory import TrajectoryObjective,CUDABandedLM,segmented_iir,robot_diagnostics


p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;manifest=json.loads((r/'MANIFEST.json').read_text());assert json.loads((r/'STATUS.json').read_text()).get('complete',False),'Finish main campaign first'
out=r/'diagnostics_common_targets';out.mkdir(exist_ok=True);budget=dict(groups=4,main_records=5,seeds=['network','sdk_raw_recovered_diagnostic'],warm_updates=[0,4],final_updates=200,maximum_returned_robot_updates=3232,capture_warmup_updates_at_most=16,neural_optimizer_updates=0,scope='SDK intermediates enter this marked diagnostic only; no speed claims')
(out/'BUDGET.json').write_text(json.dumps(budget,indent=2))
stop=threading.Event();resident_peaks=[]
def guard():
 while not stop.wait(1):
  if not torch.cuda.is_initialized():continue
  query=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout
  used=sum(int(v[1]) for v in csv.reader(query.splitlines()) if v and int(v[0])==os.getpid() and v[1].strip().isdigit());resident_peaks.append(used)
  if used>8192:
   (out/'RESOURCE_VIOLATION.json').write_text(json.dumps(dict(pid=os.getpid(),resident_mib=used)));os.kill(os.getpid(),signal.SIGTERM)
threading.Thread(target=guard,daemon=True).start()
torch.set_grad_enabled(False);torch.set_num_threads(2);torch._inductor.config.compile_threads=2;torch.cuda.set_per_process_memory_fraction(6*1024**3/torch.cuda.get_device_properties(0).total_memory);wp.config.kernel_cache_dir='/dev/shm/wuji_lm_runtime_2207_20261111/anchor_warp_cache';rows=[]
for config in ['builtin','hl_urdf']:
 for side in ['left','right']:
  main=r/'jobs'/f'development_banded_lm_{config}_{side}'/'budget_16';report=json.loads((main/'result.json').read_text());assert report['candidate_signature']==manifest['candidate_signature'];files=sorted(main.glob('trajectory_*.npz'),key=lambda f:int(f.stem.split('_')[1]));model=EMFTrajectorySolver(r/'reference',r/'assets'/f'{config}_{side}.urdf',side,device='cuda:0');hq=[];raw=[];filtered=[];initial=[];production=[];offsets=[0]
  for file in files:
   index=int(file.stem.split('_')[1]);gt=np.load(r/'sdk'/config/f'trajectory_{index}_{side}.npz');data=np.load(file);hq.append(gt['human_q']);raw.append(gt['q_sdk_raw_recovered']);filtered.append(gt['q_sdk_filtered']);initial.append(data['q_network']);production.append(data['q_raw']);offsets.extend((data['segment_offsets'][1:]+offsets[-1]).tolist())
  tensor=model.tensor;hq=tensor(np.concatenate(hq));sdk=tensor(np.concatenate(raw));sdk_filtered=tensor(np.concatenate(filtered));network=tensor(np.concatenate(initial));original=tensor(np.concatenate(production));target=prepare_robot_targets(model.sfk.forward(hq)[0],model.alignment);objectives=[];engines=[]
  from retarget.learning.wuji_ik.trajectory_pipeline import CompiledSpatial
  for stage in [1,2]:
   objective=TrajectoryObjective(CompiledSpatial(RobotRetargetObjective(model.rfk,*target,stage=stage)),offsets,tensor(model.robot.bounds.mean(1)).expand(len(offsets)-1,-1),.01);objectives.append(objective);engines.append(CUDABandedLM(objective,network,model.rfk.lower,model.rfk.upper))
  spatial=objectives[1].spatial.raw;_,jac=spatial(sdk);singular=torch.linalg.svdvals(jac[:,:,:4]);unconstrained=(spatial.thumb_weight==0)&(sdk[:,[2,3]]>=0).all(-1);baseline=float(objectives[1].evaluate(sdk,False).sum());_,stationary=robot_diagnostics(spatial,sdk,offsets,objectives[1].reference,lower=model.rfk.lower,upper=model.rfk.upper)
  group=dict(configuration=config,side=side,frames=len(sdk),sdk_joint_cost=baseline,sdk_joint_projected_gradient_max=float(stationary['projected_gradient_inf'].max()),production_common_target_cost=float(objectives[1].evaluate(original,False).sum()),thumb_gate_zero_fraction=float((spatial.thumb_weight==0).double().mean()),thumb_gate_and_antireverse_inactive_fraction=float(unconstrained.double().mean()),thumb_sigma_min_median=float(singular[:,-1].median()),thumb_sigma_min_eligible_median=float(singular[unconstrained,-1].median()) if unconstrained.any() else None,solutions=[])
  arrays=dict(q_sdk_filtered=sdk_filtered.cpu().numpy(),q_sdk_raw_recovered=sdk.cpu().numpy(),q_network=network.cpu().numpy(),q_production_raw=original.cpu().numpy(),segment_offsets=np.array(offsets),thumb_weight=spatial.thumb_weight.cpu().numpy(),thumb_singular_values=singular.cpu().numpy())
  for warm in [0,4]:
   for name,seed in [('network',network),('sdk_raw_diagnostic',sdk)]:
    q=seed
    if warm:q=engines[0](q,warm).q
    result=engines[1](q,200);q=result.q;qf=segmented_iir(q,offsets);_,state=robot_diagnostics(spatial,q,offsets,objectives[1].reference,lower=model.rfk.lower,upper=model.rfk.upper);key=f'{name}_warm{warm}';arrays[key]=q.cpu().numpy();arrays[key+'_filtered']=qf.cpu().numpy();error=qf-sdk_filtered;entry=dict(seed=name,warm_updates=warm,final_updates=200,cost=float(result.cost.sum()),filtered_sdk_rmse_rad=float(error.square().mean().sqrt()),filtered_sdk_max_rad=float(error.abs().max()),projected_gradient_max=float(state['projected_gradient_inf'].max()),stationary_fraction=float(state['stationary'].double().mean()));group['solutions'].append(entry);print('BASIN_DIAGNOSTIC',config,side,entry,flush=True)
  for warm in [0,4]:
   qa=arrays[f'network_warm{warm}'];qb=arrays[f'sdk_raw_diagnostic_warm{warm}'];group[f'seed_solution_rmse_warm{warm}']=float(np.sqrt(np.square(qa-qb).mean()));group[f'seed_solution_max_warm{warm}']=float(abs(qa-qb).max())
  np.savez_compressed(out/f'{config}_{side}.npz',**arrays);rows.append(group);(out/'summary.json').write_text(json.dumps(dict(complete=len(rows)==4,diagnostic_only=True,budget=budget,groups=rows,candidate_signature=manifest['candidate_signature']),indent=2));del model,engines,objectives;gc.collect();torch.cuda.empty_cache()
stop.set();(out/'RESOURCE.json').write_text(json.dumps(dict(peak_sampled_resident_mib=max(resident_peaks,default=0),limit_mib=8192)));print('COMMON_TARGET_DIAGNOSTICS_COMPLETE',flush=True)
