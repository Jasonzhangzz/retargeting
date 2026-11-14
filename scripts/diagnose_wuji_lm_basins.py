#!/usr/bin/env python3
"""GPU-only seed controls; SDK-seeded/oracle targets are diagnostic, never production."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
import warp as wp
from benchmark_wuji_parallel_lm import reference_math, quaternion_matrix


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--iterations',type=int,default=100)
    p.add_argument('--frames-per-trajectory',type=int,default=16)
    p.add_argument('--prediction-subdir',default='parallel_robokit')
    a=p.parse_args();root=a.root;sys.path.insert(0,str(root/'reference'))
    from human_ik_reference.kinematics import HumanHandKinematics
    from human_ik_reference.parameters import default_parameters
    from retarget.learning.wuji_ik.batched_lm import FastFingerURDF,HumanPoseObjective,RobotRetargetObjective,prepare_robot_targets,parallel_sdk_output_filter,so3_log,bounded_lm
    from retarget.learning.wuji_ik.robokit_lm import RobokitCUDALM
    rk,pre,_=reference_math(root/'reference')
    wp.config.kernel_cache_dir=str(root/'warp_cache');torch.set_num_threads(2);torch._inductor.config.compile_threads=2
    torch._dynamo.config.recompile_limit=64
    torch.cuda.set_device(0);device=torch.device('cuda:0')
    t=lambda x:torch.as_tensor(x,device=device,dtype=torch.float64)
    selection=json.loads((root/'selection.json').read_text())['trajectories'];report=[]
    for config in ('builtin','hl_urdf'):
      for side in ('left','right'):
        print('DIAGNOSE',config,side,flush=True)
        ds=[dict(np.load(root/'gt'/config/f'trajectory_{record["index"]}_{side}.npz')) for record in selection]
        prod=[dict(np.load(root/a.prediction_subdir/config/f'trajectory_{record["index"]}_{side}.npz')) for record in selection]
        # Undo the documented SDK alpha=.2 output filter for optimizer-only
        # diagnostics. Float32 roundoff is amplified slightly (<1e-4 rad).
        for d in ds:
            filtered=d['q_sdk_rad'].astype(np.float64)
            raw=filtered.copy();raw[1:]=(filtered[1:]-.8*filtered[:-1])/.2
            d['q_sdk_raw_recovered_rad']=raw
        sampled_indices=[]
        for k,(d,pr) in enumerate(zip(ds,prod)):
            length=len(d['q_sdk_rad']);count=min(a.frames_per_trajectory,length)
            worst=int(abs(pr['q_lm_filtered_rad']-d['q_sdk_rad']).max(-1).argmax())
            chosen=list(dict.fromkeys([worst]+np.linspace(0,length-1,max(1,count-1),dtype=int).tolist()))[:count]
            if len(chosen)<count:chosen+=list(set(range(length))-set(chosen))[:count-len(chosen)]
            chosen=np.array(sorted(chosen));sampled_indices.append(chosen)
            ds[k]={key:(value[chosen] if value.ndim and len(value)==length else value) for key,value in d.items()}
            prod[k]={key:(value[chosen] if value.ndim and len(value)==length else value) for key,value in pr.items()}
        offsets=np.cumsum([0]+[len(d['q_sdk_rad']) for d in ds]);n=int(offsets[-1])
        cat=lambda data,key:t(np.concatenate([d[key] for d in data]))
        sdkq=cat(ds,'q_sdk_raw_recovered_rad');hq_sdk=cat(ds,'human_q_sdk_rad');hq_prod=cat(prod,'human_q_lm_rad')
        human=HumanHandKinematics(side,root/'assets'/f'{config}_{side}.urdf',sdk_tx_rounding=True)
        human_fk=FastFingerURDF(root/'assets'/f'{config}_{side}.urdf',human.joint_names,
            ['thumb_rx_coil','index_rx_coil','middle_rx_coil','ring_rx_coil','pinky_rx_coil'],device=device,root_transform=human._tx_inverse)
        parameters=default_parameters(side);reference=t(np.concatenate(parameters['seed']));bounds=t(np.concatenate(parameters['bounds']))
        weights=t(np.ones(21));weights[:5]=.1;weights[[8,12,16,20]]=0
        position=cat(ds,'position_tx_m');rotation=quaternion_matrix(cat(ds,'orientation_xyzw'))
        hobj=HumanPoseObjective(human_fk,position.repeat_interleave(2,0),rotation.repeat_interleave(2,0),reference,weights)
        initial=torch.stack((hq_sdk,hq_prod),1).reshape(2*n,21)
        hsol=RobokitCUDALM(hobj,iterations=a.iterations,compile=False)(initial,bounds[:,0],bounds[:,1])
        hq=hsol.q.reshape(n,2,21);hcost=hsol.cost.reshape(n,2)
        single_hobj=HumanPoseObjective(human_fk,position,rotation,reference,weights)
        hcost_sdk=.5*single_hobj(hq_sdk)[0].square().sum(-1);hcost_prod=.5*single_hobj(hq_prod)[0].square().sum(-1)
        hand_metrics={}
        for label,q in [('sdk',hq_sdk),('production',hq_prod),('sdk_seed_refined',hq[:,0]),('production_seed_refined',hq[:,1])]:
            ph,rh,*_=human_fk.forward(q)
            hand_metrics[label]=dict(position_error_mean_m=(ph-position).norm(dim=-1).mean(0).cpu().tolist(),
                orientation_error_mean_rad=so3_log(rh.transpose(-1,-2)@rotation).norm(dim=-1).mean(0).cpu().tolist())
        robot=rk.HandKinematics('WujiHand2',side)
        robot_fk=FastFingerURDF(robot.urdf_path,robot.joint_names,robot.landmark_names,device=device)
        neutral=human.forward(np.concatenate(parameters['seed']))['skeleton'][:,:3,3].astype(np.float32).astype(np.float64)
        operator=pre.OPERATOR2MANO_RIGHT if side=='right' else pre.OPERATOR2MANO_LEFT
        rz=np.array([[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]])
        alignment=t(pre.estimate_frame(neutral)@operator@rz.T)
        network=cat(prod,'q_network_rad');production=cat(prod,'q_lm_raw_rad')
        seeds=torch.stack((network,production,robot_fk.lower.lerp(robot_fk.upper,.5).expand(n,-1),sdkq),1)
        seed_names=['network_direct','production_refine','mid_limits','sdk_raw_recovered_DIAGNOSTIC_ONLY']
        row=dict(configuration=config,side=side,frames=n,control_threshold_rad=1e-4,
            sampling='Uniform temporal stratification plus the worst production-error frame per trajectory; all frames remain in the main benchmark.',
            sampled_frame_indices=[x.tolist() for x in sampled_indices],
            human_cost_sdk_mean=float(hcost_sdk.mean()),human_cost_production_mean=float(hcost_prod.mean()),
            human_cost_sdk_seed_refined_mean=float(hcost[:,0].mean()),human_cost_production_seed_refined_mean=float(hcost[:,1].mean()),
            human_seed_solution_difference_rmse_rad=float((hq[:,0]-hq[:,1]).square().mean().sqrt()),
            human_refined_sdk_seed_change_max_rad=float((hq[:,0]-hq_sdk).abs().max()),human_pose_metrics=hand_metrics,
            robot_reference='SDK raw commands recovered from alpha=.2 output filtering; production metrics retain original SDK filtered output.',
            robot_seed_names=seed_names,robot_targets={})
        out=root/'basin_controls'/config;out.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(out/f'human_{side}.npz',q_refined_rad=hq.cpu().numpy(),cost=hcost.cpu().numpy(),
                            cost_sdk=hcost_sdk.cpu().numpy(),cost_production=hcost_prod.cpu().numpy(),offsets=offsets)
        for target_name,skeleton in [('production',cat(prod,'skeleton_lm_m')),('sdk_skeleton_DIAGNOSTIC_ONLY',cat(ds,'skeleton_sdk_m'))]:
            target=prepare_robot_targets(skeleton,alignment)
            expanded=[value if k==3 else value.repeat_interleave(4,0) for k,value in enumerate(target)]
            objective=RobotRetargetObjective(robot_fk,*expanded)
            engine=RobokitCUDALM(objective,iterations=a.iterations,compile=False)
            result=engine(seeds.reshape(n*4,20),robot_fk.lower,robot_fk.upper)
            q=result.q.reshape(n,4,20);cost=result.cost.reshape(n,4)
            _,physical_jacobian=objective(result.q)
            thumb_singular=torch.linalg.svdvals(physical_jacobian[:,:,:4]).reshape(n,4,4)
            thumb_tip=robot_fk.forward(result.q)[0][:,4].reshape(n,4,3)
            thumb_gate=objective.thumb_weight.reshape(n,4)[:,0]
            # Diagnostic only: test whether FP64 GPU polishing moves the native result.
            polish=bounded_lm(objective,result.q,robot_fk.lower,robot_fk.upper,iterations=40,trace_every=10)
            polish_change=(polish.q-result.q).abs()
            err=q-sdkq[:,None]
            best=cost.argmin(-1);bestq=q[torch.arange(n,device=device),best]
            summary=dict(thumb_gate_zero_fraction=float((thumb_gate==0).double().mean()),
                thumb_jacobian_rank_below_four_fraction=float((thumb_singular[:,:,-1]<1e-7).double().mean()),
                thumb_smallest_singular_value_median=float(thumb_singular[:,:,-1].median()),
                thumb_network_vs_sdk_seed_joint_rmse_rad=float((q[:,0,:4]-q[:,3,:4]).square().mean().sqrt()),
                thumb_network_vs_sdk_seed_tip_distance_mean_m=float((thumb_tip[:,0]-thumb_tip[:,3]).norm(dim=-1).mean()),
                thumb_network_vs_sdk_seed_tip_distance_max_m=float((thumb_tip[:,0]-thumb_tip[:,3]).norm(dim=-1).max()),
                native_to_fp64_gpu_polish_max_change_rad=float(polish_change.max()),
                native_to_fp64_gpu_polish_rmse_change_rad=float(polish_change.square().mean().sqrt()),
                native_to_fp64_gpu_polish_over_threshold_fraction=float((polish_change>1e-4).double().mean()),
                fp64_gpu_polish_cost_mean=polish.cost.reshape(n,4).mean(0).cpu().tolist(),
                final_cost_mean=cost.mean(0).cpu().tolist(),rmse_rad=err.square().mean((0,2)).sqrt().cpu().tolist(),
                max_error_rad=err.abs().amax((0,2)).cpu().tolist(),
                network_vs_best_solution_rmse_rad=float((q[:,0]-bestq).square().mean().sqrt()),
                network_seed_higher_cost_fraction=float((cost[:,0]>cost.amin(-1)+1e-7).double().mean()),
                production_vs_best_solution_rmse_rad=float((q[:,1]-bestq).square().mean().sqrt()),
                production_seed_higher_cost_fraction=float((cost[:,1]>cost.amin(-1)+1e-7).double().mean()),
                selected_seed_counts=torch.bincount(best,minlength=4).cpu().tolist())
            row['robot_targets'][target_name]=summary
            np.savez_compressed(out/f'robot_{side}_{target_name}.npz',q_rad=q.cpu().numpy(),
                costs=cost.cpu().numpy(),fp64_polished_q_rad=polish.q.reshape(n,4,20).cpu().numpy(),offsets=offsets,seed_names=seed_names)
            print('ROBOT_CONTROL',config,side,target_name,json.dumps(summary),flush=True)
            del engine,result,objective
        report.append(row);(root/'basin_diagnosis.json').write_text(json.dumps(report,indent=2))
        print('HUMAN_CONTROL',json.dumps({k:v for k,v in row.items() if k!='robot_targets'}),flush=True)
        del human_fk,robot_fk,hobj,hsol
        torch.cuda.empty_cache()
    (root/'BASIN_DIAGNOSIS_COMPLETE').write_text('complete\n')
    print('BASIN_DIAGNOSIS_COMPLETE',flush=True)


if __name__=='__main__':main()
