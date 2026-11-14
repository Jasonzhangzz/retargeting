#!/usr/bin/env python3
"""Diagnostic-only SDK skeleton/history controls on CUDA; never production inputs."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
import warp as wp
from benchmark_wuji_parallel_lm import reference_math


class SDKHistoryObjective:
    def __init__(self,base,reference):
        self.base,self.reference=base,reference
        self.eye=torch.eye(20,device=reference.device,dtype=reference.dtype)

    def __call__(self,q):
        r,j=self.base(q)
        return torch.cat((r,.1*(q-self.reference)),-1),torch.cat((j,(self.eye*.1)[None].expand(len(q),-1,-1)),1)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    sys.path.insert(0,str(root/'reference'))
    from human_ik_reference.kinematics import HumanHandKinematics
    from human_ik_reference.parameters import default_parameters
    from retarget.learning.wuji_ik.batched_lm import FastFingerURDF,RobotRetargetObjective,prepare_robot_targets,bounded_lm
    from retarget.learning.wuji_ik.robokit_lm import RobokitCUDALM
    rk,pre,_=reference_math(root/'reference');torch.set_num_threads(2);device=torch.device('cuda:0')
    wp.config.kernel_cache_dir=str(root/'warp_cache')
    t=lambda x:torch.as_tensor(x,device=device,dtype=torch.float64)
    old=json.loads((root/'basin_diagnosis.json').read_text());selection=json.loads((root/'selection.json').read_text())['trajectories'];report=[]
    for row in old:
        config,side=row['configuration'],row['side'];print('SDK_OBJECTIVE_CONTROL',config,side,flush=True)
        robot=rk.HandKinematics('WujiHand2',side);fk=FastFingerURDF(robot.urdf_path,robot.joint_names,robot.landmark_names,device=device)
        human=HumanHandKinematics(side,root/'assets'/f'{config}_{side}.urdf',sdk_tx_rounding=True)
        seed=np.concatenate(default_parameters(side)['seed']);neutral=human.forward(seed)['skeleton'][:,:3,3].astype(np.float32).astype(np.float64)
        operator=pre.OPERATOR2MANO_RIGHT if side=='right' else pre.OPERATOR2MANO_LEFT
        rz=np.array([[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]])
        alignment=t(pre.estimate_frame(neutral)@operator@rz.T)
        raw_all=[];prior_all=[];skeleton_all=[]
        for item,indices in zip(selection,row['sampled_frame_indices']):
            d=np.load(root/'gt'/config/f'trajectory_{item["index"]}_{side}.npz');filtered=d['q_sdk_rad'].astype(np.float64)
            raw=filtered.copy();raw[1:]=(filtered[1:]-.8*filtered[:-1])/.2
            previous=np.concatenate((robot.bounds.mean(1)[None],raw[:-1]))
            raw_all.append(raw[indices]);prior_all.append(previous[indices]);skeleton_all.append(d['skeleton_sdk_m'][indices])
        raw=t(np.concatenate(raw_all));previous=t(np.concatenate(prior_all));skeleton=t(np.concatenate(skeleton_all));n=len(raw)
        spatial=np.load(root/'basin_controls'/config/f'robot_{side}_sdk_skeleton_DIAGNOSTIC_ONLY.npz')['q_rad']
        initial=torch.cat((t(spatial),raw[:,None]),1).reshape(n*5,20)
        targets=prepare_robot_targets(skeleton,alignment)
        targets=[v if k==3 else v.repeat_interleave(5,0) for k,v in enumerate(targets)]
        base=RobotRetargetObjective(fk,*targets);objective=SDKHistoryObjective(base,previous.repeat_interleave(5,0))
        native=RobokitCUDALM(objective,iterations=100,compile=False)(initial,fk.lower,fk.upper)
        polished=bounded_lm(objective,native.q,fk.lower,fk.upper,iterations=100)
        q=polished.q.reshape(n,5,20);cost=polished.cost.reshape(n,5)
        sdk_cost=.5*objective(raw.repeat_interleave(5,0))[0].square().sum(-1).reshape(n,5)[:,0]
        sdk_res,sdk_jac=objective(raw.repeat_interleave(5,0));g=(sdk_jac*sdk_res[...,None]).sum(1).reshape(n,5,20)[:,0]
        active=((raw<=fk.lower+1e-5)&(g>0))|((raw>=fk.upper-1e-5)&(g<0));pg=g*(~active)
        difference=q-raw[:,None]
        detail=dict(configuration=config,side=side,frames=n,
            scope='Diagnostic-only: official SDK skeleton and previous SDK raw q; native CUDA LM plus FP64 GPU polishing, same spatial weights and lambda_smooth=.01. SDK output filter was inverted for raw-q comparisons.',
            seed_names=['network_spatial_solution','production_spatial_solution','mid_spatial_solution','sdk_seed_spatial_solution','sdk_raw_itself'],
            rmse_rad=difference.square().mean((0,2)).sqrt().cpu().tolist(),max_error_rad=difference.abs().amax((0,2)).cpu().tolist(),
            native_rmse_rad=(native.q.reshape(n,5,20)-raw[:,None]).square().mean((0,2)).sqrt().cpu().tolist(),
            cost_sdk_mean=float(sdk_cost.mean()),cost_optimized_mean=cost.mean(0).cpu().tolist(),
            sdk_projected_gradient_abs_max=float(pg.abs().max()),sdk_projected_gradient_abs_mean=float(pg.abs().mean()),
            cost_improved_from_sdk_seed_fraction=float((cost[:,4]<sdk_cost-1e-8).double().mean()),
            network_vs_sdk_seed_solution_rmse_rad=float((q[:,0]-q[:,4]).square().mean().sqrt()),
            network_higher_cost_than_sdk_seed_fraction=float((cost[:,0]>cost[:,4]+1e-7).double().mean()))
        report.append(detail);(root/'sdk_objective_diagnosis.json').write_text(json.dumps(report,indent=2))
        np.savez_compressed(root/'basin_controls'/config/f'robot_{side}_sdk_history_DIAGNOSTIC_ONLY.npz',q_rad=q.cpu().numpy(),sdk_raw_rad=raw.cpu().numpy(),sdk_previous_raw_rad=previous.cpu().numpy(),cost=cost.cpu().numpy(),sdk_cost=sdk_cost.cpu().numpy())
        print('FULL_OBJECTIVE_RESULT',json.dumps(detail),flush=True)
    (root/'SDK_OBJECTIVE_DIAGNOSIS_COMPLETE').write_text('complete\n');print('SDK_OBJECTIVE_DIAGNOSIS_COMPLETE',flush=True)


if __name__=='__main__':main()
