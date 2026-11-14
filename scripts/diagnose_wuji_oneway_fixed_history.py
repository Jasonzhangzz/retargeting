#!/usr/bin/env python3
"""SDK-history teacher-forced diagnostic; never a production result.

Keep source-verified raw predecessor q fixed for each independent frame. Test
zero vs one actual warm LM update, and 10 vs 200 final updates, with preceding
q, network q, and source-current q initializers. This distinguishes schedule,
initialization, stopping, and propagation without future/history feedback.
"""
import argparse,json,time,gc,os,threading,subprocess,csv,signal
from pathlib import Path
import numpy as np
import torch,warp as wp
from retarget.learning.wuji_ik.trajectory_pipeline import EMFTrajectorySolver,CompiledSpatial
from retarget.learning.wuji_ik.batched_lm import RobotRetargetObjective,prepare_robot_targets
from retarget.learning.wuji_ik.trajectory import TrajectoryObjective
from retarget.learning.wuji_ik.trajectory_oneway import OneWayLM


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;w=r/'oneway_comparison';out=w/'fixed_sdk_history_DIAGNOSTIC';out.mkdir(exist_ok=True)
    assert json.loads((w/'STATUS.json').read_text()).get('complete'), 'Release main comparison GPU first'
    check=json.loads((w/'source_slsqp_stopping/summary.json').read_text());assert check['complete'] and check['all_bitwise_equal']
    budget=dict(groups=4,seeds=['source_previous_raw','network','source_current_raw_DIAGNOSTIC'],warm_updates=[0,1],final_updates=[10,200],expected_cases=48,maximum_returned_robot_updates=4*3*sum(2*u+1 for u in [10,200]),capture_overhead_updates_at_most=16,neural_optimizer_updates=0,scope='Teacher-forced original source history, not independent production IK; first frames included with same artificial warm budget and separately scored later-only; quality only, no timing claim')
    (out/'BUDGET.json').write_text(json.dumps(budget,indent=2));torch.set_grad_enabled(False);torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(6*1024**3/torch.cuda.get_device_properties(0).total_memory);torch._inductor.config.compile_threads=2;wp.config.kernel_cache_dir='/dev/shm/wuji_lm_runtime_2207_20261111/anchor_warp_cache';rows=[];peaks=[];stop=threading.Event()
    def monitor():
        while not stop.wait(1):
            if not torch.cuda.is_initialized():continue
            q=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True)
            used=sum(int(v[1]) for v in csv.reader(q.stdout.splitlines()) if len(v)>1 and v[0].strip()==str(os.getpid()) and v[1].strip().isdigit());peaks.append(used)
            if used>8192:(out/'RESOURCE_VIOLATION.json').write_text(json.dumps(dict(resident_mib=used)));os.kill(os.getpid(),signal.SIGTERM)
    threading.Thread(target=monitor,daemon=True).start()
    for config in ['builtin','hl_urdf']:
        for side in ['left','right']:
            source=np.load(w/'source_slsqp_stopping'/f'{config}_{side}.npz')
            data=np.load(w/f'{config}_{side}'/'curves.npz');q0=data['q_sdk_raw_recovered'];refraw=source['q_source_raw'];offsets=source['record_offsets'];assert q0.shape==refraw.shape
            model=EMFTrajectorySolver(r/'reference',r/'assets'/f'{config}_{side}.urdf',side,device='cuda:0');tensor=model.tensor;human=[]
            for idx in data['record_indices']:human.append(np.load(r/'sdk'/config/f'trajectory_{idx}_{side}.npz')['human_q'])
            hq=tensor(np.concatenate(human));target=prepare_robot_targets(model.sfk.forward(hq)[0],model.alignment);qref=tensor(refraw);previous=torch.cat((qref[:1],qref[:-1])).clone();previous[tensor(offsets[:-1]).long()]=tensor(model.robot.bounds.mean(1));later=torch.ones(len(qref),device='cuda',dtype=torch.bool);later[tensor(offsets[:-1]).long()]=False
            objs=[TrajectoryObjective(CompiledSpatial(RobotRetargetObjective(model.rfk,*target,stage=s)),range(len(qref)+1),previous,.01) for s in (1,2)];engines=[OneWayLM(o,qref,model.rfk.lower,model.rfk.upper,mode='frozen_parallel') for o in objs]
            def score(q):
                residual,jac,delta=objs[1].residuals(q);gradient=(jac*residual[...,None]).sum(1)+.01*delta;active=((q<=model.rfk.lower+5e-6)&(gradient>0))|((q>=model.rfk.upper-5e-6)&(gradient<0));pg=torch.where(active,0.,gradient).abs().amax(-1);cost=.5*residual.square().sum(-1)+.005*delta.square().sum(-1);error=q-qref
                return dict(raw_sdk_rmse_rad=float(error.square().mean().sqrt()),raw_sdk_max_rad=float(error.abs().max()),later_raw_sdk_rmse_rad=float(error[later].square().mean().sqrt()),later_raw_sdk_max_rad=float(error[later].abs().max()),later_above_1e4_fraction=float((error[later].abs()>1e-4).double().mean()),cost=float(cost.sum()),later_cost=float(cost[later].sum()),projected_gradient_max=float(pg.max()),later_projected_gradient_median=float(pg[later].median()),later_stationary_fraction=float((pg[later]<1e-6).double().mean()))
            baseline=score(qref)
            from retarget.learning.wuji_ik.trajectory_oneway_cuda import native_lower_scan
            residual,jac,delta=objs[1].residuals(qref);g=(jac*residual[...,None]).sum(1)+.01*delta
            free=(~(((qref<=model.rfk.lower+1e-7)&(g>0))|((qref>=model.rfk.upper-1e-7)&(g<0)))).to(qref.dtype)
            normal=jac.transpose(-1,-2)@jac+.01*torch.eye(20,device='cuda',dtype=qref.dtype)
            matrix=normal*free[:,:,None]*free[:,None,:]+torch.diag_embed(.01*free+1-free)
            rhs=-g*free;expected=torch.linalg.solve(matrix,rhs[...,None]).squeeze(-1)
            actual=native_lower_scan(matrix,torch.zeros_like(matrix),rhs,objs[1].segment,1)
            baseline['native_linear_solve_max_error_vs_fp64']=float((actual-expected).abs().max())
            arrays=dict(source_q_raw=refraw,source_previous_raw=previous.cpu().numpy(),q_network=data['q_network'],record_offsets=offsets,record_indices=data['record_indices'])
            for seed_name,initial in [('source_previous_raw',previous),('network',tensor(data['q_network'])),('source_current_raw_DIAGNOSTIC',qref)]:
                for warm in [0,1]:
                    for final in [10,200]:
                        q=initial if warm==0 else engines[0](initial,warm).q;q=engines[1](q,final).q
                        entry=dict(configuration=config,side=side,seed=seed_name,warm_updates=warm,final_updates=final,frames=len(q),baseline=baseline,**score(q));rows.append(entry);arrays[f'{seed_name}_warm{warm}_final{final}']=q.cpu().numpy();print('FIXED_HISTORY',json.dumps(entry),flush=True)
            np.savez_compressed(out/f'{config}_{side}.npz',**arrays);(out/'summary.json').write_text(json.dumps(dict(complete=len(rows)==48,rows=rows,budget=budget,peak_resident_mib=max(peaks,default=0),neural_optimizer_updates=0),indent=2));del engines,objs,model;gc.collect();torch.cuda.empty_cache()
    stop.set()

if __name__=='__main__':main()
