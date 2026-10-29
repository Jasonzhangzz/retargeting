#!/usr/bin/env python3
"""Paired mjwarp action-solver tests at freshly optimized SPIDER checkpoints.

Every method uses the same measured state, knot seed, full objective and frozen
contact block. These local tests do not feed solver states into later cases.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
import torch
import warp as wp
from retarget.fa.adamw import AdamW
from retarget.humanoid import HumanoidProblem, Settings
from retarget.humanoid_mjwarp import MJWarpHumanoid
from retarget.humanoid_terms import stack_host
from retarget.humanoid_sample_lm import SampleLM
from retarget.humanoid_contacts import update
from run_humanoid import task_paths


from retarget.humanoid_lm_oracle import MJWarpResidualOracle as BatchOracle

def fit_controls(backend,dense):
    """Project source's 48 dense controls into the common 8-knot action space."""
    A=np.zeros((backend.dense_steps,backend.knots));boundary=np.zeros((backend.dense_steps,backend.problem.model.nu))
    previous=backend.previous.numpy()
    for i,(left,right,fraction) in enumerate(backend.maps):
        A[i,right]+=fraction
        if left>=0:A[i,left]+=1-fraction
        else:boundary[i]=(1-fraction)*previous
    dense=np.clip(dense,backend.problem.lower,backend.problem.upper)
    knots=np.linalg.lstsq(A,dense-boundary,rcond=None)[0]
    fitted=np.clip(A@knots+boundary,backend.problem.lower,backend.problem.upper)
    x=(knots-backend.knot_reference.numpy()).reshape(1,-1)/backend.settings.control_scale
    return x,float(np.sqrt(np.mean((fitted-dense)**2)))


def freeze_contacts(b,x,seed):
    shape=(b.dense_steps+1,1,4,3)
    z=np.broadcast_to(b.anchors[:,None],shape).copy();dual=np.zeros(shape)
    b.targets.z.assign(z);b.targets.dual.zero_();b.targets.forces.zero_()
    b.targets.floor_force.zero_();b.targets.floor_torque.zero_();b.dirty=True
    b.targets.penalty.assign(np.array([b.settings.rho,b.settings.anchor_weight],np.float32))
    before=b.evaluate(x,False)[0][0]
    candidate,force,floor,torque,_=update(b,b.snapshots(),z,dual,np.random.default_rng(seed))
    for name,value in [('z',candidate),('forces',force),('floor_force',floor),('floor_torque',torque)]:
        getattr(b.targets,name).assign(np.asarray(value,np.float32))
    b.dirty=True;after=b.evaluate(x,False)[0][0]
    if not np.isfinite(after) or after>before+1.e-6:
        b.targets.z.assign(z);b.targets.forces.zero_();b.targets.floor_force.zero_();b.targets.floor_torque.zero_();b.dirty=True
    return bool(np.isfinite(after) and after<=before+1.e-6)


METHODS=('adamw1','adamw2','adamw3','adamw8','sampling64','lm32','lm232','sampling64_lm32','sampling64_lm32x2','sampling64_fit_lm32')


def solve(name,x,oracle,seed):
    b=oracle.parent
    if name.startswith('adamw'):
        iterations=int(name[5:]);optimizer=AdamW(device=b.device,learning_rate=.3,weight_decay=.01)
        result,_,diag=optimizer.update(x.cpu().numpy(),b.evaluate,iterations=iterations,horizon=b.knots,
            advance=0.,evaluate_on_device=True)
        return torch.as_tensor(result,device=x.device),diag
    if name=='sampling64':
        generator=torch.Generator(device=x.device).manual_seed(seed)
        noise=torch.randn((64,x.shape[1]),device=x.device,dtype=x.dtype,generator=generator)
        candidates=torch.cat((x,x+noise),dim=0) # .05 rad / control_scale .05
        values,_=oracle(candidates)
        selected=int(values.argmin())
        return candidates[selected:selected+1],dict(samples=64,value_after=float(values[selected]))
    options=dict(rank=32,samples=0,iterations=1,seed=seed)
    if name=='lm232':options['rank']=x.shape[1]
    if name.startswith('sampling64_'):options['samples']=64
    if name.endswith('x2'):options['iterations']=2
    if name=='sampling64_fit_lm32':options['reuse_sample_residuals']=True
    return SampleLM(**options).solve(x,oracle)


def benchmark(reference,output,windows=None,perturbations=(0.,.03),methods=METHODS,dense_base=True):
    metadata=json.loads((reference/'summary.json').read_text());task=metadata['task']
    if not metadata['feasible_under_balance_audit']:
        raise ValueError('Source reference did not pass the independent balance audit')
    data=np.load(reference/'trajectory.npz')
    scene,raw=task_paths(ROOT/'data/spider',task)
    settings=Settings(multistarts=1,admm_iterations=1,trajectory_cache=False,adamw_lr=.3)
    p=HumanoidProblem(scene,raw,ref_dt=2*metadata['config']['sim_dt'],settings=settings)
    b=MJWarpHumanoid(p);oracle=BatchOracle(b)
    torch.set_num_threads(1)
    output.mkdir(parents=True,exist_ok=False)
    indices=list(range(len(data['checkpoint_qpos']))) if windows is None else list(windows)
    rows=[];all_actions=[];all_states=[]
    setup=time.perf_counter();warmed=False
    for index in indices:
        state=mujoco.MjData(p.model)
        state.qpos[:]=data['checkpoint_qpos'][index];state.qvel[:]=data['checkpoint_qvel'][index]
        state.ctrl[:]=data['checkpoint_previous_ctrl'][index];state.time=data['checkpoint_time'][index]
        mujoco.mj_forward(p.model,state)
        first=1+index*metadata['config']['ctrl_steps']//2
        tick=time.perf_counter();b.prepare(state,first,state.ctrl)
        if dense_base:
            b.use_dense_control_base(data['planned_controls'][index])
            base=np.zeros(b.variables.shape);projection_error=0.
        else:
            base,projection_error=fit_controls(b,data['planned_controls'][index])
        accepted_contact=freeze_contacts(b,base,100+index)
        oracle.sync();torch.cuda.synchronize()
        preparation=time.perf_counter()-tick
        # Both evaluator widths must compute the same fixed action problem.
        baseline=torch.as_tensor(base,device=str(b.device),dtype=torch.float64)
        primary,_=oracle(baseline,True)
        if dense_base:
            np.testing.assert_array_equal(stack_host(b.controls)[:,0],np.clip(data['planned_controls'][index],p.lower,p.upper).astype(np.float32))
        batch_value,batch_r=oracle(baseline.expand(2,-1),True)
        torch.testing.assert_close(batch_value,primary.expand_as(batch_value),rtol=5.e-6,atol=2.e-5)
        torch.testing.assert_close(batch_r.double().square().sum(dim=1),batch_value,rtol=5.e-6,atol=2.e-5)
        if not warmed:
            for name in methods:
                solve(name,baseline,oracle,4000+index)
            torch.cuda.synchronize();warmed=True
        for perturbation in perturbations:
            rng=np.random.default_rng(7000+index)
            x=baseline+torch.as_tensor(rng.normal(0,perturbation/settings.control_scale,baseline.shape),device=baseline.device)
            before=float(b.evaluate(x,False)[0][0]);order=list(methods)
            np.random.default_rng(9000+index).shuffle(order)
            for name in order:
                # A fresh callback starts at exactly the same input for each method.
                b.evaluate(x,False)
                b.forward_calls=b.backward_calls=oracle.backend.forward_calls=0;oracle.reset_counts()
                torch.cuda.synchronize();tick=time.perf_counter()
                answer,diag=solve(name,x,oracle,4000+index)
                final,_=b.evaluate(answer,False)
                torch.cuda.synchronize();seconds=time.perf_counter()-tick
                if float(final[0])>before+max(1.e-4,1.e-6*abs(before)):
                    raise AssertionError(f'{name} exported a worse actual rollout')
                snapshot=b.snapshots();controls=stack_host(b.controls)[:,0]
                nextq=b.steps[settings.substeps-1].recorded.result.qpos.numpy()[0]
                terminalq=b.steps[-1].recorded.result.qpos.numpy()[0]
                mask=p.mask[b.reference_indices]
                error=np.linalg.norm(snapshot['x'][:,0]-b.targets.z.numpy()[:,0],axis=-1)
                row=dict(task=task,checkpoint=index,reference_frame=first,action_noise_std_rad=perturbation,
                    solver=name,seconds=seconds,value_before=before,value_after=float(final[0]),
                    relative_reduction=(before-float(final[0]))/max(abs(before),1.e-12),
                    first_frame_root_height_m=float(nextq[2]),terminal_root_height_m=float(terminalq[2]),
                    contact_rms_m=float(np.sqrt(np.mean(error[mask]**2))) if mask.any() else 0.,
                    first_action_change_rms_rad=float(np.sqrt(np.mean((controls[:2]-np.clip(data['planned_controls'][index,:2],p.lower,p.upper))**2))),
                    source_control_projection_rms_rad=projection_error,contact_block_update_accepted=accepted_contact,
                    shared_preparation_seconds=preparation,primary_forward_calls=b.forward_calls,
                    analytic_backward_calls=b.backward_calls,batch_forward_calls=oracle.backend.forward_calls,
                    requested_oracle_trajectories=oracle.evaluated_trajectories,
                    padded_oracle_trajectories=oracle.padded_trajectories,diagnostics=diag)
                rows.append(row);all_actions.append(controls);all_states.append(nextq)
                print(task,index,perturbation,name,'ms',round(seconds*1000,2),'reduction',round(row['relative_reduction'],4),flush=True)
            (output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
            np.savez_compressed(output/'solutions.npz',controls=all_actions,next_qpos=all_states)
    config=dict(source_reference=str(reference),source_summary=metadata,settings=asdict(settings),weights=asdict(p.weights),
        methods=list(methods),checkpoints=indices,action_noise_std_rad=list(perturbations),
        physics='MJWarp for every paired solver; SPIDER MJWarp supplies the reference states/actions',
        action_parameterization=('Exact fixed SPIDER dense nominal plus identical 8-knot action increments' if dense_base
            else 'Same 8-knot least-squares projection of clipped SPIDER dense control for all methods'),
        objective='Unchanged full mjwarp whole-body objective with one frozen accepted joint contact block per checkpoint',
        timing='Synchronized serial warm action solve plus final mjwarp rollout; excludes common contact/reference preparation and diagnostics',
        analytical_adamw_starts=1,notes=['Independent local tests, not a closed-loop validation of the new solvers.',
            'Missing upstream contact labels: source trajectory used original pose reward with contact reward disabled.',
            'State qpos/qvel are shared exactly; qacc_warmstart is saved for source replay but is not part of MJWarp state.',
            'MJWarp targets use the existing reference interpolation; this is not a claim of reproducing SPIDER reward or control parameterization.'])
    (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    print('Saved',output,'rows',len(rows),'elapsed',time.perf_counter()-setup,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('reference',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--windows',type=int,nargs='+')
    p.add_argument('--perturbations',type=float,nargs='+',default=[0.,.03])
    p.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    p.add_argument('--dense-base',action=argparse.BooleanOptionalAction,default=True)
    a=p.parse_args();benchmark(a.reference,a.output,a.windows,a.perturbations,a.methods,a.dense_base)
