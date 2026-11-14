#!/usr/bin/env python3
"""EMF-only temporal initialization + independent batched human/robot LM benchmark."""
import argparse
import faulthandler
import signal
if __name__=='__main__':
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    print('SCRIPT_START',flush=True)
import time
_PROCESS_STARTED=time.perf_counter()
from datetime import datetime, timezone
import importlib
import json
import math
from pathlib import Path
import sys
import time
import types
import numpy as np
import h5py
import torch


def quaternion_matrix(q):
    q=q/q.norm(dim=-1,keepdim=True);x,y,z,w=q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),
        1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),-1).reshape(*q.shape[:-1],3,3)


def reference_math(path):
    package=types.ModuleType('wuji_lm_math');package.__path__=[str(path/'retarget_reference')]
    sys.modules[package.__name__]=package
    return tuple(importlib.import_module('wuji_lm_math.'+name) for name in ('kinematics','preprocess','objectives'))


def check_jacobian(objective,q):
    residual,jac=objective(q);columns=[]
    for index in range(q.shape[1]):
        step=torch.zeros_like(q);step[:,index]=1e-6
        columns.append((objective(q+step)[0]-objective(q-step)[0])/2e-6)
    difference=float((jac-torch.stack(columns,-1)).abs().max())
    if difference>2e-5:raise AssertionError('Analytic residual Jacobian mismatch '+str(difference))
    return difference


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--iterations',type=int,default=40)
    p.add_argument('--human-iterations',type=int,default=70);p.add_argument('--compile',action='store_true')
    p.add_argument('--memory-gib',type=float,default=8.);p.add_argument('--smoke',action='store_true')
    p.add_argument('--cuda-graphs',action='store_true')
    p.add_argument('--robokit-native',action='store_true')
    p.add_argument('--run-tag',default='')
    p.add_argument('--network-backend',choices=('auto','flex','flash','sdpa'),default='flex')
    p.add_argument('--anchor-weight',type=float,default=0.)
    p.add_argument('--anchor-radius',type=float,default=0.)
    p.add_argument('--max-step',type=float,default=.35)
    p.add_argument('--robot-options',type=Path,help='Per-configuration anchor/clip parameters; no SDK states')
    a=p.parse_args();suffix='_'+a.run_tag if a.run_tag else '';root=a.root;sys.path.insert(0,str(root/'reference'))
    from human_ik_reference.kinematics import HumanHandKinematics
    from human_ik_reference.parameters import default_parameters
    rk,pre,ref_objectives=reference_math(root/'reference')
    from retarget.learning.wuji_ik.batched_lm import (BatchURDF,FastFingerURDF,HumanPoseObjective,RobotRetargetObjective,JointAnchorObjective,CUDAGraphLM,
        bounded_lm,prepare_robot_targets,parallel_sdk_output_filter)
    from retarget.learning.wuji_ik.features import wrist_tx_transform,wrist_emf_features,joint_names
    from retarget.learning.wuji_ik.training import load_inference
    from retarget.learning.wuji_ik.metrics import autocast_context
    assert not any(k=='wuji_sdk' or k.startswith('wuji_sdk.') for k in sys.modules)
    torch.set_num_threads(2);device=torch.device('cuda:0');torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(a.memory_gib*1024**3/torch.cuda.get_device_properties(device).total_memory,device)
    torch._inductor.config.compile_threads=2
    torch._dynamo.config.recompile_limit=64
    if a.robokit_native:
        import warp as wp
        from retarget.learning.wuji_ik.robokit_lm import RobokitCUDALM
        wp.config.kernel_cache_dir=str(root/'warp_cache')
        print('NATIVE_RUNTIME',torch.__version__,wp.__version__,flush=True)
        target=torch.tensor([[.2,-.3],[.5,.7]],device=device,dtype=torch.float64)
        class NativeCheck:
            def __call__(self,q):return q-target,torch.eye(2,device=device,dtype=q.dtype).expand(len(q),-1,-1)
        engines=[RobokitCUDALM(NativeCheck(),iterations=20,compile=a.compile) for _ in range(2)]
        check_initial=torch.zeros_like(target);check_lower=target.new_full((2,),-1.)
        for engine in engines:engine(check_initial,check_lower,-check_lower)
        target.copy_(target.flip(0)*.8)
        for engine in engines+engines:
            torch.testing.assert_close(engine(check_initial,check_lower,-check_lower).q,target,atol=1e-6,rtol=0)
        print('NATIVE_CROSS_GRAPH_FRESH_INPUT_PASS',flush=True)
        del engines
    if a.cuda_graphs:
        target=torch.tensor([[.2,-.3],[.5,.7]],device=device,dtype=torch.float64)
        class TinyObjective:
            def __call__(self,q):return q-target,torch.eye(2,device=device,dtype=q.dtype).expand(len(q),-1,-1)
        engine=CUDAGraphLM(TinyObjective(),iterations=20)
        initial=torch.zeros_like(target);lower=target.new_full((2,),-1.);upper=-lower
        first=engine(initial,lower,upper).q
        torch.testing.assert_close(first,target,atol=1e-7,rtol=0)
        target.copy_(target.flip(0)*.8)
        second=engine(initial,lower,upper).q
        torch.testing.assert_close(second,target,atol=1e-7,rtol=0)
        if torch.allclose(first,second):raise AssertionError('Graph replay reused a cached solution')
        del engine
        print('CUDA_GRAPH_NEW_INPUT_VALIDATED',flush=True)
    model,normalizer,state=load_inference(a.checkpoint,device,backend=a.network_backend)
    if state['variant'] not in ('temporal_causal','temporal_bidirectional') or state['model_config']['width']!=1024 or state['model_config']['layers']!=4:
        raise ValueError('Expected a completed four-block temporal model')
    print('MODEL_LOADED',state['variant'],state['step'],flush=True)
    selection=json.loads((root/'selection.json').read_text());groups=[];checks=[]
    configs=('builtin',) if a.smoke else ('builtin','hl_urdf')
    sides=('right',) if a.smoke else ('left','right')
    for configuration in configs:
        for side in sides:
            urdf=root/'assets'/f'{configuration}_{side}.urdf';human=HumanHandKinematics(side,urdf,sdk_tx_rounding=True)
            human_fk=FastFingerURDF(urdf,human.joint_names,['thumb_rx_coil','index_rx_coil','middle_rx_coil','ring_rx_coil','pinky_rx_coil'],device=device,root_transform=human._tx_inverse)
            skeleton_fk=BatchURDF(urdf,human.joint_names,human.skeleton_links,device=device)
            robot=rk.HandKinematics('WujiHand2',side)
            if list(robot.joint_names)!=joint_names(side):raise ValueError('Robot joint order mismatch')
            robot_fk=FastFingerURDF(robot.urdf_path,robot.joint_names,robot.landmark_names,device=device)
            sdk_bounds=normalizer.limits[0 if side=='right' else 1].cpu().numpy()
            np.testing.assert_allclose(sdk_bounds,np.stack((robot.lower,robot.upper),-1),atol=1e-6,rtol=0)
            parameters=default_parameters(side);seed=np.concatenate(parameters['seed']);bounds=np.concatenate(parameters['bounds'])
            inputs=[];offsets=[0];records=[]
            for item in selection['trajectories'][:1] if a.smoke else selection['trajectories']:
                input_path=root/item.get('cached_h5_relative',f'inputs/trajectory_{item["index"]}/wuji_glove_offline.h5')
                if not input_path.exists():input_path=Path(item['cached_h5'])
                if not input_path.exists():input_path=Path(item['source_h5'])
                with h5py.File(input_path,'r',locking=False) as h:
                    g=h[side];length=min(8,len(g['source_timestamp_ns'])) if a.smoke else len(g['source_timestamp_ns'])
                    observed=dict(position=g['emf_poses/position'][:length],quaternion=g['emf_poses/orientation_xyzw'][:length],
                        confidence=g['emf_poses/confidence'][:length],timestamp_ns=g['source_timestamp_ns'][:length],source_seq=g['source_seq'][:length])
                inputs.append(observed);offsets.append(offsets[-1]+length);records.append(item)
            position=np.concatenate([x['position'] for x in inputs]);quaternion=np.concatenate([x['quaternion'] for x in inputs]);n=len(position)
            tensor=lambda x:torch.as_tensor(x,device=device,dtype=torch.float64)
            default=tensor(seed);seeds=[default.clone(),tensor(np.clip(np.zeros(21),bounds[:,0],bounds[:,1])),default.clone()]
            sign=-1 if side=='left' else 1
            for f in range(4):seeds[2][5+4*f:9+4*f]=tensor([sign*.8,0,sign*1.1,sign*.7])
            seeds[2][3:5]=tensor([sign*.8,sign*.7])
            initial=torch.stack(seeds)[:,None].expand(-1,n,-1).reshape(3*n,21)
            hp=tensor(position).repeat(3,1,1);hr=quaternion_matrix(tensor(quaternion)).repeat(3,1,1,1)
            posture=tensor(np.ones(21));posture[:5]=.1
            posture[torch.tensor([8,12,16,20],device=device)]=0
            human_objective=HumanPoseObjective(human_fk,hp,hr,default,posture)
            # The wrist / index MCP / middle MCP alignment is fixed by the URDF.
            neutral=human.forward(seed)['skeleton'][:,:3,3].astype(np.float32).astype(np.float64)
            operator=pre.OPERATOR2MANO_RIGHT if side=='right' else pre.OPERATOR2MANO_LEFT
            rz=np.array([[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]])
            alignment=tensor(pre.estimate_frame(neutral)@operator@rz.T)
            qtest=default[None]+tensor(np.random.default_rng(42).uniform(-.02,.02,(2,21)))
            ref_human=np.stack([human.forward(q.cpu().numpy())['tx_rx'][:,:3,3] for q in qtest])
            hdiff=float((human_fk.forward(qtest)[0]-tensor(ref_human)).abs().max())
            hobj=HumanPoseObjective(human_fk,hp[:2],hr[:2],default,posture)
            hjac=check_jacobian(hobj,qtest)
            qrobot=tensor(robot.bounds.mean(1))[None].expand(2,-1).clone()+.012
            pd,jd=robot_fk.forward(qrobot)[::2]
            rr=[robot.forward(q.cpu().numpy()) for q in qrobot]
            rdiff=float((pd-tensor(np.stack([v[0] for v in rr]))).abs().max())
            jdiff=float((jd-tensor(np.stack([v[1] for v in rr]))).abs().max())
            if max(hdiff,rdiff,jdiff)>1e-10:raise AssertionError('URDF FK mismatch')
            sample_skeleton=skeleton_fk.forward(qtest)[0]
            target=prepare_robot_targets(sample_skeleton,alignment)
            for i in range(2):
                source=pre.preprocess(sample_skeleton[i].cpu().numpy(),'WujiHand2',side)
                prepared=pre.prepare_targets(source,thumb_rel_skip_proximal=2)
                for value,expected in [(target[0][i],source),(target[1][i],prepared['L']),
                    (target[2][i],prepared['D']),(target[4][i],prepared['directions'])]:
                    np.testing.assert_allclose(value.cpu(),expected,rtol=1e-8,atol=1e-9)
            objective=RobotRetargetObjective(robot_fk,*target)
            rjac=check_jacobian(objective,qrobot)
            r,j=objective(qrobot)
            reference=ref_objectives.FrameObjective(target[0][0].cpu().numpy(),target[3].cpu().numpy(),
                laplacian=target[1][0].cpu().numpy(),differential=target[2][0].cpu().numpy(),
                weights=[1,0,0,0]+[1]*17,thumb_triplets=[(2,3,4)],
                params=dict(lambda_smooth=0,lambda_thumb_rel=.5,thumb_rel_off_rad=.5),antirev_indices=robot.antirev_indices)
            ref_cost,ref_gradient,_=reference.evaluate(qrobot[0].cpu().numpy(),rr[0][0],rr[0][1])
            cost_difference=abs(float(.5*r[0].square().sum())-ref_cost)
            gradient_difference=float(np.abs((j[0].T@r[0]).cpu().numpy()-ref_gradient).max())
            if cost_difference>1e-8 or gradient_difference>1e-7:raise AssertionError('SDK spatial objective port mismatch')
            checks.append(dict(configuration=configuration,side=side,human_fk_max_m=hdiff,robot_fk_max_m=rdiff,
                robot_fk_jac_max=jdiff,human_residual_jac_max=hjac,robot_residual_jac_max=rjac,
                reference_cost_difference=cost_difference,reference_gradient_difference=gradient_difference))
            groups.append(dict(configuration=configuration,side=side,human=human,robot=robot,inputs=inputs,offsets=offsets,records=records,
                human_fk=human_fk,skeleton_fk=skeleton_fk,robot_fk=robot_fk,initial=initial,default=default,
                human_lower=tensor(bounds[:,0]),human_upper=tensor(bounds[:,1]),human_objective=human_objective,
                alignment=alignment,wrist_tx=wrist_tx_transform(urdf),position_cpu=position,quaternion_cpu=quaternion,n=n))
    (root/'numerical_validation.json').write_text(json.dumps(checks,indent=2));print('NUMERICAL_VALIDATION_PASSED',checks,flush=True)
    compile_fn=lambda fn:torch.compile(fn,fullgraph=True,dynamic=True) if a.compile else fn
    def solve(group,key,objective,initial,lower,upper,iterations):
        max_step=group.get('robot_max_step',.35) if key.startswith('stage') else .35
        if a.robokit_native:
            if key not in group:group[key]=RobokitCUDALM(objective,iterations=iterations,compile=a.compile,max_step=max_step)
            return group[key](initial,lower,upper)
        if not a.cuda_graphs:return bounded_lm(compile_fn(objective),initial,lower,upper,iterations=iterations,max_step=max_step)
        if key not in group:group[key]=CUDAGraphLM(compile_fn(objective),iterations=iterations,max_step=max_step)
        return group[key](initial,lower,upper)
    def synchronize():torch.cuda.synchronize()
    @torch.no_grad()
    def run_once(label,save=False):
        details=[];outputs=[]
        for group in groups:
            c,side,n=group['configuration'],group['side'],group['n']
            print('RUN_GROUP',label,c,side,n,flush=True)
            synchronize();started=time.perf_counter();predictions=[]
            # Full independent episodes share one padded GPU forward; lengths mask padding.
            lengths=torch.tensor([len(observed['position']) for observed in group['inputs']],device=device)
            features_cpu=[torch.from_numpy(wrist_emf_features(observed['position'],observed['quaternion'],group['wrist_tx']))
                          for observed in group['inputs']]
            features=torch.nn.utils.rnn.pad_sequence(features_cpu,batch_first=True).to(device)
            side_tensor=torch.full((len(features),),0 if side=='right' else 1,device=device,dtype=torch.long)
            with autocast_context(device):
                qn=model(features,side_tensor,lengths)
            q_batch=normalizer.denormalize(qn.float(),side_tensor).double()
            predictions=[q_batch[i,:len(observed['position'])] for i,observed in enumerate(group['inputs'])]
            prediction=torch.cat(predictions);synchronize();nn_end=time.perf_counter()
            if label=='warmup':print('PHASE',c,side,'network_complete',flush=True)
            group['human_objective'].position.copy_(torch.as_tensor(group['position_cpu'],device=device,dtype=torch.float64).repeat(3,1,1))
            group['human_objective'].rotation.copy_(quaternion_matrix(torch.as_tensor(group['quaternion_cpu'],device=device,dtype=torch.float64)).repeat(3,1,1,1))
            human_result=solve(group,'human_engine',group['human_objective'],group['initial'],group['human_lower'],group['human_upper'],a.human_iterations)
            candidates=human_result.q.reshape(3,n,21)
            residual=group['human_objective'](human_result.q)[0].reshape(3,n,-1)
            pose_cost=residual[:,:,:30].reshape(3,n,5,6).square().sum(-1)
            finger_q=[]
            for f,sl in enumerate(group['human'].finger_slices):
                cost=pose_cost[:,:,f]+residual[:,:,30+sl.start:30+sl.stop].square().sum(-1)
                best=cost.argmin(0);finger_q.append(candidates[best,torch.arange(n,device=device),sl])
            human_q=torch.cat(finger_q,-1);skeleton=group['skeleton_fk'].forward(human_q)[0]
            target=prepare_robot_targets(skeleton,group['alignment']);synchronize();human_end=time.perf_counter()
            if label=='warmup':print('PHASE',c,side,'human_lm_complete',flush=True)
            stage1=RobotRetargetObjective(group['robot_fk'],*target,stage=1)
            stage2=RobotRetargetObjective(group['robot_fk'],*target,stage=2)
            if a.cuda_graphs or a.robokit_native:
                if 'stage1_objective' in group:
                    group['stage1_objective'].copy_targets_(stage1);group['stage2_objective'].copy_targets_(stage2)
                else:group['stage1_objective']=stage1;group['stage2_objective']=stage2
                stage1,stage2=group['stage1_objective'],group['stage2_objective']
            options=json.loads(a.robot_options.read_text()).get(c,{}) if a.robot_options else {}
            weight=options.get('weight',a.anchor_weight);radius=options.get('radius',a.anchor_radius);max_step=options.get('step',a.max_step)
            if not math.isfinite(weight) or weight<0 or not math.isfinite(max_step) or max_step<=0 or (radius is not None and (not math.isfinite(radius) or radius<0)):
                raise ValueError('Invalid anchor or clipping parameter')
            group['robot_max_step']=max_step
            if weight:
                if 'network_anchor' not in group:group['network_anchor']=prediction.clone()
                else:group['network_anchor'].copy_(prediction)
                stage1=JointAnchorObjective(stage1,group['network_anchor'],weight);stage2=JointAnchorObjective(stage2,group['network_anchor'],weight)
            lower,upper=group['robot_fk'].lower,group['robot_fk'].upper
            if radius:
                lower=torch.maximum(lower,prediction-radius);upper=torch.minimum(upper,prediction+radius)
            first=solve(group,'stage1_engine',stage1,prediction,lower,upper,min(25,a.iterations))
            if label=='warmup':print('PHASE',c,side,'robot_stage1_complete',flush=True)
            seed_cost=.5*stage2(prediction)[0].square().sum(-1);first_cost=.5*stage2(first.q)[0].square().sum(-1)
            initial=torch.where((first_cost<seed_cost)[:,None],first.q,prediction)
            second=solve(group,'stage2_engine',stage2,initial,lower,upper,a.iterations)
            synchronize();robot_end=time.perf_counter()
            filtered=torch.cat([parallel_sdk_output_filter(second.q[s:e]) for s,e in zip(group['offsets'][:-1],group['offsets'][1:])])
            synchronize();end=time.perf_counter()
            timing=dict(configuration=c,side=side,frames=n,network_seconds=nn_end-started,
                human_lm_and_targets_seconds=human_end-nn_end,robot_lm_seconds=robot_end-human_end,
                filter_seconds=end-robot_end,end_to_end_seconds=end-started,
                human_accepted_mean=None if a.robokit_native else float(human_result.accepted_steps.float().mean()),robot_accepted_mean=None if a.robokit_native else float(second.accepted_steps.float().mean()),
                robot_converged_fraction=None if a.robokit_native else float(second.converged.float().mean()),
                initial_robot_cost_mean=float(seed_cost.mean()),final_robot_cost_mean=float(second.cost.mean()))
            details.append(timing);print('TIMING',label,json.dumps(timing),flush=True)
            outputs.append((prediction,second.q,filtered,human_q,skeleton,target[0],seed_cost,second.cost))
            if save:
                out=root/('parallel_robokit'+suffix if a.robokit_native else 'parallel_graph' if a.cuda_graphs else 'parallel')/c;out.mkdir(parents=True,exist_ok=True)
                for index,(s,e) in enumerate(zip(group['offsets'][:-1],group['offsets'][1:])):
                    observed=group['inputs'][index]
                    np.savez_compressed(out/f'trajectory_{group["records"][index]["index"]}_{side}.npz',
                        **{name:value[s:e].cpu().numpy() for name,value in zip(
                            ('q_network_rad','q_lm_raw_rad','q_lm_filtered_rad','human_q_lm_rad','skeleton_lm_m','source_aligned_m','initial_robot_cost','final_robot_cost'),outputs[-1])},
                        timestamp_ns=observed['timestamp_ns'],source_seq=observed['source_seq'])
                (out/f'{side}_lm_trace.json').write_text(json.dumps(dict(human=human_result.history,robot_stage1=first.history,robot_stage2=second.history),indent=2))
        return dict(groups=details,end_to_end_seconds=sum(x['end_to_end_seconds'] for x in details)),outputs
    timings=[];synchronize();cold_started=time.perf_counter()
    warmup,reference_outputs=run_once('warmup',save=True)
    cold_seconds=time.perf_counter()-cold_started
    first_result_seconds=time.perf_counter()-_PROCESS_STARTED
    for repeat in range(1 if a.smoke else a.repeats):
        measured,outputs=run_once('measured_'+str(repeat),save=False)
        for actual,expected in zip(outputs,reference_outputs):
            for x,y in zip(actual,expected):torch.testing.assert_close(x,y,atol=1e-7,rtol=1e-7)
        timings.append(measured)
    report=dict(created_at=datetime.now(timezone.utc).isoformat(),checkpoint=str(a.checkpoint),checkpoint_step=state['step'],
        variant=state['variant'],checkpoint_completed_epochs=state['epoch'],
        information='past_and_present' if state['variant']=='temporal_causal' else 'past_present_and_future',
        checkpoint_data_signature=state['data_signature'],compile_enabled=a.compile,smoke=a.smoke,
        cuda_graphs_enabled=a.cuda_graphs or a.robokit_native,robokit_native=a.robokit_native,
        run_tag=a.run_tag,robot_options=json.loads(a.robot_options.read_text()) if a.robot_options else dict(anchor_weight=a.anchor_weight,anchor_radius=a.anchor_radius,max_step=a.max_step),lm_backend='robokit_native_cuda_cholesky' if a.robokit_native else 'torch_cuda',
        human_iterations=a.human_iterations,robot_iterations=a.iterations,network_backend=model.core.backend_report(),
        first_result_including_import_model_urdf_validation_compilation_and_capture_seconds=first_result_seconds,
        cold_warmup_seconds_including_compile_and_result_capture=cold_seconds,warmup=warmup,steady_repeats=timings,
        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu_name=torch.cuda.get_device_name(),
        torch_version=torch.__version__,scope='All native frames; full '+state['variant']+' EMF-only model per complete episode, independent batched human IK and robot LM, no serial q feedback. SDK intermediates never enter the solve.',
        differences_from_sdk='Human history penalty and robot previous-solution regularizer excluded to permit independent-frame LM. Physiological/posture bounds and SDK spatial retarget weights retained. Raw and parallel alpha=.2 output-filtered variants are both saved.',
        timing='CPU-resident raw EMF, GPU synchronized wall clock. Includes feature construction, host-to-device transfers, model, human LM, topology preparation, robot LM and parallel output filter; excludes disk I/O, URDF/model loading, validation and metric calculation.')
    (root/('smoke_report.json' if a.smoke else 'benchmark_robokit'+suffix+'_report.json' if a.robokit_native else 'benchmark_graph_report.json' if a.cuda_graphs else 'benchmark_report.json')).write_text(json.dumps(report,indent=2))
    if not a.smoke:(root/('ROBOKIT_PARALLEL'+suffix.upper()+'_COMPLETE' if a.robokit_native else 'PARALLEL_GRAPH_COMPLETE' if a.cuda_graphs else 'PARALLEL_COMPLETE')).write_text('complete\n')
    print('BENCHMARK_COMPLETE',json.dumps(dict(smoke=a.smoke,cold_seconds=cold_seconds,steady=[x['end_to_end_seconds'] for x in timings])),flush=True)


if __name__=='__main__':main()
