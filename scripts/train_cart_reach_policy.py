"""Synchronous multi-GPU Muon policy training, exact global batch, no geometry net."""
import argparse,hashlib,json,math,os,signal,sys,time
from pathlib import Path
from datetime import timedelta
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--reference-config',type=Path,required=True)
    p.add_argument('--run-id',required=True);p.add_argument('--micro-batch',type=int,default=512)
    p.add_argument('--effective-batch',type=int,default=8192);p.add_argument('--epochs',type=int,default=1000)
    p.add_argument('--preflight-only',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--unbounded-wrist',action='store_true',default=True)
    p.add_argument('--observation-chunk-size',type=int,default=128)
    p.add_argument('--single-step',action='store_true',default=True,help='One gradient step after one detached policy warmup')
    p.add_argument('--tactile-encoder',choices=['dense','pointnet'],default='pointnet')
    p.add_argument('--policy-width',type=int,default=384)
    p.add_argument('--decoder-width',type=int,default=744)
    p.add_argument('--delta-direction-weight',type=float,default=.01,help='GeoRT-style local motion direction; opt in for single-step PointNet')
    p.add_argument('--delta-direction-samples',type=int,default=128,help='Auxiliary paired examples per micro-batch')
    p.add_argument('--on-policy-state-depth',type=int,default=16,help='Cycle detached warmup depths 0..N-1; 0 retains legacy one-step warmup')
    p.add_argument('--initial-noise-rad',type=float,default=0.)
    p.add_argument('--step-noise-rad',type=float,default=0.)
    args=p.parse_args()
    if any(not math.isfinite(v) or not 0<=v<=.1 for v in (args.initial_noise_rad,args.step_noise_rad)):
        p.error('Joint noise amplitudes must be finite and within [0, 0.1] rad')
    if args.on_policy_state_depth not in (0,16) or (args.on_policy_state_depth and not args.single_step):
        p.error('On-policy state depth requires single-step mode and depth16')
    if not math.isfinite(args.delta_direction_weight) or args.delta_direction_weight<0 or args.delta_direction_samples<1:
        p.error('Invalid direction regularizer settings')
    if args.delta_direction_weight and (not args.single_step or args.tactile_encoder!='pointnet'):
        p.error('Direction regularizer requires --single-step --tactile-encoder pointnet')
    import numpy as np
    import torch,torch.distributed as dist
    import warp as wp
    from moxian_web.cart_data import load_sessions
    from moxian_web.cart_initialization import attach_ik
    from moxian_web.cart_card_cost import attach_card_targets
    from moxian_web.cart_training import windows,position_statistics,predict_trajectory,save_checkpoint,perturb_rollout_state
    from moxian_web.cart_fast import prepare_windows,PreparedBatch
    from moxian_web.cart_wrist import FloatingCartHand
    from moxian_web.cart_palm_alignment import PalmAlignedNetwork,alignment_from_topology,unbounded_wrist_alignment
    from moxian_web.cart_reach_loss import ReachObjective,SPEC,VERSION
    from moxian_web.cart_optim import policy_optimizer,scale_learning_rates,cosine_multiplier
    from moxian_web.cart_tracking import CartTracking
    from retarget.baseline.geort.training import atomic_json
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    torch.cuda.set_device(local);device=f'cuda:{local}';torch.set_num_threads(1)
    if world>1:dist.init_process_group('nccl',timeout=timedelta(minutes=30),device_id=torch.device(device))
    wp.init();stream=torch.cuda.Stream(device=device);torch.cuda.set_stream(stream);wp.set_stream(wp.stream_from_torch(stream))
    if args.effective_batch!=8192:raise ValueError('This authorized run requires effective batch8192')
    if args.effective_batch%(args.micro_batch*world):raise ValueError('Batch must divide exactly')
    accum=args.effective_batch//(args.micro_batch*world)
    folder=args.root/'models'/args.run_id;folder.mkdir(parents=True,exist_ok=True)
    stopped=False;updates=0;began=time.time();tracking=None;best=None
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def check():
        if stopped or (folder/'STOP_REQUESTED').exists():raise InterruptedError('User stop')
    def progress(phase,**kw):
        if rank:return
        row=dict(status='running',phase=phase,run_id=args.run_id,pid=os.getpid(),policy_updates=updates,elapsed_s=time.time()-began);row.update(kw)
        atomic_json(folder/'status.json',row)
        with (folder/'trace.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    def barrier():
        if world>1:dist.barrier()
    progress('loading_data',world_size=world,micro_batch=args.micro_batch,accumulation=accum,effective_batch=args.effective_batch)
    ref=json.loads(args.reference_config.read_text());torch.manual_seed(ref['seed']);rng=np.random.default_rng(ref['seed'])
    sessions,source=load_sessions(args.root,ref['train_ids'],'train',require_complete=True)
    validation,other=load_sessions(args.root,ref['validation_ids'],'validation',require_complete=True)
    assert source==other and not set(ref['train_ids'])&set(ref['validation_ids'])
    attach_ik(args.root,sessions+validation)
    # One writer populates/validates caches first; other ranks only reuse the result.
    if rank==0:attach_card_targets(args.root,sessions+validation,device,check,lambda phase,**v:progress(phase,**v))
    barrier()
    if rank:attach_card_targets(args.root,sessions+validation,device,check)
    train_windows=windows(sessions,16,1);val_windows=windows(validation,16,16)
    if len(train_windows)<8192:raise ValueError('Need at least8192 distinct windows at stride1')
    hand=FloatingCartHand(device);objective=ReachObjective(hand)
    train_rollout=predict_trajectory
    if args.single_step:
        from moxian_web.cart_single_step import predict_single_step,SingleStepReachObjective
        objective=SingleStepReachObjective(hand);train_rollout=predict_single_step
    groups=None
    if args.tactile_encoder=='pointnet':
        from moxian_web.cart_pointnet import tactile_groups_from_topology
        groups=tactile_groups_from_topology(json.loads((args.root/'recordings'/ref['train_ids'][0]/'topology.json').read_text()))
        for sid in ref['train_ids']+ref['validation_ids']:
            other_groups=tactile_groups_from_topology(json.loads((args.root/'recordings'/sid/'topology.json').read_text()))
            if groups!=other_groups:raise ValueError('Inconsistent anatomical taxel assignments')
    alignment=alignment_from_topology(hand,args.root/'recordings'/ref['train_ids'][0]/'topology.json')
    if args.unbounded_wrist:alignment=unbounded_wrist_alignment(alignment)
    mean,std=position_statistics(sessions)
    model=PalmAlignedNetwork(hand.low,hand.high,'attention',alignment=alignment,position_mean=mean,position_scale=std,
        width=args.policy_width,decoder_width=args.decoder_width,
        tactile_encoder=args.tactile_encoder,tactile_groups=groups).to(device)

    if world>1:
        for value in model.state_dict().values():dist.broadcast(value,src=0)
    model.observation_chunk_size=args.observation_chunk_size
    model.set_attention_precision('bfloat16');model.checkpoint_observations=args.tactile_encoder=='dense';model.deduplicate_observations=True
    prepare_windows(sessions,train_windows,device);prepare_windows(validation,val_windows,device)
    optimizer=policy_optimizer(model,'muon_sgd');total=math.ceil(args.epochs*len(train_windows)/8192)
    params=[p for p in model.parameters() if p.requires_grad]
    config=dict(run_id=args.run_id,
        observation_chunk_size=args.observation_chunk_size,
        tactile_encoder=args.tactile_encoder,tactile_groups=groups,
        training_mode='single_step_detached_warmup' if args.single_step else 'bptt16',
        policy_horizon=1 if args.single_step else 16,
        warmup_steps=1 if args.single_step else 0,
        policy_network=dict(width=model.width,decoder_width=model.decoder_width,parameters=sum(p.numel() for p in model.parameters())),objective_version=VERSION+('_single_step' if args.single_step else ''),loss={**SPEC,'bptt_length':1 if args.single_step else 16,'single_step_smooth_boundary':'detached preceding clean policy command; one native edge' if args.single_step else None},palm_alignment=alignment,
        train_ids=ref['train_ids'],validation_ids=ref['validation_ids'],tracking=ref['tracking'],seed=ref['seed'],
        world_size=world,micro_batch=args.micro_batch,accumulation_steps=accum,effective_batch=8192,
        batch_unit='one-step transitions after detached warmup' if args.single_step else '16-step sequences',transitions_per_update=8192*(1 if args.single_step else 16),window_stride=1,training_windows=len(train_windows),
        epochs_budget=args.epochs,maximum_updates=total,initialization='fresh_random',geometry_model=None,
        optimizer='Muon+SGD; mean accumulated gradients all-reduced once per update; clip global norm1',
        recurrent_input='previous q20+wrist6, future10 measured EMF/taxels; no future IK',
        wrist_domain=('unbounded wrist pose and increments; affine normalization only; finger mechanical bounds retained' if args.unbounded_wrist else 'existing MCP-aligned ±20mm and ±0.2rad component limits'),
        initial_states='noisy cached IK boundary; detached one-step policy warmup including wrist; current action only receives gradient' if args.single_step else 'noisy cached IK only at window boundary; no resets or detach within16steps',
        source_sha256={str(f.relative_to(Path(__file__).resolve().parents[1])):hashlib.sha256(f.read_bytes()).hexdigest() for d in ['moxian_web','retarget','scripts'] for f in (Path(__file__).resolve().parents[1]/d).rglob('*.py')},
        validation='fixed64 heldout windows noise-free; same objective, contact mean/p95/max; not full-session closed loop',
        loss_normalization='native contact and pose-smooth energy jointly /1e6; contact jitter disabled; no rational or exponential saturation')
    if args.delta_direction_weight:
        config['local_motion_direction']=dict(weight=args.delta_direction_weight,minimum_m=.001,maximum_m=.003,
            samples_per_micro_batch=args.delta_direction_samples,
            effective_samples_per_update=min(args.micro_batch,args.delta_direction_samples)*world*accum,
            input='independent finger translations shared across ten EMF/taxel frames; fixed previous q26',
            output='exact FK fingertip displacement in glove wrist frame',loss='1-cos, eps1e-5; paired FP32 forwards',
            validation='native closed-loop objective unchanged; auxiliary term logged separately in training')
        config['objective_version']+='_local_delta_direction_v1'
    if args.on_policy_state_depth:
        config['training_mode']='single_step_on_policy_detached_depth16'
        config['warmup_steps']='balanced0..15'
        config['initial_states']='IK at boundary only; current-policy rollout depth0..15 with process noise, no gradients through warmup'
        config['loss']['single_step_smooth_boundary']='preceding clean command; depth0 has no artificial edge'
        config['objective_version']+='_on_policy_depth16'
    config['loss']['step_noise_rad']=args.step_noise_rad
    config['state_noise']=dict(initial_joint_rad=args.initial_noise_rad,step_joint_rad=args.step_noise_rad)
    config['teacher_supervision']=False
    config['initial_states']=f'Cached IK at window boundary; initial joint noise ±{args.initial_noise_rad:g} rad; rollout joint noise ±{args.step_noise_rad:g} rad; detached warmup in single-step mode'
    def training_rollout(batch,depth=None):
        extra=dict(warmup_steps=15 if depth is None else depth) if args.on_policy_state_depth else {}
        return train_rollout(model,objective,batch,step_noise_rad=args.step_noise_rad,**extra)
    def training_loss(tr,batch):
        loss,parts=objective.trajectory_loss(tr)
        if args.delta_direction_weight:
            from moxian_web.cart_delta_direction import delta_direction_regularizer
            direction,extra=delta_direction_regularizer(model,hand,batch,tr,max_samples=args.delta_direction_samples)
            native=loss;loss=loss+args.delta_direction_weight*direction
            parts=dict(parts,**extra,native_total=native,delta_direction_weighted=args.delta_direction_weight*direction,total=loss)
        return loss,parts
    last=folder/'resume.pth'
    if args.resume:
        saved=torch.load(last,map_location='cpu',weights_only=False)
        if config!=saved['config']:raise ValueError('Resume config/source mismatch')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);updates=saved['updates'];best=saved['best']
        rng.bit_generator.state=saved['numpy_rng'];torch.set_rng_state(saved['rngs'][rank]['cpu']);torch.cuda.set_rng_state(saved['rngs'][rank]['cuda'],device)
    else:
        if last.exists():raise ValueError('Checkpoint already exists')
        if rank==0:atomic_json(folder/'config.json',config)
    # Independent disturbances per rank, reproducible across compatible resume.
    if not args.resume:torch.manual_seed(ref['seed']+rank+100)
    def batch_for(ids,noise=True):
        batch=[dict(train_windows[int(i)]) for i in ids]
        q=torch.stack([torch.as_tensor(w['initial_q'],device=device,dtype=torch.float32) for w in batch])
        if noise and args.initial_noise_rad:q=(q+torch.empty_like(q).uniform_(-args.initial_noise_rad,args.initial_noise_rad)).clamp(hand.low,hand.high)
        for w,value in zip(batch,q.unbind(0)):w['initial_q']=value
        return PreparedBatch(batch,device)
    def save():
        state=dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(device).cpu())
        gathered=[None]*world if rank==0 else None
        if world>1:dist.gather_object(state,gathered,dst=0)
        else:gathered=[state]
        if rank==0:save_checkpoint(last,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),updates=updates,best=best,config=config,numpy_rng=rng.bit_generator.state,rngs=gathered))
    def evaluate():
        model.eval();chosen=np.linspace(0,len(val_windows)-1,min(64,len(val_windows))).astype(int)
        sums={};gaps=[]
        with torch.no_grad():
            for start in range(0,len(chosen),16):
                batch=PreparedBatch([val_windows[i] for i in chosen[start:start+16]],device)
                tr=predict_trajectory(model,objective,batch,step_noise_rad=0.)
                _,parts=objective.trajectory_loss(tr)
                for k,v in parts.items():sums[k]=sums.get(k,0)+float(v)*len(batch)
                sq,mask,_=objective.distances(torch.cat(tr.predictions),[r for x in tr.pending for r in x[1]])
                gaps.append(sq[mask].sqrt()*1000)
        result={k:v/len(chosen) for k,v in sums.items()};g=torch.cat(gaps)
        result.update(contact_p95_mm=float(torch.quantile(g,.95)),contact_max_mm=float(g.max()),windows=len(chosen))
        model.train();return result
    try:
        if not args.resume:
            progress('preflight')
            cpu_rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state(device);before={k:v.detach().clone() for k,v in model.state_dict().items()}
            b=batch_for(np.random.default_rng(ref['seed']).choice(len(train_windows),8192,replace=False).reshape(world,-1)[rank,:args.micro_batch])
            tr=training_rollout(b)
            recurrence=torch.autograd.grad(tr.predictions[-1].square().sum(),tr.input_state if args.single_step else tr.predictions[0],retain_graph=True)[0]
            if args.single_step:assert not tr.previous_command.requires_grad and tr.input_state.is_leaf
            loss,parts=training_loss(tr,b);loss.backward();torch.cuda.synchronize()
            norms=[sum(float(p.grad.norm()) for p in head.parameters() if p.grad is not None) for head in model.decoders]
            wrist=sum(float(p.grad.norm()) for p in model.wrist_decoder.parameters() if p.grad is not None)
            assert torch.isfinite(loss) and torch.isfinite(recurrence).all() and recurrence.norm()>0
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in params) and min(norms)>0 and wrist>0
            for k,v in model.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
            # Distance derivative vs central difference through exact FK.
            q=tr.predictions[0][:4].detach().requires_grad_();rows=tr.pending[0][1][:4]
            from moxian_web.cart_gradient_check import directional_gradient_check
            def energy_fn(state):
                sq,mask,rho=objective.distances(state,rows)
                return (1600*sq.double()*mask*rho.double().square()).sum(-1)
            direction=torch.randn_like(q);direction[:,20:23]*=.01;direction/=direction.norm()
            derivative_check=directional_gradient_check(energy_fn,q,direction)
            atomic_json(folder/f'directional_gradient_rank{rank}.json',derivative_check)
            error=derivative_check['relative_error']
            if not derivative_check['passed']:raise ValueError(f'Contact derivative audit failed: {derivative_check}')
            atomic_json(folder/f'preflight_rank{rank}.json',dict(passed=True,loss=float(loss),finger_gradients=norms,wrist_gradient=wrist,state_gradient=float(recurrence.norm()),bptt_enabled=not args.single_step,directional_gradient_relative_error=error,peak_memory=torch.cuda.max_memory_allocated(),micro_batch=args.micro_batch,updates=0))
            optimizer.zero_grad();del tr,loss,b,before
            torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(gpu_rng,device)
            save();barrier();progress('preflight_complete')
            if args.preflight_only:return
        if rank==0:
            t=config['tracking'];tracking=CartTracking(folder,config,**t,resume=args.resume and (folder/'wandb_run.json').exists())
        barrier()
        if updates==0:
            if rank==0:
                baseline=evaluate();atomic_json(folder/'validation_initial.json',baseline)
                tracking.run.log({f'validation/{k}':v for k,v in baseline.items()}|{'validation/epoch':0})
            barrier()
        while updates<total:
            check();tick=time.perf_counter();model.train();optimizer.zero_grad()
            order=rng.choice(len(train_windows),8192,replace=False);local_ids=order.reshape(world,-1)[rank];parts_sum={}
            scale_learning_rates(optimizer,cosine_multiplier(updates,total,5,.1))
            for k in range(accum):
                check();b=batch_for(local_ids[k*args.micro_batch:(k+1)*args.micro_batch])
                depth=(updates*accum*world+k*world+rank)%16
                tr=training_rollout(b,depth);loss,parts=training_loss(tr,b)
                if args.on_policy_state_depth:parts['state_rollout_step']=loss.detach().new_tensor(depth+1)
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                (loss/accum).backward()
                for key,value in parts.items():parts_sum[key]=parts_sum.get(key,0.)+value.detach()/accum
                del tr,loss,b
                progress('accumulating',micro_step=k+1,micro_steps=accum,sequences_per_update=8192)
            # One weighted global mean. No optimizer step or clipping between microbatches.
            flat=torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in params])
            if world>1:dist.all_reduce(flat);flat/=world
            if not torch.isfinite(flat).all():raise ValueError('Nonfinite reduced gradients')
            offset=0
            for p in params:
                p.grad=flat[offset:offset+p.numel()].view_as(p);offset+=p.numel()
            grad_norm=float(torch.nn.utils.clip_grad_norm_(params,1.));optimizer.step();updates+=1
            vec=torch.stack(list(parts_sum.values()))
            if world>1:dist.all_reduce(vec);vec/=world
            values={k:float(v) for k,v in zip(parts_sum,vec)};elapsed=time.perf_counter()-tick
            save();progress('training',losses=values,gradient_norm=grad_norm,effective_batch=8192,world_size=world,seconds_per_update=elapsed,equivalent_epochs=updates*8192/len(train_windows))
            if rank==0:
                tracking.run.log({f'train/{k}':v for k,v in values.items()}|{'train/optimizer_step':updates,'train/effective_batch':8192,'train/micro_batch':args.micro_batch,'train/accumulation_steps':accum,'train/world_size':world,'train/gradient_norm':grad_norm,'train/seconds_per_update':elapsed})
            if updates==1 or updates%10==0 or updates==total:
                if rank==0:
                    progress('validation');metrics=evaluate();score=metrics['total'];improved=best is None or score<best['loss']
                    if improved:
                        best=dict(update=updates,loss=score,metrics=metrics)
                        save_checkpoint(folder/'best.pth',dict(model=model.state_dict(),config=config,selection=best));atomic_json(folder/'best.json',best)
                    atomic_json(folder/'validation_latest.json',dict(update=updates,metrics=metrics))
                    tracking.run.log({f'validation/{k}':v for k,v in metrics.items()}|{'validation/epoch':updates*8192/len(train_windows)})
                    progress('validated',metrics=metrics)
                barrier()
        save();progress('finished',status='finished')
        if tracking:tracking.finish(report=dict(updates=updates))
    except InterruptedError:
        # Last checkpoint is the last fully committed global update; partial gradients discarded.
        progress('stopped',status='stopped_by_user')
        if tracking:tracking.finish(report=dict(stopped=True,updates=updates))
    except BaseException as e:
        atomic_json(folder/f'failure_rank{rank}.json',dict(error=repr(e),updates=updates,pid=os.getpid()))
        if tracking:tracking.finish(error=str(e))
        raise
    finally:
        if world>1:dist.destroy_process_group()


if __name__=='__main__':main()
