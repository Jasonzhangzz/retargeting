"""Measure real CART optimizer throughput without saving or enabling a model."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from moxian_web.cart_data import load_sessions,manifest
from moxian_web.cart_training import windows,position_statistics,policy_update
from moxian_web.cart_fast import prepare_windows,choose_parallel_batch
from moxian_web.cart_protocol import POLICY_OUTPUT
from moxian_web.cart_objective import CartHand,CartObjective,policy_settings
from moxian_web.cart_collision import CollisionModel,SDFLearner
from moxian_web.cart_network import CartNetwork
from moxian_web.cart_initialization import prepare_ik,attach_ik
from retarget.baseline.geort.adapter import ROOT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT/'output/moxian/cart')
    parser.add_argument('--geometry-only',action='store_true',help='Transfer geometry/replay only, leaving the delta policy freshly initialized')
    parser.add_argument('--initialize-sdf-from',type=Path,help='Benchmark the pretrained articulated SDF distance backend')
    parser.add_argument('--dataset',required=True);parser.add_argument('--model')
    parser.add_argument('--architecture',choices=('attention','fsq'),default='attention')
    parser.add_argument('--batch-size',type=int,default=16);parser.add_argument('--micro-batch-size',type=int,default=0)
    parser.add_argument('--window-start',type=int,default=0,help='First eligible window index, useful for single-window timing')
    parser.add_argument('--window-stride',type=int,default=0,help='Window start spacing; 0 keeps length//2')
    parser.add_argument('--initial-noise-rad',type=float,default=.1)
    parser.add_argument('--length',type=int,default=16);parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--device',default='cuda:0' if torch.cuda.is_available() else 'cpu');parser.add_argument('--output',type=Path)
    parser.add_argument('--serial-losses',action='store_true',help='Reference implementation: evaluate FK and loss separately at each recurrent step')
    parser.add_argument('--attention-precision',choices=('float32','bfloat16'),default='float32')
    parser.add_argument('--checkpoint-observations',action='store_true',help='Recompute the sensor encoder to fit more genuinely parallel windows')
    parser.add_argument('--profile',type=Path,help='Optional extra profiler step; excluded from throughput measurements')
    args=parser.parse_args()
    if args.batch_size<0 or args.length<2 or args.repeats<2:parser.error('batch>=0 (0=auto), length>=2 and repeats>=2 are required')
    torch.set_num_threads(1);torch.manual_seed(17)
    _,meta=manifest(args.root,args.dataset)
    sessions,_=load_sessions(args.root,[args.dataset],meta['split'])
    prepare_ik(args.root,args.dataset);attach_ik(args.root,sessions)
    if args.window_start<0:parser.error('window-start must be nonnegative')
    selected=windows(sessions,args.length,args.window_stride)[args.window_start:]
    if not selected:raise ValueError('No continuous rollout windows')
    prepare_windows(sessions,selected,args.device)
    parallel=choose_parallel_batch(args.architecture,args.device,args.batch_size,args.micro_batch_size,args.length,
        args.attention_precision,args.checkpoint_observations)
    batch=[selected[i] for i in np.linspace(0,len(selected)-1,min(parallel,len(selected))).astype(int)]
    hand=CartHand(args.device);collision=CollisionModel(hand)
    if args.initialize_sdf_from:
        from moxian_web.cart_pretrained_sdf import load_pretrained
        collision.model,_=load_pretrained(hand,args.initialize_sdf_from)
    objective=CartObjective(hand,policy_settings(),collision);objective.cache_rows=True
    mean,scale=position_statistics(sessions)
    model=CartNetwork(hand.low,hand.high,args.architecture,position_mean=mean,position_scale=scale).to(args.device)
    learner=SDFLearner(collision);optimizer=torch.optim.AdamW(model.parameters(),lr=1.e-4)
    if args.model:
        folder=args.root/'models'/args.model;config=json.loads((folder/'config.json').read_text())
        if config['architecture_name']!=args.architecture or config['source']!=meta['source']:raise ValueError('Benchmark checkpoint does not match architecture/source')
        if not args.initialize_sdf_from and config.get('sdf_backend',{}).get('kind','link_sdf_v1')!='link_sdf_v1':
            from moxian_web.cart_pretrained_sdf import restore_geometry
            collision.model=restore_geometry(hand,config['sdf_backend']);learner=SDFLearner(collision)
        model=CartNetwork(hand.low,hand.high,args.architecture,position_mean=config['position_mean'],position_scale=config['position_scale']).to(args.device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1.e-4)
        path=folder/('resume.pth' if (folder/'resume.pth').exists() else 'warmup.pth')
        state=torch.load(path,map_location='cpu',weights_only=True)
        if not args.geometry_only:
            if config.get('policy_output')!=POLICY_OUTPUT:raise ValueError('Absolute-q checkpoint cannot initialize a delta policy; use --geometry-only')
            model.load_state_dict(state['model'])
        if config.get('contact_field'):
            if args.initialize_sdf_from:
                collision.model.load_contact_state(state['collision']);learner.load_state_dict(state['sdf_learner'],load_optimizer=False)
            else:
                collision.model.load_state_dict(state['collision']);learner.load_state_dict(state['sdf_learner'])
        else:
            missing=collision.model.load_state_dict(state['collision'],strict=False)
            expected={k for k in collision.model.state_dict() if k.startswith('contact_field.')}
            if set(missing.missing_keys)!=expected or missing.unexpected_keys:raise ValueError('Unsupported old geometry checkpoint')
            print(json.dumps(dict(contact_field_initialized=True,quality_benchmark=False)),flush=True)
        if not args.geometry_only:optimizer.load_state_dict(state['policy_optimizer'])
    model.set_attention_precision(args.attention_precision)
    model.checkpoint_observations=args.checkpoint_observations
    micro=len(batch)
    def sync():
        if args.device.startswith('cuda'):torch.cuda.synchronize(args.device)
    results=[]
    for repeat in range(args.repeats):
        sync()
        if args.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats(args.device)
        started=time.perf_counter()
        loss,_,geometry_losses,timings=policy_update(model,objective,learner,optimizer,batch,micro,initial_noise_rad=args.initial_noise_rad,batch_losses=not args.serial_losses)
        sync();elapsed=time.perf_counter()-started
        result=dict(repeat=repeat,warmup=repeat==0,elapsed_s=elapsed,windows_per_s=len(batch)/elapsed,
            transitions_per_s=len(batch)*args.length/elapsed,loss=float(loss.detach()),phase_wall_s=timings,geometry_losses=geometry_losses)
        if args.device.startswith('cuda'):result['peak_allocated_mb']=torch.cuda.max_memory_allocated(args.device)/2**20
        results.append(result);print(json.dumps(result),flush=True)
    report=dict(device=torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else 'CPU',
        architecture=args.architecture,requested_parallel_batch=args.batch_size,parallel_batch=len(batch),gradient_accumulation=False,length=args.length,window_start=args.window_start,
        window_stride=args.window_stride or max(1,args.length//2),available_windows=len(selected),
        loss_execution='serial_reference' if args.serial_losses else 'time_batched',
        policy_attention_precision=args.attention_precision,geometry_precision='float32',
        observation_checkpointing=args.checkpoint_observations,
        policy_output=POLICY_OUTPUT,geometry_only_initialization=args.geometry_only,
        sdf_backend='configuration_sdf_v1' if hasattr(collision.model,'at_frames') else 'link_sdf_v1',
        geometry_activation_recomputation=args.device.startswith('cuda'),
        initialization='offline_wuji_retargeting_ik',initial_noise_rad=args.initial_noise_rad,
        contact_loss_frames=sum(bool(row['contacts']['self_pairs'] or row['contacts']['object_contacts'])
            for w in batch for row in w['rows'][-w['length']-9:-9]),
        optimizer_steps_measured=args.repeats,steady_windows_per_s=float(np.mean([r['windows_per_s'] for r in results[1:]])),
        measurements=results,checkpoint_written=False,model_published=False)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,allow_nan=False))
    if args.profile:
        activities=[torch.profiler.ProfilerActivity.CPU]
        if args.device.startswith('cuda'):activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities) as profiler:
            policy_update(model,objective,learner,optimizer,batch,micro,initial_noise_rad=args.initial_noise_rad,batch_losses=not args.serial_losses);sync()
        args.profile.parent.mkdir(parents=True,exist_ok=True)
        events=profiler.key_averages()
        args.profile.write_text(events.table(sort_by='self_cpu_time_total',row_limit=30)+'\n'+
            events.table(sort_by='self_device_time_total',row_limit=30))
        print(json.dumps(dict(profile_device_times_available=any(e.device_time_total>0 for e in events))),flush=True)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
