#!/usr/bin/env python3
"""Stage timings and launch-layout probes; production stays stopped.

Scoring probes perform zero pose updates. The optional CEM diagnostic executes
at most two 24-update attempts on ONE frozen input; it is not production.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,gc,json,sys,time
from pathlib import Path
from dataclasses import replace
import numpy as np
import torch
import warp as wp
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.mano_patch_tools import (ManoDecoder,ShadowSkeleton,FixedContacts,SurfaceBatch,
    initialize,lm_ik,Settings,pose_problem,CemSettings,write,sha,competition)
from retarget.solver.cem import _solve_patch_cem as solve_patch_cem
from retarget.env.grasp.mano_patch_consistent import PatchConsistentAdapter
from retarget.env.grasp import spring_history_warp


def sync():
    torch.cuda.synchronize()
    return time.perf_counter()


class StageEvents:
    def __init__(self):
        self.events={}
        self.restore=[]

    def patch(self,obj,name,label):
        old=getattr(obj,name)
        def wrapped(*args,**kwargs):
            key=label(*args,**kwargs) if callable(label) else label
            if key is None:return old(*args,**kwargs)
            if key not in self.events:
                self.events[key]=(torch.cuda.Event(enable_timing=True,external=True),torch.cuda.Event(enable_timing=True,external=True))
            start,end=self.events[key]
            start.record()
            value=old(*args,**kwargs)
            end.record()
            return value
        setattr(obj,name,wrapped)
        self.restore.append((obj,name,old))

    def close(self):
        for obj,name,old in reversed(self.restore):setattr(obj,name,old)
        self.restore.clear()


def graph_timings(adapter,x,rows,repeats):
    events=StageEvents()
    events.patch(adapter,'_compute','full')
    events.patch(adapter.decoder,'forward','mano_decode')
    events.patch(adapter.fixed,'evaluate','fixed_hand_slots')
    events.patch(adapter.surfaces,'query',lambda points,*a,**kw:'hand_vertex_penetration_query' if points.shape[1]==778 else 'fixed_slot_surface_query')
    if hasattr(adapter,'vertex_penetration'):
        events.patch(adapter.vertex_penetration,'query','hand_vertex_penetration_query')
    events.patch(adapter.collision,'query','mjwarp_collision')
    events.patch(adapter.rollout,'solve','rollout_total')
    events.patch(adapter.rollout,'integrate_fused','rollout_fused_kernel_and_scratch')
    events.patch(adapter.patches,'query','anatomical_patch_query')
    events.patch(wp,'launch',lambda kernel,*a,**kw:'geometry_'+kernel.key if kernel.key in
                 ('prepare_pairs','evaluate_pairs','finish_pairs','fallback_pairs','depth_kernel') else None)
    adapter.graph=None
    try:
        t=sync();adapter.evaluate(x,rows);preparation=sync()-t
        samples=[]
        for _ in range(repeats):
            t=sync();adapter.evaluate(x,rows);wall=(sync()-t)*1000
            samples.append(dict(wall_ms=wall,**{name:start.elapsed_time(end) for name,(start,end) in events.events.items()}))
        result={name:float(np.median([v[name] for v in samples])) for name in samples[0]}
        result['rollout_preparation_and_metrics']=result['rollout_total']-result['rollout_fused_kernel_and_scratch']
        components=['mano_decode','fixed_hand_slots','hand_vertex_penetration_query','fixed_slot_surface_query',
            'mjwarp_collision','rollout_total','anatomical_patch_query']
        result['other_scoring']=result['full']-sum(result[k] for k in components)
        result['percent_of_full']={key:100*result[key]/result['full'] for key in [*components,'other_scoring']}
        return dict(preparation_seconds=preparation,median_ms=result,samples=samples)
    finally:
        events.close();adapter.graph=None


def cem_diagnostic(adapter,q,mean,iterations,out):
    """Uninstrumented wall time first; CPU profiler separately, same start/seed."""
    problem=replace(pose_problem(mean,adapter),mean=q.clone())
    settings=CemSettings(population=128,iterations=iterations)
    adapter.evaluate(q,torch.zeros(len(q),device=q.device,dtype=torch.long));sync()
    intervals=[]
    originals={key:getattr(adapter,key) for key in ['evaluate_candidates','validate']}
    def wrap(name):
        def call(*args,**kwargs):
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
            a.record();value=originals[name](*args,**kwargs);b.record()
            intervals.append((name,a,b,len(args[0])))
            return value
        return call
    for key in originals:setattr(adapter,key,wrap(key))
    try:
        start=sync();state=solve_patch_cem(problem,adapter,settings,20261105,early_stop=True);wall=sync()-start
    finally:
        for key,fn in originals.items():setattr(adapter,key,fn)
    spans={key:sum(a.elapsed_time(b) for name,a,b,n in intervals if name==key) for key in originals}
    report=dict(wall_seconds=wall,actual_pose_updates=int(state['iterations'].sum()),evaluations=int(state['evaluations'].sum()),
        passed=state['passed'].cpu().tolist(),adapter_gpu_stream_spans_ms=spans,
        remaining_wall_ms=wall*1000-sum(spans.values()),
        calls=[dict(kind=name,samples=n,gpu_stream_span_ms=a.elapsed_time(b)) for name,a,b,n in intervals],
        attribution='Adapter spans include host submission gaps between chunks. Remaining wall includes sampling, elite updates, host control and synchronization, not CPU arithmetic alone.')
    write(out/'CEM_WALL.json',report)
    np.savez(out/'diagnostic_final.npz',variables=state['parameters'].cpu().numpy(),passed=state['passed'].cpu().numpy())
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],record_shapes=False) as prof:
            other=solve_patch_cem(problem,adapter,settings,20261105,early_stop=True)
            sync()
        prof.export_chrome_trace(str(out/'cem_cpu_trace.json'))
        ops=sorted([dict(name=v.key,calls=v.count,self_cpu_ms=v.self_cpu_time_total/1000,total_cpu_ms=v.cpu_time_total/1000)
            for v in prof.key_averages()],key=lambda v:v['self_cpu_ms'],reverse=True)
        write(out/'CEM_CPU_PROFILE.json',dict(ops=ops,actual_pose_updates=int(other['iterations'].sum()),
            note='Instrumented run; synchronization waits overlap GPU work and must not be added to the wall breakdown.'))
        report['profile_pose_updates']=int(other['iterations'].sum())
    except Exception as e:
        write(out/'CEM_CPU_PROFILE.json',dict(complete=False,error=str(e)))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,default=Path('output/mano_patch_velocity_v4_20261105'))
    p.add_argument('--results',type=Path,default=Path('output/mano_patch_history_v5_gpu_fast_20261105'))
    p.add_argument('--packet',default='01_1_Large_Diameter_random')
    p.add_argument('--model',type=Path,default=Path('/home/zz/GraspSONIC/thirdparty/DexForge/assets/mano_v1_2/models/MANO_RIGHT.pkl'))
    p.add_argument('--device',default='cuda:0');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--chunks',default='128,2048');p.add_argument('--blocks',default='256,64,32')
    p.add_argument('--repeats',type=int,default=5);p.add_argument('--cem-iterations',type=int,default=24)
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'MANIFEST.json').exists():raise FileExistsError('Use a fresh profile directory')
    assert 0<=args.cem_iterations<=24
    torch.set_num_threads(1);torch.cuda.set_device(args.device);torch.manual_seed(20261105)
    mapping=json.loads((args.baseline/'assets/fixed_contact_mapping.json').read_text());assert sha(args.model)==mapping['model_sha256']
    manifest=json.loads((args.results/'MANIFEST.json').read_text());packet=next(v for v in manifest['packets'] if v['name']==args.packet)
    frozen=args.baseline/'inputs'/(packet['name']+'.npz');a=dict(np.load(frozen));assert sha(frozen)==packet['sha256']
    old=[dict(np.load(args.results/'quality'/args.packet/stage/'final.npz')) for stage in ['input','velocity_input','cem']]
    write(out/'MANIFEST.json',dict(authorization='User requests runtime breakdown and optimization opportunities',production_started=False,
        diagnostic_cem_max_pose_updates=2*args.cem_iterations,score_probe_pose_updates=0,packet=args.packet,
        input_sha256=sha(frozen),model_sha256=sha(args.model),source_sha256={f:sha(ROOT/f) for f in
            ['scripts/profile_mano_patch_pipeline.py','retarget/env/grasp/spring_history.py','retarget/env/grasp/spring_history_warp.py',
             'retarget/solver/cem.py','retarget/env/grasp/mano_comfree.py','retarget/env/grasp/mano_patch_consistent.py',
             'retarget/env/grasp/patch_query.py','retarget/env/grasp/triangle_penetration.py']},
        device=torch.cuda.get_device_name(),chunks=args.chunks,block_dimensions=args.blocks,repeats=args.repeats,
        external_competition=competition(),timing='External CUDA events inside captured full scoring graph; nested rollout substage is not added twice; diagnostic CEM separate.'))
    start=sync();decoder=ManoDecoder(args.model,args.device);decoder.configure_contact_vertices(mapping['mappings'])
    ids=list(map(int,a['object_geometry_ids']));geo={i:dict(vertices=a[f'object_{i}_vertices'],faces=a[f'object_{i}_faces'],watertight=True) for i in set(ids)}
    surface=SurfaceBatch(geo,ids,a['object_pose_world'],a['scene_scale'],args.device)
    fixed=FixedContacts(decoder,mapping['mappings'][packet['type']]);anchors=torch.tensor(old[0]['object_template_points'],device=args.device)
    q=torch.cat([torch.tensor(v['variables'],device=args.device) for v in old]);n=len(old[0]['variables'])
    probe=q[:n][torch.arange(64,device=q.device)%n].clone()
    probe+=torch.randn_like(probe)*q.new_tensor([.002]*3+[.01]*3+[.03]*(q.shape[1]-6))
    q=torch.cat((q,probe));rows=torch.cat((torch.arange(n,device=q.device).repeat(3),torch.arange(64,device=q.device)%n))
    write(out/'preparation.json',dict(model_and_geometry_seconds=sync()-start))
    original_launch=wp.launch;block=256
    def launch(kernel,*pos,**kwargs):
        if kernel is spring_history_warp.integrate:kwargs['block_dim']=block
        return original_launch(kernel,*pos,**kwargs)
    wp.launch=launch
    reports={}
    try:
        for chunk in map(int,args.chunks.split(',')):
            adapter=PatchConsistentAdapter(decoder,fixed,surface,chunk,anchors=anchors)
            x=q[torch.arange(chunk,device=q.device)%len(q)].clone();rr=rows[torch.arange(chunk,device=q.device)%len(q)].clone()
            original_cost=None
            for block in map(int,args.blocks.split(',')):
                t=sync();scores=adapter.evaluate(x,rr).clone();warmup=sync()-t
                if original_cost is None:original_cost=scores.clone()
                else:torch.testing.assert_close(scores,original_cost,atol=1e-3,rtol=2e-4)
                # _score graph contains the selected kernel variant only.
                stats=graph_timings(adapter,x,rr,args.repeats)
                stats['initial_warmup_seconds']=warmup
                stats['rollout_grid_blocks']=(chunk+block-1)//block
                reports[f'chunk{chunk}_block{block}']=stats
                write(out/'PHASES.json',reports)
            if chunk==128 and args.cem_iterations:
                block=256;adapter.graph=None
                skeleton=ShadowSkeleton(args.baseline/'assets/shadow_assets/right.xml',args.device)
                target=skeleton(a['shadow_qpos']);mean,_,_=lm_ik(decoder,initialize(decoder,target),target,Settings())
                cem=cem_diagnostic(adapter,torch.tensor(old[0]['variables'],device=q.device),mean,args.cem_iterations,out)
            del adapter;gc.collect();torch.cuda.empty_cache()
    finally:wp.launch=original_launch
    write(out/'SUMMARY.json',dict(complete=True,production_started=False,stages=reports,
        actual_diagnostic_pose_updates=cem['actual_pose_updates']+cem.get('profile_pose_updates',0) if args.cem_iterations else 0,
        external_competition_after=competition()))


if __name__=='__main__':main()
