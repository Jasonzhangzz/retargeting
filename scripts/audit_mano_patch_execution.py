#!/usr/bin/env python3
"""Bounded, zero-update audit of fused v5 scoring against the Torch integrator."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,gc,json,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.mano_patch_tools import ManoDecoder,SurfaceBatch,FixedContacts,MeshCache,write,sha,competition
from retarget.env.grasp.mano_patch_consistent import PatchConsistentAdapter
from retarget.env.grasp.spring_history import SpringHistoryRollout


def sync():
    torch.cuda.synchronize()
    return time.perf_counter()


def sample_time(fn,repeats=3):
    result=[]
    for _ in range(repeats):
        t=sync()
        fn()
        result.append(sync()-t)
    return dict(seconds=result,median_seconds=float(np.median(result)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,default=Path('output/mano_patch_velocity_v4_20261105'))
    p.add_argument('--reference',type=Path,default=Path('output/mano_patch_history_v5_20261105'))
    p.add_argument('--candidate',type=Path,default=Path('output/mano_patch_history_v5_gpu_fast_20261105'))
    p.add_argument('--model',type=Path,default=Path('/home/zz/GraspSONIC/thirdparty/DexForge/assets/mano_v1_2/models/MANO_RIGHT.pkl'))
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--chunks',default='128,2048')
    p.add_argument('--first-packet-only',action='store_true')
    args=p.parse_args()
    out=args.output
    out.mkdir(parents=True,exist_ok=True)
    if (out/'MANIFEST.json').exists():raise FileExistsError('Audit directory must be new')
    torch.set_num_threads(1)
    torch.cuda.set_device(args.device)
    torch.manual_seed(20261105)
    manifest=json.loads((args.reference/'MANIFEST.json').read_text())
    mapping=json.loads((args.baseline/'assets/fixed_contact_mapping.json').read_text())
    assert sha(args.model)==mapping['model_sha256']
    packets=manifest['packets'][:1] if args.first_packet_only else manifest['packets']
    write(out/'MANIFEST.json',dict(optimizer_updates=0,production_started=False,packets=[v['name'] for v in packets],
        reference=str(args.reference),candidate=str(args.candidate),device=torch.cuda.get_device_name(),
        source_sha256={name:sha(ROOT/name) for name in ['retarget/env/grasp/spring_history.py','retarget/env/grasp/spring_history_warp.py',
            'retarget/env/grasp/mano_comfree.py','retarget/env/grasp/mano_patch_anchor.py','retarget/env/grasp/mano_patch_consistent.py',
            'scripts/audit_mano_patch_execution.py']},external_competition=competition(),
        tolerance=dict(cost_atol=1e-3,cost_rtol=2e-4,linear_velocity_atol=2e-5,angular_velocity_atol=2e-4),
        probes='all three frozen stages plus 64 local random pose perturbations per packet; fixed seed; no optimization',
        timing='same full candidate scoring, warmed CUDA graph, three synchronized repeats, separate graph preparation'))
    decoder=ManoDecoder(args.model,args.device)
    decoder.configure_contact_vertices(mapping['mappings'])
    cache=MeshCache(args.device)
    reports=[]
    for packet in packets:
        name=packet['name']
        frozen=args.baseline/'inputs'/(name+'.npz')
        a=dict(np.load(frozen))
        assert sha(frozen)==packet['sha256']
        ids=list(map(int,a['object_geometry_ids']))
        geo={i:dict(vertices=a[f'object_{i}_vertices'],faces=a[f'object_{i}_faces'],watertight=True) for i in set(ids)}
        surface=SurfaceBatch(geo,ids,a['object_pose_world'],a['scene_scale'],args.device,mesh_cache=cache)
        fixed=FixedContacts(decoder,mapping['mappings'][packet['type']])
        stages=[dict(np.load(args.reference/'quality'/name/stage/'final.npz')) for stage in ['input','velocity_input','cem']]
        anchors=torch.tensor(stages[0]['object_template_points'],device=args.device)
        n=len(stages[0]['variables'])
        q=torch.cat([torch.tensor(stage['variables'],device=args.device) for stage in stages])
        rows=torch.arange(n,device=args.device).repeat(3)
        probes=q[:n][torch.arange(64,device=args.device)%n].clone()
        scale=probes.new_tensor([.002]*3+[.01]*3+[.03]*(probes.shape[1]-6))
        probes+=torch.randn_like(probes)*scale
        probe_rows=torch.arange(64,device=args.device)%n
        q=torch.cat((q,probes));rows=torch.cat((rows,probe_rows))
        adapter=PatchConsistentAdapter(decoder,fixed,surface,128,anchors=anchors)
        t=sync();fused_cost,fused=adapter.validate(q,rows);fused_warmup=sync()-t
        adapter.rollout.backend='torch';adapter.graph=None
        t=sync();reference_cost,reference=adapter.validate(q,rows);reference_warmup=sync()-t
        np.testing.assert_allclose(fused_cost.cpu(),reference_cost.cpu(),atol=1e-3,rtol=2e-4)
        for key,atol in [('rollout_linear_velocity',2e-5),('rollout_angular_velocity',2e-4)]:
            try:
                torch.testing.assert_close(fused[key],reference[key],atol=atol,rtol=2e-4,msg=lambda m:key+': '+m)
            except AssertionError:
                np.savez(out/'mismatch.npz',variables=q.cpu(),rows=rows.cpu(),
                    **{'fused_'+k:v.cpu() for k,v in fused.items() if k.startswith('rollout_')},
                    **{'reference_'+k:v.cpu() for k,v in reference.items() if k.startswith('rollout_')})
                raise
        # MJWarp contact output order varies between collision invocations.
        # Integrator history/forces must be compared on IDENTICAL packets;
        # comparing slot k across two independently generated packets is invalid.
        rotation=surface.rotation[rows]
        center=(rotation@adapter.centers[rows,:,None]).squeeze(-1)+surface.origin[rows]
        direct=SpringHistoryRollout(len(q),fused['contact_gap'].shape[1],device=args.device,
            settings=adapter.rollout_settings,spring=adapter.rollout.spring,backend='torch')
        same_packet=direct.solve(fused['contact_hand_points'],fused['contact_object_points'],fused['contact_normals'],
            fused['contact_gap'],center,rotation,adapter.inertias[rows],eligible=fused['rollout_eligible'])
        for key,atol in [('linear_velocity',2e-5),('angular_velocity',2e-4),('tangential_displacement_m',2e-6),
                         ('normal_force_n',2e-4),('tangent_force_n',2e-4)]:
            torch.testing.assert_close(fused['rollout_'+key],same_packet[key],atol=atol,rtol=2e-4,msg=lambda m:key+': '+m)
        gates=['valid','rollout_stable','patch_geometry_accepted','patch_joint_accepted','patch_fitted']
        for key in gates:assert torch.equal(fused[key],reference[key]),(name,key)
        # Compare the exact existing final outputs, not just aggregate rates.
        for j,stage in enumerate(stages):
            for key in gates:np.testing.assert_array_equal(fused[key][j*n:(j+1)*n].cpu(),stage[key])
        report=dict(packet=name,scores_checked=len(q),fused_warmup_seconds=fused_warmup,reference_warmup_seconds=reference_warmup,
            max_absolute_linear_velocity_delta=float((fused['rollout_linear_velocity']-reference['rollout_linear_velocity']).abs().max()),
            max_absolute_angular_velocity_delta=float((fused['rollout_angular_velocity']-reference['rollout_angular_velocity']).abs().max()),
            newton_solves_median=float(fused['rollout_newton_linear_solves'].float().median()),
            newton_solves_max=int(fused['rollout_newton_linear_solves'].max()),same_gates=True)
        if args.candidate.exists():
            current=np.load(args.candidate/'quality'/name/'cem/final.npz')
            report['same_cem_parameters']=bool(np.array_equal(current['variables'],stages[2]['variables']))
            report['same_cem_joint_decisions']=bool(np.array_equal(current['patch_joint_accepted'],stages[2]['patch_joint_accepted']))
        reports.append(report)
        write(out/'progress.json',dict(complete=False,optimizer_updates=0,packets=reports))
        # Representative complete-scoring throughput; no repeated or independent-pose claim.
        if name==packets[0]['name']:
            timings={}
            for chunk in map(int,args.chunks.split(',')):
                del adapter
                gc.collect();torch.cuda.empty_cache()
                x=q[torch.arange(chunk,device=q.device)%len(q)].clone()
                rr=rows[torch.arange(chunk,device=q.device)%len(q)].clone()
                adapter=PatchConsistentAdapter(decoder,fixed,surface,chunk,anchors=anchors)
                for backend in ['warp','torch']:
                    adapter.rollout.backend=backend;adapter.graph=None
                    t=sync();adapter.evaluate(x,rr);warmup=sync()-t
                    timings[f'{backend}_{chunk}']=dict(warmup_seconds=warmup,**sample_time(lambda owner=adapter:owner.evaluate(x,rr)))
                write(out/'TIMINGS.json',timings)
        del adapter,surface
        gc.collect();torch.cuda.empty_cache()
    write(out/'SUMMARY.json',dict(complete=True,passed=True,optimizer_updates=0,production_started=False,
        packets=reports,scores_checked=sum(v['scores_checked'] for v in reports),external_competition_after=competition()))
    print(json.dumps(reports,indent=2))


if __name__=='__main__':main()
