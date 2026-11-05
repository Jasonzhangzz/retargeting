#!/usr/bin/env python3
"""Zero-update equivalence and steady-state timings for patch/depth execution."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,gc,json,sys,time
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.mano_patch_tools import ManoDecoder,FixedContacts,SurfaceBatch,MeshCache,write,sha,competition
from retarget.env.grasp.mano_patch_consistent import PatchConsistentAdapter
from scripts.profile_mano_patch_pipeline import graph_timings,sync


def backend(adapter,name):
    adapter.geometry_backend='reference' if name=='reference' else 'optimized'
    adapter.patches.backend=name
    adapter.graph=None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,default=Path('output/mano_patch_velocity_v4_20261105'))
    p.add_argument('--results',type=Path,default=Path('output/mano_patch_history_v5_gpu_fast_20261105'))
    p.add_argument('--model',type=Path,default=Path('/home/zz/GraspSONIC/thirdparty/DexForge/assets/mano_v1_2/models/MANO_RIGHT.pkl'))
    p.add_argument('--device',default='cuda:0');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--first-packet-only',action='store_true');p.add_argument('--chunks',default='128,2048')
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'MANIFEST.json').exists():raise FileExistsError('Use a fresh audit directory')
    torch.set_num_threads(1);torch.cuda.set_device(args.device);torch.manual_seed(20261105)
    manifest=json.loads((args.results/'MANIFEST.json').read_text());packets=manifest['packets'][:1] if args.first_packet_only else manifest['packets']
    mapping=json.loads((args.baseline/'assets/fixed_contact_mapping.json').read_text());assert sha(args.model)==mapping['model_sha256']
    write(out/'MANIFEST.json',dict(authorization='User: patch查询帮我优化',production_started=False,optimizer_updates=0,
        packets=[v['name'] for v in packets],device=torch.cuda.get_device_name(),external_competition=competition(),
        source_sha256={f:sha(ROOT/f) for f in ['scripts/audit_mano_patch_geometry.py','retarget/env/grasp/patch_query.py',
            'retarget/env/grasp/triangle_penetration.py','retarget/env/grasp/mano_comfree.py','retarget/env/grasp/mano_patch_anchor.py',
            'retarget/env/grasp/mano_patch_consistent.py','scripts/profile_mano_patch_pipeline.py']},
        tolerance=dict(objective_atol=1e-3,objective_rtol=2e-4,geometry_atol_m=1e-6,geometry_rtol=2e-5),
        non_unique_witness='Coordinates may differ only when BOTH witnesses pass the unchanged slot thresholds and therefore both have exactly zero slot cost; failing witnesses remain coordinate checked.',
        probes='All 3 saved stages and 64 fixed-seed local perturbations per packet; no optimization or resampling',
        timing='Same CUDA graph stage events; five warmed repeats; same buffer, dtype, physical parameters and thresholds'))
    decoder=ManoDecoder(args.model,args.device);decoder.configure_contact_vertices(mapping['mappings']);cache=MeshCache(args.device)
    reports=[]
    for packet in packets:
        name=packet['name'];frozen=args.baseline/'inputs'/(name+'.npz');a=dict(np.load(frozen));assert sha(frozen)==packet['sha256']
        ids=list(map(int,a['object_geometry_ids']));geo={i:dict(vertices=a[f'object_{i}_vertices'],faces=a[f'object_{i}_faces'],watertight=True) for i in set(ids)}
        surface=SurfaceBatch(geo,ids,a['object_pose_world'],a['scene_scale'],args.device,mesh_cache=cache)
        fixed=FixedContacts(decoder,mapping['mappings'][packet['type']]);saved=[dict(np.load(args.results/'quality'/name/s/'final.npz')) for s in ['input','velocity_input','cem']]
        n=len(saved[0]['variables']);anchors=torch.tensor(saved[0]['object_template_points'],device=args.device)
        q=torch.cat([torch.tensor(a['variables'],device=args.device) for a in saved]);rows=torch.arange(n,device=args.device).repeat(3)
        probes=q[:n][torch.arange(64,device=q.device)%n].clone()
        probes+=torch.randn_like(probes)*q.new_tensor([.002]*3+[.01]*3+[.03]*(q.shape[1]-6))
        q=torch.cat((q,probes));rows=torch.cat((rows,torch.arange(64,device=q.device)%n))
        adapter=PatchConsistentAdapter(decoder,fixed,surface,128,anchors=anchors)
        backend(adapter,'reference');rc,r=adapter.validate(q,rows)
        backend(adapter,'optimized');oc,o=adapter.validate(q,rows)
        try:
            torch.testing.assert_close(oc,rc,atol=1e-3,rtol=2e-4)
            fields=['penetration_depth_m','triangle_object_depth_m','patch_fit_cost','patch_fit_slot_error']
            for key in fields:torch.testing.assert_close(o[key],r[key],atol=1e-6,rtol=2e-5,msg=lambda m:key+': '+m)
            nonpassing=~(o['patch_fitted']&r['patch_fitted'])
            for key in ['patch_gap_m','patch_anchor_offset_m','patch_hand_points','patch_object_points']:
                torch.testing.assert_close(o[key][nonpassing],r[key][nonpassing],atol=1e-6,rtol=2e-5,msg=lambda m:key+': '+m)
            for key in ['valid','patch_fitted','patch_geometry_accepted','patch_joint_accepted','rollout_stable']:
                assert torch.equal(o[key],r[key]),(name,key)
            for j,a in enumerate(saved):
                for key in ['patch_fitted','patch_joint_accepted']:
                    np.testing.assert_array_equal(o[key][j*n:(j+1)*n].cpu(),a[key])
        except Exception:
            np.savez(out/'mismatch.npz',variables=q.cpu(),rows=rows.cpu(),**{'reference_'+k:v.cpu() for k,v in r.items()},**{'optimized_'+k:v.cpu() for k,v in o.items()})
            write(out/'status.json',dict(complete=False,failed_packet=name));raise
        report=dict(packet=name,scores_checked=len(q),all_gates_equal=True,
            max_patch_gap_error_m=float((o['patch_gap_m']-r['patch_gap_m']).abs().max()),
            max_anchor_offset_error_m=float((o['patch_anchor_offset_m']-r['patch_anchor_offset_m']).abs().max()),
            max_penetration_error_m=float((o['penetration_depth_m']-r['penetration_depth_m']).abs().max()),
            equivalent_face_ties=int((o['patch_face_ids']!=r['patch_face_ids']).sum()),
            passing_witness_ties=int(((o['patch_hand_points']-r['patch_hand_points']).norm(dim=-1)>1e-6).sum()),
            gpu_overflow_triangles=int(o['patch_gpu_overflow_count'].sum()))
        reports.append(report);write(out/'progress.json',dict(complete=False,packets=reports))
        if name==packets[0]['name']:
            timing={}
            for chunk in map(int,args.chunks.split(',')):
                del adapter;gc.collect();torch.cuda.empty_cache()
                x=q[torch.arange(chunk,device=q.device)%len(q)].clone();rr=rows[torch.arange(chunk,device=q.device)%len(q)].clone()
                adapter=PatchConsistentAdapter(decoder,fixed,surface,chunk,anchors=anchors)
                for mode in ['reference','optimized']:
                    backend(adapter,mode)
                    timing[f'{mode}_{chunk}']=graph_timings(adapter,x,rr,5)
                    write(out/'TIMINGS.json',timing)
        del adapter,surface;gc.collect();torch.cuda.empty_cache()
    write(out/'SUMMARY.json',dict(complete=True,passed=True,optimizer_updates=0,production_started=False,
        scores_checked=sum(v['scores_checked'] for v in reports),packets=reports,external_competition_after=competition()))
    print(json.dumps(reports,indent=2))


if __name__=='__main__':main()
