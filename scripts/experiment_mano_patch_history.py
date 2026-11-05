#!/usr/bin/env python3
"""Fixed 54-input validation of tangent memory and the consistent patch goal."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,json,sys,gc,fcntl,traceback,shutil
from pathlib import Path
from dataclasses import asdict,replace
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from retarget.solver.cem import DEFAULTS
from scripts.mano_patch_tools import (ManoDecoder,ShadowSkeleton,MeshCache,SurfaceBatch,FixedContacts,
    initialize,lm_ik,Settings,pose_problem,CemSettings,solve_patch_cem,save_result,write,sync,sha,now,competition)
from retarget.env.grasp.mano_patch_consistent import PatchConsistentAdapter
from retarget.env.grasp.spring_history import SpringHistorySettings
from retarget.env.grasp.comfree_rollout import RolloutSettings


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,default=Path('output/mano_patch_velocity_v4_20261105'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cohort',default='quality6,quality48')
    p.add_argument('--chunk',type=int,default=128)
    p.add_argument('--model',type=Path,default=Path('/home/zz/GraspSONIC/thirdparty/DexForge/assets/mano_v1_2/models/MANO_RIGHT.pkl'))
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--rollout-backend',choices=['warp','torch'],default='warp')
    p.add_argument('--geometry-backend',choices=['reference','optimized'],default='optimized')
    args=p.parse_args();out=args.output
    out.mkdir(parents=True,exist_ok=True);lock=(out/'experiment.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'MANIFEST.json').exists():raise FileExistsError('Use a new immutable experiment directory')
    base=args.baseline;inputs=base/'inputs';manifest=json.loads((inputs/'MANIFEST.json').read_text())
    packets=[v for v in manifest['packets'] if v['stage'] in args.cohort.split(',')]
    model=args.model
    mapping=json.loads((base/'assets/fixed_contact_mapping.json').read_text());assert sha(model)==mapping['model_sha256']
    torch.set_num_threads(1);torch.cuda.set_device(args.device)
    spring=DEFAULTS.spring;rollout=DEFAULTS.rollout
    sources=[Path(__file__),*[ROOT/f for f in ['retarget/env/grasp/spring_history.py','retarget/env/grasp/mano_patch_consistent.py',
        'retarget/env/grasp/spring_history_warp.py','retarget/env/grasp/mano_patch_anchor.py','retarget/env/grasp/mano_comfree.py',
        'retarget/env/grasp/patch_query.py','retarget/env/grasp/triangle_penetration.py','retarget/solver/cem.py']]]
    source_hash={str(f.relative_to(ROOT)):sha(f) for f in sources}
    for f in sources:
        dest=out/'source'/f.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
    write(out/'MANIFEST.json',dict(protocol='mano_patch_consistent_spring_history_h5_v5',authorization='User: 引入切向形变历史，并将贴合目标与验收改为一致；帮忙修复计算过慢',
        production_changed=False,production_stopped=True,samples=sum(v['samples'] for v in packets),packets=packets,
        start='Frozen pure-kinematic LM final poses from the velocity-v4 experiment; original Shadow-IK variable boundaries',
        current_velocity_reference='Frozen continuation/anchor_cem_early poses, rescored with the same v5 model',
        spring=asdict(spring),rollout=asdict(rollout),rollout_backend=args.rollout_backend,geometry_backend=args.geometry_backend,cem=dict(population=128,iterations=DEFAULTS.sampling.iterations,early_stop=True,retain_feasible=True),
        chunk=args.chunk,source_sha256=source_hash,model_sha256=sha(model),mapping_sha256=sha(base/'assets/fixed_contact_mapping.json'),
        input_manifest_sha256=sha(inputs/'MANIFEST.json'),device=torch.cuda.get_device_name(),external_competition=competition(),started=now()))
    write(out/'status.json',dict(state='loading',optimizer_updates=0))
    try:
        start=sync();decoder=ManoDecoder(model,args.device);decoder.configure_contact_vertices(mapping['mappings'])
        skeleton=ShadowSkeleton(base/'assets/shadow_assets/right.xml',args.device);cache=MeshCache(args.device)
        write(out/'loading.json',dict(seconds=sync()-start));reports={}
        for packet in packets:
            name=packet['name'];write(out/'status.json',dict(state='preparing',packet=name,updated=now()))
            a=dict(np.load(inputs/(name+'.npz')));assert sha(inputs/(name+'.npz'))==packet['sha256']
            lm=base/'run/quality'/name/'kinematic_only/final.npz';reference=base/'continuation/quality'/name/'anchor_cem_early/final.npz'
            old=dict(np.load(lm));current=dict(np.load(reference));q=torch.tensor(old['variables'],device=args.device)
            rows=torch.arange(len(q),device=q.device);t=sync()
            ids=list(map(int,a['object_geometry_ids']));geo={i:dict(vertices=a[f'object_{i}_vertices'],faces=a[f'object_{i}_faces'],watertight=True) for i in set(ids)}
            surface=SurfaceBatch(geo,ids,a['object_pose_world'],a['scene_scale'],args.device,mesh_cache=cache)
            fixed=FixedContacts(decoder,mapping['mappings'][packet['type']])
            target=skeleton(a['shadow_qpos']);mean,_,_=lm_ik(decoder,initialize(decoder,target),target,Settings())
            anchors=torch.tensor(old['object_template_points'],device=q.device)
            adapter=PatchConsistentAdapter(decoder,fixed,surface,args.chunk,anchors=anchors,rollout=rollout,spring_settings=spring,
                spring_backend=args.rollout_backend,geometry_backend=args.geometry_backend)
            problem=replace(pose_problem(mean,adapter),mean=q.clone());preparation=sync()-t
            folder=out/'quality'/name;folder.mkdir(parents=True,exist_ok=True)
            shutil.copy2(base/'run/quality'/name/'anchor_provenance.json',folder/'anchor_provenance.json')
            write(folder/'INPUT.json',dict(pure_lm_sha256=sha(lm),velocity_reference_sha256=sha(reference),frozen_input_sha256=sha(inputs/(name+'.npz')),
                preparation_seconds=preparation,original_bounds_unchanged=True))
            t=sync();adapter.validate(q,rows);warmup=sync()-t
            result={}
            result['input'],_=save_result(folder/'input',q,adapter,rows)
            result['velocity_input'],_=save_result(folder/'velocity_input',torch.tensor(current['variables'],device=q.device),adapter,rows)
            write(out/'status.json',dict(state='refining',packet=name,updated=now(),max_updates_per_pose=DEFAULTS.sampling.iterations))
            t=sync();state=solve_patch_cem(problem, adapter, seed=20261105);solving=sync()-t
            result['cem'],_=save_result(folder/'cem',state['parameters'],adapter,rows,state,dict(solve_seconds=solving,
                warmup_seconds=warmup,optimization_terms=['consistent_patch','original_penetration','terminal_velocity'],new_physics=True))
            before=np.load(folder/'input/final.npz');after=np.load(folder/'cem/final.npz');passed=before['patch_joint_accepted']
            np.testing.assert_array_equal(before['variables'][passed],after['variables'][passed])
            assert after['patch_joint_accepted'][passed].all()
            reports[name]=result;write(out/'packet_reports.json',reports)
            del adapter,surface,problem,state;gc.collect();torch.cuda.empty_cache()
        write(out/'status.json',dict(state='complete',updated=now(),production_changed=False))
    except Exception:
        write(out/'failure.json',dict(error=traceback.format_exc(),updated=now()));write(out/'status.json',dict(state='failed',updated=now()));raise

if __name__=='__main__':main()
