#!/usr/bin/env python3
"""Read-only export of original native Shadow states for a portable Viser viewer."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import native_graspadmm_retest as runner


def run(root,output):
    source=runner.load_upstream(root/'upstream');runner.verify_assets(root,source)
    cfg=runner.configuration(source,root,'grasp_admm')
    quality=json.loads((root/'quality_operation.json').read_text())
    evaluation=json.loads((root/'native_evaluation.json').read_text())
    physical={r['output']:r for r in evaluation['rows']}
    diagnostics=json.loads((root/'diagnostics.json').read_text()) if (root/'diagnostics.json').exists() else None
    diagnostic={r['input']:r for r in diagnostics['rows']} if diagnostics else {}
    wrench_path=root/'wrench_evaluation.json'
    wrench={r['index']:r for r in json.loads(wrench_path.read_text())['rows']} if wrench_path.exists() else {}
    output.mkdir(parents=True,exist_ok=True);geometry={};poses={};entries=[];envs={};geom_ids={};scenes={}
    for i,r in enumerate(quality['rows']):
        inp=np.load(r['input'],allow_pickle=True).item();scene=Path(r['input']).parent.name
        if scene not in envs:
            env=source.sim.MuJoCo_OptEnv(hand_cfg=source.sim.HandCfg(cfg.hand.xml_path,freejoint=True),scene_cfg=source.files.load_scene_cfg(inp['scene_path']),sim_cfg=source.sim.MuJoCo_OptCfg(obj_margin=.001))
            envs[scene]=env;m=env._model
            ids=[g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith(env._cfg.hand_prefix) and int(m.geom_type[g])==7 and m.geom_contype[g]==0]
            if not ids:
                ids=[g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith(env._cfg.hand_prefix) and int(m.geom_type[g])==7]
            if not ids:raise RuntimeError('No original hand meshes')
            geom_ids[scene]=ids
            if not geometry:
                for k,g in enumerate(ids):
                    mid=m.geom_dataid[g];a,n=m.mesh_vertadr[mid],m.mesh_vertnum[mid];b,f=m.mesh_faceadr[mid],m.mesh_facenum[mid]
                    geometry[f'hand_{k}_vertices']=m.mesh_vert[a:a+n].copy();geometry[f'hand_{k}_faces']=m.mesh_face[b:b+f].copy()
            mesh,*_=source.core.load_obj_trimesh(inp)
            geometry[scene+'_vertices']=np.asarray(mesh.vertices,np.float32);geometry[scene+'_faces']=np.asarray(mesh.faces,np.int32)
            scenes[scene]=dict(center=mesh.bounds.mean(0).tolist(),radius=float(np.linalg.norm(mesh.bounds[1]-mesh.bounds[0])/2))
        env=envs[scene];ids=geom_ids[scene]
        row=dict(index=i,scene=scene,input=r['input'],native_operation_pass=r['native_operation_output'],native_gsr_pass=bool(physical.get(r.get('output'),{}).get('success',False)),fallback_used=r['fallback_used'],final_available=r['captured_final_state'],hand_cbody=r['hand_cbody'],input_sha256=r['input_sha256'],final_sha256=r.get('captured_final_sha256'),diagnostics=diagnostic.get(r['input']))
        row['wrench']=wrench.get(i)
        failed=[f for f in r['filters'] if not f['passed']]
        row['failure_reason']=('native_gsr_pass' if row['native_gsr_pass'] else ('six_direction_failed' if row['native_operation_pass'] else ', '.join(dict.fromkeys(f.get('name','')+': '+f.get('check','forward') for f in failed))))
        states=np.load(root/'quality/cases'/f'{i:05d}'/'final_state.npz')
        for stage,q in [('initial',inp['grasp_qpos']),('final',states['final_grasp_qpos'] if r['captured_final_state'] else None)]:
            if q is None:continue
            env.reset_qpos(np.asarray(q).reshape(-1));d=env._data
            poses[f'{i}_{stage}_positions']=d.geom_xpos[ids].copy()
            poses[f'{i}_{stage}_wxyz']=Rotation.from_matrix(d.geom_xmat[ids].reshape(-1,3,3)).as_quat(scalar_first=True)
            if stage=='initial':poses[f'{i}_initial_slots']=inp['hand_cpn_w']
        contacts=root/'quality/cases'/f'{i:05d}'/'diagnostic_contacts.npz'
        if contacts.exists():
            with np.load(contacts) as a:
                for key in a.files:poses[f'{i}_{key}']=a[key]
        entries.append(row)
    np.savez_compressed(output/'geometry.npz',**geometry);np.savez_compressed(output/'poses.npz',**poses)
    runner.write(output/'manifest.json',dict(commit=runner.COMMIT,model=cfg.hand_name,type=runner.TYPE,samples=len(entries),hand_geoms=len(geom_ids[next(iter(geom_ids))]),scenes=scenes,cases=entries,diagnostics_complete=bool(diagnostics and diagnostics['complete']),summary={k:v for k,v in diagnostics.items() if k!='rows'} if diagnostics else None,optimization_updates=0))
    runner.verify_assets(root,source);print('EXPORTED',output,len(entries),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.root.resolve(),a.output.resolve())
