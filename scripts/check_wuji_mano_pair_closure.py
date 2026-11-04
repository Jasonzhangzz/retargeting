#!/usr/bin/env python3
"""Evaluate fixed-projection closure of selected final pose pairs; never change poses."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import sys,argparse,time,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch,mujoco
from retarget.viewer.mano_results import ManoSnapshot,read,write
from retarget.env.grasp.dexonomy_grasp_static import StaticModels
from retarget.env.grasp.dexonomy_replay import closure

def evaluate(model,data,body,points,normals):
    before=time.monotonic();report,arrays=closure(model,data,body,dict(points=points,object_points=points,normals=normals))
    if 'closure_model' in report:
        m=report['closure_model'];f=arrays['closure_response_forces_n'];spin=arrays['closure_response_intrinsic_torques_nm']
        wrench=np.concatenate((f.sum(1),(np.cross(points[None]-np.array(m['center']),f)+spin).sum(1)/m['radius_m']),-1)
        loads=np.concatenate((np.zeros((1,6)),np.eye(6),-np.eye(6)))*m['test_force_n']
        residual=np.linalg.norm((wrench+loads)/m['test_force_n'],axis=-1)
        report['audited_max_wrench_residual']=float(residual[1:].max());report['witness_audit_passed']=bool(not report['certified'] or residual[1:].max()<=m['certificate_tolerance']+1e-6)
        if not report['witness_audit_passed']:raise ValueError('Closure witness residual failed independent audit')
    report['evaluation_wall_seconds']=time.monotonic()-before
    return report,arrays

def main():
    p=argparse.ArgumentParser();p.add_argument('--selection',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1)
    chosen=read(a.selection);models=StaticModels(ROOT/'data/dexonomy',2,visual=False);reports=[]
    for item in chosen['items']:
        s=ManoSnapshot(item['mano_shard']);i=item['mano_row'];e=s.manifest['cases'][i]
        record=dict(scene_path=e['scene_path'],scene_scale=np.array([e['scene_scale']]),grasp_qpos=np.zeros((1,29)))
        env,_,_=models.get(record,0);model,data=env._model,env._data
        data.qpos[:29]=s.source_pose(i);data.qpos[29:]=s.poses['object_pose_world'][i];mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data)
        with np.load(Path(item['wuji_shard'])/'poses.npz',allow_pickle=False) as z:
            wp=z['object_contact_points'][item['wuji_row']];wn=z['object_contact_normals'][item['wuji_row']]
        mano,ma=evaluate(model,data,env._obj_active_bodyid,s.poses['mapped_object_points'][i],s.poses['mapped_object_normals'][i])
        wuji,wa=evaluate(model,data,env._obj_active_bodyid,wp,wn)
        entry=dict(number=item['number'],key=item['key'],mano_row=i,wuji_row=item['wuji_row'],mano=mano,wuji2=wuji,
            mano_geometry_accepted=item['mano_geometry_accepted'],wuji_geometry_accepted=item['wuji_geometry_accepted'],
            combined_mano=bool(mano['certified'] and item['mano_geometry_accepted']),combined_wuji2=bool(wuji['certified'] and item['wuji_geometry_accepted']))
        np.savez_compressed(a.selection.parent/f"pair_{item['number']:02d}_closure_witness.npz",**{'mano_'+k:v for k,v in ma.items()},**{'wuji2_'+k:v for k,v in wa.items()})
        reports.append(entry);write(a.selection.parent/'closure_report.json',dict(complete=len(reports)==len(chosen['items']),expected=len(chosen['items']),evaluated=len(reports),
            semantics='Fixed virtual object-surface projections; free contact force witnesses; does not certify joint-torque feasibility or actual physical contact; geometry checked separately.',items=reports))
        print(json.dumps(dict(number=item['number'],key=item['key'],mano_certified=mano['certified'],wuji2_certified=wuji['certified'],mano_residual=mano.get('audited_max_wrench_residual'),wuji_residual=wuji.get('audited_max_wrench_residual'))),flush=True)
if __name__=='__main__':main()
