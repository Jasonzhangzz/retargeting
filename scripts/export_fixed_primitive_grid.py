#!/usr/bin/env python3
"""Recover/package completed grids without any optimizer updates."""
import argparse,json,sys,datetime,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.mano_patch_tools import write

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--strip-legacy-fields',action='store_true');a=p.parse_args();out=a.output;entries=[]
 manifest=json.loads((out/'MANIFEST.json').read_text())
 for folder in sorted((out/'grids_fixed').iterdir(),key=lambda p:int(p.name.split('_')[0])):
  if not (folder/'complete.json').exists():continue
  completion=json.loads((folder/'complete.json').read_text());entry=completion['entry'];pose_file=folder/'poses.npz';z=np.load(pose_file)
  if a.strip_legacy_fields and any(k in z for k in ('penetration_m','closure_certified')):
   arrays={k:z[k] for k in z.files if k not in ('penetration_m','closure_certified')};z.close();temp=folder/'poses.export.tmp.npz';np.savez_compressed(temp,**arrays);temp.replace(pose_file)
   completion['worker_output_sha256']=completion['sha256'];completion['sha256']=hashlib.sha256(pose_file.read_bytes()).hexdigest();completion['export_schema']='current_patch_cem_no_projected_closure';write(folder/'complete.json',completion);z=np.load(pose_file)
  n=len(z['size_scale']);q=z['mano_variables'];angles=z['mano_angles'];op=z['object_pose_world']
  safe=lambda arr:[float(x) if np.isfinite(x) else None for x in arr]
  data={'name':entry['name'],'kind':entry['kind'],'axis':entry['axis'],'scales':z['size_scale'].tolist(),'source_scales':[r['size_scale'] for r in entry['source_rows']],'radius_m':z['radius_m'].tolist(),'height_m':z['height_m'].tolist(),'mano_translation':q[:,:3].tolist(),'mano_root_quat':Rotation.from_rotvec(q[:,3:6]).as_quat().tolist(),'mano_joint_quat':Rotation.from_rotvec(angles.reshape(-1,3)).as_quat().reshape(n,15,4).tolist(),'wuji_qpos':z['wuji_qpos'].tolist(),'wuji_root_quat':Rotation.from_euler('XYZ',z['wuji_qpos'][:,3:6]).as_quat().tolist(),'final_cost':safe(z['final_cost']),'initial_cost':safe(z['initial_cost']),'valid':z['valid'].tolist(),'coverage':z['patch_coverage'].tolist(),'penetration_mm':(z['penetration_depth_m']*1000).tolist(),'contact_passed':z['base_patch_joint_accepted'].tolist(),'grasp_passed':z['grasp_passed' if 'grasp_passed' in z else 'patch_joint_accepted'].tolist(),'finger_gap_rms_mm':(z['finger_gap_rms_m']*1000).tolist(),'object_translation':op[:,:3].tolist(),'object_quat':op[:,[4,5,6,3]].tolist(),'object_anchors':z['object_anchors'].tolist(),'patch_fitted':z['patch_fitted'].tolist(),'fixed_wrist':True,'current_patch_cem':True}
  keys=('name','kind','axis','knots','source_knots','min_scale','max_scale','url','source_rows','wrist_max_error_m_rad','coverage_median','contact_passed','pose_and_contact_passed','tip_gap_rms_after_mm','seconds','max_adjacent_joint_step_rad','solver','cost')
  entry={k:entry[k] for k in keys};finite=z['final_cost'][np.isfinite(z['final_cost'])];entry.update(cost_median=float(np.median(finite)) if len(finite) else None,invalid_knots=int((~z['valid']).sum()),cem_updates=int(z['iterations'].sum()))
  write(out/'web/data'/f"{entry['name']}.json",data);entries.append(entry)
 write(out/'web/index.json',{'entries':entries,'complete':len(entries)==31,'total_knots':sum(e['knots'] for e in entries),'fixed_wrist':True,'solver':manifest['solver'],'cost':manifest['cost'],'run':out.name,'data_schema':'current_patch_cem_no_projected_closure','solver_sha256':manifest['source_code_sha256'].get('retarget/solver/cem.py'),'generated_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
 print(json.dumps({'types':len(entries),'knots':sum(e['knots'] for e in entries),'invalid_knots':sum(e['invalid_knots'] for e in entries),'optimizer_updates':0}))
if __name__=='__main__':main()
