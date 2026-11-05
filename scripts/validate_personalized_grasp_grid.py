#!/usr/bin/env python3
"""Validate every solved knot and browser uploads against saved geometry/FK."""
import argparse,json,hashlib,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,mujoco,torch,xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation
from retarget.env.grasp.hand import hand_tree
from retarget.env.grasp.personalized_primitive import FrozenPersonalizedDecoder
from retarget.env.grasp.primitive_problem import AnatomicalPrimitiveDecoder,canonicalize_wuji_wrist
from retarget.env.grasp.primitives import primitive_geometry

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output;index=json.loads((out/'web/index.json').read_text());fixed=(out/'MANIFEST.json').exists();source=Path(json.loads((out/('MANIFEST.json' if fixed else 'GRID_MANIFEST.json')).read_text())['source']);decoder=(AnatomicalPrimitiveDecoder if fixed else FrozenPersonalizedDecoder)(out/'saved_left','cpu');original=json.loads((source/'manifest.json').read_text())['cases'];cache={r['name']:r for r in original};counts=0;improved=0;max_decode=0;source_knots=0
 archives={}
 for entry in index['entries']:
  folder='grids_fixed' if fixed else 'grids_refined' if (out/'grids_refined'/entry['name']/'poses.npz').exists() else 'grids';z=np.load(out/folder/entry['name']/'poses.npz');archives[entry['name']]=z
  assert np.all(np.diff(z['size_scale'])>0)
  for row in entry['source_rows']:
   i=int(np.argmin(abs(z['size_scale']-row['size_scale'])));assert abs(z['size_scale'][i]-row['size_scale'])<1e-12
   np.testing.assert_array_equal(z['wuji_qpos_original' if fixed else 'wuji_qpos'][i],np.load(source/'cases'/row['name']/'grasp.npz')['qpos']);source_knots+=1
  if not fixed:assert np.all(z['final_cost']<=z['initial_cost']+1e-3)
  else:
   assert np.isfinite(z['mano_variables']).all()
   assert np.all(np.isfinite(z['final_cost']) | ~z['valid'])
   assert not np.any(z['patch_joint_accepted'] & ~z['valid'])
   assert np.max(np.ptp(z['mano_variables'][:,:6],axis=0))<1e-7
   assert np.max(np.abs(np.diff(z['mano_variables'][:,6:],axis=0)))<.100001
   canonical=int(np.argmin(abs(z['size_scale']-1)))
   wq,op=canonicalize_wuji_wrist(z['wuji_qpos_original'],canonical)
   np.testing.assert_allclose(z['wuji_qpos'],wq,atol=1e-12)
   np.testing.assert_allclose(z['object_pose_world'],op,atol=1e-12)
   local=np.array([z['object_anchor_normalized']*[r,r,h] for r,h in zip(z['radius_m'],z['height_m'])])
   rot=Rotation.from_quat(op[:,[4,5,6,3]]).as_matrix()
   np.testing.assert_allclose(z['object_anchors'],np.einsum('nij,nkj->nki',rot,local)+op[:,None,:3],atol=2e-8)
   vertices,faces=primitive_geometry(entry['kind'],float(z['radius_m'][canonical]),float(z['height_m'][canonical]))
   triangles=vertices[faces[z['object_anchor_faces'][0]]]
   expected=(triangles*z['object_anchor_barycentric'][0,:,:,None]).sum(1)
   np.testing.assert_allclose(local[canonical],expected,atol=3e-8)
   data_web=json.loads((out/'web/data'/f"{entry['name']}.json").read_text())
   np.testing.assert_allclose(data_web['coverage'],z['patch_coverage'])
   np.testing.assert_array_equal(data_web['grasp_passed'],z['grasp_passed' if 'grasp_passed' in z else 'patch_joint_accepted'])
  counts+=len(z['size_scale']);improved+=int(np.sum(z['final_cost']<z['initial_cost']))
  with torch.no_grad():v,j,_=decoder(torch.tensor(z['mano_variables']),full=True)
  max_decode=max(max_decode,float(np.max(abs(v.numpy()-z['mano_vertices']))))
  np.testing.assert_allclose(v.numpy(),z['mano_vertices'],atol=8e-7)
 render_body_ids=[v['body'] for v in json.loads((out/'web/model.json').read_text())['wuji']['meshes']]
 model=mujoco.MjModel.from_xml_string(ET.tostring(hand_tree('wuji2'),encoding='unicode'));data=mujoco.MjData(model);browser=json.loads((out/'browser_samples.json').read_text());max_browser=0;max_wuji=0
 for row in browser:
  z=archives[row['name']];i=int(np.argmin(abs(z['size_scale']-row['scale'])));assert abs(z['size_scale'][i]-row['scale'])<1e-9
  actual=np.array(row['mano']).reshape(778,3);max_browser=max(max_browser,float(np.max(abs(actual-z['mano_vertices'][i]))));np.testing.assert_allclose(actual,z['mano_vertices'][i],atol=8e-7)
  if fixed:
   np.testing.assert_allclose(row['objectPosition'],z['object_pose_world'][i,:3],atol=2e-8)
   np.testing.assert_allclose(Rotation.from_quat(row['objectQuaternion']).as_matrix(),Rotation.from_quat(z['object_pose_world'][i,[4,5,6,3]]).as_matrix(),atol=2e-8)
  data.qpos[:26]=z['wuji_qpos'][i];mujoco.mj_kinematics(model,data)
  wp=np.array(row['wujiPositions']);max_wuji=max(max_wuji,float(np.max(abs(wp[render_body_ids]-data.xpos[render_body_ids]))));np.testing.assert_allclose(wp[render_body_ids],data.xpos[render_body_ids],atol=2e-7)
  np.testing.assert_allclose(Rotation.from_quat(row['wujiRotations']).as_matrix()[render_body_ids],data.xmat.reshape(-1,3,3)[render_body_ids],atol=2e-7)
 s=json.loads((out/'saved_left/manifest.json').read_text())
 for f,h in s['sha256'].items():assert hashlib.sha256((out/'saved_left'/f).read_bytes()).hexdigest()==h,f
 result={'complete':len(index['entries'])==31,'types':len(index['entries']),'solved_mano_knots':counts,'original_wuji_knots_preserved':source_knots,'lower_cost_than_initial':improved,'max_torch_reconstruction_m':max_decode,'max_browser_skin_vertex_error_m':max_browser,'max_browser_wuji_body_error_m':max_wuji,'immutable_left_snapshot':True,'optimizer':'current patch CEM with fixed wrists/material regions' if fixed else 'legacy CEM objective','physical_grasp_success_verified':False}
 if fixed:
  result.update(fixed_wrists=True,canonical_wuji_object_geometry_preserved=True,fixed_material_anchor_coordinates=True,contact_passed=sum(int(z['patch_joint_accepted'].sum()) for z in archives.values()),median_patch_coverage=float(np.median(np.concatenate([z['patch_coverage'] for z in archives.values()]))),finger_spacing_rms_mm=float(np.sqrt(np.mean(np.concatenate([z['finger_gap_error_m'] for z in archives.values()])**2))*1000))
 (out/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
