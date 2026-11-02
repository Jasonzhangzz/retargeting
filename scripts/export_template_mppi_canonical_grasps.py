"""Export and independently audit solved 1x contact correspondences for size IK."""
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from retarget.common.results import write_json
from retarget.env.grasp.template_mppi_inputs import TemplateContacts
from retarget.env.grasp.template_mppi_reference import template_signature
from retarget.env.geometry import ObjectMeshSurface,closest_triangles


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def export(roots,output,refined10=None):
 output=Path(output).resolve()
 if output.exists() or output.parent!=ROOT/'output':raise ValueError('Use a new output directory')
 sources={}
 for root in map(lambda p:Path(p).resolve(),roots):
  audit=read(root/'independent_result_audit.json')
  if not audit['passed'] or not audit['complete']:raise ValueError(f'Unaudited source: {root}')
  for entry in read(root/'manifest.json')['cases']:
   if abs(entry['primitive']['size_scale']-1)<1.e-10:sources[entry['template']]=(root,entry)
 if len(sources)!=31:raise ValueError(f'Expected all 31 supported canonical grasps, got {len(sources)}')
 output.mkdir();results=[]
 for name,(root,entry) in sorted(sources.items(),key=lambda x:int(x[0].partition('_')[0])):
  case=(root/entry['trajectory']).parent;template=TemplateContacts(**read(case/'template_contacts.json'))
  q=np.load(root/entry['trajectory'])['qpos'][-1];summary=read(root/entry['summary']);d=summary['geometry_diagnostics']
  files=[root/entry['model'],root/entry['trajectory'],root/entry['summary'],case/'template_contacts.json',case/'projection_force_closure.json']
  refinement=None
  if name=='10_Power_Disk' and refined10 is not None:
   candidate=read(refined10)
   if candidate['diagnostics']['cost'] < d['cost']:
    refinement=output/'refined_sources/10_Power_Disk.json';write_json(refinement,candidate)
    q=np.array(candidate['qpos']);d=candidate['diagnostics'];files.append(refinement)
  model=mujoco.MjModel.from_xml_path(str(root/entry['model']));data=mujoco.MjData(model);data.qpos[:]=q
  mujoco.mj_kinematics(model,data);mujoco.mj_collision(model,data)
  ids=np.array([model.body(n).id for n in template.body_names]);rot=data.xmat[ids].reshape(-1,3,3)
  points=data.xpos[ids]+np.einsum('nij,nj->ni',rot,template.local_points)
  normals=np.einsum('nij,nj->ni',rot,template.local_normals)
  fk_error=max(np.max(np.abs(points-d['hand_points'])),np.max(np.abs(normals-d['hand_normals'])))
  obj=model.body('grasp_object').id;depth=np.zeros(2)
  for c in data.contact:
   body=model.geom_bodyid[c.geom]
   if np.all(body==obj):continue
   kind=0 if obj in body else 1;depth[kind]=max(depth[kind],-c.dist)
  surface=ObjectMeshSurface.from_model(model,obj);projected=np.array(d['projected_points']);near=closest_triangles(projected,surface.triangles)
  projection_error=np.sqrt(((near-projected[:,None])**2).sum(-1).min(-1)).max()
  m=d['closure_model'];force=np.array(d['response_contact_forces_n']);spin=np.array(d['response_intrinsic_torques_nm'])
  wrench=np.concatenate((force.sum(1),(np.cross(projected[None]-np.array(m['center']),force)+spin).sum(1)/m['radius_m']),-1)
  loads=np.concatenate((np.zeros((1,6)),np.eye(6),-np.eye(6)))*m['test_force_n'];residual=(wrench+loads)/m['test_force_n']
  force_error=float(np.max(np.abs(residual-np.array(d['scenario_residuals']))))
  checks=dict(cpu_fk=bool(fk_error<2.e-6),physical_bounds=bool(np.all(q>=model.jnt_range[:,0]-1.e-6) and np.all(q<=model.jnt_range[:,1]+1.e-6)),
   self_penetration=bool(depth[1]<=1.e-5+1.e-7),surface_projection=bool(projection_error<2.e-6),force_witness=bool(force_error<1.e-4))
  if not all(checks.values()):raise RuntimeError((name,checks,depth.tolist(),projection_error,force_error))
  record=dict(policy='optimized_canonical_grasp_contacts_v1',template=name,template_signature=template_signature(template),
   primitive=entry['primitive'],qpos=q.tolist(),object_points=d['projected_points'],object_outward_normals=d['object_normals'],
   source_sha256={str(p):sha(p) for p in files},source_case=entry['name'],source_batch=str(root),source_model=str(root/entry['model']),
   source_projected_closure=d['projected_closure'],source_coverage=d['coverage'],canonical_refinement=str(refinement) if refinement else None,
   checks=checks,cpu_fk_error=fk_error,projection_error_m=projection_error,force_residual_error=force_error)
  write_json(output/f'{name}.json',record);results.append(dict(template=name,checks=checks,source_coverage=d['coverage'],refined=refinement is not None))
 write_json(output/'canonical_audit.json',dict(passed=True,canonical_grasps=len(results),cases=results,
  source_audits={str(Path(r).resolve()/'independent_result_audit.json'):sha(Path(r)/'independent_result_audit.json') for r in roots}))
 print(output,len(results))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--refined10',type=Path)
 a=p.parse_args();export(a.source,a.output,a.refined10)
