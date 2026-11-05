#!/usr/bin/env python3
"""Current patch CEM, fixed wrists/normalized object regions and continuation."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import sys,argparse,json,shutil,fcntl,signal,time,gc
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch,mujoco,xml.etree.ElementTree as ET
from dataclasses import replace
from scipy.spatial.transform import Rotation
from retarget.env.grasp.primitive_problem import AnatomicalPrimitiveDecoder,frozen_wrist_problem,canonicalize_wuji_wrist,reference_wrist_problem
from retarget.env.grasp.personalized_primitive import anatomical_contacts
from retarget.env.grasp.mano_solver import FixedContacts,SurfaceBatch,MeshCache,initialize,lm_ik,Settings
from retarget.env.grasp.pose_gpu import WujiPoseAdapter,ModelFK
from retarget.env.grasp.hand import hand_tree
from retarget.env.grasp.primitives import primitive_geometry
from retarget.solver.cem import DEFAULTS
from retarget.solver.cem import solve_patch_cem
from retarget.env.grasp.mano_patch_consistent import PatchConsistentAdapter
from scripts.mano_patch_tools import lm_anchors,write,sha,sync
ROOT=Path(__file__).resolve().parents[1]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--previous',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--only');ap.add_argument('--compatible-resume',type=Path);ap.add_argument('--allow-export-upgrade',action='store_true');args=ap.parse_args();out=args.output;old=args.previous;out.mkdir(parents=True,exist_ok=True);lock=(out/'.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);stop=[False];signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));torch.set_num_threads(1)
 if not (out/'saved_left').exists():shutil.copytree(old/'saved_left',out/'saved_left')
 if not (out/'web').exists():shutil.copytree(old/'web',out/'web',ignore=shutil.ignore_patterns('data','first-pass'))
 index=json.loads((old/'web/index.json').read_text());source=Path(json.loads((old/'GRID_MANIFEST.json').read_text())['source'])
 manifest={'revision':'shared_current_patch_cem_fixed_wrist_v9','source':str(source),'previous':str(old.resolve()),'snapshot_sha256':sha(out/'saved_left/manifest.json'),'solver':'retarget.solver.cem.solve_patch_cem','cost':'unaltered PatchConsistentAdapter: mano_patch_consistent_spring_history_h5_v5','canonical_iterations':DEFAULTS.sampling.iterations,'anchor_lm_steps':[96,48],'size_iterations':DEFAULTS.sampling.iterations,'population':128,'continuity_radius_rad':.1,'wrist':'reference wrist optimized once within 25mm/0.35rad, then immutable for every size','wuji':'same fixed canonical wrist; original cached finger angles retained, original full poses separately saved','object_regions':'canonical WUJI surface projection once; identical face/barycentric material coordinates resized continuously','object_placement':'exact per-size global rigid frame change fixing WUJI wrist while preserving cached hand-object transforms','source_code_sha256':{f:sha(ROOT/f) for f in ['retarget/env/grasp/primitive_problem.py','retarget/env/grasp/mano_patch_consistent.py','retarget/env/grasp/mano_patch_anchor.py','retarget/env/grasp/mano_comfree.py','retarget/env/grasp/patch_query.py','retarget/env/grasp/triangle_penetration.py','retarget/env/grasp/spring_history.py','retarget/env/grasp/spring_history_warp.py','retarget/env/grasp/comfree_rollout.py','retarget/env/grasp/contact_geometry.py','scripts/mano_patch_tools.py','retarget/env/grasp/personalized_primitive.py','retarget/env/grasp/mano_contact_collision.py','retarget/env/grasp/mano_solver.py','moxian_web/hand_anatomy.py','retarget/solver/cem.py','scripts/fix_personalized_grasp_grid.py']}}
 if (out/'MANIFEST.json').exists() and json.loads((out/'MANIFEST.json').read_text())!=manifest:
  previous_manifest=json.loads((out/'MANIFEST.json').read_text());before=json.loads(json.dumps(previous_manifest));after=json.loads(json.dumps(manifest))
  before['source_code_sha256'].pop('scripts/fix_personalized_grasp_grid.py');after['source_code_sha256'].pop('scripts/fix_personalized_grasp_grid.py')
  if args.compatible_resume:
   receipt=json.loads(args.compatible_resume.read_text())
   assert receipt['passed'] and receipt['from_manifest_sha256']==sha(out/'MANIFEST.json')
   assert receipt['to_source_code_sha256']==manifest['source_code_sha256']
   config_before={k:v for k,v in previous_manifest.items() if k!='source_code_sha256'};config_after={k:v for k,v in manifest.items() if k!='source_code_sha256'}
   assert config_before==config_after
  elif not args.allow_export_upgrade or before!=after:raise ValueError('Changed run configuration; use new output directory')
  write(out/'MANIFEST_before_export_fix.json',previous_manifest)
  write(out/'EXPORT_UPGRADE.json',{'reason':'Nonfinite costs exported as null with explicit invalid status; recover browser export after partial completion','optimizer_updates':0,'pose_objective_unchanged':True})
 write(out/'MANIFEST.json',manifest)
 for f in manifest['source_code_sha256']:
  dest=out/'source'/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,dest)
 model=mujoco.MjModel.from_xml_string(ET.tostring(hand_tree('wuji2'),encoding='unicode'));native=WujiPoseAdapter(model,'cuda:0');fk=ModelFK(model,'cuda:0');m=AnatomicalPrimitiveDecoder(out/'saved_left');cache=MeshCache('cuda:0');entries=[]
 for entry in index['entries']:
  name=entry['name']
  if args.only and name not in args.only.split(','):continue
  if stop[0] or (out/'STOP').exists():write(out/'status.json',{'state':'stopped','types':len(entries)});return
  folder=out/'grids_fixed'/name
  if (folder/'complete.json').exists():entries.append(json.loads((folder/'complete.json').read_text())['entry']);continue
  z=dict(np.load(old/'grids'/name/'poses.npz'));n=len(z['size_scale']);canonical=int(np.argmin(abs(z['size_scale']-1)));qref=z['wuji_qpos'][canonical].copy();wq,object_pose=canonicalize_wuji_wrist(z['wuji_qpos'],canonical)
  target=native.keypoints(torch.tensor(wq,device='cuda:0',dtype=torch.float64)).float();mean=initialize(m,target[canonical:canonical+1]);wrist=mean[:,:6].clone()
  try:seed,_,_=lm_ik(m,mean,target[canonical:canonical+1],Settings(ik_steps=48))
  except Exception:seed=mean
  seed[:,:6]=wrist;seed[:,6:]=seed[:,6:].clamp(m.anatomical_low,m.anatomical_high)
  tc=json.loads((source/'cases'/entry['source_rows'][0]['name']/'template_contacts.json').read_text());records=anatomical_contacts(m,model,tc);m.configure_contact_vertices({name:records});fixed=FixedContacts(m,records)
  r0=float(z['radius_m'][canonical]);h0=float(z['height_m'][canonical]);geo={i:dict(zip(('vertices','faces'),primitive_geometry(entry['kind'],float(r),float(h))),watertight=True) for i,(r,h) in enumerate(zip(z['radius_m'],z['height_m']))}
  identity=np.tile([0.,0.,0.,1.,0.,0.,0.],(n,1));surface=SurfaceBatch(geo,list(geo),identity,np.ones(n),'cuda:0',mesh_cache=cache)
  p,r=fk.frames(torch.tensor(wq[canonical:canonical+1],device='cuda:0',dtype=torch.float32));ids=[model.body(rec['wuji_body']).id for rec in records];local=seed.new_tensor([rec['wuji_local_point'] for rec in records]);points=p[:,ids]+(r[:,ids]@local[None,:,:,None]).squeeze(-1)
  anchor0,_,_,anchor_faces,anchor_bary=surface.query(points,torch.tensor([canonical],device='cuda:0'));normalized=anchor0[0].cpu().numpy()/[r0,r0,h0];anchors_local=np.array([normalized*[r,r,h] for r,h in zip(z['radius_m'],z['height_m'])]);origin=object_pose[:,:3];object_rotation=Rotation.from_quat(object_pose[:,[4,5,6,3]]).as_matrix();anchors=np.einsum('nij,nkj->nki',object_rotation,anchors_local)+origin[:,None]
  surface=SurfaceBatch(geo,list(geo),object_pose,np.ones(n),'cuda:0',mesh_cache=cache)
  adapter=PatchConsistentAdapter(m,fixed,surface,128,anchors=torch.tensor(anchors,device='cuda:0',dtype=torch.float32),geometry_backend='optimized')
  assert type(adapter) is PatchConsistentAdapter
  folder.mkdir(parents=True,exist_ok=True);write(folder/'mapping.json',records)
  order=[canonical]+list(range(canonical+1,n))+list(range(canonical-1,-1,-1));solutions={};parts_saved={};costs={};initial_costs={};iterations={};start=sync();checkpoint=folder/'checkpoint.npz';problem=None
  if checkpoint.exists():
   cp=np.load(checkpoint);solutions={int(i):torch.tensor(v,device='cuda:0')[None] for i,v in zip(cp['indices'],cp['variables'])};initial_costs=dict(zip(cp['indices'].tolist(),cp['initial_costs'].tolist()));iterations=dict(zip(cp['indices'].tolist(),cp['iterations'].tolist()));wrist=solutions[canonical][:,:6].clone() if canonical in solutions else wrist
  if canonical in solutions:
   for value in solutions.values():
    assert float((value[:,:6]-wrist).abs().max())<1e-6
    value[:,:6]=wrist
  for step,i in enumerate(order):
   if i in solutions:continue
   if stop[0] or (out/'STOP').exists():return
   parent=i-1 if i>canonical else i+1;reference=solutions.get(parent) if i!=canonical else None
   initial=seed.clone() if reference is None else reference.clone()
   if reference is not None:initial[:,:6]=wrist
   problem=reference_wrist_problem(initial,m) if reference is None else frozen_wrist_problem(initial,m,reference,continuity_radius=.1);row=torch.tensor([i],device='cuda:0')
   # Existing anchor LM initializer uses one active row, without changing wrists.
   from types import SimpleNamespace
   shim=SimpleNamespace(decoder=m,fixed=fixed,required_groups=adapter.required_groups,anchors=adapter.anchors[i:i+1]);proposal=lm_anchors(problem,shim,steps=96 if reference is None else 48)['parameters']
   if reference is not None:proposal[:,:6]=wrist
   previous_cost,previous_parts=adapter.validate(initial,row);proposal_cost,proposal_parts=adapter.validate(proposal,row)
   if reference is None or (bool(proposal_parts['patch_joint_accepted'][0]) and not bool(previous_parts['patch_joint_accepted'][0])) or (bool(proposal_parts['patch_joint_accepted'][0])==bool(previous_parts['patch_joint_accepted'][0]) and float(proposal_cost[0])<float(previous_cost[0])):initial=proposal
   problem=replace(problem,mean=initial)
   before,bparts=adapter.validate(initial,row)
   class RowAdapter:
    retain_feasible=True
    def __init__(self,owner,source_row):self.owner,self.source_row=owner,source_row
    def evaluate_candidates(self,x,rows):return self.owner.evaluate_candidates(x,rows*0+self.source_row)
    def validate(self,x,rows):return self.owner.validate(x,rows*0+self.source_row)
   state=solve_patch_cem(problem, RowAdapter(adapter,i), seed=20261105 + int(name.split('_')[0]) * 1000 + i)
   solutions[i]=state['parameters'];
   if i==canonical:wrist=solutions[i][:,:6].clone()
   initial_costs[i]=float(before[0]);iterations[i]=int(state['iterations'][0]);costs[i]=float(state['cost'][0])
   if step%4==0 or step==len(order)-1:np.savez_compressed(checkpoint,indices=list(solutions),variables=np.concatenate([v.cpu().numpy() for v in solutions.values()]),initial_costs=[initial_costs[i] for i in solutions],iterations=[iterations[i] for i in solutions])
   write(out/'status.json',{'state':'solving','type':name,'completed_types':len(entries),'solved':len(solutions),'knots':n});
  q=torch.cat([solutions[i] for i in range(n)]);rows=torch.arange(n,device='cuda:0');cost,parts=adapter.validate(q,rows)
  with torch.no_grad():v,j,_=m(q,full=True)
  tip=torch.tensor([8,12,16,20],device=q.device);error=torch.diff(j[:,tip],dim=1).norm(dim=-1)-torch.diff(target[:,tip],dim=1).norm(dim=-1)
  parts.update(base_patch_joint_accepted=parts['patch_joint_accepted'].clone(),finger_gap_error_m=error,finger_gap_rms_m=error.square().mean(-1).sqrt(),grasp_passed=parts['patch_joint_accepted']&(error.square().mean(-1).sqrt()<=.006))
  angles=m.angles(q).reshape(n,15,3).cpu().numpy();variables=q.cpu().numpy();assert np.max(abs(variables[:,:6]-wrist.cpu().numpy()))<1e-7
  outputs={**z,'wuji_qpos_original':z['wuji_qpos'],'wuji_qpos':wq,'mano_variables':variables,'mano_angles':angles,'mano_vertices':v.cpu().numpy(),'mano_joints':j.cpu().numpy(),'object_pose_world':object_pose,'object_anchors':anchors,'object_anchor_normalized':normalized,'object_anchor_faces':anchor_faces.cpu().numpy(),'object_anchor_barycentric':anchor_bary.cpu().numpy(),'final_cost':cost.cpu().numpy(),'initial_cost':np.array([initial_costs.get(i,np.nan) for i in range(n)]),'iterations':np.array([iterations.get(i,0) for i in range(n)])}
  outputs.update({k:x.cpu().numpy() for k,x in parts.items() if isinstance(x,torch.Tensor)});np.savez_compressed(folder/'poses.npz',**outputs)
  oldgap=np.linalg.norm(np.diff(z['mano_joints'][:,[8,12,16,20]],axis=1),axis=-1);goal=torch.diff(target[:,tip],dim=1).norm(dim=-1).cpu().numpy();newgap=np.linalg.norm(np.diff(outputs['mano_joints'][:,[8,12,16,20]],axis=1),axis=-1)
  metrics={'wrist_max_error_m_rad':float(np.max(abs(variables[:,:6]-variables[canonical,:6]))),'coverage_median':float(np.median(outputs['patch_coverage'])),'contact_passed':int(outputs['base_patch_joint_accepted'].sum()),'pose_and_contact_passed':int(outputs['grasp_passed'].sum()),'tip_gap_rms_before_mm':float(np.sqrt(np.mean((oldgap-goal)**2))*1000),'tip_gap_rms_after_mm':float(np.sqrt(np.mean((newgap-goal)**2))*1000),'seconds':sync()-start,'max_adjacent_joint_step_rad':float(np.max(abs(np.diff(variables[:,6:],axis=0))))}
  entry=dict(entry,**metrics,solver=manifest['solver'],cost=manifest['cost']);write(folder/'complete.json',{'entry':entry,'sha256':sha(folder/'poses.npz')});entries.append(entry)
  browser={'name':name,'kind':entry['kind'],'axis':entry['axis'],'scales':z['size_scale'].tolist(),'source_scales':[r['size_scale'] for r in entry['source_rows']],'radius_m':z['radius_m'].tolist(),'height_m':z['height_m'].tolist(),'mano_translation':variables[:,:3].tolist(),'mano_root_quat':Rotation.from_rotvec(variables[:,3:6]).as_quat().tolist(),'mano_joint_quat':Rotation.from_rotvec(angles.reshape(-1,3)).as_quat().reshape(n,15,4).tolist(),'wuji_qpos':wq.tolist(),'wuji_root_quat':Rotation.from_euler('XYZ',wq[:,3:6]).as_quat().tolist(),'final_cost':[float(x) if np.isfinite(x) else None for x in cost.cpu().tolist()],'initial_cost':[float(x) if np.isfinite(x) else None for x in outputs['initial_cost'].tolist()],'valid':outputs['valid'].tolist(),'coverage':outputs['patch_coverage'].tolist(),'penetration_mm':(outputs['penetration_depth_m']*1000).tolist(),'closure_certified':outputs['patch_joint_accepted'].tolist(),'contact_passed':outputs['base_patch_joint_accepted'].tolist(),'grasp_passed':outputs['grasp_passed'].tolist(),'finger_gap_rms_mm':(outputs['finger_gap_rms_m']*1000).tolist(),'object_translation':origin.tolist(),'object_quat':object_pose[:,[4,5,6,3]].tolist(),'patch_fitted':outputs['patch_fitted'].tolist(),'object_anchors':anchors.tolist(),'fixed_wrist':True,'current_patch_cem':True}
  write(out/'web/data'/f'{name}.json',browser);write(out/'web/index.json',{'entries':entries,'complete':False,'total_knots':sum(e['knots'] for e in entries),'fixed_wrist':True,'solver':manifest['solver'],'cost':manifest['cost']});print(json.dumps(entry),flush=True);del adapter,surface,problem;gc.collect();torch.cuda.empty_cache()
 write(out/'web/index.json',{'entries':entries,'complete':not args.only,'total_knots':sum(e['knots'] for e in entries),'fixed_wrist':True,'solver':manifest['solver'],'cost':manifest['cost']});write(out/'status.json',{'state':'complete' if not args.only else 'pilot_complete','types':len(entries),'knots':sum(e['knots'] for e in entries)})
if __name__=='__main__':main()
