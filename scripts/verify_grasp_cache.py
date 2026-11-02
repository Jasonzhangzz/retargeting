"""Verify a compact snapshot; optionally compare every value with its raw source."""
import argparse,gzip,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
from scipy.spatial import cKDTree
from retarget.env.geometry import ObjectMeshSurface
from retarget.viewer.snapshot import get_snapshot,canonical
from retarget.viewer.dense_pose import DensePoseGrid


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(a):return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()


def verify(path,source=None,report_path=None):
 path=Path(path).resolve();snapshot=get_snapshot(path);entries=snapshot.manifest['cases']
 checksums=json.loads((path/'checksums.json').read_text())
 for name,expected in checksums.items():
  if sha(path/name)!=expected:raise ValueError(f'Checksum mismatch: {name}')
 preservation=json.loads(gzip.decompress((path/'preservation.json.gz').read_bytes()))
 provenance=json.loads((path/'provenance.json').read_text())
 for name,expected in provenance['repository_assets_sha256'].items():
  if sha(ROOT/name)!=expected:raise ValueError(f'Shared asset changed: {name}')
 originals={e['name']:e for e in json.loads((Path(source)/'manifest.json').read_text())['cases']} if source else {}
 rows=[];maximum_geometry_error=0.
 for i,e in enumerate(entries):
  p=preservation[i];q=snapshot.poses['qpos'][i];native=snapshot.poses['grasp_qpos'][i];count=int(snapshot.poses['contact_counts'][i])
  assert digest(q)==p['qpos_sha256'] and digest(native)==p['grasp_qpos_sha256']
  assert digest(snapshot.poses['projected_points'][i,:count])==p['projected_points_sha256']
  assert hashlib.sha256(canonical(snapshot.template(i)).encode()).hexdigest()==p['template_sha256']
  model=snapshot.model(i);data=mujoco.MjData(model);data.qpos[:]=q;mujoco.mj_kinematics(model,data);mujoco.mj_collision(model,data)
  t=snapshot.template(i);ids=[model.body(n).id for n in t['body_names']]
  hand=data.xpos[ids]+np.einsum('nij,nj->ni',data.xmat[ids].reshape(-1,3,3),t['local_points'])
  d=snapshot.diagnostics(i);object_id=model.body('grasp_object').id;depth=np.zeros(2)
  for c in data.contact:
   pair=model.geom_bodyid[c.geom]
   if np.all(pair==object_id):continue
   k=0 if object_id in pair else 1;depth[k]=max(depth[k],-c.dist)
  error=np.abs(depth-[d['max_hand_object_penetration_m'],d['max_hand_hand_penetration_m']]).max()
  assert error<2.e-5,(e['name'],'collision',error)
  item=dict(case=e['name'],collision_error_m=float(error),closure=bool(d['projected_closure']),coverage=d['coverage'])
  number=int(e['template'].partition('_')[0])
  if number==18:
   assert e['primitive']['radius_m']==.08 and e['primitive']['size_axis']=='thickness'
   assert abs(e['primitive']['height_m']-.008*e['primitive']['size_scale'])<1.e-12
  if number==26:
   assert abs(np.degrees(q[15]))<=5.0001 and d['index_contact']['fitted']
   assert d['middle_centering']['min_gap_m']>=.004-1.e-6
  if source:
   old=originals[e['name']];case=(Path(source)/old['trajectory']).parent
   with np.load(case/'grasp.npz',allow_pickle=False) as z:
    assert np.array_equal(q,z['qpos']) and np.array_equal(native,z['grasp_qpos'])
   original=json.loads((case/'projection_force_closure.json').read_text());assert sha(case/'projection_force_closure.json')==p['source_diagnostic_sha256']
   assert np.abs(hand-np.asarray(original['hand_points'])).max()<1.e-6
   old_model=mujoco.MjModel.from_xml_path(str(Path(source)/old['model']))
   a=ObjectMeshSurface.from_model(old_model,old_model.body('grasp_object').id).triangles.reshape(-1,3)
   b=ObjectMeshSurface.from_model(model,object_id).triangles.reshape(-1,3)
   geometric=max(cKDTree(a).query(b)[0].max(),cKDTree(b).query(a)[0].max());assert geometric<1.e-6,(e['name'],geometric)
   maximum_geometry_error=max(maximum_geometry_error,float(geometric));item['primitive_reconstruction_error_m']=float(geometric)
   m=original['closure_model'];points=np.array(original['projected_points']);force=np.array(original['response_contact_forces_n']);spin=np.array(original['response_intrinsic_torques_nm'])
   wrench=np.concatenate((force.sum(1),(np.cross((points-np.array(m['center']))[None],force)+spin).sum(1)/m['radius_m']),axis=-1)
   loads=np.concatenate((np.zeros((1,6)),np.eye(6),-np.eye(6)))*m['test_force_n'];residual=(wrench+loads)/m['test_force_n']
   closure_error=float(np.linalg.norm(residual[1:],axis=-1).max());certified=closure_error<=m['certificate_tolerance'] and closure_error<1/np.sqrt(6)
   assert bool(certified)==original['projected_closure'] and np.abs(residual-np.asarray(original['scenario_residuals'])).max()<1.e-4
   item['force_certificate_error']=closure_error
  rows.append(item)
  if (i+1)%100==0:print('VERIFIED',i+1,'/',len(entries),flush=True)
 index=json.loads((path/'dense_pose_index.json').read_text());dense_preservation=json.loads((path/'dense_preservation.json').read_text());nposes=0;source_points=0
 for g,record in zip(index['groups'],dense_preservation):
  grid=DensePoseGrid(path/g['file']);values=grid.values
  assert np.isfinite(values['qpos']).all() and np.all(np.diff(grid.scales)>0)
  for k,expected in record['arrays'].items():assert digest(values[k])==expected
  for scale,idx in zip(grid.source_scales,g['source_case_indices']):
   sample=grid.sample(scale);assert sample['solved'] and np.array_equal(sample['qpos'],snapshot.poses['qpos'][idx]);source_points+=1
  if 'source_projected_points' in values:
   a=values['source_projected_points'];flat=a.reshape(len(grid.source_scales),-1)
   restored=np.stack([np.interp(grid.scales,grid.source_scales,flat[:,j]) for j in range(flat.shape[1])],axis=1).reshape(len(grid.scales),*a.shape[1:])
   assert digest(restored)==record['original_projected_points_sha256']
  nposes+=len(grid.scales)
 assert source_points==len(entries)
 report=dict(passed=True,cases=len(entries),grids=len(index['groups']),dense_poses=nposes,source_poses_preserved_exactly=True,
  dense_projections_reconstructed_exactly=True,source_comparison=bool(source),maximum_primitive_reconstruction_error_m=maximum_geometry_error if source else None,
  projected_closed=sum(r['closure'] for r in rows),coverage_80_percent=sum(r['coverage']>=.8-1.e-7 for r in rows),rows=rows)
 if report_path is not None:Path(report_path).write_text(json.dumps(report,indent=2))
 print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2));return report


if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('snapshot',type=Path);parser.add_argument('--source',type=Path);parser.add_argument('--report',type=Path);a=parser.parse_args();verify(a.snapshot,a.source,a.report)
