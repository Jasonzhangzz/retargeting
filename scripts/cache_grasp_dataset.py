"""Freeze final and interpolated poses; share a single scene and primitive meshes."""
import argparse,gzip,hashlib,json,sys
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import trimesh
from retarget.common.results import write_json
from retarget.viewer.snapshot import canonical


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(a):return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()
def compact(v):return json.dumps(v,separators=(',',':'),ensure_ascii=False).encode()


def export(source,destination):
 source=Path(source).resolve();destination=Path(destination).resolve()
 if destination.exists():raise FileExistsError(destination)
 destination.mkdir(parents=True);original=json.loads((source/'manifest.json').read_text())['cases'];dense=json.loads((source/'dense_pose_index.json').read_text())
 templates=[];template_ids={};entries=[];fields=defaultdict(list);metrics=[];preservation=[];assets={};model_text=None;source_bytes=0;objects={};geometry={};points=[]
 for i,e in enumerate(original):
  old=(source/Path(e['trajectory'])).parent
  with np.load(old/'grasp.npz',allow_pickle=False) as poses:
   for key in poses.files:fields[key].append(poses[key].copy())
  d=json.loads((old/'projection_force_closure.json').read_text());points.append(np.asarray(d['projected_points'],dtype=np.float64))
  summary=json.loads((source/Path(e['summary'])).read_text())
  scalars={k:v for k,v in summary['geometry_diagnostics'].items() if not isinstance(v,(list,dict))}
  for key in ('settings','articulation_policy','grasp_posture_bounds','axial_centering','middle_centering','index_contact','adjacent_motion','adjacent_motion_limits'):
   if key in d:scalars[key]=d[key]
  metric={k:summary[k] for k in ('method','objective','template','hand','solver_completed','task_success','native_evaluated','success_criterion','optimization_seconds','final_cost') if k in summary}
  metric.update(primitive=e['primitive'],geometry_diagnostics=scalars);metrics.append(metric)
  template=json.loads((old/'template_contacts.json').read_text());key=hashlib.sha256(canonical(template).encode()).hexdigest()
  if key not in template_ids:template_ids[key]=len(templates);templates.append(template)
  tree=ET.parse(source/Path(e['model']));object_paths=[]
  for element in tree.iter():
   file=element.get('file')
   if file is None:continue
   path=((source/Path(e['model'])).parent/file).resolve()
   if path.is_relative_to(old):object_paths.append(path);element.set('file','__OBJECT__')
   else:
    if not path.is_relative_to(ROOT):raise ValueError(f'External model dependency: {path}')
    name=str(path.relative_to(ROOT));element.set('file',name);assets[name]=sha(path)
  if len(object_paths)!=1:raise ValueError('Expected one primitive mesh')
  normalized=ET.tostring(tree.getroot(),encoding='unicode')
  if model_text is None:model_text=normalized
  elif normalized!=model_text:raise ValueError('Scene templates differ')
  p=e['primitive'];kind=p['kind'];candidate=(p['radius_m'],object_paths[0],p)
  if kind not in objects or candidate[0]>objects[kind][0]:objects[kind]=candidate
  primitive={k:v for k,v in p.items() if k not in ('sources','dimensions_status')}
  entries.append(dict(name=e['name'],task='grasp',method='mppi',hand='wuji2',template=e['template'],primitive=primitive,
   phase='refinement',task_success=e['task_success'],success_criterion=e['success_criterion'],cached_case=i,
   template_id=template_ids[key],model='scene.xml',trajectory='poses.npz',summary='metrics.json.gz'))
  preservation.append(dict(case=e['name'],qpos_sha256=digest(fields['qpos'][-1]),grasp_qpos_sha256=digest(fields['grasp_qpos'][-1]),
   projected_points_sha256=digest(points[-1]),source_diagnostic_sha256=sha(old/'projection_force_closure.json'),template_sha256=key))
  for name in ('scene.xml','input_object.obj','grasp.npz','trajectory.npz','projection_force_closure.json','patch_contact_coverage.json','template_contacts.json'):
   if (old/name).is_file():source_bytes+=(old/name).stat().st_size
  source_bytes+=(source/Path(e['summary'])).stat().st_size
 for kind,(_,path,p) in objects.items():
  mesh=trimesh.load_mesh(path,process=False)
  geometry[kind+'_vertices']=np.asarray(mesh.vertices)/np.array([p['radius_m'],p['radius_m'],p['height_m']])
  geometry[kind+'_faces']=np.asarray(mesh.faces,dtype=np.int32)
 np.savez_compressed(destination/'geometry.npz',**geometry)
 maximum=max(len(p) for p in points);projected=np.zeros((len(points),maximum,3))
 for i,p in enumerate(points):projected[i,:len(p)]=p
 np.savez_compressed(destination/'poses.npz',**{k:np.stack(v) for k,v in fields.items()},projected_points=projected,contact_counts=np.array([len(p) for p in points]),
  case_names=np.array([e['name'] for e in entries]),radius_m=np.array([e['primitive']['radius_m'] for e in entries]),
  height_m=np.array([e['primitive']['height_m'] for e in entries]),size_scale=np.array([e['primitive']['size_scale'] for e in entries]),
  dimensions_m=np.array([e['primitive']['dimensions_m'] for e in entries]))
 (destination/'scene.xml').write_text(model_text)
 for name,value in (('templates.json.gz',templates),('metrics.json.gz',metrics),('preservation.json.gz',preservation)):
  (destination/name).write_bytes(gzip.compress(compact(value),compresslevel=9,mtime=0))
 write_json(destination/'manifest.json',dict(version=2,storage='compact_grasp_pose_snapshot_v2',task='grasp',method='mppi',cases=entries))
 by_name={e['name']:e for e in entries};groups=[];(destination/'dense').mkdir();dense_preservation=[];npy_bytes=0
 for group in dense['groups']:
  path=Path(group['file']);path=path if path.is_absolute() else source/path
  with np.load(path,allow_pickle=False) as values:arrays={k:values[k].copy() for k in values.files}
  original_projected=arrays.pop('projected_points',None)
  if original_projected is not None:
   arrays['source_projected_points']=original_projected[arrays['source_indices']].copy()
   flat=arrays['source_projected_points'].reshape(len(arrays['source_scales']),-1)
   restored=np.stack([np.interp(arrays['scales'],arrays['source_scales'],flat[:,j]) for j in range(flat.shape[1])],axis=1).reshape(original_projected.shape)
   if not np.array_equal(restored,original_projected):raise ValueError('Projection reconstruction is not lossless')
  target=destination/'dense'/path.name;np.savez_compressed(target,**arrays)
  dense_preservation.append(dict(file=str(target.relative_to(destination)),arrays={k:digest(v) for k,v in arrays.items()},original_projected_points_sha256=digest(original_projected) if original_projected is not None else None))
  npy_bytes+=sum(v.nbytes+128 for v in arrays.values());source_bytes+=path.stat().st_size
  groups.append({**{k:group[k] for k in ('template','kind','source_cases','dense_poses','min_scale','max_scale')},'file':str(target.relative_to(destination)),
   'source_case_indices':[by_name[e['name']]['cached_case'] for e in group['source_rows']]})
 dense={k:v for k,v in dense.items() if k not in ('groups','source_manifest','source_manifest_sha256')}
 dense.update(groups=groups,source_manifest='manifest.json',source_manifest_sha256=sha(destination/'manifest.json'))
 write_json(destination/'dense_pose_index.json',dense)
 write_json(destination/'summary.json',dict(cases=len(entries),grids=len(groups),dense_poses=dense['dense_poses'],projected_closed=sum(e['task_success'] for e in entries),native_dynamic_evaluation=False))
 write_json(destination/'provenance.json',dict(source_manifest_sha256=sha(source/'manifest.json'),repository_assets_sha256=assets,
  source_final_files_bytes=source_bytes,unique_scene_templates=1,shared_primitive_kinds=list(objects),float_dtype='float64',
  source_poses_preserved_exactly=True,dense_projection_storage='losslessly reconstructed from source knots',uncompressed_dense_npy_estimated_bytes=npy_bytes))
 write_json(destination/'dense_preservation.json',dense_preservation)
 write_json(destination/'status.json',dict(state='completed',planned=len(entries),finished=len(entries)))
 write_json(destination/'checksums.json',{str(p.relative_to(destination)):sha(p) for p in destination.rglob('*') if p.is_file()})
 total=sum(p.stat().st_size for p in destination.rglob('*') if p.is_file())
 print(json.dumps(dict(cases=len(entries),grids=len(groups),dense_poses=dense['dense_poses'],source_mb=source_bytes/1e6,snapshot_mb=total/1e6,reduction_percent=100*(1-total/source_bytes)),indent=2))


if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('source',type=Path);parser.add_argument('destination',type=Path);args=parser.parse_args();export(args.source,args.destination)
