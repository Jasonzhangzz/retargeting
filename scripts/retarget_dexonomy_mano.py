#!/usr/bin/env python3
"""Resumable multi-GPU Shadow -> MANO using the production LM/CEM preset."""
from __future__ import annotations
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import argparse,collections,concurrent.futures,datetime,gzip,hashlib,json,multiprocessing,sys,time,traceback,signal,fcntl
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from dataclasses import asdict

def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    temp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n');temp.replace(path)
def gz(path,value):Path(path).write_bytes(gzip.compress(json.dumps(value,separators=(',',':'),allow_nan=False).encode(),mtime=0))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def source_groups(source):
    source=Path(source);journal=source/'shards.jsonl'
    if journal.exists():
        entries={}
        for line in journal.read_text().splitlines():
            try:v=json.loads(line)
            except ValueError:continue
            entries[(v['type'],v['part'])]=v
        return list(entries.values())
    return [dict(type=p.parents[2].name,part=int(p.parent.name),path=str(p.parent),
                 samples=len(read(p)['cases'])) for p in sorted(source.glob('*/shards/*/manifest.json'))]

class GeometryLoader:
    def __init__(self,dataset):self.dataset=dataset;self.cache=collections.OrderedDict()
    def get(self,entry):
        import trimesh
        from retarget.env.grasp.dexonomy_grasp_static import base_scene
        scene=base_scene(str(self.dataset),entry['scene_path'])
        cfg=scene['scene'][scene['task']['obj_name']]
        scale=np.asarray(cfg['scale'],float)
        key=(cfg['file_path'],tuple(scale))
        if key not in self.cache:
            mesh=trimesh.load_mesh(cfg['file_path'],process=False)
            if isinstance(mesh,trimesh.Scene):mesh=mesh.to_mesh()
            if not isinstance(mesh,trimesh.Trimesh):raise ValueError('Object asset is not a triangle mesh')
            vertices=np.asarray(mesh.vertices,float)*scale
            faces=np.asarray(mesh.faces,np.int32).copy()
            good=np.linalg.norm(np.cross(vertices[faces[:,1]]-vertices[faces[:,0]],vertices[faces[:,2]]-vertices[faces[:,0]]),axis=1)>1e-14
            faces=faces[good]
            checked=trimesh.Trimesh(vertices=vertices,faces=faces,process=False)
            flipped=bool(checked.is_watertight and checked.volume<0)
            if flipped:faces=faces[:,[0,2,1]]
            self.cache[key]=dict(vertices=vertices.astype(np.float32),faces=faces,watertight=bool(checked.is_watertight),
                source_mesh=cfg['file_path'],base_scale=scale.tolist(),winding_flipped=flipped,sha256=sha(cfg['file_path']))
            if len(self.cache)>128:self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return key,self.cache[key]

def interleaved_order(geometry_ids):
    """Expose each object's first examples before its later warm-start candidates."""
    groups=collections.OrderedDict()
    for i,key in enumerate(geometry_ids):groups.setdefault(key,collections.deque()).append(i)
    order=[]
    while groups:
        for key in list(groups):
            order.append(groups[key].popleft())
            if not groups[key]:del groups[key]
    return np.array(order,dtype=np.int64)

def export_shard(group,output,decoder,skeleton,loader,settings,fingerprint,limit=0,mappings=None):
    import torch
    from scipy.spatial.transform import Rotation
    from retarget.env.grasp.mano_solver import SurfaceBatch,optimize,vertex_normals,FixedContacts,MeshCache,WarmStartBank
    source=Path(group['path']);original=read(source/'manifest.json')
    source_hash=sha(source/'poses.npz')
    out=Path(output)/group['type']/'shards'/f"{group['part']:07d}"
    done=out/'verification.json'
    if done.exists():
        v=read(done)
        if v.get('success') and v['source_poses_sha256']==source_hash and v['configuration_fingerprint']==fingerprint:return v
        raise ValueError('Incompatible or incomplete existing output: '+str(out))
    out.mkdir(parents=True,exist_ok=True)
    entries=original['cases'][:limit or None];n=len(entries)
    active=Path(output)/'active'/('cuda_'+str(decoder.device.index)+'.json')
    progress=dict(source=group['type']+'/'+str(group['part']),total_rows=n,solved_rows=0,stage='loading_geometry',last_updated_utc=stamp())
    write(active,progress)
    with np.load(source/'poses.npz',allow_pickle=False) as a:
        shadow=a['grasp_qpos'][:n].copy();object_pose=a['object_pose_world'][:n].copy()
        object_scale=a['object_scale'][:n].copy()
        scene_scale=a['scene_scale'][:n].copy() if 'scene_scale' in a.files else np.array([e['scene_scale'] for e in entries])
    if object_pose.ndim==3:object_pose=object_pose[:,0]
    if mappings is None or group['type'] not in mappings:raise ValueError('No reviewed mapping for '+group['type'])
    method='cem'
    mapping=mappings[group['type']]
    fixed=FixedContacts(decoder,mapping)
    source_points,source_normals=skeleton.contacts(shadow,mapping)
    target=skeleton(shadow);geometries=[];geo_keys={};geo_indices=[]
    for e in entries:
        key,geo=loader.get(e)
        if key not in geo_keys:geo_keys[key]=len(geometries);geometries.append(geo)
        geo_indices.append(geo_keys[key])
    if not hasattr(decoder,'mesh_cache'):decoder.mesh_cache=MeshCache(decoder.device)
    if not hasattr(decoder,'warm_bank'):decoder.warm_bank=WarmStartBank(decoder.device,settings)
    mesh_cache=decoder.mesh_cache;warm_bank=decoder.warm_bank
    mesh_start=(mesh_cache.builds,mesh_cache.hits)
    order=interleaved_order(geo_indices) if settings.warm_start else np.arange(n)
    results=[];batch_size=settings.batch_size;start=0;seeds=[];t0=time.monotonic()
    while start<n:
        stop=min(start+batch_size,n);ids=order[start:stop];subset=[geo_indices[int(i)] for i in ids]
        progress.update(solved_rows=start,stage='optimizing',batch_size=stop-start,last_updated_utc=stamp());write(active,progress)
        # Keep the per-shard mesh list shared; rows select their own object.
        surface=SurfaceBatch(geometries,subset,object_pose[ids],scene_scale[ids],decoder.device,mesh_cache=mesh_cache)
        seed=(settings.seed+int(hashlib.sha256((group['type']+'/'+str(group['part'])+'/'+str(start)).encode()).hexdigest()[:8],16))%(2**31)
        keys=[(group['type'],MeshCache.key(geometries[j])) for j in subset]
        warm,available,feature=warm_bank.lookup(keys,target[ids],surface) if settings.warm_start else (None,None,None)
        try:
            result=optimize(decoder,target[ids],surface,settings,seed,fixed,source_points[ids],source_normals[ids],
                warm_start=warm,warm_available=available,
                on_solved=(lambda best:warm_bank.add(keys,best,feature,surface)) if settings.warm_start else None)
        except torch.cuda.OutOfMemoryError:
            del surface;torch.cuda.empty_cache()
            if batch_size==1:raise
            batch_size=max(1,batch_size//2);continue
        results.append(result);seeds.append(dict(start=start,stop=stop,seed=seed,batch_size=batch_size,source_cached_rows=ids.tolist(),
            warm_start_used=int(result['warm_start_used'].sum()),ik_steps_mean=float(result['ik_iterations'].mean()),
            cem_steps_mean=float(result['cem_iterations'].mean()),candidate_evaluations=int(result.get('candidate_evaluations',np.zeros(1)).sum()),stop_reason_counts={str(int(k)):int(v) for k,v in zip(*np.unique(result['stop_reason'],return_counts=True))}))
        del surface;start=stop
        progress.update(solved_rows=stop,stage='solved_batch',last_updated_utc=stamp());write(active,progress)
    result={k:np.concatenate([v[k] for v in results],axis=1 if k=='history' else 0) for k in results[0]}
    inverse=np.argsort(order)
    result={k:(v[:,inverse] if k=='history' else v[inverse]) for k,v in result.items()}
    variables=result['variables'];angles=result['angles']
    quaternions=Rotation.from_rotvec(variables[:,3:6]).as_quat(scalar_first=True)
    qpos=np.concatenate((variables[:,:6],angles),-1).astype(np.float64)
    grasp=np.concatenate((variables[:,:3],quaternions,angles),-1).astype(np.float64)
    hand_pose=np.concatenate((variables[:,3:6],angles,variables[:,:3]),-1)
    arrays=collections.defaultdict(list);offsets=[0];templates=[];metrics=[];cases=[];counts=[];projections=[]
    faces=decoder.base.faces;weights=decoder.base.weights.cpu().numpy();joint_names=['wrist','index_mcp','index_pip','index_dip','middle_mcp','middle_pip','middle_dip','pinky_mcp','pinky_pip','pinky_dip','ring_mcp','ring_pip','ring_dip','thumb_cmc','thumb_mcp','thumb_ip']
    incident=np.zeros(778,int);local_bary=np.zeros((778,3))
    for face_id,face in enumerate(faces):
        for j,vertex in enumerate(face):incident[vertex]=face_id;local_bary[vertex]=np.eye(3)[j]
    quality_count=0;quality_flags=[]
    for i,e in enumerate(entries):
        distance=result['distance'][i];vertices=result['vertices'][i]
        selected=np.flatnonzero(np.abs(distance)<=settings.contact_tolerance_m)
        nn=vertex_normals(vertices,faces);object_r=Rotation.from_quat(object_pose[i,3:],scalar_first=True).as_matrix()
        root_r=Rotation.from_rotvec(variables[i,3:6]).as_matrix()
        hand=vertices[selected];projected=result['projected'][i,selected];normal=result['object_normals'][i,selected]
        joint_ids=weights[selected].argmax(-1);joint_positions=result['joints'][i] # ordered 21 points; use root-local skin anchors, not rigid phalanges
        local=(hand-variables[i,:3])@root_r
        obj_local=(projected-object_pose[i,:3])@object_r
        values=dict(hand_vertex_ids=selected.astype(np.int32),hand_face_ids=incident[selected].astype(np.int32),
            hand_barycentric=local_bary[selected],object_face_ids=result['object_face'][i,selected].astype(np.int32),
            object_barycentric=result['object_bary'][i,selected],hand_points_world=hand,object_points_world=projected,
            points_world=(hand+projected)*.5,object_normals_world=normal,hand_normals_world=nn[selected],
            hand_local_points=local,hand_local_normals=nn[selected]@root_r,object_local_points=obj_local,
            distances_m=distance[selected])
        for k,v in values.items():arrays[k].append(v)
        offsets.append(offsets[-1]+len(selected));counts.append(len(selected));projections.append(projected)
        maximum_penetration=float(np.maximum(-distance,0).max())
        fit_rmse=float(np.sqrt(np.mean(np.sum((result['joints'][i]-target[i])**2,-1))))
        quality=bool(maximum_penetration<=settings.penetration_tolerance_m and len(selected)>=2 and geometries[geo_indices[i]]['watertight'])
        mapped_hand=result['mapped_hand_points'][i]
        mapped_normals=result['mapped_hand_normals'][i]
        mapped_local=(mapped_hand-variables[i,:3])@root_r
        mapped_error=np.abs(result['mapped_distance'][i])
        normal_alignment=np.sum(mapped_normals*result['mapped_object_normals'][i],axis=-1)
        mapped_coverage=float(np.mean((mapped_error<=settings.contact_tolerance_m)&(normal_alignment<=-np.cos(np.pi/6))))
        mapped_distance_coverage=float(np.mean(mapped_error<=settings.contact_tolerance_m))
        quality=bool(quality and mapped_coverage>=.8)
        if not quality and result['stop_reason'][i]==1:result['stop_reason'][i]=3
        quality_count+=int(quality);quality_flags.append(quality)
        templates.append(dict(body_names=[rec['shadow_body'] for rec in mapping],local_points=mapped_local.tolist(),
            local_normals=(mapped_normals@root_r).tolist(),finger_qpos=angles[i].tolist(),required_cbody=[],
            mano_face_ids=[rec['mano_face_id'] for rec in mapping],mano_barycentric=[rec['mano_barycentric'] for rec in mapping],
            anchor_frame='MANO root frame at this exact pose; fixed deformable triangle/barycentric slots authoritative',
            provenance=dict(source='reviewed original Shadow template annotation -> fixed MANO triangle mapping',
                mapping_review_flags=[rec['id'] for rec in mapping if rec.get('needs_review')],
                contact_tolerance_m=settings.contact_tolerance_m)))
        metrics.append(dict(hand='mano',method=method,solver_completed=True,task_success=None,
            optimization=dict(ik_method='lm',search_method='cem',ik_iterations=int(result['ik_iterations'][i]),cem_iterations=int(result['cem_iterations'][i]),
                stop_reason={0:'budget',1:'geometry_accepted',2:'plateau',3:'geometry_recheck_failed'}[int(result['stop_reason'][i])],warm_start_used=bool(result['warm_start_used'][i])),
            success_criterion='not evaluated: no force-closure or dynamic certificate',force_closure_evaluated=False,dynamic_evaluated=False,
            native_evaluated=False,geometry_accepted=quality,initial_cost=float(result['initial_cost'][i]),final_cost=float(result['final_cost'][i]),
            geometry_diagnostics=dict(contact_count=len(selected),mapped_contact_count=len(mapping),mapped_coverage=mapped_coverage,mapped_distance_coverage=mapped_distance_coverage,
                max_mapped_gap_m=float(mapped_error.max()),mapping_review_flags=[rec['id'] for rec in mapping if rec.get('needs_review')],
                max_hand_object_penetration_m=maximum_penetration,
                joint_rmse_to_shadow_m=fit_rmse,object_watertight=geometries[geo_indices[i]]['watertight'],
                self_collision_evaluated=False,contact_query='all 778 MANO vertices against object triangle mesh',
                contact_tolerance_m=settings.contact_tolerance_m,source_grasp_preserved=True)))
        cases.append(dict(name=e['name'],task='grasp',hand='mano',method=method,template=group['type'],grasp_type=group['type'],
            phase='retargeting',task_success=None,success_criterion='geometry_only_no_force_closure_certificate',
            cached_case=i,template_id=i,model='mano_model.json',trajectory='poses.npz',summary='metrics.json.gz',
            contact_data='contacts.npz',object_geometry_id=geo_indices[i],object=e['object'],
            scene_path=e['scene_path'],scene_scale=e['scene_scale'],source_manifest=str(source/'manifest.json'),source_cached_case=i))
    maximum=max(counts,default=0);padded=np.zeros((n,maximum,3))
    for i,p in enumerate(projections):padded[i,:len(p)]=p
    np.savez_compressed(out/'poses.npz',qpos=qpos,grasp_qpos=grasp,mano_pose_51=hand_pose,
        mano_betas=np.zeros((n,10)),mano_pca=result['variables'][:,6:],mano_vertices=result['vertices'],
        mano_joints=result['joints'],projected_points=result['mapped_object_points'],
        contact_counts=np.full(n,len(mapping),np.int32),geometric_contact_counts=np.array(counts),case_names=np.array([e['name'] for e in entries]),
        object_pose_world=object_pose,object_scale=object_scale,scene_scale=scene_scale,object_geometry_ids=np.array(geo_indices),
        source_shadow_grasp_qpos=shadow,source_shadow_joints_world=target,
        mapped_hand_points=result['mapped_hand_points'],mapped_hand_normals=result['mapped_hand_normals'],
        mapped_object_points=result['mapped_object_points'],mapped_object_normals=result['mapped_object_normals'],
        mapped_distances_m=result['mapped_distance'],mapped_mano_face_ids=np.array([rec['mano_face_id'] for rec in mapping]),
        mapped_mano_barycentric=np.array([rec['mano_barycentric'] for rec in mapping]),
        source_annotation_points_world=result['source_annotation_points'],
        source_annotation_normals_world=result['source_annotation_normals'],
        source_annotation_object_projection=result['source_annotation_object_projection'],
        initial_cost=result['initial_cost'],final_cost=result['final_cost'],cem_best_cost_history=result['history'].T,
        ik_iterations=result['ik_iterations'],cem_iterations=result['cem_iterations'],stop_reason=result['stop_reason'],
        warm_start_used=result['warm_start_used'],warm_start_variables=result['warm_start_variables'],
        candidate_evaluations=result.get('candidate_evaluations',result['cem_iterations']*(settings.population+1)))
    np.savez_compressed(out/'contacts.npz',offsets=np.array(offsets,np.int64),
        mapped_hand_points_world=result['mapped_hand_points'],mapped_object_points_world=result['mapped_object_points'],
        mapped_hand_normals_world=result['mapped_hand_normals'],mapped_object_normals_world=result['mapped_object_normals'],
        mapped_object_face_ids=result['mapped_object_face_ids'],mapped_object_barycentric=result['mapped_object_barycentric'],
        mapped_distances_m=result['mapped_distance'],**{k:np.concatenate(v,axis=0) for k,v in arrays.items()})
    geometry_arrays={}
    for i,g in enumerate(geometries):geometry_arrays['object_'+str(i)+'_vertices']=g['vertices'];geometry_arrays['object_'+str(i)+'_faces']=g['faces']
    geometry_arrays['mano_faces']=faces
    np.savez_compressed(out/'geometry.npz',**geometry_arrays)
    gz(out/'templates.json.gz',templates);gz(out/'metrics.json.gz',metrics)
    model_info=dict(model='MANO',side='right',model_path=str(decoder.base.path),model_sha256=sha(decoder.base.path),
        units='metres/radians',shape='fixed beta zero, unit scale',mano_pose_51='root axis-angle(3), absolute local axis-angles(45), translation(3)',
        qpos='translation(3), root axis-angle(3), absolute local axis-angles(45)',
        grasp_qpos='translation(3), root quaternion wxyz(4), absolute local axis-angles(45)',hands_mean_already_added=True,
        pca_components=settings.pca_components,joint_order_21='wrist; thumb/index/middle/ring/pinky, each MCP/PIP/DIP/tip (thumb CMC/MCP/IP/tip)')
    write(out/'mano_model.json',model_info)
    write(out/'manifest.json',dict(version=2,storage='compact_grasp_pose_snapshot_v2',hand='mano',method=method,task='grasp',cases=cases,
        model_representation='MANO deformable LBS; MANO-aware loader required',contact_representation='fixed mapped triangle slots plus separate vertex-sampled proximity; no contact forces'))
    write(out/'object_assets.json',[{k:v for k,v in g.items() if k not in ('vertices','faces')} for g in geometries])
    write(out/'provenance.json',dict(configuration_fingerprint=fingerprint,settings=asdict(settings),source_poses_sha256=source_hash,
        source_manifest_sha256=sha(source/'manifest.json'),model_sha256=sha(decoder.base.path),batches=seeds,created_utc=stamp(),
        source_stage='grasp',fixed_mapping=True,architecture_version='mano_lm_cem_v2',
        warm_start_policy='same grasp type and exact base geometry, closest object-frame Shadow skeleton, size ratio gated; actual warm candidate saved in poses.npz',
        stop_reason_codes={'0':'budget','1':'geometry_accepted','2':'plateau','3':'geometry_recheck_failed'},history_policy='GPU resident; stopped rows padded with their final cost',
        mesh_cache=dict(builds=mesh_cache.builds-mesh_start[0],hits=mesh_cache.hits-mesh_start[1]),mapping_review_flags=[rec['id'] for rec in mapping if rec.get('needs_review')],paired_mano_ground_truth_available=False,initialization='rigid palm SVD + batched anatomical bone IK',
        contact_force_available=False,force_closure_evaluated=False,dynamic_evaluated=False))
    checks={}
    with np.load(out/'poses.npz',allow_pickle=False) as a:
        checks['source_preserved']=bool(np.array_equal(a['source_shadow_grasp_qpos'],shadow))
        checks['finite']=bool(all(np.isfinite(a[k]).all() for k in ['qpos','grasp_qpos','mano_vertices','final_cost']))
        checks['cem_monotone']=bool(np.all(a['final_cost']<=a['initial_cost']+1e-5))
        checks['dimensions']=a['qpos'].shape==(n,51) and a['grasp_qpos'].shape==(n,52)
    checks['early_exit_geometry']=bool(np.all(np.array(quality_flags)[result['stop_reason']==1]))
    checks['contact_offsets']=len(offsets)==n+1 and offsets[-1]==sum(counts)
    triangles=result['vertices'][:,faces[np.array([rec['mano_face_id'] for rec in mapping])]]
    rebuilt=(triangles*np.array([rec['mano_barycentric'] for rec in mapping])[None,:,:,None]).sum(-2)
    checks['fixed_mapping_reconstruction']=bool(np.allclose(rebuilt,result['mapped_hand_points'],atol=2e-6))
    if not all(checks.values()):raise RuntimeError('Output verification failed: '+str(checks))
    write(out/'checksums.json',{p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in ('verification.json','checksums.json')})
    verified=dict(success=True,samples=n,geometric_contacts=sum(counts),geometry_accepted=quality_count,checks=checks,
        configuration_fingerprint=fingerprint,source_poses_sha256=source_hash,seconds=round(time.monotonic()-t0,3),path=str(out))
    write(done,verified)
    progress.update(stage='written',solved_rows=n,last_updated_utc=stamp());write(active,progress)
    return verified

def reused_shards(roots,source,model_sha,mapping_fingerprint):
    """Verify and flatten historical references without copying immutable data."""
    result={};visited=set()
    for root in roots:
        root=Path(root).resolve();config=read(root/'configuration.json')
        if config['model_sha256']!=model_sha or config['mapping_fingerprint']!=mapping_fingerprint or config['source']!=str(Path(source).resolve()):
            raise ValueError('Incompatible historical output: '+str(root))
        paths=list(root.glob('*/shards/*/verification.json'))
        references=root/'reused_shards.json'
        if references.exists():
            for item in read(references):
                done=Path(item['path'])/'verification.json'
                if sha(done)!=item['verification_sha256']:raise ValueError('Historical verification changed: '+str(done))
                paths.append(done)
        for done in sorted(paths):
            done=done.resolve()
            if done in visited:continue
            visited.add(done);v=read(done)
            if not v.get('success'):continue
            key=done.parents[2].name+'/'+str(int(done.parent.name))
            if key in result:raise ValueError('Duplicate historical shard '+key)
            source_path=Path(source)/done.parents[2].name/'shards'/done.parent.name/'poses.npz'
            if int(v['samples'])!=len(read(source_path.parent/'manifest.json')['cases']):
                raise ValueError('Historical shard is partial: '+key)
            if sha(source_path)!=v['source_poses_sha256']:raise ValueError('Historical source changed: '+key)
            if not all(v.get('checks',{}).values()):raise ValueError('Historical verification failed: '+key)
            for filename,checksum in read(done.parent/'checksums.json').items():
                if sha(done.parent/filename)!=checksum:raise ValueError('Historical artifact checksum mismatch: '+key+'/'+filename)
            result[key]=dict(key=key,verification_sha256=sha(done),**v)
            result[key]['path']=str(done.parent)
    return list(result.values())


def worker(number,device,options,fingerprint):
    import torch,warp as wp
    from retarget.env.grasp.mano_solver import Settings,ManoDecoder,ShadowSkeleton
    torch.set_num_threads(1);torch.cuda.set_device(device)
    settings=Settings(**options['settings'])
    free,total=torch.cuda.mem_get_info(device)
    budget=min(settings.memory_budget_gb*2**30,free*.8)
    if budget<2*2**30:raise RuntimeError('Insufficient free VRAM on '+device)
    torch.cuda.set_per_process_memory_fraction(budget/total,device)
    wp.init()
    model=ManoDecoder(options['model'],device,settings.pca_components)
    mappings=read(options['mapping'])['mappings']
    model.configure_contact_vertices(mappings)
    skeleton=ShadowSkeleton(ROOT/'data/dexonomy/shadow_assets/right.xml');loader=GeometryLoader(ROOT/'data/dexonomy')
    output=Path(options['output']);source=Path(options['source']);completed={};failed={};started=time.monotonic()
    reused={r['key']:r for r in options.get('reused',[]) if int(hashlib.sha256(r['key'].encode()).hexdigest()[:8],16)%options['workers']==number}
    profile=dict(device=device,pid=os.getpid(),memory_budget_bytes=int(budget),started_utc=stamp(),
        configured_batch_size=settings.batch_size,population=settings.population,iterations=settings.iterations)
    active=output/'active'/('cuda_'+str(torch.device(device).index)+'.json')
    write(active,dict(source='',stage='starting',solved_rows=0,total_rows=0))
    def save(state):
        write(output/f'worker_{number}.json',dict(state=state,**profile,active=read(active) if active.exists() else None,shards_completed=len(completed)+len(reused),samples=sum(x['samples'] for x in completed.values())+sum(x['samples'] for x in reused.values()),
            samples_new=sum(x['samples'] for x in completed.values()),samples_reused=sum(x['samples'] for x in reused.values()),
            geometric_contacts=sum(x['geometric_contacts'] for x in [*completed.values(),*reused.values()]),geometry_accepted=sum(x['geometry_accepted'] for x in [*completed.values(),*reused.values()]),
            failed_shards=len(failed),errors=list(failed.values()),elapsed_seconds=round(time.monotonic()-started,2),
            peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(device),last_updated_utc=stamp()))
    save('starting')
    while True:
        if (output/'STOP').exists():save('stopped');return
        groups=source_groups(source)
        if options.get('max_shards'):groups=groups[:options['max_shards']]
        for group in groups:
            if (output/'STOP').exists():save('stopped');return
            key=group['type']+'/'+str(group['part'])
            if int(hashlib.sha256(key.encode()).hexdigest()[:8],16)%options['workers']!=number or key in completed or key in reused or key in failed:continue
            try:
                value=export_shard(group,output,model,skeleton,loader,settings,fingerprint,options.get('max_rows',0),mappings)
                completed[key]=value
                print(json.dumps(dict(worker=number,device=device,source=key,samples=value['samples'],seconds=value['seconds']),ensure_ascii=False),flush=True)
            except Exception as error:
                failed[key]=dict(source=key,error=repr(error),traceback=traceback.format_exc(),samples=group['samples'])
                write(output/group['type']/'shards'/f"{group['part']:07d}"/'failure.json',failed[key])
                print(json.dumps(failed[key]),flush=True)
            save('running')
        producer=read(source/'status.json') if (source/'status.json').exists() else {'state':'completed'}
        if not options['watch'] or producer.get('state')=='completed':
            save('completed_with_errors' if failed else 'completed');return
        save('waiting_for_source')
        for _ in range(20):
            if (output/'STOP').exists():save('stopped');return
            time.sleep(1)

def main():
    from dataclasses import asdict
    from retarget.env.grasp.mano_solver import Settings
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'results/dexonomy_shadow_grasp_full')
    p.add_argument('--output',type=Path,default=ROOT/'results/dexonomy_mano_current')
    p.add_argument('--model',type=Path,default=ROOT/'data/grasp/assets/mano/MANO_RIGHT.pkl')
    p.add_argument('--mapping',type=Path,default=ROOT/'output/dexonomy_shadow_mano_mapping_20261103/data.json')
    p.add_argument('--resume-from',type=Path,action='append',default=[])
    p.add_argument('--devices',default='cuda:0,cuda:1,cuda:2,cuda:3')
    p.add_argument('--watch',action='store_true')
    p.add_argument('--max-shards',type=int,default=0);p.add_argument('--max-rows',type=int,default=0)
    for key in ('batch_size','candidate_chunk','seed'):
        p.add_argument('--'+key.replace('_','-'),type=int,default=getattr(Settings(),key))
    p.add_argument('--memory-budget-gb',type=float,default=Settings().memory_budget_gb)
    a=p.parse_args();devices=a.devices.split(',')
    if len(devices)!=len(set(devices)):p.error('Use distinct CUDA devices')
    settings=Settings(batch_size=a.batch_size,candidate_chunk=a.candidate_chunk,seed=a.seed,memory_budget_gb=a.memory_budget_gb)
    a.output.mkdir(parents=True,exist_ok=True)
    output_lock=(a.output/'.lock').open('a+')
    try:fcntl.flock(output_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another solver is already writing this output directory')
    mapping=read(a.mapping)
    if mapping['model_sha256']!=sha(a.model):raise ValueError('Reviewed mapping was made for a different MANO model')
    frozen_mapping=dict(model_sha256=mapping['model_sha256'],mappings=mapping['mappings'])
    mapping_fingerprint=hashlib.sha256(json.dumps(frozen_mapping,sort_keys=True).encode()).hexdigest()
    payload=dict(settings=asdict(settings),source=str(a.source.resolve()),max_rows=a.max_rows,model_sha256=sha(a.model),mapping_fingerprint=mapping_fingerprint,implementation_sha256={f:sha(ROOT/f) for f in ['retarget/env/grasp/mano_solver.py','scripts/retarget_dexonomy_mano.py','scripts/run_dexonomy_mano_background.py']})
    if a.resume_from:
        references=a.output/'reused_shards.json'
        reused=reused_shards(a.resume_from,a.source,sha(a.model),mapping_fingerprint)
        if references.exists() and read(references)!=reused:
            raise ValueError('Historical shard references changed')
        write(references,reused)
        payload.update(resume_from=[str(p.resolve()) for p in a.resume_from],reused_fingerprint=hashlib.sha256(json.dumps(reused,sort_keys=True).encode()).hexdigest())
    else:reused=[]
    fingerprint=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    frozen_path=a.output/'fixed_contact_mapping.json'
    if frozen_path.exists() and read(frozen_path)!=frozen_mapping:raise ValueError('Fixed contact mapping changed')
    write(frozen_path,frozen_mapping)
    config=a.output/'configuration.json'
    if config.exists() and read(config)['fingerprint']!=fingerprint:raise ValueError('Configuration changed; use an independent output directory')
    write(config,dict(fingerprint=fingerprint,**payload,devices=devices,created_utc=stamp()))
    options=dict(source=str(a.source.resolve()),output=str(a.output.resolve()),model=str(a.model.resolve()),mapping=str(frozen_path.resolve()),settings=asdict(settings),
        workers=len(devices),watch=a.watch,max_shards=a.max_shards,max_rows=a.max_rows,reused=reused)
    write(a.output/'process.json',dict(pid=os.getpid(),devices=devices,started_utc=stamp(),options=options))
    def request_stop(signum,frame):
        (a.output/'STOP').write_text(stamp()+'\\n')
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    started=time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(devices),mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs=[pool.submit(worker,i,d,options,fingerprint) for i,d in enumerate(devices)]
        while True:
            states=[]
            for i in range(len(devices)):
                path=a.output/f'worker_{i}.json'
                if path.exists():
                    state=read(path)
                    active=a.output/'active'/('cuda_'+str(int(devices[i].split(':')[-1]))+'.json')
                    if active.exists():state['active']=read(active)
                    states.append(state)
            done=all(f.done() for f in jobs)
            status=dict(state='completed' if done else 'running',samples=sum(s.get('samples',0) for s in states),
                samples_new=sum(s.get('samples_new',0) for s in states),samples_reused=sum(s.get('samples_reused',0) for s in states),
                shards_completed=sum(s.get('shards_completed',0) for s in states),failed_shards=sum(s.get('failed_shards',0) for s in states),
                geometric_contacts=sum(s.get('geometric_contacts',0) for s in states),geometry_accepted=sum(s.get('geometry_accepted',0) for s in states),
                devices=devices,workers=states,elapsed_seconds=round(time.monotonic()-started,2),last_updated_utc=stamp(),
                force_closure_evaluated=False,dynamic_evaluated=False)
            if status['failed_shards']:status['state']='completed_with_errors' if done else 'running_with_errors'
            if done and (a.output/'STOP').exists():status['state']='stopped'
            write(a.output/'status.json',status)
            if done:
                fatal=[repr(job.exception()) for job in jobs if job.exception() is not None]
                if fatal:
                    status['state']='failed';status['worker_exceptions']=fatal;write(a.output/'status.json',status)
                    raise RuntimeError('Worker failed: '+str(fatal))
                break
            time.sleep(10)
if __name__=='__main__':main()
