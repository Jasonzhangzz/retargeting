#!/usr/bin/env python3
"""Resumable multi-GPU Shadow -> MANO using the production LM/CEM preset."""
from __future__ import annotations
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import math,argparse,collections,concurrent.futures,datetime,gzip,hashlib,json,multiprocessing,sys,time,traceback,signal,fcntl,queue
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


def retry_io(function):
    import errno,functools
    @functools.wraps(function)
    def wrapped(*args,**kwargs):
        for attempt in range(7):
            try:return function(*args,**kwargs)
            except OSError as error:
                if error.errno not in (errno.ESTALE,errno.EIO,errno.EAGAIN) or attempt==6:raise
                time.sleep(.05*2**attempt)
    return wrapped

read=retry_io(read);write=retry_io(write);sha=retry_io(sha)
@retry_io
def save_npz(path,**values):
    import torch
    np.savez_compressed(path,**{k:v.detach().cpu().numpy() if isinstance(v,torch.Tensor) else v for k,v in values.items()})

@retry_io
def load_npz(path):
    with np.load(path,allow_pickle=False) as archive:return {key:archive[key].copy() for key in archive.files}

def worker_layout(devices,workers_per_device):
    if workers_per_device<1:raise ValueError('workers-per-device must be positive')
    devices=[d.strip() for d in devices.split(',')]
    if not devices or any(not d.startswith('cuda:') or not d[5:].isdigit() for d in devices):
        raise ValueError('Use explicit CUDA devices, e.g. cuda:0,cuda:1')
    devices=['cuda:'+str(int(d[5:])) for d in devices]
    if len(devices)!=len(set(devices)):raise ValueError('Use distinct CUDA devices')
    return devices,[d for d in devices for _ in range(workers_per_device)]

def shard_owner(key,workers):
    return int(hashlib.sha256(key.encode()).hexdigest()[:8],16)%workers

def schedule_groups(groups,reused_keys,scheduled,tasks):
    """Queue each uncommitted shard once; workers claim dynamically."""
    for group in groups:
        key=group['type']+'/'+str(group['part'])
        if key not in reused_keys and key not in scheduled:
            tasks.put(group);scheduled.add(key)


def active_path(output,number):
    return Path(output)/'active'/f'worker_{number}.json'


@retry_io
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
    @retry_io
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

def export_shard(group,output,decoder,skeleton,loader,settings,fingerprint,limit=0,mappings=None,progress_path=None):
    import torch
    from scripts.calibrate_wuji_mano import rotmat
    from retarget.env.grasp.pose_gpu import vertex_normals
    from retarget.initialization.dexonomy.util.torch_util import torch_quaternion_to_matrix
    from retarget.env.grasp.mano_solver import SurfaceBatch,optimize,FixedContacts,MeshCache,WarmStartBank
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
    active=Path(progress_path) if progress_path is not None else Path(output)/'active'/('cuda_'+str(decoder.device.index)+'.json')
    progress=dict(source=group['type']+'/'+str(group['part']),total_rows=n,solved_rows=0,stage='loading_geometry',last_updated_utc=stamp())
    write(active,progress)
    a=load_npz(source/'poses.npz')
    shadow=a['grasp_qpos'][:n].copy();object_pose=a['object_pose_world'][:n].copy()
    object_scale=a['object_scale'][:n].copy()
    scene_scale=a['scene_scale'][:n].copy() if 'scene_scale' in a else np.array([e['scene_scale'] for e in entries])
    if object_pose.ndim==3:object_pose=object_pose[:,0]
    if mappings is None or group['type'] not in mappings:raise ValueError('No reviewed mapping for '+group['type'])
    method=settings.solver
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
    results=[];batch_size=settings.batch_size;start=0;t0=time.monotonic()
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
            if settings.candidate_chunk<=16:raise
            import gc
            from dataclasses import replace
            settings=replace(settings,candidate_chunk=max(16,settings.candidate_chunk//2))
            if hasattr(decoder,'collision_geometry'):del decoder.collision_geometry
            if hasattr(decoder,'shared_scorers'):decoder.shared_scorers.clear()
            gc.collect();torch.cuda.empty_cache()
            progress.update(candidate_chunk=settings.candidate_chunk);write(active,progress)
            continue
        results.append(result)
        del surface;start=stop
        progress.update(solved_rows=stop,stage='solved_batch',last_updated_utc=stamp());write(active,progress)
    result={k:torch.cat([v[k] for v in results],dim=1 if k=='history' else 0) for k in results[0]}
    inverse=np.argsort(order)
    result={k:(v[:,inverse] if k=='history' else v[inverse]) for k,v in result.items()}
    object_pose=torch.as_tensor(object_pose,device=decoder.device,dtype=torch.float32)
    faces_gpu=torch.as_tensor(decoder.base.faces,device=decoder.device,dtype=torch.long)
    variables=result['variables'];angles=result['angles']
    qpos=torch.cat((variables[:,:6],angles),-1).to(torch.float64)
    arrays=collections.defaultdict(list);offsets=[0];templates=[];metrics=[];cases=[];counts=[];projections=[]
    faces=decoder.base.faces;weights=decoder.base.weights;joint_names=['wrist','index_mcp','index_pip','index_dip','middle_mcp','middle_pip','middle_dip','pinky_mcp','pinky_pip','pinky_dip','ring_mcp','ring_pip','ring_dip','thumb_cmc','thumb_mcp','thumb_ip']
    incident=torch.full((778,),-1,device=decoder.device,dtype=torch.long)
    incident.scatter_reduce_(0,faces_gpu.flatten(),torch.arange(len(faces_gpu),device=decoder.device).repeat_interleave(3),reduce='amax')
    corner=(faces_gpu[incident]==torch.arange(778,device=decoder.device)[:,None]).long().argmax(-1)
    local_bary=torch.eye(3,device=decoder.device)[corner]
    all_normals=vertex_normals(result['vertices'],faces_gpu)
    all_object_rotations=torch_quaternion_to_matrix(object_pose[:,3:])
    all_root_rotations=rotmat(variables[:,3:6])
    quality_count=0;quality_flags=[]
    for i,e in enumerate(entries):
        distance=result['distance'][i];vertices=result['vertices'][i]
        selected=torch.nonzero(distance.abs()<=settings.contact_tolerance_m).flatten()
        nn=all_normals[i];object_r=all_object_rotations[i]
        root_r=all_root_rotations[i]
        hand=vertices[selected];projected=result['projected'][i,selected];normal=result['object_normals'][i,selected]
        joint_ids=weights[selected].argmax(-1);joint_positions=result['joints'][i] # ordered 21 points; use root-local skin anchors, not rigid phalanges
        local=(hand-variables[i,:3])@root_r
        obj_local=(projected-object_pose[i,:3])@object_r
        values=dict(hand_vertex_ids=selected.to(torch.int32),hand_face_ids=incident[selected].to(torch.int32),
            hand_barycentric=local_bary[selected],object_face_ids=result['object_face'][i,selected].to(torch.int32),
            object_barycentric=result['object_bary'][i,selected],hand_points_world=hand,object_points_world=projected,
            points_world=(hand+projected)*.5,object_normals_world=normal,hand_normals_world=nn[selected],
            hand_local_points=local,hand_local_normals=nn[selected]@root_r,object_local_points=obj_local,
            distances_m=distance[selected])
        for k,v in values.items():arrays[k].append(v)
        offsets.append(offsets[-1]+len(selected));counts.append(len(selected));projections.append(projected)
        maximum_penetration=float((-distance).clamp_min(0).max())
        fit_rmse=float((result['joints'][i]-target[i]).square().sum(-1).mean().sqrt())
        quality=bool(maximum_penetration<=settings.penetration_tolerance_m and len(selected)>=2 and geometries[geo_indices[i]]['watertight'])
        mapped_hand=result['mapped_hand_points'][i]
        mapped_normals=result['mapped_hand_normals'][i]
        mapped_local=(mapped_hand-variables[i,:3])@root_r
        mapped_error=result['mapped_distance'][i].abs()
        normal_alignment=(mapped_normals*result['mapped_object_normals'][i]).sum(-1)
        mapped_coverage=float(((mapped_error<=settings.contact_tolerance_m)&(normal_alignment<=-math.cos(math.pi/6))).float().mean())
        mapped_distance_coverage=float((mapped_error<=settings.contact_tolerance_m).float().mean())
        from retarget.env.grasp.pose_objective import acceptance,PoseObjectiveSettings
        accepted=acceptance(result['coverage'][i],result['projected_closure'][i],result['closure_valid'][i],result['collision_depth'][i],PoseObjectiveSettings())
        quality=accepted['geometry_success']
        if not quality and result['stop_reason'][i]==1:result['stop_reason'][i]=3
        quality_count+=int(quality);quality_flags.append(quality)
        templates.append(dict(body_names=[rec['shadow_body'] for rec in mapping],local_points=mapped_local.tolist(),
            local_normals=(mapped_normals@root_r).tolist(),finger_qpos=angles[i].tolist(),required_cbody=[],
            mano_face_ids=[rec['mano_face_id'] for rec in mapping],mano_barycentric=[rec['mano_barycentric'] for rec in mapping],
            anchor_frame='MANO root frame at this exact pose; fixed deformable triangle/barycentric slots authoritative',
            provenance=dict(source='reviewed original Shadow template annotation -> fixed MANO triangle mapping',
                mapping_review_flags=[rec['id'] for rec in mapping if rec.get('needs_review')],
                contact_tolerance_m=settings.contact_tolerance_m)))
        metrics.append(dict(hand='mano',method=method,solver_completed=True,task_success=accepted['soft_contact_success'],
            success_criterion='projected_polyhedral_qp_closure_v1',force_closure_evaluated=True,projected_closure=bool(result['projected_closure'][i]),
            closure_cost=float(result['closure_cost'][i]),closure_margin=float(result['closure_margin'][i]),dynamic_evaluated=False,
            native_evaluated=False,geometry_accepted=quality,
            geometry_diagnostics=dict(contact_count=len(selected),mapped_contact_count=len(mapping),mapped_coverage=mapped_coverage,mapped_distance_coverage=mapped_distance_coverage,
                max_mapped_gap_m=float(mapped_error.max()),mapping_review_flags=[rec['id'] for rec in mapping if rec.get('needs_review')],
                max_hand_object_penetration_m=maximum_penetration,max_hand_hand_penetration_m=float(result['collision_depth'][i,1]),
                joint_rmse_to_shadow_m=fit_rmse,object_watertight=geometries[geo_indices[i]]['watertight'],
                self_collision_evaluated=True,self_collision_model='MANO dominant-skin-weight rigid convex pieces; parent-child excluded',contact_query='all 778 MANO vertices against object triangle mesh',
                contact_tolerance_m=settings.contact_tolerance_m,source_grasp_preserved=True)))
        cases.append(dict(name=e['name'],task='grasp',hand='mano',method=method,template=group['type'],grasp_type=group['type'],
            phase='retargeting',task_success=accepted['soft_contact_success'],success_criterion='projected_polyhedral_qp_closure_v1',
            cached_case=i,template_id=i,model='mano_model.json',trajectory='poses.npz',summary='metrics.json.gz',
            contact_data='contacts.npz',object_geometry_id=geo_indices[i],object=e['object'],
            scene_path=e['scene_path'],scene_scale=e['scene_scale'],source_manifest=str(source/'manifest.json'),source_cached_case=i))
    maximum=max(counts,default=0);padded=variables.new_zeros((n,maximum,3))
    for i,p in enumerate(projections):padded[i,:len(p)]=p
    save_npz(out/'poses.npz',qpos=qpos,mano_pca=result['variables'][:,6:],
        mano_joints=result['joints'],projected_closure=result['projected_closure'],closure_margin=result['closure_margin'],
        collision_depth=result['collision_depth'],
        object_pose_world=object_pose,object_scale=object_scale,scene_scale=scene_scale,object_geometry_ids=np.array(geo_indices),
        mapped_hand_points=result['mapped_hand_points'],mapped_hand_normals=result['mapped_hand_normals'],
        mapped_object_points=result['mapped_object_points'],mapped_object_normals=result['mapped_object_normals'],
        mapped_distances_m=result['mapped_distance'],mapped_mano_face_ids=np.array([rec['mano_face_id'] for rec in mapping]),
        mapped_mano_barycentric=np.array([rec['mano_barycentric'] for rec in mapping]))
    save_npz(out/'contacts.npz',offsets=np.array(offsets,np.int64),
        mapped_hand_points_world=result['mapped_hand_points'],mapped_object_points_world=result['mapped_object_points'],
        mapped_hand_normals_world=result['mapped_hand_normals'],mapped_object_normals_world=result['mapped_object_normals'],
        mapped_object_face_ids=result['mapped_object_face_ids'],mapped_object_barycentric=result['mapped_object_barycentric'],
        mapped_distances_m=result['mapped_distance'],**{k:torch.cat(v,dim=0) for k,v in arrays.items()})
    geometry_arrays={}
    for i,g in enumerate(geometries):geometry_arrays['object_'+str(i)+'_vertices']=g['vertices'];geometry_arrays['object_'+str(i)+'_faces']=g['faces']
    geometry_arrays['mano_faces']=faces
    save_npz(out/'geometry.npz',**geometry_arrays)
    gz(out/'templates.json.gz',templates);gz(out/'metrics.json.gz',metrics)
    model_info=dict(model='MANO',side='right',model_path=str(decoder.base.path),model_sha256=sha(decoder.base.path),
        keypoints='mano_joints: world-frame 21x3; wrist index 0; wrist orientation qpos[3:6] axis-angle',units='metres/radians',shape='fixed beta zero, unit scale',mano_pose_51='root axis-angle(3), absolute local axis-angles(45), translation(3)',
        qpos='translation(3), root axis-angle(3), absolute local axis-angles(45)',
        grasp_qpos='translation(3), root quaternion wxyz(4), absolute local axis-angles(45)',hands_mean_already_added=True,
        pca_components=settings.pca_components,joint_order_21='wrist; thumb/index/middle/ring/pinky, each MCP/PIP/DIP/tip (thumb CMC/MCP/IP/tip)')
    write(out/'mano_model.json',model_info)
    write(out/'manifest.json',dict(version=2,storage='final_grasp_keypoints_v3',hand='mano',method=method,task='grasp',cases=cases,
        model_representation='MANO deformable LBS; MANO-aware loader required',contact_representation='fixed mapped triangle slots plus separate vertex-sampled proximity; no contact forces'))
    write(out/'object_assets.json',[{k:v for k,v in g.items() if k not in ('vertices','faces')} for g in geometries])
    write(out/'provenance.json',dict(configuration_fingerprint=fingerprint,settings=asdict(settings),source_poses_sha256=source_hash,
        source_manifest_sha256=sha(source/'manifest.json'),model_sha256=sha(decoder.base.path),created_utc=stamp(),
        source_stage='grasp',fixed_mapping=True,architecture_version='mano_shared_objective_cem_batch_v1',objective='template_historical_wrench_qp_axis_v1',closure_backend='historical wrench_qp; 4 sliding rays; no torsional rays; 13 wrench targets; 96 FISTA updates',
        warm_start_policy='disabled; repository MANO palm and bone IK initialization',
        stop_reason_codes={'0':'budget','1':'geometry_accepted','2':'plateau','3':'geometry_recheck_failed'},history_policy='not persisted; final keypoints, pose, object transforms and contacts only',
        mesh_cache=dict(builds=mesh_cache.builds-mesh_start[0],hits=mesh_cache.hits-mesh_start[1]),mapping_review_flags=[rec['id'] for rec in mapping if rec.get('needs_review')],paired_mano_ground_truth_available=False,initialization='rigid palm SVD + batched anatomical bone IK',
        contact_force_available=False,force_closure_evaluated=True,dynamic_evaluated=False))
    checks={}
    a={k:torch.as_tensor(v,device=decoder.device) for k,v in load_npz(out/'poses.npz').items()}
    checks['source_preserved']=sha(source/'poses.npz')==source_hash
    checks['finite']=bool(all(torch.isfinite(a[k]).all() for k in ['qpos','mano_pca','mano_joints','mapped_object_points','object_pose_world']))
    checks['cem_monotone']=bool((result['final_cost']<=result['initial_cost']+1e-5+1e-5*result['initial_cost'].abs()).all())
    checks['dimensions']=a['qpos'].shape==(n,51) and a['mano_joints'].shape==(n,21,3)
    checks['final_pose_preserved']=torch.equal(a['qpos'],qpos) and torch.equal(a['mano_joints'],result['joints'])
    checks['early_exit_geometry']=bool(torch.tensor(quality_flags,device=decoder.device)[result['stop_reason']==1].all())
    checks['contact_offsets']=len(offsets)==n+1 and offsets[-1]==sum(counts)
    triangles=result['vertices'][:,faces_gpu[torch.tensor([rec['mano_face_id'] for rec in mapping],device=decoder.device)]]
    rebuilt=(triangles*variables.new_tensor([rec['mano_barycentric'] for rec in mapping])[None,:,:,None]).sum(-2)
    checks['fixed_mapping_reconstruction']=bool(torch.allclose(rebuilt,result['mapped_hand_points'],atol=2e-6))
    if not all(checks.values()):raise RuntimeError('Output verification failed: '+str(checks))
    write(out/'checksums.json',{p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in ('verification.json','checksums.json')})
    verified=dict(success=True,samples=n,projected_closure=int(result['projected_closure'].count_nonzero()),geometric_contacts=sum(counts),geometry_accepted=quality_count,checks=checks,
        configuration_fingerprint=fingerprint,source_poses_sha256=source_hash,seconds=round(time.monotonic()-t0,3),
        actual_pose_batch_max=min(batch_size,n),timing_scope='solve_and_final_export_after_geometry_load; includes graph setup when needed',path=str(out))
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


def worker(number,device,options,fingerprint,tasks):
    import torch,warp as wp
    from retarget.env.grasp.mano_solver import Settings,ManoDecoder,ShadowSkeleton
    torch.set_num_threads(1);torch.cuda.set_device(device)
    settings=Settings(**options['settings'])
    free,total=torch.cuda.mem_get_info(device)
    budget=min(settings.memory_budget_gb*2**30,free*.8)/options['workers_per_device']
    if budget<2*2**30:raise RuntimeError('Insufficient free VRAM on '+device)
    torch.cuda.set_per_process_memory_fraction(budget/total,device)
    wp.init()
    model=ManoDecoder(options['model'],device,settings.pca_components)
    model.artifact_root=Path(options['output'])
    mappings=read(options['mapping'])['mappings']
    model.configure_contact_vertices(mappings)
    skeleton=ShadowSkeleton(ROOT/'data/dexonomy/shadow_assets/right.xml',device);loader=GeometryLoader(ROOT/'data/dexonomy')
    output=Path(options['output']);source=Path(options['source']);completed={};failed={};started=time.monotonic()
    reused={r['key']:r for r in options.get('reused',[]) if shard_owner(r['key'],options['workers'])==number}
    profile=dict(worker=number,device=device,pid=os.getpid(),memory_budget_bytes=int(budget),started_utc=stamp(),
        configured_batch_size=settings.batch_size,population=settings.population,iterations=settings.iterations)
    active=active_path(output,number)
    write(active,dict(source='',stage='starting',solved_rows=0,total_rows=0))
    def save(state):
        write(output/f'worker_{number}.json',dict(state=state,**profile,active=read(active) if active.exists() else None,shards_completed=len(completed)+len(reused),samples=sum(x['samples'] for x in completed.values())+sum(x['samples'] for x in reused.values()),
            samples_new=sum(x['samples'] for x in completed.values()),projected_closure=sum(x.get('projected_closure',0) for x in [*completed.values(),*reused.values()]),samples_reused=sum(x['samples'] for x in reused.values()),
            geometric_contacts=sum(x['geometric_contacts'] for x in [*completed.values(),*reused.values()]),geometry_accepted=sum(x['geometry_accepted'] for x in [*completed.values(),*reused.values()]),
            failed_shards=len(failed),errors=list(failed.values()),elapsed_seconds=round(time.monotonic()-started,2),
            peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(device),last_updated_utc=stamp()))
    save('starting')
    while True:
        if (output/'STOP').exists():save('stopped');return
        try:group=tasks.get(timeout=2)
        except queue.Empty:
            save('waiting_for_source');continue
        if group is None:
            save('completed_with_errors' if failed else 'completed');return
        key=group['type']+'/'+str(group['part'])
        try:
            value=export_shard(group,output,model,skeleton,loader,settings,fingerprint,options.get('max_rows',0),mappings,progress_path=active)
            completed[key]=value
            print(json.dumps(dict(worker=number,device=device,source=key,samples=value['samples'],seconds=value['seconds']),ensure_ascii=False),flush=True)
        except Exception as error:
            failed[key]=dict(source=key,error=repr(error),traceback=traceback.format_exc(),samples=group['samples'])
            write(output/group['type']/'shards'/f"{group['part']:07d}"/'failure.json',failed[key])
            print(json.dumps(failed[key]),flush=True)
        save('running')

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
    p.add_argument('--workers-per-device',type=int,default=1)
    p.add_argument('--watch',action='store_true')
    p.add_argument('--solver',choices=('cem',),default='cem')
    p.add_argument('--max-shards',type=int,default=0);p.add_argument('--max-rows',type=int,default=0)
    for key in ('batch_size','candidate_chunk','seed','population','iterations'):
        p.add_argument('--'+key.replace('_','-'),type=int,default=getattr(Settings(),key))
    p.add_argument('--memory-budget-gb',type=float,default=Settings().memory_budget_gb)
    a=p.parse_args()
    try:devices,worker_devices=worker_layout(a.devices,a.workers_per_device)
    except ValueError as error:p.error(str(error))
    settings=Settings(solver=a.solver,batch_size=a.batch_size,candidate_chunk=a.candidate_chunk,seed=a.seed,memory_budget_gb=a.memory_budget_gb,population=a.population,iterations=a.iterations)
    a.output.mkdir(parents=True,exist_ok=True)
    output_lock=(a.output/'.lock').open('a+')
    try:fcntl.flock(output_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another solver is already writing this output directory')
    mapping=read(a.mapping)
    if mapping['model_sha256']!=sha(a.model):raise ValueError('Reviewed mapping was made for a different MANO model')
    frozen_mapping=dict(model_sha256=mapping['model_sha256'],mappings=mapping['mappings'])
    mapping_fingerprint=hashlib.sha256(json.dumps(frozen_mapping,sort_keys=True).encode()).hexdigest()
    from retarget.env.grasp.pose_objective import PoseObjectiveSettings
    from retarget.solver.base import Settings as SamplerSettings
    payload=dict(objective='template_historical_wrench_qp_axis_v1',closure_backend='historical wrench_qp; 4 sliding rays; no torsional rays; 13 wrench targets; 96 FISTA updates',object_mass_kg=.1,objective_settings=asdict(PoseObjectiveSettings()),sampler_settings=asdict(SamplerSettings(type='cem',population=settings.population,iterations=settings.iterations)),output_format="final_grasp_keypoints_v3",scheduling="dynamic_queue",worker_devices=worker_devices,workers_per_device=a.workers_per_device,settings=asdict(settings),source=str(a.source.resolve()),max_rows=a.max_rows,model_sha256=sha(a.model),mapping_fingerprint=mapping_fingerprint,implementation_sha256={f:sha(ROOT/f) for f in ['retarget/env/grasp/mano_solver.py','retarget/env/grasp/mano_adapter.py','retarget/env/grasp/pose_objective.py','retarget/env/grasp/closure_types.py','retarget/env/grasp/projected_qp.py','retarget/env/grasp/pose_gpu.py','retarget/env/grasp/pose_inputs.py','retarget/env/grasp/qp_projection.py','retarget/solver/cem.py','scripts/retarget_dexonomy_mano.py','scripts/run_dexonomy_mano_background.py']})
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
        workers=len(worker_devices),workers_per_device=a.workers_per_device,watch=a.watch,max_shards=a.max_shards,max_rows=a.max_rows,reused=reused)
    write(a.output/'process.json',dict(pid=os.getpid(),devices=devices,started_utc=stamp(),options=options))
    def request_stop(signum,frame):
        (a.output/'STOP').write_text(stamp()+'\\n')
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    started=time.monotonic()
    context=multiprocessing.get_context('spawn')
    with context.Manager() as manager, concurrent.futures.ProcessPoolExecutor(max_workers=len(worker_devices),mp_context=context) as pool:
        tasks=manager.Queue();scheduled=set();reused_keys={r['key'] for r in reused};source_closed=False
        jobs=[pool.submit(worker,i,d,options,fingerprint,tasks) for i,d in enumerate(worker_devices)]
        while True:
            if not source_closed and not (a.output/'STOP').exists():
                groups=source_groups(a.source)
                if a.max_shards:groups=groups[:a.max_shards]
                schedule_groups(groups,reused_keys,scheduled,tasks)
                producer=read(a.source/'status.json') if (a.source/'status.json').exists() else {'state':'completed'}
                if not a.watch or producer.get('state')=='completed':
                    for _ in worker_devices:tasks.put(None)
                    source_closed=True
            states=[]
            for i in range(len(worker_devices)):
                path=a.output/f'worker_{i}.json'
                if path.exists():
                    state=read(path)
                    active=active_path(a.output,i)
                    if active.exists():state['active']=read(active)
                    states.append(state)
            done=all(f.done() for f in jobs)
            status=dict(state='completed' if done else 'running',samples=sum(s.get('samples',0) for s in states),
                samples_new=sum(s.get('samples_new',0) for s in states),samples_reused=sum(s.get('samples_reused',0) for s in states),
                projected_closure=sum(s.get('projected_closure',0) for s in states),shards_completed=sum(s.get('shards_completed',0) for s in states),failed_shards=sum(s.get('failed_shards',0) for s in states),
                geometric_contacts=sum(s.get('geometric_contacts',0) for s in states),geometry_accepted=sum(s.get('geometry_accepted',0) for s in states),
                devices=devices,workers=states,elapsed_seconds=round(time.monotonic()-started,2),last_updated_utc=stamp(),
                force_closure_evaluated=True,dynamic_evaluated=False)
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
