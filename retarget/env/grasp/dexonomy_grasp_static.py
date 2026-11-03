"""Static Shadow reconstruction: source poses, geometric contacts, compact shards.

No integration, force/torque computation, grasp optimization or closure solve.
"""
from __future__ import annotations
from collections import OrderedDict,defaultdict
from functools import lru_cache
from pathlib import Path
import copy,gzip,hashlib,io,json,os,pickle,tarfile,time,traceback,warnings
import numpy as np
from .dexonomy_replay import ROOT,NumpyUnpickler,load_record,write_json,sha,relative_pose,make_env

STAGES=('grasp',)
GRASP_STAGE=0
STORAGE='dexonomy_shadow_grasp_snapshot_v1'
_CACHE=None

def decode_record(raw):
    f=io.BytesIO(raw);v=np.lib.format.read_magic(f)
    (np.lib.format.read_array_header_1_0 if v==(1,0) else np.lib.format.read_array_header_2_0)(f)
    data=NumpyUnpickler(f).load().item()
    for stage in STAGES:
        a=np.asarray(data[stage+'_qpos'])
        if a.ndim!=2 or a.shape[1]!=29 or not np.isfinite(a).all():raise ValueError('Invalid Shadow source poses')
    if len({len(data[s+'_qpos']) for s in STAGES})!=1:raise ValueError('Stage lengths differ')
    return data

@lru_cache(maxsize=256)
def base_scene(dataset,name):
    root=(Path(dataset)/'extracted').resolve()
    name=str(name).replace('\\','/').split('assets/object/',1)[-1]
    path=(root/name).resolve()
    if not path.is_relative_to(root):raise ValueError('Invalid scene path')
    scene=load_record(path)
    for cfg in scene['scene'].values():
        for k,v in list(cfg.items()):
            if k.endswith('_path') and isinstance(v,str):
                p=(path.parent/v).resolve()
                if not p.is_relative_to(root):raise ValueError('Asset escapes dataset')
                cfg[k]=str(p)
    return scene

def scaled_scene(dataset,record,row):
    scene=copy.deepcopy(base_scene(str(Path(dataset).resolve()),str(record['scene_path'])))
    n=len(record['grasp_qpos'])
    factor=float(np.asarray(record.get('scene_scale',np.ones(n))).reshape(-1)[row])
    if not np.isfinite(factor) or factor<=0:raise ValueError('Invalid scale')
    for cfg in scene['scene'].values():
        if cfg['type']=='rigid_object':
            cfg['scale']=np.asarray(cfg['scale'],float)*factor
            cfg['pose']=np.asarray(cfg['pose'],float).copy();cfg['pose'][:3]*=factor
    return scene,factor

def build_static_env(dataset,scene,visual=False):
    """Assemble official geometry without reset, forward dynamics or integration."""
    import sys,mujoco
    sys.path.insert(0,str(ROOT/'third_party/Dexonomy'))
    from dexonomy.sim.mujoco_env import MuJoCo_BaseEnv,MuJoCo_EvalCfg
    env=MuJoCo_BaseEnv.__new__(MuJoCo_BaseEnv)
    env._cfg=MuJoCo_EvalCfg();env._spec=mujoco.MjSpec()
    env._spec.meshdir=str(ROOT/'third_party/Dexonomy')
    env._spec.option.cone=mujoco.mjtCone.mjCONE_ELLIPTIC
    env._spec.option.impratio=10
    env._spec.compiler.discardvisual=not visual
    env._add_hand(str((Path(dataset)/'shadow_assets/right.xml').resolve()),True,False,None)
    env._obj_init_qpos=[];env._obj_nv=0
    for name,cfg in scene['scene'].items():
        if cfg['type']=='rigid_object':env._add_rigid_object(name,**cfg)
        elif cfg['type']=='plane':env._add_plane(name,**cfg)
        else:raise ValueError('Static Shadow release expects rigid scenes')
    env._obj_init_qpos=np.asarray(env._obj_init_qpos,float)
    env._model=env._spec.compile();env._data=mujoco.MjData(env._model)
    env._obj_active_bodyid=env._model.body(env._cfg.obj_prefix+scene['task']['obj_name']+'world').id
    return env

class StaticModels:
    def __init__(self,dataset,capacity=8,visual=False):
        import mujoco
        self.dataset=Path(dataset).resolve();self.capacity=capacity;self.visual=visual;self.cache=OrderedDict()
        self.compiled=0;self.hits=0
        warnings.filterwarnings('ignore',message="Attach conflict when attaching.*")
        def warning(message):
            if not message.startswith("Attach conflict when attaching"):
                print('MUJOCO_WARNING',message,flush=True)
        mujoco.set_mju_user_warning(warning)
    def get(self,record,row):
        factor=float(np.asarray(record.get('scene_scale',np.ones(len(record['grasp_qpos'])))).reshape(-1)[row])
        key=(str(record['scene_path']),factor)
        if key in self.cache:
            self.hits+=1;self.cache.move_to_end(key);return self.cache[key]
        scene,factor=scaled_scene(self.dataset,record,row)
        env=build_static_env(self.dataset,scene,self.visual)
        if env._model.nq!=36 or len(env.get_hand_qpos())!=29:raise ValueError('Expected one Shadow hand and one rigid object')
        item=(env,scene,factor);self.cache[key]=item;self.compiled+=1
        if len(self.cache)>self.capacity:self.cache.popitem(last=False)
        return item

def geometry_contacts(model,data,object_body):
    """Positions/normals only. Never reads or computes contact forces."""
    descendants={int(object_body)}
    for b in range(object_body+1,model.nbody):
        if int(model.body_parentid[b]) in descendants:descendants.add(b)
    rows=defaultdict(list);object_depth=0.;self_depth=0.
    for contact in data.contact:
        b1,b2=map(int,model.geom_bodyid[[contact.geom1,contact.geom2]])
        o1,o2=b1 in descendants,b2 in descendants
        if o1==o2:
            if not o1 and b1>0 and b2>0 and model.body_mocapid[b1]<0 and model.body_mocapid[b2]<0:self_depth=max(self_depth,-float(contact.dist))
            continue
        hand=b2 if o1 else b1
        if hand==0 or model.body_mocapid[hand]>=0:continue
        normal=np.asarray(contact.frame[:3]).copy()*(1 if o1 else -1)
        point=np.asarray(contact.pos).copy();distance=float(contact.dist)
        hp=point+.5*distance*normal;op=point-.5*distance*normal
        hrot=data.xmat[hand].reshape(3,3);orot=data.xmat[object_body].reshape(3,3)
        rows['points_world'].append(point);rows['object_normals_world'].append(normal)
        rows['hand_points_world'].append(hp);rows['object_points_world'].append(op)
        rows['hand_local_points'].append(hrot.T@(hp-data.xpos[hand]))
        rows['hand_local_normals'].append(hrot.T@(-normal))
        rows['object_local_points'].append(orot.T@(op-data.xpos[object_body]))
        rows['distances_m'].append(distance);rows['hand_body_ids'].append(hand)
        rows['hand_geom_ids'].append(int(contact.geom2 if o1 else contact.geom1))
        rows['object_geom_ids'].append(int(contact.geom1 if o1 else contact.geom2))
        object_depth=max(object_depth,-distance)
    vector_names=('points_world','object_normals_world','hand_points_world','object_points_world','hand_local_points','hand_local_normals','object_local_points')
    out={k:np.asarray(rows[k],np.float32).reshape(-1,3) for k in vector_names}
    out.update({k:np.asarray(rows[k],np.int32 if k.endswith('_ids') else np.float32) for k in ('distances_m','hand_body_ids','hand_geom_ids','object_geom_ids')})
    return out,object_depth,self_depth

def static_sample(models,record,row):
    import mujoco
    env,scene,factor=models.get(record,row)
    model,data=env._model,env._data;body=env._obj_active_bodyid
    source=np.stack([record[s+'_qpos'][row] for s in STAGES])
    if np.max(np.abs(np.linalg.norm(source[:,3:7],axis=1)-1))>1e-3:raise ValueError('Invalid source quaternion')
    object_pose=np.asarray(env._obj_init_qpos,float).copy()
    states=[];observations=[];depths=[];limits=[]
    for q in source:
        data.qpos[:29]=q;data.qpos[29:]=object_pose;data.qvel[:]=0
        mujoco.mj_kinematics(model,data);mujoco.mj_collision(model,data)
        obs,od,sd=geometry_contacts(model,data,body)
        states.append(data.qpos.copy());observations.append(obs);depths.append([od,sd])
        violation=0.
        for j in range(model.njnt):
            if model.jnt_limited[j]:
                value=data.qpos[model.jnt_qposadr[j]]
                violation=max(violation,float(model.jnt_range[j,0]-value),float(value-model.jnt_range[j,1]))
        limits.append(violation)
    cfg=scene['scene'][scene['task']['obj_name']]
    names=np.asarray([model.body(b).name for b in range(model.nbody)],str)
    result={'source_stage_hand_qpos':source,'stage_qpos':np.stack(states),
        'object_pose_world':object_pose,'object_pose_in_hand':relative_pose(source[GRASP_STAGE],object_pose),
        'object_scale':np.asarray(cfg['scale'],float),'scene_scale':factor,
        'contacts':observations,'penetration_m':np.asarray(depths),'joint_limit_violation_rad':np.asarray(limits),
        'body_names':names,'scene_path':str(record['scene_path'])}
    return result

def atomic_npz(path,**arrays):
    path=Path(path);temp=path.with_suffix('.tmp.npz')
    np.savez_compressed(temp,**arrays);temp.replace(path)

def write_shard(output,typ,part,samples,failures,source_records):
    """Seven shared files per shard; no per-pose scene or image files."""
    folder=Path(output)/typ/'shards'/f'{part:07d}';folder.mkdir(parents=True,exist_ok=True)
    if not samples:
        write_json(folder/'failures.json',failures)
        write_json(folder/'summary.json',{'samples':0,'errors':len(failures)})
        return {'type':typ,'part':part,'samples':0,'errors':len(failures),'contacts':0,'path':str(folder)}
    n=len(samples);states=np.stack([s['stage_qpos'] for s in samples])
    original=np.stack([s['source_stage_hand_qpos'] for s in samples])
    counts=np.asarray([[len(c['points_world']) for c in s['contacts']] for s in samples],np.int32)
    maximum=int(counts[:,GRASP_STAGE].max());points=np.zeros((n,maximum,3),np.float32);normals=np.zeros_like(points)
    offsets=[0];flattened=defaultdict(list);templates=[];metrics=[];entries=[]
    names=samples[0]['body_names']
    for i,s in enumerate(samples):
        for obs in s['contacts']:
            for k,v in obs.items():flattened[k].append(v)
            offsets.append(offsets[-1]+len(obs['points_world']))
        obs=s['contacts'][GRASP_STAGE];count=len(obs['points_world']);points[i,:count]=obs['object_points_world'];normals[i,:count]=obs['object_normals_world']
        body_names=s['body_names'][obs['hand_body_ids']].tolist()
        templates.append({'body_names':body_names,'local_points':obs['hand_local_points'].tolist(),
            'local_normals':obs['hand_local_normals'].tolist(),'finger_qpos':original[i,GRASP_STAGE,7:].tolist(),
            'required_cbody':[],'provenance':{'source':'static geometric collision at original grasp target','not_original_contact_labels':True}})
        metric={'hand':'shadow','method':'dexonomy_static','task_success':None,'dynamic_evaluated':False,
            'force_closure_evaluated':False,'geometry_diagnostics':{
                'contact_count':count,'max_hand_object_penetration_m':float(s['penetration_m'][GRASP_STAGE,0]),
                'max_hand_hand_penetration_m':float(s['penetration_m'][GRASP_STAGE,1]),'joint_limit_violation_rad':float(s['joint_limit_violation_rad'][GRASP_STAGE])},
            'stage_contact_counts':counts[i].tolist(),'stage_penetration_m':s['penetration_m'].tolist(),
            'stage_joint_limit_violation_rad':s['joint_limit_violation_rad'].tolist()}
        metrics.append(metric);ref=s['source_ref'];case=f'{part:07d}_{i:05d}'
        entries.append({'name':case,'task':'grasp','hand':'shadow','method':'dexonomy_static','template':typ,'grasp_type':typ,
            'cached_case':i,'template_id':i,'trajectory':'poses.npz','summary':'metrics.json.gz','contact_data':'contacts.npz',
            'scene_path':s['scene_path'],'scene_scale':s['scene_scale'],'task_success':None,
            'object':{'source_record':ref['archive_member'],'source_row':ref['row']}})
    arrays={k:np.concatenate(v,axis=0) for k,v in flattened.items()}
    arrays.update(offsets=np.asarray(offsets,np.int64),stage_names=np.asarray(STAGES),body_names=names)
    atomic_npz(folder/'contacts.npz',**arrays)
    atomic_npz(folder/'poses.npz',qpos=states[:,GRASP_STAGE],grasp_qpos=original[:,GRASP_STAGE],source_stage_hand_qpos=original,stage_qpos=states,
        object_pose_world=np.stack([s['object_pose_world'] for s in samples]),
        object_pose_in_hand=np.stack([s['object_pose_in_hand'] for s in samples]),
        object_scale=np.stack([s['object_scale'] for s in samples]),scene_scale=np.asarray([s['scene_scale'] for s in samples]),
        projected_points=points,contact_normals=normals,contact_counts=counts[:,GRASP_STAGE],stage_contact_counts=counts,
        case_names=np.asarray([e['name'] for e in entries]))
    for name,value in [('templates.json.gz',templates),('metrics.json.gz',metrics)]:
        (folder/name).write_bytes(gzip.compress(json.dumps(value,allow_nan=False).encode(),compresslevel=3,mtime=0))
    write_json(folder/'manifest.json',{'version':4,'storage':STORAGE,'task':'grasp','hand':'shadow','dataset_root':str(Path(samples[0]['dataset']).resolve()),
        'stage_names':list(STAGES),'contact_offsets_layout':'sample-major, grasp only','cases':entries})
    write_json(folder/'source_records.json',source_records);write_json(folder/'failures.json',failures)
    summary={'samples':n,'errors':len(failures),'contacts':int(counts.sum()),'grasp_target_with_contacts':int((counts[:,GRASP_STAGE]>0).sum()),
        'source_poses_preserved':True,'dynamic_evaluated':False,'force_closure_evaluated':False,'stage_names':['grasp']}
    write_json(folder/'summary.json',summary)
    checks={p.name:sha(p) for p in folder.iterdir() if p.is_file() and p.name not in {'checksums.json','verification.json'}}
    write_json(folder/'checksums.json',checks)
    # A direct save/load audit on every shard, not a second physics pass.
    with np.load(folder/'poses.npz',allow_pickle=False) as saved:
        if not np.array_equal(saved['source_stage_hand_qpos'],original):raise ValueError('Source pose changed while saving')
    with np.load(folder/'contacts.npz',allow_pickle=False) as saved:
        if len(saved['offsets'])!=n*len(STAGES)+1 or saved['offsets'][-1]!=len(saved['points_world']):raise ValueError('Invalid ragged contact offsets')
        if not np.isfinite(saved['points_world']).all():raise ValueError('Nonfinite contacts')
    write_json(folder/'verification.json',{'success':True,'samples':n,'source_poses_exact':True,'contact_offsets_valid':True,
        'source_fingerprint':hashlib.sha256(json.dumps(source_records,sort_keys=True).encode()).hexdigest()})
    return {'type':typ,'part':part,**summary,'path':str(folder)}

def worker_init(dataset):
    global _CACHE
    _CACHE=StaticModels(dataset,8)

def process_batch(args):
    dataset,output,typ,part,records=args
    folder=Path(output)/typ/'shards'/f'{part:07d}'
    refs=[{'archive_member':name,'sha256':hashlib.sha256(raw).hexdigest(),'selected_rows':rows} for name,raw,rows in records]
    fingerprint=hashlib.sha256(json.dumps(refs,sort_keys=True).encode()).hexdigest()
    v=folder/'verification.json'
    if v.exists():
        old=json.loads(v.read_text())
        if old.get('success') and old.get('source_fingerprint')==fingerprint:
            return {'type':typ,'part':part,**json.loads((folder/'summary.json').read_text()),'path':str(folder),'resumed':True}
        raise ValueError('Resume source fingerprint mismatch: '+str(folder))
    global _CACHE
    if _CACHE is None or str(_CACHE.dataset)!=str(Path(dataset).resolve()):_CACHE=StaticModels(dataset,8)
    samples=[];failures=[]
    for name,raw,rows in records:
        try:
            record=decode_record(raw);indices=range(len(record['grasp_qpos'])) if rows is None else rows
        except Exception as error:
            failures.append({'archive_member':name,'error':str(error)});continue
        for row in indices:
            try:
                sample=static_sample(_CACHE,record,row)
                sample.update(dataset=str(dataset),source_ref={'archive_member':name,'row':int(row)})
                samples.append(sample)
            except Exception as error:
                failures.append({'archive_member':name,'row':int(row),'error':str(error),'traceback':traceback.format_exc()})
    result=write_shard(output,typ,part,samples,failures,refs)
    result.update(models_compiled=_CACHE.compiled,model_cache_hits=_CACHE.hits)
    return result

def archive_jobs(dataset,output,records_per_shard=32):
    dataset=Path(dataset).resolve();part=defaultdict(int);batch=[];active=None
    with tarfile.open(dataset/'Dexonomy_GRASP_shadow.tar.gz','r|gz') as tar:
        for member in tar:
            tar.members.clear()
            if not member.isfile() or not member.name.endswith('.npy') or '/succ_collect/' not in member.name:continue
            path=Path(member.name);parts=path.parts;typ=parts[parts.index('succ_collect')+1]
            if active is not None and typ!=active and batch:
                yield (str(dataset),str(output),active,part[active],batch);part[active]+=1;batch=[]
            active=typ;batch.append((member.name,tar.extractfile(member).read(),None))
            if len(batch)>=records_per_shard:
                yield (str(dataset),str(output),typ,part[typ],batch);part[typ]+=1;batch=[]
        if batch:yield (str(dataset),str(output),active,part[active],batch)

def run_static_full(dataset,output,workers=8,records_per_shard=32):
    import concurrent.futures,multiprocessing
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True);started=time.time()
    status={'state':'running','mode':'static_grasp_geometry','workers':workers,'records_per_shard':records_per_shard,
        'samples_cached':0,'records_processed':0,'shards_completed':0,'geometric_contacts':0,'errors':0,'by_type':{},
        'dynamic_evaluated':False,'force_closure_evaluated':False,'stage_names':['grasp']}
    provenance={'dataset_root':str(Path(dataset).resolve()),'download_manifest_sha256':sha(Path(dataset)/'download_manifest.json'),
        'static_source_sha256':sha(Path(__file__)),'model_reference':'shared released scenes plus Shadow assets','source_pose_optimization':False,'grasp_only':True,
        'contact_semantics':'MuJoCo native zero-margin geometric contacts at original source targets; no forces, no integration',
        'geometry_compiler_discardvisual':True,'geometric_ids_reference':'collision-only compiled model; use body names/points when displaying a visual model',
        'stages':list(STAGES)}
    journal=output/'shards.jsonl';known={}
    if journal.exists():
        with journal.open() as previous:
            for line in previous:
                try:
                    item=json.loads(line)
                    if item.get('samples',0)>0 and (Path(item['path'])/'verification.json').exists():known[(item['type'],int(item['part']))]=item
                except (ValueError,KeyError):continue
    for item in known.values():
        status['samples_cached']+=item['samples'];status['shards_completed']+=1;status['geometric_contacts']+=item['contacts'];status['errors']+=item['errors']
        group=status['by_type'].setdefault(item['type'],{'samples':0,'shards':0,'errors':0})
        group['samples']+=item['samples'];group['shards']+=1;group['errors']+=item['errors']
    baseline_samples=status['samples_cached'];status['samples_reused']=baseline_samples
    write_json(output/'provenance.json',provenance);write_json(output/'status.json',status)
    write_json(output/'index.json',{'storage':'dexonomy_shadow_grasp_index_v1','groups':[{'grasp_type':typ,**v,'shards':typ+'/shards'} for typ,v in status['by_type'].items()]})
    with journal.open('a') as log,concurrent.futures.ProcessPoolExecutor(max_workers=workers,
        initializer=worker_init,initargs=(str(Path(dataset).resolve()),),mp_context=multiprocessing.get_context('spawn')) as pool:
        pending={};jobs=iter(archive_jobs(dataset,output,records_per_shard));exhausted=False
        while pending or not exhausted:
            while len(pending)<2*workers and not exhausted:
                try:
                    job=next(jobs);pending[pool.submit(process_batch,job)]=job
                except StopIteration:exhausted=True
            if not pending:break
            completed,_=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in completed:
                job=pending.pop(future)
                try:result=future.result()
                except Exception as error:
                    result={'type':job[2],'part':job[3],'samples':0,'errors':1,'contacts':0,'fatal_batch_error':str(error),'traceback':traceback.format_exc(),'source_records':[{'archive_member':name,'rows':rows,'sha256':hashlib.sha256(raw).hexdigest()} for name,raw,rows in job[4]]}
                    write_json(output/'batch_errors'/f'{job[2]}_{job[3]:07d}.json',result)
                key=(result['type'],int(result['part']))
                if key in known and result.get('fatal_batch_error'):
                    status.update(state='failed_resume_verification',error=result['fatal_batch_error'])
                    write_json(output/'status.json',status)
                    raise RuntimeError(result['fatal_batch_error'])
                if key not in known:
                    known[key]=result;log.write(json.dumps(result)+'\n');log.flush()
                    status['samples_cached']+=result['samples'];status['shards_completed']+=1
                    status['geometric_contacts']+=result['contacts'];status['errors']+=result['errors']
                    group=status['by_type'].setdefault(result['type'],{'samples':0,'shards':0,'errors':0})
                    group['samples']+=result['samples'];group['shards']+=1;group['errors']+=result['errors']
                status['records_processed']+=len(job[4])
                status['elapsed_seconds']=round(time.time()-started,1)
                status['samples_per_second']=round((status['samples_cached']-baseline_samples)/max(1.,time.time()-started),2)
                write_json(output/'status.json',status)
                write_json(output/result['type']/'status.json',group)
                print(json.dumps({k:v for k,v in status.items() if k!='by_type'}),flush=True)
    status['state']='completed' if not status['errors'] else 'completed_with_errors'
    write_json(output/'status.json',status);write_json(output/'summary.json',status)
    write_json(output/'index.json',{'storage':'dexonomy_shadow_grasp_index_v1','groups':[{'grasp_type':typ,**v,'shards':'%s/shards'%typ} for typ,v in status['by_type'].items()]})
    return status

def preview_jobs(dataset,plan,output):
    """The existing 100-case stratified plan, with no outcome-based selection."""
    grouped=defaultdict(OrderedDict)
    for item in plan:
        path=Path(item['path']);path=path if path.is_absolute() else ROOT/path
        # Read source files directly; retain the original archive member identity.
        parts=path.parts;member='/'.join(parts[parts.index('Dexonomy_GRASP_shadow'):])
        key=(str(path),member);grouped[item['type']].setdefault(key,[]).append(int(item['row']))
    for typ,records in grouped.items():
        yield (str(Path(dataset).resolve()),str(Path(output).resolve()),typ,0,
            [(member,Path(path).read_bytes(),rows) for (path,member),rows in records.items()])

def run_static_preview(dataset,plan_path,output,workers=8):
    import concurrent.futures,multiprocessing
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    jobs=list(preview_jobs(dataset,json.loads(Path(plan_path).read_text()),output));started=time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,initializer=worker_init,
        initargs=(str(Path(dataset).resolve()),),mp_context=multiprocessing.get_context('spawn')) as pool:
        results=list(pool.map(process_batch,jobs))
    report={'mode':'static_grasp_geometry','samples':sum(r['samples'] for r in results),'errors':sum(r['errors'] for r in results),
        'grasp_types':len(results),'geometric_contacts':sum(r['contacts'] for r in results),'elapsed_seconds':time.time()-started,'groups':results}
    write_json(output/'summary.json',report);write_json(output/'index.json',{'storage':'dexonomy_shadow_grasp_index_v1','groups':results})
    return report
