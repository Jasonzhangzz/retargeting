#!/usr/bin/env python3
"""Finite, resumable dataset campaign around the pinned native MANO operation."""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import argparse,collections,copy,fcntl,hashlib,importlib.metadata,json,multiprocessing as mp
import gzip,shutil,signal,sys,tempfile,time,traceback
import numpy as np
import native_graspadmm_retest as native
import native_mano_graspadmm_variant as variant
try:
    from shadow_mano_seed import ShadowManoTransfer,BODY_MAP
except ImportError:
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from retarget.env.grasp.shadow_mano_seed import ShadowManoTransfer,BODY_MAP

CONTEXT={}
SCRIPT_NAMES=['solve_dataset_mano_graspadmm.py','native_graspadmm_retest.py',
              'native_mano_graspadmm_variant.py','native_grasp_extensions.py',
              'shadow_mano_seed.py','diagnose_native_graspadmm.py','shadow_seed_diagnostics.py']


def stat_identity(path):
    st=Path(path).stat()
    return dict(size=st.st_size,mtime_ns=st.st_mtime_ns)


def prepare(a):
    root=a.root;root.mkdir(parents=True,exist_ok=True)
    if (root/'MANIFEST.json').exists():raise RuntimeError('Campaign already prepared; use run to resume')
    source=native.load_upstream(a.native_root/'upstream')
    shutil.copytree(a.mano_root/'hand',root/'hand',dirs_exist_ok=True)
    (root/'upstream').symlink_to(source.root,target_is_directory=True)
    # Convert the original Shadow annotations with the original converter.
    cfg=native.configuration(source,root,'tmpl',workers=1)
    cfg.hand.xml_path=str(source.root/'assets/hand/shadow/right.xml')
    cfg.raw_anno_dir=str(source.root/'assets/hand/shadow/raw_anno')
    cfg.hand_bgroup_path=str(source.root/'assets/hand/shadow/body_group.yaml')
    cfg.hand_kp_path=str(source.root/'assets/hand/shadow/keypoint.yaml')
    cfg.init_tmpl_dir=str(root/'shadow_templates')
    source.convert.operate_tmpl(cfg)
    import mujoco
    official=mujoco.MjModel.from_xml_path(cfg.hand.xml_path)
    dataset=mujoco.MjModel.from_xml_path(str(a.dataset/'shadow_assets/right.xml'))
    if official.nq!=dataset.nq or [official.joint(i).name for i in range(official.njnt)]!=[dataset.joint(i).name for i in range(dataset.njnt)]:
        raise RuntimeError('Dataset Shadow joint ordering does not match upstream')
    np.testing.assert_allclose(official.jnt_range,dataset.jnt_range,atol=1e-12)
    journal=(a.source/'shards.jsonl').read_bytes();(root/'source_journal_snapshot.jsonl').write_bytes(journal)
    latest={}
    for line in journal.splitlines():
        row=json.loads(line);latest[(row['type'],row['part'])]=row
    groups=collections.defaultdict(list)
    for row in latest.values():
        folder=Path(row['path']);folder=folder if folder.is_absolute() else a.source/folder
        if not (root/'shadow_templates'/f'{row["type"]}.npy').is_file():raise RuntimeError('Missing original Shadow template '+row['type'])
        row.update(path=str(folder),identity={name:stat_identity(folder/name) for name in ('manifest.json','poses.npz')})
        groups[row['type']].append(row)
    # Round robin types, then rows within each input shard. All snapshot rows enter the denominator.
    ordered=[]
    for typ in groups:groups[typ].sort(key=lambda r:r['part'])
    for k in range(max(map(len,groups.values()))):
        for typ in sorted(groups,key=lambda x:int(x.split('_')[0])):
            if k<len(groups[typ]):ordered.append(groups[typ][k])
    native.write(root/'source_shards.json',ordered)
    assets={str(p.relative_to(root)):native.sha(p) for directory in ('hand','shadow_templates') for p in (root/directory).rglob('*') if p.is_file()}
    script_hashes={name:native.sha(root/name) for name in SCRIPT_NAMES}
    # Freeze the existing posthoc certificate implementation as well as the native solver.
    certificates={name:native.sha(a.production/name) for name in ('retarget/env/grasp/projected_qp.py','retarget/env/grasp/qp_projection.py')}
    manifest=dict(campaign='dataset_shadow_to_native_mano_graspadmm_20261104',created_utc=native.stamp(),
        authorization='User: launch large-scale MANO GraspADMM on server, dataset Shadow grasp poses as initialization',
        upstream_commit=native.COMMIT,upstream_source_manifest_sha256=native.sha(source.root/'GRASPADMM_UPSTREAM.json'),
        source=str(a.source),dataset=str(a.dataset),production=str(a.production),source_snapshot_sha256=native.sha(root/'source_shards.json'),
        source_journal_snapshot_sha256=native.sha(root/'source_journal_snapshot.jsonl'),
        source_samples=sum(r['samples'] for r in ordered),source_shards=len(ordered),types=list(groups),
        workers=a.workers,chunk_size=a.chunk_size,blas_threads=1,cpu_affinity=sorted(os.sched_getaffinity(0)),
        hand='Upstream segmented rigid MANO: 20 joints; external wrist; repaired XML spelling',
        initialization='Dataset Shadow qpos, original per-type Shadow contact annotations, anatomical point IK; source GSR not re-evaluated or assumed',
        contact_mapping=dict(BODY_MAP),mapping_note='Proximal/middle/distal stay on corresponding finger segment; Shadow pinky metacarpal maps to MANO palm, which has no separate pinky metacarpal',
        solver='Unmodified upstream _single_grasp: 50 x 10, Clarabel, MuJoCo virtual forces, dual, fallback, filter, squeeze, pregrasp',
        penetration_m=dict(hand_object=.002,self=.002),evaluation='All finite final states: static geometry plus production 13-case/96-step projected QP. All native outputs: original six-force GSR plus six-force/six-pure-torque rollouts.',
        budgets=dict(input_rows=sum(r['samples'] for r in ordered),bone_ik_max_evaluations=80,contact_ik_max_evaluations=80,admm_outer=50,hand_substeps=10),
        failure_policy='Every input row retained. Missing states are NaN with explicit status. No failed-row resampling. No CPU/GPU solver substitution.',
        storage='Compact committed shards, final states/contact data/keypoints/object pose only; no optimization history. Scratch native files removed after packing.',
        resume='Only complete, fingerprint and SHA-verified chunks reused. Inflight chunks may recompute after crash; recorded explicitly. No automatic restart after STOP.',
        source_policy='Finite snapshot of currently packed rows; later producer rows require a new campaign',
        callback_required=False,production_gpu_pause='none',python=sys.executable,
        versions={k:importlib.metadata.version(k) for k in native.VERSIONS},assets_sha256=assets,
        scripts_sha256=script_hashes,certificate_sha256=certificates)
    native.write(root/'MANIFEST.json',manifest)
    native.write(root/'status.json',dict(state='prepared',total=manifest['source_samples'],updated_utc=native.stamp()))


def validate(root):
    m=json.loads((root/'MANIFEST.json').read_text())
    for group in ('assets_sha256','scripts_sha256'):
        for name,expected in m[group].items():
            if native.sha(root/name)!=expected:raise RuntimeError('Frozen campaign file changed: '+name)
    for name,expected in m['certificate_sha256'].items():
        if native.sha(Path(m['production'])/name)!=expected:raise RuntimeError('Certificate implementation changed: '+name)
    for key,name in [('source_snapshot_sha256','source_shards.json'),('source_journal_snapshot_sha256','source_journal_snapshot.jsonl')]:
        if native.sha(root/name)!=m[key]:raise RuntimeError('Source snapshot changed')
    for k,v in native.VERSIONS.items():
        if importlib.metadata.version(k).split('+')[0]!=v:raise RuntimeError('Pinned dependency mismatch: '+k)
    return m


def worker_init(root_text):
    global CONTEXT
    root=Path(root_text);m=json.loads((root/'MANIFEST.json').read_text())
    source=native.load_upstream(root/'upstream');cfg=variant.configuration(source,root,'grasp_admm')
    native.worker_init(str(source.root),source.OmegaConf.to_container(cfg));os.chdir(source.root)
    from shadow_seed_diagnostics import original
    CONTEXT=dict(root=root,manifest=m,source=source,cfg=cfg,
        transfer=ShadowManoTransfer(source.root/'assets/hand/shadow/right.xml',cfg.hand.xml_path),
        templates={},shards=collections.OrderedDict(),scenes=collections.OrderedDict())
    os.nice(10)


def load_shard(group):
    c=CONTEXT;key=group['path']
    if key not in c['shards']:
        folder=Path(key)
        for name,identity in group['identity'].items():
            if stat_identity(folder/name)!=identity:raise RuntimeError('Frozen source shard changed: '+str(folder/name))
        entries=json.loads((folder/'manifest.json').read_text())['cases'];arrays=dict(np.load(folder/'poses.npz'))
        if len(entries)!=group['samples'] or len(arrays['grasp_qpos'])!=group['samples']:raise RuntimeError('Source shard row count mismatch')
        hashes={name:native.sha(folder/name) for name in group['identity']}
        c['shards'][key]=(entries,arrays,hashes)
        if len(c['shards'])>2:c['shards'].popitem(last=False)
    return c['shards'][key]


def source_slots(typ):
    c=CONTEXT
    if typ not in c['templates']:
        data=np.load(c['root']/'shadow_templates'/f'{typ}.npy',allow_pickle=True).item()
        hand=c['transfer'].shadow;q=data['grasp_qpos'].reshape(-1);hand.set(q[7:]);local=[]
        from scipy.spatial.transform import Rotation
        r=Rotation.from_quat(q[3:7],scalar_first=True).as_matrix()
        for b,p in zip(data['hand_cbody'],data['hand_cpn_w']):
            bid=hand.model.body(b).id;local.append(((p[:3]-q[:3])@r-hand.data.xpos[bid])@hand.data.xmat[bid].reshape(3,3))
        if not local:raise RuntimeError('No annotated source contacts for '+typ)
        c['templates'][typ]=(data,np.asarray(local))
    return c['templates'][typ]


def scene_for(entry,arrays,index,scratch):
    c=CONTEXT;dataset=Path(c['manifest']['dataset'])
    name=str(entry['scene_path']).replace('\\','/').split('assets/object/',1)[-1]
    path=(dataset/'extracted'/name).resolve()
    if not path.is_relative_to((dataset/'extracted').resolve()):raise ValueError('Scene escapes dataset')
    if str(path) not in c['scenes']:
        scene=np.load(path,allow_pickle=True).item()
        for obj in scene['scene'].values():
            for key,value in obj.items():
                if key.endswith('_path') and isinstance(value,str):
                    obj[key]=str((path.parent/value).resolve())
                    if not Path(obj[key]).is_file():raise FileNotFoundError(obj[key])
        c['scenes'][str(path)]=(scene,native.sha(path))
        if len(c['scenes'])>8:c['scenes'].popitem(last=False)
    scene,source_sha=c['scenes'][str(path)];scene=copy.deepcopy(scene)
    if len(scene['scene'])!=1:raise ValueError('Native ADMM requires a single rigid object')
    factor=float(arrays['scene_scale'][index]);obj=scene['scene'][scene['task']['obj_name']]
    np.testing.assert_allclose(np.asarray(obj['scale'])*factor,arrays['object_scale'][index],atol=1e-7,rtol=1e-6)
    expected=np.array(obj['pose'],float);expected[:3]*=factor
    np.testing.assert_allclose(expected,arrays['object_pose_world'][index],atol=1e-7,rtol=1e-6)
    obj['scale']=arrays['object_scale'][index].copy();obj['pose']=arrays['object_pose_world'][index].copy()
    scene['scene_id']='dataset_pose';target=scratch/'scene.npy';np.save(target,scene)
    return target,source_sha,scene


def pack_arrays(items):
    """Ragged contacts and pregrasp poses use offsets, never object-pickled arrays."""
    fixed={};ragged={};offsets={}
    shapes={'initial_qpos':(27,),'final_qpos':(27,),'last_available_qpos':(27,),'squeeze_qpos':(27,),
            'keypoints_world':(21,3),'object_pose_world':(7,),'object_scale':(3,),'source_shadow_qpos':(29,)}
    widths={'hand_cpn_b':6,'hand_slots':6,'hand_projections':3,'hand_projected_normals':3,
            'target_points':3,'target_normals':3,'cp_o':3,'pregrasp_qpos':27,'source_hand_slots':6}
    for key,shape in shapes.items():fixed[key]=np.stack([np.asarray(x.get(key,np.full(shape,np.nan)),np.float32).reshape(shape) for x in items])
    for key,width in widths.items():
        values=[np.asarray(x.get(key,[]),np.float32).reshape(-1,width) for x in items]
        ragged[key]=np.concatenate(values,axis=0);offsets[key+'_offsets']=np.r_[0,np.cumsum([len(v) for v in values])].astype(np.int64)
    return dict(fixed,**ragged,**offsets)


def solve_row(group,index,entry,arrays,scratch):
    c=CONTEXT;source=c['source'];cfg=copy.deepcopy(c['cfg']);start=time.perf_counter()
    receipt=dict(source_shard=group['path'],source_row=index,grasp_type=group['type'],source_entry=entry,
                 status='not_started',native_operation_output=False,native_gsr_pass=False,
                 torque_passed=False,combined_wrench_passed=False)
    values=dict(source_shadow_qpos=arrays['grasp_qpos'][index],object_pose_world=arrays['object_pose_world'][index],object_scale=arrays['object_scale'][index])
    try:
        template,local=source_slots(group['type'])
        t=time.perf_counter();q,names,contacts,info=c['transfer'].fit(arrays['grasp_qpos'][index],template['hand_cbody'],local)
        receipt.update(mapping_seconds=time.perf_counter()-t,mapping_max_error_m=info['max_contact_error_m'],mapping_within_2mm=info['contact_fit_within_2mm'],hand_cbody=names)
        values.update(initial_qpos=q,source_hand_slots=info['source_hand_cpn_w'])
        scene_path,scene_sha,scene=scene_for(entry,arrays,index,scratch);receipt['source_scene_sha256']=scene_sha
        data=dict(hand_name='mano',tmpl_name=group['type'],hand_cbody=names,
            required_cbody=[list(dict.fromkeys(BODY_MAP[b] for b in g)) for g in template['required_cbody']],
            grasp_qpos=q[None].astype(np.float32),hand_cpn_w=contacts.astype(np.float32),
            scene_path=str(scene_path),n_evo=np.array([0],np.float32),ext_wrench=np.zeros(6,np.float32),
            obj_pose=values['object_pose_world'],obj_scale=values['object_scale'])
        mesh=source.core.load_obj_trimesh(data)[0]
        import trimesh
        projected,_,faces=trimesh.proximity.closest_point(mesh,np.asarray(info['source_hand_cpn_w'])[:,:3])
        bary=trimesh.triangles.points_to_barycentric(mesh.triangles[faces],projected)
        normal=(mesh.vertex_normals[mesh.faces[faces]]*bary[...,None]).sum(1);normal/=np.linalg.norm(normal,axis=1,keepdims=True)
        center=np.asarray(mesh.center_mass,np.float32);task=scene['task']
        if task['type'] in ('force_closure','keyframe'):wrench=np.zeros(6,np.float32)
        elif task['type']=='slide':wrench=np.r_[np.asarray(task['axis'])*.1,np.zeros(3)].astype(np.float32)
        elif task['type']=='hinge':wrench=np.r_[np.cross(task['axis'],center-task['pos'])*.1,np.zeros(3)].astype(np.float32)
        else:raise ValueError('Unsupported original task: '+task['type'])
        data.update(obj_cpn_w=np.c_[projected,normal].astype(np.float32),ext_center=center,ext_wrench=wrench)
        cfg.init_dir=str(scratch/'init');cfg.grasp_dir=str(scratch/'grasp');cfg.debug_dir=str(scratch/'debug');cfg.tmpl_name=group['type']
        path=Path(cfg.init_dir)/'pose.npy';path.parent.mkdir();np.save(path,data)
        record=native.execute_job((str(path),source.OmegaConf.to_container(cfg),'quality',scratch/'state'))
        receipt.update(status='native_output' if record['native_operation_output'] else 'native_rejected',
                       native_operation_output=record['native_operation_output'],operation_seconds=record['operation_seconds'],
                       fallback_used=record['fallback_used'],filters=record['filters'],exception=record['exception'],
                       captured_final_state=record['captured_final_state'])
        state=dict(np.load(scratch/'state/final_state.npz'))
        for name in ('hand_cpn_b','cp_o','last_available_qpos'):
            if name in state:values[name]=state[name]
        if record['captured_final_state']:
            final=state['final_grasp_qpos'].reshape(-1);values['final_qpos']=final
            hand=c['transfer'].mano
            points=hand.world_points(final);from scipy.spatial.transform import Rotation
            wrist=hand.data.xpos[hand.palm]@Rotation.from_quat(final[3:7],scalar_first=True).as_matrix().T+final[:3]
            values['keypoints_world']=np.vstack((wrist,points.reshape(20,3)))
            t=time.perf_counter();import shadow_seed_diagnostics as diagnostics
            row,packet=diagnostics.job((record,scratch/'state'));receipt['geometry']=row
            if row.get('failure')=='diagnostic_exception':raise RuntimeError(row['exception'])
            if packet is not None:values.update(dict(np.load(scratch/'state/diagnostic_contacts.npz')))
            receipt['geometry_seconds']=time.perf_counter()-t
        else:packet=None
        if record['native_operation_output']:
            saved=np.load(record['output'],allow_pickle=True).item()
            values.update(squeeze_qpos=saved['squeeze_qpos'].reshape(-1),pregrasp_qpos=saved['pregrasp_qpos'])
            t=time.perf_counter();evalcfg=variant.configuration(source,c['root'],'eval')
            evaluation=native.evaluate_job((record['output'],source.OmegaConf.to_container(evalcfg)))
            receipt.update(native_gsr_pass=evaluation['success'],native_gsr_observed=evaluation['observed'],native_gsr_exception=evaluation['exception'],gsr_seconds=time.perf_counter()-t)
            t=time.perf_counter();extended=variant.extended_eval_job((index,record['output'],source.OmegaConf.to_container(evalcfg)))
            for trial in extended.get('trials',[]):trial.pop('final_qpos',None)
            receipt.update(torque_passed=extended['torque_passed'],combined_wrench_passed=extended['combined_passed'],extended_wrench=extended,wrench_seconds=time.perf_counter()-t)
        receipt['complete']=not bool(receipt.get('exception'))
    except Exception:
        receipt.update(status='exception',exception=traceback.format_exc(),complete=False);packet=None
    receipt['total_compute_seconds']=time.perf_counter()-start
    # Transient paths are not reusable after scratch cleanup; preserve permanent source identities above.
    if 'geometry' in receipt:receipt['geometry'].pop('input',None)
    return receipt,values,packet


def solve_chunk(job):
    group,start,end,job_id=job;c=CONTEXT;root=c['root'];t=time.perf_counter()
    entries,arrays,hashes=load_shard(group);rows=[];values=[];packets=[]
    heartbeat=root/'workers'/f'{os.getpid()}.json'
    with tempfile.TemporaryDirectory(prefix='mano_admm_',dir='/tmp') as tmp:
        for index in range(start,end):
            if (root/'STOP').exists():break
            scratch=Path(tmp)/str(index);scratch.mkdir()
            native.write(heartbeat,dict(pid=os.getpid(),job_id=job_id,source_row=index,updated_utc=native.stamp()))
            row_seed=20261104+group['part']*2048+index+int(group['type'].split('_')[0])*10000000
            import random,torch
            random.seed(row_seed);np.random.seed(row_seed%(2**32));torch.manual_seed(row_seed)
            row,v,p=solve_row(group,index,entries[index],arrays,scratch);rows.append(row);values.append(v);packets.append(p)
            shutil.rmtree(scratch)
        if len(rows)!=end-start:
            # Never commit a partial chunk as complete.
            return dict(job_id=job_id,interrupted=True,computed_uncommitted=len(rows))
        tclosure=time.perf_counter()
        import diagnose_native_graspadmm
        diagnostic_rows=[r.get('geometry',dict(projected_hand=False,projected_targets=False,slot_joint_pass=False)) for r in rows]
        diagnose_native_graspadmm.closure_batches(diagnostic_rows,packets,Path(c['manifest']['production']))
        for r,d in zip(rows,diagnostic_rows):
            d['tolerated_collision_pass']=d.get('valid_final_state',False) and d.get('hand_object_depth_m',np.inf)<=.002 and d.get('self_depth_m',np.inf)<=.002
            d['diagnostic_joint_pass']=d.get('projected_hand',False) and d.get('slot_joint_pass',False) and d['tolerated_collision_pass'] and d.get('native_contact_count',0)>0
            r['geometry']=d
        folder=root/'shards'/job_id;folder.parent.mkdir(parents=True,exist_ok=True)
        staging=folder.with_name(folder.name+f'.tmp.{os.getpid()}');staging.mkdir(exist_ok=False)
        np.savez_compressed(staging/'poses.npz',**pack_arrays(values))
        native.write(staging/'rows.json',rows)
        with gzip.open(staging/'rows.json.gz','wb',compresslevel=6) as stream:stream.write((staging/'rows.json').read_bytes())
        (staging/'rows.json').unlink()
        receipt=dict(job_id=job_id,manifest_sha256=native.sha(root/'MANIFEST.json'),source_sha256=hashes,
            type=group['type'],source_start=start,source_end=end,samples=len(rows),
            mapped=sum('initial_qpos' in v for v in values),valid_final_states=sum('final_qpos' in v for v in values),
            native_outputs=sum(r['native_operation_output'] for r in rows),gsr_passed=sum(r['native_gsr_pass'] for r in rows),
            projected_closure=sum(r['geometry'].get('projected_hand',False) for r in rows),
            joint_quality=sum(r['geometry']['diagnostic_joint_pass'] for r in rows),
            torque_passed=sum(r['torque_passed'] for r in rows),exceptions=sum(not r['complete'] for r in rows),
            chunk_wall_seconds=time.perf_counter()-t,closure_seconds=time.perf_counter()-tclosure,
            files_sha256={n:native.sha(staging/n) for n in ('poses.npz','rows.json.gz')},completed_utc=native.stamp())
        native.write(staging/'receipt.json',receipt);staging.rename(folder)
        native.write(heartbeat,dict(pid=os.getpid(),job_id=job_id,state='committed',updated_utc=native.stamp()))
        return receipt


def jobs_for(groups,size,pilot=False):
    seen=set()
    for group in groups:
        if pilot:
            if group['type'] in seen:continue
            seen.add(group['type']);yield(group,0,1,f'pilot_{group["type"]}_{group["part"]:07d}');continue
        for start in range(0,group['samples'],size):
            yield(group,start,min(start+size,group['samples']),f'{group["type"]}_{group["part"]:07d}_{start:04d}')


def run(a):
    root=a.root;m=validate(root);fp=native.sha(root/'MANIFEST.json');start=time.perf_counter()
    lock=(root/'.run.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'STOP').exists():raise RuntimeError('STOP is present; explicit resume authorization required')
    source=native.load_upstream(root/'upstream')
    if native.sha(source.root/'GRASPADMM_UPSTREAM.json')!=m['upstream_source_manifest_sha256']:raise RuntimeError('Upstream identity changed')
    identity=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],cmdline=Path('/proc/self/cmdline').read_bytes().replace(b'\0',b' ').decode(),manifest_sha256=fp,started_utc=native.stamp())
    native.write(root/'process.json',identity)
    groups=json.loads((root/'source_shards.json').read_text());jobs=list(jobs_for(groups,m['chunk_size'],a.stage=='pilot'))
    counters=collections.Counter();pending=[];receipts=[]
    for job in jobs:
        path=root/'shards'/job[3]/'receipt.json'
        if path.exists():
            r=json.loads(path.read_text())
            if r['manifest_sha256']!=fp:raise RuntimeError('Result belongs to another campaign')
            for name,h in r['files_sha256'].items():
                if native.sha(path.parent/name)!=h:raise RuntimeError('Committed shard corrupted')
            receipts.append(r)
        else:pending.append(job)
    fields=['samples','mapped','valid_final_states','native_outputs','gsr_passed','projected_closure','joint_quality','torque_passed','exceptions']
    for r in receipts:
        for k in fields:counters[k]+=r[k]
    def status(state,**extra):
        native.write(root/('pilot_status.json' if a.stage=='pilot' else 'status.json'),dict(state=state,pid=os.getpid(),manifest_sha256=fp,workers=m['workers'],
            total=sum(j[2]-j[1] for j in jobs),chunks_total=len(jobs),chunks_completed=len(receipts),**counters,
            elapsed_seconds=time.perf_counter()-start,updated_utc=native.stamp(),**extra))
    status('running',resumed_committed=len(receipts))
    context=mp.get_context('spawn')
    # Bound submissions to twice worker count, instead of enqueuing millions of row jobs.
    from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
    iterator=iter(pending);futures={};stopping=False
    def request_stop(*_):
        nonlocal stopping
        stopping=True;(root/'STOP').touch()
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    with ProcessPoolExecutor(max_workers=m['workers'],mp_context=context,initializer=worker_init,initargs=(str(root),)) as pool:
        while True:
            stopping=stopping or (root/'STOP').exists()
            while not stopping and len(futures)<2*m['workers']:
                try:job=next(iterator)
                except StopIteration:break
                futures[pool.submit(solve_chunk,job)]=job[3]
            if not futures:break
            done,_=wait(futures,timeout=5,return_when=FIRST_COMPLETED)
            for f in done:
                job_id=futures.pop(f)
                try:r=f.result()
                except BaseException:
                    native.write(root/'failure.json',dict(job_id=job_id,exception=traceback.format_exc(),updated_utc=native.stamp()));request_stop();status('failed');raise
                if not r.get('interrupted'):
                    receipts.append(r)
                    for k in fields:counters[k]+=r[k]
            status('stopping' if stopping else 'running')
    state='stopped' if stopping else 'complete'
    status(state)
    if a.stage=='pilot':
        exceptions=[r for r in receipts if r['exceptions']]
        native.write(root/'pilot_receipts.json',receipts)
        if exceptions:raise RuntimeError('Pilot contains implementation/input exceptions; full launch blocked')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',choices=['prepare','pilot','run'],required=True)
    for name in ('source','dataset','production','native-root','mano-root'):p.add_argument('--'+name,type=Path)
    p.add_argument('--workers',type=int,default=16);p.add_argument('--chunk-size',type=int,default=64)
    a=p.parse_args();a.root=a.root.resolve()
    if a.stage=='prepare':prepare(a)
    else:run(a)

if __name__=='__main__':main()
