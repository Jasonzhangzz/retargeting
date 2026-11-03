"""Reconstruct released Shadow grasps and cache measured MuJoCo contacts by type.

Source poses are never optimized. Closure uses the repository's unchanged
ProjectedComFreeClosure on actual collision contacts, with its default settings.
"""
from __future__ import annotations
import copy,gzip,hashlib,io,json,os,pickle,time,traceback
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
STAGES=('pregrasp','grasp','squeeze')
STORAGE='dexonomy_shadow_snapshot_v1'

class NumpyUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        if module in {'numpy','numpy.core.multiarray','numpy._core.multiarray'} and name in {'_reconstruct','ndarray','dtype','scalar'}:
            return super().find_class(module,name)
        raise ValueError(f'Unsupported pickle class: {module}.{name}')

def load_record(path):
    with Path(path).open('rb') as f:
        version=np.lib.format.read_magic(f)
        (np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0)(f)
        data=NumpyUnpickler(f).load().item()
    if not isinstance(data,dict):raise ValueError('Expected a dictionary record')
    return data

def write_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temporary.replace(path)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def relative_pose(hand,objects):
    from scipy.spatial.transform import Rotation as R
    h=R.from_quat(np.asarray(hand)[3:7],scalar_first=True)
    o=R.from_quat(np.asarray(objects)[3:7],scalar_first=True)
    return np.r_[h.inv().apply(np.asarray(objects)[:3]-np.asarray(hand)[:3]),(h.inv()*o).as_quat(scalar_first=True)]

def resolve_scene(dataset,record,row):
    extracted=Path(dataset)/'extracted'
    name=str(record['scene_path']).replace('\\','/')
    if 'assets/object/' in name:name=name.split('assets/object/',1)[1]
    path=(extracted/name).resolve()
    if not path.is_relative_to(extracted.resolve()):raise ValueError('Invalid scene path')
    scene=copy.deepcopy(load_record(path))
    factor=float(np.asarray(record.get('scene_scale',np.ones(len(record['grasp_qpos'])))).reshape(-1)[row])
    if not np.isfinite(factor) or factor<=0:raise ValueError('Invalid scene_scale')
    for cfg in scene['scene'].values():
        for key,value in list(cfg.items()):
            if key.endswith('_path') and isinstance(value,str):
                target=(path.parent/value).resolve()
                if not target.is_relative_to(extracted.resolve()) or not target.is_file():raise FileNotFoundError(target)
                cfg[key]=str(target)
        if cfg['type']=='rigid_object':
            cfg['scale']=np.asarray(cfg['scale'],float)*factor
            cfg['pose']=np.asarray(cfg['pose'],float).copy();cfg['pose'][:3]*=factor
            info=json.loads(Path(cfg['info_path']).read_text())
            volume=info['mass']/(info['density']*float(info['scale'])**3)
            cfg['density']=.1/(volume*float(np.prod(cfg['scale'])))
    return scene,path,factor

def make_env(dataset,scene):
    import sys
    sys.path.insert(0,str(ROOT/'third_party/Dexonomy'))
    from dexonomy.sim import MuJoCo_EvalEnv,MuJoCo_EvalCfg,HandCfg
    return MuJoCo_EvalEnv(HandCfg(str((Path(dataset)/'shadow_assets/right.xml').resolve()),freejoint=True),
        scene_cfg=scene,sim_cfg=MuJoCo_EvalCfg(miu_coef=[.6,.02]),debug_view=False)

def contacts(model,data,object_body):
    """Contact normal points out of object towards hand; forces act on object.

    MuJoCo stores the midpoint and signed separation. Endpoints are reconstructed
    symmetrically; each hand endpoint is also saved in its actual link frame.
    """
    import mujoco
    object_bodies={object_body}
    for b in range(object_body+1,model.nbody):
        if int(model.body_parentid[b]) in object_bodies:object_bodies.add(b)
    rows=defaultdict(list);self_depth=0.;object_depth=0.
    for cid,c in enumerate(data.contact):
        b1,b2=map(int,model.geom_bodyid[[c.geom1,c.geom2]])
        o1,o2=b1 in object_bodies,b2 in object_bodies
        if o1==o2:
            if not o1 and b1>0 and b2>0 and model.body_mocapid[b1]<0 and model.body_mocapid[b2]<0:self_depth=max(self_depth,-float(c.dist))
            continue
        hand=b2 if o1 else b1
        if hand==0 or model.body_mocapid[hand]>=0:continue
        normal=np.asarray(c.frame[:3]).copy()*(1 if o1 else -1)
        point=np.asarray(c.pos).copy();hp=point+.5*float(c.dist)*normal;op=point-.5*float(c.dist)*normal
        wrench=np.zeros(6)
        mujoco.mj_contactForce(model,data,cid,wrench)
        frame=np.asarray(c.frame).reshape(3,3);sign=-1 if o1 else 1
        rows['points'].append(point);rows['normals'].append(normal);rows['hand_points'].append(hp);rows['object_points'].append(op)
        rows['hand_local_normals'].append(np.asarray(data.xmat[hand]).reshape(3,3).T@normal)
        rows['hand_local_points'].append(np.asarray(data.xmat[hand]).reshape(3,3).T@(hp-data.xpos[hand]))
        rows['object_local_points'].append(np.asarray(data.xmat[object_body]).reshape(3,3).T@(op-data.xpos[object_body]))
        rows['hand_body_ids'].append(hand);rows['distances'].append(float(c.dist));rows['force_on_object_world'].append(sign*(frame.T@wrench[:3]));rows['torque_on_object_world'].append(sign*(frame.T@wrench[3:]))
        rows['hand_geom_ids'].append(int(c.geom2 if o1 else c.geom1));rows['object_geom_ids'].append(int(c.geom1 if o1 else c.geom2))
        object_depth=max(object_depth,-float(c.dist))
    vectors={'points','normals','hand_points','object_points','hand_local_points','hand_local_normals','object_local_points','force_on_object_world','torque_on_object_world'}
    result={k:np.asarray(rows[k],float).reshape(-1,3) for k in vectors}
    result.update({k:np.asarray(rows[k],int if k.endswith('_ids') else float) for k in ('hand_body_ids','hand_geom_ids','object_geom_ids','distances')})
    result['object_penetration_m']=object_depth;result['self_penetration_m']=self_depth
    return result

def closure(model,data,body,observation):
    import torch
    from .projected_comfree import ClosureSettings,ProjectedComFreeClosure
    n=len(observation['points'])
    if n<2:return {'certified':False,'reason':'fewer_than_two_contacts','contacts':n},{}
    # The attached scene root may be massless; aggregate the rigid subtree.
    descendants={body}
    for b in range(body+1,model.nbody):
        if int(model.body_parentid[b]) in descendants:descendants.add(b)
    mass=float(sum(model.body_mass[b] for b in descendants))
    center=sum(model.body_mass[b]*data.xipos[b] for b in descendants)/mass
    inertia=np.zeros((3,3))
    for b in descendants:
        rotation=np.asarray(data.ximat[b]).reshape(3,3);offset=data.xipos[b]-center
        inertia+=rotation@np.diag(model.body_inertia[b])@rotation.T+model.body_mass[b]*(np.dot(offset,offset)*np.eye(3)-np.outer(offset,offset))
    radius=max(float(np.linalg.norm(observation['points']-center,axis=1).max()),.005)
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) not in descendants:continue
        mesh=int(model.geom_dataid[g])
        if mesh<0:continue
        vertices=model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh]+model.mesh_vertnum[mesh]]
        world=vertices@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g]
        radius=max(radius,float(np.linalg.norm(world-center,axis=1).max()))
    solver=ProjectedComFreeClosure(ClosureSettings(cuda_graph=False),center=center,radius=radius,
        object_mass=mass,object_inertia=inertia)
    points=torch.as_tensor(observation['object_points'][None],dtype=torch.float64)
    normals=torch.as_tensor(observation['normals'][None],dtype=torch.float64)
    result=solver.solve(points,normals)
    force,torque=solver.contact_wrenches(result.row_forces,normals)
    report={'certified':bool(result.certified[0]),'valid':bool(result.valid[0]),'converged':bool(result.converged[0]),
        'certified_margin':float(result.certified_margin[0]),'cost':float(result.cost[0]),'contacts':n,
        'scenario_rms':result.scenario_rms[0].tolist(),'closure_model':solver.metadata}
    arrays={'closure_preloads_n':result.preloads[0].numpy(),'closure_row_forces':result.row_forces[0].numpy(),
        'closure_response_forces_n':force[0].numpy(),'closure_response_intrinsic_torques_nm':torque[0].numpy(),
        'closure_scenario_residuals':result.scenario_residuals[0].numpy()}
    return report,arrays

def replay_one(dataset,path,row,destination,case_id,dynamic=True):
    import mujoco
    record=load_record(path);scene,scene_path,factor=resolve_scene(dataset,record,row)
    source=np.stack([np.asarray(record[s+'_qpos'][row],float) for s in STAGES])
    if source.shape!=(3,29) or not np.isfinite(source).all():raise ValueError('Expected finite three-stage Shadow qpos (3,29)')
    if np.any(np.abs(np.linalg.norm(source[:,3:7],axis=1)-1)>.001):raise ValueError('Nonunit source hand quaternion')
    env=make_env(dataset,scene);model,data=env._model,env._data;body=env._obj_active_bodyid
    if len(env.get_hand_qpos())!=29:raise ValueError('Shadow model joint order/dimension mismatch')
    object_cfg=scene['scene'][scene['task']['obj_name']]
    all_contacts=[];states=[];reports=[];certificates={};object_poses=[];relative=[]
    for i,stage in enumerate(STAGES):
        env.reset_qpos(source[i]);mujoco.mj_forward(model,data)
        states.append(data.qpos.copy());pose=env.get_active_obj_pose().copy();object_poses.append(pose);relative.append(relative_pose(source[i],pose))
        obs=contacts(model,data,body);all_contacts.append(obs)
        report,arrays=closure(model,data,body,obs) if i>0 else ({'certified':False,'reason':'pregrasp_not_tested'}, {})
        certificates.update({stage+'_'+k:v for k,v in arrays.items()});reports.append(report)
    joint_names=[model.joint(j).name for j in range(model.njnt) if model.jnt_type[j]!=mujoco.mjtJoint.mjJNT_FREE]
    limit_violation=0.
    for state in states:
        for j in range(model.njnt):
            if model.jnt_limited[j]:
                value=state[model.jnt_qposadr[j]];limit_violation=max(limit_violation,float(model.jnt_range[j,0]-value),float(value-model.jnt_range[j,1]))
    trajectory=[];dynamic_report={'performed':False};dynamic_contacts=[];executed_report=reports[1];executed_state=states[1].copy();executed_pose=object_poses[1].copy();executed_obs=all_contacts[1]
    if dynamic:
        from scipy.spatial.transform import Rotation as R,Slerp
        env.reset_qpos(source[0]);mujoco.mj_forward(model,data)
        for start,end in zip(source[:-1],source[1:]):
            rotations=Slerp([0,1],R.from_quat(np.stack([start[3:7],end[3:7]]),scalar_first=True))
            for alpha in np.linspace(0,1,11)[1:]:
                control=(1-alpha)*start+alpha*end;control[3:7]=rotations([alpha]).as_quat(scalar_first=True)[0]
                env.set_ctrl(control);env.step_sim(5);trajectory.append(data.qpos.copy())
        env.set_ctrl(source[-1]);env.step_sim(50);mujoco.mj_forward(model,data);trajectory.append(data.qpos.copy())
        obs=contacts(model,data,body);dynamic_contacts.append(obs)
        executed_report,executed_arrays=closure(model,data,body,obs)
        certificates.update({'executed_'+k:v for k,v in executed_arrays.items()})
        executed_state=data.qpos.copy();executed_pose=env.get_active_obj_pose().copy();executed_obs=obs
        reference=object_poses[-1];pose=env.get_active_obj_pose().copy();translation=float(np.linalg.norm(pose[:3]-reference[:3]));angle=float((R.from_quat(reference[3:],scalar_first=True).inv()*R.from_quat(pose[3:],scalar_first=True)).magnitude())
        dynamic_report={'performed':True,'translation_m':translation,'rotation_rad':angle,'contact_count':len(obs['points']),
            'hold_within_threshold':bool(translation<.02 and angle<np.deg2rad(15)),'test':'three_stage_execution_and_0.2s_settle_no_gravity',
            'note':'post-execution closure is computed separately; not an official six-axis dynamic success certificate'}
    destination=Path(destination);models=destination/'models';models.mkdir(exist_ok=True,parents=True)
    # MjSpec records resolved absolute source asset paths in this remote cache.
    model_path=models/(case_id+'.xml');model_path.write_text(env._spec.to_xml())
    npz=destination/'cases'/(case_id+'.npz');npz.parent.mkdir(exist_ok=True)
    arrays={'source_stage_hand_qpos':source,'stage_qpos':np.stack(states),'object_pose_world':np.stack(object_poses),
        'object_pose_in_hand':np.stack(relative),'object_scale':np.asarray(object_cfg['scale']),
        'trajectory_qpos':np.asarray(trajectory).reshape(-1,model.nq),'executed_qpos':executed_state,'executed_object_pose_world':executed_pose,'executed_object_pose_in_hand':relative_pose(executed_state[:29],executed_pose),**certificates}
    for stage,obs in zip(STAGES,all_contacts):
        arrays.update({stage+'_'+k:v for k,v in obs.items() if isinstance(v,np.ndarray)})
    if dynamic_contacts:arrays.update({'dynamic_final_'+k:v for k,v in dynamic_contacts[0].items() if isinstance(v,np.ndarray)})
    np.savez_compressed(npz,**arrays)
    template={'body_names':[model.body(int(b)).name for b in executed_obs['hand_body_ids']],
        'local_points':executed_obs['hand_local_points'].tolist(),
        'local_normals':(-executed_obs['hand_local_normals']).tolist(),
        'finger_qpos':executed_state[7:29].tolist(),'required_cbody':[],
        'provenance':{'source':'MuJoCo collision at actual replayed pose','not_original_contact_labels':True}}
    source_limit_violation=limit_violation;limit_violation=0.
    for j in range(model.njnt):
        if model.jnt_limited[j]:
            value=executed_state[model.jnt_qposadr[j]];limit_violation=max(limit_violation,float(model.jnt_range[j,0]-value),float(value-model.jnt_range[j,1]))
    depths=[executed_obs['object_penetration_m'],executed_obs['self_penetration_m']]
    metric={'hand':'shadow','method':'dexonomy_replay','task_success':executed_report['certified'],
        'success_criterion':'repository_projected_comfree_replayed_actual_contacts','geometry_diagnostics':{
            'projected_closure':executed_report['certified'],'max_hand_object_penetration_m':depths[0],
            'max_hand_hand_penetration_m':depths[1],'joint_limit_violation_rad':limit_violation,'source_joint_limit_violation_rad':source_limit_violation,
            'contact_count':len(executed_obs['points'])},'stage_closure':dict(zip(STAGES,reports)),'executed_closure':executed_report,
        'dynamic':dynamic_report,'source_scene':str(scene_path),'scene_scale':factor,'joint_names':joint_names,
        'object_pose_convention':'local-to-world xyz+wxyz; relative pose maps object frame into hand root frame',
        'contact_force_convention':'on object in world; static forces are constraint reactions at prescribed qpos, closure witnesses are separate arrays'}
    write_json(destination/'cases'/(case_id+'.json'),metric)
    return metric,template

def select_pilot(source,total=100,seed=7):
    """Spread deterministic selection across types, objects and available sizes."""
    source=Path(source);types=sorted(p for p in source.iterdir() if p.is_dir())
    result=[]
    for ti,typ in enumerate(types):
        quota=total//len(types)+(ti<total%len(types));objects=sorted(p for p in typ.iterdir() if p.is_dir())
        # Deterministic hash order avoids filesystem/archive order bias.
        objects.sort(key=lambda p:hashlib.sha256(f'{seed}/{typ.name}/{p.name}'.encode()).digest())
        eligible=[]
        for obj in objects:
            files=sorted(obj.rglob('*.npy'))
            if files:eligible.append(files)
            if len(eligible)>=max(1,(quota+1)//2):break
        for k in range(quota):
            files=eligible[k%len(eligible)];levels=max(1,(quota+len(eligible)-1)//len(eligible));level=k//len(eligible);path=files[round(level*(len(files)-1)/max(1,levels-1))]
            n=len(load_record(path)['grasp_qpos']);row=int.from_bytes(hashlib.sha256(f'{seed}/{path}/{k}'.encode()).digest()[:4],'little')%n
            result.append({'type':typ.name,'path':str(path),'row':row})
    return result

def pack_type(folder,entries):
    folder=Path(folder);poses=defaultdict(list);templates=[];metrics=[];cases=[]
    count=max([len(t['body_names']) for _,t,_ in entries]+[0])
    for i,(metric,template,item) in enumerate(entries):
        case=item['case'];path=folder/'cases'/(case+'.npz')
        with np.load(path,allow_pickle=False) as d:
            q=d['executed_qpos'];poses['qpos'].append(q);poses['grasp_qpos'].append(q[:29]);poses['source_grasp_qpos'].append(d['source_stage_hand_qpos'][1]);poses['pregrasp_qpos'].append(d['source_stage_hand_qpos'][0]);poses['squeeze_qpos'].append(d['source_stage_hand_qpos'][2]);poses['object_pose_world'].append(d['executed_object_pose_world']);poses['object_pose_in_hand'].append(d['executed_object_pose_in_hand']);poses['object_scale'].append(d['object_scale']);prefix='dynamic_final_' if metric['dynamic']['performed'] else 'grasp_';points=d[prefix+'object_points'];normals=d[prefix+'normals'];pad=np.zeros((count,3));pad[:len(points)]=points;poses['projected_points'].append(pad);pad=np.zeros((count,3));pad[:len(points)]=normals;poses['contact_normals'].append(pad);poses['contact_counts'].append(len(points))
        poses['case_names'].append(case);templates.append(template);metrics.append(metric)
        cases.append({'name':case,'task':'grasp','hand':'shadow','method':'dexonomy_replay','template':item['type'],
            'grasp_type':item['type'],'cached_case':i,'template_id':i,'model':'models/'+case+'.xml','trajectory':'poses.npz',
            'contact_data':'cases/'+case+'.npz','summary':'metrics.json.gz','task_success':metric['task_success'],
            'success_criterion':metric['success_criterion'],'object':{'source_record':item['path'],'source_row':item['row']}})
    np.savez_compressed(folder/'poses.npz',**{k:np.asarray(v) for k,v in poses.items()})
    for name,value in [('templates.json.gz',templates),('metrics.json.gz',metrics)]:
        (folder/name).write_bytes(gzip.compress(json.dumps(value,allow_nan=False).encode(),mtime=0))
    write_json(folder/'manifest.json',{'version':3,'storage':STORAGE,'task':'grasp','hand':'shadow','cases':cases})
    write_json(folder/'summary.json',{'cases':len(cases),'closure_passed':sum(m['task_success'] for m in metrics),
        'source_poses_preserved':True,'original_contact_labels_available':False,'contacts_recomputed':True})
    write_json(folder/'checksums.json',{str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file() and p.name not in {'checksums.json','verification.json'}})

def _pilot_job(args):
    import torch
    torch.set_num_threads(1)
    dataset,root,item,dynamic=args
    folder=Path(root)/item['type'];folder.mkdir(exist_ok=True)
    try:
        metric,template=replay_one(dataset,item['path'],item['row'],folder,item['case'],dynamic)
        return metric,template,item,None
    except Exception as error:
        return None,None,item,{'error':str(error),'traceback':traceback.format_exc()}


def run_pilot(dataset,destination,total=100,seed=7,dynamic=True,workers=8):
    import torch
    torch.set_num_threads(1)
    root=Path(destination);root.mkdir(exist_ok=True,parents=True)
    write_json(root/'provenance.json',{'dataset_manifest_sha256':sha(Path(dataset)/'download_manifest.json'),'hand_assets':json.loads((Path(dataset)/'shadow_assets/provenance.json').read_text()),'source':str(Path(dataset).resolve()),'object_mass_kg':.1,'extra_scene_scale_applied':True,'pose_optimization_performed':False})
    source=Path(dataset)/'extracted/Dexonomy_GRASP_shadow/succ_collect';plan=select_pilot(source,total,seed)
    write_json(root/'plan.json',plan);groups=defaultdict(list);failures=[];start=time.time()
    import concurrent.futures,multiprocessing
    jobs=[]
    for i,item in enumerate(plan):
        case=f'{i:06d}_{Path(item["path"]).parent.parent.name}_{Path(item["path"]).stem}_{item["row"]}'
        jobs.append((str(dataset),str(root),{**item,'case':case},dynamic))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for i,(metric,template,item,error) in enumerate(pool.map(_pilot_job,jobs)):
            if error:
                failures.append({**item,**error});print(error['traceback'],flush=True)
            else:
                groups[item['type']].append((metric,template,item))
                print(json.dumps({'finished':i+1,'total':total,'type':item['type'],'closure':metric['task_success'],'contacts':metric['geometry_diagnostics']['contact_count']}),flush=True)
            write_json(root/'status.json',{'state':'running','planned':len(plan),'finished':i+1,'errors':len(failures),'elapsed_seconds':time.time()-start})
    for typ,entries in groups.items():pack_type(root/typ,entries)
    passed=sum(m['task_success'] for entries in groups.values() for m,_,_ in entries);replayed=sum(map(len,groups.values()))
    report={'planned':len(plan),'replayed':replayed,'errors':len(failures),'closure_passed':passed,
        'closure_pass_rate':passed/len(plan),'grasp_types':len(groups),'seed':seed,'elapsed_seconds':time.time()-start,
        'full_batch_threshold':.90,'eligible_for_full_batch':passed/len(plan)>=.90 and not failures,
        'closure_method':'unchanged repository ProjectedComFreeClosure default settings on actual contacts after three-stage execution and settling',
        'by_type':{typ:{'samples':len(entries),'passed':sum(m['task_success'] for m,_,_ in entries)} for typ,entries in groups.items()}}
    write_json(root/'index.json',{'storage':'dexonomy_shadow_grasp_type_index_v1','groups':[{'grasp_type':typ,'manifest':typ+'/manifest.json','cases':len(entries)} for typ,entries in groups.items()]})
    write_json(root/'failures.json',failures);write_json(root/'summary.json',report);write_json(root/'status.json',{'state':'completed',**report})
    print(json.dumps(report,indent=2),flush=True)
    return report

def verify_cache(folder):
    """Audit checksums, source preservation, FK anchors and closure witnesses."""
    import mujoco
    root=Path(folder);manifest=json.loads((root/'manifest.json').read_text());checks=json.loads((root/'checksums.json').read_text())
    for path,expected in checks.items():
        if sha(root/path)!=expected:raise ValueError('Checksum mismatch: '+path)
    with np.load(root/'poses.npz',allow_pickle=False) as raw:poses={k:raw[k] for k in raw.files}
    templates=json.loads(gzip.decompress((root/'templates.json.gz').read_bytes()));metrics=json.loads(gzip.decompress((root/'metrics.json.gz').read_bytes()))
    errors=[];max_fk=0.;max_force_balance=0.
    for i,entry in enumerate(manifest['cases']):
        source=load_record(entry['object']['source_record']);row=entry['object']['source_row']
        if not np.array_equal(poses['source_grasp_qpos'][i],source['grasp_qpos'][row]):raise ValueError('Source pose changed')
        model=mujoco.MjModel.from_xml_path(str(root/entry['model']));data=mujoco.MjData(model);data.qpos[:]=poses['qpos'][i];mujoco.mj_forward(model,data)
        template=templates[i];count=int(poses['contact_counts'][i]);ids=[model.body(n).id for n in template['body_names']]
        if count:
            points=data.xpos[ids]+np.einsum('nij,nj->ni',data.xmat[ids].reshape(-1,3,3),np.asarray(template['local_points']))
            with np.load(root/entry['contact_data'],allow_pickle=False) as contact:
                prefix='dynamic_final_' if metrics[i]['dynamic']['performed'] else 'grasp_';err=float(np.max(np.abs(points-contact[prefix+'hand_points'])));max_fk=max(max_fk,err)
                if err>1e-7:raise ValueError('Hand contact frame mismatch')
                cert=metrics[i]['executed_closure']
                if 'closure_model' in cert:
                    cm=cert['closure_model'];cprefix='executed_' if metrics[i]['dynamic']['performed'] else 'grasp_';force=contact[cprefix+'closure_response_forces_n'];torque=contact[cprefix+'closure_response_intrinsic_torques_nm'];p=contact[prefix+'object_points'];center=np.asarray(cm['center']);radius=cm['radius_m'];test=cm['test_force_n']
                    wrench=np.concatenate((force.sum(1),(np.cross(p[None]-center,force)+torque).sum(1)/radius),axis=-1)
                    loads=np.concatenate((np.zeros((1,6)),np.eye(6),-np.eye(6)))*test
                    residual=(wrench+loads)/test
                    saved=contact[cprefix+'closure_scenario_residuals']
                    # Solver scenario residuals contain additional regularizers;
                    # compare the six physical wrench components explicitly.
                    physical=np.linalg.norm(residual[1:],axis=-1)
                    if cert['certified'] and float(physical.max())>cm['certificate_tolerance']+1e-6:raise ValueError('Invalid closure force witness')
                    max_force_balance=max(max_force_balance,float(physical.max()))
    report={'success':True,'cases':len(manifest['cases']),'max_contact_fk_error_m':max_fk,'max_probe_wrench_residual':max_force_balance,'source_poses_preserved':True,'checksum_files':len(checks)}
    write_json(root/'verification.json',report);return report


def full_type_job(args):
    import torch
    torch.set_num_threads(1)
    dataset,destination,typ,shard_size,dynamic=args
    source=Path(dataset)/'extracted/Dexonomy_GRASP_shadow/succ_collect'/typ
    folder=Path(destination)/typ;folder.mkdir(parents=True,exist_ok=True)
    chunk=[];part=0;total=0;errors=0;passed=0
    def flush(chunk,part):
        nonlocal errors,passed
        out=folder/'shards'/f'{part:06d}';out.mkdir(parents=True,exist_ok=True)
        if (out/'verification.json').exists():
            v=json.loads((out/'verification.json').read_text())
            if v.get('success'):
                summary=json.loads((out/'summary.json').read_text());passed+=summary['closure_passed'];errors+=len(json.loads((out/'failures.json').read_text()));return
        entries=[];failures=[]
        for item in chunk:
            case=item['case']
            try:
                metric,template=replay_one(dataset,item['path'],item['row'],out,case,dynamic)
                entries.append((metric,template,item));passed+=bool(metric['task_success'])
            except Exception as e:failures.append({**item,'error':str(e),'traceback':traceback.format_exc()})
        errors+=len(failures);write_json(out/'failures.json',failures)
        if entries:pack_type(out,entries);verify_cache(out)
        write_json(folder/'status.json',{'state':'running','shards_completed':part+1,'attempted':total,'errors':errors,'closure_passed':passed})
    for path in sorted(source.rglob('*.npy')):
        n=len(load_record(path)['grasp_qpos'])
        for row in range(n):
            chunk.append({'type':typ,'path':str(path),'row':row,'case':f'{total:09d}_{path.parent.parent.name}_{path.stem}_{row}'})
            total+=1
            if len(chunk)>=shard_size:flush(chunk,part);chunk=[];part+=1
    if chunk:flush(chunk,part);part+=1
    result={'type':typ,'state':'completed','attempted':total,'errors':errors,'closure_passed':passed,'shards':part}
    write_json(folder/'status.json',result);return result


def run_full(dataset,destination,workers=8,shard_size=256,dynamic=True):
    import concurrent.futures
    source=Path(dataset)/'extracted/Dexonomy_GRASP_shadow/succ_collect';root=Path(destination);root.mkdir(parents=True,exist_ok=True)
    types=sorted(p.name for p in source.iterdir() if p.is_dir());results=[]
    write_json(root/'status.json',{'state':'running','grasp_types':types,'workers':workers,'shard_size':shard_size})
    write_json(root/'provenance.json',{'dataset_manifest_sha256':sha(Path(dataset)/'download_manifest.json'),'hand_assets':json.loads((Path(dataset)/'shadow_assets/provenance.json').read_text()),'code_sha256':sha(Path(__file__)),'closure_source_sha256':sha(Path(__file__).with_name('projected_comfree.py')),'source_poses_preserved_in_separate_arrays':True,'passed_and_failed_closure_cases_retained':True,'nominal_object_mass_kg':.1})
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        jobs=[pool.submit(full_type_job,(str(dataset),str(destination),typ,shard_size,dynamic)) for typ in types]
        for job in concurrent.futures.as_completed(jobs):
            results.append(job.result());write_json(root/'progress.json',results)
    summary={'state':'completed','by_type':results,'attempted':sum(r['attempted'] for r in results),'errors':sum(r['errors'] for r in results)}
    write_json(root/'summary.json',summary);write_json(root/'status.json',summary)
