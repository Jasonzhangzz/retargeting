"""Frozen HO3D grasp windows and common MJWarp evaluation for two solvers."""
from pathlib import Path
import json,hashlib,time,copy
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from .scene import build_scene,fit_kinematic_reference
from .execution import GPUExecution
from .runner import find_object_mesh,metrics


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_motion(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def select_cases(motions,mano,output,frames=64,seed=42,objects=None):
    """Selection uses source contact labels, never either solver's outcomes."""
    from retarget.task.ho3d.source import vertex_finger_owners
    root=Path(motions);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((root/'manifest.json').read_text());owner=vertex_finger_owners(str(mano))
    candidates={}
    for entry in manifest['segments']:
        if entry['frames']<frames or (objects and entry['object'] not in objects):continue
        path=root/Path(entry['path']).name
        with np.load(path,allow_pickle=False) as z:
            c=z['vertex_contact'];v=z['vertex_contact_valid']
            active=np.stack([c[:,owner==f].sum(1)>=3 for f in range(5)],axis=1)
            phase=(active.sum(1)>=2)&v
        valid=np.flatnonzero(np.convolve(phase.astype(int),np.ones(frames,dtype=int),'valid')==frames)
        if len(valid):candidates.setdefault(entry['object'],[]).append((path,entry,int(valid[0])))
    rng=np.random.default_rng(seed);cases=[]
    for name in sorted(candidates):
        pool=candidates[name];path,entry,start=pool[int(rng.integers(len(pool)))]
        case=out/name;case.mkdir(exist_ok=True);motion=load_motion(path);length=len(motion['frame_ids'])
        cut={k:(v[start:start+frames] if v.ndim and len(v)==length and k!='mano_faces' else v) for k,v in motion.items()}
        # Avoid shape heuristics for static mesh faces and metadata; frame data
        # are all explicitly recoverable in the source manifest/schema.
        cut['grasp_phase']=np.ones(frames,np.float32);cut['time_s']=np.arange(frames)/float(cut['fps'])
        np.savez_compressed(case/'motion.npz',**cut)
        cases.append(dict(object=name,path=str(case.resolve()),source=str(path.resolve()),source_sha256=digest(path),start_index=start,start_frame=int(cut['frame_ids'][0]),frames=frames))
    result=dict(selection_seed=seed,frames_per_case=frames,selection='One segment per object, uniformly from source-labelled contiguous grasp candidates; earliest qualifying window in selected segment',cases=cases)
    (out/'manifest.json').write_text(json.dumps(result,indent=2));return result


def prepare_common(case,models):
    case=Path(case);motion=load_motion(case/'motion.npz');mesh=find_object_mesh(models,str(motion['object_name']))
    model,xml=build_scene(mesh,timestep=1/300)
    (case/'scene.xml').write_text(xml)
    start=time.perf_counter();hand,info=fit_kinematic_reference(model,motion['mano_joints'])
    reference=np.c_[hand,motion['object_pose']];np.save(case/'reference.npy',reference)
    execution=GPUExecution(model);initial=execution.establish(hand[0],motion['object_pose'][0],12)
    np.savez(case/'initial.npz',qpos=initial.qpos,qvel=initial.qvel,ctrl=initial.ctrl,qacc_warmstart=initial.qacc_warmstart,time=np.array(initial.time))
    (case/'preparation.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,ik=info,mesh=str(mesh),mesh_sha256=digest(mesh),initial_state_sha256=digest(case/'initial.npz'),initialization='GPU IK + 12-frame MJWarp pinned-object closure; identical state for both methods'),indent=2))
    prepare_dex_scene(case,xml,mesh)


def prepare_dex_scene(case,xml,mesh):
    """Native DexForge source-Gaussian/mesh-target scene, same joint ordering.

    The independent evaluator retains the original native mesh collision scene.
    """
    from .scene import hand_tree
    case=Path(case);seq=case/'sequence';scene=seq/'scene/wuji2';scene.mkdir(parents=True,exist_ok=True)
    raw=seq/'mano_raw';raw.mkdir(exist_ok=True)
    motion=load_motion(case/'motion.npz');name=str(motion['object_name']).split('_',1)[1]
    (raw/'meta.json').write_text(json.dumps(dict(object=name,camera=dict(fovy=60.),source='ho3d_mano_raw',dataset='HO3D',hand_side='right')))
    obj=motion['object_pose'];xyzw=np.c_[obj[:,[4,5,6,3]],obj[:,:3]]
    np.save(raw/'camera_pose_7.npy',np.repeat(np.array([[0,0,0,1,0,-.5,.5]]),len(obj),axis=0))
    tree=ET.fromstring(xml);tree.find('option').set('cone','pyramidal');tree.find('option').set('integrator','Euler');flag=tree.find('./option/flag')
    if flag is not None:tree.find('option').remove(flag)
    tree.find(".//body[@name='tracked_object']").set('name','obj')
    tree.find(".//freejoint[@name='tracked_object_joint']").set('name','obj_joint')
    body=tree.find(".//body[@name='obj']");geom=body.find('geom')
    # Compile native mesh mass properties before replacing the collision target.
    import mujoco
    model=mujoco.MjModel.from_xml_string(xml);bid=model.body('tracked_object').id
    ET.SubElement(body,'inertial',pos=' '.join(map(str,model.body_ipos[bid])),quat=' '.join(map(str,model.body_iquat[bid])),mass=str(model.body_mass[bid]),diaginertia=' '.join(map(str,model.body_inertia[bid])))
    geom.set('contype','0');geom.set('conaffinity','0');geom.set('mass','0')
    ET.SubElement(body,'geom',name='obj_gs',type='sdfmesh',file=str(Path(mesh).resolve()),contype='4',conaffinity='1',condim='3',friction='.8 .005 .0001')
    assets=case/'wuji_adapter';assets.mkdir(exist_ok=True);qp=assets/'qp';qp.mkdir(exist_ok=True)
    hand=hand_tree('wuji2');finger_map=[3,0,1,2,5]
    for b in tree.findall('.//body'):
        bname=b.get('name','')
        if not (bname=='right_palm_link' or (bname.startswith('right_finger') and bname[-1] in '1234')):continue
        source=Path('data/grasp/assets/wuji2/qp_right')/(bname+'.npz')
        if not source.exists():continue
        with np.load(source) as z:
            # Deterministic spatial subsampling, at most 64 spheres per rigid body.
            ids=np.linspace(0,len(z['local_pos'])-1,min(64,len(z['local_pos']))).astype(int)
            pos=z['local_pos'][ids];normal=z['local_normal'][ids];radius=z['radius'][ids]
        # The repository QP files are surface queries with ZERO radius, not
        # simulator sphere clouds. Give collision spheres a positive 2 mm
        # interior radius while preserving the sampled outer surface points.
        radius=np.maximum(radius,.002)
        f=finger_map[int(bname[len('right_finger')])-1] if bname.startswith('right_finger') else 4
        link=int(bname[-1]) if f!=4 else 0
        np.savez(qp/(bname+'.npz'),local_pos=pos,local_normal=normal,finger_idx=np.full(len(pos),f,np.int32),link_id=np.full(len(pos),link,np.int32))
        cloud=assets/(bname+'.npz');np.savez(cloud,local_pos=pos-normal*radius[:,None],radius=radius,local_normal=normal)
        for child in b.findall('geom'):child.set('contype','0');child.set('conaffinity','0')
        ET.SubElement(b,'geom',name='gs_'+bname,type='gs',file=str(cloud.resolve()),contype='1',conaffinity='4',friction='.8 .005 .0001',condim='3')
    for g in tree.findall('.//geom'):
        if g.get('type')!='gs' and g.get('name')!='obj_gs':g.set('contype','0');g.set('conaffinity','0')
    (scene/(name+'_sdf.xml')).write_text(ET.tostring(tree,encoding='unicode'))
    # CA's kinematic hand retains exact native geometry, labelled for capsule extraction.
    import re
    names=['thumb','index','middle','ring','pinky']
    for i,b in enumerate(hand.findall('.//body')):
        bname=b.get('name','');match=re.match(r'right_finger([1-5])_',bname)
        for j,g in enumerate(b.findall('geom')):
            if match and g.get('contype','1')!='0':g.set('name',f'collision_{names[int(match[1])-1]}_{i}_{j}')
    (assets/'hand.xml').write_text(ET.tostring(hand,encoding='unicode'))
    # mesh scene is exclusively for rendering; the benchmark evaluator uses scene.xml.
    visual=ET.fromstring(xml);visual.find(".//body[@name='tracked_object']").set('name','obj');visual.find(".//freejoint").set('name','obj_joint')
    (scene/(name+'_mesh.xml')).write_text(ET.tostring(visual,encoding='unicode'))
    np.save(case/'object_xyzw.npy',xyzw)


def restore_initial(model,case):
    import mujoco
    data=mujoco.MjData(model)
    with np.load(Path(case)/'initial.npz') as z:
        for k in ('qpos','qvel','ctrl','qacc_warmstart'):getattr(data,k)[:]=z[k]
        data.time=float(z['time'])
    return data


def dense_reference(reference,substeps=10,padding=200):
    x=np.arange((len(reference)-1)*substeps+1)/substeps
    dense=np.stack([np.interp(x,np.arange(len(reference)),col) for col in reference.T],1)
    dense[:,29:33]=Slerp(np.arange(len(reference)),Rotation.from_quat(reference[:,29:33],scalar_first=True))(x).as_quat(scalar_first=True)
    return np.r_[dense,np.repeat(dense[-1:],padding,axis=0)]
