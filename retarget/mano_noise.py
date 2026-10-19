"""Shared MANO finger perturbation and exact SDF template preparation.

MANO model weights remain external; generated inputs are ordinary prepared
experiments usable by the solver and viewer without MANO dependencies.
"""
import hashlib
import inspect
import json
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import least_squares, linprog
from scipy.spatial.transform import Rotation
import yaml
ROOT=Path(__file__).resolve().parents[1]
FINGERS={'thumb':(12,1,1),'index':(0,2,5),'middle':(3,3,9),'ring':(9,4,13),'pinky':(6,5,17)}


def nearest_sdf(surface,points):
    """Exact triangle-mesh signed-distance nearest points, with safe KD culling."""
    from retarget.contact_exploration import closest_triangles
    points=np.asarray(points,float);flat=points.reshape(-1,3)
    output=np.empty_like(flat);normal=np.empty_like(flat)
    for row,point in enumerate(flat):
        _,seed=surface.triangle_tree.query(point)
        first=closest_triangles(point[None],surface.triangles[[seed]])[0,0]
        upper=np.linalg.norm(first-point)
        ids=np.asarray(surface.triangle_tree.query_ball_point(point,upper+surface.max_triangle_radius+1e-10),int)
        candidates=closest_triangles(point[None],surface.triangles[ids])[0]
        choice=np.argmin(np.sum((candidates-point)**2,axis=1))
        output[row]=candidates[choice];normal[row]=surface.normals[ids[choice]]
    delta=flat-output;distance=np.linalg.norm(delta,axis=1)
    sign=np.where(np.sum(delta*normal,axis=1)>=0.,1.,-1.)
    phi=sign*distance
    gradient=np.where((distance>1e-12)[:,None],sign[:,None]*delta/np.maximum(distance[:,None],1e-12),normal)
    # This is p - phi(p) grad(phi), using the closest-feature SDF gradient.
    projected=flat-phi[:,None]*gradient
    np.testing.assert_allclose(projected,output,atol=1e-10)
    return projected.reshape(points.shape),normal.reshape(points.shape),phi.reshape(points.shape[:-1])


def mano_layer(args):
    import torch
    if not hasattr(inspect,'getargspec'):inspect.getargspec=inspect.getfullargspec
    for name,value in {'int':int,'float':float,'complex':complex,'object':object,'str':str,'unicode':str}.items():
        if name not in vars(np):setattr(np,name,value)
    sys.path.insert(0,str(args.manopth))
    from manopth.manolayer import ManoLayer
    return ManoLayer(flat_hand_mean=False,ncomps=45,side='right',mano_root=str(args.mano_models),use_pca=True)


def decode(layer,pose,shape,world_rotation,world_position):
    import torch
    with torch.no_grad():
        v,j=layer(torch.from_numpy(pose[:,:48].astype(np.float32)),
            torch.from_numpy(shape.astype(np.float32))[None].expand(len(pose),-1),
            torch.from_numpy(pose[:,48:].astype(np.float32)))
    vertices=v.numpy()/1000;joints=j.numpy()/1000
    return (np.einsum('fvi,fji->fvj',vertices,world_rotation)+world_position[:,None],
            np.einsum('fvi,fji->fvj',joints,world_rotation)+world_position[:,None])


def directions(points):
    delta=np.diff(points,axis=0)
    return delta/np.maximum(np.linalg.norm(delta,axis=1,keepdims=True),1e-12)


def palm_frame(wrist, roots):
    """Anatomical frame from the wrist and five MCP/CMC roots, independent of grasp."""
    origin=np.mean(np.r_[wrist[None],roots],axis=0)
    x=roots[1]-roots[4];x/=np.linalg.norm(x)
    forward=np.mean(roots[1:],axis=0)-wrist
    z=np.cross(x,forward);z/=np.linalg.norm(z)
    y=np.cross(z,x)
    return origin,np.stack((x,y,z),axis=-1)


def retarget_from_mano(model, joints):
    """Direct palm + absolute bone-direction IK, with no feasible robot reference.

    This routine deliberately accepts no baseline trajectory, contact targets,
    object poses or masks. The robot neutral geometry defines the palm frame;
    subsequent IK warm starts come only from the preceding generated frame.
    """
    import mujoco
    from time import perf_counter
    started=perf_counter();data=mujoco.MjData(model)
    qids=np.arange(26);joint_ids=np.array([np.flatnonzero(model.jnt_qposadr==c)[0] for c in qids])
    lo,hi=model.jnt_range[joint_ids].T
    neutral=np.clip(model.qpos0[:26],lo+1e-8,hi-1e-8)
    data.qpos[:26]=neutral;mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data)
    chains={};robot_roots=[]
    for name,(_,robot,mano) in FINGERS.items():
        bodies=[model.body(f'right_finger{robot}_link{i}').id for i in (2,3,4)]
        site=model.site(f'right_{name}_tip').id
        chains[name]=(bodies,site,mano,np.arange(6+4*(robot-1),6+4*robot))
        robot_roots.append(data.xpos[bodies[0]].copy())
    robot_origin,robot_basis=palm_frame(np.zeros(3),np.asarray(robot_roots))
    output=np.empty((len(joints),26));errors=[];evaluations=[]
    # TRF can terminate prematurely when all flexion coordinates start within
    # 1e-8 of a lower bound. Use a generic, slightly flexed neutral seed.
    previous=np.clip(neutral,lo+.1,hi-.1)
    for frame,human in enumerate(joints):
        roots=human[[FINGERS[name][2] for name in FINGERS]]
        origin,basis=palm_frame(human[0],roots)
        rotation=basis@robot_basis.T
        base=previous.copy();base[:3]=origin-rotation@robot_origin
        # The XML uses serial X/Y/Z hinge joints: R = Rx @ Ry @ Rz.
        base[3:6]=Rotation.from_matrix(rotation).as_euler('XYZ')
        if frame:base[3:6]+=2*np.pi*np.round((previous[3:6]-base[3:6])/(2*np.pi))
        for name,(bodies,site,mano,cols) in chains.items():
            desired=directions(human[mano:mano+4])
            def evaluate(values):
                q=base.copy();q[cols]=values
                data.qpos[:26]=q;mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data)
                points=np.r_[data.xpos[bodies],data.site_xpos[site][None]].copy();jac=[]
                for body in bodies:
                    jp=np.zeros((3,model.nv));jr=np.zeros_like(jp)
                    mujoco.mj_jacBody(model,data,jp,jr,body);jac.append(jp[:,cols])
                jp=np.zeros((3,model.nv));jr=np.zeros_like(jp)
                mujoco.mj_jacSite(model,data,jp,jr,site);jac.append(jp[:,cols])
                vec=np.diff(points,axis=0);length=np.maximum(np.linalg.norm(vec,axis=1),1e-12);unit=vec/length[:,None]
                derivative=np.einsum('bij,bjk->bik',(np.eye(3)[None]-unit[:,:,None]*unit[:,None,:])/length[:,None,None],np.diff(jac,axis=0))
                return (unit-desired).ravel(),derivative.reshape(-1,4)
            fitted=least_squares(lambda v:evaluate(v)[0],np.clip(previous[cols],lo[cols]+1e-4,hi[cols]-1e-4),
                jac=lambda v:evaluate(v)[1],bounds=(lo[cols],hi[cols]),max_nfev=64,
                ftol=1e-7,xtol=1e-7,gtol=1e-7)
            base[cols]=fitted.x;errors.append(float(np.sqrt(np.mean(fitted.fun**2))));evaluations.append(int(fitted.nfev))
        output[frame]=base;previous=base
    return output,dict(method='MANO_palm_frame_absolute_chain_IK',
        original_wuji_reference_used=False,object_or_contact_objective_used=False,
        initialization='slightly_flexed_model_neutral_then_previous_generated_frame',
        chain_direction_rms=float(np.mean(errors)),chain_direction_rms_max=float(np.max(errors)),
        evaluations=sum(evaluations),seconds=perf_counter()-started)


def task_lp(points,normals,active,force,torque,center):
    from retarget.contact_solvers import contact_basis,grasp_matrix
    p,n=points[active],normals[active]
    if not len(p):return False
    b=contact_basis(n);normal,t1,t2=b[...,0],b[...,1],b[...,2]
    primitive=np.stack((normal+t1,normal-t1,normal+t2,normal-t2),axis=-1)
    G=grasp_matrix(p[None],primitive[None],center)[0]
    result=linprog(np.zeros(G.shape[1]),A_eq=G,b_eq=np.r_[force,torque],
        A_ub=np.kron(np.eye(len(p)),np.ones((1,4))),b_ub=np.full(len(p),10.),bounds=(0,None),method='highs')
    return bool(result.success)


def prepare_inputs(args, *, conditions=None, task_feasibility=True, noise_seed=None):
    import mujoco
    from retarget.contact_exploration import ObjectMeshSurface
    from forceaware.targets import object_reference_velocity
    from forceaware.config import object_xml_stem
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    source=args.source;base=args.experiment
    original={k:v.copy() for k,v in np.load(base/'contactaware/contact_guidance.npz').items()}
    semantic_path=source/'sequence/mano_raw/semantic_contacts.npz'
    if not semantic_path.exists():semantic_path=source/'original_MANO_semantic_contacts.npz'
    semantic=np.load(semantic_path)
    np.testing.assert_allclose(semantic['anchors'],original['contact_pos_obj'],atol=1e-7)
    np.testing.assert_array_equal(semantic['mask'],original['contact_mask'])
    pose=np.load(source/'sequence/mano_raw/hand_pose_51.npy').reshape(-1,51).astype(np.float32)
    shape=np.load(source/'sequence/mano_raw/hand_shape_10.npy')
    camera=np.linalg.inv(np.load(source/'sequence/mano_raw/extrinsics_4x4.npy'))
    raw_object=np.load(source/'sequence/mano_raw/object_pose_camera_7.npy').reshape(-1,7)
    ref_object=np.load(base/'contactaware/object_pose_7.npy');frames=len(pose)
    raw_rotation=camera[:3,:3][None]@Rotation.from_quat(raw_object[:,:4]).as_matrix()
    raw_position=raw_object[:,4:]@camera[:3,:3].T+camera[:3,3]
    ref_rotation=Rotation.from_quat(ref_object[:,:4]).as_matrix()
    correction=ref_rotation@raw_rotation.transpose(0,2,1)
    world_rotation=correction@camera[:3,:3]
    world_position=ref_object[:,4:]+np.einsum('fij,fj->fi',correction,camera[:3,3]-raw_position)
    layer=mano_layer(args);vertices,joints=decode(layer,pose,shape,world_rotation,world_position)
    visual=np.load(base/'demonstration_visual.npz')
    max_decode_error=float(np.max(np.abs(vertices-visual['vertices'])))
    assert max_decode_error<1e-6,('Raw MANO does not match the bundled demonstration',max_decode_error)
    components=layer.th_selected_comps.numpy();mean=layer.th_hands_mean.numpy()[0]
    inverse_components=np.linalg.pinv(components);decoded=pose[:,3:48]@components+mean
    scene=base/'sequence/scene/wuji_right'/f'{object_xml_stem(base/"sequence")}_mesh.xml'
    model=mujoco.MjModel.from_xml_path(str(scene.resolve()))
    obj=model.body('obj').id;surface=ObjectMeshSurface.from_model(model,obj)
    vertex_ids=semantic['mano_vertex'].astype(int)
    original_local=np.einsum('fci,fij->fcj',vertices[:,vertex_ids]-ref_object[:,None,4:],ref_rotation)
    zero_points,zero_normals,zero_phi=nearest_sdf(surface,original_local)
    mask=original['contact_mask']>0
    mass=float(model.body_mass[obj]);center=np.asarray(model.body_ipos[obj])
    twist=object_reference_velocity(ref_object,np.arange(frames),ref_dt=1/30,end_frame=frames-1)
    acceleration=np.gradient(twist,1/30,axis=0)
    # Scene's COM inertia is isotropic. Generalize its inertial rotation too.
    ri=Rotation.from_quat(model.body_iquat[obj][[1,2,3,0]]).as_matrix();inertia=ri@np.diag(model.body_inertia[obj])@ri.T
    forces=np.einsum('fji,fj->fi',ref_rotation,mass*(acceleration[:,:3]-model.opt.gravity))
    omega=np.einsum('fji,fj->fi',ref_rotation,twist[:,3:]);alpha=np.einsum('fji,fj->fi',ref_rotation,acceleration[:,3:])
    torques=alpha@inertia.T+np.cross(omega,omega@inertia.T)
    minimum=np.min(np.einsum('vi,fji->fvj',surface.triangles.reshape(-1,3),ref_rotation)[:,:,2],axis=1)+ref_object[:,6]
    airborne=minimum>.005
    if conditions is None:
        conditions=[('original',None,0.),('projected_zero',None,0.)]
        conditions += [(f'{finger}_{round(amplitude*1000):03d}',finger,amplitude) for finger in args.fingers for amplitude in args.amplitudes]
    manifest=dict(experiment='decoded MANO perturbation -> exact mesh SDF closest-point templates -> same object trajectory',
        raw_source=str(source),example=str(base),mano_decode_max_error_m=max_decode_error,duration=args.duration,
        hand_updates_per_window=1,multistarts=8,search_radius_mm=5.,patterns=args.fingers,amplitudes_rad=args.amplitudes,
        reference_policy='direct perturbed MANO palm+bone IK; neutral initialization; no original Wuji reference',
        perturbation_schedule='full amplitude from first frame',
        contact_policy='original masks, weights, slot/vertex correspondences retained; all projected afresh',
        airborne_frames=np.flatnonzero(airborne).tolist(),conditions=[])
    source_hash={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [ROOT/'retarget/fa/solver.py',ROOT/'retarget/contact_solvers.py',ROOT/'retarget/contact_lm_warp.py',
         ROOT/'retarget/mano_noise.py',ROOT/'data/example/forceaware.yaml']}
    manifest['solver_source_sha256']=source_hash
    for name,pattern,amplitude in conditions:
        case=out/'inputs'/name
        if case.exists():raise FileExistsError(case)
        (case/'contactaware').mkdir(parents=True)
        (case/'sequence').symlink_to(base/'sequence',target_is_directory=True)
        rng=np.random.default_rng(noise_seed) if noise_seed is not None else None
        points=original['contact_pos_obj'].copy();normal=zero_normals
        pert_pose=pose.copy();v,j=vertices,joints;fit={}
        if pattern is not None:
            selected=list(FINGERS) if pattern=='all' else [pattern]
            altered=decoded.copy();ramp=np.ones(frames)
            for finger in selected:
                joint=FINGERS[finger][0]
                for axis in [3*k+1 for k in range(joint,joint+3)]:
                    offset=(1. if decoded[21,axis]>=0 else -1.) if rng is None else rng.uniform(-1.,1.)
                    altered[:,axis]+=ramp*offset*amplitude
            pert_pose[:,3:48]=(altered-mean)@inverse_components
            v,j=decode(layer,pert_pose,shape,world_rotation,world_position)
        qref,fit=retarget_from_mano(model,j)
        if name!='original':
            local=np.einsum('fci,fij->fcj',v[:,vertex_ids]-ref_object[:,None,4:],ref_rotation)
            points,normal,phi=nearest_sdf(surface,local)
        else:phi=zero_phi
        np.testing.assert_array_equal(pert_pose[:,:3],pose[:,:3])
        np.testing.assert_array_equal(pert_pose[:,48:],pose[:,48:])
        guidance={k:value.copy() for k,value in original.items()};guidance['contact_pos_obj']=points.astype(np.float32)
        np.savez_compressed(case/'contactaware/contact_guidance.npz',**guidance)
        for filename in ['object_pose_7.npy','camera_pose_7.npy']:
            (case/'contactaware'/filename).symlink_to(base/'contactaware'/filename)
        (case/'contactaware/config.json').write_text((base/'contactaware/config.json').read_text())
        np.save(case/'contactaware/wuji_right_qpos.npy',qref.astype(np.float32))
        np.savez_compressed(case/'demonstration_visual.npz',vertices=v.astype(np.float32),faces=visual['faces'],
            object_pose=visual['object_pose'],times=visual['times'],frame_ids=visual['frame_ids'])
        np.savez_compressed(case/'mano_input.npz',hand_pose_51=pert_pose,mano_vertex_ids=vertex_ids,
            object_template_normals=normal,mano_contact_distance=phi,mano_joints=j.astype(np.float32))
        config=yaml.safe_load((base/'forceaware.yaml').read_text());(case/'forceaware.yaml').write_text(yaml.safe_dump(config))
        shifts=np.linalg.norm(points-zero_points,axis=-1)[mask]*1000
        allshifts=np.linalg.norm(points-original['contact_pos_obj'],axis=-1)[mask]*1000
        active_frames=np.flatnonzero(airborne & np.any(mask,axis=1))
        if name=='original':
            _,normal,_=nearest_sdf(surface,points)
        feasible=[task_lp(points[f],normal[f],mask[f],forces[f],torques[f],center) for f in active_frames] if task_feasibility else []
        record=dict(name=name,pattern=pattern,amplitude_rad=amplitude,input=str(case),
            max_projection_shift_from_zero_mm=float(shifts.max(initial=0)),mean_projection_shift_from_zero_mm=float(shifts.mean()),
            max_shift_from_original_template_mm=float(allshifts.max(initial=0)),mean_shift_from_original_template_mm=float(allshifts.mean()),
            task_feasible_airborne_fraction=float(np.mean(feasible)) if feasible else None,
            task_feasible_airborne_frames={str(f):bool(ok) for f,ok in zip(active_frames,feasible)},fit=fit,noise_seed=noise_seed)
        manifest['conditions'].append(record)
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print('prepared',json.dumps(record),flush=True)
    np.savez_compressed(out/'reference_task.npz',forces=forces,torques=torques,airborne=airborne,normals_zero=zero_normals)
    return manifest


def resolve_mano_resources(models=None, manopth=None):
    import os
    config_path=Path.home()/'.config/retarget/mano.json'
    resources=json.loads(config_path.read_text()) if config_path.exists() else {}
    models=models or os.environ.get('MANO_MODEL_DIR') or resources.get('mano_models') or ROOT/'assets/mano'
    manopth=manopth or os.environ.get('MANOPTH_DIR') or resources.get('manopth')
    models=Path(models).expanduser().resolve()
    if not (models/'MANO_RIGHT.pkl').is_file():
        raise FileNotFoundError('MANO_RIGHT.pkl missing: set --noise-mano-models DIR or MANO_MODEL_DIR')
    if manopth is None:
        import importlib.util
        if importlib.util.find_spec('manopth') is None:
            raise ModuleNotFoundError('MANO noise needs manopth: install it or set --noise-manopth DIR / MANOPTH_DIR')
        manopth=ROOT
    manopth=Path(manopth).expanduser().resolve()
    return models,manopth


def prepare_run(args):
    """Build one immutable noisy experiment before allocating solver state."""
    from types import SimpleNamespace
    base=args.experiment.resolve()
    models,manopth=resolve_mano_resources(args.noise_mano_models,args.noise_manopth)
    source=(args.noise_source or base).expanduser().resolve()
    case=args.output_base.resolve()/args.executor/args.case_name
    if any((case/name).exists() for name in ('rollout.npz','partial.npz','windows.jsonl','live_state.npz')):
        raise FileExistsError(f'{case} contains a previous run; use a new --case-name')
    output=args.output_base.resolve()/'noise_inputs'/args.case_name
    if output.exists():raise FileExistsError(f'Noise inputs already exist: {output}; use a new --case-name')
    preparation=SimpleNamespace(source=source,experiment=base,output=output,
        manopth=manopth,mano_models=models,duration=args.duration,
        fingers=[args.noise_finger],amplitudes=[args.noise_amplitude])
    manifest=prepare_inputs(preparation,conditions=[('noise',args.noise_finger,args.noise_amplitude)],
        task_feasibility=False,noise_seed=args.noise_seed)
    metadata=dict(enabled=True,source_experiment=str(base),raw_source=str(source),
        finger=args.noise_finger,amplitude_rad=args.noise_amplitude,seed=args.noise_seed,
        hand_reference='direct_MANO_palm_and_absolute_chain_IK',original_wuji_reference_used=False,
        perturbation_schedule='full_from_first_frame',
        mode='flexion_bias' if args.noise_seed is None else 'seeded_uniform_joint_bias',
        projection='exact_mesh_SDF_nearest',prepared_input=manifest['conditions'][0]['input'],
        manifest=str(output/'manifest.json'),mano_models=str(models),manopth=str(manopth))
    (output/'noise.json').write_text(json.dumps(metadata,indent=2)+'\n')
    args.experiment=Path(metadata['prepared_input'])
    args.noise_metadata=metadata
    print(f'[noise] {args.noise_finger} {args.noise_amplitude:g} rad -> {args.experiment}',flush=True)

