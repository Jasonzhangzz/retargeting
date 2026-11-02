"""Separate unverified CA proposals, acquired MPPI references and legacy CA."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .forceaware_reference import _read_ca


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scene_files(scene):
    """Hash actual file-backed assets of the generated ordinary CA scene."""
    root = ET.parse(scene).getroot()
    compiler = root.find("compiler")
    attributes = {} if compiler is None else compiler.attrib
    paths = [scene]
    for node in root.findall("./asset/*"):
        if not node.get("file"):
            continue
        name = node.get("file")
        if attributes.get("strippath", "false") == "true":
            name = Path(name).name
        directory = {"mesh": "meshdir", "texture": "texturedir"}.get(node.tag, "assetdir")
        base = attributes.get(directory, attributes.get("assetdir", ""))
        path = (scene.parent/Path(base)/name).resolve()
        if not path.is_file():
            raise ValueError(f"SPIDER input scene asset is missing: {path}")
        paths.append(path)
    return paths


@dataclass
class ContactAwareInput:
    source: Path
    scene: Path
    qpos: np.ndarray
    qvel: np.ndarray
    contact_bodies: np.ndarray
    contact_local: np.ndarray
    object_local: np.ndarray
    active_slots: np.ndarray
    source_hashes: dict
    reference_kind: str = 'accepted_ca'
    required_groups: np.ndarray | None = None
    contact_groups: np.ndarray | None = None
    contact_outward: np.ndarray | None = None
    seed_source: Path | None = None


def read_input(source):
    source=Path(source).resolve()
    if (source/'spider_seed.npz').is_file():
        from .spider_seed import load_seed
        source,metadata,raw,model=load_seed(source)
        scene=source/metadata['scene']
        files=[source/'spider_seed.npz',source/'spider_seed.json',*scene_files(scene)]
        return ContactAwareInput(source,scene,raw['qpos'],np.zeros(32),raw['bodies'],raw['local'],
            np.zeros_like(raw['local']),np.arange(len(raw['bodies'])),{str(p):sha256(p) for p in files},
            reference_kind='ca_proposal',required_groups=raw['required_groups'],seed_source=source)
    if (source/'spider_reference.npz').is_file():
        metadata=__import__('json').loads((source/'spider_reference.json').read_text())
        with np.load(source/'spider_reference.npz',allow_pickle=False) as archive:
            raw={key:archive[key].copy() for key in archive.files}
        if metadata.get('force_closure_validated') is not True:
            raise ValueError('An MPPI reference requires independent force-closure acceptance')
        scene=Path(metadata['scene'])
        files=[source/'spider_reference.npz',source/'spider_reference.json',*scene_files(scene)]
        return ContactAwareInput(source,scene,raw['qpos'],np.zeros(32),raw['bodies'],raw['local'],
            raw['object_local'],np.arange(len(raw['bodies'])),{str(p):sha256(p) for p in files},
            reference_kind='mppi_acquired',required_groups=raw['required_groups'],
            contact_groups=raw['groups'],contact_outward=raw['outward'],seed_source=Path(metadata['seed_source']))
    source, card, report, guidance, poses = _read_ca(source)
    scene = source/"runtime_scene.xml"
    if not scene.is_file():
        scene = source/"sequence/scene/wuji_right/primitive_mesh.xml"
    if not scene.is_file():
        scene = source/"sequence/scene/wuji_right/primitive_sdf.xml"
    if not scene.is_file():
        raise ValueError("SPIDER ContactAware input is missing its ordinary physical scene")
    slots = np.flatnonzero(card["reference_contact_mask"])
    # CA pose convention: quaternion XYZW followed by XYZ.
    pose = poses[0]
    qpos = np.r_[card["hand_reference_qpos"], pose[4:7], pose[3], pose[:3]].astype(np.float64)
    files = [source/"card_input.npz", source/"contactaware_reference.json", *scene_files(scene)]
    directory = source/"contactaware"
    if not directory.is_dir():
        directory = source/"sequence/retarget/wuji_right/contactaware"
    files += list(directory.glob("*.npz")) + list(directory.glob("*.npy"))
    return ContactAwareInput(source, scene, qpos, np.zeros(32),
        guidance["hand_query_body_ids"][0, slots].astype(int),
        guidance["hand_query_local_pos"][0, slots].astype(np.float64),
        guidance["contact_pos_obj"][0, slots].astype(np.float64), slots,
        {str(path): sha256(path) for path in files})


def prepare_model(reference, destination, dt, *, solver_iterations=20, ls_iterations=50):
    """Canonical ordinary mesh scene, with explicit original SPIDER solver budget."""
    source = mujoco.MjModel.from_xml_path(str(reference.scene))
    if (source.nq, source.nv, source.nu) != (33, 32, 26):
        raise ValueError("SPIDER grasp requires the full WUJI2 and one free object")
    source.opt.timestep = dt
    source.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    source.opt.iterations, source.opt.ls_iterations = solver_iterations, ls_iterations
    path = Path(destination)/"model.xml"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Reuse only the precision-preserving ordinary-XML serialization helper;
    # it does not load FA dynamics, replace geometry, or invoke optimization.
    from .forceaware_scene import _canonical_tree
    tree, _, _, _ = _canonical_tree(reference.scene, source, path.parent, dt)
    tree.write(path, encoding="utf-8")
    model = mujoco.MjModel.from_xml_path(str(path))
    for name in ("body_mass", "body_inertia", "body_pos", "body_quat", "actuator_gainprm",
                 "actuator_biasprm", "actuator_ctrlrange", "actuator_forcerange", "geom_friction"):
        np.testing.assert_allclose(getattr(model, name), getattr(source, name), rtol=0, atol=1e-12)
    return model, path


def preload(reference, model_path, horizon):
    """Existing exact-bias static input proposal; no FA optimization is invoked."""
    if reference.reference_kind == 'mppi_acquired':
        return acquired_preload(reference,model_path)
    from .forceaware_preload import prepare_preload
    artifacts = SimpleNamespace(root=reference.source, viewer_xml=model_path)
    cfg = SimpleNamespace(sequence=SimpleNamespace(hand="wuji_right"),
                          optimizer=SimpleNamespace(horizon=horizon))
    minimum = dict(thumb=.1, index=.1, middle=.1, ring=.2, pinky=.3)
    model = mujoco.MjModel.from_xml_path(str(reference.scene))
    from .forceaware_validation import finger_body_labels
    labels = finger_body_labels(model)[reference.contact_bodies]
    if set(labels.tolist()) == set(range(5)):
        command, report = prepare_preload(artifacts, cfg, minimum_per_contact=minimum, minimum_total=4.)
    else:
        # Pinches and palmar grasps use only their accepted CA contact set.
        # Keep the existing FA five-finger default unchanged for its callers.
        levels = np.asarray([.1, .1, .1, .2, .3])
        minima = np.asarray([levels[label] if label >= 0 else .1 for label in labels])
        command, report = prepare_preload(artifacts, cfg, minimum_per_contact=minima,
            minimum_total=4., require_all_fingers=False)
    if command is None:
        raise ValueError("Accepted CA grasp has no feasible bounded static control initialization")
    return command.astype(np.float32), report


def save_acquired_reference(destination,objective,qpos,observation,data):
    """Export an MPPI reference under its own schema, never as accepted CA."""
    from retarget.common.results import write_json
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    model=objective.cpu;obj=model.body('grasp_object').id;R=data.xmat[obj].reshape(3,3)
    bodies=observation['bodies'];local=[]
    for body,point in zip(bodies,observation['hand_points']):
        local.append(data.xmat[body].reshape(3,3).T@(point-data.xpos[body]))
    np.savez_compressed(destination/'spider_reference.npz',qpos=np.r_[qpos,objective.seed['qpos'][26:]],
        bodies=bodies,local=np.asarray(local),object_local=(observation['object_points']-data.xpos[obj])@R,
        outward=-observation['normals']@R,groups=observation['groups'],required_groups=objective.seed['required_groups'])
    write_json(destination/'spider_reference.json',dict(version=1,kind='mppi_acquired',
        force_closure_validated=True,scene=str(objective.source/objective.metadata['scene']),
        seed_source=str(objective.source),ca_success=objective.metadata['ca_success'],
        hand_reference_source='fixed-object MPPI optimized configuration',template_angles_are_reference=False))
    return destination.resolve()


def acquired_preload(reference,model_path):
    """Bounded static-load QP from acquired material points, independent of CA acceptance."""
    from .forceaware_preload import _cpu_load_matrices,_solve_clarabel_load
    model=mujoco.MjModel.from_xml_path(str(model_path));data=mujoco.MjData(model)
    data.qpos[:]=reference.qpos;mujoco.mj_forward(model,data)
    obj=model.body('grasp_object').id;R=data.xmat[obj].reshape(3,3)
    points=[];jac=[]
    for body,local in zip(reference.contact_bodies,reference.contact_local):
        point=data.xpos[body]+data.xmat[body].reshape(3,3)@local
        jp,jr=np.zeros((3,model.nv)),np.zeros((3,model.nv))
        mujoco.mj_jac(model,data,jp,jr,point,int(body));points.append(point);jac.append(jp[:,:26])
    from retarget.env.geometry import ObjectMeshSurface
    radius=max(np.linalg.norm(ObjectMeshSurface.from_model(model,obj).triangles.reshape(-1,3),axis=1).max(),.01)
    physical=(model.geom_contype|model.geom_conaffinity)!=0
    friction=float(model.geom_friction[physical,0].min())
    values=dict(model=model,qref=reference.qpos[:26],outward=reference.contact_outward@R.T,
        arms=np.asarray(points)-data.xpos[obj],jacobian=np.asarray(jac),radius=radius,
        mass=float(model.body_mass[obj]),bias=data.qfrc_bias[:26].copy(),friction=friction)
    counts=np.bincount(reference.contact_groups,minlength=6)
    minima=np.asarray([.5/max(counts[g],1) for g in reference.contact_groups])
    command,report=_solve_clarabel_load(_cpu_load_matrices(values),minima,
        minimum_total=max(values['mass']*9.81*2.,.5*int(reference.required_groups.sum())),iterations=10000)
    if not report['feasible']:raise ValueError('Acquired grasp has no feasible bounded static control initialization')
    return command.astype(np.float32),report


def settle_acquired_grip(reference,model_path,command,destination,*,seconds=.4):
    """Build real contact preload against the fixed object, then release from rest.

    Only the acquisition model loses its object joint. The free-body model is
    never modified, and its entire subsequent 0.4 s trajectory is audited.
    """
    import json
    from types import SimpleNamespace
    from .spider_seed import load_seed
    from .spider_geometry import ContactRequirements,exact_closure
    from retarget.common.results import write_json
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    from retarget.env.geometry import ObjectMeshSurface
    tree=ET.parse(model_path);body=tree.find(".//body[@name='grasp_object']")
    for child in list(body):
        if child.tag in ('joint','freejoint'):body.remove(child)
    body.set('pos',' '.join(map(str,reference.qpos[26:29])))
    body.set('quat',' '.join(map(str,reference.qpos[29:33])))
    fixed_path=destination/'fixed_object.xml';tree.write(fixed_path,encoding='utf-8')
    fixed=mujoco.MjModel.from_xml_path(str(fixed_path));data=mujoco.MjData(fixed)
    data.qpos[:]=reference.qpos[:26];data.qvel[:]=0.;data.ctrl[:]=command
    _,metadata,seed,_=load_seed(reference.seed_source)
    requirements=ContactRequirements(fixed,seed);obj=fixed.body('grasp_object').id
    free=mujoco.MjModel.from_xml_path(str(model_path));probe=mujoco.MjData(free)
    free_requirements=ContactRequirements(free,seed)
    trace=[np.r_[data.qpos,reference.qpos[26:]]];vel=[np.r_[data.qvel,np.zeros(6)]];loads=[]
    best=None
    radius=max(float(np.linalg.norm(ObjectMeshSurface.from_model(fixed,obj).triangles.reshape(-1,3),axis=1).max()),.01)
    for step in range(round(seconds/fixed.opt.timestep)):
        mujoco.mj_step(fixed,data)
        trace.append(np.r_[data.qpos,reference.qpos[26:]]);vel.append(np.r_[data.qvel,np.zeros(6)])
        loads.append(requirements.observe(data,loaded_only=True)['loads'])
        if (step+1)%10==0:
            probe.qpos[:]=np.r_[data.qpos,reference.qpos[26:]];probe.qvel[:]=0.;probe.ctrl[:]=command
            probe.qacc_warmstart[:]=0.;mujoco.mj_forward(free,probe)
            actual=free_requirements.observe(probe,loaded_only=True)
            if (actual['loads'][reference.required_groups]>=.05).all():
                fc=exact_closure(actual,center=probe.xpos[obj],radius=radius)
                score=float(np.linalg.norm(probe.qacc[26:29])+radius*np.linalg.norm(probe.qacc[29:32]))
                if fc['success'] and (best is None or score<best[0]):
                    best=(score,probe.qpos[:26].copy(),step+1,actual,fc)
    observation=requirements.observe(data,loaded_only=True)
    closure=exact_closure(observation,center=data.xpos[obj],radius=radius)
    loaded=bool((observation['loads'][reference.required_groups]>=.05).all())
    accepted=best is not None
    np.savez_compressed(destination/'trajectory.npz',qpos=trace,qvel=vel,ctrl=np.repeat(np.asarray(command)[None],len(loads),0),
        time=np.arange(len(trace))*fixed.opt.timestep,group_normal_force_n=loads)
    result=dict(accepted=accepted,object_fixed_only_during_acquisition=True,seconds=seconds,
        required_groups=reference.required_groups,final_loads_n=observation['loads'],force_closure=closure,
        hand_velocity_before_release=float(np.linalg.norm(data.qvel)),release_velocity_policy='all DOFs start at zero; no settling steps removed from free hold')
    if accepted:
        _,q,step,observation,closure=best
        probe.qpos[:]=np.r_[q,reference.qpos[26:]];probe.qvel[:]=0.;probe.ctrl[:]=command;mujoco.mj_forward(free,probe)
        objective=SimpleNamespace(cpu=free,seed=seed,source=reference.seed_source,metadata=metadata)
        result.update(selected_fixed_step=step,free_initial_acceleration_score=best[0],
            final_loads_n=observation['loads'],force_closure=closure)
        result['reference']=str(save_acquired_reference(destination/'reference',objective,q,observation,probe))
        write_json(destination/'settling.json',result)
        return read_input(result['reference']),np.asarray(command,np.float32),result
    write_json(destination/'settling.json',result)
    return reference,np.asarray(command,np.float32),result
