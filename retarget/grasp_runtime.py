"""Build static Dexonomy inputs for CARD CEM or the shared AdamW solver."""
from dataclasses import replace
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np

from retarget.dexonomy import ROOT


def prepare_experiment(directory, hand, guess, args):
    """The only task-specific transformation: static references and a new mesh.

    The object is free, as required by the shared physical solver, with a
    stationary target and zero initial velocity. No floor or lift trajectory.
    Mesh dimensions/recentering are already baked into hand.surface.
    """
    import trimesh
    import yaml
    config = yaml.safe_load(args.solver_config.read_text())
    is_card = getattr(args, 'grasp_optimizer', 'adamw') == 'cem'
    directory = Path(directory)
    scene = directory/'sequence/scene/wuji_right'
    metadata = directory/'sequence/mano_raw'
    guidance = directory/'contactaware'
    for path in (scene, metadata, guidance):
        path.mkdir(parents=True, exist_ok=True)
    mesh_path = directory/'object.obj'
    vertices, faces = np.unique(hand.surface.triangles.reshape(-1, 3), axis=0, return_inverse=True)
    trimesh.Trimesh(vertices=vertices, faces=faces.reshape(-1, 3), process=False).export(mesh_path)
    from retarget.grasp_hand import hand_tree, model_hand
    tree = hand_tree(model_hand(hand.model))
    tree.find('option').set('gravity', f'0 0 {-getattr(args,"planning_gravity",9.81)}')
    ET.SubElement(tree.find('asset'), 'mesh', name='grasp_object_mesh', file=str(mesh_path.resolve()))
    obj = ET.SubElement(tree.find('worldbody'), 'body', name='grasp_object')
    radius = float(np.linalg.norm(vertices, axis=1).max())
    mass = args.object_mass
    ET.SubElement(obj, 'inertial', pos='0 0 0', mass=str(mass),
                  diaginertia=' '.join([str(max(.4*mass*radius**2, 1e-7))]*3))
    ET.SubElement(obj, 'joint', name='obj_joint', type='free', armature='.0001')
    ET.SubElement(obj, 'geom', name='object_visual', type='mesh', mesh='grasp_object_mesh',
                  contype='0', conaffinity='0', mass='0', rgba='.25 .65 .85 1', group='1')
    ET.SubElement(obj, 'geom', name='object_collision', type='sdfmesh', file=str(mesh_path.resolve()),
                  contype='4', conaffinity='1', condim='3', priority='1',
                  friction=f'{args.friction} .005 .0001', mass='0')
    from retarget.grasp_physics import apply_object_physics,reference_physics
    values = dict(reference_physics(str(getattr(args,'physics_reference',
        ROOT/'data/example/sequence/scene/wuji_right/sugar_box_sdf.xml'))))
    values.update(friction=[args.friction,*values['friction'][1:]],
                  impratio=getattr(args,'impratio',values['impratio']),
                  solref=getattr(args,'contact_solref',values['solref']),
                  solimp=getattr(args,'contact_solimp',values['solimp']))
    apply_object_physics(tree,values)
    # The customized collision backend supplies both contact Jacobians and
    # native mesh SDFs to the common objective; no CPU mesh-distance fallback.
    (scene/'primitive_sdf.xml').write_text(ET.tostring(tree, encoding='unicode'))
    (metadata/'meta.json').write_text(json.dumps(dict(object='primitive')))
    reference = np.tile(guess.hand_qpos,(2,1))
    if is_card:
        from retarget.grasp_cem import force_priorities
        supplied = dict(object_points=guess.object_points, object_normals=guess.object_normals)
        if hasattr(hand, 'contact_reference_normals'):
            supplied['region_reference_normals'] = hand.contact_reference_normals
        forces = getattr(guess, 'contact_forces', None)
        if forces is not None:
            force_priorities(forces, len(guess.object_points))
            supplied['contact_forces'] = forces
        np.savez_compressed(directory/'card_input.npz', **supplied)
    load = None if is_card else guess.metadata.get('load_conditioning')
    if not is_card and getattr(args,'load_conditioning',False) and load is None:
        if hasattr(args, 'gpu_nodes') and not getattr(args, 'legacy_serial', False):
            from retarget.grasp_load import load_commands_batch
            commands, loads = load_commands_batch(hand.model, hand.bodies, hand.local,
                guess.hand_qpos[None], guess.object_points[None], guess.object_normals[None],
                device=args.init_device, mass=args.object_mass, friction=args.friction, radius=radius)
            command, load = commands[0], loads[0]
        else:
            from retarget.grasp_load import load_command
            command,load = load_command(hand.model,hand.bodies,hand.local,guess.hand_qpos,
                guess.object_points,guess.object_normals,device=args.init_device,
                mass=args.object_mass,friction=args.friction,radius=radius)
        load['command'] = command.tolist()
    conditioned = bool(not is_card and getattr(args,'load_conditioning',False) and load and load['feasible'])
    if load is not None:
        (directory/'load_conditioning.json').write_text(json.dumps(load,indent=2))
    if conditioned:
        command = np.asarray(load['command'])
        np.savez_compressed(directory/'load_input.npz',command=command,
            gravity_command=load['gravity_command'],
            warm=np.r_[np.zeros(26),np.tile(command-guess.hand_qpos,config['optimizer']['horizon'])])
    depth = getattr(args,'closing_depth_mm',0.)
    if is_card:
        depth = 0.
    if depth is None:
        depth = max(0.,-config['contact']['target_phi']*1000)
    if depth>0 and not conditioned:
        from retarget.grasp_reference import closing_posture,closing_sequence
        closed,info = closing_posture(hand.model,hand.bodies,hand.local,hand.normals,
            guess.hand_qpos,guess.object_points,guess.object_normals,depth=depth/1000,
            device=args.init_device)
        reference = closing_sequence(guess.hand_qpos,closed,duration=args.closing_seconds,
                                     ref_dt=config['simulator']['ref_dt'])
        (directory/'closing_reference.json').write_text(json.dumps(dict(info,
            initial_hand_qpos=guess.hand_qpos.tolist(),closed_hand_qpos=closed.tolist(),
            duration_seconds=args.closing_seconds,object_contacts_modified=False),indent=2))
    frames = len(reference)
    np.save(guidance/'wuji_right_qpos.npy', reference.astype(np.float32))
    # ForceAware's existing convention is quaternion xyzw, then xyz.
    np.save(guidance/'object_pose_7.npy', np.tile([0., 0., 0., 1., 0., 0., 0.], (frames, 1)).astype(np.float32))
    count = len(hand.local)
    np.savez_compressed(guidance/'contact_guidance.npz',
                        contact_mask=np.ones((frames, count), np.float32),
                        contact_weight=np.ones((frames, count), np.float32),
                        contact_pos_obj=np.tile(guess.object_points, (frames, 1, 1)).astype(np.float32),
                        hand_query_body_ids=np.tile(hand.bodies, (frames, 1)).astype(np.int32),
                        hand_query_local_pos=np.tile(hand.local, (frames, 1, 1)).astype(np.float32))
    moving = depth>0 and not conditioned
    config['sequence'].update(start=0, hold_frame=-1 if moving else 0,
                              hold_seconds=0. if moving else .01)
    (directory/'forceaware.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    return directory


class GraspRuntime:
    """A static CARD runtime, or the original common AdamW runtime."""
    def __init__(self, experiment, args, *, scene=None, gravity=None):
        import warp as wp
        import comfree_warp
        from forceaware.config import load_retarget_config
        from forceaware.runtime import setup_runtime
        from forceaware.targets import build_index, load_sequence, initial_state, make_window
        from forceaware.workflow import _configure_comfree_physics, _collision_output_layout
        from retarget.fa.solver import optimizer_class, settings_from_args
        begun = time.perf_counter()
        self.args = args
        self.experiment = Path(experiment)
        self.grasp_optimizer = getattr(args, 'grasp_optimizer', 'adamw')
        self.cfg = cfg = load_retarget_config(experiment/'sequence', experiment/'forceaware.yaml',
                                              contactaware_dir=experiment/'contactaware')
        worlds = args.multistarts
        self.cfg = cfg = replace(cfg, optimizer=replace(cfg.optimizer, n_iter=args.iterations,
            multistart_count=worlds, multistart_seed=args.seed))
        if scene is not None:
            self.cfg = cfg = replace(cfg, sequence=replace(cfg.sequence, xml=Path(scene)))
        setup_runtime(cfg)
        wp.set_device(args.init_device)
        if not wp.get_device().is_cuda:
            raise RuntimeError('Grasp solving requires CUDA')
        self.device_model, self.model = comfree_warp.load_model(str(cfg.sequence.xml), device=args.init_device)
        if gravity is not None:
            self.model.opt.gravity[:] = (0., 0., -gravity)
            self.device_model.opt.gravity.fill_((0., 0., -gravity))
        _configure_comfree_physics(cfg, self.model, self.device_model, timestep=cfg.simulator.time_grid.mpc_dt)
        # MuJoCo's "last XML" is process-global. Snapshot it while this scene
        # is still the last loaded model, before another node is constructed.
        import mujoco
        scene_snapshot = Path(experiment)/'runtime_scene.xml'
        mujoco.mj_saveLastXML(str(scene_snapshot), self.model)
        self.scene_xml = scene_snapshot.read_text()
        self.index = index = build_index(self.model)
        bodies, enabled = _collision_output_layout(self.device_model)
        self.sequence = sequence = load_sequence(cfg, output_body_ids=bodies, enabled_output_indices=enabled)
        self.qpos, self.qvel = initial_state(self.model, index, sequence, start_frame=0.,
                                            ref_dt=cfg.simulator.time_grid.ref_dt, hold=True)
        self.targets = make_window(sequence, executed_steps=0, horizon=cfg.optimizer.horizon,
            time_grid=cfg.simulator.time_grid, start_frame=0., end_frame=cfg.sequence.end,
            hold_frame=cfg.sequence.hold_frame)
        self.settings = settings_from_args(args)
        if self.grasp_optimizer == 'cem':
            if not (self.experiment/'card_input.npz').exists():
                raise ValueError('Old dynamics-CEM experiment: rebuild inputs with run_grasp.py or scripts/refine_grasp_results.py')
            from retarget.grasp_cem import initialize_runtime
            initialize_runtime(self)
            self.setup_seconds = time.perf_counter()-begun
            return
        self.optimizer = optimizer_class(self.settings)(self.model, self.device_model, index,
            guidance_count=sequence.contact.mask.shape[1], cfg=cfg)
        self.initial_warm = np.zeros(index.action_dim*(1+cfg.optimizer.horizon),np.float32)
        self.initial_action = self.qpos[index.hand_qpos].copy()
        self.load_input = None
        if getattr(args,'load_conditioning',True) and (experiment/'load_input.npz').exists():
            with np.load(experiment/'load_input.npz') as supplied:
                self.load_input = {key:supplied[key].copy() for key in supplied.files}
            self.initial_warm = self.load_input['warm'].astype(np.float32)
            self.initial_action = self.load_input['command'].copy()
        self.setup_seconds = time.perf_counter()-begun

    def solve(self):
        """Dispatch static CEM, or solve one original run.py AdamW window."""
        if self.grasp_optimizer == 'cem':
            from retarget.grasp_device import solve_nodes
            return solve_nodes([self])[0]
        import warp as wp
        opt, index = self.optimizer, self.index
        plan = opt.solve(self.qpos, self.qvel, self.targets,
                         warm=self.initial_warm,
                         previous_action=self.initial_action, allow_init=False,
                         seed=self.cfg.optimizer.multistart_seed, execution_steps=1, start_time=0.)
        selected = int(plan.diag['selected'])
        snapshot = opt.objective.snapshot(selected)
        wp.synchronize_device(opt.backend.device)
        # run.py executes the selected interval, not the full prediction
        # horizon. Contact snapshot row 0 is the corresponding post-step state
        # (the disabled pre-step-0 query is omitted by snapshot()).
        self.output_step = 1
        self.output_workspace = (opt.backend.loss.immediate[self.output_step]
                                 if self.output_step < len(opt.backend.steps)
                                 else opt.backend.loss.terminal)
        self.predicted_terminal_qpos = opt.backend.steps[-1].recorded.result.qpos.numpy()[selected].copy()
        return plan, snapshot, plan.next_qpos.copy()
