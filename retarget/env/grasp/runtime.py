"""Build static grasp environment inputs and compile its forward model."""

from dataclasses import replace
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import numpy as np
from retarget.config.run import ROOT


def prepare_experiment(directory, hand, guess, args):
    """The only task-specific transformation: static references and a new mesh.

    The object is free, as required by the shared physical solver, with a
    stationary target and zero initial velocity. No floor or lift trajectory.
    Mesh dimensions/recentering are already baked into hand.surface.
    """
    import trimesh
    import yaml

    config = yaml.safe_load(args.solver_config.read_text())
    directory = Path(directory)
    scene = directory / "sequence/scene/wuji_right"
    metadata = directory / "sequence/mano_raw"
    guidance = directory / "contactaware"
    for path in (scene, metadata, guidance):
        path.mkdir(parents=True, exist_ok=True)
    mesh_path = directory / "object.obj"
    vertices, faces = np.unique(
        hand.surface.triangles.reshape(-1, 3), axis=0, return_inverse=True
    )
    trimesh.Trimesh(
        vertices=vertices, faces=faces.reshape(-1, 3), process=False
    ).export(mesh_path)
    from retarget.env.grasp.hand import hand_tree, model_hand

    tree = hand_tree(model_hand(hand.model))
    tree.find("option").set(
        "gravity", f"0 0 {-getattr(args, 'planning_gravity', 9.81)}"
    )
    ET.SubElement(
        tree.find("asset"),
        "mesh",
        name="grasp_object_mesh",
        file=str(mesh_path.resolve()),
    )
    obj = ET.SubElement(tree.find("worldbody"), "body", name="grasp_object")
    radius = float(np.linalg.norm(vertices, axis=1).max())
    mass = args.object_mass
    ET.SubElement(
        obj,
        "inertial",
        pos="0 0 0",
        mass=str(mass),
        diaginertia=" ".join([str(max(0.4 * mass * radius**2, 1e-7))] * 3),
    )
    ET.SubElement(obj, "joint", name="obj_joint", type="free", armature=".0001")
    ET.SubElement(
        obj,
        "geom",
        name="object_visual",
        type="mesh",
        mesh="grasp_object_mesh",
        contype="0",
        conaffinity="0",
        mass="0",
        rgba=".25 .65 .85 1",
        group="1",
    )
    ET.SubElement(
        obj,
        "geom",
        name="object_collision",
        type="mesh",
        mesh="grasp_object_mesh",
        contype="4",
        conaffinity="1",
        condim="3",
        priority="1",
        friction=f"{args.friction} .005 .0001",
        mass="0",
    )
    from retarget.env.grasp.physics import apply_object_physics, grasp_physics

    values = grasp_physics(args)
    apply_object_physics(tree, values)
    # Physics uses standard mesh collision; task surface queries are separate.
    (scene / "primitive_sdf.xml").write_text(ET.tostring(tree, encoding="unicode"))
    (metadata / "meta.json").write_text(json.dumps(dict(object="primitive")))
    hand_reference = getattr(guess, "contactaware_reference_qpos", guess.hand_qpos)
    reference = np.tile(hand_reference, (2, 1))
    from retarget.env.grasp.rollout import force_priorities

    supplied = dict(
        object_points=guess.object_points, object_normals=guess.object_normals
    )
    if hasattr(hand, "ik_reference_qpos"):
        supplied["gesture_reference_qpos"] = hand.ik_reference_qpos.copy()
    for name in ("pose_intent_qpos", "pose_intent_radius", "pose_intent_center", "intent_palm_position",
                 "intent_palm_rotation", "cylinder_middle_slots", "cylinder_middle_half_band"):
        if hasattr(hand, name):
            supplied[name] = np.asarray(getattr(hand, name)).copy()
    if hasattr(guess, "contactaware_reference_qpos"):
        q_reference = np.asarray(guess.contactaware_reference_qpos, dtype=float)
        if q_reference.shape != (26,) or not np.isfinite(q_reference).all():
            raise ValueError("ContactAware must produce a finite 26-coordinate hand reference")
        supplied["hand_reference_qpos"] = q_reference.copy()
        supplied["hand_reference_source"] = np.asarray("dexforge_contactaware")
        supplied["hand_initial_qpos"] = np.asarray(guess.hand_qpos, dtype=float).copy()
        from retarget.common.results import write_json

        write_json(directory / "contactaware_reference.json", guess.metadata["contactaware"])
    if hasattr(hand, "semantic_contact_masks"):
        masks = hand.semantic_contact_masks
        supplied.update(
            semantic_patch_offsets=np.r_[0, np.cumsum([len(mask) for mask in masks])],
            semantic_patch_masks=np.concatenate(masks),
            semantic_patch_labels=np.asarray(hand.semantic_patch_labels),
            semantic_reference_local=hand.semantic_contact_local,
            semantic_reference_normals=hand.semantic_contact_normals,
        )
    if hasattr(hand, "contact_reference_normals"):
        supplied["region_reference_normals"] = hand.contact_reference_normals
    if hasattr(hand, "object_orientation"):
        supplied["object_axis_palm"] = hand.object_orientation["axis_palm"]
        supplied["object_axis_limit_deg"] = hand.object_orientation["maximum_angle_deg"]
    if hasattr(hand, "wrist_pose_reference"):
        ref = hand.wrist_pose_reference
        supplied["wrist_pose_reference"] = ref["object_pose_wrist"]
        supplied["wrist_pose_prior_settings"] = [
            ref["weight"],
            ref["translation_scale_mm"],
            ref["rotation_scale_deg"],
        ]
        (directory / "wrist_pose_reference.json").write_text(json.dumps(ref, indent=2))
    if getattr(hand, "tip_patches", None):
        patch_bodies = sorted(hand.tip_patches)
        supplied["tip_patch_bodies"] = np.asarray(patch_bodies, dtype=np.int32)
        supplied["tip_patch_centers"] = [
            hand.tip_patches[b]["center"] for b in patch_bodies
        ]
        supplied["tip_patch_radii"] = [
            hand.tip_patches[b]["radius_m"] for b in patch_bodies
        ]
        supplied["tip_patch_weights"] = [
            hand.tip_patches[b]["weight"] for b in patch_bodies
        ]
    forces = getattr(guess, "contact_forces", None)
    if forces is not None:
        force_priorities(forces, len(guess.object_points))
        supplied["contact_forces"] = forces
    count = len(hand.local)
    contact_mask = np.asarray(getattr(guess, "contact_mask", np.ones(count)), dtype=np.float32)
    if (contact_mask.shape != (count,) or not np.isfinite(contact_mask).all()
            or not np.isin(contact_mask, [0.0, 1.0]).all() or not contact_mask.any()):
        raise ValueError("Reference contact mask must select at least one finite binary contact slot")
    supplied["reference_contact_mask"] = contact_mask.copy()
    np.savez_compressed(directory / "card_input.npz", **supplied)
    frames = len(reference)
    np.save(guidance / "wuji_right_qpos.npy", reference.astype(np.float32))
    # ForceAware's existing convention is quaternion xyzw, then xyz.
    np.save(
        guidance / "object_pose_7.npy",
        np.tile([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], (frames, 1)).astype(np.float32),
    )
    # ContactAware may release individual unreachable slots; this does not
    # release a mandatory finger, which the front end validates separately.
    np.savez_compressed(
        guidance / "contact_guidance.npz",
        contact_mask=np.tile(contact_mask, (frames, 1)),
        contact_weight=np.ones((frames, count), np.float32),
        contact_pos_obj=np.tile(guess.object_points, (frames, 1, 1)).astype(np.float32),
        hand_query_body_ids=np.tile(hand.bodies, (frames, 1)).astype(np.int32),
        hand_query_local_pos=np.tile(hand.local, (frames, 1, 1)).astype(np.float32),
    )
    config["sequence"].update(start=0, hold_frame=0, hold_seconds=0.01)
    (directory / "model.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    return directory


class GraspRuntime:
    def __init__(self, experiment, args, *, scene=None, gravity=None):
        import warp as wp
        from retarget.env.mjwarp import load_model
        from retarget.config.dex import load_retarget_config
        from retarget.task.dex.reference import (
            build_index,
            load_sequence,
            initial_state,
            make_window,
        )
        from retarget.env.dex.physics import (
            _configure_comfree_physics,
            _collision_output_layout,
        )

        begun = time.perf_counter()
        self.args = args
        self.experiment = Path(experiment)
        self.cfg = cfg = load_retarget_config(
            experiment / "sequence",
            experiment / "model.yaml",
            contactaware_dir=experiment / "contactaware",
        )
        if scene is not None:
            self.cfg = cfg = replace(
                cfg, sequence=replace(cfg.sequence, xml=Path(scene))
            )
        wp.init()
        wp.set_device(args.init_device)
        if not wp.get_device().is_cuda:
            raise RuntimeError("Grasp solving requires CUDA")
        self.device_model, self.model = load_model(
            str(cfg.sequence.xml), device=args.init_device
        )
        if gravity is not None:
            self.model.opt.gravity[:] = (0.0, 0.0, -gravity)
            self.device_model.opt.gravity.fill_((0.0, 0.0, -gravity))
        _configure_comfree_physics(
            cfg, self.model, self.device_model, timestep=cfg.simulator.time_grid.mpc_dt
        )
        # MuJoCo's "last XML" is process-global. Snapshot it while this scene
        # is still the last loaded model, before another node is constructed.
        import mujoco

        scene_snapshot = Path(experiment) / "runtime_scene.xml"
        mujoco.mj_saveLastXML(str(scene_snapshot), self.model)
        self.scene_xml = scene_snapshot.read_text()
        self.index = index = build_index(self.model)
        bodies, enabled = _collision_output_layout(self.model)
        self.sequence = sequence = load_sequence(
            cfg, output_body_ids=bodies, enabled_output_indices=enabled
        )
        self.qpos, self.qvel = initial_state(
            self.model,
            index,
            sequence,
            start_frame=0.0,
            ref_dt=cfg.simulator.time_grid.ref_dt,
            hold=True,
        )
        with np.load(experiment / "card_input.npz", allow_pickle=False) as supplied:
            if "hand_initial_qpos" in supplied:
                actual_hand = supplied["hand_initial_qpos"]
                if actual_hand.shape != (26,) or not np.isfinite(actual_hand).all():
                    raise ValueError("Actual hand initial state must contain 26 finite coordinates")
                self.qpos[np.asarray(index.hand_qpos)] = actual_hand
        self.targets = make_window(
            sequence,
            executed_steps=0,
            horizon=cfg.prediction.horizon,
            time_grid=cfg.simulator.time_grid,
            start_frame=0.0,
            end_frame=cfg.sequence.end,
            hold_frame=cfg.sequence.hold_frame,
        )
        from retarget.env.grasp.rollout import initialize_runtime

        initialize_runtime(self)
        self.setup_seconds = time.perf_counter() - begun
