# /// script
# requires-python = ">=3.10"
# dependencies = ["robokit[mjwarp]"]
#
# [tool.uv.sources]
# robokit = { path = "../../../..", editable = true }
# ///
# pyright: reportMissingImports=false
# pyright: reportArgumentType=false
"""Retarget a Sharpa pen rollout to Wuji Hand 2.2, then refine it with the existing SimMPC.

First export a source with ``07_sharpa_pen.py --headless --output results/sharpa_pen.npz``.
Run this example with ``--source results/sharpa_pen.npz --headless``. ``--retarget-only``
saves the kinematic reference without claiming physical success. View either saved motion
with ``--replay reference`` or ``--replay rollout`` (neither reruns optimization).
"""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import trimesh
import viser
import warp as wp

from robokit.assets.robots.end_effectors import sharpa_hand, wuji_hand_2_2
from robokit.geom import CapsuleGeom, WarpScene
from robokit.helpers.hand_retargeting import HandRetargeterOffline, contact_config, refine_config
from robokit.helpers.hand_retargeting.presets import sharpa
from robokit.helpers.hand_retargeting.presets import wuji_hand_2_2 as wuji
from robokit.helpers.sim_mpc import SimMPC, SimMPCConfig, SimMPCStage
from robokit.robo import Robot
from robokit.sim import MjWarpSim
from robokit.terms import ControlTask, SensorTask, StateTask, SupportTask
from robokit.utils.ee_coord_utils import ee_coord_conversion
from robokit.utils.visualize_utils import ViserMjcf
from robokit.utils.warp_utils import wp_vec7
from robokit.xform.numpy import matrix_to_quaternion, quaternion_apply, quaternion_invert, quaternion_multiply


ROOT = np.array([0, 0, 0.5, -0.5, -0.5, 0.5, 0.5], np.float32)
SKIN, RADIUS, HALF_LENGTH = 0.0025, 0.0045, 0.0655
HEIGHT_WEIGHT = 300000.0  # penalize the brief height loss during a finger handoff
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
GROUPS = ["r_thumb_", "r_index_finger_", "r_middle_finger_", "r_ring_finger_", "r_pinky_"]


def build_scene(T_world_pen: np.ndarray) -> mujoco.MjSpec:
    """Build the fixed-wrist, 20-actuator scene with the original pen physics.

    >>> # xdoctest: +SKIP
    >>> build_scene(np.array([0.14, 0, 0.58, 1, 0, 0, 0])).compile().nu
    20
    """
    scene = mujoco.MjSpec.from_file(str(wuji_hand_2_2.MJCF_PATH_RIGHT))
    for mesh in scene.meshes:
        mesh.file = str(wuji_hand_2_2.MJCF_PATH_RIGHT.parent / scene.meshdir / mesh.file)
    scene.meshdir, scene.option.timestep = "", 0.005
    scene.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    scene.body("r_mount").pos, scene.body("r_mount").quat = ROOT[:3], ROOT[3:]
    for index, geom in enumerate(scene.geoms):
        # The asset's geoms are unnamed; SupportTask matches geom names, not body names.
        geom.name = f"{geom.parent.name}_geom_{index}"
        if geom.contype or geom.conaffinity:
            geom.friction, geom.solmix = [1.0, 0.005, 1e-4], 0.0
            geom.solref, geom.solimp[:3] = [0.01, 1.0], [0.99, 0.999, 0.001]
    pen = scene.worldbody.add_body(name="pen", pos=T_world_pen[:3], quat=T_world_pen[3:])
    pen.add_freejoint(name="pen_free")
    pen.add_geom(
        name="pen", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[RADIUS, HALF_LENGTH, 0],
        quat=[np.sqrt(0.5), 0, np.sqrt(0.5), 0], solref=[0.01, 1.0], solimp=[0.99, 0.999, 0.001, 0.5, 2],
        margin=SKIN, mass=0.01, friction=[1.0, 0.005, 1e-4], condim=4, rgba=[0.15, 0.45, 0.9, 1],
    )  # fmt: skip
    body = mujoco.mjtObj.mjOBJ_XBODY
    for name, kind in (("pos", mujoco.mjtSensor.mjSENS_FRAMEPOS), ("axis", mujoco.mjtSensor.mjSENS_FRAMEXAXIS)):
        scene.add_sensor(name=name, type=kind, objtype=body, objname="pen")
    for actuator in scene.actuators:
        actuator.biasprm[2] *= 2
        scene.add_sensor(name=actuator.name, type=mujoco.mjtSensor.mjSENS_ACTUATORVEL,
                         objtype=mujoco.mjtObj.mjOBJ_ACTUATOR, objname=actuator.name)  # fmt: skip
    return scene


def solve_retargeting(source: Path, stride: int, device: str) -> dict:
    """Use the repository's offline/contact/refinement solvers to transfer a source rollout.

    >>> # xdoctest: +SKIP
    >>> result = solve_retargeting(Path("results/sharpa_pen.npz"), 4, "cpu")
    >>> result["q"].shape[1]
    20
    """
    with np.load(source, allow_pickle=False) as archive:
        motion = {name: archive[name] for name in archive.files}
    source_model = mujoco.MjModel.from_xml_string(str(motion["scene_xml"]))
    if len(motion["qpos"]) < 3 or not np.isfinite(motion["qpos"]).all():
        raise ValueError("Source must contain at least three finite motion frames.")
    indices = np.unique(np.r_[np.arange(0, len(motion["qpos"]), stride), len(motion["qpos"]) - 1])
    if len(indices) < 3:
        indices = np.linspace(0, len(motion["qpos"]) - 1, 3, dtype=int)
    frames = len(indices)
    source_robot = Robot.load(sharpa_hand.URDF_PATH_RIGHT)
    robot = Robot.load(
        wuji_hand_2_2.URDF_PATH_RIGHT,
        load_meshes=True,
        load_collision_spheres=True,
        collision_spheres_path=wuji_hand_2_2.COLLISION_SPHERE_PATH_RIGHT,
        self_collision_ignore_path=wuji_hand_2_2.SPHERE_SELF_COLLISION_IGNORE_PATH_RIGHT,
    )
    names = wuji.spec.target_names
    source_links = dict(sharpa._LINKS)  # all 21 landmarks, including the finger joints
    source_joint_ids = [source_model.joint(name).qposadr[0] for name in source_robot.spec.actuated_joint_names]
    source_state = source_robot.state(q=motion["qpos"][indices][:, source_joint_ids].astype(np.float32))
    source_robot.forward_kinematics(source_state)
    T_source_link = source_state.T_world_link.numpy()
    points_source = T_source_link[:, [source_robot.spec.link_names.index(source_links[name]) for name in names], :3]

    # The raw source URDF uses +x for the palm normal, +z toward the fingers.
    # Both scene installations put those semantic axes at world +z and +x.
    q_wxyz_world_source = np.asarray(motion["T_world_base"])[3:]
    q_wxyz_wuji_source = matrix_to_quaternion(
        ee_coord_conversion(sharpa_hand.EE_COORD_SPEC_RIGHT, wuji_hand_2_2.EE_COORD_SPEC_RIGHT)
    )
    q_wxyz_world_source_new = quaternion_multiply(ROOT[3:], q_wxyz_wuji_source)
    q_wxyz_world_world_source = quaternion_multiply(q_wxyz_world_source_new, quaternion_invert(q_wxyz_world_source))
    points_world = ROOT[:3] + quaternion_apply(q_wxyz_world_source_new, points_source)
    source_zero = source_robot.state(q=np.zeros((1, source_robot.spec.num_actuated_joints), np.float32))
    target_zero = robot.state(q=np.zeros((1, robot.spec.num_actuated_joints), np.float32))
    source_robot.forward_kinematics(source_zero)
    robot.forward_kinematics(target_zero)
    source_tips = source_zero.T_world_link.numpy()[
        0, [source_robot.spec.link_names.index(source_links[n]) for n in wuji.spec.contact_target_names], :3
    ]
    target_tips = target_zero.T_world_link.numpy()[
        0, [robot.spec.link_names.index(wuji.spec.target_link_names[n]) for n in wuji.spec.contact_target_names], :3
    ]
    scale = float(np.median(np.linalg.norm(target_tips, axis=1) / np.linalg.norm(source_tips, axis=1)))
    points_world = (ROOT[:3] + scale * (points_world - ROOT[:3])).astype(np.float32)
    base = np.broadcast_to(ROOT, (1, frames, 7)).copy()
    limits = robot.spec.actuated_joint_limits
    init_q = np.broadcast_to(
        np.clip((limits[:, 0] + limits[:, 1]) / 2, limits[:, 0], limits[:, 1]), (1, frames, 20)
    ).copy()
    init = robot.state(
        q=wp.array(init_q, dtype=float, device=device), T_world_base=wp.array(base, dtype=wp_vec7, device=device)
    )
    config = replace(wuji.offline, optimize_base=False)
    config.solver = replace(config.solver, cuda_graph_mode="none" if device == "cpu" else "full")
    # The input points already use target base axes; no MANO orientation conversion applies.
    target_spec = replace(wuji.spec, target_coord_spec=wuji.spec.root_link_coord_spec)
    root_quat = np.broadcast_to(ROOT[3:], (frames, 4)).copy()
    tracking = HandRetargeterOffline(config, robot, target_spec, device=device)
    packed = tracking.solve_numpy(points_world, root_quat, init_state=init)

    # Transfer at most one actual contact per finger segment and frame, including shaft contacts.
    contact_names = tuple(name for chain in target_spec.target_chains for name in chain[:-1])
    contact_spec = replace(target_spec, contact_target_names=contact_names)
    link_indices = [robot.spec.link_names.index(target_spec.target_link_names[name]) for name in contact_names]
    init = robot.state(
        q=wp.array(packed[None, :, 7:], dtype=float, device=device),
        T_world_base=wp.array(base, dtype=wp_vec7, device=device),
    )
    robot.forward_kinematics(init)
    T_world_link = init.T_world_link.numpy()[0][:, link_indices]
    contact_points = np.zeros((frames, len(contact_names), 3), np.float32)
    local_points = np.zeros_like(contact_points)
    contact_mask = np.zeros(contact_points.shape[:2], bool)
    distance = np.full(contact_mask.shape, np.inf)
    source_data = mujoco.MjData(source_model)
    pen_body = source_model.body("pen").id
    T_world_pen = motion["T_world_pen"][indices].copy()
    T_world_pen[:, :3] = ROOT[:3] + quaternion_apply(
        q_wxyz_world_world_source, T_world_pen[:, :3] - motion["T_world_base"][:3]
    )
    T_world_pen[:, 3:] = quaternion_multiply(q_wxyz_world_world_source, T_world_pen[:, 3:])
    source_body_slots = {}
    for body_id in range(1, source_model.nbody):
        parent = body_id
        while parent:
            match = next(
                (j for j, n in enumerate(contact_names) if source_links[n] == source_model.body(parent).name), None
            )
            if match is not None:
                source_body_slots[body_id] = match
                break
            parent = int(source_model.body_parentid[parent])
    for frame, source_index in enumerate(indices):
        source_data.qpos[:] = motion["qpos"][source_index]
        mujoco.mj_forward(source_model, source_data)
        for contact in source_data.contact:
            bodies = source_model.geom_bodyid[contact.geom]
            if pen_body not in bodies or contact.dist > SKIN:
                continue
            other = int(bodies[1] if bodies[0] == pen_body else bodies[0])
            slot = source_body_slots.get(other)
            if slot is None or contact.dist >= distance[frame, slot]:
                continue
            distance[frame, slot] = contact.dist
            contact_mask[frame, slot] = True
            p_world = ROOT[:3] + quaternion_apply(q_wxyz_world_world_source, contact.pos - motion["T_world_base"][:3])
            p_pen = quaternion_apply(quaternion_invert(T_world_pen[frame, 3:]), p_world - T_world_pen[frame, :3])
            center = np.array([np.clip(p_pen[0], -HALF_LENGTH, HALF_LENGTH), 0, 0])
            normal = p_pen - center
            normal /= max(np.linalg.norm(normal), 1e-8)
            contact_points[frame, slot] = T_world_pen[frame, :3] + quaternion_apply(
                T_world_pen[frame, 3:], center + (RADIUS + SKIN) * normal
            )
            # Express a surface candidate relative to the corresponding semantic joint, not a link origin.
            src_point = (
                quaternion_apply(q_wxyz_world_source, points_source[frame, names.index(contact_names[slot])])
                + motion["T_world_base"][:3]
            )
            candidate = points_world[frame, names.index(contact_names[slot])] + scale * quaternion_apply(
                q_wxyz_world_world_source, contact.pos - src_point
            )
            local_points[frame, slot] = quaternion_apply(
                quaternion_invert(T_world_link[frame, slot, 3:]), candidate - T_world_link[frame, slot, :3]
            )
    for slot, name in enumerate(contact_names):
        active = contact_mask[:, slot]
        if not active.any():
            continue
        mesh = robot.spec.link_collision_geometries[contact_spec.target_link_names[name]].to_mesh()
        # Distal pads are separate fixed links; include them in the distal segment surface.
        if name.endswith(("_dip", "_ip")):
            finger = name.split("_")[0]
            prefix = f"r_{finger}" + ("_finger" if finger not in ("thumb", "pinky") else "")
            pad = robot.spec.link_collision_geometries[f"{prefix}_tip_sensor_frame"].to_mesh()
            mesh = trimesh.util.concatenate([mesh, pad])
        local_points[active, slot], _, _ = trimesh.proximity.closest_point(mesh, local_points[active, slot])

    # One analytic capsule scene per frame keeps collision queries aligned with the moving pen.
    matrices = np.tile(np.eye(4, dtype=np.float32), (frames, 1, 1))
    q_wxyz_capsule = quaternion_multiply(T_world_pen[:, 3:], np.array([np.sqrt(0.5), 0, np.sqrt(0.5), 0]))
    for axis in range(3):
        matrices[:, :3, axis] = quaternion_apply(q_wxyz_capsule, np.eye(3)[axis])
    matrices[:, :3, 3] = T_world_pen[:, :3]
    scene = WarpScene(frames, device).add(
        CapsuleGeom(
            np.full(frames, RADIUS),
            np.full(frames, HALF_LENGTH),
            np.arange(frames + 1, dtype=np.int32),
            poses=wp.array(matrices, dtype=wp.mat44, device=device),
        )
    )
    scene_indices = wp.array(np.arange(frames, dtype=np.int32)[None], dtype=int, device=device)
    contacts = contact_config(config)
    contacts.task("contact").contact_margin = 0.0
    contacts.task("collision").margin = 0.0
    direct = HandRetargeterOffline(
        contacts, robot, contact_spec, scene=scene, scene_indices=scene_indices, device=device
    )
    packed = direct.solve_with_contact_numpy(
        points_world, contact_points, contact_mask, init_state=init, local_contact_points=local_points
    )
    init = robot.state(
        q=wp.array(packed[None, :, 7:], dtype=float, device=device),
        T_world_base=wp.array(base, dtype=wp_vec7, device=device),
    )
    refinement = refine_config(config)
    refinement.task("contact").contact_margin = 0.0
    refinement.task("contact").weight = 300.0
    refinement.task("collision").margin = 0.0
    refinement.task("collision").weight = 100.0
    refine = HandRetargeterOffline(
        refinement, robot, contact_spec, scene=scene, scene_indices=scene_indices, device=device
    )
    packed = refine.solve_with_contact_numpy(
        points_world, contact_points, contact_mask, init_state=init, local_contact_points=local_points
    )
    q = np.clip(packed[:, 7:], limits[:, 0], limits[:, 1])
    if not np.isfinite(q).all():
        raise RuntimeError("Retargeting returned non-finite joint positions.")
    final = robot.state(
        q=wp.array(q[None], dtype=float, device=device), T_world_base=wp.array(base, dtype=wp_vec7, device=device)
    )
    robot.forward_kinematics(final)
    actual = final.T_world_link.numpy()[0][:, link_indices]
    actual = actual[..., :3] + quaternion_apply(actual[..., 3:], local_points)
    errors = np.linalg.norm(actual - contact_points, axis=-1)[contact_mask]
    print(
        f"retarget: {frames} frames, {contact_mask.sum()} segment contacts, "
        f"contact RMS {1000 * np.sqrt(np.mean(errors**2)) if errors.size else 0:.2f} mm",
        flush=True,
    )
    return dict(
        time=(indices * float(motion["timestep"])),
        q=q,
        joint_names=np.array(robot.spec.actuated_joint_names),
        T_world_base=packed[:, :7],
        T_world_pen=T_world_pen,
        target_points=points_world,
        contact_names=np.array(contact_names),
        contact_points=contact_points,
        local_contact_points=local_points,
        contact_mask=contact_mask,
        contact_errors=errors,
        target_scale=scale,
        source=str(source.resolve()),
    )


def compute_metrics(model: mujoco.MjModel, qpos: np.ndarray, initial_qpos: np.ndarray) -> dict:
    """Score physical states against the three-turn, no-drop, 2 mm overlap criteria.

    >>> # xdoctest: +SKIP
    >>> compute_metrics(model, qpos, initial_qpos)["passed"]
    True
    """
    data = mujoco.MjData(model)
    if not np.isfinite(qpos).all():
        return dict(
            passed=False,
            turns=None,
            dropped=True,
            max_height_loss_mm=None,
            finger_overlap_mm=None,
            finite=False,
            duration=len(qpos) * model.opt.timestep,
        )
    pen_body = model.body("pen").id
    pen_adr = int(model.jnt_qposadr[model.body_jntadr[pen_body]])
    axis = quaternion_apply(qpos[:, pen_adr + 3 : pen_adr + 7], np.array([1, 0, 0]))
    start_axis = quaternion_apply(initial_qpos[pen_adr + 3 : pen_adr + 7], np.array([1, 0, 0]))
    heading = np.unwrap(np.r_[np.arctan2(start_axis[1], start_axis[0]), np.arctan2(axis[:, 1], axis[:, 0])])
    deadline = round(15.0 / model.opt.timestep)
    turns = float((heading[min(len(qpos), deadline)] - heading[0]) / (2 * np.pi))
    finite = bool(np.isfinite(qpos).all())
    dropped = bool(np.any(qpos[:, pen_adr + 2] < initial_qpos[pen_adr + 2] - 0.03))
    height_loss = max(0.0, float(initial_qpos[pen_adr + 2] - qpos[:, pen_adr + 2].min()))
    overlap = 0.0
    for q in qpos:
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_collision(model, data)
        for contact in data.contact:
            if pen_body not in model.geom_bodyid[contact.geom]:
                overlap = max(overlap, -float(contact.dist))
    return dict(
        passed=bool(finite and len(qpos) >= deadline and turns >= 3.0 and not dropped and overlap <= 0.002),
        turns=turns,
        dropped=dropped,
        max_height_loss_mm=height_loss * 1000,
        finger_overlap_mm=overlap * 1000,
        finite=finite,
        duration=len(qpos) * model.opt.timestep,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--reference", type=Path, help="Reuse a previously solved retarget.npz.")
    parser.add_argument("--output", type=Path, default=Path("results/wuji_pen"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--headless", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--retarget-only", action="store_true")
    mode.add_argument("--hold-only", action="store_true", help="Validate and save just the initial 1 s physical hold.")
    mode.add_argument("--replay", choices=("reference", "rollout"))
    args = parser.parse_args()
    if args.steps < 1 or args.stride < 1 or args.samples < 2:
        parser.error("--steps and --stride must be positive; --samples must be at least 2")
    if args.source is None and args.reference is None and args.replay is None:
        parser.error("--source or --reference is required unless replaying saved motion")
    if args.replay is None:
        wp.config.log_level = wp.LOG_WARNING
        wp.init()
        wp.set_device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    reference_path = args.output / "retarget.npz"
    if args.replay is not None or args.reference is not None:
        with np.load(args.reference or reference_path, allow_pickle=False) as archive:
            reference = {name: archive[name] for name in archive.files}
        if args.replay is None:
            np.savez_compressed(reference_path, **reference)
    else:
        reference = solve_retargeting(args.source, args.stride, args.device)
        np.savez_compressed(reference_path, **reference)
    scene = build_scene(reference["T_world_pen"][0])
    model = scene.compile()
    joint_indices = np.array([model.joint(name).qposadr[0] for name in reference["joint_names"]])
    actuator_columns = [
        list(reference["joint_names"]).index(model.joint(int(j)).name) for j in model.actuator_trnid[:, 0]
    ]
    pen_adr = int(model.joint("pen_free").qposadr[0])
    state = mujoco.MjData(model)
    state.qpos[joint_indices], state.ctrl[:] = reference["q"][0], reference["q"][0, actuator_columns]
    mujoco.mj_forward(model, state)
    reference_qpos = np.tile(state.qpos, (len(reference["q"]), 1))
    reference_qpos[:, joint_indices] = reference["q"]
    reference_qpos[:, pen_adr : pen_adr + 7] = reference["T_world_pen"]
    np.savez_compressed(
        args.output / "reference.npz", qpos=reference_qpos, time=reference["time"], scene_xml=scene.to_xml()
    )
    if args.retarget_only:
        report = dict(
            status="retarget_only",
            physical_validation=False,
            contact_count=int(reference["contact_mask"].sum()),
            contact_rms_mm=float(1000 * np.sqrt(np.mean(reference["contact_errors"] ** 2)))
            if reference["contact_errors"].size
            else None,
        )
        (args.output / "retarget_report.json").write_text(json.dumps(report, indent=2) + "\n")
        if args.headless:
            return
        args.replay = "reference"
    if args.replay is None:
        hold_rows = []
        for _ in range(200):
            mujoco.mj_step(model, state)
            hold_rows.append(state.qpos.copy())
        hold = compute_metrics(model, np.asarray(hold_rows), reference_qpos[0])
        hold.pop("passed")
        hold["held"] = bool(hold["finite"] and not hold["dropped"] and hold["finger_overlap_mm"] <= 2.0)
        initial_qpos = state.qpos.copy()
        print(f"hold: {json.dumps(hold)}", flush=True)
        np.savez_compressed(
            args.output / f"hold_seed{args.seed}.npz",
            qpos=np.asarray(hold_rows),
            initial_qpos=reference_qpos[0],
            scene_xml=scene.to_xml(),
            timestep=model.opt.timestep,
        )
        (args.output / f"hold_seed{args.seed}.json").write_text(json.dumps(hold, indent=2) + "\n")
        if not hold["held"]:
            report = dict(passed=False, status="initial_hold_failed", seed=args.seed, hold=hold)
            (args.output / f"report_seed{args.seed}.json").write_text(json.dumps(report, indent=2) + "\n")
            raise SystemExit(1)
        if args.hold_only:
            return
        H = 100
        times = np.arange(((args.steps + 7) // 8) * 8 + H + 1) * model.opt.timestep
        ctrl_ref = np.stack(
            [np.interp(times, reference["time"], reference["q"][:, j]) for j in actuator_columns], axis=1
        )
        axis, pos = int(model.sensor("axis").adr[0]), int(model.sensor("pos").adr[0])
        ref = np.zeros((1, len(times), model.nsensordata), np.float32)
        ref[0, :, pos : pos + 3] = state.qpos[pen_adr : pen_adr + 3]
        sensor_ref = wp.array(ref, dtype=float)
        config = SimMPCConfig(
            num_samples=args.samples,
            horizon_steps=H,
            node_steps=25,
            ctrl_steps=8,
            interpolation="cubic",
            stages=[SimMPCStage(2, [(0.5, None), (1.0, 0.01)])],
            std_growth=1.0,
            terminal_scale=1.0,
            ctrl_std=0.12,
            seed=args.seed,
        )
        config.add(SensorTask(sensor_ref, dict(axis=60.0), residual="heading"), terminal_scale=10.0)
        config.add(SensorTask(sensor_ref, dict(axis=[0.0, 0.0, 400.0])))
        config.add(SensorTask(sensor_ref, dict(pos=[20000.0, 20000.0, HEIGHT_WEIGHT])))
        config.add(SensorTask(sensor_ref, {a.name: 0.03 for a in scene.actuators}))
        config.add(ControlTask(0.22, ctrl_ref))
        config.add(SupportTask("pen", GROUPS, 2, 10.0, SKIN))
        hand = np.array([int(b) != model.body("pen").id for b in model.geom_bodyid])
        config.add(
            StateTask(
                initial_qpos[None],
                np.zeros((1, model.nv)),
                np.zeros(model.nv),
                penetration_groups=(hand, hand),
                penetration_scale=1e6,
                margin=0.001,
            )
        )
        clock = 0
        first_axis = quaternion_apply(state.qpos[pen_adr + 3 : pen_adr + 7], np.array([1, 0, 0]))
        first = heading = float(np.arctan2(first_axis[1], first_axis[0]))

        def lead(plan: list) -> None:
            nonlocal clock, heading
            if plan:
                q_wxyz = sim.data.qpos.numpy()[0, pen_adr + 3 : pen_adr + 7]
                direction = quaternion_apply(q_wxyz, np.array([1, 0, 0]))
                angle = np.arctan2(direction[1], direction[0])
                heading += (angle - heading + np.pi) % (2 * np.pi) - np.pi
                clock += config.ctrl_steps
                if args.headless and clock % 200 == 0:
                    print(
                        f"progress: {clock * model.opt.timestep:.1f} s, {(heading - first) / (2 * np.pi):.2f} turns",
                        flush=True,
                    )
            angles = np.minimum(
                np.minimum(first + 2.7 * model.opt.timestep * (clock + np.arange(H + 1)), heading + 2.0),
                first + 2 * np.pi * 3.2,
            )
            ref[0, clock : clock + H + 1, axis : axis + 2] = np.stack([np.cos(angles), np.sin(angles)], axis=1)
            sensor_ref.assign(ref)

        lead([])
        sim = MjWarpSim(model, 1, state, nconmax=128, njmax=800)
        mpc = SimMPC(config, sim, ctrl_ref)
        start = time.perf_counter()
        print(f"planning {args.steps * model.opt.timestep:.2f} s, seed {args.seed} ...", flush=True)
        (rollout,), _ = mpc.execute(sim, args.steps, callback=lead)
        # execute returns pre-step states; append the final integrated state for physical scoring.
        scored = np.concatenate([rollout["qpos"][1:], sim.data.qpos.numpy()])
        report = compute_metrics(model, scored, initial_qpos)
        report.update(
            seed=args.seed,
            num_samples=args.samples,
            horizon_steps=H,
            height_weight=HEIGHT_WEIGHT,
            hold=hold,
            planned_seconds=time.perf_counter() - start,
        )
        report["passed"] = bool(report["passed"] and hold["held"])
        np.savez_compressed(
            args.output / f"rollout_seed{args.seed}.npz",
            **rollout,
            final_qpos=sim.data.qpos.numpy()[0],
            initial_qpos=initial_qpos,
            scene_xml=scene.to_xml(),
            timestep=model.opt.timestep,
        )
        (args.output / f"report_seed{args.seed}.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        if args.headless:
            raise SystemExit(0 if report["passed"] else 1)
    elif args.replay == "rollout":
        with np.load(args.output / f"rollout_seed{args.seed}.npz", allow_pickle=False) as archive:
            rollout = {name: archive[name] for name in archive.files}
    if args.headless:
        return
    if args.replay == "reference":
        rows, interval = reference_qpos, float(np.median(np.diff(reference["time"])))
    else:
        rows, interval = rollout["qpos"], model.opt.timestep
    server = viser.ViserServer()
    viewer = ViserMjcf(server, scene.to_xml())
    print("replaying; Ctrl+C to stop", flush=True)
    while True:
        for q in rows:
            viewer.update_qpos(q)
            time.sleep(interval)
        time.sleep(1.0)


if __name__ == "__main__":
    main()
