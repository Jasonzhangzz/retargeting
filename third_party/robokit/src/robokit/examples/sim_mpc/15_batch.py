# /// script
# requires-python = ">=3.10"
# dependencies = ["robokit[mjwarp]"]
#
# [tool.uv.sources]
# robokit = { path = "../../../..", editable = true }
# ///
# pyright: reportMissingImports=false
# pyright: reportArgumentType=false
"""SimMPC batch: one Sharpa right hand tracks four retargeted OakInk-v2 clips, each with its own object, at once.

SPIDER: https://arxiv.org/abs/2511.09484.
"""

import glob
import time
from dataclasses import replace

import mujoco
import numpy as np
import trimesh
import viser
import warp as wp

from robokit.assets import fetch
from robokit.assets.robots.end_effectors import sharpa_hand
from robokit.geom import MeshGeom, WarpScene
from robokit.helpers.hand_retargeting import HandRetargeterOffline, HandSpec, refine_config
from robokit.helpers.hand_retargeting.presets._common import FIVE_FINGERTIPS
from robokit.helpers.hand_retargeting.presets.sharpa import _LINKS, offline, spec
from robokit.helpers.sim_mpc import SimMPC, SimMPCConfig, SimMPCStage
from robokit.robo import Robot
from robokit.sim import MjWarpSim
from robokit.terms import StateTask
from robokit.utils.visualize_utils import ViserMjcf
from robokit.xform.numpy import quaternion_apply, quaternion_invert, quaternion_multiply, quaternion_to_matrix
from robokit.xform.numpy.rotation_conversions import matrix_to_euler_angles


REPO, REVISION = "retarget/retarget_example", "8b0cdf2b24cc86d1c859050c5b0317ddbec59a63"  # human hands, objects, 50 Hz
CLIPS = dict(lift_board="O02_0018_00001", pour_tube="O02_0011_00003", stir_beaker="O02_0033_00001",
             wipe_board="O02_0055_00002")  # OakInk-v2 right-hand clips and their objects  # fmt: skip
TARGETS = ("wrist", *FIVE_FINGERTIPS)


def main():
    wp.config.log_level = wp.LOG_WARNING  # no per-module load lines
    wp.init()

    # --- data: each clip's human wrist pose, five fingertips and object pose, and each object's meshes ---
    clip_dir = lambda name: f"processed/oakinkv2/mano/right/{name}/0"
    object_dir = lambda name: f"processed/oakinkv2/assets/objects/{CLIPS[name]}"
    files = [f"{clip_dir(n)}/trajectory_keypoints.npz" for n in CLIPS] + [f"{object_dir(n)}/**" for n in CLIPS]
    root = fetch(files, REVISION, REPO)
    clips = {n: np.load(root / clip_dir(n) / "trajectory_keypoints.npz") for n in CLIPS}

    # --- scenes: the Sharpa on six position-controlled base joints and the clip's object free on a table; the scenes differ in the object's meshes, geoms and inertia, and the table's height ---
    specs = []
    for name, clip in clips.items():
        scene = mujoco.MjSpec.from_file(str(sharpa_hand.MJCF_PATH_RIGHT))
        for mesh in scene.meshes:  # absolute mesh paths, so the batched model and the viewer's copies find them
            mesh.file = str(sharpa_hand.MJCF_PATH_RIGHT.parent / scene.meshdir / mesh.file)
        scene.meshdir = ""
        for joint in scene.joints:  # SPIDER's hand dynamics: heavy, stiff fingers without torque limits
            joint.armature, joint.actfrclimited = 1.0, mujoco.mjtLimited.mjLIMITED_FALSE
        for actuator in scene.actuators:
            actuator.gainprm[0], actuator.biasprm[:3] = 300.0, [0.0, -300.0, 1.0]
        for geom in scene.geoms:  # the hand touches only the object, and the object the table
            geom.conaffinity = 0
        scene.option.timestep = 0.01
        hand = scene.body("right_hand_C_MC")
        hand.gravcomp = 1.0  # the base actuators stand in for an arm
        for kind, axis in zip(["pos"] * 3 + ["rot"] * 3, [*np.eye(3)] * 2):
            joint = f"{kind}_{'xyz'[int(np.argmax(axis))]}"
            hand.add_joint(name=joint, type=getattr(mujoco.mjtJoint, "mjJNT_SLIDE" if kind == "pos" else "mjJNT_HINGE"),
                           axis=axis, armature=1.0)  # fmt: skip
            scene.add_actuator(
                name=joint, target=joint, trntype=mujoco.mjtTrn.mjTRN_JOINT, biastype=mujoco.mjtBias.mjBIAS_AFFINE,
                gainprm=[1000] + [0] * 9, biasprm=[0, -1000, -100] + [0] * 7,
            )  # fmt: skip
        folder = root / object_dir(name)
        vertices = trimesh.load(str(folder / "visual.obj"), force="mesh").vertices
        vertices = quaternion_apply(clip["qpos_obj_right"][0, 3:], vertices) + clip["qpos_obj_right"][0, :3]
        scene.worldbody.add_geom(
            name="table", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 1], pos=[0, 0, vertices[:, 2].min()],
            contype=2, conaffinity=0,
        )  # fmt: skip
        body = scene.worldbody.add_body(name="object")
        body.add_freejoint(name="object")
        scene.add_mesh(name="object", file=str(folder / "visual.obj"))
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="object", contype=0, conaffinity=0, density=0)
        for k, path in enumerate(sorted(glob.glob(str(folder / "convex/*.obj")))):  # density 1000
            scene.add_mesh(name=f"object_{k}", file=path)
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"object_{k}", group=3, contype=0, conaffinity=3)
        specs.append(scene)
    model = specs[0].compile()  # the joints and actuators every scene shares
    print(f"{len(specs)} scenes: {model.nq} qpos, {model.nu} actuators")

    # --- retarget: the wrist and fingertips to the Sharpa's base pose and 22 joints, in the object's frame, where a collision term keeps the hand out of the static object mesh ---
    config = replace(offline, terms=[*offline.terms, refine_config(offline).task("collision")])
    robot = Robot.load(
        str(sharpa_hand.URDF_PATH_RIGHT), load_collision_spheres=True,
        collision_spheres_path=str(sharpa_hand.COLLISION_SPHERE_PATH_RIGHT),
    )  # fmt: skip
    hand_spec = HandSpec(
        floating_base=True, target_names=TARGETS, target_link_names={n: _LINKS[n] for n in TARGETS},
        target_chains=tuple(("wrist", tip) for tip in FIVE_FINGERTIPS), root_target_name="wrist",
        contact_target_names=FIVE_FINGERTIPS, root_link_coord_spec=spec.root_link_coord_spec,
        target_coord_spec=spec.target_coord_spec,
    )  # fmt: skip
    identity = wp.array(np.eye(4, dtype=np.float32)[None], dtype=wp.mat44)
    base, fingers = (
        model.jnt_qposadr[model.joint("pos_x").id],
        [model.joint(n).id for n in robot.spec.actuated_joint_names],
    )
    refs = []
    for name, clip in clips.items():
        T_world_object = clip["qpos_obj_right"]
        q_wxyz_object_world = quaternion_invert(T_world_object[:, 3:])
        mesh = trimesh.load(str(root / object_dir(name) / "visual.obj"), force="mesh")
        objects = WarpScene(1, wp.get_device()).add(MeshGeom([mesh], np.array([0, 1], np.int32), poses=identity))
        points = np.concatenate([clip["qpos_wrist_right"][:, None, :3], clip["qpos_finger_right"][..., :3]], 1)
        points = quaternion_apply(q_wxyz_object_world[:, None], points - T_world_object[:, None, :3])
        q_wxyz_object_wrist = quaternion_multiply(q_wxyz_object_world, clip["qpos_wrist_right"][:, 3:])
        packed = HandRetargeterOffline(config, robot, hand_spec, scene=objects).solve_numpy(points, q_wxyz_object_wrist)
        ref = np.zeros((len(T_world_object), model.nq))
        ref[:, base : base + 3] = T_world_object[:, :3] + quaternion_apply(T_world_object[:, 3:], packed[:, :3])
        q_wxyz_world_base = quaternion_multiply(T_world_object[:, 3:], packed[:, 3:7])
        ref[:, base + 3 : base + 6] = np.unwrap(
            matrix_to_euler_angles(quaternion_to_matrix(q_wxyz_world_base), "rxyz"), axis=0
        )
        ref[:, model.jnt_qposadr[fingers]] = packed[:, 7:]
        quat = T_world_object[:, 3:]
        ref[:, -7:] = np.c_[T_world_object[:, :3], quat * np.sign(np.sum(quat * quat[:1], 1))[:, None]]
        refs.append(ref)
        print(f"retargeted {name}: {len(ref)} frames")

    # --- references: upsampled to the 100 Hz sim step, then held while the horizon runs past the clip ---
    end = min(len(r) for r in refs) - 1
    t = np.r_[np.arange(0, end, 0.5), np.full(200, end)]
    qpos_ref = np.stack([np.stack([np.interp(t, np.arange(len(r)), c) for c in r.T], 1) for r in refs])
    qpos_ref[..., -4:] /= np.linalg.norm(qpos_ref[..., -4:], axis=-1, keepdims=True)
    ctrl_ref = qpos_ref[..., model.jnt_qposadr[model.actuator_trnid[:, 0]]]

    # --- SimMPC: one batch of the four clips; each instance tracks its own hand and object references ---
    names = [model.joint(model.dof_jntid[i]).name for i in range(model.nv)]
    weight = [
        1.0 if n == "object" else 0.1 if n.startswith("pos_") else 0.03 if n.startswith("rot_") else 0.003
        for n in names
    ]
    weight[-3:] = [0.3] * 3  # object rotation
    cost = StateTask(qpos_ref, np.zeros((len(t), model.nv)), np.array(weight))
    ctrl_std = np.where([model.actuator(i).name[:4] in ("pos_", "rot_") for i in range(model.nu)], 0.02, 0.06)
    config = SimMPCConfig(
        batch_size=len(CLIPS), num_samples=256, horizon_steps=200, node_steps=40, ctrl_steps=40,
        stages=[SimMPCStage(8)],
        min_improvement=0.0, std_growth=2.0, noise_decay=0.93, terminal_scale=1.0, ctrl_std=ctrl_std,
    )  # fmt: skip
    config.add(cost)
    steps = (len(t) - config.horizon_steps) // config.ctrl_steps * config.ctrl_steps
    print("building SimMPC (the first run compiles kernels) ...")
    sim = MjWarpSim(specs, [1] * len(specs), nconmax=120, njmax=500)  # one scene per clip; plans on copies of it
    sim.data.qpos.assign(qpos_ref[:, 0])  # each clip's start
    mpc = SimMPC(config, sim, ctrl_ref)
    print(f"planning {steps / 100:.1f} s of motion for {len(CLIPS)} clips at once ...")
    start = time.perf_counter()
    report = lambda plans: print(f"  t={plans[0]['t0'] / 100:.1f} s: costs", *[f"{p['plan_cost']:.3f}" for p in plans])
    rollouts, _ = mpc.execute(sim, steps, callback=report)
    print(f"planned in {time.perf_counter() - start:.0f} s")
    for name, rollout, ref in zip(CLIPS, rollouts, qpos_ref):
        error = np.linalg.norm(rollout["qpos"][:, -7:-4] - ref[:steps, -7:-4], axis=1)
        print(f"{name}: object position error {100 * error.mean():.1f} cm mean, {100 * error.max():.1f} max")

    # --- viewer: the four clips side by side, replaying the simulated motion or the kinematic reference ---
    server = viser.ViserServer()
    print("replaying; Ctrl+C to stop")
    reference = server.gui.add_checkbox("Show kinematic reference", False)
    viewers = []
    for b, scene in enumerate(specs):
        x, y = qpos_ref[b, 0, -7:-5]
        server.scene.add_frame(f"/clip{b}", position=(-x, 0.5 * b - y, 0.0), show_axes=False)
        server.scene.add_grid(f"/clip{b}/table", width=0.4, height=0.4, position=(x, y, scene.geom("table").pos[2]))
        viewers.append(ViserMjcf(server, scene.to_xml(), root_node_name=f"/clip{b}"))
    while True:
        for k in range(0, steps, 2):
            for viewer, rollout, ref in zip(viewers, rollouts, qpos_ref):
                viewer.update_qpos(ref[k] if reference.value else rollout["qpos"][k])
            time.sleep(2 * model.opt.timestep)
        time.sleep(1.0)  # hold the last frame before replaying


if __name__ == "__main__":
    main()
