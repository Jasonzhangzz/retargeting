# /// script
# requires-python = ">=3.10"
# dependencies = ["robokit[mjwarp]"]
#
# [tool.uv.sources]
# robokit = { path = "../../../..", editable = true }
# ///
# pyright: reportMissingImports=false
# pyright: reportArgumentType=false
"""SimMPC: a Sharpa hand lifts an OakInk object, tracking a human clip that robokit retargets kinematically.

SPIDER: https://arxiv.org/abs/2511.09484.
"""

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


REPO, REVISION = "retarget/retarget_full", "b6050b02146fc412d3db8cd539634dc71a117474"  # SPIDER's released data
CLIP = "oakink_0000_right_S10021_00"  # a human right hand lifts an object 30 cm; 20 frames, 80 ms apart
HUMAN, SCENE = f"processed/oakink_lifted/mano/right/{CLIP}/0", f"processed/oakink_lifted/sharpa/right/{CLIP}"
OBJECT = f"processed/oakink_lifted/assets/objects/{CLIP}"
TARGETS = ("wrist", *FIVE_FINGERTIPS)


def main():
    wp.config.log_level = wp.LOG_WARNING  # no per-module load lines
    wp.init()

    # --- data: the human hand's wrist pose and five fingertips, the object's pose, and SPIDER's Sharpa scene ---
    files = [f"{HUMAN}/trajectory_keypoints.npz", f"{SCENE}/scene.xml", f"{OBJECT}/**"]
    root = fetch([*files, "processed/oakink_lifted/assets/robots/sharpa/**"], REVISION, REPO)
    clip = np.load(root / HUMAN / "trajectory_keypoints.npz")
    print(f"loaded {len(clip['qpos_obj_right'])} frames of {CLIP}")

    # --- scene: the Sharpa on six position-controlled base joints, the object free on a small plate; SPIDER's contact pairs, and its run's solver settings ---
    scene = mujoco.MjSpec.from_file(str(root / SCENE / "scene.xml"))
    scene.meshdir = f"{(root / SCENE / scene.meshdir).resolve()}/"  # absolute, for the viewer's copy
    scene.option.iterations, scene.option.integrator = 20, mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model = scene.compile()
    print(f"scene: {model.nq} qpos, {model.nu} actuators")

    # --- retarget: the wrist and fingertips to the Sharpa's base pose and 22 joints, in the object's frame, where a collision term keeps the hand out of the static object mesh ---
    T_world_object = clip["qpos_obj_right"]
    q_wxyz_object_world = quaternion_invert(T_world_object[:, 3:])
    identity = wp.array(np.eye(4, dtype=np.float32)[None], dtype=wp.mat44)
    obj = trimesh.load(str(root / OBJECT / "visual.obj"), force="mesh")
    mesh = MeshGeom([obj], np.array([0, 1], np.int32), poses=identity)
    object_scene = WarpScene(1, wp.get_device()).add(mesh)
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
    points = np.concatenate([clip["qpos_wrist_right"][:, None, :3], clip["qpos_finger_right"][..., :3]], 1)
    points = quaternion_apply(q_wxyz_object_world[:, None], points - T_world_object[:, None, :3])
    q_wxyz_object_wrist = quaternion_multiply(q_wxyz_object_world, clip["qpos_wrist_right"][:, 3:])
    retargeter = HandRetargeterOffline(config, robot, hand_spec, scene=object_scene)
    packed = retargeter.solve_numpy(points, q_wxyz_object_wrist)
    ref = np.zeros((len(T_world_object), model.nq))
    t_world_base = T_world_object[:, :3] + quaternion_apply(T_world_object[:, 3:], packed[:, :3])
    ref[:, :3] = t_world_base - model.body("right_hand_base").pos  # the base joints slide from the base body
    q_wxyz_world_base = quaternion_multiply(T_world_object[:, 3:], packed[:, 3:7])
    ref[:, 3:6] = np.unwrap(matrix_to_euler_angles(quaternion_to_matrix(q_wxyz_world_base), "rxyz"), axis=0)
    ref[:, model.jnt_qposadr[[model.joint(n).id for n in robot.spec.actuated_joint_names]]] = packed[:, 7:]
    quat = T_world_object[:, 3:]
    ref[:, -7:] = np.c_[T_world_object[:, :3], quat * np.sign(np.sum(quat * quat[:1], 1))[:, None]]

    # --- reference: upsampled to the 10 ms sim step, then held while the horizon runs past the clip ---
    end = len(ref) - 1
    t = np.r_[np.arange(0, end, 0.125), np.full(200, end)]
    qpos_ref = np.stack([np.interp(t, np.arange(len(ref)), c) for c in ref.T], 1)
    qpos_ref[:, -4:] /= np.linalg.norm(qpos_ref[:, -4:], axis=1, keepdims=True)
    ctrl_ref = qpos_ref[:, model.jnt_qposadr[model.actuator_trnid[:, 0]]]
    state = mujoco.MjData(model)
    state.qpos[:], state.ctrl[:] = qpos_ref[0], ctrl_ref[0]
    mujoco.mj_forward(model, state)

    # --- SimMPC: track the hand and the object, weighting the object most; SPIDER's sampling budget ---
    names = [model.joint(model.dof_jntid[i]).name for i in range(model.nv)]
    weight = [1.0 if "object" in n else 0.1 if "_pos_" in n else 0.03 if "_rot_" in n else 0.003 for n in names]
    weight[-3:] = [0.3] * 3  # object rotation
    cost = StateTask(qpos_ref, np.zeros((len(qpos_ref), model.nv)), np.array(weight))
    ctrl_std = np.where(np.arange(model.nu) < 6, 0.02, 0.2)  # base, fingers
    config = SimMPCConfig(
        num_samples=1024, horizon_steps=200, node_steps=50, ctrl_steps=25,
        stages=[SimMPCStage(15, [(0.5, None), (1.0, 0.02)]), SimMPCStage(1)],  # 1.28x faster
        min_improvement=-1.0,
        std_growth=2.0, noise_decay=0.75, terminal_scale=1.0, ctrl_std=ctrl_std,
    ).add(cost)  # fmt: skip
    steps = (len(qpos_ref) - config.horizon_steps) // config.ctrl_steps * config.ctrl_steps
    print("building SimMPC (the first run compiles kernels) ...")
    sim = MjWarpSim(model, 1, state, nconmax=120, njmax=350)
    mpc = SimMPC(config, sim, ctrl_ref)
    print(f"planning {steps / 100:.1f} s of motion ...")
    start = time.perf_counter()
    report = lambda plans: plans[0]["t0"] % 50 or print(f"  t={plans[0]['t0'] / 100:.1f} s: cost {plans[0]['plan_cost']:.3f}")  # fmt: skip
    (rollout,), _ = mpc.execute(sim, steps, callback=report)

    # --- check: SPIDER's success, a mean object error within 10 cm and 0.5 rad ---
    q, q_ref = rollout["qpos"][:, -4:], qpos_ref[:steps, -4:]
    position = np.linalg.norm(rollout["qpos"][:, -7:-4] - qpos_ref[:steps, -7:-4], axis=1).mean()
    rotation = np.mean(2 * np.arccos(np.clip(np.abs(np.sum(q * q_ref, 1)), 0, 1)))
    lift = rollout["qpos"][:, -5].max() - qpos_ref[0, -5]
    print(
        f"object: {'pass' if position <= 0.1 and rotation <= 0.5 else 'fail'}, lifted {100 * lift:.1f} cm, "
        f"mean error {100 * position:.1f} cm (<= 10) and {rotation:.2f} rad (<= 0.5)"
    )
    print(f"planned in {time.perf_counter() - start:.0f} s")

    # --- viewer: replay the simulated motion, or the kinematic reference it tracks ---
    server = viser.ViserServer()
    print("replaying; Ctrl+C to stop")
    server.scene.add_grid("/ground", width=1.0, height=1.0)
    reference = server.gui.add_checkbox("Show kinematic reference", False)
    viewer = ViserMjcf(server, scene.to_xml())
    while True:
        for k in range(0, steps, 2):
            viewer.update_qpos(qpos_ref[k] if reference.value else rollout["qpos"][k])
            time.sleep(2 * model.opt.timestep)
        time.sleep(1.0)  # hold the last frame before replaying


if __name__ == "__main__":
    main()
