# /// script
# requires-python = ">=3.10"
# dependencies = ["robokit[mjwarp,pt-cpu]"]
#
# [tool.uv.sources]
# robokit = { path = "../../../..", editable = true }
# ///
# pyright: reportMissingImports=false
"""SimMPC: a Unitree G1 carries a box 1.4 m, tracking a human OMOMO clip that robokit retargets with its contacts."""

import time

import mujoco
import numpy as np
import trimesh
import viser
import warp as wp

from robokit.assets import fetch
from robokit.assets.datasets import omomo
from robokit.geom import MeshGeom, PlaneGeom, WarpScene
from robokit.helpers.humanoid_retargeting import HumanoidRetargeterOnline, build_interaction_mesh_frame
from robokit.helpers.humanoid_retargeting.loaders import load_omomo_clip
from robokit.helpers.humanoid_retargeting.presets.g1_smplh import G1_HEIGHT, g1_smplh
from robokit.helpers.sim_mpc import SimMPC, SimMPCConfig, SimMPCStage
from robokit.sim import MjWarpSim
from robokit.terms import StateTask
from robokit.utils.visualize_utils import ViserMjcf
from robokit.utils.warp_utils import wp_vec7
from robokit.xform.numpy import matrix_to_quaternion, quaternion_to_matrix


REPO, REVISION = "retarget/retarget_example", "8b0cdf2b24cc86d1c859050c5b0317ddbec59a63"  # G1 scene
G1 = "processed/amass/unitree_g1/humanoid/sprint/scene.xml"
SUBJECT, OBJECT = "sub3", "largebox"
CLIP = f"{SUBJECT}_{OBJECT}_003"  # a person picks a box off the floor, carries it and sets it down; 30 Hz
CAPSULES = {  # extra collision capsules on each side: forearm, wrist, shin, thigh
    "elbow": ([-0.01, 0, -0.01, 0.08, 0, -0.01], 0.035),
    "wrist_pitch": ([-0.01, 0, 0, 0.06, 0, 0], 0.035),
    "knee": ([0.01, 0, 0, 0.01, 0, -0.15], 0.045),
    "hip_yaw": ([0, 0, -0.03, -0.06, 0, -0.17], 0.055),
}


def main():
    wp.config.log_level = wp.LOG_WARNING  # no per-module load lines
    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"

    # --- data: the human's joints and the box's pose, scaled to the G1; the box at full size, resting on the floor ---
    robot = g1_smplh.load_robot()
    clip = str(omomo.clip_path(CLIP)), str(omomo.height_dict_path()), SUBJECT, g1_smplh.human_joint_names
    T_world_human, T_world_object_vec7, _, _ = load_omomo_clip(*clip, robot_height=G1_HEIGHT)
    box = trimesh.load(str(omomo.mesh_path(OBJECT)), force="mesh")
    assert isinstance(box, trimesh.Trimesh)
    print(f"loaded {len(T_world_human)} frames of {CLIP}")

    # --- retarget: an interaction-mesh solve per frame keeps the hands on the box ---
    points, _ = trimesh.sample.sample_surface_even(box, 100, seed=42)
    T_world_object = np.tile(np.eye(4, dtype=np.float32), (len(T_world_object_vec7), 1, 1))
    T_world_object[:, :3, :3] = quaternion_to_matrix(T_world_object_vec7[:, 3:])
    T_world_object[:, :3, 3] = T_world_object_vec7[:, :3]
    poses = wp.from_numpy(T_world_object[:1], dtype=wp.mat44, device=device)
    object_geom = MeshGeom([box], np.array([0, 1], dtype=np.int32), poses=poses)
    world = WarpScene(1, device).add(object_geom).add(PlaneGeom(np.array([0, 1], dtype=np.int32)))
    retargeter = HumanoidRetargeterOnline(g1_smplh, robot=robot, scene=world, device=device)
    keys = [retargeter.human_joint_names.index(entry.human_joint) for entry in g1_smplh.link_mapping.values()]
    frame = build_interaction_mesh_frame(T_world_human[0, keys, :3], points, T_world_object_vec7[0])
    retargeter.interaction_task.set_frame(*frame)
    retargeter.warmup(1)
    base = T_world_human[0, retargeter.human_joint_names.index(g1_smplh.human_root_name), :3]
    yaw = np.arctan2(*(T_world_object_vec7[0, :2] - base[:2])[::-1])
    T_world_base = np.array([*base, np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], dtype=np.float32)
    retargeter.reset(wp.from_numpy(T_world_base[None], dtype=wp_vec7, device=device))
    start, retargeted = time.perf_counter(), []
    for i in range(len(T_world_human)):
        object_geom.update(poses=wp.from_numpy(T_world_object[i : i + 1], dtype=wp.mat44, device=device))
        frame = build_interaction_mesh_frame(T_world_human[i, keys, :3], points, T_world_object_vec7[i])
        retargeter.interaction_task.set_frame(*frame)
        retargeted.append(retargeter.solve_numpy(T_world_human[i : i + 1])[0])
    print(f"retargeted in {time.perf_counter() - start:.0f} s")

    # --- scene: the G1 scene, where only feet and hands touch the floor, and a 1 kg cuboid fit to the box in its clip frame; it rests on the floor, and the feet, hands, arms, legs and torso touch it ---
    root = fetch([G1, "processed/amass/assets/robots/unitree_g1/**"], REVISION, REPO)
    scene = mujoco.MjSpec.from_file(str(root / G1))
    scene.meshdir = str((root / G1).parent / scene.meshdir)  # absolute, so the viewer's copy finds the meshes
    scene.option.timestep, scene.option.iterations, scene.option.ls_iterations = 0.02, 5, 10
    scene.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    T_cuboid_object, extents = trimesh.bounds.oriented_bounds(box)
    T_object_cuboid = np.linalg.inv(T_cuboid_object)
    body = scene.worldbody.add_body(name="box")
    body.add_freejoint()
    body.add_geom(
        name="box", type=mujoco.mjtGeom.mjGEOM_BOX, size=extents / 2, pos=T_object_cuboid[:3, 3],
        quat=matrix_to_quaternion(T_object_cuboid[:3, :3]), mass=1.0, contype=0, conaffinity=0, rgba=[0.8, 0.6, 0.4, 1],
    )  # fmt: skip
    touching = [p.geomname1 for p in scene.pairs]
    for side in ("left", "right"):
        for link, (fromto, radius) in CAPSULES.items():
            g = scene.body(f"{side}_{link}_link").add_geom(name=f"{side}_{link}", type=mujoco.mjtGeom.mjGEOM_CAPSULE)
            g.fromto, g.size, g.contype, g.conaffinity = fromto, [radius, 0, 0], 0, 0
            touching.append(g.name)
    g = scene.body("torso_link").add_geom(name="torso", type=mujoco.mjtGeom.mjGEOM_CAPSULE)
    g.fromto, g.size, g.contype, g.conaffinity = [0.01, 0, 0.08, 0.01, 0, 0.2], [0.07, 0, 0], 0, 0
    floor = scene.pairs[0]
    for name in [*touching, "torso", "floor"]:
        scene.add_pair(geomname1=name, geomname2="box", condim=3, friction=floor.friction, solref=floor.solref)
    model = scene.compile()
    print(f"scene: {model.nq} qpos, {model.nu} actuators")

    # --- reference: robot and box upsampled from 30 Hz to the 50 Hz sim step, held for a horizon past the end; the start is the first frame ---
    config = SimMPCConfig(
        num_samples=2048, horizon_steps=40, node_steps=5, ctrl_steps=5, stages=[SimMPCStage(16)], min_improvement=0.01,
        patience=1, std_growth=2.0, noise_decay=0.1 ** (1 / 16), terminal_scale=1.0, ctrl_std=0.05,
    )  # fmt: skip
    ref = np.c_[np.stack(retargeted), T_world_object_vec7]
    for a in (3, 39):  # no sign flips between neighboring quaternions
        ref[1:, a : a + 4] *= np.cumprod(np.sign(np.sum(ref[1:, a : a + 4] * ref[:-1, a : a + 4], 1)))[:, None]
    t = np.arange(0, len(ref) - 1, 0.6)
    qpos_ref = np.stack([np.interp(t, np.arange(len(ref)), c) for c in ref.T], 1)
    n = len(qpos_ref)
    qpos_ref = np.r_[qpos_ref, np.repeat(qpos_ref[-1:], config.horizon_steps, 0)]
    for a in (3, 39):
        qpos_ref[:, a : a + 4] /= np.linalg.norm(qpos_ref[:, a : a + 4], axis=1, keepdims=True)
    state = mujoco.MjData(model)
    state.qpos[:], state.ctrl[:] = qpos_ref[0], qpos_ref[0, 7:36]
    mujoco.mj_forward(model, state)

    # --- SimMPC: track the pelvis and box positions (3x), the joints and the box rotation ---
    weight = np.r_[3.0, 3.0, 3.0, np.ones(32), 3.0, 3.0, 3.0, 1.0, 1.0, 1.0]
    config.add(StateTask(qpos_ref, np.zeros((len(qpos_ref), model.nv)), weight))
    steps = (n - 1) // config.ctrl_steps * config.ctrl_steps
    print("building SimMPC (the first run compiles kernels) ...")
    sim = MjWarpSim(model, 1, state, nconmax=120, njmax=500)
    mpc = SimMPC(config, sim, qpos_ref[:, 7:36])
    print(f"planning {steps / 50:.1f} s of motion ...")
    start = time.perf_counter()
    report = lambda p: p[0]["t0"] % 50 or print(f"  t={p[0]['t0'] / 50:.1f} s: cost {p[0]['plan_cost']:.3f}")
    (rollout,), _ = mpc.execute(sim, steps, callback=report)

    # --- check: pelvis error, no fall, the box lifted and set down within 10 cm of its goal ---
    sim, ref = rollout["qpos"], qpos_ref[:steps]
    pelvis = np.linalg.norm(sim[:, :3] - ref[:, :3], axis=1).mean()
    box_error = np.linalg.norm(sim[:, 36:39] - ref[:, 36:39], axis=1).mean()
    goal = np.linalg.norm(sim[-1, 36:39] - qpos_ref[n - 1, 36:39])
    lowest, lift = sim[:, 2].min(), sim[:, 38].max() - sim[0, 38]
    ok = pelvis <= 0.1 and lowest >= ref[:, 2].min() - 0.1 and goal <= 0.1
    print(
        f"G1: pelvis error {100 * pelvis:.1f} cm mean, lowest pelvis {lowest:.2f} m; box error {100 * box_error:.1f} cm "
        f"mean, lifted {100 * lift:.0f} cm, {100 * goal:.1f} cm from its goal: {'pass' if ok else 'FAIL'}"
    )
    print(f"planned in {time.perf_counter() - start:.0f} s")

    # --- viewer: replay the simulated motion, or the kinematic reference it tracks ---
    server = viser.ViserServer()
    print("replaying; Ctrl+C to stop")
    server.scene.add_grid("/ground", width=4.0, height=4.0, position=(*qpos_ref[0, :2], 0.0))
    reference = server.gui.add_checkbox("Show kinematic reference", False)
    viewer = ViserMjcf(server, scene.to_xml())
    while True:
        for k in range(steps):
            viewer.update_qpos(qpos_ref[k] if reference.value else sim[k])
            time.sleep(model.opt.timestep)
        time.sleep(1.0)  # hold the last frame before replaying


if __name__ == "__main__":
    main()
