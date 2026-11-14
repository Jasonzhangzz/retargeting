# /// script
# requires-python = ">=3.10"
# dependencies = ["robokit[mjwarp]"]
#
# [tool.uv.sources]
# robokit = { path = "../../../..", editable = true }
# ///
# pyright: reportMissingImports=false
"""SimMPC: a Unitree G1 climbs a 0.7 m box on hands and knees, tracking human mocap that robokit retargets."""

import time

import mujoco
import numpy as np
import trimesh
import viser
import warp as wp

from robokit.assets import fetch
from robokit.assets.datasets import climb
from robokit.geom import MeshGeom, PlaneGeom, WarpScene
from robokit.helpers.humanoid_retargeting import HumanoidRetargeterOnline, build_interaction_mesh_frame
from robokit.helpers.humanoid_retargeting.loaders import load_mocap_clip
from robokit.helpers.humanoid_retargeting.presets.g1_mocap import G1_HEIGHT, g1_mocap
from robokit.helpers.sim_mpc import SimMPC, SimMPCConfig, SimMPCStage
from robokit.sim import MjWarpSim
from robokit.terms import SensorTask, StateTask
from robokit.utils.visualize_utils import ViserMjcf
from robokit.utils.warp_utils import wp_vec7
from robokit.xform.numpy import matrix_to_quaternion


REPO, REVISION = "retarget/retarget_example", "8b0cdf2b24cc86d1c859050c5b0317ddbec59a63"  # G1 scene
G1 = "processed/amass/unitree_g1/humanoid/sprint/scene.xml"
CLIP, START, FRAMES = "mocap_climb_seq_1", 84, 300  # at 30 Hz: hands on a high box, a person climbs it from a low step
SITES = ("left_foot", "right_foot", "left_palm", "right_palm")  # feet and palms, tracked in the world


def main():
    wp.config.log_level = wp.LOG_WARNING  # no per-module load lines
    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"

    # --- data: the human's joints and the boxes, scaled to the G1 ---
    clip = climb.sequence_dir(CLIP)
    robot = g1_mocap.load_robot()
    npy = str(next(clip.glob("*.npy")))
    T_world_human, _, scale = load_mocap_clip(npy, g1_mocap.human_joint_names, robot_height=G1_HEIGHT)
    T_world_human = T_world_human[:FRAMES]
    terrain = trimesh.load(str(clip / "multi_boxes.obj"), force="mesh")
    assert isinstance(terrain, trimesh.Trimesh)
    terrain.apply_scale(scale)
    print(f"loaded {FRAMES} frames of {CLIP}")

    # --- retarget: an interaction-mesh solve per frame keeps the feet and hands on the boxes ---
    points, _ = trimesh.sample.sample_surface_even(terrain, 100, seed=42)
    masks = np.array([0, 1], dtype=np.int32)
    world = WarpScene(1, device).add(MeshGeom([terrain], masks)).add(PlaneGeom(masks))
    retargeter = HumanoidRetargeterOnline(g1_mocap, robot=robot, scene=world, device=device)
    keys = [retargeter.human_joint_names.index(entry.human_joint) for entry in g1_mocap.link_mapping.values()]
    T_world_terrain = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    retargeter.interaction_task.set_frame(
        *build_interaction_mesh_frame(T_world_human[0, keys, :3], points, T_world_terrain)
    )
    retargeter.warmup(1)
    base = T_world_human[0, retargeter.human_joint_names.index(g1_mocap.human_root_name), :3]
    yaw = np.arctan2(*(points.mean(0)[:2] - base[:2])[::-1])
    T_world_base = np.array([*base, np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], dtype=np.float32)
    retargeter.reset(wp.from_numpy(T_world_base[None], dtype=wp_vec7, device=device))
    start, retargeted = time.perf_counter(), []
    for frame in T_world_human:
        retargeter.interaction_task.set_frame(*build_interaction_mesh_frame(frame[keys, :3], points, T_world_terrain))
        retargeted.append(retargeter.solve_numpy(frame[None])[0])
    print(f"retargeted in {time.perf_counter() - start:.0f} s")

    # --- scene: the G1 scene, where only feet and hands touch the floor, and a cuboid fit to each box, which feet, hands and knees touch like the floor; position sensors on the feet and palms ---
    root = fetch([G1, "processed/amass/assets/robots/unitree_g1/**"], REVISION, REPO)
    scene = mujoco.MjSpec.from_file(str(root / G1))
    scene.meshdir = str((root / G1).parent / scene.meshdir)  # absolute, so the viewer's copy finds the meshes
    scene.option.timestep, scene.option.iterations, scene.option.ls_iterations = 0.02, 5, 10
    scene.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    for side in ("left", "right"):
        knee = scene.body(f"{side}_knee_link").add_geom(name=f"{side}_knee", type=mujoco.mjtGeom.mjGEOM_CAPSULE)
        knee.fromto, knee.size, knee.contype, knee.conaffinity = [0.01, 0, 0, 0.01, 0, -0.15], [0.045, 0, 0], 0, 0
    floor, touching = scene.pairs[0], [p.geomname1 for p in scene.pairs] + ["left_knee", "right_knee"]
    for i, path in enumerate(sorted(clip.glob("box_models/*.obj"))):
        mesh = trimesh.load(str(path), force="mesh")
        mesh.apply_scale(scale)
        T_cuboid_world, extents = trimesh.bounds.oriented_bounds(mesh)
        T_world_cuboid = np.linalg.inv(T_cuboid_world)
        scene.worldbody.add_geom(
            name=f"box{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=extents / 2, pos=T_world_cuboid[:3, 3],
            quat=matrix_to_quaternion(T_world_cuboid[:3, :3]), contype=0, conaffinity=0, rgba=[0.55, 0.6, 0.7, 1],
        )  # fmt: skip
        for g in touching:
            scene.add_pair(geomname1=g, geomname2=f"box{i}", condim=3, friction=floor.friction, solref=floor.solref)
    for site in SITES:
        scene.add_sensor(
            name=site, type=mujoco.mjtSensor.mjSENS_FRAMEPOS, objtype=mujoco.mjtObj.mjOBJ_SITE, objname=site
        )
    model = scene.compile()
    print(f"scene: {model.nq} qpos, {model.nu} actuators")

    # --- reference: upsampled from 30 Hz to the 50 Hz sim step, held for a horizon past the end ---
    config = SimMPCConfig(
        num_samples=2048, horizon_steps=40, node_steps=5, ctrl_steps=5, stages=[SimMPCStage(16)], min_improvement=0.01,
        patience=1, std_growth=2.0, noise_decay=0.1 ** (1 / 16), terminal_scale=1.0, ctrl_std=0.05,
    )  # fmt: skip
    ref = np.stack(retargeted)[START:]  # retargeted from the first frame, so the start pose has settled
    ref[1:, 3:7] *= np.cumprod(np.sign(np.sum(ref[1:, 3:7] * ref[:-1, 3:7], 1)))[:, None]  # no quaternion sign flips
    t = np.arange(0, len(ref) - 1, 0.6)
    qpos_ref = np.stack([np.interp(t, np.arange(len(ref)), c) for c in ref.T], 1)
    n = len(qpos_ref)
    qpos_ref = np.r_[qpos_ref, np.repeat(qpos_ref[-1:], config.horizon_steps, 0)]
    qpos_ref[:, 3:7] /= np.linalg.norm(qpos_ref[:, 3:7], axis=1, keepdims=True)
    state, sensor_ref = mujoco.MjData(model), []
    for q in qpos_ref:  # the feet and palms where the reference puts them
        state.qpos[:] = q
        mujoco.mj_forward(model, state)
        sensor_ref.append(state.sensordata.copy())
    state.qpos[:], state.ctrl[:] = qpos_ref[0], qpos_ref[0, 7:]
    mujoco.mj_forward(model, state)

    # --- SimMPC: track the pelvis (position weighted 3x) and joints, and the feet and palms in the world, so a foot clears a box edge where the reference lifts it; the reference joints are the commands ---
    config.add(StateTask(qpos_ref, np.zeros((len(qpos_ref), model.nv)), np.r_[3.0, 3.0, 3.0, np.ones(model.nv - 3)]))
    config.add(SensorTask(np.array(sensor_ref), dict.fromkeys(SITES, 10.0)))
    steps = (n - 1) // config.ctrl_steps * config.ctrl_steps
    print("building SimMPC (the first run compiles kernels) ...")
    sim = MjWarpSim(model, 1, state, nconmax=120, njmax=500)
    mpc = SimMPC(config, sim, qpos_ref[:, 7:])
    print(f"planning {steps / 50:.1f} s of motion ...")
    start = time.perf_counter()
    report = lambda p: p[0]["t0"] % 50 or print(f"  t={p[0]['t0'] / 50:.1f} s: cost {p[0]['plan_cost']:.3f}")
    (rollout,), _ = mpc.execute(sim, steps, callback=report)

    # --- check: pelvis and joint errors; no fall; standing on the box at the end ---
    sim, ref = rollout["qpos"], qpos_ref[:steps]
    position = np.linalg.norm(sim[:, :3] - ref[:, :3], axis=1).mean()
    rotation = 2 * np.arccos(np.clip(np.abs(np.sum(sim[:, 3:7] * ref[:, 3:7], 1)), 0, 1)).mean()
    joints = np.linalg.norm(sim[:, 7:] - ref[:, 7:], axis=1).mean()
    lowest, final = sim[:, 2].min(), sim[-1, 2]
    ok = position <= 0.1 and rotation <= 0.5 and lowest >= 0.5 and final >= ref[-1, 2] - 0.1
    print(
        f"G1: pelvis error {100 * position:.1f} cm, {rotation:.2f} rad mean; joint error {joints:.2f} rad mean; "
        f"lowest pelvis {lowest:.2f} m, final {final:.2f} m: {'pass' if ok else 'FAIL'}"
    )
    print(f"planned in {time.perf_counter() - start:.0f} s")

    # --- viewer: replay the simulated motion, or the kinematic reference it tracks ---
    server = viser.ViserServer()
    print("replaying; Ctrl+C to stop")
    server.scene.add_grid("/ground", width=6.0, height=6.0)
    reference = server.gui.add_checkbox("Show kinematic reference", False)
    viewer = ViserMjcf(server, scene.to_xml())
    while True:
        for k in range(steps):
            viewer.update_qpos(qpos_ref[k] if reference.value else sim[k])
            time.sleep(model.opt.timestep)
        time.sleep(1.0)  # hold the last frame before replaying


if __name__ == "__main__":
    main()
