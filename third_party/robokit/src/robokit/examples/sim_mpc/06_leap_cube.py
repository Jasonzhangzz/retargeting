# /// script
# requires-python = ">=3.10"
# dependencies = ["robokit[mjwarp]"]
#
# [tool.uv.sources]
# robokit = { path = "../../../..", editable = true }
# ///
# pyright: reportMissingImports=false
# pyright: reportArgumentType=false
"""SimMPC: a palm-up LEAP hand turns a cube in hand to one random goal after another."""

import time

import mujoco
import numpy as np
import viser
import warp as wp
from robot_descriptions import leap_hand_mj_description

from robokit.helpers.sim_mpc import SimMPC, SimMPCConfig, SimMPCStage
from robokit.sim import MjWarpSim
from robokit.terms import SensorTask
from robokit.utils.visualize_utils import ViserMjcf


STEPS = 2000  # 20 s at the 10 ms step
FACES = [([1, 0, 0], (220, 40, 40)), ([-1, 0, 0], (245, 140, 30)), ([0, 1, 0], (40, 170, 80)),
         ([0, -1, 0], (40, 100, 220)), ([0, 0, 1], (245, 245, 245)), ([0, 0, -1], (245, 215, 40))]  # fmt: skip


def main():
    wp.config.log_level = wp.LOG_WARNING  # no per-module load lines
    wp.init()

    # --- scene: Menagerie's LEAP hand tilted palm-up, soft weak position servos, capsule fingertips; a 7 cm cube ---
    scene = mujoco.MjSpec.from_file(leap_hand_mj_description.MJCF_PATH)
    for mesh in scene.meshes:  # absolute mesh paths, so the viewer's copy finds them
        mesh.file = f"{leap_hand_mj_description.PACKAGE_PATH}/{scene.meshdir}/{mesh.file}"
    scene.meshdir = ""
    o = scene.option
    o.timestep, o.iterations, o.ls_iterations, o.impratio = 0.01, 4, 10, 0.1
    o.cone, o.disableflags = mujoco.mjtCone.mjCONE_PYRAMIDAL, mujoco.mjtDisableBit.mjDSBL_EULERDAMP
    scene.body("palm").pos, scene.body("palm").quat = [0.124, -0.0305, -0.06045], [0, 1, 0, -0.175]
    for geom in scene.geoms:  # the hand touches itself, stiffly, and the cube; solmix 0 leaves those to the cube's
        if geom.conaffinity:  # collision geoms: contype 4 meets the hand, conaffinity 1 the cube, neither the floor
            geom.contype, geom.conaffinity, geom.margin = 4, 5, 0.002  # fingers push apart 2 mm out; see the cube's gap
        geom.friction, geom.solmix = [0.2, 0.005, 1e-4], 0.0
        geom.solref, geom.solimp[:3] = [0.02, 1.0], [0.99, 0.999, 0.0005]
        if geom.name.endswith("_tip"):  # the tip mesh stays as a visual; a capsule (no mesh to refit it to) touches
            thumb = geom.name == "th_tip"
            geom.parent.add_geom(meshname=geom.meshname, material=geom.material, contype=0, conaffinity=0, group=1,
                                 density=0, type=mujoco.mjtGeom.mjGEOM_MESH)  # fmt: skip
            geom.type, geom.meshname, geom.size[:2] = mujoco.mjtGeom.mjGEOM_CAPSULE, "", [0.012, 0.008]
            geom.friction = [0.7, 0.05, 2e-4]
            geom.pos, geom.quat = [0, -0.04, -0.015] if thumb else [0, -0.03, 0.015], [0.717, 0.717, 0, 0]
    for joint in scene.joints:
        joint.damping[0], joint.armature, joint.frictionloss = 0.05, 0.001, 0.0
    for actuator in scene.actuators:  # kp 0.2, kv 0.01, |force| <= 1
        actuator.gainprm[0], actuator.biasprm[:3], actuator.forcerange = 0.2, [0, -0.2, -0.01], [-1, 1]
        actuator.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
    scene.worldbody.add_site(name="grasp", pos=[0.11, 0, 0.03])  # where the cube should stay
    cube = scene.worldbody.add_body(name="cube", pos=[0.11, 0, 0.1])
    cube.add_freejoint(name="cube")
    cube.add_geom(
        name="cube", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.035] * 3, mass=0.108, friction=[0.3, 0.05, 0.01], contype=1,
        conaffinity=2, solref=[0.01, 1.5], solimp=[0.0, 0.95, 0.001, 0.5, 2], rgba=[0.9, 0.5, 0.1, 1],
        gap=0.002,  # cancels the hand's margin: hand-cube contacts still act at contact
    )  # fmt: skip
    for axis, color in FACES:  # a coloured plate on each face shows the orientation; no contact, no mass
        cube.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=np.where(axis, 5e-4, 0.035), pos=0.0345 * np.array(axis),
                      contype=0, conaffinity=0, group=1, density=0, rgba=[*np.array(color) / 255, 1])  # fmt: skip
    scene.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 1], pos=[0, 0, -0.25], contype=2)
    scene.add_sensor(name="position", type=mujoco.mjtSensor.mjSENS_FRAMEPOS, objtype=mujoco.mjtObj.mjOBJ_BODY,
                     objname="cube", reftype=mujoco.mjtObj.mjOBJ_SITE, refname="grasp")  # fmt: skip
    scene.add_sensor(name="quat", type=mujoco.mjtSensor.mjSENS_FRAMEQUAT, objtype=mujoco.mjtObj.mjOBJ_BODY,
                     objname="cube")  # fmt: skip
    model = scene.compile()
    print(f"scene: {model.nq} qpos, {model.nu} actuators")

    # --- goals: uniform on SO(3), each at least 90 deg from the last; the planner reads the current one in place ---
    rng = np.random.default_rng(0)

    def draw(last: np.ndarray) -> np.ndarray:
        while True:
            q = rng.normal(size=4)
            q /= np.linalg.norm(q)
            if 2 * np.arccos(min(abs(q @ last), 1.0)) >= np.pi / 2:
                return q

    goal, a = draw(np.array([1.0, 0, 0, 0])), model.sensor("quat").adr[0]
    ref = np.zeros((1, 1, model.nsensordata), np.float32)
    ref[0, 0, a : a + 4] = goal
    sensor_ref = wp.array(ref, dtype=float)

    # --- SimMPC: hold the cube over the palm, turn it gently toward the goal; a 0.8 s plan refined 3 times ---
    cupped = np.where(np.isin(np.arange(model.nu) % 4, [1]) & (np.arange(model.nu) < 12), 0.0, 0.8)  # rest pose
    cupped[15] = 0.0  # thumb tip joint
    ctrl_ref = np.tile(cupped, (STEPS + 101, 1))
    config = SimMPCConfig(
        num_samples=64, horizon_steps=80, node_steps=27, ctrl_steps=5, interpolation="cubic",
        stages=[SimMPCStage(3, [(0.25, None), (1.0, 0.02)])],
        terminal_scale=1.0, ctrl_std=0.2, qpos_std=0.005,
    )  # fmt: skip
    config.add(SensorTask(sensor_ref, dict(quat=0.05), residual="quaternion", freeze_reference=True))
    config.add(SensorTask(sensor_ref, dict(position=50.0), freeze_reference=True))
    state = mujoco.MjData(model)
    state.qpos[:16], state.ctrl[:] = cupped, cupped
    mujoco.mj_forward(model, state)
    print("building SimMPC (the first run compiles kernels) ...")
    sim = MjWarpSim(model, 1, state, nconmax=64, njmax=512)
    mpc = SimMPC(config, sim, ctrl_ref)

    # --- after each control period: a cube within 0.4 rad of its goal gets the next one ---
    goals, reached = [], 0

    def next_goal(plan: list):
        nonlocal goal, reached
        goals.extend([goal] * config.ctrl_steps)
        q = sim.data.qpos.numpy()[0, -4:]
        if 2 * np.arccos(min(abs(q @ goal), 1.0)) < 0.4:
            reached += 1
            goal = draw(goal)
            ref[0, 0, a : a + 4] = goal
            sensor_ref.assign(ref)

    print(f"planning {STEPS / 100:.0f} s of motion ...")
    start = time.perf_counter()
    (rollout,), _ = mpc.execute(sim, STEPS, callback=next_goal)
    dropped = bool(np.any(rollout["qpos"][:, -5] < -0.05))
    print(f"cube: {'pass' if reached >= 3 and not dropped else 'fail'}, {reached} goals reached in "
          f"{STEPS / 100:.0f} s, {'dropped' if dropped else 'never dropped'}")  # fmt: skip
    print(f"planned in {time.perf_counter() - start:.0f} s")

    # --- viewer: replay the simulated motion with the current goal as a ghost cube beside the hand ---
    server = viser.ViserServer()
    print("replaying; Ctrl+C to stop")
    server.scene.add_grid("/ground", width=1.0, height=1.0, position=(0.1, 0.0, -0.25))
    viewer = ViserMjcf(server, scene.to_xml())
    ghost = server.scene.add_frame("/goal", show_axes=False, position=(0.11, 0.15, 0.1))
    for i, (axis, color) in enumerate(FACES):
        dimensions, position = np.where(axis, 1e-3, 0.07), 0.0345 * np.array(axis)
        server.scene.add_box(f"/goal/{i}", color=color, dimensions=dimensions, position=position)
    while True:
        for k in range(0, STEPS, 2):
            viewer.update_qpos(rollout["qpos"][k])
            ghost.wxyz = goals[k]
            time.sleep(2 * model.opt.timestep)
        time.sleep(1.0)  # hold the last frame before replaying


if __name__ == "__main__":
    main()
