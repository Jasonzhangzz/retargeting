# /// script
# requires-python = ">=3.10"
# dependencies = ["robokit[mjwarp]"]
#
# [tool.uv.sources]
# robokit = { path = "../../../..", editable = true }
# ///
# pyright: reportMissingImports=false
# pyright: reportArgumentType=false
"""SimMPC: a Franka FR3 with a pusher stick pushes a T block on a table to a goal 15 cm away, turned 45 degrees."""

import time
from pathlib import Path

import mujoco
import numpy as np
import viser
import warp as wp
from robot_descriptions import fr3_mj_description

from robokit.helpers.sim_mpc import CartesianControl, SimMPC, SimMPCConfig, SimMPCStage
from robokit.sim import MjWarpSim
from robokit.terms import AlignTask, SensorTask, StateTask
from robokit.utils.visualize_utils import ViserMjcf
from robokit.xform.numpy import quaternion_apply, quaternion_invert, quaternion_multiply


STEPS = 1000  # 10 s at 100 Hz
# start: the arm holds the pusher tip 3.9 cm up at (0.696, -0.081); the block rests at (0.539, -0.064), turned 33 deg
QPOS0 = [0.21119, 0.86587, -0.39529, -1.54592, 0.41525, 2.32759, -1.91761, 0.5389, -0.0638, 0.031, 0.95804, 0, 0, 0.28662]  # fmt: skip
GOAL = [0.5, 0.08, 0.031, 0.99480, 0, 0, -0.10182]  # the block 14.9 cm away, turned 45 deg back (to -12 deg)


def main():
    wp.config.log_level = wp.LOG_WARNING  # no per-module load lines
    wp.init()

    # --- scene: the FR3 on joint-velocity servos with a pusher on its flange; a T block on a table ---
    scene = mujoco.MjSpec.from_file(fr3_mj_description.MJCF_PATH)
    scene.meshdir = str(Path(fr3_mj_description.MJCF_PATH).parent / scene.meshdir)  # absolute, for the viewer's copy
    scene.delete(scene.key("home"))
    for geom in [g for g in scene.geoms if g.group == 3]:  # arm collision meshes: only the pusher touches the block
        scene.delete(geom)
    for actuator, limit in zip(scene.actuators, [2.62] * 4 + [5.26, 4.18, 5.26]):  # rad/s
        actuator.gainprm[0], actuator.biasprm[:3] = 50.0, [0.0, 0.0, -50.0]
        actuator.inheritrange, actuator.ctrlrange = 0.0, [-limit, limit]
        actuator.forcelimited, actuator.forcerange = True, scene.joint(actuator.target).actfrcrange
    pusher = scene.body("fr3_link7").add_body(name="pusher", pos=[0, 0, 0.107])
    pusher.add_geom(
        name="stick", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.01, 0.01, 0.0625], pos=[0, 0, 0.0625], conaffinity=0
    )
    pusher.add_body(name="tip", pos=[0, 0, 0.125]).add_geom(
        size=[0.015, 0, 0], friction=[0.01, 0.005, 1e-4], solref=[0.03, 1.1], solimp=[0.5, 0.95, 0.002, 0.5, 2],
        conaffinity=0,
    )  # fmt: skip
    for body in scene.bodies[1:]:  # the arm's controller cancels gravity
        body.gravcomp = 1.0
    soft = dict(solref=[0.03, 1.0], solimp=[0.5, 0.95, 0.002, 0.5, 2])
    scene.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.35, 0.45, 0.0075], pos=[0.5, 0, -0.0015], friction=[0.6, 0.005, 1e-4],
        rgba=[0.65, 0.65, 0.65, 1], contype=2, conaffinity=0, **soft,
    )  # fmt: skip
    block = scene.worldbody.add_body(name="block")
    block.add_freejoint(name="block")
    bars = [([0.05, 0.025, 0.025], 0.0, 0.087), ([0.025, 0.08, 0.025], 0.075, 0.123)]  # half sizes, x, mass
    for size, x, mass in bars:
        block.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX, size=size, pos=[x, 0, 0], mass=mass, friction=[0.1, 0.001, 0.01],
            rgba=[0.9, 0.5, 0.1, 1], contype=0, conaffinity=3, **soft,
        )  # fmt: skip  # the block touches the pusher (bit 1) and the table (bit 2)
    frame = dict(type=mujoco.mjtSensor.mjSENS_FRAMEPOS, objtype=mujoco.mjtObj.mjOBJ_BODY)
    for i, xy in enumerate([(-0.03, 0), (0.075, 0.05), (0.075, -0.05)]):  # points on the block that draw the tip
        block.add_site(name=f"side_{i}", pos=[*xy, 0])
        scene.add_sensor(name=f"tip_to_side_{i}", objname="tip", reftype=mujoco.mjtObj.mjOBJ_SITE, refname=f"side_{i}", **frame)  # fmt: skip
    scene.add_sensor(name="block", objname="block", **frame)  # a body's frame sensors read its center of mass
    scene.add_sensor(name="tip", objname="tip", **frame)
    scene.option.timestep, scene.option.iterations, scene.option.ls_iterations = 0.01, 10, 16
    scene.option.cone, scene.option.jacobian = mujoco.mjtCone.mjCONE_ELLIPTIC, mujoco.mjtJacobian.mjJAC_DENSE
    model = scene.compile()
    state = mujoco.MjData(model)
    state.qpos[:] = QPOS0
    mujoco.mj_forward(model, state)
    print(f"scene: {model.nq} qpos, {model.nu} actuators")

    # --- SimMPC: sample the pusher's planar velocity, mapped to joint velocities that hold it upright 3.9 cm up; costs: the block's center to the goal's (50 per m) and its angle (5 per rad), the tip near the block and behind it as seen from the goal ---
    T = STEPS + 50  # past the end by one horizon
    goal_com = np.asarray(GOAL[:3]) + quaternion_apply(np.asarray(GOAL[3:]), model.body("block").ipos)
    sensor_ref = np.zeros((T, model.nsensordata))
    sensor_ref[:, model.sensor("block").adr[0] : model.sensor("block").adr[0] + 3] = goal_com
    qpos_ref, qvel_ref = np.tile(QPOS0[:7] + GOAL, (T, 1)), np.zeros((T, model.nv))
    control = CartesianControl(
        model, "tip", position_ref=[0, 0, 0.039], q_wxyz_ref=[0, 0.7071, 0.7071, 0], position_gain=[0, 0, 1],
        rotation_gain=[1, 1, 1],
    )  # fmt: skip
    ctrl_range = np.array([[-0.25, 0.25]] * 2 + [[0.0, 0.0]] * 5)  # x, y velocity in m/s; the other commands unused
    config = SimMPCConfig(
        num_samples=64, horizon_steps=50, node_steps=17, ctrl_steps=10, std_growth=1.0,
        stages=[SimMPCStage(1, [(0.25, None), (1.0, 0.02)])],
        interpolation="cubic", settle_steps=2, terminal_scale=1.0, ctrl_std=np.r_[0.3, 0.3, [0] * 5],
        ctrl_range=ctrl_range, control_map=control,
    )  # fmt: skip
    config.add(SensorTask(sensor_ref, {"block": 50.0}, norm=True), terminal_scale=501.0)
    config.add(StateTask(qpos_ref, qvel_ref, np.r_[[0] * 10, 5, 5, 5]), terminal_scale=501.0)
    config.add(SensorTask(sensor_ref, {f"tip_to_side_{i}": 0.05 for i in range(3)}, norm=True))
    config.add(AlignTask("block", "tip", np.tile(goal_com, (T, 1)), weight=0.9))
    print("building SimMPC (the first run compiles kernels) ...")
    sim = MjWarpSim(model, 1, state, nconmax=42, njmax=147)  # plans on copies of it, executes on it
    mpc = SimMPC(config, sim, np.zeros((T, model.nu)))
    print(f"planning {STEPS / 100:.0f} s of motion ...")
    start = time.perf_counter()
    report = lambda plans: plans[0]["t0"] % 200 or print(f"  t={plans[0]['t0'] / 100:.0f} s: cost {plans[0]['plan_cost']:.1f}")  # fmt: skip
    (rollout,), _ = mpc.execute(sim, STEPS, callback=report)
    planned = time.perf_counter() - start

    # --- check: over the last 1 s the block's center stays within 2 cm of the goal's, its angle within 0.2 rad ---
    q = rollout["qpos"][-100:, 7:]
    position = np.linalg.norm(q[:, :3] + quaternion_apply(q[:, 3:], model.body("block").ipos) - goal_com, axis=1)
    turn = quaternion_multiply(quaternion_invert(np.asarray(GOAL[3:])), q[:, 3:])
    angle = 2 * np.arctan2(np.linalg.norm(turn[:, 1:], axis=1), np.abs(turn[:, 0]))
    print(
        f"push: {'pass' if position.max() <= 0.02 and angle.max() <= 0.2 else 'fail'}, last 1 s at most "
        f"{100 * position.max():.1f} cm and {angle.max():.2f} rad off the goal (pass <= 2 cm, 0.2 rad)"
    )
    print(f"planned in {planned:.0f} s")

    # --- viewer: replay the simulated push with the goal pose as a green ghost ---
    server = viser.ViserServer()
    print("replaying; Ctrl+C to stop")
    server.scene.add_grid("/ground", width=1.0, height=1.0, position=(0.5, 0.0, -0.01))
    viewer = ViserMjcf(server, scene.to_xml())
    server.scene.add_frame("/goal", show_axes=False, position=GOAL[:3], wxyz=GOAL[3:])
    for size, x, _ in bars:
        server.scene.add_box(f"/goal/{x}", (40, 200, 40), 2 * np.asarray(size), opacity=0.3, position=(x, 0, 0))
    while True:
        for k in range(0, STEPS, 2):
            viewer.update_qpos(rollout["qpos"][k])
            time.sleep(2 * model.opt.timestep)
        time.sleep(1.0)  # hold the last frame before replaying


if __name__ == "__main__":
    main()
