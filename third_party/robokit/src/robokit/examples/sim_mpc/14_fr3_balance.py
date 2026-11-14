# /// script
# requires-python = ">=3.10"
# dependencies = ["robokit[mjwarp]"]
#
# [tool.uv.sources]
# robokit = { path = "../../../..", editable = true }
# ///
# pyright: reportMissingImports=false
# pyright: reportArgumentType=false
"""SimMPC: a Franka FR3 tilts a plate on its flange to roll a ball after a drifting goal point on the plate."""

import time
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np
import tyro
import viser
import warp as wp
from robot_descriptions import fr3_mj_description

from robokit.helpers.sim_mpc import SimMPC, SimMPCConfig, SimMPCStage
from robokit.sim import MjWarpSim
from robokit.terms import SensorTask, StateTask
from robokit.utils.visualize_utils import ViserMjcf


STEPS, PHASE, REST = 1000, 11.458, 0.04415  # 10 s at 100 Hz; drift phase (s); the ball's resting height on the plate
# start (seed 11): the arm holds the plate level with the ball resting on it
QPOS0 = [1.37404, -1.40987, -0.97431, -2.5512, 2.16327, 1.74726, 2.01986, 0.30211, 0.02328, 0.74355, -0.47285, 0, 0, -0.88115]  # fmt: skip
PASS = 0.0061  # accumulated squared error a 2,800-sample MPPI reaches from this start in 10 s


def main(planner: Literal["mppi", "ilqr"] = "mppi"):
    """Balance the ball with MPPI or iLQR (`--planner ilqr`) on the same costs."""
    wp.config.log_level = wp.LOG_WARNING  # no per-module load lines
    wp.init()

    # --- scene: the FR3 on joint-velocity servos with a plate on its flange; a ball on the plate ---
    scene = mujoco.MjSpec.from_file(fr3_mj_description.MJCF_PATH)
    scene.meshdir = str(Path(fr3_mj_description.MJCF_PATH).parent / scene.meshdir)  # absolute, for the viewer's copy
    scene.delete(scene.key("home"))
    for geom in [g for g in scene.geoms if g.group == 3]:  # arm collision meshes: only the plate touches the ball
        scene.delete(geom)
    for actuator, limit in zip(scene.actuators, [2.62] * 4 + [5.26, 4.18, 5.26]):  # rad/s
        actuator.gainprm[0], actuator.biasprm[:3] = 50.0, [0.0, 0.0, -50.0]
        actuator.inheritrange, actuator.ctrlrange = 0.0, [-limit, limit]
        actuator.forcelimited, actuator.forcerange = True, scene.joint(actuator.target).actfrcrange
    mount = scene.body("fr3_link7").add_body(name="mount", pos=[0, 0, 0.107])
    mount.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.018, 0.025, 0], pos=[0, 0, 0.025], mass=0.125, contype=0,
        conaffinity=0, rgba=[0.1, 0.1, 0.1, 1],
    )  # fmt: skip
    mount.add_body(name="plate", pos=[0, 0, 0.05025], quat=[0.930418, 0, 0, -0.366501]).add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.195, 0.25, 0.00425], mass=1.0, friction=[0.3, 0.005, 0.002],
        solimp=[0.0, 0.95, 0.005, 0.5, 2.0], rgba=[0.6, 0.6, 0.65, 1],
    )  # fmt: skip
    for body in scene.bodies[1:]:  # the arm's controller cancels gravity
        body.gravcomp = 1.0
    ball = scene.worldbody.add_body(name="ball")
    ball.add_freejoint(name="ball")
    ball.add_geom(
        size=[0.04, 0, 0], mass=0.1, friction=[0.06, 0.001, 0.002], condim=6, solimp=[0.9, 0.99, 0.001, 0.5, 2],
        rgba=[0.9, 0.5, 0.1, 1],
    )  # fmt: skip  # condim 6: the ball rolls, against rolling friction
    scene.add_sensor(
        name="ball_on_plate", type=mujoco.mjtSensor.mjSENS_FRAMEPOS, objtype=mujoco.mjtObj.mjOBJ_BODY, objname="ball",
        reftype=mujoco.mjtObj.mjOBJ_BODY, refname="plate",
    )  # fmt: skip
    scene.option.timestep, scene.option.iterations, scene.option.ls_iterations = 0.01, 16, 20
    scene.option.cone, scene.option.jacobian = mujoco.mjtCone.mjCONE_ELLIPTIC, mujoco.mjtJacobian.mjJAC_DENSE
    model = scene.compile()
    state = mujoco.MjData(model)
    state.qpos[:] = QPOS0
    mujoco.mj_forward(model, state)
    print(f"scene: {model.nq} qpos, {model.nu} actuators")

    # --- goal: a point on the plate drifting on a slow sine around (0.05, 0.04), in the plate's frame ---
    t = PHASE + np.arange(STEPS + 60) * model.opt.timestep  # past the end by one horizon
    drift = 0.04 * np.sin(2 * np.pi * 0.06 * t)
    goal = np.c_[0.05 + drift, 0.04 + drift, np.full_like(t, REST)]

    # --- SimMPC: sample joint velocities (iLQR: refine them, each held 0.05 s, slower, over a shorter horizon); costs: the ball's squared distance to the goal (30 per m^2) and the arm's distance to its level-plate pose (15 per rad); each plan holds the goal it starts from ---
    held, qpos_ref, qvel_ref = dict(freeze_reference=True), np.tile(QPOS0, (len(t), 1)), np.zeros((len(t), model.nv))
    ilqr = planner == "ilqr"
    config = SimMPCConfig(
        num_samples=1400, horizon_steps=40 if ilqr else 60, node_steps=5 if ilqr else 20, ctrl_steps=5,
        stages=[SimMPCStage(1)] if ilqr else [SimMPCStage(2, [(0.25, None), (1.0, 0.02)])], std_growth=1.0,
        interpolation="cubic", terminal_scale=1.0, ctrl_std=0.3,
        ctrl_range=(0.3 if ilqr else 0.6) * model.actuator_ctrlrange, planner=planner,  # of the joint speed limits
    )  # fmt: skip
    config.add(SensorTask(goal, {"ball_on_plate": 30.0}, **held), terminal_scale=701.0)
    config.add(StateTask(qpos_ref, qvel_ref, np.r_[[15] * 7, [0] * 6], **held))
    print("building SimMPC (the first run compiles kernels) ...")
    sim = MjWarpSim(model, 1, state, nconmax=3, njmax=39)  # plans on copies of it, executes on it
    mpc = SimMPC(config, sim, np.zeros((len(t), model.nu)))
    print(f"planning {STEPS / 100:.0f} s of motion ...")
    start = time.perf_counter()
    report = lambda plans: plans[0]["t0"] % 200 or print(f"  t={plans[0]['t0'] / 100:.0f} s: cost {plans[0]['plan_cost']:.2f}")  # fmt: skip
    (rollout,), _ = mpc.execute(sim, STEPS, callback=report)
    planned = time.perf_counter() - start

    # --- check: the ball never drops 16 cm below the plate; its squared distance to the goal, every 0.05 s x 0.05 s ---
    ball_on_plate = rollout["sensordata"][::5]
    error = 0.05 * np.sum((ball_on_plate - goal[::5][: len(ball_on_plate)]) ** 2)
    fell = bool(np.any(ball_on_plate[:, 2] < REST - 0.16))
    distance = np.linalg.norm(ball_on_plate[:, :2] - goal[::5][: len(ball_on_plate), :2], axis=1)
    print(
        f"balance: {'pass' if error <= PASS and not fell else 'fail'}, {'fell' if fell else 'kept the ball'}, "
        f"accumulated squared error {error:.4f} (pass <= {PASS}), {100 * distance.mean():.1f} cm from the goal on average"
    )
    print(f"planned in {planned:.0f} s")

    # --- viewer: replay the simulated motion with the goal as a green ghost on the plate ---
    server = viser.ViserServer()
    print("replaying; Ctrl+C to stop")
    server.scene.add_grid("/ground", width=1.0, height=1.0, position=(0.3, 0.0, 0.0))
    viewer = ViserMjcf(server, scene.to_xml())
    ghost = server.scene.add_icosphere("/goal", radius=0.04, color=(40, 200, 40), opacity=0.3)
    data, plate = mujoco.MjData(model), model.body("plate").id
    while True:
        for k in range(0, STEPS, 2):
            viewer.update_qpos(rollout["qpos"][k])
            data.qpos[:] = rollout["qpos"][k]
            mujoco.mj_kinematics(model, data)
            ghost.position = data.xpos[plate] + data.xmat[plate].reshape(3, 3) @ goal[k]
            time.sleep(2 * model.opt.timestep)
        time.sleep(1.0)  # hold the last frame before replaying


if __name__ == "__main__":
    tyro.cli(main)
