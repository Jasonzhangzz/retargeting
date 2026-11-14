# /// script
# requires-python = ">=3.10"
# dependencies = ["robokit[mjwarp]"]
#
# [tool.uv.sources]
# robokit = { path = "../../../..", editable = true }
# ///
# pyright: reportMissingImports=false
"""SimMPC: a torque-driven Unitree Go2 makes four 40 cm jumps, landing each one's feet inside 10 cm target circles.

DIAL-MPC: https://arxiv.org/abs/2409.15610.
"""

import time
from pathlib import Path

import mujoco
import numpy as np
import viser
import warp as wp
from robot_descriptions import go2_mj_description

from robokit.helpers.sim_mpc import SimMPC, SimMPCConfig
from robokit.sim import MjWarpSim
from robokit.terms import ContactTask, SensorTask
from robokit.utils.visualize_utils import ViserMjcf


FEET, FOOT_RADIUS = ("FR", "FL", "RR", "RL"), 0.033  # holds the foot mesh, with room for the soft contact's 1 cm sink
STEPS, STAGE, DT = 400, 50, 0.02  # 8 s; a jump each second (50 steps of 20 ms)
# floor capsules (fromto, radius) in each link's frame, fitted to the meshes: the whole thigh below the hip head, and the calf's floor side (its arch always faces up), ending at the foot sphere so that the foot still lands
CAPSULES = {"thigh": ([-0.01, 0, -0.05, -0.0025, 0, -0.2], 0.03), "calf": ([0, 0, 0.01, 0.001, 0, -0.195], 0.02)}


def main():
    wp.config.log_level = wp.LOG_WARNING  # no per-module load lines
    wp.init()

    # --- scene: Menagerie's Go2 with a heavier base, light joint damping, only the legs meeting the floor, and PD servos (kp 30, no torque limit) on the joints, their targets clipped to the jump's joint ranges ---
    scene = mujoco.MjSpec.from_file(str(Path(go2_mj_description.MJCF_PATH).parent / "scene_mjx.xml"))
    scene.meshdir = str(Path(go2_mj_description.MJCF_PATH).parent / "assets")  # absolute, for the viewer's copy
    scene.option.timestep, scene.option.iterations, scene.option.impratio = DT, 2, 1.0
    scene.body("base").mass = 7.921
    for joint in scene.joints[1:]:  # the leg joints
        joint.damping[0] = 0.65
    for geom in scene.geoms:
        if geom.group == 3:  # collision shapes: contype 0 touches only the floor, conaffinity 0 not even that
            geom.contype, geom.conaffinity = 0, int(geom.name in FEET)
        if geom.name in FEET:
            geom.priority, geom.condim, geom.size[0] = 0, 3, FOOT_RADIUS
    for leg in FEET:
        for link, (fromto, radius) in CAPSULES.items():
            scene.body(f"{leg}_{link}").add_geom(
                type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=fromto, size=[radius, 0, 0], group=3, contype=0, conaffinity=1,
            )  # fmt: skip
    ranges = {"hip": [-0.5, 0.5], "thigh": [0.4, 2.0], "calf": [-2.3, -1.3]}
    for actuator in scene.actuators:
        lo_hi = [0.4, 1.4] if actuator.name in ("RR_thigh", "RL_thigh") else ranges[actuator.name[3:]]
        actuator.gainprm[0], actuator.biasprm[:3], actuator.ctrlrange = 30.0, [0.0, -30.0, 0.0], lo_hi
        actuator.forcelimited = mujoco.mjtLimited.mjLIMITED_FALSE
    for leg in ("RL", "RR"):
        scene.joint(f"{leg}_thigh_joint").range = [-0.5236, 4.5379]
    site, body = mujoco.mjtObj.mjOBJ_SITE, mujoco.mjtObj.mjOBJ_XBODY
    for foot in FEET:
        scene.add_sensor(name=foot, type=mujoco.mjtSensor.mjSENS_FRAMEPOS, objtype=site, objname=f"{foot}_foot")
    scene.add_sensor(name="torso_pos", type=mujoco.mjtSensor.mjSENS_FRAMEPOS, objtype=body, objname="base")
    scene.add_sensor(name="torso_zaxis", type=mujoco.mjtSensor.mjSENS_FRAMEZAXIS, objtype=body, objname="base")
    scene.add_sensor(  # the world x axis seen from the torso: its heading is minus the torso's yaw
        name="heading", type=mujoco.mjtSensor.mjSENS_FRAMEXAXIS, objtype=body, objname="world", reftype=body,
        refname="base",
    )  # fmt: skip
    model = scene.compile()
    state = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, state, 0)  # standing, 27 cm
    mujoco.mj_forward(model, state)

    # --- targets: the torso 40 cm further each second, and each stage's four foot circles ---
    T = STEPS + 64  # the horizon runs past the episode
    stage = np.minimum(np.arange(T) // STAGE, 4)
    torso = np.c_[0.4 * np.arange(5), np.zeros(5), np.full(5, 0.27)]
    circles = torso[:, None] + np.array([[0.2, -0.135, -0.27], [0.2, 0.135, -0.27], [-0.2, -0.135, -0.27],
                                         [-0.2, 0.135, -0.27]])  # fmt: skip  # (stage, foot, xyz) in FEET order
    adr = {model.sensor(i).name: model.sensor_adr[i] for i in range(model.nsensor)}
    sensor_ref = np.zeros((T, model.nsensordata))
    sensor_ref[:, adr["torso_pos"] : adr["torso_pos"] + 3] = torso[stage]
    sensor_ref[:, adr["torso_zaxis"] + 2] = sensor_ref[:, adr["heading"]] = 1.0  # upright, heading 0
    for i, foot in enumerate(FEET):
        sensor_ref[:, adr[foot] : adr[foot] + 3] = circles[stage, i]
    allowed = np.c_[circles[..., :2].reshape(-1, 2), np.full(20, 0.1)].reshape(5, 4, 3).transpose(1, 0, 2)
    costs = [
        SensorTask(sensor_ref, {"torso_pos": 1.0, "torso_zaxis": 1.0, **{f: [10.0, 10.0, 0.0] for f in FEET}}),
        SensorTask(sensor_ref, {"heading": 0.3}, residual="heading"),
        ContactTask(FEET, circles[stage], np.full((T, 4), 0.1), allowed[None], weight=0.1, penalty=0.1, margin=0.001),
    ]  # pull the feet over the stage's circles (at their edge as hard as a landing pays), reward feet down inside them, and penalize touches outside every circle

    # --- nodes: a quadratic not-a-knot spline through 6 nodes over the horizon, as a (steps + 1, nodes) basis ---
    x, xs = np.linspace(0, 20, 6), np.arange(20 + 1.0)
    knots = np.r_[x[0] - (x[1] - x[0]) / 2, (x[1:] + x[:-1]) / 2, x[-1] + (x[-1] - x[-2]) / 2]
    n, h, dt = len(knots), np.diff(knots), x - knots[:-1]
    A = np.diag(np.r_[1.0, 2 * dt[1:] - dt[1:] ** 2 / h[1:] - dt[:-1] ** 2 / h[:-1] + h[:-1], 1.0])
    A += np.diag(np.r_[-(1 + h[0] / h[1]), dt[1:] ** 2 / h[1:]], 1) + np.diag(np.r_[h[0] / h[1], np.zeros(n - 3)], 2)
    A += np.diag(np.r_[h[:-1] - 2 * dt[:-1] + dt[:-1] ** 2 / h[:-1], -(1 + h[-1] / h[-2])], -1)
    A += np.diag(np.r_[np.zeros(n - 3), h[-1] / h[-2]], -2)
    i = np.clip(np.digitize(xs, knots) - 1, 0, n - 2)
    t, hi, di = xs - knots[i], h[i], dt[i]
    slopes = np.linalg.solve(A, np.r_[np.zeros((1, 6)), 2 * np.diff(np.eye(6), axis=0), np.zeros((1, 6))])
    b0, b1 = slopes[i], slopes[i + 1]
    basis = np.eye(6)[i] - b0 * di[:, None] - (b1 - b0) * (di**2 / (2 * hi))[:, None] + b0 * t[:, None]
    basis += (b1 - b0) * (t**2 / (2 * hi))[:, None]

    # --- SimMPC: replan every step over a 0.4 s horizon of 6 nodes ---
    limits = model.actuator_ctrlrange
    config = SimMPCConfig(
        num_samples=1229, horizon_steps=20, node_steps=4, ctrl_steps=1, init_iterations=10,  # 0.6x DIAL's 2048
        elite_ratio=0.02,  # the DIAL benchmark's config
        min_improvement=-np.inf, std_growth=0.9**-4, noise_decay=0.5, min_std=1e-6,
        terminal_scale=1.0, ctrl_std=0.9**4 * (limits[:, 1] - limits[:, 0]) / 2, interpolation=basis,
    )  # fmt: skip
    for cost in costs:
        config.add(cost)
    print("building SimMPC (the first run compiles kernels) ...")
    sim = MjWarpSim(model, 1, state, nconmax=120, njmax=500)  # plans on copies of it, executes on it
    mpc = SimMPC(config, sim, np.tile(limits.mean(1), (T, 1)))
    print(f"planning {STEPS * DT:.0f} s of motion ...")
    start = time.perf_counter()
    report = lambda plans: plans[0]["t0"] % STAGE or print(f"  t={plans[0]['t0'] * DT:.0f} s: cost {plans[0]['plan_cost']:.3f}")  # fmt: skip
    (rollout,), _ = mpc.execute(sim, STEPS, callback=report)

    # --- check: per landing, the worst step of its contact window (0.5-0.9 s into the stage) scores 0.1 per foot down inside its circle, minus 0.1 per foot down elsewhere (the previous circle excepted); 0.4 is perfect ---
    feet = np.stack([rollout["sensordata"][:, adr[f] : adr[f] + 3] for f in FEET], 1)
    k = np.arange(len(feet))
    inside = np.sum((feet[:, None, :, :2] - circles[None, :, :, :2]) ** 2, -1) <= 0.1**2  # (step, stage, foot)
    down, now, before = feet[..., 2] <= FOOT_RADIUS + 0.001, inside[k, stage[k]], inside[k, stage[k] - 1] & (stage[k] > 0)[:, None]  # fmt: skip
    reward = 0.1 * np.sum(down & now, 1) - 0.1 * (np.sum(down & ~now, 1) - np.sum(down & before, 1))
    since = k % STAGE
    window = (since >= STAGE // 2) & ((since < STAGE * 9 // 10) | (stage[k] == 4))
    score = np.mean([reward[window & (stage[k] == j)].min() / 0.4 for j in range(1, 5)])
    upright = rollout["sensordata"][:, adr["torso_zaxis"] + 2].min()
    passed = score >= 0.99 and upright > 0.3
    print(f"jumps: {'pass' if passed else 'FAIL'}, landing score {score:.2f} of 1, torso tilt at most "
          f"{np.degrees(np.arccos(upright)):.0f} deg, travelled {rollout['qpos'][-1, 0]:.2f} m")  # fmt: skip
    print(f"planned in {time.perf_counter() - start:.0f} s")

    # --- viewer: replay the simulated jumps over the target circles ---
    server = viser.ViserServer()
    print("replaying; Ctrl+C to stop")
    server.scene.add_grid("/ground", width=3.0, height=1.0, position=(0.8, 0.0, 0.0))
    for i, (x, y, _) in enumerate(circles.reshape(-1, 3)):
        server.scene.add_icosphere(f"/circles/{i}", radius=0.02, color=(255, 120, 0), position=(x, y, 0.0))
    viewer = ViserMjcf(server, scene.to_xml())
    while True:
        for q in rollout["qpos"]:
            viewer.update_qpos(q)
            time.sleep(DT)
        time.sleep(1.0)  # hold the last frame before replaying


if __name__ == "__main__":
    main()
