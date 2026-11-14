# /// script
# requires-python = ">=3.10"
# dependencies = ["robokit[mjwarp]"]
#
# [tool.uv.sources]
# robokit = { path = "../../../..", editable = true }
# ///
# pyright: reportMissingImports=false
"""SimMPC: a torque-driven Unitree H1 jogs, ramping its forward speed to 2 m/s over 3 s, from velocity and gait costs."""

import time
from pathlib import Path

import mujoco
import numpy as np
import viser
import warp as wp
from robot_descriptions import h1_mj_description

from robokit.helpers.sim_mpc import SimMPC, SimMPCConfig, SimMPCStage
from robokit.sim import MjWarpSim
from robokit.terms import SensorTask
from robokit.utils.visualize_utils import ViserMjcf


STEPS, DT, SPEED = 400, 0.02, 2.0  # 8 s


def main():
    wp.config.log_level = wp.LOG_WARNING  # no per-module load lines
    wp.init()

    # --- scene: Menagerie's H1 touching the floor with one capsule per foot, and PD servos (kp 200, or 60 on the ankles and arms, kd = kp / 40) whose torques clip to the motors' limits, their targets to the jog's ranges ---
    scene = mujoco.MjSpec.from_file(str(Path(h1_mj_description.MJCF_PATH).parent / "scene.xml"))
    scene.meshdir = str(Path(h1_mj_description.MJCF_PATH).parent / "assets")  # absolute, for the viewer's copy
    scene.option.timestep, scene.option.iterations, scene.option.ls_iterations = DT, 2, 5
    scene.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_EULERDAMP
    for geom in scene.geoms:
        if geom.group == 3:  # body collision shapes
            geom.contype = geom.conaffinity = 0
    for side in ("left", "right"):
        scene.body(f"{side}_ankle_link").add_geom(
            name=f"{side}_foot", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.014, 0, 0], group=3, conaffinity=0,
            fromto=[0.115, 0, -0.056, -0.035, 0, -0.056], density=0,
        )  # fmt: skip
    leg, arm = [[-0.3, 0.3], [-0.3, 0.3], [-1.0, 1.0], [0.0, 1.74], [-0.6, 0.4]], [[-0.78, 0.78]] + [[-0.3, 0.3]] * 3
    for actuator, lo_hi in zip(scene.actuators, leg * 2 + [[-0.5, 0.5]] + arm * 2):
        kp = 60.0 if any(n in actuator.name for n in ("ankle", "shoulder", "elbow")) else 200.0
        actuator.forcerange, actuator.forcelimited = actuator.ctrlrange, mujoco.mjtLimited.mjLIMITED_TRUE
        actuator.gainprm[0], actuator.biasprm[:3], actuator.ctrlrange = kp, [0.0, -kp, -kp / 40], lo_hi
        actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
    scene.body("torso_link").add_site(name="torso")
    sensor, site = mujoco.mjtSensor, mujoco.mjtObj.mjOBJ_SITE
    body, geom = mujoco.mjtObj.mjOBJ_XBODY, mujoco.mjtObj.mjOBJ_GEOM
    scene.add_sensor(name="torso_vel", type=sensor.mjSENS_VELOCIMETER, objtype=site, objname="torso")
    scene.add_sensor(name="torso_gyro", type=sensor.mjSENS_GYRO, objtype=site, objname="torso")
    scene.add_sensor(name="torso_pos", type=sensor.mjSENS_FRAMEPOS, objtype=site, objname="torso")
    scene.add_sensor(name="pelvis_zaxis", type=sensor.mjSENS_FRAMEZAXIS, objtype=body, objname="pelvis")
    scene.add_sensor(  # the world x axis seen from the torso: its heading is minus the torso's yaw
        name="heading", type=sensor.mjSENS_FRAMEXAXIS, objtype=body, objname="world", reftype=site, refname="torso"
    )
    for side in ("left", "right"):  # foot height: the capsule's distance to the floor
        scene.add_sensor(name=f"{side}_foot", type=sensor.mjSENS_GEOMDIST, objtype=geom, objname=f"{side}_foot",
                         reftype=geom, refname="floor", cutoff=10.0)  # fmt: skip
    for actuator in scene.actuators:
        scene.add_sensor(name=f"frc_{actuator.name}", type=sensor.mjSENS_ACTUATORFRC, objtype=mujoco.mjtObj.mjOBJ_ACTUATOR,
                         objname=actuator.name)  # fmt: skip
    model = scene.compile()
    state = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, state, 0)  # standing, knees bent
    mujoco.mj_forward(model, state)

    # --- targets: forward speed ramping to 2 m/s, the torso at 1.3 m, upright, heading 0, alternating 20 cm foot lifts at 2 Hz (each foot down 30% of a cycle), and small torques ---
    T = STEPS + 64  # the horizon runs past the episode
    t = np.arange(T) * DT
    velocity = np.minimum(SPEED * t / 3.0, SPEED)
    angle = (t[:, None] * 4 * np.pi + 2 * np.pi - 2 * np.pi * np.array([0.0, 0.5])) % (2 * np.pi) - np.pi
    lift = 0.2 * np.abs(np.cos(np.clip(angle * 0.5 / 0.7, -np.pi / 2, np.pi / 2)))  # (T, left/right)
    adr = {model.sensor(i).name: model.sensor_adr[i] for i in range(model.nsensor)}
    sensor_ref = np.zeros((T, model.nsensordata))
    sensor_ref[:, adr["torso_vel"]], sensor_ref[:, adr["torso_pos"] + 2] = velocity, 1.3
    sensor_ref[:, adr["pelvis_zaxis"] + 2] = sensor_ref[:, adr["heading"]] = 1.0
    sensor_ref[:, adr["left_foot"]], sensor_ref[:, adr["right_foot"]] = lift.T
    weight = {"torso_vel": [1.0, 1.0, 0.0], "torso_gyro": [0.0, 0.0, (np.pi / 180) ** 2], "torso_pos": [0.0, 0.0, 0.5],
              "pelvis_zaxis": 0.5, "left_foot": 5.0, "right_foot": 5.0}  # fmt: skip
    weight |= {f"frc_{model.actuator(i).name}": 0.01 / model.actuator_forcerange[i, 1] ** 2 for i in range(model.nu)}
    costs = [SensorTask(sensor_ref, weight), SensorTask(sensor_ref, {"heading": 0.1}, residual="heading")]

    # --- nodes: a quadratic not-a-knot spline through 6 nodes over the horizon, as a (steps + 1, nodes) basis ---
    x, xs = np.linspace(0, 25, 6), np.arange(25 + 1.0)
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

    # --- SimMPC: replan every step over a 0.5 s horizon of 6 nodes ---
    limits = model.actuator_ctrlrange
    config = SimMPCConfig(
        num_samples=2048, horizon_steps=25, node_steps=5, ctrl_steps=1, init_iterations=10,
        stages=[SimMPCStage(4, [(0.25, None), (1.0, 0.04)])],  # 1.4x faster, tracking error 0.009 vs 0.017 fine
        min_improvement=-np.inf, std_growth=1.0, noise_decay=0.5, min_std=1e-6,
        terminal_scale=1.0, ctrl_std=(limits[:, 1] - limits[:, 0]) / 2, interpolation=basis,
    )  # fmt: skip
    for cost in costs:
        config.add(cost)
    print("building SimMPC (the first run compiles kernels) ...")
    sim = MjWarpSim(model, 1, state, nconmax=120, njmax=500)  # plans on copies of it, executes on it
    mpc = SimMPC(config, sim, np.tile(limits.mean(1), (T, 1)))
    print(f"planning {STEPS * DT:.0f} s of motion ...")
    start = time.perf_counter()
    report = lambda plans: plans[0]["t0"] % 50 or print(f"  t={plans[0]['t0'] * DT:.0f} s: cost {plans[0]['plan_cost']:.3f}")  # fmt: skip
    (rollout,), _ = mpc.execute(sim, STEPS, callback=report)

    # --- check: mean squared velocity error (forward and sideways, plus the yaw rate in degrees/s scaled by pi/180), and no fall (pelvis tipped past ~73 deg or torso below 0.5 m) ---
    data = rollout["sensordata"]
    error = np.sum((data[:, adr["torso_vel"] :][:, :2] - np.c_[velocity[:STEPS], 0 * velocity[:STEPS]]) ** 2, 1)
    error = np.mean(error + (data[:, adr["torso_gyro"] + 2] * np.pi / 180) ** 2)
    fell = data[:, adr["pelvis_zaxis"] + 2].min() < 0.3 or data[:, adr["torso_pos"] + 2].min() < 0.5
    passed = error < 0.025 and not fell
    print(f"jog: {'pass' if passed else 'FAIL'}, velocity tracking error {error:.4f} (m/s)^2, "
          f"{'fell' if fell else 'no fall'}, final speed {data[-50:, adr['torso_vel']].mean():.2f} m/s, "
          f"travelled {rollout['qpos'][-1, 0]:.1f} m")  # fmt: skip
    print(f"planned in {time.perf_counter() - start:.0f} s")

    # --- viewer: replay the simulated jog ---
    server = viser.ViserServer()
    print("replaying; Ctrl+C to stop")
    server.scene.add_grid("/ground", width=12.0, height=2.0, position=(5.0, 0.0, 0.0))
    viewer = ViserMjcf(server, scene.to_xml())
    while True:
        for q in rollout["qpos"]:
            viewer.update_qpos(q)
            time.sleep(DT)
        time.sleep(1.0)  # hold the last frame before replaying


if __name__ == "__main__":
    main()
