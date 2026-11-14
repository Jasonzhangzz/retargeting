# /// script
# requires-python = ">=3.10"
# dependencies = ["robokit[mjwarp]"]
#
# [tool.uv.sources]
# robokit = { path = "../../../..", editable = true }
# ///
# pyright: reportMissingImports=false
"""SimMPC: repair a noisy robot demonstration, a Panda fitting a square nut onto its peg (MimicGen Square D0). The demonstration's recorded arm joints get a smooth 0.05 rad per-joint offset, standing in for joints fitted from video; replayed open-loop they miss the nut, so SimMPC samples offsets around them to track the demo's nut motion."""

import re
import time
import zlib

import mujoco
import numpy as np
import viser
import warp as wp

from robokit.assets import fetch
from robokit.helpers.sim_mpc import SimMPC, SimMPCConfig, SimMPCStage
from robokit.sim import MjWarpSim
from robokit.terms import ControlTask, GraspTask, StateTask
from robokit.utils.visualize_utils import ViserMjcf


# demo 0 of MimicGen's robot/square_d0_panda.hdf5 (its MuJoCo XML, 20 Hz sim states and actions) and robosuite's meshes
DEMO = "datasets/mimicgen/square_d0_panda_demo0"
DT, N = 0.002, 25  # sim step; sim steps per 20 Hz demo step


def main():
    wp.config.log_level = wp.LOG_WARNING  # no per-module load lines
    wp.init()

    # --- data: the demo's sim state at every 20 Hz step and its gripper command ---
    root = fetch([f"{DEMO}/**"]) / DEMO
    demo = np.load(root / "square_d0_panda_demo0.npz")
    states, actions = demo["states"], demo["actions"]
    print(f"loaded {len(states)} steps of a Square D0 Panda demo")

    # --- scene: the demo's own XML; no arm self-collisions, no air (NaN forces on MuJoCo Warp), no contacts for the round nut parked under the table, the arm's torque motors as joint position servos ---
    xml = (root / "square_d0_panda_demo0.xml").read_text().replace('file="', f'file="{root}/')
    links = re.findall(r'<body name="(robot0_[^"]*)"', xml)
    pairs = "".join(f'<exclude body1="{a}" body2="{b}"/>' for i, a in enumerate(links) for b in links[i + 1 :])
    xml = xml.replace("</mujoco>", f"<contact>{pairs}</contact></mujoco>")
    scene = mujoco.MjSpec.from_string(xml)
    scene.option.timestep, scene.option.density, scene.option.viscosity = DT, 0.0, 0.0
    scene.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    for geom in scene.geoms:
        if geom.parent.name.startswith("RoundNut"):
            geom.contype = geom.conaffinity = 0
    arm = np.array([a.biastype == mujoco.mjtBias.mjBIAS_NONE for a in scene.actuators])  # the gripper's are servos
    for actuator in [a for a, is_arm in zip(scene.actuators, arm) if is_arm]:
        actuator.gainprm[0], actuator.biasprm[:3] = 2000.0, [0.0, -2000.0, -200.0]
        actuator.biastype, actuator.ctrllimited = mujoco.mjtBias.mjBIAS_AFFINE, mujoco.mjtLimited.mjLIMITED_FALSE
    model = scene.compile()
    joints = model.actuator_trnid[:, 0]
    nut = model.body("SquareNut_main").id
    adr, dof = model.jnt_qposadr[model.body_jntadr[nut]], model.jnt_dofadr[model.body_jntadr[nut]]
    qpos_demo, qvel_demo = states[:, 1 : 1 + model.nq], states[:, 1 + model.nq :]

    # --- the observed demonstration: recorded joints plus noise as arm targets, the recorded gripper command ---
    q = qpos_demo[:, model.jnt_qposadr[joints]]
    rng = np.random.default_rng(zlib.crc32(b"square_d0_panda_demo0_noise2"))  # the benchmark's draw for this demo
    knots = rng.normal(0.0, 0.05, (int((len(q) - 1) * N * DT / 0.5) + 2, arm.sum()))  # a knot every 0.5 s
    t = np.arange(len(q)) * N * DT / 0.5
    q[:, arm] += np.stack([np.interp(t, np.arange(len(knots)), k) for k in knots.T], 1)
    lo, hi = model.actuator_ctrlrange.T
    closed = np.where(np.abs(hi - q[0]) > np.abs(lo - q[0]), hi, lo)
    command = np.where(actions[:, -1:] > 0, closed, np.where(closed == hi, lo, hi))
    ctrl_ref = np.repeat(np.where(arm, q[1:], command[:-1]), N, axis=0)  # each target held over its demo step
    upsample = lambda x: np.stack(
        [np.interp(np.arange((len(x) - 1) * N + 1) / N, np.arange(len(x)), c) for c in x.T], 1
    )
    qpos_ref, qvel_ref = upsample(qpos_demo), upsample(qvel_demo)
    qpos_ref[:, adr + 3 : adr + 7] /= np.linalg.norm(qpos_ref[:, adr + 3 : adr + 7], axis=1, keepdims=True)
    observed = qpos_ref.copy()
    observed[:, model.jnt_qposadr[joints[arm]]] = upsample(q[:, arm])  # what the joint pull follows
    state = mujoco.MjData(model)
    state.qpos[:], state.qvel[:], state.ctrl[:] = qpos_demo[0], qvel_demo[0], ctrl_ref[0]
    mujoco.mj_forward(model, state)

    # --- open-loop: the observed targets, stepped in MuJoCo ---
    data, open_loop = mujoco.MjData(model), []
    data.qpos[:], data.qvel[:], data.ctrl[:] = qpos_demo[0], qvel_demo[0], ctrl_ref[0]
    for c in ctrl_ref:
        data.ctrl[:] = c
        mujoco.mj_step(model, data)
        open_loop.append(data.qpos.copy())
    open_loop = np.array(open_loop)

    # --- SimMPC: track the nut from the first close, the gripper's offset from the nut around the grasp, and pull weakly toward the observed joints ---
    config = SimMPCConfig(
        num_samples=512, horizon_steps=500, node_steps=125, ctrl_steps=125, min_improvement=0.005,
        stages=[SimMPCStage(8, [(0.25, None), (1.0, 0.004)])],
        patience=1, ctrl_std=arm * 0.015,
    )  # fmt: skip
    pad = config.horizon_steps + config.ctrl_steps
    extend = lambda x: np.vstack([x, np.repeat(x[-1:], pad, 0)])
    shut = np.flatnonzero(actions[:, -1] > 0)
    close, release = shut[0] * N, np.flatnonzero((np.arange(len(actions)) > shut[0]) & (actions[:, -1] <= 0))[0] * N
    weight = np.zeros((len(qpos_ref) + pad, model.nv))
    weight[:, model.jnt_dofadr[joints[arm]]] = 0.03
    weight[close:, dof : dof + 3], weight[close:, dof + 3 : dof + 6] = 1.0, 0.1  # per m and per rad
    gripper = model.site_bodyid[model.site("gripper0_grip_site").id]
    offset = np.zeros((len(qpos_demo), 3))  # the demo's gripper relative to the nut, in world coordinates
    for i, row in enumerate(qpos_demo):
        data.qpos[:] = row
        mujoco.mj_kinematics(model, data)
        offset[i] = data.xpos[gripper] - data.xpos[nut]
    steps = np.arange(len(qpos_ref) + pad)
    on, carrying = (steps >= close - 0.5 / DT) & (steps <= release), (steps >= close) & (steps <= release)
    costs = [
        ControlTask(0.03, extend(ctrl_ref), norm=True),
        StateTask(extend(observed), extend(qvel_ref), weight, pose_tolerance=0.005),
        GraspTask(
            gripper, nut, extend(upsample(offset)), on * 1.0, tolerance=0.005 * (1 + carrying), object_frame=False
        ),
    ]
    for cost in costs:
        config.add(cost)
    print("building SimMPC (the first run compiles kernels) ...")
    mpc = SimMPC(config, MjWarpSim(model, 1, state, nconmax=400, njmax=2000), extend(ctrl_ref))
    print(f"planning {len(ctrl_ref) * DT:.1f} s of motion ...")
    begin, repaired = time.perf_counter(), []
    for t0 in range(0, len(ctrl_ref), config.ctrl_steps):  # plan on MuJoCo Warp, step in MuJoCo as the open-loop does
        commands, (plan,) = mpc.plan(state, t0)
        print(f"  t={t0 * DT:.2f} s: cost {plan['plan_cost']:.3f}")
        for ctrl in commands.numpy()[:, 0]:
            state.ctrl[:] = ctrl
            mujoco.mj_step(model, state)
            repaired.append(state.qpos.copy())
    repaired = np.array(repaired[: len(ctrl_ref)])
    for name, qpos in [("open-loop", open_loop), ("SimMPC", repaired)]:
        error = np.linalg.norm(qpos[-1, adr : adr + 3] - qpos_ref[-1, adr : adr + 3])
        print(f"{name}: nut ends {100 * error:.1f} cm from the demo's, {'pass' if error < 0.02 else 'fail'} (< 2 cm)")
    print(f"planned in {time.perf_counter() - begin:.0f} s")

    # --- viewer: replay the repaired motion, the open-loop replay or the recorded demo ---
    server = viser.ViserServer()
    print("replaying; Ctrl+C to stop")
    server.scene.add_grid("/ground", width=2.0, height=2.0)
    shown = server.gui.add_dropdown("Replay", ("SimMPC", "open-loop", "recorded demo"))
    viewer = ViserMjcf(server, xml)
    replays = {"SimMPC": repaired, "open-loop": open_loop, "recorded demo": qpos_ref[1:]}
    while True:
        for k in range(0, len(ctrl_ref), 10):
            viewer.update_qpos(replays[shown.value][k])
            time.sleep(10 * DT)
        time.sleep(1.0)  # hold the last frame before replaying


if __name__ == "__main__":
    main()
