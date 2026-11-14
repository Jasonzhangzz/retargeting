# /// script
# requires-python = ">=3.10"
# dependencies = ["robokit[mjwarp]"]
#
# [tool.uv.sources]
# robokit = { path = "../../../..", editable = true }
# ///
# pyright: reportMissingImports=false
"""SimMPC: a Unitree G1 backflips in physics with iLQR, tracking the mocap flip of mjbatch's `examples/g1_flip.py`."""

import time
import urllib.request
from pathlib import Path

import mujoco
import numpy as np
import viser
import warp as wp
from robot_descriptions import g1_mj_description
from scipy.optimize import nnls

from robokit.assets import ROBOKIT_HOME
from robokit.helpers.sim_mpc import SimMPC, SimMPCConfig
from robokit.opt.ilqr_optimizer import ILQROptimizerConfig
from robokit.sim import MjWarpSim
from robokit.terms import ControlTask, SensorTask
from robokit.utils.visualize_utils import ViserMjcf


REVISION = "4cba7799b0e5265cdb96f6d520a491ace4dad9ae"  # kevinzakka/mjbatch: the G1 on torque motors, and the clip
FRAMES, SUB, LIFT = 200, 2, 0.001  # clip frames tracked (50 Hz; the ankles roll after 200); sim steps a frame; feet up
BODIES = (  # bodies the cost tracks, mjbatch's
    "pelvis", "torso_link", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link", "right_hip_roll_link",
    "right_knee_link", "right_ankle_roll_link", "left_shoulder_roll_link", "left_elbow_link",
    "right_shoulder_roll_link", "right_elbow_link",
)  # fmt: skip
# weights per m^2, rad^2, (m/s)^2, (rad/s)^2 and (N m)^2: mjbatch's, but body positions and orientations twice theirs; the pelvis's position loose in x and y, tight in z
W_POS, W_ROOT, W_ROT, W_VEL, W_ANG, W_CTRL = 200.0, [1.0, 1.0, 1000.0], 200.0, 5.0, 1.0, 1e-2
PASS = 0.03  # mean body position error (m) relative to the pelvis; mjbatch's iLQR reaches 0.020


def main():
    wp.config.log_level = wp.LOG_WARNING  # no per-module load lines
    wp.init()

    # --- data: mjbatch's G1 on menagerie's meshes, with frame sensors on the tracked bodies, and its retargeted flip ---
    root = ROBOKIT_HOME / "mjbatch" / REVISION
    root.mkdir(parents=True, exist_ok=True)
    for name in ("g1.xml", "theme.xml", "flip.npz"):
        if not (root / name).exists():
            url = f"https://raw.githubusercontent.com/kevinzakka/mjbatch/{REVISION}/examples/assets/{name}"
            urllib.request.urlretrieve(url, root / name)
    scene = mujoco.MjSpec.from_file(str(root / "g1.xml"))
    scene.meshdir = str(Path(g1_mj_description.PACKAGE_PATH) / "assets")
    for body in BODIES:  # position, orientation and velocities, in the world
        for kind in ("POS", "QUAT", "LINVEL", "ANGVEL"):
            kind_type, xbody = getattr(mujoco.mjtSensor, f"mjSENS_FRAME{kind}"), mujoco.mjtObj.mjOBJ_XBODY
            scene.add_sensor(name=f"{body}_{kind}", type=kind_type, objtype=xbody, objname=body)
    model = scene.compile()
    clip = {k: v.astype(float) for k, v in np.load(root / "flip.npz").items()}
    print(f"scene: {model.nq} qpos, {model.nu} torque motors; clip: {FRAMES} of {len(clip['joint_pos'])} frames")

    # --- reference: the clip's states (free joint spin in the pelvis frame) halfway between frames, as a knot's costs read the state one sim step in; their sensor readings, and torques by inverse dynamics as mjbatch's warm start, with the contact forces that best explain the pelvis's rows (nonnegative least squares), since MuJoCo's come from the clip's foot penetration, far over the limits ---
    keep, data = model.jnt_bodyid[1:] - 2, mujoco.MjData(model)  # the clip's columns of the model's hinges
    qpos = np.c_[clip["body_pos_w"][:, 0] + [0, 0, LIFT], clip["body_quat_w"][:, 0], clip["joint_pos"][:, keep]]
    qvel = np.c_[clip["body_lin_vel_w"][:, 0], clip["body_ang_vel_w"][:, 0], clip["joint_vel"][:, keep]]
    for q, v in zip(qpos, qvel):
        mujoco.mju_normalize4(q[3:7])
        mujoco.mju_rotVecQuat(v[3:6], v[3:6].copy(), q[3:7] * [1, -1, -1, -1])
    sensor_ref, torque, limit = [], [], model.actuator_ctrlrange[:, 1]
    for t in range(FRAMES + 1):
        data.qpos[:], data.qvel[:] = (qpos[t] + qpos[t + 1]) / 2, (qvel[t] + qvel[t + 1]) / 2
        data.qacc[:] = (qvel[t + 1] - qvel[t]) * clip["fps"][0]
        mujoco.mju_normalize4(data.qpos[3:7])
        mujoco.mj_inverse(model, data)
        contact = data.efc_type[: data.nefc] == mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL
        J = data.efc_J.reshape(-1, model.nv)[: data.nefc][contact]
        force = data.qfrc_inverse + data.qfrc_constraint  # without the contacts
        A = np.r_[J[:, :6].T, J[:, 6:].T / limit[:, None], 1e-3 * np.eye(len(J))]
        f = nnls(A, np.r_[force[:6], force[6:] / limit, np.zeros(len(J))])[0] if len(J) else []
        sensor_ref.append(data.sensordata.copy())
        torque.append(np.clip(force[6:] - J[:, 6:].T @ f, -limit, limit))
    sensor_ref = np.repeat(sensor_ref, SUB, axis=0)
    data.qpos[:], data.qvel[:] = qpos[0], qvel[0]
    mujoco.mj_forward(model, data)

    # --- SimMPC: iLQR on torques held a frame, 50 frames ahead, planned every 10 frames, 20 iterations (no CUDA graph, whose capture alone takes 50 s); mjbatch's costs (quadratic where it is robust): the bodies' positions relative to the pelvis, the pelvis's own, their orientations and velocities, and the torques ---
    weight = {"pelvis_POS": W_ROOT} | {f"{b}_{k}": w for b in BODIES for k, w in (("LINVEL", W_VEL), ("ANGVEL", W_ANG))}
    config = SimMPCConfig(
        horizon_steps=50 * SUB, node_steps=SUB, ctrl_steps=10 * SUB, max_iterations=20, terminal_scale=1.0,
        planner="ilqr", ilqr=ILQROptimizerConfig(
            improve=True, feedback_search=False, feedback_cap=0.1, min_regularization=1.0, cuda_graph_mode="none"
        ),
    )  # fmt: skip
    config.add(SensorTask(sensor_ref, {f"{b}_POS": W_POS for b in BODIES[1:]}, relative="pelvis_POS"))
    config.add(SensorTask(sensor_ref, weight))  # pyright: ignore[reportArgumentType]
    config.add(SensorTask(sensor_ref, {f"{b}_QUAT": W_ROT for b in BODIES}, residual="quaternion"))
    config.add(ControlTask(W_CTRL))
    print("building SimMPC (the first run compiles kernels) ...")
    sim = MjWarpSim(model, 1, data, nconmax=64, njmax=256)  # plans on copies of it, executes on it
    mpc = SimMPC(config, sim, np.repeat(torque, SUB, axis=0))
    print(f"planning {FRAMES / 50:.0f} s of motion ...")
    start = time.perf_counter()
    report = lambda p: p[0]["t0"] % 50 or print(f"  t={p[0]['t0'] / 100:.1f} s: cost {p[0]['plan_cost']:.2f}")
    (rollout,), _ = mpc.execute(sim, FRAMES * SUB, callback=report)
    planned = time.perf_counter() - start

    # --- check: each frame's tracked bodies relative to the pelvis, against the clip's; the torso upright at the end ---
    ids, xpos = [model.body(b).id for b in BODIES], []
    for q in rollout["qpos"][::SUB]:
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        xpos.append(data.xpos[ids].copy())
    xpos, target = np.array(xpos), clip["body_pos_w"][:FRAMES, np.array(ids) - 1] + [0, 0, LIFT]
    error = np.linalg.norm((xpos - xpos[:, :1]) - (target - target[:, :1]), axis=2).mean()
    upright = data.xmat[model.body("torso_link").id][8]  # z of the torso's up axis at the last frame
    print(
        f"flip: {'pass' if error <= PASS and upright >= 0.9 else 'FAIL'}, body error {100 * error:.1f} cm "
        f"(pass <= {100 * PASS:.0f}), torso up {upright:.2f} at the end (pass >= 0.9)"
    )
    print(f"planned in {planned:.0f} s")

    # --- viewer: replay the simulated motion, or the clip it tracks ---
    server = viser.ViserServer()
    print("replaying; Ctrl+C to stop")
    server.scene.add_grid("/ground", width=4.0, height=4.0)
    reference = server.gui.add_checkbox("Show the clip", False)
    viewer = ViserMjcf(server, scene.to_xml())
    while True:
        for k in range(0, FRAMES * SUB, SUB):
            viewer.update_qpos(qpos[k // SUB] if reference.value else rollout["qpos"][k])
            time.sleep(SUB * model.opt.timestep)
        time.sleep(1.0)  # hold the last frame before replaying


if __name__ == "__main__":
    main()
