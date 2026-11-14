# /// script
# requires-python = ">=3.10"
# dependencies = ["robokit[mjwarp]"]
#
# [tool.uv.sources]
# robokit = { path = "../../../..", editable = true }
# ///
# pyright: reportMissingImports=false
"""SimMPC: a Unitree G1 sprints in physics, tracking an AMASS clip retargeted kinematically to the G1."""

import time

import mujoco
import numpy as np
import viser
import warp as wp

from robokit.assets import fetch
from robokit.helpers.sim_mpc import SimMPC, SimMPCConfig, SimMPCStage
from robokit.sim import MjWarpSim
from robokit.terms import StateTask
from robokit.utils.visualize_utils import ViserMjcf


REPO, REVISION = "retarget/retarget_example", "8b0cdf2b24cc86d1c859050c5b0317ddbec59a63"  # G1 scene and clips
CLIP = "processed/amass/unitree_g1/humanoid/sprint"  # 64 frames, 33.5 ms apart: pelvis pose and 29 joints


def main():
    wp.config.log_level = wp.LOG_WARNING  # no per-module load lines
    wp.init()

    # --- data: the G1's kinematic sprint, and its scene, where only the feet touch the floor ---
    root = fetch(
        [f"{CLIP}/0/trajectory_kinematic.npz", f"{CLIP}/scene.xml", "processed/amass/assets/robots/unitree_g1/**"],
        REVISION, REPO,
    )  # fmt: skip
    clip = np.load(root / CLIP / "0/trajectory_kinematic.npz")
    scene = mujoco.MjSpec.from_file(str(root / CLIP / "scene.xml"))
    scene.option.timestep, scene.option.iterations, scene.option.ls_iterations = 1 / 60, 5, 10
    scene.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model = scene.compile()
    print(f"loaded {len(clip['qpos'])} frames of {CLIP}; scene: {model.nq} qpos, {model.nu} actuators")

    # --- reference: the clip upsampled 2x to the 60 Hz sim step; the start is its first frame, at the clip's velocity ---
    n = len(clip["qpos"])
    qpos_ref = np.stack([np.interp(np.linspace(0, n - 1, 2 * n), np.arange(n), c) for c in clip["qpos"].T], 1)
    state = mujoco.MjData(model)
    state.qpos[:], state.qvel[:], state.ctrl[:] = qpos_ref[0], clip["qvel"][0], qpos_ref[0, 7:]
    mujoco.mj_forward(model, state)

    # --- SimMPC: track the pelvis (position weighted 3x) and joints, around the reference joints as commands ---
    cost = StateTask(qpos_ref, np.zeros((len(qpos_ref), model.nv)), np.r_[3.0, 3.0, 3.0, np.ones(model.nv - 3)])
    config = SimMPCConfig(
        num_samples=2048, horizon_steps=48, node_steps=6, ctrl_steps=6, stages=[SimMPCStage(16)], min_improvement=0.01,
        patience=1, std_growth=2.0, noise_decay=0.1 ** (1 / 16), terminal_scale=1.0, ctrl_std=0.05,
    ).add(cost)  # fmt: skip
    steps = -(-len(qpos_ref) // config.ctrl_steps) * config.ctrl_steps
    print("building SimMPC (the first run compiles kernels) ...")
    sim = MjWarpSim(model, 1, state, nconmax=120, njmax=500)
    mpc = SimMPC(config, sim, qpos_ref[:, 7:])
    print(f"planning {steps / 60:.1f} s of motion ...")
    start = time.perf_counter()
    report = lambda p: p[0]["t0"] % 30 or print(f"  t={p[0]['t0'] / 60:.1f} s: cost {p[0]['plan_cost']:.3f}")
    (rollout,), _ = mpc.execute(sim, steps, callback=report)

    # --- check: pelvis and joint errors at the clip's frames (each frame two sim steps); no fall ---
    sim = rollout["qpos"][1::2][:n]
    ref = clip["qpos"][: len(sim)]
    position = np.linalg.norm(sim[:, :3] - ref[:, :3], axis=1).mean()
    rotation = 2 * np.arccos(np.clip(np.abs(np.sum(sim[:, 3:7] * ref[:, 3:7], 1)), 0, 1)).mean()
    joints = np.linalg.norm(sim[:, 7:] - ref[:, 7:], axis=1).mean()
    lowest = rollout["qpos"][:, 2].min()
    ok = position <= 0.1 and rotation <= 0.5 and lowest >= 0.4
    print(
        f"G1: pelvis error {100 * position:.1f} cm, {rotation:.2f} rad mean; joint error {joints:.2f} rad mean; "
        f"lowest pelvis {lowest:.2f} m: {'pass' if ok else 'FAIL'}"
    )
    print(f"planned in {time.perf_counter() - start:.0f} s")

    # --- viewer: replay the simulated motion, or the kinematic reference it tracks ---
    server = viser.ViserServer()
    print("replaying; Ctrl+C to stop")
    server.scene.add_grid("/ground", width=6.0, height=6.0, position=(*qpos_ref[n, :2], 0.0))
    reference = server.gui.add_checkbox("Show kinematic reference", False)
    viewer = ViserMjcf(server, str(root / CLIP / "scene.xml"))
    while True:
        for k in range(steps):
            viewer.update_qpos(qpos_ref[min(k, len(qpos_ref) - 1)] if reference.value else rollout["qpos"][k])
            time.sleep(model.opt.timestep)
        time.sleep(1.0)  # hold the last frame before replaying


if __name__ == "__main__":
    main()
