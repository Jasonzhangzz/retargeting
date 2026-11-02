"""Original FA mesh replay on a valid independent MuJoCo integration grid.

The original fixed 2ms replay grid cannot represent controls spaced by 0.1ms.
This adapter calls the original loading, integration, errors and rendering
functions with a smaller explicit sim_dt. It does not change FA optimization,
initial states, controls, contact parameters or the MuJoCo integration method.
"""

import json
from pathlib import Path

import numpy as np


def replay_sim_dt(mpc_dt):
    """Choose the requested adapter grid; upstream still validates divisibility."""
    value = float(mpc_dt)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("Independent replay requires a positive finite MPC interval")
    return min(.002, value)


def replay_mesh(cfg, rollout, output):
    """Save the official replay artifacts, with only its sim_dt made explicit.

    ``output`` is the final replay artifact directory. This fallback is for the
    confirmed unsupported 2ms grid, after official optimization has completed.
    """
    from retarget.dynamics.dexforge_upstream import activate_upstream, require_environment
    require_environment(); activate_upstream()
    import mujoco
    from forceaware.replay import (
        DEFAULT_REPLAY_SIM_DT, MUJOCO_CPU, load_rollout, object_qadr,
        replay_errors, replay_mujoco, replay_substeps, stats,
    )
    from forceaware.render import sequence_camera_pose_and_fovy
    from tools.render import TrajectoryRenderRequest, render_scene_trajectory_video, scene_mesh_xml_for

    rollout = Path(rollout).expanduser().resolve()
    scene = scene_mesh_xml_for(Path(cfg.sequence.xml).expanduser().resolve())
    model = mujoco.MjModel.from_xml_path(str(scene))
    source = load_rollout(rollout, scene, model)
    sim_dt = replay_sim_dt(cfg.simulator.time_grid.mpc_dt)
    substeps = replay_substeps(source.action_dt, sim_dt)
    result = replay_mujoco(source, source.ctrl.shape[0], sim_dt)
    if not all(np.isfinite(value).all() for value in (result.qpos, result.qvel, result.ctrl)):
        raise RuntimeError("MuJoCo replay produced nonfinite states or controls")
    destination = Path(output).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    np.savez(destination / "replay.npz", qpos=result.qpos, qvel=result.qvel,
             ctrl=result.ctrl, time=result.time)
    errors = replay_errors(source, result.qpos, object_qadr(model))
    report = dict(input_rollout=str(rollout), scene_xml=str(scene), simulator=MUJOCO_CPU,
        sim_dt=sim_dt, action_dt=source.action_dt, steps=int(result.ctrl.shape[0]),
        reference_available=source.object_reference_qpos is not None,
        summary={name: stats(values) for name, values in errors.items()},
        independent_replay_time_grid_adapter=True, original_default_replay_supported=False,
        original_default_replay_sim_dt=DEFAULT_REPLAY_SIM_DT,
        physics_substeps_per_control=substeps,
        adapter_scope="only independent replay sim_dt; original replay_mujoco and rendering functions",
        same_initial_state_and_controls=True, upstream_optimization_modified=False)
    (destination / "replay.json").write_text(json.dumps(report, indent=2) + "\n")
    camera, fovy = sequence_camera_pose_and_fovy(cfg.sequence.seq_dir,
        camera_pose_path=cfg.sequence.contact_guidance_npz.parent / "camera_pose_7.npy")
    render_scene_trajectory_video(TrajectoryRenderRequest(
        scene_xml=scene, out_path=destination / "replay.mp4", fps=30,
        camera_pose=camera, camera_fovy=fovy, qpos=result.qpos,
        qpos_frames=result.time * 30, height=480, width=640))
    print(f"[replay_saved: explicit independent time grid] {destination}", flush=True)
    return destination


def main(argv=None):
    """Replay an already saved bundle without repeating FA optimization."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Prepared ForceAware reference bundle")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--rollout", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    from retarget.dynamics.dexforge_upstream import activate_upstream, require_environment
    require_environment(); activate_upstream()
    from retarget.env.grasp.forceaware_reference import register_wuji2_hand
    from forceaware.config import load_retarget_config
    register_wuji2_hand()
    cfg = load_retarget_config(args.root / "sequence", args.config or args.root / "forceaware_model.yaml")
    native = cfg.sequence.seq_dir / cfg.sequence.output_dir_name
    return replay_mesh(cfg, args.rollout or native / "rollout.npz", args.output or native / "replay/mujoco_cpu")


if __name__ == "__main__":
    main()
