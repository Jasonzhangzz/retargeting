"""Compare global and segmented optimization through identical waypoints in Viser."""

import dataclasses
import time
from pathlib import Path
from typing import Dict, Optional, cast

import numpy as np
import tyro
import viser
import warp as wp
import yourdfpy
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf

from robokit.helpers.motion_plan import GradientTrajectoryOptimizerConfig, MotionPlanner, MotionPlanResult, presets
from robokit.helpers.motion_plan.trajectory_retimer import TimingVar, TrajectoryRetimer
from robokit.robo import Robot


def main(
    port: int = 8092,
    duration: float = 900,
    headless: bool = False,
    output: Path = Path("runs/waypoint_planning/example"),
    replay: Optional[Path] = None,
):
    """Optimize the same waypoints globally and by segment; save and optionally replay."""
    urdf = load_robot_description("panda_description")
    results: Dict[str, MotionPlanResult[np.ndarray]] = {}
    T_world_tool = {}
    if replay is not None:
        for name in ("Global", "Segmented"):
            with np.load(replay / (name.lower().replace(" ", "_") + ".npz")) as saved:
                results[name] = MotionPlanResult(
                    q_traj=saved["q_traj"].copy(),
                    motion_time=saved["motion_time"].copy(),
                    waypoint_indices=tuple(int(i) for i in saved["waypoint_indices"]),
                )
                T_world_mid = saved["T_world_mid"].copy()
                T_world_tool[name] = saved["T_world_tool"].copy()
        if not headless:
            _show(urdf, results, T_world_mid, T_world_tool, port, duration)
        return
    from robokit.assets.robots.arms import franka_panda

    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    robot = Robot.load(
        urdf,
        load_collision_spheres=True,
        collision_spheres_path=franka_panda.COLLISION_SPHERE_PATH,
        self_collision_ignore_path=franka_panda.SPHERE_SELF_COLLISION_IGNORE_PATH,
    )
    # No scene obstacles, to focus on the ordered hard anchors; deterministic LM makes middle-anchor smoothing visible.
    lm = cast(GradientTrajectoryOptimizerConfig, presets.offline_pose_lm.stages[0]).solver.stages[0].optimizer
    stage = cast(GradientTrajectoryOptimizerConfig, presets.offline_joint_lbfgs.stages[0])
    config = dataclasses.replace(
        presets.offline_joint_lbfgs,
        cuda_graph="stages" if device != "cpu" else "none",
        init_sample_range=0,
        stages=(
            dataclasses.replace(
                stage,
                solver=dataclasses.replace(
                    stage.solver, stages=[dataclasses.replace(stage.solver.stages[0], num_candidates=1, optimizer=lm)]
                ),
            ),
        ),
    )
    planner = MotionPlanner(config, robot, "panda_hand", device=device)
    q = np.tile(robot.spec.midrange_q.astype(np.float32), (3, 1))
    q[:, :7] = [[-0.4, -0.8, 0, -2.3, 0, 1.6, 0.5], [0, -0.6, 0, -2, 0, 1.5, 0.8], [0.4, -0.8, 0, -2.3, 0, 1.6, 1.1]]
    state = robot.forward_kinematics(robot.state(q=wp.from_numpy(q.copy(), dtype=wp.float32, device=device)))
    T_world_mid = state.T_world_link.numpy()[1, robot.link_names.index("panda_hand")].copy()
    waypoints = list(q)
    results["Global"] = planner.solve_offline_numpy(waypoints)
    indices = results["Global"].waypoint_indices
    assert indices is not None
    parts = []
    for index, (left, right) in enumerate(zip(indices[:-1], indices[1:])):
        segment = MotionPlanner(
            dataclasses.replace(config, num_timesteps=right - left), robot, "panda_hand", device=device
        )
        parts.append(segment.solve_offline_numpy(waypoints[index : index + 2]).q_traj)
    results["Segmented"] = MotionPlanResult(
        q_traj=np.concatenate([part[:, :-1] for part in parts[:-1]] + [parts[-1]], axis=1),
        waypoint_indices=indices,
    )
    # Playback pauses at the endpoints; include those holds in derivative limits.
    retime = config.retime
    assert retime is not None
    limits = (config.acceleration_limit, config.jerk_limit)
    retimer = TrajectoryRetimer(
        robot, indices[-1] + 1, config.dt, *limits, dataclasses.replace(retime, virtual_hold_frames=2), device
    )
    output.mkdir(parents=True, exist_ok=True)
    for name, result in results.items():
        timing = TimingVar.full(1, config.dt, device)
        retimer.retime(wp.from_numpy(result.q_traj.copy(), dtype=wp.float32, device=device), timing)
        result.motion_time = timing.dt.numpy().copy() * indices[-1]
        assert result.motion_time is not None and result.waypoint_indices is not None
        state = robot.forward_kinematics(
            robot.state(q=wp.from_numpy(result.q_traj.copy(), dtype=wp.float32, device=device))
        )
        T_world_tool[name] = state.T_world_link.numpy()[0, :, robot.link_names.index("panda_hand")].copy()
        np.savez(
            output / (name.lower().replace(" ", "_") + ".npz"),
            q_traj=result.q_traj,
            motion_time=result.motion_time,
            waypoint_indices=result.waypoint_indices,
            T_world_mid=T_world_mid,
            T_world_tool=T_world_tool[name],
        )
        print(f"{name}: anchors={result.waypoint_indices}, motion_time={result.motion_time[0]:.3f}s", flush=True)
    if headless:
        return
    _show(urdf, results, T_world_mid, T_world_tool, port, duration)


def _show(
    urdf: yourdfpy.URDF,
    results: Dict[str, MotionPlanResult[np.ndarray]],
    T_world_mid: np.ndarray,
    T_world_tool: Dict[str, np.ndarray],
    port: int,
    duration: float,
):
    """Replay atomic robot poses at 60 Hz using each trajectory's planned timing."""
    server = viser.ViserServer(host="0.0.0.0", port=port)

    @server.on_client_connect
    def set_camera(client: viser.ClientHandle):
        client.camera.position = (1.25, -1.25, 1.0)
        client.camera.look_at = (0.25, 0.0, 0.4)

    server.scene.add_grid("/ground", width=2, height=2)
    server.scene.add_frame("/middle_target", position=T_world_mid[:3], wxyz=T_world_mid[3:], show_axes=False)
    visual = ViserUrdf(server, urdf, root_node_name="/robot")
    for name, targets in T_world_tool.items():
        points = np.stack([targets[:-1, :3], targets[1:, :3]], axis=1)
        server.scene.add_line_segments(
            "/path/" + name,
            points=points,
            colors=(255, 180, 40) if name == "Global" else (40, 180, 255),
            thickness=3,
            thickness_units="screen",
        )
    select = server.gui.add_dropdown("Optimization", options=list(results), initial_value="Global")
    playback_timing = server.gui.add_dropdown(
        "Timing", options=("Planned duration", "Equal duration"), initial_value="Planned duration"
    )
    play = server.gui.add_checkbox("Play", initial_value=True)
    frame = server.gui.add_slider(
        "Frame", min=0, max=next(iter(results.values())).q_traj.shape[1] - 1, step=1, initial_value=0
    )
    status = server.gui.add_text("Anchors", str(results[select.value].waypoint_indices), disabled=True)
    motion_time = server.gui.add_text("Motion time (s)", "", disabled=True)
    playback_time = server.gui.add_text("Playback time (s)", "", disabled=True)
    common_duration = max(float(result.motion_time[0]) for result in results.values() if result.motion_time is not None)
    deadline = time.monotonic() + duration
    previous_time = time.monotonic()
    previous_controls = (select.value, playback_timing.value, play.value)
    displayed_frame = 0
    position, direction, hold_until = 0.0, 1, 0.0
    while time.monotonic() < deadline:
        now = time.monotonic()
        elapsed = now - previous_time
        previous_time = now
        result = results[select.value]
        assert result.motion_time is not None
        controls = (select.value, playback_timing.value, play.value)
        if controls != previous_controls or int(frame.value) != displayed_frame:
            position = float(frame.value)
            elapsed = 0.0
            hold_until = now
        previous_controls = controls
        seconds = common_duration if playback_timing.value == "Equal duration" else float(result.motion_time[0])
        last = result.q_traj.shape[1] - 1
        if play.value and now >= hold_until:
            position += direction * elapsed * last / seconds
            if position >= last or position <= 0:
                position = float(np.clip(position, 0, last))
                direction = -1 if position == last else 1
                hold_until = now + 0.5  # Reverse after a pause, avoiding an end-to-start teleport.
        left = int(position)
        alpha = position - left
        q = (1 - alpha) * result.q_traj[0, left] + alpha * result.q_traj[0, min(left + 1, last)]
        displayed_frame = int(round(position))
        with server.atomic():
            visual.update_cfg(q)
            if play.value and int(frame.value) != displayed_frame:
                frame.value = displayed_frame
            status.value = str(result.waypoint_indices)
            motion_time.value = f"{result.motion_time[0]:.3f}"
            playback_time.value = f"{seconds:.3f}"
        time.sleep(max(0.0, 1 / 60 - (time.monotonic() - now)))
    server.stop()


if __name__ == "__main__":
    tyro.cli(main)
