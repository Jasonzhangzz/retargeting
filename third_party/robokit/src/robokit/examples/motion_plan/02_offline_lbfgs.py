"""Offline L-BFGS planning with four candidates and fixed-endpoint finetuning, replayed in Viser."""

import time
from typing import Optional

import numpy as np
import trimesh
import viser
import warp as wp
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf

from robokit.assets.robots.arms import franka_panda
from robokit.geom import BoxGeom, WarpScene
from robokit.helpers.motion_plan import MotionPlanner, presets
from robokit.robo import Robot


PRESET = presets.offline_pose_lbfgs  # Use presets.offline_pose_lm for LM.


def main():
    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    urdf = load_robot_description("panda_description")

    # --- scene ---
    T_world_box = np.eye(4, dtype=np.float32)[None]
    T_world_box[0, :3, 3] = (0.45, 0.0, 0.2)
    box = BoxGeom(
        np.array([[0.04, 0.25, 0.2]], np.float32),
        np.array([0, 1], np.int32),
        poses=wp.from_numpy(T_world_box, dtype=wp.mat44, device=device),
    )
    scene = WarpScene(1, device).add(box)

    # --- planner ---
    robot = Robot.load(urdf, load_collision_spheres=True, collision_spheres_path=franka_panda.COLLISION_SPHERE_PATH)
    cur_q = robot.spec.midrange_q.astype(np.float32)
    planner = MotionPlanner(
        PRESET,
        robot,
        "panda_hand",
        scene=scene,
        device=device,
    )

    # --- viewer ---
    server = viser.ViserServer()
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")
    urdf_vis.update_cfg(cur_q)
    box_mesh = trimesh.creation.box(extents=(0.08, 0.5, 0.4)).apply_translation((0.45, 0.0, 0.2))
    server.scene.add_mesh_trimesh("/world/box", box_mesh)
    goal = server.scene.add_transform_controls("/goal", scale=0.2, position=(0.6, 0.0, 0.4), wxyz=(0.0, 1.0, 0.0, 0.0))
    replan = server.gui.add_button("Replan")
    status = server.gui.add_text("Status", "-", disabled=True)

    @replan.on_click
    def solve(_: Optional[viser.GuiEvent] = None):
        nonlocal cur_q
        # --- solve ---
        T_world_target = np.asarray((*goal.position, *goal.wxyz), dtype=np.float32)
        result = planner.solve_offline_numpy([cur_q, {"panda_hand": T_world_target}])
        metrics = planner.compute_metrics_numpy(result.q_traj, T_world_target=T_world_target)
        assert metrics.success is not None  # A pose goal supplies an endpoint success metric.
        if metrics.penetration_mm[0, -1] > 5.0:
            status.value = "goal in collision"
        else:
            status.value = "success" if metrics.success[0] else "failed"
        # --- visualize ---
        for cur_q in result.q_traj[0]:
            with server.atomic():
                urdf_vis.update_cfg(cur_q)
            time.sleep(0.05)

    solve()  # pyright: ignore[reportCallIssue]
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
