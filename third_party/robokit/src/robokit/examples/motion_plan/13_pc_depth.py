# pyright: reportArgumentType=false
"""Online L-BFGS follows a draggable goal using four-view PC or single-view depth."""

import time
from threading import Event
from typing import List, Tuple

import numpy as np
import trimesh
import tyro
import viser
import warp as wp
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf

from robokit.assets.robots.arms import franka_panda
from robokit.geom import BoxGeom, SdfVolume, VolumeGeom, WarpScene
from robokit.helpers.motion_plan import MotionPlanEvaluator, MotionPlanner, presets
from robokit.robo import Robot
from robokit.utils.warp_utils import wp_vec7


# --- synthetic observations and reconstruction display ----------------------
@wp.kernel
def _render_depth_kernel(
    mesh: wp.uint64, T_world_camera: wp.mat44, depth: wp.array2d(dtype=float), points: wp.array(dtype=wp.vec3)
):
    v, u = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    ray = wp.normalize(wp.vec3((float(u) - 63.5) / 100.0, (float(v) - 63.5) / 100.0, 1.0))  # pyright: ignore[reportCallIssue]
    origin = wp.transform_point(T_world_camera, wp.vec3(0.0))  # pyright: ignore[reportCallIssue]
    direction = wp.transform_vector(T_world_camera, ray)  # pyright: ignore[reportCallIssue]
    hit = wp.mesh_query_ray(mesh, origin, direction, 3.0)
    depth[v, u] = wp.where(hit.result, hit.t * ray[2], 0.0)
    points[v * 128 + u] = wp.where(hit.result, origin + hit.t * direction, wp.vec3(wp.nan))


@wp.kernel
def _extract_occupied_kernel(
    volume: wp.uint64,
    observations: wp.array3d(dtype=wp.vec2),
    points: wp.array(dtype=wp.vec3),
    count: wp.array(dtype=int),
):
    i, j, k = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    if observations[i, j, k][0] < 0.0:
        points[wp.atomic_add(count, 0, 1)] = wp.volume_index_to_world(volume, wp.vec3(float(i), float(j), float(k)))  # pyright: ignore[reportCallIssue]


# --- data preparation --------------------------------------------------------
def _build_observations() -> Tuple[WarpScene, wp.mat33, List[Tuple[wp.array, wp.array, wp.mat44]]]:
    """Render four views: PC uses all four; depth uses the first for partial observation.

    Returns:
        Ground-truth scene, camera intrinsics, and `(depth, world XYZ, T_world_camera)` views; depth and XYZ are GPU arrays in meters; camera axes are right, down, forward.
    """
    box = trimesh.creation.box((0.08, 0.5, 0.4)).apply_translation((0.45, 0, 0.2))
    mesh = wp.Mesh(wp.array(box.vertices, dtype=wp.vec3), wp.array(box.faces.ravel(), dtype=int))
    T_world_box = np.eye(4, dtype=np.float32)[None]
    T_world_box[0, :3, 3] = (0.45, 0, 0.2)
    truth = WarpScene(1, "cuda:0").add(
        BoxGeom(np.array([[0.04, 0.25, 0.2]]), np.array([0, 1]), poses=wp.array(T_world_box, dtype=wp.mat44))
    )
    intrinsics = wp.mat33(100, 0, 63.5, 0, 100, 63.5, 0, 0, 1)
    observations = []
    for x, y in ((1, 0), (0, 1), (-1, 0), (0, -1)):
        t_world_camera = wp.vec3(0.45 + x, y, 0.8)
        forward = wp.vec3(wp.normalize(wp.vec3(0.45, 0, 0.2) - t_world_camera))  # pyright: ignore[reportCallIssue, reportOperatorIssue]
        right = wp.vec3(wp.normalize(wp.cross(forward, wp.vec3(0, 0, 1))))  # pyright: ignore[reportCallIssue]
        down = wp.vec3(wp.cross(forward, right))  # pyright: ignore[reportCallIssue]
        T_world_camera = wp.matrix_from_cols(
            wp.vec4(*right, 0), wp.vec4(*down, 0), wp.vec4(*forward, 0), wp.vec4(*t_world_camera, 1)  # pyright: ignore[reportCallIssue]
        )  # fmt: skip
        depth, points = wp.zeros((128, 128)), wp.empty(128 * 128, dtype=wp.vec3)
        wp.launch(_render_depth_kernel, depth.shape, [mesh.id, T_world_camera, depth, points])
        observations.append((depth, points, T_world_camera))
    return truth, intrinsics, observations


def main(port: int = 8080):
    wp.init()
    wp.set_device("cuda:0")
    truth, intrinsics, observations = _build_observations()
    urdf = load_robot_description("panda_description")
    robot = Robot.load(urdf, load_collision_spheres=True, collision_spheres_path=franka_panda.COLLISION_SPHERE_PATH)
    cur_q = robot.spec.midrange_q.astype(np.float32)

    # --- one persistent collision volume and the existing planner ---
    volume = SdfVolume.from_points(wp.empty(0, dtype=wp.vec3), np.array([-1, -1, -0.2]), np.array([1, 1, 1.5]))
    scene = WarpScene(1, "cuda:0").add(VolumeGeom([volume], np.array([0, 1])))
    planner = MotionPlanner(presets.online_pose_lbfgs, robot, "panda_hand", scene=scene, device="cuda:0")
    evaluator = MotionPlanEvaluator(robot, "panda_hand", 1, scene=truth, device="cuda:0")
    fusion = volume._fusion  # only inspected to display occupied voxels
    assert fusion is not None
    occupied, count = wp.empty(int(np.prod(fusion.shape)), dtype=wp.vec3), wp.zeros(1, dtype=int)

    # --- viewer ---
    server = viser.ViserServer(port=port, label="PC / depth planning")
    server.scene.set_up_direction("+z")
    server.gui.add_markdown(
        "PC: 4 views. Depth: 1 partial view. Orange: input XYZ. Blue: occupied voxels. Drag the goal to replan."
    )
    selection = server.gui.add_dropdown("Input", ("Point cloud", "Depth"))
    running = server.gui.add_checkbox("Run", True)
    status = server.gui.add_markdown("Building map")
    image = observations[0][0].numpy()
    server.gui.add_image(
        (image.clip(0, 2) * 127.5).astype(np.uint8)[..., None].repeat(3, axis=2), label="Depth view 1 (0–2 m)"
    )
    input_cloud = server.scene.add_point_cloud("/input", np.empty((0, 3)), colors=(240, 160, 60), point_size=0.005)
    reconstructed = server.scene.add_point_cloud("/map", np.empty((0, 3)), colors=(80, 160, 240), point_size=0.02)
    robot_view = ViserUrdf(server, urdf, root_node_name="/robot")
    goal = server.scene.add_transform_controls("/goal", scale=0.15, position=(0.6, 0, 0.4), wxyz=(0, 1, 0, 0))

    @server.on_client_connect
    def set_camera(client: viser.ClientHandle):
        client.camera.position = (1.5, -1.5, 1.2)
        client.camera.look_at = (0.4, 0, 0.3)

    rebuild = Event()
    selection.on_update(lambda _: rebuild.set())
    rebuild.set()
    online_state = None
    while True:
        started = time.perf_counter()
        if rebuild.is_set():
            rebuild.clear()
            input_type = selection.value
            views = observations if input_type == "Point cloud" else observations[:1]
            volume.update(reset=True)
            for depth, points, T_world_camera in views:
                if input_type == "Point cloud":
                    volume.update(points=points)
                else:
                    volume.update(depth=depth, intrinsics=intrinsics, T_world_camera=T_world_camera)
            points_np = np.concatenate([points.numpy() for _, points, _ in views])
            input_cloud.points = points_np[np.isfinite(points_np).all(axis=1)]
            count.zero_()
            wp.launch(_extract_occupied_kernel, fusion.shape, [volume.volume.id, fusion, occupied, count])
            reconstructed.points = occupied.numpy()[: int(count.numpy()[0])]
            online_state = None
            status.content = "Map ready; drag the goal"
            print(f"{input_type}: map ready ({len(views)} views)", flush=True)
        if running.value and server.get_clients():
            target = np.asarray((*goal.position, *goal.wxyz), dtype=np.float32)
            # --- solve: online replanning ---
            result = planner.solve_online_numpy(cur_q, T_world_target=target, online_state=online_state)
            online_state = result.online_state
            cur_q = result.q_traj[0, 1].copy()
            metrics = evaluator.compute_metrics(
                robot.state(q=wp.array(cur_q[None, None])), T_world_target=wp.array(target[None], dtype=wp_vec7)
            )
            penetration = float(metrics.penetration_mm.numpy().max())
            assert metrics.success is not None and metrics.position_error_m is not None
            reached = bool(metrics.success.numpy()[0]) and penetration < 0.1
            error = float(metrics.position_error_m.numpy()[0]) * 1000
            status.content = (
                f"{'Reached' if reached else 'Tracking'} · goal {error:.1f} mm · box overlap {penetration:.1f} mm"
            )
        robot_view.update_cfg(cur_q)
        time.sleep(max(0.0, planner.config.dt - (time.perf_counter() - started)))


if __name__ == "__main__":
    tyro.cli(main)
