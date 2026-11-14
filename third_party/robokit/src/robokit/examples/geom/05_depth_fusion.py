# pyright: reportArgumentType=false
"""Watch noisy depth images build a TSDF surface; pause or restart fusion."""

import time
from threading import Event

import numpy as np
import trimesh
import tyro
import viser
import warp as wp

from robokit.geom import SdfVolume


# --- depth observations -----------------------------------------------------
@wp.kernel
def _render_depth_kernel(
    mesh: wp.uint64,
    T_world_camera: wp.mat44,
    intrinsics: wp.mat33,
    frame: int,
    depth: wp.array2d(dtype=wp.float32),
    points: wp.array(dtype=wp.vec3),
):
    v, u = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    ray = wp.normalize(  # pyright: ignore[reportCallIssue]
        wp.vec3((float(u) - intrinsics[0, 2]) / intrinsics[0, 0], (float(v) - intrinsics[1, 2]) / intrinsics[1, 1], 1.0)  # pyright: ignore[reportOperatorIssue]
    )
    origin = wp.transform_point(T_world_camera, wp.vec3(0.0))  # pyright: ignore[reportCallIssue]
    direction = wp.transform_vector(T_world_camera, ray)  # pyright: ignore[reportCallIssue]
    hit = wp.mesh_query_ray(mesh, origin, direction, 3.0 / ray[2])
    state = wp.rand_init(frame, v * depth.shape[1] + u)
    d = wp.where(hit.result, hit.t * ray[2] + 0.005 * wp.randn(state), 0.0)
    depth[v, u] = d
    points[v * depth.shape[1] + u] = origin + direction * (d / ray[2])


@wp.kernel
def _extract_surface_kernel(
    volume_id: wp.uint64,
    observations: wp.array3d(dtype=wp.vec2),
    points: wp.array(dtype=wp.vec3),
    count: wp.array(dtype=wp.int32),
):
    i, j, k = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    current = observations[i, j, k]
    if current[1] == 0.0:
        return
    for axis in range(3):
        neighbor = wp.vec3i(i, j, k)
        neighbor[axis] += 1  # pyright: ignore[reportIndexIssue]
        if neighbor[axis] >= observations.shape[axis]:  # pyright: ignore[reportIndexIssue]
            continue
        following = observations[neighbor[0], neighbor[1], neighbor[2]]  # pyright: ignore[reportIndexIssue]
        # only observed sign changes reconstruct a surface; unknown space adds no caps
        if following[1] > 0.0 and (current[0] > 0.0) != (following[0] > 0.0):
            index = wp.vec3(float(i), float(j), float(k))
            index[axis] += current[0] / (current[0] - following[0])  # pyright: ignore[reportIndexIssue]
            points[wp.atomic_add(count, 0, 1)] = wp.volume_index_to_world(volume_id, index)  # pyright: ignore[reportCallIssue]


def main(port: int = 8080):
    wp.init()
    wp.set_device("cuda:0")
    # --- scene used only to generate depth images ---
    scene = trimesh.creation.box((0.8, 0.6, 0.04))
    scene += trimesh.creation.box((0.28, 0.28, 0.35)).apply_translation((0.2, 0, 0.2))
    scene += trimesh.creation.icosphere(subdivisions=3, radius=0.16).apply_translation((-0.2, 0, 0.2))
    mesh = wp.Mesh(wp.array(scene.vertices, dtype=wp.vec3), wp.array(scene.faces.ravel(), dtype=wp.int32))
    intrinsics = wp.mat33(100, 0, 63.5, 0, 100, 63.5, 0, 0, 1)
    transforms = []
    for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        T_world_camera = np.eye(4, dtype=np.float32)
        T_world_camera[:3, 3] = [np.cos(angle), np.sin(angle), 0.6]
        forward = np.array([0, 0, 0.2]) - T_world_camera[:3, 3]
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0, 0, 1])
        right /= np.linalg.norm(right)
        T_world_camera[:3, :3] = np.column_stack((right, np.cross(forward, right), forward))
        transforms.append(T_world_camera)

    # --- persistent fusion and display buffers ---
    bounds = np.array([[-0.5, -0.4, -0.1], [0.5, 0.4, 0.6]])
    volume = SdfVolume.from_points(wp.empty(0, dtype=wp.vec3), *bounds)
    observations = volume._fusion  # inspect the retained TSDF only to display its zero surface
    assert observations is not None
    depth = wp.zeros((128, 128), dtype=wp.float32)
    points = wp.empty(128 * 128, dtype=wp.vec3)
    surface = wp.empty(3 * int(np.prod(observations.shape)), dtype=wp.vec3)
    count = wp.zeros(1, dtype=wp.int32)

    # --- viewer ---
    server = viser.ViserServer(port=port, label="Depth fusion")
    server.scene.set_up_direction("+z")
    server.gui.add_markdown(
        "Depth → orange XYZ → blue TSDF. 12 views, 5 mm noise. Depth: black = invalid, white = 2 m."
    )
    playing = server.gui.add_checkbox("Play", True)
    replay = server.gui.add_button("Restart fusion")
    depth_image = server.gui.add_image(np.zeros((128, 128, 3), np.uint8), label="Current depth (m)")
    clouds = []
    for root, x, color in (("/depth", -0.65, (240, 160, 60)), ("/reconstruction", 0.65, (80, 160, 240))):
        server.scene.add_frame(root, show_axes=False, position=(x, 0, 0))
        server.scene.add_label(f"{root}/title", root[1:], position=(0, 0, 0.75))
        clouds.append(server.scene.add_point_cloud(f"{root}/points", np.empty((0, 3)), colors=color, point_size=0.01))

    @server.on_client_connect
    def set_camera(client: viser.ClientHandle):
        client.camera.position = (1.4, -2.2, 1.6)
        client.camera.look_at = (0, 0, 0.2)

    restart = Event()
    replay.on_click(lambda _: restart.set())

    # --- render, integrate, reconstruct, display ---
    index = 0
    while True:
        if restart.is_set():
            restart.clear()
            volume.update(reset=True)
            index = 0
            playing.value = True
        if playing.value and server.get_clients() and index < len(transforms):
            T_world_camera = wp.mat44(transforms[index])
            wp.launch(_render_depth_kernel, depth.shape, [mesh.id, T_world_camera, intrinsics, index, depth, points])
            volume.update(depth=depth, intrinsics=intrinsics, T_world_camera=T_world_camera)
            count.zero_()
            wp.launch(_extract_surface_kernel, observations.shape, [volume.volume.id, observations, surface, count])
            size = int(count.numpy()[0])
            assert size > 0
            # readbacks and image conversion are for the viewer, not the fusion path
            depth_np = depth.numpy()
            depth_image.image = (depth_np.clip(0, 2) * 127.5).astype(np.uint8)[..., None].repeat(3, axis=2)
            clouds[0].points = points.numpy()[depth_np.ravel() > 0]
            clouds[1].points = surface.numpy()[:size]
            index += 1
            print(f"Fused frame {index}/12: {size} reconstructed surface points", flush=True)
            if index == len(transforms):
                playing.value = False
        time.sleep(0.5)


if __name__ == "__main__":
    tyro.cli(main)
