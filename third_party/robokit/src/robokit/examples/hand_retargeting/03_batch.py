"""Batched offline retargeting: solve several DexYCB trajectories together."""

import time

import numpy as np
import viser
import warp as wp
from viser.extras import ViserUrdf

from robokit.assets.robots.end_effectors import shadow_hand
from robokit.helpers.hand_retargeting import HandRetargeterOffline
from robokit.helpers.hand_retargeting.dexycb_utils import load_dexycb_grasp
from robokit.helpers.hand_retargeting.presets.shadow import offline, spec
from robokit.robo import Robot


NUM_CAPTURES = 4


def main() -> None:
    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"

    captures = [load_dexycb_grasp(i, device=device) for i in range(NUM_CAPTURES)]
    num_frames = min(len(data["keypoints"]) for data in captures)

    robot = Robot.load(str(shadow_hand.URDF_PATH_RIGHT))
    retargeting = HandRetargeterOffline(offline, robot, spec, device=device)
    packed = retargeting.solve_numpy(
        np.stack([data["keypoints"][:num_frames] for data in captures]),
        np.stack([data["wrist_quat_wxyz"][:num_frames] for data in captures]),
    )
    print(f"solved {NUM_CAPTURES} trajectories x {num_frames} frames")
    T_world_base, q = packed[..., :7], packed[..., 7:]

    cell = np.zeros((NUM_CAPTURES, 3), np.float32)
    cell[:, 0] = np.arange(NUM_CAPTURES) * 0.3
    offset = cell - T_world_base[:, 0, :3]

    server = viser.ViserServer()
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/grid", 1.0, 1.0, position=(0.5, -0.5, 0.0))
    robot_frames = [server.scene.add_frame(f"/hands/robot_{i}", show_axes=False) for i in range(NUM_CAPTURES)]
    urdf_visualizers = [
        ViserUrdf(server, shadow_hand.URDF_PATH_RIGHT, root_node_name=f"/hands/robot_{i}") for i in range(NUM_CAPTURES)
    ]
    object_frames = [server.scene.add_frame(f"/hands/object_{i}", show_axes=False) for i in range(NUM_CAPTURES)]
    for i, data in enumerate(captures):
        server.scene.add_mesh_simple(
            f"/hands/object_{i}/mesh",
            data["object_mesh"].vertices,
            data["object_mesh"].faces,
            color=(180, 180, 190),
            opacity=0.6,
        )
    manos = [
        server.scene.add_mesh_simple(
            f"/hands/mano_{i}",
            captures[i]["mano_vertices"][0] + offset[i],
            captures[i]["mano_faces"],
            color=(120, 200, 130),
            opacity=0.5,
        )
        for i in range(NUM_CAPTURES)
    ]
    playing = server.gui.add_checkbox("Play", True)
    frame_slider = server.gui.add_slider("Frame", 0, num_frames - 1, 1, 0)

    while True:
        if playing.value:
            frame_slider.value = (frame_slider.value + 1) % num_frames
        frame_index = frame_slider.value
        T_display_base = T_world_base[:, frame_index].copy()
        T_display_base[:, :3] += offset
        with server.atomic():
            for i in range(NUM_CAPTURES):
                robot_frames[i].position = T_display_base[i, :3]
                robot_frames[i].wxyz = T_display_base[i, 3:]
                urdf_visualizers[i].update_cfg(q[i, frame_index])
                object_frames[i].position = captures[i]["object_pos"][frame_index] + offset[i]
                object_frames[i].wxyz = captures[i]["object_quat_wxyz"][frame_index]
                manos[i].vertices = captures[i]["mano_vertices"][frame_index] + offset[i]
        time.sleep(0.03)


if __name__ == "__main__":
    main()
