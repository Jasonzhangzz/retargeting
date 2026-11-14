"""Batched online retargeting: track several DexYCB captures with xArm7 and XHand."""

import time

import numpy as np
import viser
import warp as wp

from robokit.assets.robots.assembly import xarm7_xhand as xarm7_xhand_asset
from robokit.helpers.hand_retargeting import HandRetargeterOnline
from robokit.helpers.hand_retargeting.dexycb_utils import load_dexycb_grasp
from robokit.helpers.hand_retargeting.presets import xarm7_xhand
from robokit.robo import Robot
from robokit.utils.visualize_utils import ViserBatchUrdf
from robokit.xform.numpy import quaternion_apply, quaternion_invert


NUM_CAPTURES = 4
CELL_SPACING = 1.5
T_WORLD_BASE = np.array([0.9, 0.9, 0.0, 0.70710678, 0.0, 0.0, -0.70710678], np.float32)


def main() -> None:
    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"

    robot = Robot.load(str(xarm7_xhand_asset.URDF_PATH), load_meshes=True)
    target_index = {name: index for index, name in enumerate(xarm7_xhand.spec.target_names)}
    contact_target_indices = [target_index[name] for name in xarm7_xhand.spec.contact_target_names]
    contact_links = [xarm7_xhand.spec.target_link_names[name] for name in xarm7_xhand.spec.contact_target_names]
    contact_link_indices = [robot.spec.link_names.index(name) for name in contact_links]

    retargeting = HandRetargeterOnline(xarm7_xhand.online, robot, xarm7_xhand.spec, device=device)
    retargeting.warmup(NUM_CAPTURES)

    captures = [load_dexycb_grasp(i, device=device) for i in range(NUM_CAPTURES)]
    num_frames = min(len(data["keypoints"]) for data in captures)
    keypoints = np.stack([data["keypoints"][:num_frames] for data in captures])
    q_wxyz_base_world = quaternion_invert(T_WORLD_BASE[3:])
    keypoints_base = quaternion_apply(q_wxyz_base_world, keypoints - T_WORLD_BASE[:3]).astype(np.float32)

    offset = np.zeros((NUM_CAPTURES, 3), np.float32)
    offset[:, 0] = np.arange(NUM_CAPTURES) * CELL_SPACING
    T_world_base = np.tile(T_WORLD_BASE, (NUM_CAPTURES, 1))
    T_world_base[:, :3] += offset

    server = viser.ViserServer()
    server.scene.add_grid(
        "/grid",
        2.0 + CELL_SPACING * NUM_CAPTURES,
        2.0,
        position=((NUM_CAPTURES - 1) * CELL_SPACING / 2 + 0.5, 0.5, 0.0),
    )
    batch_urdf = ViserBatchUrdf(server, robot, batch_size=NUM_CAPTURES, root_node_name="/arms")
    object_frames = [server.scene.add_frame(f"/object_{i}", show_axes=False) for i in range(NUM_CAPTURES)]
    for i, data in enumerate(captures):
        server.scene.add_mesh_simple(
            f"/object_{i}/mesh",
            data["object_mesh"].vertices,
            data["object_mesh"].faces,
            color=(180, 180, 190),
            opacity=0.6,
        )
    manos = [
        server.scene.add_mesh_simple(
            f"/mano_{i}",
            captures[i]["mano_vertices"][0] + offset[i],
            captures[i]["mano_faces"],
            color=(120, 200, 130),
            opacity=0.5,
        )
        for i in range(NUM_CAPTURES)
    ]
    frame = 0
    while True:
        frame_index = frame % num_frames
        frame += 1
        packed = retargeting.solve_numpy(keypoints_base[:, frame_index])

        if frame % 60 == 1:
            state = robot.state(q=wp.from_numpy(packed[:, 7:], dtype=wp.float32, device=device))
            robot.forward_kinematics(state)
            positions = state.T_world_link.numpy()[:, contact_link_indices, :3]
            error = (
                np.linalg.norm(positions - keypoints_base[:, frame_index, contact_target_indices], axis=2).mean(axis=1)
                * 1000
            )
            print(f"frame {frame} batch {NUM_CAPTURES} tips err mm {np.round(error, 1)}")

        with server.atomic():
            batch_urdf.update_cfg(packed[:, 7:], T_world_base=T_world_base)
            for i in range(NUM_CAPTURES):
                object_frames[i].position = captures[i]["object_pos"][frame_index] + offset[i]
                object_frames[i].wxyz = captures[i]["object_quat_wxyz"][frame_index]
                manos[i].vertices = captures[i]["mano_vertices"][frame_index] + offset[i]
        time.sleep(0.03)


if __name__ == "__main__":
    main()
