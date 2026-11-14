"""Online arm-hand retargeting: track DexYCB with a fixed xArm7 and XHand."""

import time

import numpy as np
import viser
import warp as wp
from viser.extras import ViserUrdf

from robokit.assets.robots.assembly import xarm7_xhand as xarm7_xhand_asset
from robokit.helpers.hand_retargeting import HandRetargeterOnline
from robokit.helpers.hand_retargeting.dexycb_utils import load_dexycb_grasp
from robokit.helpers.hand_retargeting.presets import xarm7_xhand
from robokit.robo import Robot
from robokit.xform.numpy import quaternion_apply, quaternion_invert


T_WORLD_BASE = np.array([0.9, 0.9, 0.0, 0.70710678, 0.0, 0.0, -0.70710678], np.float32)


def main() -> None:
    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"

    robot = Robot.load(str(xarm7_xhand_asset.URDF_PATH))
    target_index = {name: index for index, name in enumerate(xarm7_xhand.spec.target_names)}
    contact_target_indices = [target_index[name] for name in xarm7_xhand.spec.contact_target_names]
    contact_links = [xarm7_xhand.spec.target_link_names[name] for name in xarm7_xhand.spec.contact_target_names]
    contact_link_indices = [robot.spec.link_names.index(name) for name in contact_links]

    retargeting = HandRetargeterOnline(xarm7_xhand.online, robot, xarm7_xhand.spec, device=device)
    retargeting.warmup(1)

    data = load_dexycb_grasp(0, device=device)
    q_wxyz_base_world = quaternion_invert(T_WORLD_BASE[3:])
    keypoints_base = quaternion_apply(q_wxyz_base_world, data["keypoints"] - T_WORLD_BASE[:3]).astype(np.float32)
    num_frames = len(data["keypoints"])

    server = viser.ViserServer()
    server.scene.add_grid("/grid", 1.0, 1.0, position=(0.5, 0.5, 0.0))
    robot_frame = server.scene.add_frame("/robot", show_axes=False)
    robot_frame.position, robot_frame.wxyz = T_WORLD_BASE[:3], T_WORLD_BASE[3:]
    urdf_vis = ViserUrdf(server, xarm7_xhand_asset.URDF_PATH, root_node_name="/robot")
    object_frame = server.scene.add_frame("/object", show_axes=False)
    server.scene.add_mesh_simple(
        "/object/mesh",
        data["object_mesh"].vertices,
        data["object_mesh"].faces,
        color=(180, 180, 190),
        opacity=0.6,
    )
    mano = server.scene.add_mesh_simple(
        "/mano", data["mano_vertices"][0], data["mano_faces"], color=(120, 200, 130), opacity=0.5
    )
    frame = 0
    while True:
        frame_index = frame % num_frames
        frame += 1
        packed = retargeting.solve_numpy(keypoints_base[frame_index])

        if frame % 60 == 1:
            state = robot.state(q=wp.from_numpy(packed[:, 7:], dtype=wp.float32, device=device))
            robot.forward_kinematics(state)
            positions = state.T_world_link.numpy()[0, contact_link_indices, :3]
            error = np.linalg.norm(positions - keypoints_base[frame_index, contact_target_indices], axis=1)
            print(f"frame {frame} tips err {error.mean() * 1000:.1f} mm")

        with server.atomic():
            urdf_vis.update_cfg(packed[0, 7:])
            object_frame.position = data["object_pos"][frame_index]
            object_frame.wxyz = data["object_quat_wxyz"][frame_index]
            mano.vertices = data["mano_vertices"][frame_index]
        time.sleep(0.03)


if __name__ == "__main__":
    main()
