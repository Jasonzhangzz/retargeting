"""Online hand retargeting: track Manus keypoints with a Sharpa hand."""

import pickle
import time

import numpy as np
import viser
import warp as wp
from viser.extras import ViserUrdf

from robokit.assets import fetch
from robokit.assets.robots.end_effectors import sharpa_hand
from robokit.helpers.hand_retargeting import HandRetargeterOnline
from robokit.helpers.hand_retargeting.presets import sharpa
from robokit.robo import Robot
from robokit.utils.visualize_utils import ViserHandSkeleton
from robokit.xform.numpy import quaternion_apply


def main() -> None:
    wp.init()
    device = "cuda" if wp.is_cuda_available() else "cpu"

    robot = Robot.load(
        str(sharpa_hand.URDF_PATH_RIGHT),
        load_collision_spheres=True,
        collision_spheres_path=sharpa_hand.COLLISION_SPHERE_PATH_RIGHT,
        self_collision_ignore_path=sharpa_hand.SPHERE_SELF_COLLISION_IGNORE_PATH_RIGHT,
    )
    retargeting = HandRetargeterOnline(sharpa.online, robot, sharpa.spec, device=device)
    retargeting.warmup(1)
    data_dir = fetch(["datasets/hand_retargeting_vector/**"]) / "datasets/hand_retargeting_vector"
    with open(data_dir / "manus_data_right.pkl", "rb") as file:
        keypoints = np.asarray(pickle.load(file, encoding="latin1")["keypoints"], dtype=np.float32)
    keypoints -= keypoints[:, :1]

    fk_state = robot.forward_kinematics(robot.state(q=robot.zero_q))
    t_base_wrist = fk_state.T_world_link.numpy().reshape(-1, 7)[robot.spec.link_names.index("right_hand_C_MC"), :3]
    server = viser.ViserServer()
    server.scene.add_grid("/grid", 1.0, 1.0)
    human = ViserHandSkeleton(server, "/human", position=(0.0, -0.1, 0.0))
    robot_frame = server.scene.add_frame("/robot", show_axes=False)
    urdf_vis = ViserUrdf(server, sharpa_hand.URDF_PATH_RIGHT, root_node_name="/robot")

    frame = 0
    while True:
        frame_index = frame % len(keypoints)
        frame += 1
        packed = retargeting.solve_numpy(keypoints[frame_index])
        q_wxyz_world_base = packed[0, 3:7]
        with server.atomic():
            robot_frame.position = -quaternion_apply(q_wxyz_world_base, t_base_wrist) + (0.0, 0.1, 0.0)
            robot_frame.wxyz = q_wxyz_world_base
            urdf_vis.update_cfg(packed[0, 7:])
            human.update(keypoints[frame_index])
        time.sleep(0.01)


if __name__ == "__main__":
    main()
