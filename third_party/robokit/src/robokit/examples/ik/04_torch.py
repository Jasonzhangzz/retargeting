"""Batch IK example with torch tensor I/O: torch tensors in, torch tensors out."""

import time

import numpy as np
import torch
import warp as wp
from robot_descriptions.loaders.yourdfpy import load_robot_description

from robokit.helpers.ik import IKSolver, presets
from robokit.robo import Robot
from robokit.xform.warp.torch_wrappers import rot_tl_to_tf_mat


BATCH_SIZE = 128


def main():
    # --- setup ---
    wp.config.quiet = True
    wp.init()
    device = "cuda" if wp.is_cuda_available() else "cpu"

    target_pos = torch.stack(
        [
            torch.linspace(0.45, 0.6, BATCH_SIZE, dtype=torch.float32, device=device),
            torch.linspace(-0.18, 0.18, BATCH_SIZE, dtype=torch.float32, device=device),
            torch.linspace(0.35, 0.6, BATCH_SIZE, dtype=torch.float32, device=device),
        ],
        dim=-1,
    )
    target_matrices = rot_tl_to_tf_mat(tl=target_pos)

    # --- IK ---
    urdf = load_robot_description("panda_description")
    robot = Robot.load(urdf)
    ik = IKSolver(presets.basic, robot=robot, target_links="panda_hand", device=device)
    ik.warmup(batch_size=BATCH_SIZE)

    # --- solve ---
    t0 = time.perf_counter()
    result = ik.solve_torch(target_matrices)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # --- check ---
    fk_state = robot.forward_kinematics(robot.state(q=wp.from_torch(result.q)))
    hand_link_index = robot.spec.link_names.index("panda_hand")
    T_world_hand = fk_state.get_T_world_link(hand_link_index).numpy()
    pos_errors = np.linalg.norm(T_world_hand[:, :3] - target_pos.cpu().numpy(), axis=-1)
    quat_dot = np.abs(T_world_hand[:, 3])
    rot_errors = 2.0 * np.arccos(np.clip(quat_dot, -1.0, 1.0))
    success = (pos_errors < 0.01) & (rot_errors < 0.1)

    print(f"Batch size:     {BATCH_SIZE}")
    print(f"First solve:    {elapsed_ms:.2f} ms")
    print(f"Success rate:   {success.mean() * 100:.0f}%")
    print(f"Pos error mean: {np.mean(pos_errors) * 1000:.2f} mm")
    print(f"Rot error mean: {np.mean(rot_errors):.4f} rad")


if __name__ == "__main__":
    main()
