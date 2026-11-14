"""G1 humanoid IK with fixed feet and CoM stabilization."""

import time

import numpy as np
import viser
import warp as wp
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf

from robokit.helpers.ik import IKContext, IKSolver, IKSolverConfig
from robokit.opt.lm_optimizer import LMOptimizerConfig
from robokit.opt.population_solver import PopulationSolverConfig, Stage
from robokit.opt.sampler import RobertsSamplerConfig
from robokit.robo import Robot
from robokit.terms.dense.com_position_task import ComPositionTask
from robokit.terms.dense.configuration_task import ConfigurationTask
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.position_task import PositionTask
from robokit.terms.dense.rotation_task import RotationTask
from robokit.terms.dense.smoothness_task import SmoothnessTask
from robokit.utils.warp_utils import stack, wp_vec7


# Hands are draggable targets; fixed position and rotation targets pin the feet.
TARGET_LINKS = ["left_rubber_hand", "right_rubber_hand"]
FOOT_LINKS = ["left_ankle_roll_link", "right_ankle_roll_link"]


def main():
    wp.init()
    device = "cuda" if wp.is_cuda_available() else "cpu"
    urdf = load_robot_description("g1_description")
    robot = Robot.load(urdf)
    foot_idx = [robot.spec.link_names.index(n) for n in FOOT_LINKS]

    # Home pose: lift the base so the feet rest on the ground (z=0); references for the balance tasks.
    base = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    probe = robot.forward_kinematics(robot.state(T_world_base=wp.from_numpy(base, dtype=wp_vec7, device=device)))
    base[0, 2] = -float(probe.T_world_link.numpy()[0, :, 2].min())
    home = robot.forward_kinematics(robot.state(T_world_base=wp.from_numpy(base, dtype=wp_vec7, device=device)))
    home_feet = stack([home.get_T_world_link(i) for i in foot_idx], axis=1)
    home_com = robot.compute_center_of_mass(home).com_world.numpy()[0]

    # --- IK ---
    config = (
        IKSolverConfig(
            enable_T_world_base=True,
            solver=PopulationSolverConfig(
                stages=[
                    Stage(10, 64, LMOptimizerConfig(lm_lambda=10.0)),
                    Stage(16, 4, LMOptimizerConfig(lm_lambda=1.0)),
                    Stage(0, 1),
                ],
                cuda_graph_mode="full",
            ),
            sampler=RobertsSamplerConfig(keep_first=True, scale=1.0),
        )
        .add(PositionTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=1.5))
        .add(RotationTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=0.3))
        .add(PositionTask(robot, foot_idx, home_feet, weight=5.0, fixed_target=True))
        .add(RotationTask(robot, foot_idx, home_feet, weight=2.0, fixed_target=True))
        .add(ComPositionTask(robot, target_com_position=home_com, weight=[1.0, 1.0, 0.0]))
        .add(PositionLimit(weight=50.0))
        # base_weight locks base roll/pitch, frees xyz+yaw (so the pelvis can shift to balance)
        .add(
            ConfigurationTask(
                rest_q=IKContext.REST_Q,
                T_world_base_rest=IKContext.REST_BASE,
                weight=0.05,
                base_weight=[0.0, 0.0, 0.0, 100.0, 100.0, 0.0],
            )
        )
        .add(SmoothnessTask(prev_var=IKContext.PREVIOUS_STATE, weight=0.03, base_weight=0.0))
    )
    ik = IKSolver(config, robot=robot, target_links=TARGET_LINKS, device=device)
    ik.warmup(batch_size=1)

    # --- viewer ---
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=3, height=3)
    base_frame = server.scene.add_frame(
        "/pelvis",
        show_axes=False,  # this node only roots the robot; don't draw a 0.5 m axes triad at the pelvis
        position=home.T_world_base.numpy()[..., :3].squeeze(0),
        wxyz=home.T_world_base.numpy()[..., 3:].squeeze(0),
    )
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/pelvis")
    urdf_vis.update_cfg(robot.spec.zero_q)
    ik_targets = []
    for name in TARGET_LINKS:
        pose = home.get_T_world_link(robot.spec.link_names.index(name))
        ik_targets.append(
            server.scene.add_transform_controls(
                f"/ik_targets/{name}",
                scale=0.15,
                position=pose.numpy()[..., :3].squeeze(0),
                wxyz=pose.numpy()[..., 3:].squeeze(0),
            )
        )
    timing = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    prev = home  # previous solve: warm-start + smoothness reference for the next one
    while True:
        # --- IK: one solve per frame ---
        t0 = time.time()
        positions = [t.position for t in ik_targets]
        wxyzs = [t.wxyz for t in ik_targets]
        solved = ik.solve_numpy(
            np.stack([np.concatenate([p, q]) for p, q in zip(positions, wxyzs)], axis=-2),
            prev_state=prev,
            init_state=prev,
            rest_state=home,
        )
        timing.value = 0.99 * timing.value + 0.01 * (time.time() - t0) * 1000

        # --- viewer: visualize the result ---
        base_np = solved.T_world_base.numpy()[0]
        with server.atomic():
            urdf_vis.update_cfg(solved.q.numpy()[0])
            base_frame.position = base_np[:3]
            base_frame.wxyz = base_np[3:]
        prev = solved
        time.sleep(0.001)


if __name__ == "__main__":
    main()
