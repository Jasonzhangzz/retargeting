"""Mimic-joint IK example: ability hand tracking 5 fingertip targets."""

import time

import numpy as np
import viser
import warp as wp
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf

from robokit.helpers.ik import IKContext, IKSolver, IKSolverConfig
from robokit.opt.lm_optimizer import LMOptimizerConfig
from robokit.opt.population_solver import PopulationSolverConfig, Stage
from robokit.robo import Robot
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.position_task import PositionTask
from robokit.terms.dense.smoothness_task import SmoothnessTask


def main():
    wp.init()
    device = "cuda" if wp.is_cuda_available() else "cpu"

    urdf = load_robot_description("ability_hand_description")
    target_link_names = ["thumb_anchor", "index_anchor", "middle_anchor", "ring_anchor", "pinky_anchor"]

    robot = Robot.load(urdf)
    init_state = robot.forward_kinematics(robot.state())  # zero-pose FK for the initial gizmo placements

    # --- IK ---
    config = (
        IKSolverConfig(
            solver=PopulationSolverConfig(
                stages=[Stage(30, 1, LMOptimizerConfig(lm_lambda=1.0))], cuda_graph_mode="full"
            ),
        )
        .add(PositionTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=1.0))
        .add(PositionLimit(weight=50.0))
        .add(SmoothnessTask(prev_var=IKContext.PREVIOUS_STATE, weight=0.1))
    )
    ik = IKSolver(config, robot=robot, target_links=target_link_names, device=device)
    ik.warmup(batch_size=1)

    # --- viewer ---
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")
    urdf_vis.update_cfg(robot.spec.zero_q)

    ik_targets = []
    for link_name in target_link_names:
        link_idx = robot.spec.link_names.index(link_name)
        T_world_link = init_state.get_T_world_link(link_idx)
        ik_target = server.scene.add_transform_controls(
            f"/ik_targets/{link_name}",
            scale=0.05,
            position=T_world_link.numpy()[0, :3],
            wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        ik_targets.append(ik_target)

    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    prev_q = robot.spec.zero_q
    prev_state = robot.state(q=wp.from_numpy(prev_q.reshape(1, -1).astype(np.float32), dtype=wp.float32, device=device))

    while True:
        # --- IK: solve (5 fingertip targets) ---
        start_time = time.time()
        positions = [t.position for t in ik_targets]
        wxyzs = [t.wxyz for t in ik_targets]
        state = ik.solve_numpy(
            np.stack([np.concatenate([p, q]) for p, q in zip(positions, wxyzs)], axis=-2),
            prev_state=prev_state,
            init_state=prev_state,
        )
        solved_q_np = state.q.numpy()[0]
        elapsed_time = time.time() - start_time
        timing_handle.value = 0.99 * timing_handle.value + 0.01 * (elapsed_time * 1000)

        # --- viewer: render ---
        urdf_vis.update_cfg(solved_q_np)
        prev_state = state
        time.sleep(0.001)


if __name__ == "__main__":
    main()
