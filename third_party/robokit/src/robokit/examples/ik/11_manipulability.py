"""Manipulability-aware IK example: slide the ManipulabilityTask weight live, no solver rebuild."""

import numpy as np
import viser
import warp as wp
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf

from robokit.helpers.ik import IKContext, IKSolver, IKSolverConfig, presets
from robokit.opt.sampler import RobertsSamplerConfig
from robokit.robo import Robot
from robokit.terms.dense.manipulability_task import ManipulabilityTask
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.position_task import PositionTask
from robokit.terms.dense.rotation_task import RotationTask
from robokit.terms.dense.smoothness_task import SmoothnessTask


def main():
    wp.init()
    device = "cuda" if wp.is_cuda_available() else "cpu"
    urdf = load_robot_description("panda_description")

    # --- IK ---
    robot = Robot.load(urdf)
    link_idx = robot.spec.link_names.index("panda_hand")
    manip_task = ManipulabilityTask(frame_index=IKContext.TARGET_LINK, weight=0.01)
    config = (
        IKSolverConfig(solver=presets.smooth.solver, sampler=RobertsSamplerConfig(keep_first=True, scale=1.0))
        .add(PositionTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=20.0))
        .add(RotationTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=10.0))
        .add(PositionLimit(weight=50.0))
        .add(SmoothnessTask(prev_var=IKContext.PREVIOUS_STATE, weight=1.0))
        .add(manip_task)
    )
    ik = IKSolver(config, robot=robot, target_links="panda_hand", device=device)
    ik.warmup(batch_size=1)
    init_state = robot.forward_kinematics(
        robot.state(q=wp.from_numpy(robot.spec.midrange_q[None], dtype=wp.float32, device=device))
    )

    # --- viewer ---
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    urdf_vis = ViserUrdf(server, urdf)
    urdf_vis.update_cfg(robot.spec.midrange_q)

    ik_target = server.scene.add_transform_controls(
        "/ik_target", scale=0.2, position=(0.5, 0.0, 0.5), wxyz=(0, 0, 1, 0)
    )
    weight_slider = server.gui.add_slider("Manipulability weight", min=0.0, max=0.1, step=1e-3, initial_value=0.01)
    weight_slider.on_update(lambda _: ik.set_weight(manip_task.name, weight_slider.value))
    manip_readout = server.gui.add_number("Manipulability", 0.0, step=1e-4, disabled=True)

    prev_state = init_state
    while True:
        # --- IK: one solve per frame ---
        solved = ik.solve_numpy(
            np.concatenate([ik_target.position, ik_target.wxyz]),
            prev_state=prev_state,
            init_state=prev_state,
        )
        prev_state = solved

        # --- viewer: visualize the result + Yoshikawa manipulability sqrt(det(J Jᵀ)), J = v + w x p ---
        urdf_vis.update_cfg(solved.q.numpy()[0])
        state = robot.state(q=solved.q)
        robot.forward_kinematics(state)
        robot.compute_motion_subspace(state)
        Jsp = state.get_link_jacobian(link_idx, reference_frame="spatial").numpy()[0]  # (6, dof)
        p = state.get_T_world_link(link_idx).numpy()[0, :3]
        J = Jsp[:3] + np.cross(Jsp[3:], p[:, None], axisa=0, axisb=0, axisc=0)  # (3, dof)
        manip_readout.value = float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))


if __name__ == "__main__":
    main()
