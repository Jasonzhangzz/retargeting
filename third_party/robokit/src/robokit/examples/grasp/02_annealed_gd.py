"""Optimize xHand mesh grasps with annealed GD, redraw contacts, and browse results in viser."""

import time

import numpy as np
import torch
import trimesh
import viser
import warp as wp
import yourdfpy
from viser.extras import ViserUrdf

from robokit.assets.robots.end_effectors import xhand
from robokit.geom import MeshGeom, WarpScene
from robokit.helpers.grasp import ContactVar
from robokit.lie.se3 import se3_to_matrix
from robokit.opt.gd_optimizer import AnnealedGDConfig, AnnealedGDOptimizer
from robokit.opt.var_values import VarKey, VarValues
from robokit.robo.robot import Robot
from robokit.robo.robot_spec import load_link_points
from robokit.terms.dense.force_closure_task import ForceClosureTask
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.scene_collision_task import SceneCollisionTask
from robokit.terms.dense.scene_distance_task import SceneDistanceTask
from robokit.terms.dense.self_collision_task import SelfCollisionTask
from robokit.utils.ee_coord_utils import HandCoordinateSpec, ee_coord_conversion
from robokit.utils.warp_utils import wp_vec7
from robokit.xform.utils import look_at_rotation
from robokit.xform.warp.torch_wrappers import matrix_to_quaternion


NUM_GRASPS = 32
NUM_CONTACTS = 5
XHAND_COORD = HandCoordinateSpec(palm_forward_axis="x", four_fingers_up_axis="z")
OPENGL_COORD = HandCoordinateSpec(palm_forward_axis="-z", four_fingers_up_axis="y")


def grasp_terms(robot: Robot, target_scene: WarpScene, collision_scene: WarpScene, device: str) -> list:
    """The grasp energy over a `contacts` var; GD consumes each weight `w` as the potential `0.5 * w * r²`."""
    scale = lambda w: float(np.sqrt(2.0 * w))  # noqa: E731
    distance = SceneDistanceTask(
        robot=robot,
        warp_meshes=target_scene,
        num_contact_points=NUM_CONTACTS,
        points_var_key=VarKey.CONTACTS,
        weight=scale(100.0),
        residual_mode="sqrt_abs",
    )
    return [
        distance,
        ForceClosureTask(
            robot=robot,
            warp_meshes=target_scene,
            num_contact_points=NUM_CONTACTS,
            points_var_key=VarKey.CONTACTS,
            weight=scale(1.0),
            distance_task=distance,  # one contact query serves both terms
        ),
        PositionLimit(robot=robot, weight=scale(1.0), residual_mode="sqrt_abs"),
        SceneCollisionTask(
            robot=robot,
            scene=collision_scene,
            scene_indices=wp.zeros(1, dtype=wp.int32, device=device),
            weight=scale(100.0),
            margin=0.005,
            residual_mode="sqrt_abs",
        ),
        SelfCollisionTask(robot=robot, weight=scale(10.0), margin=0.02, residual_mode="sqrt_abs"),
    ]


def main():
    wp.init()
    device = "cuda" if wp.is_cuda_available() else "cpu"
    torch_device = torch.device(device)

    robot = Robot.load(
        xhand.URDF_GLB_PATH_RIGHT,
        load_meshes=True,
        load_collision_spheres=True,
        collision_spheres_path=xhand.COLLISION_SPHERE_PATH_RIGHT,
        self_collision_ignore_path=xhand.SPHERE_SELF_COLLISION_IGNORE_PATH_RIGHT,
    )
    hand_points, hand_link_indices = load_link_points(robot.link_names, xhand.CONTACT_POINTS_PATH_RIGHT)

    # --- scene: one graspable mesh at the origin ---
    mesh = trimesh.creation.cylinder(radius=0.035, height=0.19)
    scene_offsets = np.array([0, 1], np.int32)
    target_scene = WarpScene(1, device).add(MeshGeom([mesh], scene_offsets))
    collision_scene = WarpScene(1, device).add(MeshGeom([mesh], scene_offsets))

    # --- init: sample points around the mesh, back off along the surface normal, point the palm inwards ---
    query = wp.from_numpy(
        (np.random.default_rng(0).random((NUM_GRASPS, 3), dtype=np.float32) * 2.0 - 1.0) * 0.15,
        dtype=wp.vec3,
        device=device,
    )
    _, _, closest = target_scene.query_sdf(query, wp.from_numpy(scene_offsets, dtype=wp.int32, device=device))
    closest = wp.to_torch(closest)
    approach_dir = torch.nn.functional.normalize(wp.to_torch(query) - closest, dim=-1)
    tl = closest + approach_dir * torch.empty(NUM_GRASPS, 1, device=torch_device).uniform_(0.13, 0.20)
    rot = look_at_rotation(tl, closest, torch.tensor([0.0, 0.0, 1.0], device=torch_device)) @ ee_coord_conversion(
        XHAND_COORD, OPENGL_COORD, return_tensors="pt"
    ).to(torch_device)
    init_base = torch.cat([tl, matrix_to_quaternion(rot)], dim=-1)
    init_q = torch.full((NUM_GRASPS, robot.num_actuated_joints), 0.1, device=torch_device)

    # --- variables: the hand pose, clamped to its joint limits after every step, and the contacts it grasps with ---
    state = robot.state(
        q=wp.from_torch(init_q.contiguous(), dtype=wp.float32),
        T_world_base=wp.from_torch(init_base.contiguous(), dtype=wp_vec7),
    )
    state.clamp_q_on_integrate = True
    # every step proposes a pose and, with `switch_probability`, fresh contacts; Metropolis keeps or reverts both
    contacts = ContactVar(hand_points, hand_link_indices, NUM_GRASPS, NUM_CONTACTS, 0.5, device=device)
    var = VarValues({VarKey.ROBOT: state, VarKey.CONTACTS: contacts})
    terms = grasp_terms(robot, target_scene, collision_scene, device)

    # --- optimize ---
    config = AnnealedGDConfig(max_iter=2000, cuda_graph_mode="iter", learning_rate=0.005, seed=0)
    solver = AnnealedGDOptimizer(terms, config, device)
    start = time.perf_counter()
    values, costs = solver.solve(var)
    wp.synchronize()
    print(f"{NUM_GRASPS} grasps in {time.perf_counter() - start:.2f}s, mean cost {costs.numpy().mean():.3f}")

    final_state = values.get(VarKey.ROBOT)
    q = wp.to_torch(final_state.q).cpu().numpy()
    base_mat = wp.to_torch(se3_to_matrix(final_state.T_world_base)).cpu().numpy()

    # --- viewer: slide through the optimized grasps ---
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=0.5, height=0.5)
    server.scene.add_mesh_trimesh("/object", mesh)
    base_frame = server.scene.add_frame("/robot_base", show_axes=False)
    urdf_vis = ViserUrdf(server, yourdfpy.URDF.load(str(xhand.URDF_GLB_PATH_RIGHT)), root_node_name="/robot_base")
    slider = server.gui.add_slider("Grasp", 0, NUM_GRASPS - 1, 1, 0)

    def show(_=None):
        idx = int(slider.value)
        with server.atomic():
            base_frame.position = base_mat[idx, :3, 3]
            base_frame.wxyz = trimesh.transformations.quaternion_from_matrix(base_mat[idx])
            urdf_vis.update_cfg(q[idx])

    slider.on_update(show)
    show()
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
