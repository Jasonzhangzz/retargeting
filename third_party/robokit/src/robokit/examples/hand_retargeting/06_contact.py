"""Offline xArm7-XHand contact retargeting around an object mesh."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import trimesh
import warp as wp

from robokit.assets.robots.assembly import xarm7_xhand
from robokit.geom import MeshGeom, WarpScene
from robokit.helpers.hand_retargeting import HandRetargeterOffline, contact_config, refine_config
from robokit.helpers.hand_retargeting.dexycb_utils import load_dexycb_grasp
from robokit.helpers.hand_retargeting.presets.xarm7_xhand import offline, spec
from robokit.robo import Robot
from robokit.utils.visualize_utils import RetargetingPlayback, view_retargeting
from robokit.utils.warp_utils import wp_vec7
from robokit.xform.numpy import quaternion_apply, quaternion_invert, quaternion_multiply


T_WORLD_BASE = np.array([0.9, 0.9, 0.0, 0.70710678, 0.0, 0.0, -0.70710678], np.float32)
CONTACT_START_DISTANCE = 0.020
CONTACT_FULL_DISTANCE = 0.006
CONTACT_EMBED_DEPTH = 0.005


def main() -> None:
    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"

    robot = Robot.load(
        str(xarm7_xhand.URDF_PATH),
        load_meshes=True,
        load_collision_spheres=True,
        collision_spheres_path=xarm7_xhand.COLLISION_SPHERE_PATH,
    )
    # --- contacts: project DexYCB fingertips onto the object mesh ---
    data = load_dexycb_grasp(0, device=device)
    keypoints = data["keypoints"]
    object_pos = data["object_pos"]
    object_quat_wxyz = data["object_quat_wxyz"]
    num_frames = len(keypoints)
    contact_target_indices = np.asarray(
        [spec.target_names.index(name) for name in spec.contact_target_names], dtype=np.int32
    )
    contact_link_indices = np.asarray(
        [robot.spec.link_names.index(spec.target_link_names[name]) for name in spec.contact_target_names],
        dtype=np.int32,
    )
    q_wxyz_object_world = quaternion_invert(object_quat_wxyz)
    keypoints_object = quaternion_apply(q_wxyz_object_world[:, None], keypoints - object_pos[:, None]).astype(
        np.float32
    )
    fingertips_object = keypoints_object[:, contact_target_indices]
    closest, distance, _ = trimesh.proximity.closest_point(data["object_mesh"], fingertips_object.reshape(-1, 3))
    contact_points_object = closest.reshape(fingertips_object.shape).astype(np.float32)
    contact_mask = distance.reshape(fingertips_object.shape[:2]) < CONTACT_START_DISTANCE

    # --- initial solve: contact tracking in the fixed robot-base frame ---
    q_wxyz_base_world = quaternion_invert(T_WORLD_BASE[3:])
    keypoints_base = quaternion_apply(q_wxyz_base_world, keypoints - T_WORLD_BASE[:3]).astype(np.float32)
    contact_points_world = object_pos[:, None] + quaternion_apply(object_quat_wxyz[:, None], contact_points_object)
    contact_points_base = quaternion_apply(q_wxyz_base_world, contact_points_world - T_WORLD_BASE[:3]).astype(
        np.float32
    )
    direct_config = contact_config(offline)
    direct_config.task("collision").weight = 0.0
    direct = HandRetargeterOffline(direct_config, robot, spec, device=device)
    packed = direct.solve_with_contact_numpy(keypoints_base, contact_points_base, contact_mask)

    # --- refinement: keep the moving object mesh static in object coordinates ---
    T_object_base = np.empty((num_frames, 7), dtype=np.float32)
    T_object_base[:, :3] = quaternion_apply(q_wxyz_object_world, T_WORLD_BASE[:3] - object_pos)
    T_object_base[:, 3:] = quaternion_multiply(q_wxyz_object_world, T_WORLD_BASE[3:])
    init_state = robot.state(
        q=wp.from_numpy(packed[None, :, 7:], dtype=wp.float32, device=device),
        T_world_base=wp.from_numpy(T_object_base[None], dtype=wp_vec7, device=device),
    )
    robot.forward_kinematics(init_state)
    T_object_link = init_state.T_world_link.numpy()[0][:, contact_link_indices]
    contact_outward = fingertips_object - contact_points_object
    contact_outward /= np.maximum(np.linalg.norm(contact_outward, axis=-1, keepdims=True), 1e-8)
    contact_inward_local = quaternion_apply(quaternion_invert(T_object_link[..., 3:]), -contact_outward)
    local_contact_points = np.zeros_like(contact_points_object)
    for contact_index, link_index in enumerate(contact_link_indices):
        link_mesh = robot.spec.link_visual_geometries[robot.spec.link_names[link_index]].to_mesh()
        vertices = np.asarray(link_mesh.vertices, np.float32)
        local_contact_points[:, contact_index] = vertices[
            np.argmax(contact_inward_local[:, contact_index] @ vertices.T, axis=1)
        ]

    root_link_index = robot.spec.link_names.index(spec.target_link_names[spec.root_target_name])
    arm_joint_mask = robot.spec.link_ancestor_joints_mask[root_link_index]
    arm_dof_mask = np.any(robot.spec.joints_to_actuated_mapping[arm_joint_mask] != 0.0, axis=0)
    hand_joint_names = tuple(np.asarray(robot.spec.actuated_joint_names)[~arm_dof_mask])
    identity = wp.from_numpy(np.eye(4, dtype=np.float32)[None], dtype=wp.mat44, device=device)
    scene = WarpScene(1, device).add(MeshGeom([data["object_mesh"]], np.array([0, 1], dtype=np.int32), poses=identity))
    refinement = refine_config(
        replace(
            offline,
            optimize_base=False,
            active_joint_names=hand_joint_names,
            locked_prefix_frames=2,
        )
    )
    refinement.task("contact").weight = 300.0
    refinement.task("contact").contact_margin = 0.0
    refinement.task("anchor").dense_task.weight = 10.0
    refinement.task("acceleration").weight = 20.0
    refinement.task("collision").weight = 100.0
    refinement.task("collision").margin = 0.0
    contact_points_robot = T_object_link[..., :3] + quaternion_apply(T_object_link[..., 3:], local_contact_points)
    contact_strength = np.clip(
        (CONTACT_START_DISTANCE - distance.reshape(contact_mask.shape))
        / (CONTACT_START_DISTANCE - CONTACT_FULL_DISTANCE),
        0.0,
        1.0,
    ).astype(np.float32)
    contact_strength = contact_strength * contact_strength * (3.0 - 2.0 * contact_strength)
    contact_points_target = contact_points_object - CONTACT_EMBED_DEPTH * contact_outward
    contact_points_target = contact_points_robot + contact_strength[..., None] * (
        contact_points_target - contact_points_robot
    )
    refine = HandRetargeterOffline(refinement, robot, spec, scene=scene, device=device)
    packed = refine.solve_with_contact_numpy(
        keypoints_object,
        contact_points_target,
        contact_mask,
        init_state=init_state,
        local_contact_points=local_contact_points,
    )
    q, T_object_base = packed[:, 7:], packed[:, :7]
    T_world_base = np.empty_like(T_object_base)
    T_world_base[:, :3] = object_pos + quaternion_apply(object_quat_wxyz, T_object_base[:, :3])
    T_world_base[:, 3:] = quaternion_multiply(object_quat_wxyz, T_object_base[:, 3:])
    print(f"contact retargeting of {num_frames} frames ({int(contact_mask.any(1).sum())} touching)")

    # --- viewer: shared with the batch example ---
    def load_clip(path: Path) -> RetargetingPlayback:
        return RetargetingPlayback(
            qpos=np.concatenate([T_world_base, q], axis=1),
            object_meshes=[data["object_mesh"]],
            T_world_object=np.concatenate([object_pos, object_quat_wxyz], axis=1)[:, None],
            targets_world=contact_points_world,
            contact_mask=contact_mask,
            human_vertices=data["mano_vertices"],
            human_faces=data["mano_faces"],
        )

    view_retargeting(robot, {data["capture_name"]: Path("clip")}, load_clip)


if __name__ == "__main__":
    main()
