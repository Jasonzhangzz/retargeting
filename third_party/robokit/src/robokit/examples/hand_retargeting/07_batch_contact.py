"""Retarget every right-hand DexYCB capture in bounded geometry-aware batches."""

import argparse
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import trimesh
import warp as wp

from robokit.geom import MeshGeom, WarpScene
from robokit.helpers.hand_retargeting import HandRetargeterOffline, contact_config, refine_config
from robokit.helpers.hand_retargeting.dexycb_utils import DexYCBVideoDataset, load_dexycb_grasp
from robokit.helpers.hand_retargeting.presets.xarm7_xhand import offline, spec
from robokit.robo import Robot
from robokit.utils.visualize_utils import RetargetingPlayback, view_retargeting
from robokit.utils.warp_utils import wp_vec7
from robokit.xform.numpy import quaternion_apply, quaternion_invert, quaternion_multiply


T_WORLD_BASE = np.array([0.9, 0.9, 0.0, 0.70710678, 0.0, 0.0, -0.70710678], np.float32)
CONTACT_START_DISTANCE = 0.020
CONTACT_FULL_DISTANCE = 0.006
CONTACT_EMBED_DEPTH = 0.005


def _view_results(
    output_dir: Path, robot: Optional[Robot] = None, dataset: Optional[DexYCBVideoDataset] = None
) -> None:
    files = {path.stem: path for path in sorted(output_dir.glob("*.npz")) if not path.name.endswith(".tmp.npz")}
    if not files:
        print("No saved clips to visualize.")
        return
    from robokit.assets.datasets import dexycb
    from robokit.assets.robots.assembly import xarm7_xhand

    if robot is None:
        robot = Robot.load(str(xarm7_xhand.URDF_PATH), load_meshes=True)
    if dataset is None:
        dataset = DexYCBVideoDataset(dexycb.DIR, is_right=True)
    indices = {dataset[index]["capture_name"]: index for index in range(len(dataset))}
    mesh_cache: Dict[Path, trimesh.Trimesh] = {}
    target_indices = [spec.target_names.index(name) for name in spec.contact_target_names]

    def load_clip(path: Path) -> RetargetingPlayback:
        data = load_dexycb_grasp(indices[path.stem], device="cpu", dataset=dataset, mesh_cache=mesh_cache)
        with np.load(path, allow_pickle=False) as result:
            frames = np.searchsorted(data["frame_indices"], result["frame_indices"])
            if np.any(frames >= len(data["frame_indices"])) or not np.array_equal(
                data["frame_indices"][frames], result["frame_indices"]
            ):
                raise ValueError(f"saved source frames do not match {path.name}")
            positions, quaternions = result["object_pos"], result["object_quat_wxyz"]
            fingertips = data["keypoints"][frames][:, target_indices]
            points_object = quaternion_apply(quaternion_invert(quaternions)[:, None], fingertips - positions[:, None])
            closest, _, _ = trimesh.proximity.closest_point(data["object_mesh"], points_object.reshape(-1, 3))
            targets = positions[:, None] + quaternion_apply(quaternions[:, None], closest.reshape(fingertips.shape))
            return RetargetingPlayback(
                qpos=result["qpos"],
                object_meshes=[data["object_mesh"]],
                T_world_object=np.concatenate([positions, quaternions], axis=1)[:, None],
                targets_world=targets,
                contact_mask=result["contact_mask"],
                human_vertices=data["mano_vertices"][frames],
                human_faces=data["mano_faces"],
            )

    view_retargeting(robot, files, load_clip)


def _solve_batch(
    captures: List[Dict[str, Any]],
    robot: Robot,
    device: str,
    direct: HandRetargeterOffline,
    refine: HandRetargeterOffline,
) -> Tuple[np.ndarray, np.ndarray]:
    batch_size = len(captures)
    lengths = np.asarray([len(data["keypoints"]) for data in captures], dtype=np.int32)
    num_frames = int(lengths.max())

    def pad(name: str) -> np.ndarray:
        return np.stack(
            [
                np.pad(
                    data[name], ((0, num_frames - len(data[name])),) + ((0, 0),) * (data[name].ndim - 1), mode="edge"
                )
                for data in captures
            ]
        )

    # --- contacts: project each capture's fingertips onto its own object mesh ---
    keypoints = pad("keypoints")
    object_pos = pad("object_pos")
    object_quat_wxyz = pad("object_quat_wxyz")
    q_wxyz_object_world = quaternion_invert(object_quat_wxyz)
    keypoints_object = quaternion_apply(q_wxyz_object_world[:, :, None], keypoints - object_pos[:, :, None]).astype(
        np.float32
    )
    contact_target_indices = np.asarray([spec.target_names.index(name) for name in spec.contact_target_names])
    contact_link_indices = np.asarray(
        [robot.spec.link_names.index(spec.target_link_names[name]) for name in spec.contact_target_names]
    )
    fingertips_object = keypoints_object[:, :, contact_target_indices]
    contact_points_object = np.empty_like(fingertips_object)
    distance = np.empty(fingertips_object.shape[:-1], dtype=np.float32)
    for i, data in enumerate(captures):
        closest, distances, _ = trimesh.proximity.closest_point(
            data["object_mesh"], fingertips_object[i].reshape(-1, 3)
        )
        contact_points_object[i] = closest.reshape(fingertips_object[i].shape)
        distance[i] = distances.reshape(distance[i].shape)
    contact_mask = distance < CONTACT_START_DISTANCE
    contact_mask &= np.arange(num_frames)[None, :, None] < lengths[:, None, None]

    # --- initial solve: batched contact tracking in the fixed robot-base frame ---
    q_wxyz_base_world = quaternion_invert(T_WORLD_BASE[3:])
    keypoints_base = quaternion_apply(q_wxyz_base_world, keypoints - T_WORLD_BASE[:3]).astype(np.float32)
    contact_points_world = object_pos[:, :, None] + quaternion_apply(
        object_quat_wxyz[:, :, None], contact_points_object
    )
    contact_points_base = quaternion_apply(q_wxyz_base_world, contact_points_world - T_WORLD_BASE[:3]).astype(
        np.float32
    )
    direct.warmup(batch_size, num_frames)
    direct.set_valid_lengths(lengths)
    packed = direct.solve_with_contact_numpy(keypoints_base, contact_points_base, contact_mask)

    # --- refinement: one static object mesh per batch item ---
    T_object_base = np.empty((batch_size, num_frames, 7), dtype=np.float32)
    T_object_base[..., :3] = quaternion_apply(q_wxyz_object_world, T_WORLD_BASE[:3] - object_pos)
    T_object_base[..., 3:] = quaternion_multiply(q_wxyz_object_world, T_WORLD_BASE[3:])
    init_state = robot.state(
        q=wp.from_numpy(np.ascontiguousarray(packed[..., 7:]), dtype=wp.float32, device=device),
        T_world_base=wp.from_numpy(T_object_base, dtype=wp_vec7, device=device),
    )
    robot.forward_kinematics(init_state)
    T_object_link = init_state.T_world_link.numpy()[:, :, contact_link_indices]
    contact_outward = fingertips_object - contact_points_object
    contact_outward /= np.maximum(np.linalg.norm(contact_outward, axis=-1, keepdims=True), 1e-8)
    contact_inward_local = quaternion_apply(quaternion_invert(T_object_link[..., 3:]), -contact_outward)
    local_contact_points = np.empty_like(contact_points_object)
    for contact_index, link_index in enumerate(contact_link_indices):
        link_mesh = robot.spec.link_visual_geometries[robot.spec.link_names[link_index]].to_mesh()
        vertices = np.asarray(link_mesh.vertices, np.float32)
        local_contact_points[:, :, contact_index] = vertices[
            np.argmax(contact_inward_local[:, :, contact_index] @ vertices.T, axis=-1)
        ]

    contact_points_robot = T_object_link[..., :3] + quaternion_apply(T_object_link[..., 3:], local_contact_points)
    contact_strength = np.clip(
        (CONTACT_START_DISTANCE - distance) / (CONTACT_START_DISTANCE - CONTACT_FULL_DISTANCE), 0.0, 1.0
    )
    contact_strength = contact_strength * contact_strength * (3.0 - 2.0 * contact_strength)
    contact_points_target = contact_points_object - CONTACT_EMBED_DEPTH * contact_outward
    contact_points_target = contact_points_robot + contact_strength[..., None] * (
        contact_points_target - contact_points_robot
    )
    refine.warmup(batch_size, num_frames)
    refine.set_valid_lengths(lengths)
    packed = refine.solve_with_contact_numpy(
        keypoints_object,
        contact_points_target,
        contact_mask,
        init_state=init_state,
        local_contact_points=local_contact_points,
    )
    T_object_base = packed[..., :7].copy()
    packed[..., :3] = object_pos + quaternion_apply(object_quat_wxyz, T_object_base[..., :3])
    packed[..., 3:7] = quaternion_multiply(object_quat_wxyz, T_object_base[..., 3:])
    return packed, contact_mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/dexycb"), help="NPZ output directory (default: %(default)s)"
    )
    parser.add_argument("--batch-size", type=int, default=64, help="captures per solve")
    parser.add_argument("--limit", type=int, default=0, help="first N captures; zero processes all")
    viewer = parser.add_mutually_exclusive_group()
    viewer.add_argument("--no-viz", action="store_true", help="save results and exit without playback")
    viewer.add_argument("--viz-only", action="store_true", help="browse saved results without solving")
    args = parser.parse_args()
    if args.batch_size < 1 or args.limit < 0:
        parser.error("--batch-size must be positive and --limit must be nonnegative")
    if args.viz_only:
        _view_results(args.output_dir)
        return

    from robokit.assets.datasets import dexycb
    from robokit.assets.robots.assembly import xarm7_xhand

    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    robot = Robot.load(
        str(xarm7_xhand.URDF_PATH),
        load_meshes=True,
        load_collision_spheres=True,
        collision_spheres_path=xarm7_xhand.COLLISION_SPHERE_PATH,
    )
    dataset = DexYCBVideoDataset(dexycb.DIR, is_right=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pending: List[Tuple[int, int]] = []
    count = min(args.limit, len(dataset)) if args.limit else len(dataset)
    for index in range(count):
        sequence = dataset[index]
        if (args.output_dir / f"{sequence['capture_name']}.npz").exists():
            continue
        num_frames = int(np.count_nonzero(np.linalg.norm(sequence["hand_pose"][:, 0, :], axis=1) > 1e-5))
        if num_frames >= 3:
            pending.append((num_frames, index))

    # --- solve: sort by length to limit padding; reuse both solvers and swap scene meshes ---
    pending.sort()
    completed = 0
    print(f"{len(pending)}/{count} right-hand captures pending")
    if not pending:
        if not args.no_viz:
            _view_results(args.output_dir, robot, dataset)
        return
    # A few batch slots need more iterations to reach the single-capture cost range.
    solver = replace(offline.solver, max_iter=35)
    direct_config = contact_config(replace(offline, solver=solver))
    direct_config.task("collision").weight = 0.0
    direct = HandRetargeterOffline(direct_config, robot, spec, device=device)
    root_link_index = robot.spec.link_names.index(spec.target_link_names[spec.root_target_name])
    arm_joint_mask = robot.spec.link_ancestor_joints_mask[root_link_index]
    arm_dof_mask = np.any(robot.spec.joints_to_actuated_mapping[arm_joint_mask] != 0.0, axis=0)
    hand_joint_names = tuple(np.asarray(robot.spec.actuated_joint_names)[~arm_dof_mask])
    refinement = refine_config(
        replace(
            offline,
            optimize_base=False,
            active_joint_names=hand_joint_names,
            locked_prefix_frames=2,
            solver=solver,
        )
    )
    refinement.task("contact").weight = 300.0
    refinement.task("contact").contact_margin = 0.0
    refinement.task("anchor").dense_task.weight = 10.0
    refinement.task("acceleration").weight = 20.0
    refinement.task("collision").weight = 100.0
    refinement.task("collision").margin = 0.0
    object_meshes: Dict[Path, trimesh.Trimesh] = {}
    meshes: Dict[Path, wp.Mesh] = {}
    mesh_geom: Optional[MeshGeom] = None
    refine: Optional[HandRetargeterOffline] = None
    active_meshes: List[wp.Mesh] = []
    for start in range(0, len(pending), args.batch_size):
        batch_indices = [index for _, index in pending[start : start + args.batch_size]]
        captures = [
            load_dexycb_grasp(index, device=device, dataset=dataset, mesh_cache=object_meshes)
            for index in batch_indices
        ]
        captures.extend([captures[-1]] * (args.batch_size - len(captures)))
        batch_meshes = []
        for data in captures:
            path = data["object_mesh_path"]
            if path not in meshes:
                mesh = data["object_mesh"]
                meshes[path] = wp.Mesh(
                    points=wp.array(mesh.vertices.view(np.ndarray), dtype=wp.vec3, device=device),
                    indices=wp.array(mesh.faces.view(np.ndarray).ravel(), dtype=int, device=device),
                )
            batch_meshes.append(meshes[path])
        if refine is None:
            identity = wp.from_numpy(
                np.repeat(np.eye(4, dtype=np.float32)[None], args.batch_size, axis=0),
                dtype=wp.mat44,
                device=device,
            )
            mesh_geom = MeshGeom(batch_meshes, np.arange(args.batch_size + 1, dtype=np.int32), poses=identity)
            scene = WarpScene(args.batch_size, device).add(mesh_geom)
            scene_indices = wp.from_numpy(np.arange(args.batch_size, dtype=np.int32), dtype=wp.int32, device=device)
            refine = HandRetargeterOffline(
                refinement, robot, spec, scene=scene, device=device, scene_indices=scene_indices
            )
        else:
            assert mesh_geom is not None
            if any(current is not upcoming for current, upcoming in zip(active_meshes, batch_meshes)):
                mesh_geom.update(meshes=batch_meshes)
        active_meshes = batch_meshes
        began = time.perf_counter()
        packed, contact_mask = _solve_batch(captures, robot, device, direct, refine)
        for i, data in enumerate(captures[: len(batch_indices)]):
            output = args.output_dir / f"{data['capture_name']}.npz"
            temporary = output.with_suffix(".tmp.npz")
            length = len(data["keypoints"])
            np.savez_compressed(
                temporary,
                qpos=packed[i, :length],
                frame_indices=data["frame_indices"],
                contact_mask=contact_mask[i, :length],
                object_pos=data["object_pos"],
                object_quat_wxyz=data["object_quat_wxyz"],
            )
            temporary.replace(output)
        completed += len(batch_indices)
        print(f"saved {completed}/{len(pending)} captures in {time.perf_counter() - began:.1f}s")
    if not args.no_viz:
        _view_results(args.output_dir, robot, dataset)


if __name__ == "__main__":
    main()
