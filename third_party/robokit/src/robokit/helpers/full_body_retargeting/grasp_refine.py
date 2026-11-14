"""Stable grasps from captured hand contacts."""

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
import warp as wp

from robokit.geom import MeshGeom, WarpScene
from robokit.helpers.full_body_retargeting.clip import FullBodyClip, HandTrack
from robokit.helpers.full_body_retargeting.config import HandTrajectory
from robokit.helpers.full_body_retargeting.retargeting import FullBodyRetargeter
from robokit.opt.lm_optimizer import LMOptimizer, LMOptimizerConfig
from robokit.opt.var_values import VarKey, VarValues
from robokit.robo import Robot
from robokit.robo.robot_state import RobotState
from robokit.terms.dense.configuration_task import ConfigurationTask
from robokit.terms.dense.force_closure_task import ForceClosureTask
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.scene_collision_task import SceneCollisionTask
from robokit.terms.dense.scene_distance_task import SceneDistanceTask
from robokit.utils.warp_utils import wp_vec7
from robokit.xform.numpy import (
    axis_angle_to_matrix,
    inverse_tf_mat,
    matrix_to_axis_angle,
    pose7_to_tf_mat,
    tf_mat_to_pose7,
)


# one link per contact slot in CONTACT_MANO_INDICES order; the thumb pairs intermediate+distal
_CONTACT_LINK_NAMES = (
    "thumb_intermediate",
    "thumb_distal",
    *(f"{finger}_intermediate" for finger in ("index", "middle", "ring", "pinky") for _ in range(2)),
)
_MIN_RUN_FRAMES = 3  # shortest any-contact span worth a grasp
_MAX_WRIST_SHIFT = 0.03  # m, how far an optimized base may move from the captured pose
_MAX_WRIST_TURN = np.deg2rad(20.0)
_MAX_PENETRATION = 0.008  # m, deepest allowed collision-sphere penetration into the object


@dataclass
class GraspRefineResult:
    """Whole-body result and the grasp markers used by the viewer."""

    qpos: np.ndarray  # (frames, 7 + dofs + 7 * num_objects)
    targets_world: Dict[str, np.ndarray]  # (frames, slots, 3) object-surface targets per side
    contacts_world: Dict[str, np.ndarray]  # (frames, slots, 3) achieved hand pad points per side
    contact_masks: Dict[str, np.ndarray]  # (frames, slots) True while a grasp is fully held


@dataclass
class _Hand:
    track: HandTrack
    robot: Robot
    scene: Optional[WarpScene]
    q: np.ndarray
    T_object_base: np.ndarray
    contact_links: np.ndarray


def _apply_se3(T_dst_src: np.ndarray, points_src: np.ndarray) -> np.ndarray:
    return np.einsum("...ij,...j->...i", T_dst_src[..., :3, :3], points_src) + T_dst_src[..., :3, 3]


def _contact_runs(mask: np.ndarray) -> List[Tuple[int, int, np.ndarray]]:
    """Sustained any-contact spans of a `(frames, slots)` mask as `(start, stop, slots)`."""
    edges = np.flatnonzero(np.diff(np.r_[False, np.asarray(mask.any(axis=1)), False])).reshape(-1, 2)
    runs = []
    for start, stop in edges:
        if stop - start < _MIN_RUN_FRAMES:
            continue
        counts = mask[start:stop].sum(axis=0).reshape(5, 2)  # slots pair up per finger
        fingers = np.flatnonzero(counts.max(axis=1))
        if 0 not in fingers:  # a grasp opposes the thumb, so always include it
            fingers = np.r_[0, fingers]
        if len(fingers) > 1:
            runs.append((int(start), int(stop), 2 * fingers + counts[fingers].argmax(axis=1)))
    return runs


@dataclass
class _GraspBatch:
    geom: MeshGeom
    solver: LMOptimizer
    values: VarValues
    pads: wp.array
    links: wp.array
    anchor: ConfigurationTask


def _refine_grasps(
    runs: List[Tuple[_Hand, int, int, np.ndarray]], device: str, cache: Dict[Tuple, _GraspBatch]
) -> List[Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Solve compatible static grasps together, one object scene per run and unchanged candidate enumeration."""
    robot = runs[0][0].robot
    meshes = robot.spec.get_link_meshes(mode="collision")
    num_slots = len(runs[0][3])
    num_runs = len(runs)
    pad_batches, link_batches, anchors, initial_q, initial_base = [], [], [], [], []
    for hand, start, stop, slots in runs:
        assert hand.scene is not None
        key = (start + stop) // 2
        links = hand.contact_links[slots]
        T_object_link = robot.forward_kinematics_via_matrix_numpy(
            hand.q[key : key + 1], hand.T_object_base[key : key + 1], device
        )[0, links]
        num_options = 2 if len(slots) <= 3 else 4
        options = []
        for link, T_object_link_i in zip(links, T_object_link):
            mesh = meshes[robot.spec.link_names[link]]
            local = np.asarray(mesh.vertices, dtype=np.float32)
            world = local @ T_object_link_i[:3, :3].T + T_object_link_i[:3, 3]
            signed_dists, normal, _ = hand.scene.query_sdf(
                wp.from_numpy(world, dtype=wp.vec3, device=device),
                scene_offsets=wp.array([0, len(world)], dtype=wp.int32, device=device),
            )
            alignment = np.einsum(
                "ij,ij->i", np.asarray(mesh.vertex_normals) @ T_object_link_i[:3, :3].T, normal.numpy()
            )
            facing = np.flatnonzero(alignment < 0.0)  # vertex normal points against the object surface
            facing = facing if len(facing) >= num_options else np.argsort(alignment)[:num_options]
            options.append(local[facing[np.argsort(np.abs(signed_dists.numpy()[facing]))[:num_options]]])

        rows = product(*options) if len(slots) <= 3 else zip(*options)
        pads = np.repeat(np.asarray(list(rows), dtype=np.float32), 2, axis=0)
        link_rows = np.repeat(links[None], len(pads), axis=0)
        q_anchor = np.repeat(hand.q[key : key + 1], len(pads), axis=0)
        q_init = q_anchor.copy()
        q_init[1::2] = 0.5 * (q_init[1::2] + np.maximum(robot.spec.actuated_joint_limits[:, 0], 0.0))
        T_object_base_init = np.repeat(tf_mat_to_pose7(hand.T_object_base[key : key + 1]), len(pads), axis=0).astype(
            np.float32
        )
        pad_batches.append(pads)
        link_batches.append(link_rows)
        anchors.append(q_anchor)
        initial_q.append(q_init)
        initial_base.append(T_object_base_init)

    pads = np.concatenate(pad_batches)
    link_rows = np.concatenate(link_batches)
    q_anchor = np.concatenate(anchors)
    q_init = np.concatenate(initial_q)
    T_object_base_init = np.concatenate(initial_base)
    meshes = [hand.track.object_mesh for hand, _, _, _ in runs if hand.track.object_mesh is not None]
    cache_key = (id(robot), num_slots)
    batch = cache.get(cache_key)
    if batch is None or batch.geom.num_scenes != num_runs:
        geom = MeshGeom(meshes, np.arange(num_runs + 1, dtype=np.int32))
        scene = WarpScene(num_runs, device).add(geom)
        pads_wp = wp.from_numpy(pads, dtype=wp.vec3, device=device)
        links_wp = wp.from_numpy(link_rows, dtype=wp.int32, device=device)
        base_wp = wp.from_numpy(T_object_base_init, dtype=wp_vec7, device=device)
        state = robot.state(q=wp.from_numpy(q_init, dtype=wp.float32, device=device), T_world_base=base_wp)
        state.base_step_scale = 3e-4
        # the thumb carries as much force weight as all other slots together
        force_weights = (1.0, *((1.0 / (num_slots - 1),) * (num_slots - 1)))
        # the pads are enumerated above, so they are pinned rather than chosen by the solver
        distance = SceneDistanceTask(
            robot=robot,
            warp_meshes=scene,
            num_contact_points=num_slots,
            local_contact_points=pads_wp,
            contact_points_link_indices=links_wp,
            weight=600.0,
            residual_mode="abs",
        )
        anchor_task = ConfigurationTask(
            robot,
            rest_q=q_anchor,
            T_world_base_rest=base_wp,
            weight=0.5,
            base_weight=(600.0, 600.0, 600.0, 80.0, 80.0, 80.0),
        )
        terms = [
            distance,
            ForceClosureTask(
                robot=robot,
                warp_meshes=scene,
                num_contact_points=num_slots,
                local_contact_points=pads_wp,
                contact_points_link_indices=links_wp,
                contact_force_weights=force_weights,
                weight=20.0,
                distance_task=distance,  # one contact query serves both terms
            ),
            PositionLimit(robot=robot, weight=10.0, residual_mode="abs"),
            SceneCollisionTask(
                robot=robot,
                scene=scene,
                scene_indices=wp.array(np.arange(num_runs), dtype=wp.int32, device=device),
                weight=150.0,
                margin=0.0,
                residual_mode="abs",
            ),
            anchor_task,
        ]
        solver = LMOptimizer(terms, LMOptimizerConfig(max_iter=48, lm_lambda=2.0), device)
        batch = _GraspBatch(geom, solver, VarValues({VarKey.ROBOT: state}), pads_wp, links_wp, anchor_task)
        cache[cache_key] = batch
    else:
        batch.geom.update(meshes=meshes)
        wp.copy(batch.pads, wp.from_numpy(pads, dtype=wp.vec3, device=device))
        wp.copy(batch.links, wp.from_numpy(link_rows, dtype=wp.int32, device=device))
        state = cast(RobotState, batch.values.get(VarKey.ROBOT))
        wp.copy(state.q, wp.from_numpy(q_init, dtype=wp.float32, device=device))
        wp.copy(state.T_world_base, wp.from_numpy(T_object_base_init, dtype=wp_vec7, device=device))
        state.invalidate()
        batch.anchor.set_target(
            rest_q=wp.from_numpy(q_anchor, dtype=wp.float32, device=device), T_world_base_rest=state.T_world_base
        )
        scene = cast(SceneDistanceTask, batch.solver.terms[0]).warp_meshes
    optimized = cast(RobotState, batch.solver.solve(batch.values)[0].get(VarKey.ROBOT))
    q = np.clip(
        optimized.q.numpy(), robot.spec.actuated_joint_limits[:, 0], robot.spec.actuated_joint_limits[:, 1]
    ).astype(np.float32)
    T_object_base = optimized.T_world_base.numpy()
    T_object_base_mat = pose7_to_tf_mat(T_object_base)
    T_pad_link = robot.forward_kinematics_via_matrix_numpy(q, T_object_base_mat, device)[
        np.arange(len(pads))[:, None], link_rows
    ]
    points = np.einsum("bcij,bcj->bci", T_pad_link[..., :3, :3], pads) + T_pad_link[..., :3, 3]
    signed_dists, normals, _ = scene.query_sdf(
        wp.from_numpy(points.reshape(-1, 3).astype(np.float32), dtype=wp.vec3, device=device),
        scene_offsets=wp.array(
            np.arange(num_runs + 1) * (points.size // (3 * num_runs)), dtype=wp.int32, device=device
        ),
    )
    pad_normals = normals.numpy().reshape(len(pads), num_slots, 3)
    finger_normal = pad_normals[:, 1:].mean(axis=1)
    finger_normal /= np.maximum(np.linalg.norm(finger_normal, axis=1, keepdims=True), 1e-8)
    opposition = np.einsum("bi,bi->b", pad_normals[:, 0], finger_normal)  # thumb vs finger normals
    shift = np.linalg.norm(T_object_base[:, :3] - T_object_base_init[:, :3], axis=1)
    turn = 2.0 * np.arccos(
        np.clip(np.abs(np.einsum("bi,bi->b", T_object_base[:, 3:], T_object_base_init[:, 3:])), 0.0, 1.0)
    )
    sphere_links = robot.spec.collision_spheres_link_indices
    T_sphere_link = robot.forward_kinematics_via_matrix_numpy(q, T_object_base_mat, device)[:, sphere_links]
    centers = (
        np.einsum("bsij,sj->bsi", T_sphere_link[..., :3, :3], robot.spec.local_collision_sphere_centers)
        + T_sphere_link[..., :3, 3]
    )
    collision_signed_dists, _, _ = scene.query_sdf(
        wp.from_numpy(centers.reshape(-1, 3).astype(np.float32), dtype=wp.vec3, device=device),
        scene_offsets=wp.array(
            np.arange(num_runs + 1) * (centers.size // (3 * num_runs)), dtype=wp.int32, device=device
        ),
    )
    sphere_signed_dists = collision_signed_dists.numpy().reshape(len(pads), -1) - robot.spec.collision_sphere_radii
    penetration = np.maximum(-sphere_signed_dists, 0.0).max(axis=1)
    surface = np.abs(signed_dists.numpy()).reshape(len(pads), num_slots).mean(axis=1)
    results = []
    num_candidates = len(pads) // num_runs
    for offset in range(0, len(pads), num_candidates):
        rows = slice(offset, offset + num_candidates)
        valid = np.flatnonzero(
            (shift[rows] < _MAX_WRIST_SHIFT) & (turn[rows] < _MAX_WRIST_TURN) & (penetration[rows] < _MAX_PENETRATION)
        )
        if not len(valid):
            results.append(None)
            continue
        # among the half closest to the surface, prefer the most opposed thumb-vs-finger normals
        opposed = valid[opposition[rows][valid] < 0.0]
        candidates = opposed if len(opposed) else valid
        close = candidates[surface[rows][candidates] <= np.median(surface[rows][candidates])]
        best = int(close[np.argmin(opposition[rows][close])])
        best += offset
        results.append((q[best], T_object_base_mat[best], pads[best]))

    return results


def _hold_grasps_batch(
    hands: List[_Hand], fps: List[int], device: str, cache: Dict[Tuple, _GraspBatch]
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Blend each hand's runs in order; batch the next independent run from every hand."""
    markers = []
    runs = []
    free = []
    for hand in hands:
        contacts = np.zeros((len(hand.q), len(hand.contact_links), 3), dtype=np.float32)
        markers.append((np.zeros_like(contacts), contacts, np.zeros(contacts.shape[:2], dtype=bool)))
        runs.append([] if hand.scene is None else _contact_runs(hand.track.contact_mask[: len(hand.q)]))
        free.append((hand.q.copy(), hand.T_object_base.copy()))
    for run_index in range(max((len(row) for row in runs), default=0)):
        groups: Dict[Tuple, List[int]] = {}
        for i, row in enumerate(runs):
            if run_index < len(row):
                groups.setdefault((id(hands[i].robot), len(row[run_index][2])), []).append(i)
        for indices in groups.values():
            grasps = _refine_grasps([(hands[i], *runs[i][run_index]) for i in indices], device, cache)
            for i, grasp in zip(indices, grasps):
                hand = hands[i]
                start, stop, slots = runs[i][run_index]
                num_frames = len(hand.q)
                window = max(3, round(fps[i] * 0.3))
                q_free, T_free = free[i]
                targets, contacts, mask = markers[i]
                if grasp is None:
                    continue
                q_grasp, T_grasp, points = grasp
                # smoothstep toward the grasp, extended one window past the run for the release
                frames = np.arange(start, min(stop + window + 1, num_frames))
                alpha = np.minimum(
                    np.clip((frames - start) / window, 0.0, 1.0),
                    np.clip((stop + window - frames) / window, 0.0, 1.0),
                )
                alpha = alpha * alpha * (3.0 - 2.0 * alpha)
                alpha[alpha > 0.95] = 1.0  # snap the plateau so held frames match the grasp exactly
                hand.q[frames] = (1.0 - alpha[:, None]) * q_free[frames] + alpha[:, None] * q_grasp

                delta = T_free[frames, :3, :3].transpose(0, 2, 1) @ T_grasp[:3, :3]
                hand.T_object_base[frames, :3, :3] = T_free[frames, :3, :3] @ axis_angle_to_matrix(
                    alpha[:, None] * matrix_to_axis_angle(delta)
                )
                hand.T_object_base[frames, :3, 3] = (1.0 - alpha[:, None]) * T_free[frames, :3, 3] + alpha[
                    :, None
                ] * T_grasp[:3, 3]

                T_pad_link = hand.robot.forward_kinematics_via_matrix_numpy(
                    hand.q[frames], hand.T_object_base[frames], device
                )[:, hand.contact_links[slots]]
                pads_object = np.einsum("fcij,cj->fci", T_pad_link[..., :3, :3], points) + T_pad_link[..., :3, 3]
                _, _, closest = hand.scene.query_sdf(
                    wp.from_numpy(pads_object.reshape(-1, 3).astype(np.float32), dtype=wp.vec3, device=device),
                    scene_offsets=wp.array([0, pads_object.size // 3], dtype=wp.int32, device=device),
                )
                contacts[frames[:, None], slots[None]] = pads_object
                targets[frames[:, None], slots[None]] = closest.numpy().reshape(pads_object.shape)
                mask[frames[:, None], slots[None]] = (alpha == 1.0)[:, None]
    return markers


def solve_grasp_refine(
    clip: FullBodyClip,
    retargeter: FullBodyRetargeter,
    *,
    max_frames: int = 0,
    scene_clearance: bool = False,
) -> GraspRefineResult:
    """Retarget the captured hands and hold one refined grasp per sustained contact run.

    Args:
        clip: Loaded interaction clip.
        retargeter: Shared retargeter; its standalone hands run the hand pass.
        max_frames: Frame cap; 0 keeps all frames.
        scene_clearance: Push the solved body out of penetrated scene parts.

    Returns:
        Whole-body qpos and world-frame grasp markers per side.
    """
    return solve_grasp_refine_batch([clip], retargeter, max_frames=max_frames, scene_clearance=scene_clearance)[0]


def solve_grasp_refine_batch(
    clips: List[FullBodyClip],
    retargeter: FullBodyRetargeter,
    *,
    max_frames: int = 0,
    scene_clearance: bool = False,
) -> List[GraspRefineResult]:
    """Batch 01's passes with unchanged weights; clips must share hand sides but may differ in length and object mesh.

    Example:
        >>> # xdoctest: +REQUIRES(env:ROBOKIT_PARAHOME_DOCS==1)
        >>> from robokit.assets.datasets import parahome
        >>> from robokit.helpers.full_body_retargeting.parahome_utils import find_parahome_intervals, load_parahome_clip
        >>> root = str(parahome.data_root())
        >>> clip = load_parahome_clip(root, find_parahome_intervals(root, "s1")[2])
        >>> results = solve_grasp_refine_batch([clip, clip], FullBodyRetargeter(), max_frames=5)
        >>> len(results), len(results[0].qpos)
        (2, 5)
    """
    if not clips or any(clip.sides != clips[0].sides for clip in clips):
        raise ValueError("clips must have the same hand sides and a positive batch size")
    lengths = [clip.num_frames if max_frames == 0 else min(max_frames, clip.num_frames) for clip in clips]
    if min(lengths) < 3:
        raise ValueError("at least 3 frames are required")
    device = retargeter.device
    hands: List[Dict[str, _Hand]] = [{} for _ in clips]
    for side in clips[0].sides:
        robot = retargeter.hand_robot(side)
        packed = retargeter.solve_hand_pass_batch(
            side, [clip.hands[side] for clip in clips], lengths, retargeter.config.grasp_hand_pass
        )
        for i, (clip, length, row) in enumerate(zip(clips, lengths, packed)):
            track = clip.hands[side]
            hands[i][side] = _Hand(
                track=track,
                robot=robot,
                scene=None
                if track.object_mesh is None
                else WarpScene(1, device).add(MeshGeom([track.object_mesh], np.asarray([0, 1], dtype=np.int32))),
                q=row[:, 7:].copy(),
                T_object_base=inverse_tf_mat(pose7_to_tf_mat(track.T_native_object[:length]))
                @ pose7_to_tf_mat(row[:, :7]),
                contact_links=np.asarray(
                    [robot.spec.link_names.index(name) for name in _CONTACT_LINK_NAMES], dtype=np.int32
                ),
            )
    flat_hands = [hand for row in hands for hand in row.values()]
    markers = _hold_grasps_batch(
        flat_hands, [clip.fps for clip in clips for _ in clip.sides], device, retargeter._grasp_batches
    )
    trajectories = []
    for row, length in zip(hands, lengths):
        trajectories.append(
            {
                side: HandTrajectory(
                    q=hand.q,
                    T_world_base=tf_mat_to_pose7(
                        pose7_to_tf_mat(hand.track.T_world_object[:length]) @ hand.T_object_base
                    ),
                )
                for side, hand in row.items()
            }
        )
    qpos = retargeter.solve_batch(clips, trajectories, valid_lengths=lengths, scene_clearance=scene_clearance)
    results = []
    offset = 0
    for row, length, packed in zip(hands, lengths, qpos):
        targets, contacts, masks = {}, {}, {}
        for side, hand in row.items():
            T_world_object = pose7_to_tf_mat(hand.track.T_world_object[:length])[:, None]
            target, contact, mask = markers[offset]
            targets[side] = _apply_se3(T_world_object, target)
            contacts[side] = _apply_se3(T_world_object, contact)
            masks[side] = mask
            offset += 1
        results.append(GraspRefineResult(packed, targets, contacts, masks))
    return results


__all__ = ["GraspRefineResult", "solve_grasp_refine", "solve_grasp_refine_batch"]
