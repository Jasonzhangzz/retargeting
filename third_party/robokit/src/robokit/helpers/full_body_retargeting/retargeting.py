"""Full-body driver: `FullBodyClip` -> qpos `[T, 7 + dof + 7 * num_objects]`; every stage runs per engaged hand, so one- and two-handed clips take the same path:

1. hand pass - captured MANO keypoints onto the standalone hand, in the native world;
2. body pass - whole-body tracking with each grasp wrist injected as a high-weight target track;
3. arm snap - an arm-only IK puts each palm exactly at the hand-pass pose;
4. contact refine - fingers re-solve at the achieved wrist against that hand's object, with sphere-vs-mesh collision. Stages 2-4 iterate twice so the body absorbs the wrist correction.
"""

from dataclasses import replace
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch
import warp as wp
from scipy.ndimage import uniform_filter1d

from robokit.geom import MeshGeom, WarpScene
from robokit.helpers.full_body_retargeting.clip import CONTACT_MANO_INDICES, FullBodyClip, HandTrack
from robokit.helpers.full_body_retargeting.config import (
    _HAND_BASE_LINK,
    _ORIENTATION_LINKS,
    FullBodyRetargetingConfig,
    HandTrajectory,
    _HandModel,
)
from robokit.helpers.full_body_retargeting.scene_clearance import apply_scene_clearance
from robokit.helpers.hand_retargeting import HandRetargeterOffline, HandRetargeterOfflineConfig
from robokit.helpers.hand_retargeting.presets import TARGET_NAMES
from robokit.helpers.hand_retargeting.presets import inspire as inspire_preset
from robokit.helpers.humanoid_retargeting.offline import HumanoidRetargeterOffline
from robokit.helpers.ik import IKSolver
from robokit.robo import Robot
from robokit.smplx import SMPLX_JOINT_NAMES
from robokit.utils.ee_coord_utils import HandCoordinateSpec
from robokit.utils.warp_utils import wp_vec7
from robokit.xform.numpy import inverse_tf_mat, normalize_quaternion, pose7_to_tf_mat, tf_mat_to_pose7


if TYPE_CHECKING:
    from robokit.helpers.full_body_retargeting.grasp_refine import _GraspBatch


def _pad_tracks(tracks: List[np.ndarray]) -> np.ndarray:
    num_frames = max(len(track) for track in tracks)
    return np.stack(
        [np.pad(track, ((0, num_frames - len(track)),) + ((0, 0),) * (track.ndim - 1), mode="edge") for track in tracks]
    ).astype(np.float32)


class FullBodyRetargeter:
    """Whole-body pass + captured-finger hand retargeting with an optional contact refine; see module docstring."""

    def __init__(
        self,
        config: Optional[FullBodyRetargetingConfig] = None,
        device: str = "cuda:0",
        load_meshes: bool = False,
    ):
        from robokit.helpers.humanoid_retargeting.presets.g1_inspire_smplh import g1_inspire_smplh
        from robokit.helpers.humanoid_retargeting.presets.g1_offline_mapping import (
            g1_offline_mapping as kin_g1_offline_mapping,
        )

        self.config = config or FullBodyRetargetingConfig()
        self.device = device
        self.body_robot = g1_inspire_smplh.load_robot(load_meshes=load_meshes)

        self._hands = self._load_hands(self.body_robot, device)
        self._arm_ik: Optional[IKSolver] = None  # built for a (sides, frames) shape; rebuilt on change
        self._arm_ik_key: Tuple = ()
        self._grasp_batches: Dict[Tuple, "_GraspBatch"] = {}
        self._hand_passes: Dict[Tuple, HandRetargeterOffline] = {}
        self._refiners: Dict[Tuple[str, bool], Tuple[object, int, HandRetargeterOffline]] = {}
        self._contact_target_names = tuple(TARGET_NAMES[m] for m in CONTACT_MANO_INDICES)

        self._kin_robot = kin_g1_offline_mapping.load_robot()
        kin_names = self._kin_robot.spec.actuated_joint_names
        self._kin = HumanoidRetargeterOffline(
            kin_g1_offline_mapping, self.config.body, robot=self._kin_robot, device=device
        )
        self._human_joint_columns = [SMPLX_JOINT_NAMES.index(n) for n in self._kin.human_joint_names]
        body_joint_names = self.body_robot.spec.actuated_joint_names
        self._kin_joint_columns = np.array([body_joint_names.index(n) for n in kin_names])

    def solve(
        self,
        clip: FullBodyClip,
        /,
        max_frames: int = 0,
        *,
        hand_trajectories: Optional[Dict[str, HandTrajectory]] = None,
        scene_clearance: bool = False,
    ) -> np.ndarray:
        """Retarget one clip.

        Args:
            clip: Loaded interaction clip.
            max_frames: Frame cap; 0 keeps all frames.
            hand_trajectories: Optional externally solved standalone-hand trajectories per side; each replaces that side's hand pass and skips its contact refine.
            scene_clearance: Push the solved body out of penetrated scene parts.

        Returns:
            Packed `(frames, 7 + dofs + 7 * num_objects)` base pose, joints, and object poses.
        """
        num_frames = clip.num_frames if max_frames == 0 else min(max_frames, clip.num_frames)
        sides = [side for side in clip.sides if side in self._hands]
        if not sides:
            raise ValueError(f"clip has no supported hands: {clip.sides}")
        hand_trajectories = hand_trajectories or {}
        if not set(hand_trajectories) <= set(sides):
            raise ValueError(f"unknown hand trajectories: {set(hand_trajectories) - set(sides)}")
        object_sides = [side for side in sides if clip.hands[side].object_mesh is not None]
        if not object_sides:
            raise ValueError(f"no grasped object among hands {sides}")
        if set(hand_trajectories) == set(sides):
            return self.solve_batch(
                [clip], [hand_trajectories], valid_lengths=[num_frames], scene_clearance=scene_clearance
            )[0]

        finger_q: Dict[str, np.ndarray] = {}
        T_world_hand_link: Dict[str, np.ndarray] = {}
        for side in sides:
            hand = self._hands[side]
            track = clip.hands[side]
            if side in hand_trajectories:
                trajectory = hand_trajectories[side]
                if trajectory.q.shape[0] < num_frames or trajectory.q.shape[1:] != (
                    hand.robot.spec.num_actuated_joints,
                ):
                    raise ValueError(
                        f"{side} trajectory q must be (>= {num_frames}, {hand.robot.spec.num_actuated_joints})"
                    )
                if trajectory.T_world_base.shape[0] < num_frames or trajectory.T_world_base.shape[1:] != (7,):
                    raise ValueError(f"{side} trajectory T_world_base must be (>= {num_frames}, 7)")
                finger_q[side] = trajectory.q[:num_frames].astype(np.float32)
                T_world_hand_base = pose7_to_tf_mat(trajectory.T_world_base[:num_frames]).astype(np.float32)
                if num_frames >= 3:
                    finger_q[side] = uniform_filter1d(finger_q[side], self.config.smooth_window, axis=0, mode="nearest")
                    T_world_object = pose7_to_tf_mat(track.T_world_object[:num_frames])
                    T_object_base = self._smooth_pose7(
                        tf_mat_to_pose7(inverse_tf_mat(T_world_object) @ T_world_hand_base), self.config.smooth_window
                    )
                    T_world_hand_base = T_world_object @ pose7_to_tf_mat(T_object_base)
            else:
                packed = self.solve_hand_pass(side, track, num_frames)
                finger_q[side] = packed[:, 7:]  # (T, hand_dof) standalone actuated order
                # native -> body world through this hand's object: shared rotation, scaled translation
                T_world_native = pose7_to_tf_mat(track.T_world_object[:num_frames]) @ inverse_tf_mat(
                    pose7_to_tf_mat(track.T_native_object[:num_frames])
                )
                T_world_hand_base = (T_world_native @ pose7_to_tf_mat(packed[:, :7])).astype(np.float32)
            T_world_hand_link[side] = hand.robot.forward_kinematics_via_matrix_numpy(
                finger_q[side], T_world_hand_base, self.device
            )

        # pass 1 solves with a soft base, pass 2 re-welds the arms to it and freezes the base
        finger_final = dict(finger_q)
        refine_sides = [
            side
            for side in sides
            if side not in hand_trajectories and bool(clip.hands[side].contact_mask[:num_frames].any())
        ]
        qpos = np.zeros((num_frames, 0), dtype=np.float32)
        num_dofs = self.body_robot.spec.num_actuated_joints
        for iteration in range(2 if refine_sides else 1):
            qpos = self._solve_body_batch([clip], [T_world_hand_link], [num_frames], bool(hand_trajectories))[0]
            T_world_base = pose7_to_tf_mat(qpos[:, :7])
            final_iteration = iteration == 1
            if refine_sides:  # one FK of the achieved body serves both refine sides
                T_world_body_link = self.body_robot.forward_kinematics_via_matrix_numpy(
                    qpos[:, 7 : 7 + num_dofs],
                    T_world_base,  # the snap leaves the base as-is
                    device=self.device,
                )
            for side in refine_sides:
                track = clip.hands[side]
                refined_q, T_object_base = self._refine(
                    clip,
                    side,
                    T_world_body_link=T_world_body_link,
                    finger_q=finger_q[side],
                    num_frames=num_frames,
                    optimize_base=not final_iteration,
                )
                merged = np.where(
                    np.asarray(track.contact_mask[:num_frames].any(axis=1))[:, None], refined_q, finger_q[side]
                ).astype(np.float32)
                if final_iteration:
                    finger_final[side] = merged
                    continue
                # feed the corrected hand back into the body anchors, low-passed in the object frame
                finger_q[side] = merged
                if num_frames >= 5:
                    T_object_base = self._smooth_pose7(T_object_base, self.config.object_frame_smooth_window)
                T_world_hand_base = pose7_to_tf_mat(track.T_world_object[:num_frames]) @ pose7_to_tf_mat(T_object_base)
                T_world_hand_link[side] = self._hands[side].robot.forward_kinematics_via_matrix_numpy(
                    finger_q[side], T_world_hand_base, self.device
                )

        body_joint_names = self.body_robot.spec.actuated_joint_names
        for side in sides:
            hand = self._hands[side]
            prefix = side[0].upper() + "_"
            hand_columns = 7 + np.array(
                [body_joint_names.index(prefix + name) for name in hand.robot.spec.actuated_joint_names]
            )
            limits = self.body_robot.spec.actuated_joint_limits[hand_columns - 7]
            qpos[:, hand_columns] = np.clip(finger_final[side], limits[:, 0], limits[:, 1])
        if scene_clearance and clip.scene is not None:
            qpos = apply_scene_clearance(self.body_robot, clip, qpos, self.device)
        return qpos

    def solve_hand_pass(
        self, side: str, track: HandTrack, num_frames: int, config: Optional[HandRetargeterOfflineConfig] = None
    ) -> np.ndarray:
        """Captured MANO keypoints -> packed `(T, 7 + dof)` on a floating hand, scaled by robot/subject hand-span."""
        return self.solve_hand_pass_batch(side, [track], [num_frames], config)[0]

    def solve_hand_pass_batch(
        self,
        side: str,
        tracks: List[HandTrack],
        valid_lengths: List[int],
        config: Optional[HandRetargeterOfflineConfig] = None,
    ) -> List[np.ndarray]:
        """Batch-retarget hands with masked padding and per-subject hand-span scale; `config` defaults to `hand_pass`.

        Example:
            >>> # xdoctest: +REQUIRES(env:ROBOKIT_PARAHOME_DOCS==1)
            >>> from robokit.assets.datasets import parahome
            >>> from robokit.helpers.full_body_retargeting.parahome_utils import find_parahome_intervals, load_parahome_clip
            >>> root = str(parahome.data_root())
            >>> clip = load_parahome_clip(root, find_parahome_intervals(root, "s1")[2])
            >>> solver = FullBodyRetargeter()
            >>> packed = solver.solve_hand_pass_batch("right", [clip.hands["right"]] * 2, [3, 5])
            >>> [len(row) for row in packed]
            [3, 5]
        """
        if not tracks or len(tracks) != len(valid_lengths):
            raise ValueError("tracks and valid_lengths must have the same positive length")
        if any(length < 3 or length > len(track.mano) for track, length in zip(tracks, valid_lengths)):
            raise ValueError("valid lengths must lie between 3 and the track length")
        hand = self._hands[side]
        scales = np.asarray(
            [
                np.clip(
                    hand.span / np.median(np.linalg.norm(track.mano[:length, 12] - track.mano[:length, 0], axis=-1)),
                    *self.config.hand_span_clamp,
                )
                for track, length in zip(tracks, valid_lengths)
            ],
            dtype=np.float32,
        )
        config = config or self.config.hand_pass
        key = (side, id(config))
        if key not in self._hand_passes:
            self._hand_passes[key] = HandRetargeterOffline(config, hand.robot, hand.spec, device=self.device)
        solver = self._hand_passes[key]
        solver.warmup(len(tracks), max(valid_lengths))
        solver.set_valid_lengths(np.asarray(valid_lengths, dtype=np.int32))
        solver.set_target_scales(scales)
        packed = solver.solve_numpy(
            _pad_tracks([track.mano[:length] for track, length in zip(tracks, valid_lengths)]),
            root_quat_wxyz=_pad_tracks(
                [track.q_wxyz_world_wrist[:length] for track, length in zip(tracks, valid_lengths)]
            ),
        )
        return [row[:length].copy() for row, length in zip(packed, valid_lengths)]

    def solve_batch(
        self,
        clips: List[FullBodyClip],
        hand_trajectories: List[Dict[str, HandTrajectory]],
        *,
        valid_lengths: Optional[List[int]] = None,
        scene_clearance: bool = False,
    ) -> List[np.ndarray]:
        """Batch body and arm passes on external hand trajectories; only hand sides must match across clips.

        Example:
            >>> # xdoctest: +REQUIRES(env:ROBOKIT_PARAHOME_DOCS==1)
            >>> from robokit.assets.datasets import parahome
            >>> from robokit.helpers.full_body_retargeting.parahome_utils import find_parahome_intervals, load_parahome_clip
            >>> root = str(parahome.data_root())
            >>> clip = load_parahome_clip(root, find_parahome_intervals(root, "s1")[2])
            >>> solver = FullBodyRetargeter()
            >>> trajectories = {side: HandTrajectory(np.zeros((3, solver.hand_robot(side).num_actuated_joints), np.float32), clip.hands[side].T_world_object[:3]) for side in clip.sides}
            >>> len(solver.solve_batch([clip], [trajectories], valid_lengths=[3])[0])
            3
        """
        if not clips or len(clips) != len(hand_trajectories):
            raise ValueError("clips and hand_trajectories must have the same positive length")
        sides = clips[0].sides
        lengths = valid_lengths if valid_lengths is not None else [clip.num_frames for clip in clips]
        if len(lengths) != len(clips) or any(
            length < 3 or length > clip.num_frames for clip, length in zip(clips, lengths)
        ):
            raise ValueError("valid lengths must lie between 3 and the clip length")
        if not sides or any(clip.sides != sides for clip in clips):
            raise ValueError("clips must have the same supported hand sides")
        if any(not clip.object_names for clip in clips):
            raise ValueError("no grasped object among hands")
        finger_q: List[Dict[str, np.ndarray]] = [{} for _ in clips]
        hand_links: List[Dict[str, np.ndarray]] = [{} for _ in clips]
        for side in sides:
            hand = self._hands[side]
            bases = []
            for i, (clip, trajectories, length) in enumerate(zip(clips, hand_trajectories, lengths)):
                if set(trajectories) != set(sides):
                    raise ValueError("an external trajectory is required for every hand")
                trajectory = trajectories[side]
                if (
                    trajectory.q.ndim != 2
                    or trajectory.q.shape[0] < length
                    or trajectory.q.shape[1] != hand.robot.spec.num_actuated_joints
                ):
                    raise ValueError(f"invalid {side} trajectory q shape")
                if (
                    trajectory.T_world_base.ndim != 2
                    or trajectory.T_world_base.shape[0] < length
                    or trajectory.T_world_base.shape[1] != 7
                ):
                    raise ValueError(f"invalid {side} trajectory T_world_base shape")
                finger_q[i][side] = uniform_filter1d(
                    trajectory.q[:length].astype(np.float32), self.config.smooth_window, axis=0, mode="nearest"
                )
                T_world_object = pose7_to_tf_mat(clip.hands[side].T_world_object[:length])
                T_world_base = pose7_to_tf_mat(trajectory.T_world_base[:length]).astype(np.float32)
                T_object_base = self._smooth_pose7(
                    tf_mat_to_pose7(inverse_tf_mat(T_world_object) @ T_world_base), self.config.smooth_window
                )
                bases.append(T_world_object @ pose7_to_tf_mat(T_object_base))
            links = hand.robot.forward_kinematics_via_matrix_numpy(
                np.concatenate([row[side] for row in finger_q]), np.concatenate(bases), self.device
            )
            offset = 0
            for i, length in enumerate(lengths):
                hand_links[i][side] = links[offset : offset + length]
                offset += length
        results = self._solve_body_batch(clips, hand_links, lengths, True)
        joint_names = self.body_robot.spec.actuated_joint_names
        for clip, fingers, qpos in zip(clips, finger_q, results):
            for side in sides:
                columns = np.asarray(
                    [
                        joint_names.index(side[0].upper() + "_" + name)
                        for name in self._hands[side].robot.spec.actuated_joint_names
                    ]
                )
                limits = self.body_robot.spec.actuated_joint_limits[columns]
                qpos[:, 7 + columns] = np.clip(fingers[side], limits[:, 0], limits[:, 1])
            if scene_clearance and clip.scene is not None:
                qpos[:] = apply_scene_clearance(self.body_robot, clip, qpos, self.device)
        return results

    def _solve_body_batch(
        self,
        clips: List[FullBodyClip],
        hand_links: List[Dict[str, np.ndarray]],
        lengths: List[int],
        external_hands: bool,
    ) -> List[np.ndarray]:
        sides = clips[0].sides
        wrist_indices = {side: self._hands[side].robot.spec.link_names.index(_HAND_BASE_LINK) for side in sides}
        target_overrides = {
            f"{side}_wrist_yaw_link": _pad_tracks(
                [
                    tf_mat_to_pose7(links[side][:, wrist_indices[side]] @ self._hands[side].T_handbase_wristyaw).astype(
                        np.float32
                    )
                    for links in hand_links
                ]
            )
            for side in sides
        }
        kin_packed = self._kin.solve_numpy(
            _pad_tracks(
                [clip.T_native_human[:length, self._human_joint_columns] for clip, length in zip(clips, lengths)]
            ),
            valid_lengths=np.asarray(lengths, dtype=np.int32),
            human_heights=np.asarray([clip.human_height for clip in clips], dtype=np.float32),
            target_overrides=target_overrides,
        )
        num_dofs = self.body_robot.spec.num_actuated_joints
        results = []
        for clip, length, packed in zip(clips, lengths, kin_packed):
            q_body = np.zeros((length, num_dofs), dtype=np.float32)
            q_body[:, self._kin_joint_columns] = packed[:length, 7:]
            if length >= 3:
                q_body = uniform_filter1d(q_body, self.config.smooth_window, axis=0, mode="nearest")
            object_tracks = {
                clip.hands[side].object_name: clip.hands[side].T_world_object[:length]
                for side in clip.sides
                if clip.hands[side].object_mesh is not None
            }
            object_tail = np.concatenate([object_tracks[name] for name in clip.object_names], axis=1)
            qpos = np.concatenate([packed[:length, :7], q_body, object_tail], axis=1, dtype=np.float32)
            if length >= 3:
                qpos[:, :7] = self._smooth_pose7(qpos[:, :7], self.config.smooth_window)
            results.append(qpos)

        # Arm IK has independent frames, so pack valid frames without temporal padding.
        key = (tuple(sides), sum(lengths), external_hands)
        if self._arm_ik_key != key:
            solver = self.config.arm_snap.solver
            if external_hands:
                solver = replace(
                    solver, stages=[replace(solver.stages[0], iters=self.config.arm_snap_trajectory_iters)]
                )
            ik_cfg = replace(self.config.arm_snap, solver=solver)
            links = [side[0].upper() + "_" + name for side in sides for name in (_HAND_BASE_LINK, *_ORIENTATION_LINKS)]
            self._arm_ik = IKSolver(ik_cfg, robot=self.body_robot, target_links=links, device=self.device)
            active = {
                index
                for index, name in enumerate(self.body_robot.spec.actuated_joint_names)
                for side in sides
                if name.startswith(f"{side}_") and any(part in name for part in ("shoulder", "elbow", "wrist"))
            }
            if external_hands:
                active.update(
                    index
                    for index, name in enumerate(self.body_robot.spec.actuated_joint_names)
                    if name.startswith("waist_")
                )
            self._arm_ik.set_active_joint_mask([1.0 if index in active else 0.0 for index in range(num_dofs)])
            self._arm_ik_key = key
        assert self._arm_ik is not None
        targets = []
        for qpos, links in zip(results, hand_links):
            T_base_world = inverse_tf_mat(pose7_to_tf_mat(qpos[:, :7]))
            targets.append(
                np.stack(
                    [
                        T_base_world @ links[side][:, self._hands[side].robot.spec.link_names.index(name)]
                        for side in sides
                        for name in (_HAND_BASE_LINK, *_ORIENTATION_LINKS)
                    ],
                    axis=1,
                ).astype(np.float32)
            )
        snap = (
            self._arm_ik.solve_torch(
                torch.from_numpy(np.concatenate(targets)).to(self.device),
                init_q=torch.from_numpy(np.concatenate([qpos[:, 7 : 7 + num_dofs] for qpos in results])).to(
                    self.device
                ),
            )
            .q.cpu()
            .numpy()
        )
        offset = 0
        for qpos, length in zip(results, lengths):
            qpos[:, 7 : 7 + num_dofs] = snap[offset : offset + length]
            offset += length
        return results

    def hand_robot(self, side: str) -> Robot:
        """Return the standalone hand robot for `side`, with its fingertip guard spheres."""
        return self._hands[side].robot

    def _refine(
        self,
        clip: FullBodyClip,
        side: str,
        *,
        T_world_body_link: np.ndarray,
        finger_q: np.ndarray,
        num_frames: int,
        optimize_base: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Refine contacts in one hand's object frame at the achieved wrist, attracting fingertips to the surface."""
        track = clip.hands[side]
        hand = self._hands[side]
        hand_robot = hand.robot
        T_world_wrist = T_world_body_link[
            :, self.body_robot.spec.link_names.index(f"{side[0].upper()}_{_HAND_BASE_LINK}")
        ]
        T_object_world = inverse_tf_mat(pose7_to_tf_mat(track.T_world_object[:num_frames]))
        T_object_base = tf_mat_to_pose7(T_object_world @ T_world_wrist @ inverse_tf_mat(hand.T_base_wrist[None])[0])

        # contact labels and normals into the object frame, via the native pose (labels are native)
        T_object_native = inverse_tf_mat(pose7_to_tf_mat(track.T_native_object[:num_frames]))
        R_object_native = T_object_native[:, :3, :3]
        points_object = (
            np.einsum("fij,fsj->fsi", R_object_native, track.contact_points[:num_frames])
            + T_object_native[:, None, :3, 3]
        )
        normals_object = np.einsum("fij,fsj->fsi", R_object_native, track.contact_normals[:num_frames])
        slot_link_indices = np.array(
            [
                hand_robot.spec.link_names.index(inspire_preset.spec.target_link_names[name])
                for name in self._contact_target_names
            ],
            dtype=np.int32,
        )
        T_object_link = hand_robot.forward_kinematics_via_matrix_numpy(
            finger_q, pose7_to_tf_mat(T_object_base), self.device
        )
        local_contact_points = self._inward_contact_points(
            hand_robot, slot_link_indices, T_object_link[:, slot_link_indices], normals_object
        )

        key = (side, optimize_base)
        cached = self._refiners.get(key)
        assert track.object_mesh is not None
        if cached is None or cached[0] is not track.object_mesh or cached[1] != num_frames:
            refiner = HandRetargeterOffline(
                replace(self.config.contact_refine, optimize_base=optimize_base),
                hand_robot,
                replace(hand.spec, contact_target_names=self._contact_target_names),
                scene=WarpScene(1, self.device).add(MeshGeom([track.object_mesh], np.asarray([0, 1], dtype=np.int32))),
                device=self.device,
            )
            refiner.warmup(1, num_frames)
            self._refiners[key] = (track.object_mesh, num_frames, refiner)
        refiner = self._refiners[key][2]
        packed = refiner.solve_with_contact_numpy(
            np.zeros((num_frames, len(inspire_preset.spec.target_names), 3), dtype=np.float32),
            points_object,
            track.contact_mask[:num_frames],
            init_state=hand_robot.state(
                q=wp.from_numpy(finger_q.astype(np.float32)[None], dtype=wp.float32, device=self.device),
                T_world_base=wp.from_numpy(T_object_base[None].astype(np.float32), dtype=wp_vec7, device=self.device),
            ),
            local_contact_points=local_contact_points,
        )
        return packed[:, 7:], packed[:, :7]

    def _load_hands(self, body_robot: Robot, device: str) -> Dict[str, _HandModel]:
        """Load the standalone Inspire hands and measure their assembly-relative frames."""
        from robokit.assets.robots.end_effectors import inspire_hand

        tip_names = [inspire_preset.spec.target_link_names[TARGET_NAMES[i]] for i in (4, 8, 12, 16, 20)]
        T_world_body_link = body_robot.forward_kinematics_via_matrix_numpy(
            np.zeros((1, body_robot.spec.num_actuated_joints), dtype=np.float32), device=device
        )[0]
        sources = (
            ("right", inspire_hand.URDF_PATH_RIGHT, inspire_hand.COLLISION_SPHERE_PATH_RIGHT),
            ("left", inspire_hand.URDF_PATH_LEFT, inspire_hand.COLLISION_SPHERE_PATH_LEFT),
        )
        hands = {}
        for side, path, spheres_path in sources:
            raw = Robot.load(
                path,
                load_meshes=True,
                mesh_dir=str(path.parent),
                load_collision_spheres=True,
                collision_spheres_path=spheres_path,
            )
            # append a small sphere at each fingertip so the contact refine guards the tips too
            link_indices = np.concatenate(
                [
                    raw.spec.collision_spheres_link_indices,
                    np.asarray([raw.spec.link_names.index(name) for name in tip_names], dtype=np.int32),
                ]
            )
            centers = np.concatenate([raw.spec.local_collision_sphere_centers, np.zeros((5, 3), dtype=np.float32)])
            radii = np.concatenate(
                [raw.spec.collision_sphere_radii, np.full(5, self.config.tip_sphere_radius, dtype=np.float32)]
            )
            order = np.asarray(
                sorted(range(len(link_indices)), key=lambda i: raw.spec.link_names[link_indices[i]]), dtype=np.int32
            )
            robot = Robot(
                replace(
                    raw.spec,
                    local_collision_sphere_centers=centers[order],
                    collision_sphere_radii=radii[order],
                    collision_spheres_link_indices=link_indices[order],
                )
            )
            hand_spec = replace(
                inspire_preset.spec,
                root_link_coord_spec=(
                    HandCoordinateSpec(palm_forward_axis="x", four_fingers_up_axis="-y")
                    if side == "left"
                    else inspire_preset.spec.root_link_coord_spec
                ),
            )
            T_world_link_rest = robot.forward_kinematics_via_matrix_numpy(
                np.zeros((1, robot.spec.num_actuated_joints), dtype=np.float32), device=device
            )[0]
            prefix = side[0].upper() + "_"
            T_world_wrist_yaw = T_world_body_link[body_robot.spec.link_names.index(f"{side}_wrist_yaw_link")]
            T_world_hand_base = T_world_body_link[body_robot.spec.link_names.index(f"{prefix}{_HAND_BASE_LINK}")]
            middle_tip_index = robot.spec.link_names.index(hand_spec.target_link_names["middle_tip"])
            wrist_index = robot.spec.link_names.index(_HAND_BASE_LINK)
            hands[side] = _HandModel(
                robot=robot,
                spec=hand_spec,
                T_base_wrist=T_world_link_rest[wrist_index],
                T_handbase_wristyaw=inverse_tf_mat(T_world_hand_base[None])[0] @ T_world_wrist_yaw,
                span=float(
                    np.linalg.norm(T_world_link_rest[middle_tip_index, :3, 3] - T_world_link_rest[wrist_index, :3, 3])
                ),
            )
        return hands

    @staticmethod
    def _smooth_pose7(pose7: np.ndarray, window: int) -> np.ndarray:
        """Low-pass a `(T, 7)` pose track, hemisphere-aligning quaternions so no average crosses a sign flip."""
        out = pose7.copy()
        out[:, :3] = uniform_filter1d(out[:, :3], window, axis=0, mode="nearest")
        quats = out[:, 3:]
        quats[1:] *= np.cumprod(np.where(np.einsum("fi,fi->f", quats[1:], quats[:-1]) < 0.0, -1.0, 1.0))[:, None]
        quats[:] = normalize_quaternion(uniform_filter1d(quats, window, axis=0, mode="nearest"))
        return out

    def _inward_contact_points(
        self, robot: Robot, slot_indices: np.ndarray, T_obj_link: np.ndarray, normals_obj: np.ndarray
    ) -> np.ndarray:
        """Per-slot link-mesh surface point facing furthest into the object; origin for meshless links."""
        inward_local = np.einsum("fsji,fsj->fsi", T_obj_link[..., :3, :3], -normals_obj)
        points = np.zeros(inward_local.shape, dtype=np.float32)
        for slot, link in enumerate(slot_indices):
            geom = robot.spec.link_visual_geometries.get(robot.spec.link_names[link])
            vertices = None if geom is None else np.asarray(geom.to_mesh().vertices, np.float32)
            if vertices is not None and vertices.shape[0]:
                points[:, slot] = vertices[np.argmax(inward_local[:, slot] @ vertices.T, axis=1)]
        return points


__all__ = ["FullBodyRetargeter"]
