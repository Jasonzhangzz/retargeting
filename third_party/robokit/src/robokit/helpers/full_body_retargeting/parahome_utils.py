"""Load ParaHome interactions (30 fps, metres, z-up) into `FullBodyClip`; hand objects inferred from proximity."""

import json
import pickle
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Tuple

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from robokit.assets.body_models import smplx as smplx_assets
from robokit.assets.datasets import parahome
from robokit.helpers.full_body_retargeting.clip import (
    CONTACT_MANO_INDICES,
    HANDS,
    FullBodyClip,
    HandTrack,
    SceneGeometry,
    mano_index,
)
from robokit.smplx import (
    SMPLX_JOINT_NAMES,
    SMPLX_STATIC_LANDMARK_NAMES,
    BodyModelSpecTensors,
    BodyModelState,
    body_lbs_warp,
    load_smplx,
)
from robokit.xform.numpy import tf_mat_to_pose7


CONTACT_DISTANCE = 0.025
PARAHOME_FPS = 30
GRASPABLE_DIAGONAL = 0.6  # m; parts bigger than this are furniture, not grasp targets
NUM_PROBE_FRAMES = 12  # frames sampled per annotated span when scoring hand-part proximity
ENGAGE_DISTANCE = 0.05  # m, maximum hand-to-part association distance for an engaged hand
ENGAGE_MULTI_RATIO = 0.25  # fraction of probe frames with >= 2 touching keypoints
ENGAGE_SINGLE_RATIO = 0.5  # fraction of probe frames with >= 1 touching keypoint
# ParaHome hand skeleton in MANO-21 order; right starts at 48, left at 23.
NATIVE_MANO21 = (0, 1, 22, 23, 24, 18, 19, 20, 21, 14, 15, 16, 17, 10, 11, 12, 13, 6, 7, 8, 9)
NATIVE_HAND_OFFSET = {"right": 48, "left": 23}


@dataclass(frozen=True)
class ParaHomeInterval:
    """One annotated interaction: a frame span of a sequence, with a target part per engaged hand."""

    sequence: str  # "s1"
    start: int  # inclusive, in native 30 fps frames
    end: int  # exclusive
    text: str  # the action annotation
    parts: Dict[str, str] = field(default_factory=dict)
    distances: Dict[str, float] = field(default_factory=dict)

    @property
    def sides(self) -> List[str]:
        return [side for side in HANDS if side in self.parts]

    @property
    def label(self) -> str:
        parts = "+".join(f"{side[0].upper()}:{self.parts[side]}" for side in self.sides)
        return f"{parts} {self.text}"


class _Candidate(NamedTuple):
    part: str
    score: float
    distance: float
    multi_ratio: float
    single_ratio: float


@lru_cache(maxsize=8)
def _read_pickle(path: Path) -> Any:
    return pickle.loads(path.read_bytes())


@lru_cache(maxsize=64)
def _part_mesh(part: str) -> trimesh.Trimesh:
    mesh = trimesh.load(str(parahome.mesh_dir() / f"{part}.obj"), force="mesh")
    assert isinstance(mesh, trimesh.Trimesh)
    return mesh


@lru_cache(maxsize=2)
def _model_tensors(gender: str, num_betas: int, device: str) -> BodyModelSpecTensors:
    model_npz = {"male": smplx_assets.MALE_NPZ_PATH, "female": smplx_assets.FEMALE_NPZ_PATH}[gender]
    spec = load_smplx(str(model_npz), gender, num_betas=num_betas)  # type: ignore[arg-type]
    return BodyModelSpecTensors(spec=spec, device=device)


def find_parahome_intervals(data_root: str, sequence: str) -> List[ParaHomeInterval]:
    """Mine annotated spans for the object part used by each hand.

    Args:
        data_root: ParaHome directory containing `seq` and `smplx_seq`.
        sequence: Sequence id such as `"s1"`.

    Returns:
        Intervals with at least one engaged hand, in annotation order.
    """
    root = Path(data_root)
    seq_dir = root / "seq" / sequence
    annotations = json.loads((seq_dir / "text_annotation.json").read_text())
    joints = _read_pickle(seq_dir / "joint_positions.pkl")
    transforms = _read_pickle(seq_dir / "object_transformations.pkl")

    trees: Dict[str, cKDTree] = {}
    for part in sorted({k for frame in transforms.values() for k in frame}):
        mesh = _part_mesh(part)
        if float(np.linalg.norm(mesh.extents)) <= GRASPABLE_DIAGONAL:
            trees[part] = cKDTree(np.asarray(mesh.vertices, dtype=np.float64))

    native_slots = np.asarray(NATIVE_MANO21)[np.asarray(CONTACT_MANO_INDICES)]
    slots = {side: NATIVE_HAND_OFFSET[side] + native_slots for side in HANDS}
    intervals = []
    for span, text in annotations.items():
        start, end = (int(x) for x in span.split())
        end = min(end, len(joints))
        if end - start < PARAHOME_FPS:
            continue
        probes = [f for f in range(start, end, max(1, (end - start) // NUM_PROBE_FRAMES)) if f in transforms]
        best: Dict[str, _Candidate] = {}
        for part, tree in trees.items():
            frames = [f for f in probes if part in transforms[f]]
            if len(frames) < len(probes) // 2 or not frames:
                continue
            for side, rows in slots.items():
                points_part = []
                for frame in frames:
                    T_world_part = np.asarray(transforms[frame][part], dtype=np.float32)
                    points_part.append((joints[frame, rows] - T_world_part[:3, 3]) @ T_world_part[:3, :3])
                distance = tree.query(np.concatenate(points_part))[0].reshape(len(frames), -1)
                multi_distance = float(np.median(np.sort(distance, axis=1)[:, :3].mean(axis=1)))
                single_distance = float(np.median(distance.min(axis=1)))
                contact_count = (distance < CONTACT_DISTANCE).sum(axis=1)
                multi_ratio = float(np.mean(contact_count >= 2))
                single_ratio = float(np.mean(contact_count >= 1))
                # single-point contact counts, at a one-contact-distance penalty over multi-point
                association_distance = min(multi_distance, single_distance + CONTACT_DISTANCE)
                score = association_distance + 0.5 * CONTACT_DISTANCE * (1.0 - multi_ratio)
                if side not in best or score < best[side].score:
                    best[side] = _Candidate(part, score, association_distance, multi_ratio, single_ratio)
        engaged = {
            side: candidate
            for side, candidate in best.items()
            if candidate.distance <= ENGAGE_DISTANCE
            and (candidate.multi_ratio >= ENGAGE_MULTI_RATIO or candidate.single_ratio >= ENGAGE_SINGLE_RATIO)
        }
        if engaged:
            intervals.append(
                ParaHomeInterval(
                    sequence=sequence,
                    start=start,
                    end=end,
                    text=text,
                    parts={side: candidate.part for side, candidate in engaged.items()},
                    distances={side: candidate.distance for side, candidate in engaged.items()},
                )
            )
    return intervals


def load_parahome_clip(
    data_root: str,
    interval: ParaHomeInterval,
    device: str = "cuda:0",
    load_scene: bool = False,
    target_height: float = 1.32,
) -> FullBodyClip:
    """Load one interval, its hand tracks, and optionally the room.

    Args:
        data_root: ParaHome directory containing `seq` and `smplx_seq`.
        interval: Mined interaction span to load.
        device: Warp device for the SMPL-X linear blend skinning.
        load_scene: Also load the untouched room parts.
        target_height: Robot height in metres that sets the world scale (default: G1).

    Returns:
        Clip with the subject floored at z = 0 in both worlds.
    """
    root = Path(data_root)
    seq_dir = root / "seq" / interval.sequence
    transforms = _read_pickle(seq_dir / "object_transformations.pkl")

    targets = set(interval.parts.values())
    frames = [f for f in range(interval.start, interval.end) if targets <= set(transforms.get(f, {}))]
    if not frames:
        raise ValueError(f"{sorted(targets)} are not all present for any frame of the span")
    num_frames = len(frames)

    params = _read_pickle(root / "smplx_seq" / interval.sequence / "smplx_params.pkl")
    pose = _read_pickle(root / "smplx_seq" / interval.sequence / "smplx_pose.pkl")
    gender = str(params["gender"])
    betas = params["beta"].cpu().numpy().astype(np.float32)
    spec_tensors = _model_tensors(gender, betas.shape[1], device)
    frame_rows = np.asarray(frames)
    full_pose = np.concatenate(
        [
            pose["global_orient"].cpu().numpy()[frame_rows].reshape(num_frames, 1, 3),
            pose["body_pose"].cpu().numpy()[frame_rows].reshape(num_frames, 21, 3),
            np.zeros((num_frames, 3, 3), np.float32),
            pose["hand_pose"].cpu().numpy()[frame_rows].reshape(num_frames, 30, 3),
        ],
        axis=1,
    ).astype(np.float32)
    state = BodyModelState(
        betas=np.repeat(betas, num_frames, axis=0),
        full_pose_aa=full_pose,
        transl=pose["transl"].cpu().numpy()[frame_rows].astype(np.float32),
    )
    out = body_lbs_warp(spec_tensors, state, return_landmarks=True)
    T_native_joint = out["T_world_joint"].numpy()
    landmarks = out["landmarks"].numpy()

    foot_indices = [
        SMPLX_STATIC_LANDMARK_NAMES.index(n) for n in ["left_heel", "right_heel", "left_big_toe", "right_big_toe"]
    ]
    floor_z = float(landmarks[:, foot_indices, 2].min())
    tpose = BodyModelState(
        betas=betas,
        full_pose_aa=np.zeros((1, 55, 3), dtype=np.float32),
        transl=np.zeros((1, 3), dtype=np.float32),
    )
    tpose_vertices = body_lbs_warp(spec_tensors, tpose, return_landmarks=False)["vertices"].numpy()[0]
    height = float(tpose_vertices[:, 1].max() - tpose_vertices[:, 1].min())
    scale = target_height / height

    joints = T_native_joint[..., :3] - [0.0, 0.0, floor_z]
    landmarks = landmarks - [0.0, 0.0, floor_z]

    track_cache: Dict[str, Tuple[trimesh.Trimesh, np.ndarray, np.ndarray, np.ndarray]] = {}

    def object_tracks(part: str) -> Tuple[trimesh.Trimesh, np.ndarray, np.ndarray, np.ndarray]:
        if part in track_cache:  # both hands often grasp the same part; load it once
            return track_cache[part]
        mesh = _part_mesh(part).copy()
        matrices = np.stack([np.asarray(transforms[f][part], dtype=np.float32) for f in frames])
        T_native_object = tf_mat_to_pose7(matrices)
        T_native_object[:, 2] -= floor_z
        T_world_object = T_native_object.copy()
        T_world_object[:, :3] *= scale
        track_cache[part] = (mesh, T_native_object, T_world_object, matrices[:, :3, :3])
        return track_cache[part]

    slots = np.asarray(CONTACT_MANO_INDICES)
    tracks: Dict[str, HandTrack] = {}
    for side in HANDS:
        joint_indices, tip_indices = mano_index(side)
        mano = np.stack(
            [landmarks[:, tip] if joint < 0 else joints[:, joint] for joint, tip in zip(joint_indices, tip_indices)],
            axis=1,
        ).astype(np.float32)
        wrist = SMPLX_JOINT_NAMES.index(f"{side}_wrist")
        wrist_position = joints[:, wrist].astype(np.float32)
        q_wxyz_world_wrist = T_native_joint[:, wrist, 3:].astype(np.float32)
        if side in interval.parts:
            part = interval.parts[side]
            mesh, T_native_object, T_world_object, R_world_object = object_tracks(part)
            points_object = np.einsum(
                "fij,fsj->fsi", np.swapaxes(R_world_object, 1, 2), mano[:, slots] - T_native_object[:, None, :3]
            )
            closest, distance, triangle = trimesh.proximity.closest_point(mesh, points_object.reshape(-1, 3))
            mask = (distance < CONTACT_DISTANCE).reshape(num_frames, len(slots))
            points_native = (
                np.einsum("fij,fsj->fsi", R_world_object, closest.reshape(num_frames, -1, 3).astype(np.float32))
                + T_native_object[:, None, :3]
            )
            normals_object = np.asarray(mesh.face_normals, dtype=np.float32)[triangle].reshape(num_frames, -1, 3)
            object_name, object_mesh = part, mesh
            contact_points = (points_native * mask[..., None]).astype(np.float32)
            contact_normals = (np.einsum("fij,fsj->fsi", R_world_object, normals_object) * mask[..., None]).astype(
                np.float32
            )
        else:
            T_native_object = np.concatenate([wrist_position, q_wxyz_world_wrist], axis=1)
            T_world_object = T_native_object.copy()
            T_world_object[:, :3] *= scale
            object_name, object_mesh = None, None
            mask = np.zeros((num_frames, len(slots)), dtype=bool)
            contact_points = np.zeros((num_frames, len(slots), 3), dtype=np.float32)
            contact_normals = np.zeros((num_frames, len(slots), 3), dtype=np.float32)
        tracks[side] = HandTrack(
            mano=mano,
            q_wxyz_world_wrist=q_wxyz_world_wrist,
            object_name=object_name,
            object_mesh=object_mesh,
            T_native_object=T_native_object,
            T_world_object=T_world_object,
            contact_points=contact_points,
            contact_normals=contact_normals,
            contact_mask=mask,
        )

    scene = None
    if load_scene:
        names, meshes, T_world_parts = [], [], []
        for part in sorted({k for f in frames for k in transforms[f]} - targets):
            if any(part not in transforms[f] for f in frames):
                continue
            T_world_part = tf_mat_to_pose7(
                np.stack([np.asarray(transforms[f][part], dtype=np.float32) for f in frames])
            )
            T_world_part[:, :3] = (T_world_part[:, :3] - [0.0, 0.0, floor_z]) * scale
            part_mesh = _part_mesh(part).copy()
            part_mesh.apply_scale(scale)
            names.append(part)
            meshes.append(part_mesh)
            T_world_parts.append(T_world_part)
        stacked = np.stack(T_world_parts) if T_world_parts else np.zeros((0, num_frames, 7), dtype=np.float32)
        scene = SceneGeometry(
            names=names,
            meshes=meshes,
            T_world_part=stacked,
            static=np.ptp(stacked, axis=1).max(axis=-1) < 1e-6 if names else np.zeros(0, dtype=bool),
        )

    return FullBodyClip(
        T_native_human=np.concatenate([joints, T_native_joint[..., 3:]], axis=-1).astype(np.float32),
        human_height=height,
        scale=scale,
        fps=PARAHOME_FPS,
        hands=tracks,
        scene=scene,
        frame_indices=frame_rows,
    )


__all__ = ["ParaHomeInterval", "find_parahome_intervals", "load_parahome_clip"]
