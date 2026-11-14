"""DexYCB loading, MANO geometry, and contact generation."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import trimesh

from robokit.smplx import (
    MANO_NUM_JOINTS,
    MANOPTH_KEYPOINT_PERMUTATION,
    BodyModelSpecTensors,
    Side,
    body_lbs_warp,
    from_dexycb_pose,
    load_mano,
)
from robokit.xform.numpy import (
    axis_angle_to_quaternion,
    matrix_to_quaternion,
    quaternion_multiply,
    quaternion_to_matrix,
)


@lru_cache(maxsize=None)
def _get_mano_spec_tensors(side: Side, mano_root: Path, device: str) -> BodyModelSpecTensors:
    spec = load_mano(mano_root / f"MANO_{side.upper()}.npz", side=side)
    return BodyModelSpecTensors(spec=spec, device=device)


class DexYCBVideoDataset:
    """Read indexed DexYCB capture arrays.

    Args:
        data_dir: Directory containing the DexYCB index and pose archives.
        is_right: Whether to select right-hand captures instead of left-hand captures.
    """

    def __init__(self, data_dir: Path, is_right: bool = True) -> None:
        self._data_dir = Path(data_dir)
        self._index: Dict[str, Any] = json.loads((self._data_dir / "index.json").read_text())
        with np.load(self._data_dir / "poses.npz") as poses:
            self._poses: Dict[str, np.ndarray] = dict(poses)

        hand_side = "right" if is_right else "left"
        self._captures: List[Dict[str, Any]] = [
            capture for capture in self._index["captures"] if hand_side in capture["mano_sides"]
        ]

    def __len__(self) -> int:
        return len(self._captures)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        meta = self._captures[item]
        hand_pose = self._poses[meta["pose_keys"]["pose_m"]]
        object_poses = self._poses[meta["pose_keys"]["pose_y"]]
        object_pose = object_poses[:, meta["ycb_grasp_ind"]]

        extrinsic_name = meta["extrinsics"]
        extrinsic = self._index["calibration"]["extrinsics"][extrinsic_name]["extrinsics"]["apriltag"]
        extrinsic_mat = np.array(extrinsic, dtype=np.float32).reshape([3, 4])
        extrinsic_mat = np.concatenate([extrinsic_mat, np.array([[0, 0, 0, 1]], dtype=np.float32)], axis=0)

        mano_name = meta["mano_calib"][0]
        mano_parameters = np.array(self._index["calibration"]["mano"][mano_name]["betas"], dtype=np.float32)

        return {
            "hand_pose": hand_pose,
            "hand_shape": mano_parameters,
            "extrinsics": extrinsic_mat,
            "object_pose_camera_xyzw_xyz": object_pose,
            "object_mesh_path": self._data_dir / meta["grasp_object_mesh"],
            "capture_name": meta["id"],
        }


def compute_world_hand_geometry(
    hand_pose: np.ndarray,
    hand_shape: np.ndarray,
    extrinsics: np.ndarray,
    mano_root: Path,
    is_right: bool,
    device: str = "cuda:0",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute MANO geometry and transform it from camera to world coordinates.

    Args:
        hand_pose: DexYCB MANO pose parameters for `T` frames.
        hand_shape: MANO shape coefficients.
        extrinsics: Camera extrinsic matrix.
        mano_root: Directory containing the MANO model files.
        is_right: Whether to use the right-hand MANO model.
        device: Warp device used for linear blend skinning.

    Returns:
        World-frame vertices `[T, 778, 3]`, OpenPose-ordered keypoints
        `[T, 21, 3]`, and triangle faces `[F, 3]`.
    """
    side: Side = "right" if is_right else "left"
    spec_tensors = _get_mano_spec_tensors(side, mano_root, device)
    spec = spec_tensors.spec

    state = from_dexycb_pose(hand_pose, hand_shape, spec, flat_hand_mean=False)
    out = body_lbs_warp(spec_tensors, state, return_landmarks=True)

    joint_positions = out["T_world_joint"].numpy()[..., :3]
    landmarks = out["landmarks"].numpy()
    keypoints_native = np.concatenate([joint_positions, landmarks], axis=1)
    keypoints = keypoints_native[:, MANOPTH_KEYPOINT_PERMUTATION].astype(np.float32)
    verts = out["vertices"].numpy()

    T_world_camera = np.linalg.inv(extrinsics)
    R_world_camera = T_world_camera[:3, :3]
    t_world_camera = T_world_camera[:3, 3]
    vertices_world = (verts @ R_world_camera.T + t_world_camera).astype(np.float32)
    keypoints_world = (keypoints @ R_world_camera.T + t_world_camera).astype(np.float32)
    return vertices_world, keypoints_world, spec.faces


def compute_dexycb_contacts(
    dexycb_dir: Path,
    mano_root: Path,
    device: str = "cuda:0",
    limit: int = 0,
    contact_threshold: float = 0.02,
) -> Dict[str, np.ndarray]:
    """Compute DexYCB fingertip-object contacts.

    Frame cases:
    - Valid hand pose: use the closest object-surface point, with contact below `contact_threshold`.
    - Invalid hand pose: leave contacts masked out.

    Args:
        dexycb_dir: Directory containing indexed DexYCB captures.
        mano_root: Directory containing the MANO model files.
        device: Warp device used for MANO geometry.
        limit: Maximum captures processed per hand side, or zero for all captures.
        contact_threshold: Maximum fingertip-to-surface contact distance.

    Returns:
        Arrays keyed by `<capture>_points` and `<capture>_mask`, with shapes `[F_raw, 5, 3]` and `[F_raw, 5]` respectively.
    """
    tip_indices = np.flatnonzero(np.asarray(MANOPTH_KEYPOINT_PERMUTATION) >= MANO_NUM_JOINTS)
    queries: Dict[Path, trimesh.proximity.ProximityQuery] = {}
    out: Dict[str, np.ndarray] = {}
    for is_right in (True, False):
        dataset = DexYCBVideoDataset(dexycb_dir, is_right=is_right)
        num_captures = min(len(dataset), limit) if limit else len(dataset)
        for i in range(num_captures):
            sequence = dataset[i]
            hand_pose = sequence["hand_pose"]
            num_raw = hand_pose.shape[0]
            points_raw = np.zeros((num_raw, len(tip_indices), 3), np.float32)
            mask_raw = np.zeros((num_raw, len(tip_indices)), bool)
            valid_mask = np.linalg.norm(hand_pose[:, 0, :], axis=1) > 1e-5
            valid_indices = np.flatnonzero(valid_mask)
            if len(valid_indices):
                _, keypoints, _ = compute_world_hand_geometry(
                    hand_pose[valid_mask],
                    sequence["hand_shape"],
                    sequence["extrinsics"],
                    mano_root,
                    is_right,
                    device=device,
                )
                # Camera-frame object poses to world.
                T_world_camera = np.linalg.inv(sequence["extrinsics"])
                R_world_camera = T_world_camera[:3, :3]
                t_world_camera = T_world_camera[:3, 3]
                object_state = sequence["object_pose_camera_xyzw_xyz"][valid_indices].astype(np.float32)
                q_wxyz_camera_object = object_state[:, [3, 0, 1, 2]]
                q_wxyz_camera_object /= np.linalg.norm(q_wxyz_camera_object, axis=1, keepdims=True)
                R_world_object = R_world_camera @ quaternion_to_matrix(q_wxyz_camera_object).astype(np.float32)
                t_world_object = (object_state[:, 4:] @ R_world_camera.T + t_world_camera).astype(np.float32)

                mesh_path = sequence["object_mesh_path"]
                if mesh_path not in queries:
                    mesh = trimesh.load(mesh_path, process=False)
                    queries[mesh_path] = trimesh.proximity.ProximityQuery(mesh)
                query = queries[mesh_path]

                tips_world = keypoints[:, tip_indices]
                tips_object = np.einsum("fji,fpj->fpi", R_world_object, tips_world - t_world_object[:, None])
                closest, dist, _ = query.on_surface(tips_object.reshape(-1, 3).astype(np.float64))
                closest = closest.reshape(tips_world.shape).astype(np.float32)
                points_raw[valid_indices] = np.einsum("fij,fpj->fpi", R_world_object, closest) + t_world_object[:, None]
                mask_raw[valid_indices] = dist.reshape(tips_world.shape[:2]) < contact_threshold
            capture_name = sequence["capture_name"]
            out[f"{capture_name}_points"] = points_raw
            out[f"{capture_name}_mask"] = mask_raw
            if (i + 1) % 50 == 0 or i + 1 == num_captures:
                print(f"[{'right' if is_right else 'left'} {i + 1}/{num_captures}]", flush=True)
    return out


def load_dexycb_contacts(
    dexycb_dir: Path,
    capture_name: str,
    frame_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load precomputed fingertip-object contact targets for one DexYCB capture.

    Args:
        dexycb_dir: Directory containing `contacts.npz`.
        capture_name: Capture identifier used as the archive key prefix.
        frame_indices: Raw capture frames to select.

    Returns:
        World-frame closest surface points `[T, 5, 3]` and contact mask
        `[T, 5]`, ordered thumb through little finger.
    """
    with np.load(dexycb_dir / "contacts.npz") as data:
        return data[f"{capture_name}_points"][frame_indices], data[f"{capture_name}_mask"][frame_indices]


def load_dexycb_keypoints(
    dexycb_dir: Path,
    mano_root: Path,
    capture_index: int,
    is_right: bool = True,
    device: str = "cuda:0",
    dataset: Optional[DexYCBVideoDataset] = None,
) -> Dict[str, Any]:
    """Load valid hand and object trajectories for one DexYCB capture.

    Args:
        dexycb_dir: Directory containing indexed DexYCB captures.
        mano_root: Directory containing the MANO model files.
        capture_index: Capture index within the selected hand side.
        is_right: Whether to select right-hand captures instead of left-hand captures.
        device: Warp device used for MANO geometry.
        dataset: Optional already-loaded capture index, useful when processing many captures.

    Returns:
        World-frame hand and object trajectories, MANO geometry, source frame
        indices, mesh path, and capture name.
    """
    if dataset is None:
        dataset = DexYCBVideoDataset(dexycb_dir, is_right=is_right)
    sequence = dataset[capture_index]
    hand_pose = sequence["hand_pose"]
    valid_mask = np.linalg.norm(hand_pose[:, 0, :], axis=1) > 1e-5
    valid_indices = np.flatnonzero(valid_mask)
    valid_hand_pose = hand_pose[valid_mask]
    if valid_hand_pose.shape[0] == 0:
        raise ValueError("No valid hand pose frames found in DexYCB sequence.")

    vertices_world, joints_world, faces = compute_world_hand_geometry(
        valid_hand_pose,
        sequence["hand_shape"],
        sequence["extrinsics"],
        mano_root,
        is_right,
        device=device,
    )

    q_wxyz_camera_wrist = axis_angle_to_quaternion(valid_hand_pose[:, 0, :3]).astype(np.float32)
    T_world_camera = np.linalg.inv(sequence["extrinsics"])
    R_world_camera = T_world_camera[:3, :3]
    t_world_camera = T_world_camera[:3, 3]
    q_wxyz_world_camera = matrix_to_quaternion(R_world_camera).astype(np.float32)
    wrist_quat_wxyz = quaternion_multiply(q_wxyz_world_camera, q_wxyz_camera_wrist).astype(np.float32)
    object_state = sequence["object_pose_camera_xyzw_xyz"][valid_indices].astype(np.float32)
    q_wxyz_camera_object = object_state[:, [3, 0, 1, 2]]
    q_wxyz_camera_object = q_wxyz_camera_object / np.linalg.norm(q_wxyz_camera_object, axis=1, keepdims=True)
    object_quat_wxyz = quaternion_multiply(q_wxyz_world_camera, q_wxyz_camera_object).astype(np.float32)
    object_pos = (object_state[:, 4:] @ R_world_camera.T + t_world_camera).astype(np.float32)

    return {
        "keypoints": joints_world,
        "wrist_quat_wxyz": wrist_quat_wxyz,
        "object_pos": object_pos,
        "object_quat_wxyz": object_quat_wxyz,
        "object_mesh_path": sequence["object_mesh_path"],
        "frame_indices": valid_indices.astype(np.int64),
        "capture_name": sequence["capture_name"],
        "mano_vertices": vertices_world,
        "mano_faces": faces,
    }


def load_dexycb_grasp(
    capture_index: int = 0,
    is_right: bool = True,
    device: str = "cuda:0",
    dataset: Optional[DexYCBVideoDataset] = None,
    mesh_cache: Optional[Dict[Path, trimesh.Trimesh]] = None,
) -> Dict[str, Any]:
    """Load one DexYCB capture and its object mesh.

    Args:
        capture_index: Capture index within the selected hand side.
        is_right: Whether to select right-hand captures instead of left-hand captures.
        device: Warp device used for MANO geometry.
        dataset: Optional already-loaded capture index, useful when processing many captures.
        mesh_cache: Optional object mesh cache shared across captures.

    Returns:
        Hand and object trajectories plus `object_mesh`, `mano_vertices`, and
        `mano_faces`.

    Example:
        >>> load_dexycb_grasp(0)["keypoints"].shape[1:]  # doctest: +SKIP
        (21, 3)
    """
    from robokit.assets.body_models import mano
    from robokit.assets.datasets import dexycb

    data = load_dexycb_keypoints(dexycb.DIR, mano.DIR, capture_index, is_right=is_right, device=device, dataset=dataset)
    path = data["object_mesh_path"]
    if mesh_cache is not None and path in mesh_cache:
        data["object_mesh"] = mesh_cache[path]
    else:
        data["object_mesh"] = cast(trimesh.Trimesh, trimesh.load(path, process=False))
        if mesh_cache is not None:
            mesh_cache[path] = data["object_mesh"]
    return data


def main() -> None:
    """Compute fingertip contacts for all DexYCB captures."""
    import argparse

    import warp as wp

    from robokit.assets.body_models import mano
    from robokit.assets.datasets import dexycb

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None, help="output npz (default: <dexycb dir>/contacts.npz)")
    parser.add_argument("--limit", type=int, default=0, help="only process the first N captures per side")
    parser.add_argument("--upload", action="store_true", help="upload the npz to the HF dataset")
    args = parser.parse_args()

    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    dexycb_dir = dexycb.DIR
    out_path = args.out if args.out is not None else dexycb_dir / "contacts.npz"

    out = compute_dexycb_contacts(dexycb_dir, mano.DIR, device=device, limit=args.limit)
    np.savez_compressed(out_path, **cast(Dict[str, Any], out))
    touching = sum(bool(out[k].any()) for k in out if k.endswith("_mask"))
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB, {len(out) // 2} captures, {touching} touching)")

    if args.upload:
        from huggingface_hub import HfApi

        from robokit.assets import HF_REPO_ID

        info = HfApi().upload_file(
            path_or_fileobj=str(out_path),
            path_in_repo="datasets/dexycb/contacts.npz",
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message="datasets/dexycb: add precomputed fingertip contact points",
        )
        print(f"uploaded: {info}")


__all__ = [
    "DexYCBVideoDataset",
    "compute_dexycb_contacts",
    "compute_world_hand_geometry",
    "load_dexycb_contacts",
    "load_dexycb_grasp",
    "load_dexycb_keypoints",
]


if __name__ == "__main__":
    main()
