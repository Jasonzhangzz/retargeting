"""Contact-aware hand-object retargeting: load a clip, track hands, refine onto object contacts, then play or scrub."""

import argparse
import importlib
import json
import time
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import viser
import warp as wp
from viser.extras import ViserUrdf

from robokit.geom import MeshGeom, WarpScene
from robokit.helpers.hand_retargeting import HandRetargeterOffline, HandRetargeterOfflineConfig, HandSpec
from robokit.helpers.hand_retargeting.config import refine_config
from robokit.helpers.hand_retargeting.presets import sharpa
from robokit.robo import Robot
from robokit.smplx.constants import Side
from robokit.utils.ee_coord_utils import HandCoordinateSpec
from robokit.utils.warp_utils import wp_vec7
from robokit.xform.numpy import inverse_tf_mat, pose7_to_tf_mat, quaternion_apply, tf_mat_to_pose7


MANO16_LINKS = (
    "palm",
    "thumb1",
    "thumb2",
    "thumb3",
    "index1",
    "index2",
    "index3",
    "middle1",
    "middle2",
    "middle3",
    "ring1",
    "ring2",
    "ring3",
    "pinky1",
    "pinky2",
    "pinky3",
)


@dataclass(frozen=True)
class ObjectPart:
    """One rigid part of an articulated object."""

    name: str
    vertices: np.ndarray
    faces: np.ndarray


@dataclass(frozen=True)
class HumanHandTrack:
    """One wrist-local MANO-21 hand track and its object-part contacts."""

    keypoints_wrist: np.ndarray
    T_world_wrist: np.ndarray
    contact_points_part: np.ndarray
    contact_normals_part: np.ndarray
    contact_part: np.ndarray
    contact_mask: np.ndarray


@dataclass(frozen=True)
class ArticulatedObjectTrack:
    """Object root, articulation, rigid parts, and each part's world pose."""

    name: str
    T_world_root: np.ndarray
    qpos: np.ndarray
    parts: Tuple[ObjectPart, ...]
    T_world_part: np.ndarray


@dataclass(frozen=True)
class HandObjectClip:
    """Robot-independent bimanual clip with meter-based [x, y, z, qw, qx, qy, qz] poses."""

    name: str
    fps: float
    hands: Dict[Side, HumanHandTrack]
    object: ArticulatedObjectTrack

    @property
    def num_frames(self) -> int:
        """Number of synchronized frames.

        >>> obj = ArticulatedObjectTrack("o", np.zeros((3, 7)), np.zeros((3, 1)), (), np.zeros((3, 0, 7)))
        >>> HandObjectClip("x", 30.0, {}, obj).num_frames
        3
        """
        return int(self.object.T_world_root.shape[0])

    def keypoints_world(self, side: Side) -> np.ndarray:
        """Return one hand's MANO-21 keypoints in world coordinates.

        >>> pose = np.array([[1, 2, 3, 1, 0, 0, 0]], dtype=np.float32)
        >>> hand = HumanHandTrack(np.zeros((1, 21, 3)), pose, np.zeros((1, 1, 3)), np.zeros((1, 1, 3)), np.zeros((1, 1), np.int8), np.zeros((1, 1), bool))
        >>> obj = ArticulatedObjectTrack("o", pose, np.zeros((1, 1)), (), np.zeros((1, 0, 7)))
        >>> HandObjectClip("x", 30.0, {"right": hand}, obj).keypoints_world("right")[0, 0].tolist()
        [1.0, 2.0, 3.0]
        """
        hand = self.hands[side]
        T_world_wrist = hand.T_world_wrist
        return (T_world_wrist[:, None, :3] + quaternion_apply(T_world_wrist[:, None, 3:], hand.keypoints_wrist)).astype(
            np.float32
        )

    def contacts_world(self, side: Side) -> np.ndarray:
        """Return one hand's semantic contacts in world coordinates.

        >>> pose = np.array([[[1, 2, 3, 1, 0, 0, 0]]], dtype=np.float32)
        >>> hand = HumanHandTrack(np.zeros((1, 21, 3)), np.zeros((1, 7)), np.zeros((1, 1, 3)), np.zeros((1, 1, 3)), np.zeros((1, 1), np.int8), np.ones((1, 1), bool))
        >>> obj = ArticulatedObjectTrack("o", pose[:, 0], np.zeros((1, 1)), (), pose)
        >>> HandObjectClip("x", 30.0, {"right": hand}, obj).contacts_world("right")[0, 0].tolist()
        [1.0, 2.0, 3.0]
        """
        hand = self.hands[side]
        frames = np.arange(self.num_frames)[:, None]
        T_world_part = self.object.T_world_part[frames, hand.contact_part]
        return (T_world_part[..., :3] + quaternion_apply(T_world_part[..., 3:], hand.contact_points_part)).astype(
            np.float32
        )

    def contact_normals_world(self, side: Side) -> np.ndarray:
        """Return one hand's outward object contact normals in world coordinates.

        >>> pose = np.array([[[0, 0, 0, 1, 0, 0, 0]]], dtype=np.float32)
        >>> hand = HumanHandTrack(np.zeros((1, 21, 3)), np.zeros((1, 7)), np.zeros((1, 1, 3)), np.array([[[0, 0, 1]]], np.float32), np.zeros((1, 1), np.int8), np.ones((1, 1), bool))
        >>> obj = ArticulatedObjectTrack("o", pose[:, 0], np.zeros((1, 1)), (), pose)
        >>> HandObjectClip("x", 30.0, {"right": hand}, obj).contact_normals_world("right")[0, 0].tolist()
        [0.0, 0.0, 1.0]
        """
        hand = self.hands[side]
        frames = np.arange(self.num_frames)[:, None]
        T_world_part = self.object.T_world_part[frames, hand.contact_part]
        return quaternion_apply(T_world_part[..., 3:], hand.contact_normals_part).astype(np.float32)

    def select(self, frames: Union[np.ndarray, slice]) -> "HandObjectClip":
        """Select frames while sharing immutable object meshes.

        >>> obj = ArticulatedObjectTrack("o", np.zeros((3, 7)), np.zeros((3, 1)), (), np.zeros((3, 0, 7)))
        >>> HandObjectClip("x", 30.0, {}, obj).select(slice(1, 3)).num_frames
        2
        """
        hands: Dict[Side, HumanHandTrack] = {
            side: HumanHandTrack(
                hand.keypoints_wrist[frames],
                hand.T_world_wrist[frames],
                hand.contact_points_part[frames],
                hand.contact_normals_part[frames],
                hand.contact_part[frames],
                hand.contact_mask[frames],
            )
            for side, hand in self.hands.items()
        }
        obj = self.object
        return HandObjectClip(
            self.name,
            self.fps,
            hands,
            ArticulatedObjectTrack(
                obj.name,
                obj.T_world_root[frames],
                obj.qpos[frames],
                obj.parts,
                obj.T_world_part[frames],
            ),
        )


def load_hand_object_clip(path: Union[str, Path]) -> HandObjectClip:
    """Load a numeric-only canonical hand-object NPZ.

    >>> load_hand_object_clip("clip.npz").num_frames  # doctest: +SKIP
    """
    with np.load(path) as data:
        parts = tuple(
            ObjectPart(
                str(name),
                data[f"object_part_{index}_vertices"].copy(),
                data[f"object_part_{index}_faces"].copy(),
            )
            for index, name in enumerate(data["object_part_names"].tolist())
        )
        hands: Dict[Side, HumanHandTrack] = {
            side: HumanHandTrack(
                data[f"{side}_keypoints_wrist"].copy(),
                data[f"{side}_wrist_pose_world"].copy(),
                data[f"{side}_contact_points_part"].copy(),
                data[f"{side}_contact_normals_part"].copy(),
                data[f"{side}_contact_part"].copy(),
                data[f"{side}_contact_mask"].copy(),
            )
            for side in data["sides"].tolist()
        }
        obj = ArticulatedObjectTrack(
            str(data["object_name"]),
            data["object_root_pose_world"].copy(),
            data["object_qpos"].copy(),
            parts,
            data["object_part_pose_world"].copy(),
        )
        return HandObjectClip(str(data["name"]), float(data["fps"]), hands, obj)


CONTACT_LINK_TARGETS = (
    ("palm", "wrist"),
    ("thumb2", "thumb_mcp"),
    ("thumb3", "thumb_tip"),
    ("index2", "index_pip"),
    ("index3", "index_tip"),
    ("middle2", "middle_pip"),
    ("middle3", "middle_tip"),
    ("ring2", "ring_pip"),
    ("ring3", "ring_tip"),
    ("pinky2", "pinky_pip"),
    ("pinky3", "pinky_tip"),
)
FINGERTIP_CONTACTS = tuple(pair for pair in CONTACT_LINK_TARGETS if pair[1].endswith("_tip"))
CONTACT_CLEARANCE = 0.01


def _compute_inward_contact_points(
    robot: Robot, link_indices: np.ndarray, T_world_link: np.ndarray, normals: np.ndarray
) -> np.ndarray:
    """Return each link mesh point facing furthest into the object."""
    inward_local = np.einsum("fsji,fsj->fsi", T_world_link[..., :3, :3], -normals)
    points = np.zeros(inward_local.shape, dtype=np.float32)
    for slot, link in enumerate(link_indices):
        geom = robot.spec.link_visual_geometries.get(robot.spec.link_names[link])
        vertices = None if geom is None else np.asarray(geom.to_mesh().vertices, np.float32)
        if vertices is not None and vertices.shape[0]:
            points[:, slot] = vertices[np.argmax(inward_local[:, slot] @ vertices.T, axis=1)]
    return points


@dataclass(frozen=True)
class RobotHandTrack:
    """Retargeted joint trajectory and floating wrist base.

    >>> RobotHandTrack(np.zeros((2, 6)), np.zeros((2, 7))).qpos.shape
    (2, 6)
    """

    qpos: np.ndarray
    T_world_base: np.ndarray


@dataclass(frozen=True)
class HandObjectMotion:
    """Retargeted hands paired with the source object's exact kinematic track."""

    clip: HandObjectClip
    hands: Dict[Side, RobotHandTrack]


def hand_object_configs(
    offline: HandRetargeterOfflineConfig, surface_contact: bool = False
) -> Tuple[HandRetargeterOfflineConfig, HandRetargeterOfflineConfig]:
    """Derive smooth hand tracking and contact refinement from an offline preset; only refinement keeps `optimize_base`.

    >>> track, refine = hand_object_configs(HandRetargeterOfflineConfig())
    >>> track.task("velocity").weight, refine.task("contact").weight
    (30.0, 300.0)
    """
    track = replace(offline, optimize_base=True, terms=deepcopy(offline.terms))
    track.task("velocity").weight = 30.0
    track.task("acceleration").weight = 10.0
    track.task("root_smoothness").base_weight[3:6] = 50.0
    refine = refine_config(replace(track, optimize_base=offline.optimize_base))
    refine.task("acceleration").weight = 100.0
    refine.task("contact").contact_margin = 0.0
    if surface_contact:
        refine.task("collision").weight = 100.0
        refine.task("collision").margin = 0.0
        refine.task("contact").weight = 100.0
        refine.task("surface_contact").weight = 300.0
        refine.task("anchor").dense_task.weight = 20.0
        refine.task("base_anchor").dense_task.base_weight = 300.0
    else:
        refine.task("collision").weight = 0.0
        refine.task("contact").weight = 300.0
        refine.task("anchor").dense_task.weight = 10.0
    return track, refine


class HandObjectRetargeter:
    """Two-stage kinematic solve: `track` follows the hand keypoints, then `refine` anchors that motion to contacts."""

    def __init__(
        self,
        robots: Dict[Side, Robot],
        specs: Dict[Side, HandSpec],
        track: HandRetargeterOfflineConfig,
        refine: HandRetargeterOfflineConfig,
        device: str,
        surface_contact: bool = False,
    ) -> None:
        self.robots = robots
        self.specs = specs
        self.track = track
        self.refine = refine
        self.device = device
        self.surface_contact = surface_contact

    def solve(self, clip: HandObjectClip) -> HandObjectMotion:
        """Retarget the available hands without dynamics or object motion optimization.

        >>> retargeter.solve(clip)  # doctest: +SKIP
        """
        contact_indices = [MANO16_LINKS.index(name) for name, _ in CONTACT_LINK_TARGETS]
        target_names = tuple(target for _, target in CONTACT_LINK_TARGETS)
        fingertip_indices = [MANO16_LINKS.index(name) for name, _ in FINGERTIP_CONTACTS]
        fingertip_targets = tuple(target for _, target in FINGERTIP_CONTACTS)
        outputs: Dict[Side, RobotHandTrack] = {}
        T_world_bottom = pose7_to_tf_mat(clip.object.T_world_part[:, 0])
        T_bottom_world = inverse_tf_mat(T_world_bottom)
        R_bottom_world = T_bottom_world[:, :3, :3]
        scene = None
        scene_indices = None
        if self.surface_contact:
            part_meshes = [
                wp.Mesh(
                    points=wp.from_numpy(part.vertices, dtype=wp.vec3, device=self.device),
                    indices=wp.from_numpy(part.faces.reshape(-1), dtype=wp.int32, device=self.device),
                )
                for part in clip.object.parts
            ]
            num_parts = len(part_meshes)
            scene = WarpScene(clip.num_frames, self.device).add(
                MeshGeom(
                    [mesh for _ in range(clip.num_frames) for mesh in part_meshes],
                    np.arange(0, (clip.num_frames + 1) * num_parts, num_parts, dtype=np.int32),
                    poses=wp.from_numpy(
                        np.ascontiguousarray(
                            (T_bottom_world[:, None] @ pose7_to_tf_mat(clip.object.T_world_part)).reshape(-1, 4, 4)
                        ),
                        dtype=wp.mat44,
                        device=self.device,
                    ),
                )
            )
            scene_indices = wp.from_numpy(
                np.arange(clip.num_frames, dtype=np.int32)[None], dtype=wp.int32, device=self.device
            )
        for side in clip.hands:
            robot = self.robots[side]
            spec = replace(self.specs[side], contact_target_names=target_names)
            points_world = clip.keypoints_world(side)
            points_bottom = (
                np.einsum("fij,ftj->fti", R_bottom_world, points_world) + T_bottom_world[:, None, :3, 3]
            ).astype(np.float32)
            wrist_bottom = tf_mat_to_pose7(T_bottom_world @ pose7_to_tf_mat(clip.hands[side].T_world_wrist)).astype(
                np.float32
            )
            points = points_bottom if self.surface_contact else points_world
            root_quat = wrist_bottom[:, 3:] if self.surface_contact else clip.hands[side].T_world_wrist[:, 3:]
            direct = HandRetargeterOffline(self.track, robot, spec, device=self.device)
            packed = direct.solve_numpy(points, root_quat_wxyz=root_quat)
            if not self.surface_contact:
                link_indices = np.asarray(
                    [robot.spec.link_names.index(spec.target_link_names[name]) for name in target_names],
                    dtype=np.int32,
                )
                normals = clip.contact_normals_world(side)[:, contact_indices]
                init_state = robot.state(
                    q=wp.from_numpy(packed[None, :, 7:].copy(), dtype=wp.float32, device=self.device),
                    T_world_base=wp.from_numpy(packed[None, :, :7].copy(), dtype=wp_vec7, device=self.device),
                )
                robot.forward_kinematics(init_state)
                T_world_link = pose7_to_tf_mat(init_state.T_world_link.numpy()[0][:, link_indices])
                local_contacts = _compute_inward_contact_points(robot, link_indices, T_world_link, normals)
                refine = HandRetargeterOffline(self.refine, robot, spec, device=self.device)
                packed = refine.solve_with_contact_numpy(
                    points,
                    clip.contacts_world(side)[:, contact_indices] + CONTACT_CLEARANCE * normals,
                    clip.hands[side].contact_mask[:, contact_indices],
                    init_state=init_state,
                    local_contact_points=local_contacts,
                )
                outputs[side] = RobotHandTrack(packed[:, 7:], packed[:, :7])
                continue

            fingertip_links = np.asarray(
                [robot.spec.link_names.index(spec.target_link_names[name]) for name in fingertip_targets],
                dtype=np.int32,
            )
            sphere_indices = []
            for link_index in fingertip_links:
                sphere_link = robot.spec.link_names[link_index].replace("_fingertip", "_elastomer")
                candidates = np.flatnonzero(
                    robot.spec.collision_spheres_link_indices == robot.spec.link_names.index(sphere_link)
                )
                surface_y = (
                    robot.spec.local_collision_sphere_centers[candidates, 1]
                    - robot.spec.collision_sphere_radii[candidates]
                )
                sphere_indices.append(int(candidates[np.argmin(surface_y)]))
            refine_spec = replace(
                self.specs[side],
                contact_target_names=fingertip_targets,
                contact_sphere_indices=tuple(sphere_indices),
            )

            contacts_world = clip.contacts_world(side)[:, fingertip_indices]
            normals_world = clip.contact_normals_world(side)[:, fingertip_indices]
            contact_mask = clip.hands[side].contact_mask[:, fingertip_indices]
            contacts_bottom = (
                np.einsum("fij,fsj->fsi", R_bottom_world, contacts_world) + T_bottom_world[:, None, :3, 3]
            ).astype(np.float32)
            normals_bottom = np.einsum("fij,fsj->fsi", R_bottom_world, normals_world).astype(np.float32)
            init_state = robot.state(
                q=wp.from_numpy(packed[None, :, 7:].copy(), dtype=wp.float32, device=self.device),
                T_world_base=wp.from_numpy(packed[None, :, :7].copy(), dtype=wp_vec7, device=self.device),
            )
            robot.forward_kinematics(init_state)
            T_world_link = pose7_to_tf_mat(init_state.T_world_link.numpy()[0][:, fingertip_links])
            local_contacts = _compute_inward_contact_points(robot, fingertip_links, T_world_link, normals_bottom)
            refine = HandRetargeterOffline(
                self.refine, robot, refine_spec, scene=scene, device=self.device, scene_indices=scene_indices
            )
            packed = refine.solve_with_contact_numpy(
                points_bottom,
                contacts_bottom,
                contact_mask,
                init_state=init_state,
                local_contact_points=local_contacts,
                surface_contact_mask=contact_mask,
            )
            packed[:, :7] = tf_mat_to_pose7(T_world_bottom @ pose7_to_tf_mat(packed[:, :7]))
            outputs[side] = RobotHandTrack(packed[:, 7:], packed[:, :7])
        return HandObjectMotion(clip, outputs)


DATASETS = ("dexmachina", "gigahands", "oakink2")


def _load_dataset(
    dataset: str, device: str, original_contact: bool
) -> Tuple[Path, Dict[str, str], HandObjectRetargeter]:
    asset_root = importlib.import_module(f"robokit.assets.datasets.{dataset}").data_root()
    manifest = json.loads((asset_root / "manifest.json").read_text())
    clips = {f"{index:02d} {clip['label']}": clip["path"] for index, clip in enumerate(manifest["clips"], 1)}
    if not clips:
        raise ValueError(f"{dataset}: manifest contains no clips")
    from robokit.assets.robots.end_effectors import sharpa_hand  # downloads on import

    urdf_paths: Dict[Side, Path] = {"right": sharpa_hand.URDF_PATH_RIGHT, "left": sharpa_hand.URDF_PATH_LEFT}
    sphere_paths: Dict[Side, Path] = {
        "right": sharpa_hand.COLLISION_SPHERE_PATH_RIGHT,
        "left": sharpa_hand.COLLISION_SPHERE_PATH_LEFT,
    }
    robots: Dict[Side, Robot] = {
        side: Robot.load(
            str(path),
            load_meshes=True,
            mesh_dir=str(path.parent),
            load_collision_spheres=dataset != "gigahands",
            collision_spheres_path=sphere_paths[side] if dataset != "gigahands" else None,
        )
        for side, path in urdf_paths.items()
    }
    specs: Dict[Side, HandSpec] = {
        "right": sharpa.spec,
        "left": replace(
            sharpa.spec,
            target_link_names={
                name: link.replace("right_", "left_") for name, link in sharpa.spec.target_link_names.items()
            },
            root_link_coord_spec=HandCoordinateSpec("-z", "-y"),
        ),
    }
    surface_contact = dataset != "gigahands" and not original_contact
    track, refine = hand_object_configs(replace(sharpa.offline, optimize_base=dataset == "gigahands"), surface_contact)
    if dataset == "gigahands":
        for config in (track, refine):
            config.task("root_smoothness").base_weight[0:3] = 30.0
            config.task("root_smoothness").base_weight[3:6] = 50.0
        refine.task("base_anchor").dense_task.base_weight = 50.0
    retargeter = HandObjectRetargeter(robots, specs, track, refine, device, surface_contact)
    return asset_root, clips, retargeter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default=DATASETS[0])
    parser.add_argument("--frames", type=int, default=0, help="maximum sampled frames; 0 keeps the full clip")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--original-contact", action="store_true", help="use point contacts for DexMachina/OakInk2")
    args = parser.parse_args()
    if args.frames < 0:
        parser.error("--frames must be nonnegative")
    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    loaded_dataset = args.dataset
    asset_root, clips, retargeter = _load_dataset(loaded_dataset, device, args.original_contact)

    labels = tuple(clips)

    def solve(label: str) -> HandObjectMotion:
        clip = load_hand_object_clip(asset_root / clips[label])
        if args.frames:
            indices = np.linspace(0, clip.num_frames - 1, min(args.frames, clip.num_frames)).round().astype(np.int32)
            clip = clip.select(indices)
        start = time.time()
        motion = retargeter.solve(clip)
        print(
            f"retargeted {loaded_dataset}/{label}: {clip.num_frames} frames in {time.time() - start:.2f}s on {device}",
            flush=True,
        )
        return motion

    if args.no_viewer:
        solve(labels[0])
        return

    contact_indices = [MANO16_LINKS.index(name) for name, _ in CONTACT_LINK_TARGETS]
    server = viser.ViserServer()
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/ground", width=3.0, height=3.0)
    with server.gui.add_folder("Source"):
        dataset_dd = server.gui.add_dropdown("Dataset", DATASETS, initial_value=loaded_dataset)
        clip_dd = server.gui.add_dropdown("Clip", labels, initial_value=labels[0])
    slider = server.gui.add_slider("frame", 0, 1, 1, 0)
    playing = server.gui.add_checkbox("play", True)
    show_contacts = server.gui.add_checkbox("contacts", True)
    status = server.gui.add_text("status", initial_value="solving...", disabled=True)
    dirty = [True]
    dataset_dd.on_update(lambda _: dirty.__setitem__(0, True))
    clip_dd.on_update(lambda _: dirty.__setitem__(0, True))

    from robokit.assets.robots.end_effectors import sharpa_hand  # downloads on import

    urdf_paths: Dict[Side, Path] = {"right": sharpa_hand.URDF_PATH_RIGHT, "left": sharpa_hand.URDF_PATH_LEFT}
    robot_frames = {side: server.scene.add_frame(f"/robot/{side}", show_axes=False) for side in urdf_paths}
    robot_urdfs = {side: ViserUrdf(server, path, root_node_name=f"/robot/{side}") for side, path in urdf_paths.items()}
    contacts = {
        side: [
            server.scene.add_icosphere(f"/contacts/{side}/{slot}", radius=0.004, color=(40, 220, 80), visible=False)
            for slot in range(len(contact_indices))
        ]
        for side in urdf_paths
    }

    solved: Dict[str, HandObjectMotion] = {}
    current: Optional[HandObjectMotion] = None
    object_meshes = []
    contact_world: Dict[Side, np.ndarray] = {}
    offset = np.zeros(3, dtype=np.float32)
    built = ("", "")
    while True:
        if dirty[0]:
            dirty[0] = False
            # Snapshot both controls; only this loop rebuilds the scene and clip options.
            dataset, label = dataset_dd.value, clip_dd.value
            if dataset != loaded_dataset:
                status.value = f"loading {dataset}..."
                current = None
                solved.clear()  # Keep cached motions only for the selected dataset.
                asset_root, clips, retargeter = _load_dataset(dataset, device, args.original_contact)
                loaded_dataset = dataset
                label = next(iter(clips))
                clip_dd.options = tuple(clips)
                clip_dd.value = label
            if label not in clips:
                label = next(iter(clips))
                clip_dd.value = label
            selection = (dataset, label)
            if selection != built:
                status.value = f"solving {dataset}/{label}..."
                if label not in solved:
                    solved[label] = solve(label)
                current = solved[label]
                clip = current.clip
                for side, handle in robot_frames.items():
                    handle.visible = side in current.hands
                    for contact in contacts[side]:
                        contact.visible = False
                server.scene.remove_by_name("/object")
                object_meshes = [
                    server.scene.add_mesh_simple(
                        f"/object/{part.name}", part.vertices, part.faces, color=(210, 150, 70), opacity=0.8
                    )
                    for part in clip.object.parts
                ]
                contact_world = {side: clip.contacts_world(side)[:, contact_indices] for side in clip.hands}
                offset = np.array([0.0, 0.0, 1.0], np.float32) - clip.object.T_world_root[0, :3]
                slider.value = 0
                slider.max = clip.num_frames - 1
                status.value = f"{dataset}/{label}: {clip.num_frames} frames"
                built = selection

        if current is not None:
            clip = current.clip
            frame = min(slider.value, clip.num_frames - 1)
            if playing.value:
                frame = (frame + 1) % clip.num_frames
                slider.value = frame
            with server.atomic():
                for side, track in current.hands.items():
                    pose = track.T_world_base[frame]
                    robot_frames[side].position = pose[:3] + offset
                    robot_frames[side].wxyz = pose[3:]
                    robot_urdfs[side].update_cfg(track.qpos[frame])
                    mask = clip.hands[side].contact_mask[frame, contact_indices]
                    for slot, handle in enumerate(contacts[side]):
                        handle.visible = bool(show_contacts.value and mask[slot])
                        handle.position = contact_world[side][frame, slot] + offset
                for part_index, mesh in enumerate(object_meshes):
                    pose = clip.object.T_world_part[frame, part_index]
                    mesh.position, mesh.wxyz = pose[:3] + offset, pose[3:]
        time.sleep(1.0 / (current.clip.fps if current is not None else 30.0))


if __name__ == "__main__":
    main()
