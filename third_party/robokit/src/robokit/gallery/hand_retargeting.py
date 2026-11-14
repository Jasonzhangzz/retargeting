"""Hand retargeting gallery: any hand on any base, online or offline, with or without wrist motion.

Usage: ``uv run -m robokit.gallery.hand_retargeting [--mode online] [--ee xhand] [--base xarm7]``
"""

import argparse
import asyncio
import pickle
import time
import weakref
import xml.etree.ElementTree as ET
from copy import deepcopy
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import trimesh
import viser
import warp as wp
from viser.extras import ViserUrdf

from robokit.assets import fetch
from robokit.assets.body_models import mano
from robokit.assets.datasets import dexycb
from robokit.assets.robots.assembly import xarm7_ability as xarm7_ability_asset
from robokit.assets.robots.assembly import xarm7_mano as xarm7_mano_asset
from robokit.assets.robots.assembly import xarm7_xhand as xarm7_xhand_asset
from robokit.assets.robots.end_effectors import (
    ability_hand,
    allegro_hand,
    barrett_hand,
    dclaw_gripper,
    inspire_hand,
    leap_hand,
    paxini_hand,
    schunk_hand,
    shadow_hand,
    sharpa_hand,
)
from robokit.assets.robots.end_effectors import xhand as xhand_asset
from robokit.gallery.robot_models import Coordinate, EntryData, entry_data
from robokit.helpers.hand_retargeting import (
    HandRetargeterOffline,
    HandRetargeterOfflineConfig,
    HandRetargeterOnline,
    HandRetargeterOnlineConfig,
    HandSpec,
)
from robokit.helpers.hand_retargeting.dexycb_utils import DexYCBVideoDataset, load_dexycb_keypoints
from robokit.helpers.hand_retargeting.presets import (
    ability,
    allegro,
    barrett,
    dclaw,
    inspire,
    leap,
    paxini,
    shadow,
    sharpa,
    svh,
    xarm7_ability,
    xarm7_mano,
    xarm7_xhand,
    xhand,
)
from robokit.robo import Robot
from robokit.utils.assembly_utils import assemble_urdf, compute_mount_transform, merge_collision_spheres
from robokit.utils.visualize_utils import ViserHandSkeleton
from robokit.utils.warp_utils import wp_vec7
from robokit.xform.numpy import quaternion_apply, quaternion_invert, quaternion_to_matrix


@dataclass(frozen=True)
class Preset:
    """One gallery robot: topology, per-mode solver settings, and asset paths."""

    spec: HandSpec
    urdf: Path
    online: Optional[HandRetargeterOnlineConfig] = None
    offline: Optional[HandRetargeterOfflineConfig] = None
    spheres: Optional[Path] = None
    ignore: Optional[Path] = None
    mount: Optional[Coordinate] = None
    online_wrist: Optional[HandRetargeterOnlineConfig] = None


def preset(
    spec: HandSpec,
    asset: Any,
    online: Any = None,
    offline: Any = None,
    spheres: bool = True,
    side: str = "_RIGHT",
    *,
    online_wrist: HandRetargeterOnlineConfig,
) -> Preset:
    """Preset for one hand, reading its paths and mount off the asset module's `side` constants."""
    data = entry_data((asset.__name__, f"URDF_PATH{side}", f"COLLISION_SPHERE_PATH{side}" if spheres else None))
    return Preset(
        spec, data.path, online, offline, data.spheres, data.ignore, data.mounts[0], online_wrist=online_wrist
    )


PRESETS: Dict[str, Preset] = {
    "ability": preset(ability.spec, ability_hand, ability.online, ability.offline, online_wrist=ability.online_wrist),
    "allegro": preset(allegro.spec, allegro_hand, allegro.online, allegro.offline, online_wrist=allegro.online_wrist),
    "barrett": preset(barrett.spec, barrett_hand, barrett.online, side="", online_wrist=barrett.online_wrist),
    "dclaw": preset(dclaw.spec, dclaw_gripper, dclaw.online, side="", online_wrist=dclaw.online_wrist),
    "inspire": preset(inspire.spec, inspire_hand, inspire.online, inspire.offline, online_wrist=inspire.online_wrist),
    "leap": preset(leap.spec, leap_hand, leap.online, leap.offline, spheres=False, online_wrist=leap.online_wrist),
    "paxini": preset(
        paxini.spec, paxini_hand, paxini.online, paxini.offline, spheres=False, online_wrist=paxini.online_wrist
    ),
    "shadow": preset(shadow.spec, shadow_hand, shadow.online, shadow.offline, online_wrist=shadow.online_wrist),
    "sharpa": preset(sharpa.spec, sharpa_hand, sharpa.online, sharpa.offline, online_wrist=sharpa.online_wrist),
    "svh": preset(svh.spec, schunk_hand, svh.online, svh.offline, spheres=False, online_wrist=svh.online_wrist),
    "xhand": preset(xhand.spec, xhand_asset, xhand.online, xhand.offline, online_wrist=xhand.online_wrist),
}
# Rigs shipped as ready-made URDFs; every other pair is assembled on demand.
BAKED: Dict[Tuple[str, str], Preset] = {
    ("ability", "xarm7"): Preset(
        xarm7_ability.spec,
        xarm7_ability_asset.URDF_PATH,
        xarm7_ability.online,
        xarm7_ability.offline,
        xarm7_ability_asset.COLLISION_SPHERE_PATH,
    ),
    ("ability_no_mimic", "xarm7"): Preset(
        xarm7_ability.spec_no_mimic,
        xarm7_ability_asset.URDF_NO_MIMIC_PATH,
        None,
        xarm7_ability.offline_no_mimic,
        xarm7_ability_asset.COLLISION_SPHERE_PATH,
    ),
    ("mano", "xarm7"): Preset(
        xarm7_mano.spec,
        xarm7_mano_asset.URDF_PATH,
        None,
        xarm7_mano.offline,
        xarm7_mano_asset.COLLISION_SPHERE_PATH,
    ),
    ("xhand", "xarm7"): Preset(
        xarm7_xhand.spec,
        xarm7_xhand_asset.URDF_PATH,
        xarm7_xhand.online,
        xarm7_xhand.offline,
        xarm7_xhand_asset.COLLISION_SPHERE_PATH,
        xarm7_xhand_asset.SPHERE_SELF_COLLISION_IGNORE_PATH,
    ),
}
ARM_NAMES = ("fr3", "gen3", "iiwa14", "piper", "rizon4", "ur10", "ur5e", "xarm6", "xarm7", "yam", "z1")
HUMANOID_NAMES = ("unitree_g1", "unitree_h1_2", "fourier_gr3", "vega_1")
# Bases a hand can ride: the floating hand itself, every single-mount arm, and humanoid right arms.
FLOATING = "floating"
EE_PREFIX = "ee_"
ASSEMBLY_DIR = Path.home() / ".cache" / "robokit" / "hand_retargeting"
BASE_NAMES = (FLOATING,) + ARM_NAMES + HUMANOID_NAMES
MODES = ("online", "offline")
# The with-wrist clip moves in DexYCB world; fingers-only replays relative fingers on a static base.
WITH_WRIST, FINGERS_ONLY = "with-wrist", "fingers-only"
MOTIONS = (WITH_WRIST, FINGERS_ONLY)
MOTION_FPS = {WITH_WRIST: 60, FINGERS_ONLY: 90}
# Place source keypoints beside fingers-only robots; their static origin view would otherwise overlap.
TARGET_OFFSET = {WITH_WRIST: (0.0, 0.0, 0.0), FINGERS_ONLY: (0.0, -0.25, 0.0)}
MOTION_CAMERA = {
    WITH_WRIST: ((1.55, 0.6, 0.42), (0.55, 0.6, 0.16)),
    FINGERS_ONLY: ((0.34, 0.10, 0.20), (0.0, -0.12, 0.0)),
}
ARM_BASE_POSE = np.array([0.9, 0.9, 0.0, 0.70710678, 0.0, 0.0, -0.70710678], dtype=np.float32)
# Vega's root sits at its wheels, so lower it to put its hands at the height the other humanoids rest at.
BASE_HEIGHT = {"vega_1": -0.9}
# Joints frozen at the base's rest pose, so only the arm reaches.
LOCKED_JOINTS = {
    "vega_1": (
        "B_wheel_j1",
        "B_wheel_j2",
        "R_wheel_j1",
        "R_wheel_j2",
        "L_wheel_j1",
        "L_wheel_j2",
        "torso_j1",
        "torso_j2",
        "torso_j3",
    )
}
IDENTITY_POSE = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
DEVICE = "cuda:0"


def ee_options(mode: str) -> List[str]:
    """Hands whose preset supports `mode`."""
    return [name for name, p in PRESETS.items() if (p.online if mode == "online" else p.offline) is not None]


@lru_cache(maxsize=None)
def base_data(name: str) -> EntryData:
    """Asset constants for one base, imported on first use so startup fetches nothing."""
    if name in ARM_NAMES:
        return entry_data((f"robokit.assets.robots.arms.{name}", "URDF_PATH", "COLLISION_SPHERE_PATH"))
    # A humanoid mounts onto its handless variant where one ships, without its body spheres.
    model_attr = "URDF_WITHOUT_HAND_PATH" if name in ("unitree_g1", "fourier_gr3") else "URDF_PATH"
    return entry_data((f"robokit.assets.robots.humanoids.{name}", model_attr, None))


def base_options(motion: str) -> List[str]:
    """Bases a motion can drive; a fingers-only clip has no world placement for an arm."""
    return [FLOATING] if motion == FINGERS_ONLY else list(BASE_NAMES)


class Gallery:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.mano_root = Path(args.mano_root) if args.mano_root else mano.DIR
        dataset = DexYCBVideoDataset(dexycb.DIR, is_right=True)
        self.capture_names = [dataset[index]["capture_name"] for index in range(len(dataset))]
        self.capture_index = args.capture_index % len(self.capture_names)
        self.mode = args.mode
        self.ee_name = args.ee
        self.base_name = args.base
        self.spec = PRESETS[self.ee_name].spec
        self.motion = args.motion
        self.sequence_cache: Dict[Tuple[Any, ...], Dict] = {}
        self.model_cache: Dict[Tuple[Any, ...], Tuple[Robot, HandSpec, Any]] = {}
        self.solution_cache: Dict[Tuple[Any, ...], Tuple[np.ndarray, np.ndarray]] = {}
        self.rig_cache: Dict[Tuple[str, str], Preset] = {}
        self.reloading = False
        self.reload_epoch = 0
        self.suppress_frame_update = False
        self.suppress_robot_update = False
        self.suppress_sequence_update = False

        self.server = viser.ViserServer(host="0.0.0.0", port=args.port)
        self.server.scene.set_up_direction("+z")
        # Callbacks reach the gallery weakly: the server holds them and the gallery holds the server.
        gallery = weakref.ref(self)

        @self.server.on_client_connect
        async def _on_client_connect(client: viser.ClientHandle):
            self = cast(Gallery, gallery())
            while client.camera._state.update_timestamp == 0.0:
                await asyncio.sleep(0.05)
            self.point_camera(client)

        self.robot_frame = self.server.scene.add_frame("/robot", show_axes=False)
        self.urdf_vis: Optional[ViserUrdf] = None
        self.object_handle = None
        self.table_handle = None
        self.mano_handle = None
        self.target_frame = self.server.scene.add_frame("/targets", show_axes=False)
        self.fingertip_handles = [
            self.server.scene.add_icosphere(f"/targets/fingertips/{idx}", radius=0.003, color=(40, 140, 255))
            for idx in range(max(len(preset.spec.contact_target_names) for preset in PRESETS.values()))
        ]
        self.skeleton = ViserHandSkeleton(self.server, "/targets/hand")

        with self.server.gui.add_folder("Source", expand_by_default=True):
            self.mode_dropdown = self.server.gui.add_dropdown("Mode", options=MODES, initial_value=self.mode)
            self.ee_dropdown = self.server.gui.add_dropdown(
                "EE", options=ee_options(self.mode), initial_value=self.ee_name
            )
            self.base_dropdown = self.server.gui.add_dropdown(
                "Base", options=base_options(self.motion), initial_value=self.base_name
            )
            self.status_text = self.server.gui.add_text("Status", initial_value="Loading...", disabled=True)
        with self.server.gui.add_folder("Motion", expand_by_default=True):
            self.motion_dropdown = self.server.gui.add_dropdown("Wrist", options=MOTIONS, initial_value=self.motion)
            self.sequence_dropdown = self.server.gui.add_dropdown(
                "Sequence",
                options=self.sequence_names(self.motion),
                initial_value=self.sequence_names(self.motion)[self.capture_index],
            )
        with self.server.gui.add_folder("Playback", expand_by_default=True):
            self.frame_slider = self.server.gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0)
            self.playing = self.server.gui.add_checkbox("Play", initial_value=True)
            self.fps_slider = self.server.gui.add_slider(
                "FPS", min=5, max=120, step=5, initial_value=MOTION_FPS[self.motion]
            )

        @self.mode_dropdown.on_update
        def _on_mode_update(event: viser.GuiEvent):
            self = cast(Gallery, gallery())
            self.mode = str(event.target.value)
            self.sync_robot_options()

        @self.ee_dropdown.on_update
        def _on_ee_update(event: viser.GuiEvent):
            self = cast(Gallery, gallery())
            if self.suppress_robot_update:
                return
            self.ee_name = str(event.target.value)
            self.reload()

        @self.base_dropdown.on_update
        def _on_base_update(event: viser.GuiEvent):
            self = cast(Gallery, gallery())
            if self.suppress_robot_update:
                return
            self.base_name = str(event.target.value)
            self.reload()

        @self.motion_dropdown.on_update
        def _on_motion_update(event: viser.GuiEvent):
            self = cast(Gallery, gallery())
            self.motion = str(event.target.value)
            self.fps_slider.value = MOTION_FPS[self.motion]
            for client in self.server.get_clients().values():
                self.point_camera(client)
            names = self.sequence_names(self.motion)
            self.capture_index = 0
            self.suppress_sequence_update = True
            self.sequence_dropdown.options = names
            self.sequence_dropdown.value = names[0]
            self.suppress_sequence_update = False
            self.sync_robot_options()

        @self.sequence_dropdown.on_update
        def _on_sequence_update(event: viser.GuiEvent):
            self = cast(Gallery, gallery())
            if self.suppress_sequence_update:
                return
            self.capture_index = self.sequence_names(self.motion).index(str(event.target.value))
            self.reload()

        @self.frame_slider.on_update
        def _on_frame_update(_event: viser.GuiEvent):
            self = cast(Gallery, gallery())
            if self.reloading or self.suppress_frame_update:
                return
            self.frame_idx = int(self.frame_slider.value)
            self.update(self.frame_idx)

        self.reload()

    def sequence_names(self, motion: str) -> List[str]:
        """Sequences the given motion offers, in capture-index order."""
        return ["manus_right"] if motion == FINGERS_ONLY else self.capture_names

    def point_camera(self, client: viser.ClientHandle) -> None:
        """Frame the scene for the current motion."""
        position, look_at = MOTION_CAMERA[self.motion]
        client.camera.position = position
        client.camera.look_at = look_at
        client.camera.up_direction = (0.0, 0.0, 1.0)

    def sync_robot_options(self) -> None:
        """Restrict the EE and Base dropdowns to what the mode and motion allow, then reload."""
        ees, bases = ee_options(self.mode), base_options(self.motion)
        if self.ee_name not in ees:
            self.ee_name = ees[0]
        if self.base_name not in bases:
            self.base_name = bases[0]
        self.suppress_robot_update = True
        self.ee_dropdown.options = ees
        self.ee_dropdown.value = self.ee_name
        self.base_dropdown.options = bases
        self.base_dropdown.value = self.base_name
        self.suppress_robot_update = False
        self.reload()

    def rig(self, ee_name: str, base_name: str) -> Preset:
        """Preset for one hand on one base: the hand alone, a baked rig, or one assembled on demand."""
        if base_name == FLOATING:
            return PRESETS[ee_name]
        if (ee_name, base_name) in BAKED:
            return BAKED[(ee_name, base_name)]
        key = (ee_name, base_name)
        if key not in self.rig_cache:
            preset = PRESETS[ee_name]
            base = base_data(base_name)
            assert preset.mount is not None and preset.online_wrist is not None
            base_link, base_mount_spec, _ = base.mounts[0]
            ee_link, ee_mount_spec, _ = preset.mount
            out_dir = ASSEMBLY_DIR / f"{base_name}_{ee_name}"
            urdf = assemble_urdf(
                out_dir / "robot.urdf",
                base.path,
                preset.urdf,
                base_link,
                ee_link,
                compute_mount_transform(base_mount_spec, ee_mount_spec),
                EE_PREFIX,
            )
            locked = LOCKED_JOINTS.get(base_name, ())
            tree = ET.parse(urdf)
            for joint in tree.getroot().findall("joint"):
                if joint.get("name") in locked and joint.get("type") != "fixed":
                    # Fold the rest angle into the joint origin so it no longer moves.
                    origin = joint.find("origin")
                    if origin is None:
                        origin = ET.SubElement(joint, "origin")
                    T = trimesh.transformations.compose_matrix(
                        angles=[float(v) for v in origin.get("rpy", "0 0 0").split()],
                        translate=[float(v) for v in origin.get("xyz", "0 0 0").split()],
                    )
                    axis = [float(v) for v in joint.find("axis").get("xyz").split()]
                    T = T @ trimesh.transformations.rotation_matrix(base.rest_q[joint.attrib["name"]], axis)
                    origin.set("xyz", " ".join(map(str, T[:3, 3])))
                    origin.set("rpy", " ".join(map(str, trimesh.transformations.euler_from_matrix(T, axes="sxyz"))))
                    joint.set("type", "fixed")
            tree.write(urdf)
            spheres = merge_collision_spheres(
                out_dir / "collision_spheres.yaml",
                [(path, prefix) for path, prefix in ((base.spheres, ""), (preset.spheres, EE_PREFIX)) if path],
            )
            spec = replace(
                preset.spec,
                floating_base=False,
                target_link_names={k: EE_PREFIX + v for k, v in preset.spec.target_link_names.items()},
                init_q_by_name={k: v for k, v in base.rest_q.items() if k not in locked},
                rest_q_by_name={EE_PREFIX + k: v for k, v in preset.spec.rest_q_by_name.items()},
            )

            # Correspondences are keyed by link name, which assembly prefixes.
            def shift(config: Optional[HandRetargeterOnlineConfig]) -> Optional[HandRetargeterOnlineConfig]:
                if config is None:
                    return None
                config = replace(
                    config,
                    terms=deepcopy(config.terms),
                    pinch_correspondences=tuple(
                        (EE_PREFIX + a, EE_PREFIX + b, c, d) for a, b, c, d in config.pinch_correspondences
                    ),
                )
                for task in config.terms:  # a fingers-only config has vectors but no directions
                    if task.name in ("vectors", "directions"):
                        task.weight = {
                            (EE_PREFIX + a, EE_PREFIX + b, c, d): w for (a, b, c, d), w in task.weight.items()
                        }
                return config

            self.rig_cache[key] = replace(
                preset,
                urdf=urdf,
                spec=spec,
                online=shift(preset.online),
                online_wrist=shift(preset.online_wrist),
                spheres=spheres,
                ignore=None,
            )
        return self.rig_cache[key]

    def model(self, ee_name: str, base_name: str, mode: str, motion: str) -> Tuple[Robot, HandSpec, Any]:
        """Loaded robot, solved-for spec, and solver, built once per setting."""
        key = (ee_name, base_name, mode, motion)
        if key not in self.model_cache:
            preset = self.rig(ee_name, base_name)
            static_base = motion == FINGERS_ONLY
            spec = replace(preset.spec, floating_base=False) if static_base else preset.spec
            robot = Robot.load(
                str(preset.urdf),
                load_collision_spheres=preset.spheres is not None,
                collision_spheres_path=preset.spheres,
                self_collision_ignore_path=preset.ignore,
            )
            if mode == "online":
                config = preset.online_wrist if not static_base and preset.online_wrist is not None else preset.online
                assert config is not None
                solver: Any = HandRetargeterOnline(config, robot, spec, device=DEVICE)
                solver.warmup(1)
            else:
                assert preset.offline is not None
                config = replace(preset.offline, optimize_base=not static_base)
                solver = HandRetargeterOffline(config, robot, spec, device=DEVICE)
            self.model_cache[key] = (robot, spec, solver)
        return self.model_cache[key]

    def sequence(self) -> Dict[str, Any]:
        """Source frames for the current motion, loaded once."""
        key = (FINGERS_ONLY,) if self.motion == FINGERS_ONLY else (WITH_WRIST, self.capture_index)
        if key not in self.sequence_cache:
            if self.motion == FINGERS_ONLY:
                path = fetch(["datasets/hand_retargeting_vector/**"]) / "datasets/hand_retargeting_vector"
                with open(path / "manus_data_right.pkl", "rb") as file:
                    points = np.asarray(pickle.load(file, encoding="latin1")["keypoints"], dtype=np.float32)
                self.sequence_cache[key] = {
                    "keypoints": points - points[:, :1],
                    "capture_name": "manus_right",
                }
            else:
                self.sequence_cache[key] = load_dexycb_keypoints(
                    dexycb_dir=dexycb.DIR,
                    mano_root=self.mano_root,
                    capture_index=self.capture_index,
                    is_right=True,
                    device=DEVICE,
                )
        return self.sequence_cache[key]

    def reload(self) -> None:
        self.reload_epoch += 1
        if self.reloading:
            return
        self.reloading = True
        self.playing.value = False
        reload_epoch = self.reload_epoch
        capture_index = self.capture_index
        rig_name = self.ee_name if self.base_name == FLOATING else f"{self.base_name}+{self.ee_name}"
        preset = self.rig(self.ee_name, self.base_name)
        self.status_text.value = f"Loading {self.motion} / {rig_name}..."

        data = self.sequence()
        points = data["keypoints"].astype(np.float32)
        static_base = self.motion == FINGERS_ONLY
        root_quat_wxyz = (
            np.tile(np.array([1, 0, 0, 0], np.float32), (points.shape[0], 1))
            if static_base
            else data["wrist_quat_wxyz"].astype(np.float32)
        )
        num_frames = points.shape[0]
        start_time = time.time()
        robot, spec, solver = self.model(self.ee_name, self.base_name, self.mode, self.motion)
        base_pose = ARM_BASE_POSE.copy()
        base_pose[2] = BASE_HEIGHT.get(self.base_name, 0.0)

        # Arm targets are expressed in the robot base frame; floating hands use world points.
        solver_points = points
        if not static_base and self.mode == "online" and not spec.floating_base:
            solver_points = quaternion_apply(quaternion_invert(base_pose[3:]), points - base_pose[:3]).astype(
                np.float32
            )

        online: Optional[HandRetargeterOnline] = None
        if self.mode == "online":
            online = cast(HandRetargeterOnline, solver)
            online.reset()
            q = np.tile(robot.spec.zero_q.astype(np.float32), (num_frames, 1))
            T_world_base = np.tile(IDENTITY_POSE if static_base else base_pose, (num_frames, 1))
        else:
            solution_key = (rig_name, self.mode, self.motion, capture_index)
            if solution_key not in self.solution_cache:
                init_state = None
                if not spec.floating_base and not static_base:
                    q_init = np.tile(robot.spec.zero_q.astype(np.float32), (1, num_frames, 1))
                    joint_index = {name: index for index, name in enumerate(robot.spec.actuated_joint_names)}
                    for name, value in spec.init_q_by_name.items():
                        q_init[:, :, joint_index[name]] = value
                    # Seed the arm's IK branch from the first three motion frames.
                    first_state = robot.state(
                        q=wp.from_numpy(q_init[:, :3].copy(), dtype=wp.float32, device=DEVICE),
                        T_world_base=wp.from_numpy(np.tile(base_pose, (1, 3, 1)), dtype=wp_vec7, device=DEVICE),
                    )
                    first = cast(HandRetargeterOffline, solver).solve_numpy(
                        solver_points[None, :3], root_quat_wxyz[None, :3], first_state
                    )
                    q_init[:] = first[:, :1, 7:]
                    init_state = robot.state(
                        q=wp.from_numpy(q_init, dtype=wp.float32, device=DEVICE),
                        T_world_base=wp.from_numpy(
                            np.tile(base_pose, (1, num_frames, 1)), dtype=wp_vec7, device=DEVICE
                        ),
                    )
                packed = cast(HandRetargeterOffline, solver).solve_numpy(
                    solver_points[None], root_quat_wxyz[None], init_state
                )
                self.solution_cache[solution_key] = (packed[0, :, 7:].copy(), packed[0, :, :7].copy())
            q, T_world_base = self.solution_cache[solution_key]
            if static_base:
                T_world_base = np.tile(IDENTITY_POSE, (num_frames, 1))
        elapsed = time.time() - start_time

        if not static_base:
            object_mesh = cast(trimesh.Trimesh, trimesh.load(data["object_mesh_path"], process=False, force="mesh"))
            object_vertices = np.asarray(object_mesh.vertices, dtype=np.float32)
            R_world_object = quaternion_to_matrix(data["object_quat_wxyz"][0]).astype(np.float32)
            object_world_vertices = object_vertices @ R_world_object.T + data["object_pos"][0]
            table_xy = np.concatenate([points[:, :, :2].reshape(-1, 2), data["object_pos"][:, :2]], axis=0)
            table_min_xy, table_max_xy = np.min(table_xy, axis=0) - 0.15, np.max(table_xy, axis=0) + 0.15
            table_z = float(np.min(object_world_vertices[:, 2]) - 0.002)

        if reload_epoch != self.reload_epoch:
            self.reloading = False
            self.reload()
            return

        with self.server.atomic():
            for handle in (self.urdf_vis, self.object_handle, self.table_handle, self.mano_handle):
                if handle is not None:
                    handle.remove()
            self.object_handle = self.table_handle = self.mano_handle = None
            self.urdf_vis = ViserUrdf(self.server, preset.urdf, root_node_name="/robot/urdf")
            if not static_base:
                self.table_handle = self.server.scene.add_mesh_simple(
                    "/table",
                    vertices=np.array(
                        [
                            [table_min_xy[0], table_min_xy[1], table_z],
                            [table_max_xy[0], table_min_xy[1], table_z],
                            [table_max_xy[0], table_max_xy[1], table_z],
                            [table_min_xy[0], table_max_xy[1], table_z],
                        ],
                        dtype=np.float32,
                    ),
                    faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
                    color=(180, 230, 190),
                    opacity=0.55,
                    side="double",
                )
                self.object_handle = self.server.scene.add_mesh_simple(
                    "/dexycb_object",
                    vertices=object_vertices,
                    faces=np.asarray(object_mesh.faces, dtype=np.int32),
                    color=(150, 210, 255),
                )
            self.target_frame.position = TARGET_OFFSET[self.motion]
            self.skeleton.visible = static_base
            self.data = data
            self.points = points
            self.solver_points = solver_points
            self.spec = spec
            self.root_index = spec.target_names.index(spec.root_target_name)
            self.contact_indices = [spec.target_names.index(n) for n in spec.contact_target_names]
            for idx, handle in enumerate(self.fingertip_handles):
                handle.visible = idx < len(self.contact_indices)
            if "mano_vertices" in data:
                self.mano_handle = self.server.scene.add_mesh_simple(
                    "/target_mano",
                    vertices=data["mano_vertices"][0],
                    faces=data["mano_faces"],
                    color=(230, 190, 150),
                    opacity=0.35,
                    side="double",
                )
            self.online = online
            self.q = q
            self.T_world_base = T_world_base
            self.num_frames = num_frames
            self.frame_idx = 0
            self.suppress_frame_update = True
            self.frame_slider.max = num_frames - 1
            self.frame_slider.value = 0
            self.suppress_frame_update = False
            self.status_text.value = (
                f"Loaded {rig_name} ({self.mode}, {self.motion}), {num_frames} frames, {elapsed:.1f}s"
            )
            self.update(0)
        self.reloading = False
        if reload_epoch != self.reload_epoch:
            self.reload()
            return
        self.playing.value = True
        print(
            f"[hand-retargeting-gallery] mode={self.mode} rig={rig_name} "
            f"motion={self.motion} sequence={capture_index + 1}/{len(self.sequence_names(self.motion))} "
            f"name={data['capture_name']} frames={num_frames} setup={elapsed:.2f}s"
        )

    def update(self, frame_idx: int) -> None:
        assert self.urdf_vis is not None
        if self.online is not None:
            packed = self.online.solve_numpy(self.solver_points[frame_idx])
            self.q[frame_idx] = packed[0, 7:]
            if self.spec.floating_base:
                self.T_world_base[frame_idx] = packed[0, :7]
        if self.mano_handle is not None:
            self.mano_handle.vertices = self.data["mano_vertices"][frame_idx]
        if self.skeleton.visible:
            self.skeleton.update(self.points[frame_idx])
        fingertips = self.points[frame_idx, self.contact_indices]
        for handle, fingertip in zip(self.fingertip_handles, fingertips):
            handle.position = fingertip
        self.robot_frame.position = self.T_world_base[frame_idx, :3]
        self.robot_frame.wxyz = self.T_world_base[frame_idx, 3:]
        self.urdf_vis.update_cfg(self.q[frame_idx])
        if self.object_handle is not None:
            self.object_handle.position = self.data["object_pos"][frame_idx]
            self.object_handle.wxyz = self.data["object_quat_wxyz"][frame_idx]

    def run(self) -> None:
        print(f"[hand-retargeting-gallery] Open: http://localhost:{self.server.get_port()}")
        while True:
            if self.playing.value and not self.reloading:
                self.frame_idx = (self.frame_idx + 1) % self.num_frames
                self.suppress_frame_update = True
                self.frame_slider.value = self.frame_idx
                self.suppress_frame_update = False
                self.update(self.frame_idx)
            time.sleep(1.0 / float(self.fps_slider.value))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, default="offline")
    parser.add_argument("--ee", choices=tuple(PRESETS), default="xhand")
    parser.add_argument("--base", choices=BASE_NAMES, default=FLOATING)
    parser.add_argument("--capture-index", type=int, default=0, help="Initial DexYCB capture index")
    parser.add_argument("--mano-root", type=str, default="")
    parser.add_argument(
        "--motion",
        choices=MOTIONS,
        default=WITH_WRIST,
        help="Whether the source motion drives the wrist; fingers-only replays a glove clip on a static base",
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--smoke-test", action="store_true", help="Initialize, draw one frame, and exit")
    args = parser.parse_args()
    if args.ee not in ee_options(args.mode):
        parser.error(f"{args.ee} has no {args.mode} preset")
    if args.motion == FINGERS_ONLY and args.base != FLOATING:
        parser.error("fingers-only runs on a floating base")

    wp.init()
    gallery = Gallery(args)
    if args.smoke_test:
        print("[hand-retargeting-gallery] smoke test complete")
        gallery.server.stop()
        return
    gallery.run()


if __name__ == "__main__":
    main()
