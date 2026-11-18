"""Any base (arm or humanoid) with any end effector, assembled at runtime and cached per asset revision. `BASES` and `EES` name the parts that can be mounted; `assemble(base, ee)` builds the URDF and merged collision spheres of one pair once and caches them under `ASSEMBLY_DIR`."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import yourdfpy

from robokit.assets import REVISION, ROBOKIT_HOME
from robokit.assets.robots.arms import franka_panda, ur5e, xarm6, xarm7, yam
from robokit.assets.robots.end_effectors import (
    ability_hand,
    allegro_hand,
    barrett_hand,
    dclaw_gripper,
    inspire_hand,
    leap_hand,
    mano_hand,
    orcahand,
    paxini_hand,
    robotiq_2f85,
    schunk_hand,
    sharpa_hand,
    xhand,
    yam_gripper,
)
from robokit.assets.robots.humanoids import unitree_g1, vega_1
from robokit.robo import Robot
from robokit.utils.assembly_utils import assemble_urdf, compute_mount_transform, merge_collision_spheres
from robokit.utils.ee_coord_utils import MountCoordinateSpec


ASSEMBLY_DIR = ROBOKIT_HOME / "assembly" / REVISION
"""Keyed by the asset revision, so an asset bump rebuilds every assembly instead of serving a stale one."""
EE_PREFIX = "ee_"
"""Every ee link and joint is renamed with this, so a hand root called "base" cannot collide with a base's own link."""


@dataclass(frozen=True)
class Part:
    """A base or an end effector on its own: its URDF, its flange, and its collision spheres."""

    urdf: Path
    mount_link: str
    mount_spec: MountCoordinateSpec
    spheres: Path


BASES = {
    "yam": Part(yam.URDF_PATH, yam.MOUNT_LINK_NAME, yam.MOUNT_COORD_SPEC, yam.COLLISION_SPHERE_PATH),
    "xarm7": Part(xarm7.URDF_GLB_PATH, xarm7.MOUNT_LINK_NAME, xarm7.MOUNT_COORD_SPEC, xarm7.COLLISION_SPHERE_PATH),
    "xarm6": Part(xarm6.URDF_PATH, xarm6.MOUNT_LINK_NAME, xarm6.MOUNT_COORD_SPEC, xarm6.COLLISION_SPHERE_PATH),
    "franka_panda": Part(
        franka_panda.ARM_URDF_PATH, franka_panda.ARM_MOUNT_LINK_NAME, franka_panda.ARM_MOUNT_COORD_SPEC,
        franka_panda.ARM_COLLISION_SPHERE_PATH,
    ),
    "ur5e": Part(ur5e.URDF_PATH, ur5e.MOUNT_LINK_NAME, ur5e.MOUNT_COORD_SPEC, ur5e.COLLISION_SPHERE_PATH),
    "g1": Part(
        unitree_g1.URDF_WITHOUT_HAND_PATH, unitree_g1.MOUNT_LINK_NAME_RIGHT, unitree_g1.MOUNT_COORD_SPEC_RIGHT,
        unitree_g1.COLLISION_SPHERE_PATH,
    ),
    "vega_1": Part(vega_1.URDF_PATH, vega_1.MOUNT_LINK_NAME_RIGHT, vega_1.MOUNT_COORD_SPEC_RIGHT, vega_1.COLLISION_SPHERE_PATH),
}  # fmt: skip

EES = {
    "xhand_right": Part(xhand.URDF_GLB_PATH_RIGHT, xhand.MOUNT_LINK_NAME_RIGHT, xhand.MOUNT_COORD_SPEC_RIGHT, xhand.COLLISION_SPHERE_PATH_RIGHT),
    "ability_right": Part(ability_hand.URDF_GLB_PATH_RIGHT, ability_hand.MOUNT_LINK_NAME_RIGHT, ability_hand.MOUNT_COORD_SPEC_RIGHT, ability_hand.COLLISION_SPHERE_PATH_RIGHT),
    "mano_right": Part(mano_hand.URDF_PATH_RIGHT, mano_hand.MOUNT_LINK_NAME_RIGHT, mano_hand.MOUNT_COORD_SPEC_RIGHT, mano_hand.COLLISION_SPHERE_PATH_RIGHT),
    "sharpa_right": Part(sharpa_hand.URDF_PATH_RIGHT, sharpa_hand.MOUNT_LINK_NAME_RIGHT, sharpa_hand.MOUNT_COORD_SPEC_RIGHT, sharpa_hand.COLLISION_SPHERE_PATH_RIGHT),
    "allegro_right": Part(allegro_hand.URDF_GLB_PATH_RIGHT, allegro_hand.MOUNT_LINK_NAME_RIGHT, allegro_hand.MOUNT_COORD_SPEC_RIGHT, allegro_hand.COLLISION_SPHERE_PATH_RIGHT),
    "dclaw_right": Part(dclaw_gripper.URDF_PATH, dclaw_gripper.MOUNT_LINK_NAME, dclaw_gripper.MOUNT_COORD_SPEC, dclaw_gripper.COLLISION_SPHERE_PATH),
    "paxini_right": Part(paxini_hand.URDF_MERGED_PATH_RIGHT, paxini_hand.MOUNT_LINK_NAME_RIGHT, paxini_hand.MOUNT_COORD_SPEC_RIGHT, paxini_hand.COLLISION_SPHERE_PATH_RIGHT),
    "inspire_right": Part(inspire_hand.URDF_PATH_RIGHT, inspire_hand.MOUNT_LINK_NAME_RIGHT, inspire_hand.MOUNT_COORD_SPEC_RIGHT, inspire_hand.COLLISION_SPHERE_PATH_RIGHT),
    "orca_right": Part(orcahand.URDF_PATH_RIGHT, orcahand.MOUNT_LINK_NAME_RIGHT, orcahand.MOUNT_COORD_SPEC_RIGHT, orcahand.COLLISION_SPHERE_PATH_RIGHT),
    "robotiq_2f85": Part(robotiq_2f85.URDF_PATH, robotiq_2f85.MOUNT_LINK_NAME, robotiq_2f85.MOUNT_COORD_SPEC, robotiq_2f85.COLLISION_SPHERE_PATH),
    "yam_gripper": Part(yam_gripper.URDF_PATH, yam_gripper.MOUNT_LINK_NAME, yam_gripper.MOUNT_COORD_SPEC, yam_gripper.COLLISION_SPHERE_PATH),
    "leap_right": Part(leap_hand.URDF_PATH_RIGHT, leap_hand.MOUNT_LINK_NAME_RIGHT, leap_hand.MOUNT_COORD_SPEC_RIGHT, leap_hand.COLLISION_SPHERE_PATH_RIGHT),
    "barrett_right": Part(barrett_hand.URDF_GLB_PATH, barrett_hand.MOUNT_LINK_NAME, barrett_hand.MOUNT_COORD_SPEC, barrett_hand.COLLISION_SPHERE_PATH),
    "schunk_right": Part(schunk_hand.URDF_GLB_PATH_RIGHT, schunk_hand.MOUNT_LINK_NAME_RIGHT, schunk_hand.MOUNT_COORD_SPEC_RIGHT, schunk_hand.COLLISION_SPHERE_PATH_RIGHT),
}  # fmt: skip


@lru_cache(maxsize=None)
def assemble(base: str, ee: str) -> Tuple[Path, Optional[Path], str]:
    """Assemble `base` + `ee` into a cached URDF plus merged collision spheres; returns them and the ee root link. The rotation comes from the two mount specs. The offset comes from their mount points; a part that declares none gets one from its mount-link mesh, pushed out so nothing sits behind the flange. The ee hangs off the base by its ROOT link, not its mount link: a floating grasp is a pose of the hand URDF's root, so the root is the link IK has to aim, and for the hands whose root sits above the mount link (xhand, ability, leap, orca, schunk) the two frames differ by a rotation, an offset, or both. Folding `T_root_mount` into the mount joint leaves the mount link exactly where the mount specs put it."""
    base_part, ee_part = BASES[base], EES[ee]
    out_dir = ASSEMBLY_DIR / f"{base}_{ee}"
    urdf = out_dir / f"{base}_{ee}.urdf"
    ee_urdf = yourdfpy.URDF.load(str(ee_part.urdf), load_meshes=False, build_scene_graph=True)
    root = str(ee_urdf.base_link)
    if not urdf.exists():
        meshes = []
        for part in (base_part, ee_part):
            if part.mount_spec.mount_point_xyz is not None:
                meshes.append(None)
                continue
            spec = Robot.load(part.urdf, load_meshes=True, mesh_dir=part.urdf.parent).spec
            scene = spec.link_visual_geometries[part.mount_link]
            meshes.append((scene if scene.geometry else spec.link_collision_geometries[part.mount_link]).to_mesh())
        T_mount_ee = compute_mount_transform(
            base_part.mount_spec,
            ee_part.mount_spec,
            auto_mount_point=True,
            base_mount_mesh=meshes[0],
            ee_mount_mesh=meshes[1],
        )
        T_root_mount = ee_urdf.get_transform(ee_part.mount_link)  # every root -> mount chain is fixed joints
        assemble_urdf(
            urdf,
            base_part.urdf,
            ee_part.urdf,
            base_part.mount_link,
            root,
            T_mount_ee @ np.linalg.inv(T_root_mount),
            EE_PREFIX,
        )
    spheres = merge_collision_spheres(
        out_dir / "collision_spheres.yaml", [(base_part.spheres, ""), (ee_part.spheres, EE_PREFIX)]
    )
    return urdf, spheres, root
