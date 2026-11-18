"""Whole-body scene clearance: one IK per clip (collision, feet/hands pinned, rest pose); assumes G1 + Inspire."""

from typing import Dict, List

import numpy as np
import torch
import warp as wp
from scipy.spatial import cKDTree

from robokit.geom import WarpScene
from robokit.geom.geoms import MeshGeom
from robokit.helpers.full_body_retargeting.clip import FullBodyClip
from robokit.helpers.ik import IKSolver
from robokit.helpers.ik.config import IKContext, IKSolverConfig
from robokit.opt.lm_optimizer import LMOptimizerConfig
from robokit.opt.population_solver import PopulationSolverConfig, Stage
from robokit.opt.sampler import RobertsSamplerConfig
from robokit.robo import Robot
from robokit.smplx import SMPLX_JOINT_NAMES
from robokit.terms.dense.configuration_task import ConfigurationTask
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.position_task import PositionTask
from robokit.terms.dense.scene_collision_task import SceneCollisionTask
from robokit.xform.numpy import pose7_to_tf_mat, tf_mat_to_pose7


_SCENE_MARGIN = 0.005  # m, target clearance between core spheres and the scene
_SUPPORT_DIST = 0.03  # m, human-to-scene distance that counts as weight-bearing
_SUPPORT_FRAC = 0.5  # fraction of frames within _SUPPORT_DIST that marks a support link
_MAX_CORE_SPHERES = 240  # cap guarded spheres so the LM Jacobian fits GPU shared memory
# robot link name -> SMPL-X joint of its human region, for the weight-bearing gate (G1 naming)
_SIDED_SEGMENTS = {
    "hip": "hip",
    "knee": "knee",
    "ankle_pitch": "ankle",
    "ankle_roll": "foot",
    "shoulder": "shoulder",
    "elbow": "elbow",
    "wrist": "wrist",
}
_CENTRAL_SEGMENTS = {"pelvis": "pelvis", "waist": "pelvis", "torso": "spine2", "head": "head"}


def apply_scene_clearance(robot: Robot, clip: FullBodyClip, qpos: np.ndarray, device: str) -> np.ndarray:
    """Eject the core body from the scene it over-penetrates; see the module docstring."""
    scene = clip.scene
    if scene is None or not scene.names:
        return qpos
    num_frames = qpos.shape[0]
    spec = robot.spec
    num_dofs = spec.num_actuated_joints
    num_spheres = spec.local_collision_sphere_centers.shape[0]
    sphere_links = [spec.link_names[i] for i in spec.collision_spheres_link_indices]

    # human weight-bearing links (sitting, leaning) are left out of the guard, so support survives
    smplx_joint_index = {name: index for index, name in enumerate(SMPLX_JOINT_NAMES)}
    human_joints_world = clip.T_native_human[:num_frames, :, :3] * clip.scale
    trees = [cKDTree(np.asarray(scene.meshes[part].vertices, dtype=np.float32)) for part in range(len(scene.names))]
    R_world_part = [
        pose7_to_tf_mat(scene.T_world_part[part][:num_frames])[:, :3, :3] for part in range(len(scene.names))
    ]
    support: set = set()
    for link_name in set(sphere_links):
        side, _, rest = link_name.partition("_")
        segments = _SIDED_SEGMENTS if side in ("left", "right") else _CENTRAL_SEGMENTS
        joint = next(
            (v for k, v in segments.items() if (rest if side in ("left", "right") else link_name).startswith(k)), None
        )
        joint_index = (
            None if joint is None else smplx_joint_index.get(f"{side}_{joint}" if side in ("left", "right") else joint)
        )
        if joint_index is None:
            continue
        distance = np.full(num_frames, np.inf, dtype=np.float32)
        for part in range(len(scene.names)):
            t_world_part = scene.T_world_part[part][:num_frames, :3]
            distance = np.minimum(
                distance,
                trees[part].query(
                    np.einsum("tji,tj->ti", R_world_part[part], human_joints_world[:, joint_index] - t_world_part)
                )[0],
            )
        if float((distance < _SUPPORT_DIST).mean()) > _SUPPORT_FRAC:
            support.add(link_name)

    def is_core(link_name: str) -> bool:
        return (
            not link_name.startswith(("R_", "L_"))
            and all(k not in link_name for k in ("ankle", "wrist", "hand"))
            and link_name not in support
        )

    core = [i for i in range(num_spheres) if is_core(sphere_links[i])]
    if not core:
        return qpos
    if len(core) > _MAX_CORE_SPHERES:  # per-link subsample so every guarded link keeps representatives
        by_link: Dict[int, List[int]] = {}
        for i in core:
            by_link.setdefault(int(spec.collision_spheres_link_indices[i]), []).append(i)
        per_link = max(1, _MAX_CORE_SPHERES // len(by_link))
        core = [i for spheres in by_link.values() for i in spheres[:: max(1, len(spheres) // per_link)][:per_link]]

    # the whole scene, one BVH per part instanced at every frame pose; scene t owns query row t
    num_parts = len(scene.names)
    wp_meshes = [
        wp.Mesh(
            points=wp.array(np.asarray(mesh.vertices, np.float32), dtype=wp.vec3, device=device),
            indices=wp.array(np.asarray(mesh.faces, np.int32).reshape(-1), device=device),
        )
        for mesh in scene.meshes
    ]
    instance_meshes = [wp_meshes[part] for _ in range(num_frames) for part in range(num_parts)]
    # instance order is (frame, part): frame-major, matching scene_offsets = arange(frames + 1) * parts
    T_world_instance = pose7_to_tf_mat(scene.T_world_part[:, :num_frames].transpose(1, 0, 2).reshape(-1, 7))
    warp_scene = WarpScene(num_scenes=num_frames, device=device).add(
        MeshGeom(
            instance_meshes,
            scene_offsets=np.arange(num_frames + 1) * num_parts,
            poses=wp.array(T_world_instance, dtype=wp.mat44, device=device),
        )
    )

    hold_links = ["left_ankle_roll_link", "right_ankle_roll_link", "L_hand_base_link", "R_hand_base_link"]
    missing = [link_name for link_name in hold_links if link_name not in spec.link_names]
    if missing:
        raise ValueError(f"scene clearance assumes G1 + Inspire link names; missing {missing}")
    ik_cfg = (
        IKSolverConfig(
            base_init_sample_range=(0.0, 0.0, 0.0),
            enable_T_world_base=True,
            base_active_mask=(0.0,) * 6,
            solver=PopulationSolverConfig(
                stages=[Stage(20, optimizer=LMOptimizerConfig(lm_lambda=1.0))], cuda_graph_mode="none"
            ),
            sampler=RobertsSamplerConfig(keep_first=True, scale=0.0),
        )
        .add(
            PositionTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=400.0)
        )  # feet + hands are functional contacts: pin hard
        .add(
            SceneCollisionTask(
                scene=warp_scene,
                sphere_indices=core,
                weight=600.0,
                margin=_SCENE_MARGIN,
                scene_indices=wp.from_numpy(np.arange(num_frames, dtype=np.int32), dtype=wp.int32, device=device),
            )
        )
        .add(ConfigurationTask(robot, weight=15.0, base_weight=15.0))  # regularizes toward rest_q below
        .add(PositionLimit(weight=50.0))
    )
    ik = IKSolver(ik_cfg, robot=robot, target_links=hold_links, device=device)
    finger_dofs = {i for i, name in enumerate(spec.actuated_joint_names) if name.startswith(("R_", "L_"))}
    ik.set_active_joint_mask([0.0 if i in finger_dofs else 1.0 for i in range(num_dofs)])

    T_world_base = pose7_to_tf_mat(qpos[:, :7]).astype(np.float32)
    T_world_link = robot.forward_kinematics_via_matrix_numpy(qpos[:, 7 : 7 + num_dofs], T_world_base, device=device)
    T_world_hold = T_world_link[:, [spec.link_names.index(link_name) for link_name in hold_links]]
    q_current = torch.from_numpy(qpos[:, 7 : 7 + num_dofs].astype(np.float32)).to(device)
    T_world_base_current = torch.from_numpy(T_world_base).to(device)
    snap = ik.solve_torch(
        torch.from_numpy(T_world_hold.astype(np.float32)).to(device),
        init_q=q_current,
        init_T_world_base=T_world_base_current,
        rest_q=q_current,
        rest_T_world_base=T_world_base_current,  # ConfigurationTask holds the current pose, not the default rest
    )
    qpos = qpos.copy()
    qpos[:, 7 : 7 + num_dofs] = snap.q.cpu().numpy()
    qpos[:, :7] = tf_mat_to_pose7(snap.T_world_base.cpu().numpy())
    return qpos


__all__ = ["apply_scene_clearance"]
