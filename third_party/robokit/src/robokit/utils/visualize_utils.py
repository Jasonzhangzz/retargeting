import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import trimesh
import viser
import warp as wp

from robokit.robo import Robot
from robokit.utils.warp_utils import wp_vec7


_HAND_SKELETON_EDGES = np.array(
    [(0, b) for b in (1, 5, 9, 13, 17)] + [(a, a + 1) for a in (1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19)]
)


@dataclass
class RetargetingPlayback:
    """Valid frames for robot, object, contact, and optional captured-hand playback."""

    qpos: np.ndarray
    fps: float = 30.0
    object_meshes: List[trimesh.Trimesh] = field(default_factory=list)
    T_world_object: Optional[np.ndarray] = None  # (frames, objects, 7)
    targets_world: Optional[np.ndarray] = None  # (frames, contacts, 3)
    contacts_world: Optional[np.ndarray] = None
    contact_mask: Optional[np.ndarray] = None  # (frames, contacts)
    human_vertices: Optional[np.ndarray] = None
    human_faces: Optional[np.ndarray] = None


def view_retargeting(robot: Robot, clips: Dict[str, Path], load_clip: Callable[[Path], RetargetingPlayback]) -> None:
    """Browse retargeted clips with the robot renderer; clips load when selected, FK runs on CPU, Ctrl-C quits.

    Example:
        >>> # xdoctest: +SKIP
        >>> from robokit.helpers.humanoid_retargeting.presets.g1_inspire_smplh import g1_inspire_smplh
        >>> robot = g1_inspire_smplh.load_robot(load_meshes=True)
        >>> qpos = np.zeros((3, 7 + robot.num_actuated_joints), np.float32)
        >>> qpos[:, 3] = 1.0
        >>> def load_clip(path: Path) -> RetargetingPlayback:
        ...     return RetargetingPlayback(qpos)
        >>> view_retargeting(robot, {"demo": Path("demo.npz")}, load_clip)
    """
    if not clips:
        print("No saved clips to visualize.")
        return
    server = viser.ViserServer()
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/ground", width=6, height=6)
    robot_viz = ViserBatchUrdf(server, robot, batch_size=1, root_node_name="/robot")
    selection = server.gui.add_dropdown("Clip", options=tuple(clips))
    playing = server.gui.add_checkbox("Play", True)
    frame_slider = server.gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0)
    show_contacts = server.gui.add_checkbox("Contacts", True)
    show_human = server.gui.add_checkbox("Captured hand", True, visible=False)
    info = server.gui.add_markdown("")
    current = ""
    clip: Optional[RetargetingPlayback] = None
    objects: List[viser.MeshHandle] = []
    targets: List[viser.IcosphereHandle] = []
    contacts: List[viser.IcosphereHandle] = []
    human: Optional[viser.MeshHandle] = None
    camera_center = np.array([0.0, 0.0, 1.0])
    num_dofs = robot.num_actuated_joints

    @server.on_client_connect
    def center_camera(client: viser.ClientHandle) -> None:
        # Moving the camera also moves look_at; set the target last.
        client.camera.position = camera_center + [2.0, -3.0, 1.5]
        client.camera.look_at = camera_center

    try:
        while True:
            if selection.value != current:
                selected = selection.value
                upcoming = load_clip(clips[selected])
                length = len(upcoming.qpos)
                if length < 1 or upcoming.qpos.ndim != 2 or upcoming.qpos.shape[1] < 7 + num_dofs:
                    raise ValueError(f"invalid qpos for {selected}")
                arrays = (
                    upcoming.T_world_object,
                    upcoming.targets_world,
                    upcoming.contacts_world,
                    upcoming.contact_mask,
                    upcoming.human_vertices,
                )
                if upcoming.fps <= 0 or any(array is not None and len(array) != length for array in arrays):
                    raise ValueError(f"inconsistent frame counts or FPS for {selected}")
                if upcoming.object_meshes and (
                    upcoming.T_world_object is None
                    or upcoming.T_world_object.shape != (length, len(upcoming.object_meshes), 7)
                ):
                    raise ValueError(f"inconsistent object tracks for {selected}")
                clip, current = upcoming, selected
                with server.atomic():
                    for handle in objects + targets + contacts + ([] if human is None else [human]):
                        handle.remove()
                    objects = [
                        server.scene.add_mesh_simple(f"/objects/{i}", mesh.vertices, mesh.faces, color=(230, 130, 40))
                        for i, mesh in enumerate(clip.object_meshes)
                    ]
                    targets = [
                        server.scene.add_icosphere(f"/contacts/target_{i}", radius=0.004, color=(40, 220, 80))
                        for i in range(0 if clip.targets_world is None else clip.targets_world.shape[1])
                    ]
                    contacts = [
                        server.scene.add_icosphere(f"/contacts/pad_{i}", radius=0.0028, color=(190, 50, 230))
                        for i in range(0 if clip.contacts_world is None else clip.contacts_world.shape[1])
                    ]
                    human = (
                        server.scene.add_mesh_simple(
                            "/human", clip.human_vertices[0], clip.human_faces, color=(120, 200, 130), opacity=0.35
                        )
                        if clip.human_vertices is not None and clip.human_faces is not None
                        else None
                    )
                    show_human.visible = human is not None
                    frame_slider.value = 0
                    frame_slider.max = max(length - 1, 1)
                    info.content = f"{length} frames · {clip.fps:g} FPS"
                    camera_center = clip.qpos[0, :3].astype(np.float64) + [0.0, 0.0, 0.5]
                    for client in server.get_clients().values():
                        center_camera(client)
            assert clip is not None
            frame = min(int(frame_slider.value), len(clip.qpos) - 1)
            with server.atomic(), wp.ScopedDevice("cpu"):
                robot_viz.update_cfg(clip.qpos[frame : frame + 1, 7 : 7 + num_dofs], clip.qpos[frame : frame + 1, :7])
                if clip.T_world_object is not None:
                    for i, handle in enumerate(objects):
                        handle.position = clip.T_world_object[frame, i, :3]
                        handle.wxyz = clip.T_world_object[frame, i, 3:]
                for handles, points in ((targets, clip.targets_world), (contacts, clip.contacts_world)):
                    if points is not None:
                        for i, handle in enumerate(handles):
                            handle.position = points[frame, i]
                            handle.visible = bool(
                                show_contacts.value and (clip.contact_mask is None or clip.contact_mask[frame, i])
                            )
                if human is not None and clip.human_vertices is not None:
                    human.vertices = clip.human_vertices[frame]
                    human.visible = show_human.value
            time.sleep(1.0 / clip.fps)
            if playing.value and selection.value == current:
                frame_slider.value = (int(frame_slider.value) + 1) % len(clip.qpos)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            server.stop()
        except RuntimeError:
            if not server.get_event_loop().is_closed():
                raise


class ViserHandSkeleton:
    """21-keypoint hand skeleton rendered as points and line segments."""

    def __init__(
        self,
        target: Union[viser.ViserServer, viser.ClientHandle],
        name: str = "/hand",
        color: Tuple[int, int, int] = (90, 200, 120),
        point_size: float = 0.006,
        position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        target.scene.add_frame(name, show_axes=False, position=position)
        keypoints = np.zeros((21, 3), dtype=np.float32)
        self._points = target.scene.add_point_cloud(
            f"{name}/keypoints", keypoints, np.full((21, 3), color, np.uint8), point_size=point_size
        )
        self._bones = target.scene.add_line_segments(
            f"{name}/bones", keypoints[_HAND_SKELETON_EDGES], color, thickness=3.0, thickness_units="screen"
        )

    def update(self, keypoints: np.ndarray):
        self._points.points = keypoints
        self._bones.points = keypoints[_HAND_SKELETON_EDGES]

    @property
    def visible(self) -> bool:
        return bool(self._points.visible)

    @visible.setter
    def visible(self, value: bool):
        self._points.visible = value
        self._bones.visible = value


class ViserBatchUrdf:
    """Render batched robot meshes with Viser and GPU forward kinematics."""

    def __init__(
        self,
        target: Union[viser.ViserServer, viser.ClientHandle],
        robot: Robot,
        batch_size: int,
        root_node_name: str = "/",
        scale: float = 1.0,
        mesh_color_override: Optional[Tuple[float, float, float]] = None,
        opacity: Optional[float] = None,
        flat_shading: bool = True,
    ):
        self._robot = robot
        self._batch_size = batch_size
        self._scale = scale
        self._target = target

        self._link_mesh_handles: List[viser.BatchedMeshHandle] = []
        self._link_indices: List[int] = []

        root = root_node_name.rstrip("/")
        identity_wxyzs = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (batch_size, 1))
        zero_positions = np.zeros((batch_size, 3), dtype=np.float32)

        color_kwargs = {}
        if mesh_color_override is not None:
            color_kwargs["batched_colors"] = np.array([int(c * 255) for c in mesh_color_override], dtype=np.uint8)

        for link_index, link_name in enumerate(robot.spec.link_names):
            if link_name not in robot.spec.link_visual_geometries:
                continue
            scene = robot.spec.link_visual_geometries[link_name]
            if len(scene.geometry) == 0:
                continue
            mesh = scene.to_mesh()
            if len(mesh.vertices) == 0:
                continue

            vertices = np.array(mesh.vertices, dtype=np.float32) * scale
            faces = np.array(mesh.faces, dtype=np.uint32)

            node_name = f"{root}/{link_name}" if root else f"/{link_name}"
            handle = target.scene.add_batched_meshes_simple(
                node_name,
                vertices=vertices,
                faces=faces,
                batched_wxyzs=identity_wxyzs,
                batched_positions=zero_positions,
                opacity=opacity,
                flat_shading=flat_shading,
                cast_shadow=False,  # batched meshes self-shadow -> acne; robots needn't cast/receive shadows
                receive_shadow=False,
                **color_kwargs,
            )
            self._link_mesh_handles.append(handle)
            self._link_indices.append(link_index)

    def update_cfg(
        self,
        configuration: np.ndarray,
        T_world_base: Optional[np.ndarray] = None,
    ):
        warp_T_world_base: Optional[wp.array] = None
        if T_world_base is not None:
            warp_T_world_base = wp.from_numpy(T_world_base.astype(np.float32), dtype=wp_vec7)

        state = self._robot.state(q=configuration, T_world_base=warp_T_world_base)
        state = self._robot.forward_kinematics(state)

        all_transforms = state.T_world_link.numpy()
        base_positions = state.T_world_base.numpy()[:, :3]

        for handle, link_index in zip(self._link_mesh_handles, self._link_indices):
            positions = base_positions + (all_transforms[:, link_index, :3] - base_positions) * self._scale
            wxyzs = all_transforms[:, link_index, 3:]
            handle.batched_positions = positions
            handle.batched_wxyzs = wxyzs

    def get_actuated_joint_names(self) -> Tuple[str, ...]:
        return tuple(self._robot.spec.actuated_joint_names)

    def get_actuated_joint_limits(self) -> Dict[str, Tuple[float, float]]:
        limits = self._robot.spec.actuated_joint_limits
        return {name: (float(lo), float(hi)) for name, (lo, hi) in zip(self._robot.spec.actuated_joint_names, limits)}

    def remove(self):
        for handle in self._link_mesh_handles:
            handle.remove()
        self._link_mesh_handles.clear()
        self._link_indices.clear()

    @property
    def show_visual(self) -> bool:
        return bool(self._link_mesh_handles and self._link_mesh_handles[0].visible)

    @show_visual.setter
    def show_visual(self, value: bool):
        for handle in self._link_mesh_handles:
            handle.visible = value


class ViserMjcf:
    """Render an MJCF model with MuJoCo and Viser."""

    def __init__(
        self,
        target: Union[viser.ViserServer, viser.ClientHandle],
        mjcf: str,
        root_node_name: str = "/",
        keep_world_link: bool = True,
    ):
        import mujoco  # type: ignore  # lazily imported; mujoco ships only in the `mj` extra

        self._model = (
            mujoco.MjModel.from_xml_path(mjcf) if os.path.isfile(mjcf) else mujoco.MjModel.from_xml_string(mjcf)
        )
        self._data = mujoco.MjData(self._model)
        mujoco.mj_forward(self._model, self._data)
        base_body = (
            0 if keep_world_link else next(b for b in range(1, self._model.nbody) if self._model.body_parentid[b] == 0)
        )
        self._t_source_base = self._data.xpos[base_body].copy()
        self._R_source_base = self._data.xmat[base_body].reshape(3, 3).copy()
        self._nodes: List[viser.GlbHandle] = []
        self._geom_ids: List[int] = []

        # actuated joints (hinge/slide), in model order -- the config vector update_cfg expects
        self._joint_names: List[str] = []
        self._joint_qposadr: List[int] = []
        self._joint_limits: List[Tuple[float, float]] = []
        self._joint_equalities = [
            eid
            for eid in range(self._model.neq)
            if int(self._model.eq_type[eid]) == int(mujoco.mjtEq.mjEQ_JOINT) and self._model.eq_active0[eid]
        ]
        dependent_joints = {int(self._model.eq_obj1id[eid]) for eid in self._joint_equalities}
        for jid in range(self._model.njnt):
            joint_type = int(self._model.jnt_type[jid])
            if jid in dependent_joints or joint_type not in (
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            ):
                continue
            self._joint_names.append(mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, jid))
            self._joint_qposadr.append(int(self._model.jnt_qposadr[jid]))
            if self._model.jnt_limited[jid]:
                limit = (float(self._model.jnt_range[jid, 0]), float(self._model.jnt_range[jid, 1]))
            elif joint_type == int(mujoco.mjtJoint.mjJNT_HINGE):
                limit = (-np.pi, np.pi)
            else:
                limit = (-1.0, 1.0)
            self._joint_limits.append(limit)

        # a visual geom never collides: no contype/conaffinity and no explicit contact pair
        paired = set(self._model.pair_geom1.tolist()) | set(self._model.pair_geom2.tolist())
        visual = [
            self._model.geom_contype[g] == 0 and self._model.geom_conaffinity[g] == 0 and g not in paired
            for g in range(self._model.ngeom)
        ]
        # show collision-only bodies as visuals so every body stays visible
        visual_bodies = {int(self._model.geom_bodyid[g]) for g in range(self._model.ngeom) if visual[g]}
        self._show_in_visual: List[bool] = []
        self._show_in_collision: List[bool] = []
        root = root_node_name.rstrip("/")
        for geom_idx in range(self._model.ngeom):
            if self._model.geom_type[geom_idx] == mujoco.mjtGeom.mjGEOM_PLANE:  # floors: callers add a grid
                continue
            is_visual = visual[geom_idx]
            mesh = self._geom_to_trimesh(mujoco, geom_idx)
            matid = int(self._model.geom_matid[geom_idx])  # menagerie colors come from materials, not geom_rgba
            rgba = self._model.mat_rgba[matid] if matid >= 0 else self._model.geom_rgba[geom_idx]
            mesh.visual.vertex_colors = (rgba * 255).astype(np.uint8)
            self._nodes.append(target.scene.add_mesh_trimesh(f"{root}/geom_{geom_idx}", mesh))
            self._geom_ids.append(geom_idx)
            no_visual = int(self._model.geom_bodyid[geom_idx]) not in visual_bodies  # collidable-only body
            self._show_in_visual.append(bool(is_visual or no_visual))
            self._show_in_collision.append(not bool(is_visual))

        self._visual_on, self._collision_on = True, False  # default: visual meshes only
        self._apply_visibility()
        self.update_cfg(np.zeros(len(self._joint_names)))

    def _apply_visibility(self):
        for node, sv, sc in zip(self._nodes, self._show_in_visual, self._show_in_collision):
            node.visible = (sv and self._visual_on) or (sc and self._collision_on)

    @property
    def show_visual(self) -> bool:
        return self._visual_on

    @show_visual.setter
    def show_visual(self, visible: bool):  # mirror viser's ViserUrdf property API
        self._visual_on = visible
        self._apply_visibility()

    @property
    def show_collision(self) -> bool:
        return self._collision_on

    @show_collision.setter
    def show_collision(self, visible: bool):
        self._collision_on = visible
        self._apply_visibility()

    def _geom_to_trimesh(self, mujoco, geom_idx: int) -> trimesh.Trimesh:
        geom_type = int(self._model.geom_type[geom_idx])
        size = self._model.geom_size[geom_idx].astype(np.float64)
        if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            return trimesh.creation.icosphere(radius=float(size[0]))
        if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
            return trimesh.creation.box(extents=2.0 * size[:3])
        if geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            return trimesh.creation.cylinder(radius=float(size[0]), height=2.0 * float(size[1]))
        if geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            return trimesh.creation.capsule(radius=float(size[0]), height=2.0 * float(size[1]))
        if geom_type == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
            mesh = trimesh.creation.icosphere(radius=1.0)
            mesh.apply_scale(size[:3])
            return mesh
        data_id = int(self._model.geom_dataid[geom_idx])  # mesh geom
        v0, vn = int(self._model.mesh_vertadr[data_id]), int(self._model.mesh_vertnum[data_id])
        f0, fn = int(self._model.mesh_faceadr[data_id]), int(self._model.mesh_facenum[data_id])
        vertices = np.asarray(self._model.mesh_vert[v0 : v0 + vn], dtype=np.float64).reshape(-1, 3)
        faces = np.asarray(self._model.mesh_face[f0 : f0 + fn], dtype=np.int64).reshape(-1, 3)
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    def update_cfg(self, configuration: np.ndarray):
        if configuration.shape != (len(self._joint_qposadr),):
            raise ValueError(f"Expected configuration shape {(len(self._joint_qposadr),)}, got {configuration.shape}.")
        for value, qposadr in zip(configuration, self._joint_qposadr):
            self._data.qpos[qposadr] = value
        for eid in self._joint_equalities:
            dependent_joint = int(self._model.eq_obj1id[eid])
            independent_joint = int(self._model.eq_obj2id[eid])
            dependent_qposadr = int(self._model.jnt_qposadr[dependent_joint])
            delta = 0.0
            if independent_joint >= 0:
                independent_qposadr = int(self._model.jnt_qposadr[independent_joint])
                delta = self._data.qpos[independent_qposadr] - self._model.qpos0[independent_qposadr]
            self._data.qpos[dependent_qposadr] = self._model.qpos0[
                dependent_qposadr
            ] + np.polynomial.polynomial.polyval(delta, self._model.eq_data[eid, :5])
        self.update_qpos(self._data.qpos)

    def update_qpos(self, qpos: np.ndarray):
        """Show the model at a full MuJoCo `qpos`, free and ball joints included."""
        import mujoco  # type: ignore

        self._data.qpos[:] = qpos
        mujoco.mj_forward(self._model, self._data)
        for geom_idx, node in zip(self._geom_ids, self._nodes):
            T = np.eye(4)
            T[:3, :3] = self._R_source_base.T @ self._data.geom_xmat[geom_idx].reshape(3, 3)
            node.position = self._R_source_base.T @ (self._data.geom_xpos[geom_idx] - self._t_source_base)
            node.wxyz = trimesh.transformations.quaternion_from_matrix(T)

    def get_actuated_joint_names(self) -> Tuple[str, ...]:
        return tuple(self._joint_names)

    def get_actuated_joint_limits(self) -> Dict[str, Tuple[float, float]]:
        return {name: limit for name, limit in zip(self._joint_names, self._joint_limits)}

    def remove(self):
        for node in self._nodes:
            node.remove()
        self._nodes.clear()
