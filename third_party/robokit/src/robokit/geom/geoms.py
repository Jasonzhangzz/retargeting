# pyright: reportArgumentType=false
# pyright: reportOptionalMemberAccess=false
"""Typed geoms for `WarpScene`: triangle meshes, SDF volumes, analytical primitives."""

from typing import List, Literal, Optional, Tuple, Union, overload

import numpy as np
import trimesh
import warp as wp

from robokit.geom.sdf_kernels import (
    VolumeObservations,
    build_scene_per_point_kernel,
    clear_sdf_query_kernel,
    query_sdf_on_meshes_kernel,
    query_sdf_on_meshes_with_sdf_kernel,
    query_sdf_on_primitives_kernel,
    query_sdf_on_volumes_kernel,
)
from robokit.geom.sdf_volume import SdfVolume
from robokit.utils.warp_utils import wp_device_type
from robokit.xform.warp.transforms import inverse_tf_mat_kernel


Device = Optional[Union[str, object]]


MeshList = Union[List[wp.Mesh], List[trimesh.Trimesh], List[str]]
MAX_DIST = 1e6  # BVH search radius; also the "no hit" value every SDF query starts from
# mesh refine fires within this many voxels of the surface (trilinear error is sub-voxel)
REFINE_BAND_VOXELS = 4.0


def _to_wp_meshes(meshes: MeshList, device_wp: wp_device_type) -> List[wp.Mesh]:
    out: List[wp.Mesh] = []
    for m in meshes:
        if isinstance(m, wp.Mesh):
            out.append(m)
            continue
        if isinstance(m, str):
            m = trimesh.load(m, process=False, force="mesh")
        v = m.vertices.view(np.ndarray)
        f = m.faces.view(np.ndarray)
        out.append(
            wp.Mesh(
                points=wp.array(v, dtype=wp.vec3, device=device_wp),
                indices=wp.array(np.ravel(f), dtype=int, device=device_wp),
            )
        )
    return out


def _pad_params_to_vec4(cols: np.ndarray) -> np.ndarray:
    """Zero-pad an `(N, k<=4)` float array to the primitive kernel's `(N, 4)` vec4 payload."""
    params = np.zeros((cols.shape[0], 4), dtype=np.float32)
    params[:, : cols.shape[1]] = cols
    return params


class BaseGeom:
    """Base geom.

    Lifecycle:
        geom = BoxGeom(...)  # store raw inputs (CPU)
        scene.add(geom)      # calls geom.build(device): allocate capacity-padded GPU arrays
        geom.update(...)     # fixed buffers update in place; resource swaps advance graph_revision
    """

    TYPE_ID = -1  # primitive shape id; unused by MeshGeom/VolumeGeom
    # buffers only some geoms build; the rest stay None so a scene query reads any geom alike
    params_wp: Optional[wp.array] = None
    mesh_ids_wp: Optional[wp.array] = None
    volume_ids_wp: Optional[wp.array] = None
    aabb_min_wp: Optional[wp.array] = None
    aabb_max_wp: Optional[wp.array] = None
    paddings_wp: Optional[wp.array] = None

    def __init__(
        self,
        count: int,
        scene_offsets: np.ndarray,
        poses: Optional[wp.array],
        scales: Optional[wp.array],
        capacity: Optional[int],
    ):
        if count == 0:
            raise ValueError("Must pass at least one element.")
        self.count = count
        self.capacity = count if capacity is None else int(capacity)
        if self.capacity < count:
            raise ValueError(f"capacity {self.capacity} < element count {count}.")
        # CSR offsets, length num_scenes+1: scene s owns [offsets[s], offsets[s+1]); validated at build
        self._scene_offsets_np = np.asarray(scene_offsets, dtype=np.int32).reshape(-1)
        self.poses_wp = poses  # user's forward poses; kept live for the torch-grad path
        self.enable_inv_poses = poses is not None
        self.enable_scales = scales is not None
        self._scales = scales
        self._params_np: Optional[np.ndarray] = None  # set by the four shape classes
        self.device_wp: Optional[wp_device_type] = None  # set at build; None = not built yet
        self._buffers: List[Tuple[wp.array, wp.array, wp.array]] = []  # see query_sdf
        self._graph_revision = 0

    @property
    def graph_revision(self) -> int:
        """Revision of GPU resources referenced by captured graphs."""
        return self._graph_revision

    @property
    def kernel(self):
        return query_sdf_on_primitives_kernel

    @property
    def refine_band(self) -> float:
        return 0.0

    @property
    def launch_params(self) -> Tuple:
        return (self.TYPE_ID, self.params_wp)

    @overload
    def query_sdf(
        self,
        points: wp.array,
        scene_offsets: Optional[wp.array] = ...,
        scene_indices: Optional[wp.array] = ...,
        query_mask: Optional[wp.array] = ...,
        distance_only: Literal[False] = ...,
        out_signed_dists: Optional[wp.array] = ...,
        out_normals: Optional[wp.array] = ...,
        out_closest_points: Optional[wp.array] = ...,
        record_tape: bool = ...,
        out_normal_jacobians: Optional[wp.array] = ...,
        out_observed: Optional[wp.array] = ...,
    ) -> Tuple[wp.array, wp.array, wp.array]: ...
    @overload
    def query_sdf(
        self,
        points: wp.array,
        scene_offsets: Optional[wp.array] = ...,
        scene_indices: Optional[wp.array] = ...,
        query_mask: Optional[wp.array] = ...,
        *,
        distance_only: Literal[True],
        out_signed_dists: Optional[wp.array] = ...,
        out_normals: Optional[wp.array] = ...,
        out_closest_points: Optional[wp.array] = ...,
        record_tape: bool = ...,
        out_normal_jacobians: Optional[wp.array] = ...,
        out_observed: Optional[wp.array] = ...,
    ) -> wp.array: ...
    def query_sdf(
        self,
        points: wp.array,
        scene_offsets: Optional[wp.array] = None,
        scene_indices: Optional[wp.array] = None,
        query_mask: Optional[wp.array] = None,
        distance_only: bool = False,
        out_signed_dists: Optional[wp.array] = None,
        out_normals: Optional[wp.array] = None,
        out_closest_points: Optional[wp.array] = None,
        record_tape: bool = True,
        out_normal_jacobians: Optional[wp.array] = None,
        out_observed: Optional[wp.array] = None,
    ) -> Union[wp.array, Tuple[wp.array, wp.array, wp.array]]:
        """Query this geom alone (same semantics as `WarpScene.query_sdf`); call `build` first."""
        if self.device_wp is None:
            raise ValueError("geom must be built before query_sdf; call geom.build(device).")
        if (scene_offsets is None) == (scene_indices is None):
            raise ValueError("query_sdf requires exactly one of scene_offsets or scene_indices.")
        n = points.size
        if out_observed is not None:
            if out_observed.dtype != wp.bool or out_observed.size != n or out_observed.device != self.device_wp:
                raise ValueError("out_observed must be a bool array with one entry per point on the geometry device.")
            out_observed = out_observed.reshape((n,))
        if out_normal_jacobians is not None:
            if distance_only:
                raise ValueError("normal Jacobians require distance_only=False.")
            out_normal_jacobians = out_normal_jacobians.reshape((n,))
            out_normal_jacobians.zero_()
        if not self._buffers or self._buffers[-1][0].size < n:
            # keep outgrown buffers: graphs captured with them still write them on replay
            self._buffers.append(
                (
                    wp.empty((n,), dtype=wp.int32, device=self.device_wp),
                    wp.empty((n,), dtype=wp.vec3, device=self.device_wp),
                    wp.empty((n,), dtype=wp.int32, device=self.device_wp),
                )
            )
        scene_per_point, local_coords, element_indices = self._buffers[-1]

        # --- outputs ---
        # Flatten to (n,). Reuse caller-provided buffers or allocate new ones; distances start at MAX_DIST.
        if out_signed_dists is None:
            out_signed_dists = wp.empty((n,), dtype=wp.float32, device=self.device_wp)
        signed_dists = out_signed_dists.reshape((n,))
        wp.launch(
            clear_sdf_query_kernel,
            n,
            [signed_dists, out_observed, query_mask, MAX_DIST],
            device=self.device_wp,
            record_tape=record_tape,
        )
        if distance_only:
            # skip the normal/closest math (~2x faster)
            normals = closest_points = self._unused_vec3
        else:
            if out_normals is None:
                out_normals = wp.zeros((n,), dtype=wp.vec3, device=self.device_wp)
            if out_closest_points is None:
                out_closest_points = wp.zeros((n,), dtype=wp.vec3, device=self.device_wp)
            normals = out_normals.reshape((n,))
            closest_points = out_closest_points.reshape((n,))

        # --- point-to-scene mapping ---
        if scene_indices is not None:
            scene_per_point = scene_indices.reshape((n,))
        else:
            wp.launch(
                kernel=build_scene_per_point_kernel,
                dim=n,
                inputs=[scene_offsets, scene_per_point],
                device=self.device_wp,
                record_tape=record_tape,
            )

        self._launch(
            points.reshape((n,)),
            scene_per_point,
            query_mask if query_mask is not None else self._unused_int32,
            query_mask is not None,  # enable_query_mask
            not distance_only,  # enable_normals
            signed_dists,
            normals,
            closest_points,
            local_coords,
            element_indices,
            record_tape,
            out_normal_jacobians,
            out_observed,
        )

        if distance_only:
            return signed_dists
        return signed_dists, normals, closest_points

    def _launch(
        self,
        points: wp.array,
        scene_per_point: wp.array,
        query_mask: wp.array,
        enable_query_mask: bool,
        enable_normals: bool,
        out_signed_dists: wp.array,
        out_normals: wp.array,
        out_closest_points: wp.array,
        out_local_coords: wp.array,
        out_element_indices: wp.array,
        record_tape: bool = True,
        out_normal_jacobians: Optional[wp.array] = None,
        out_observed: Optional[wp.array] = None,
    ) -> None:
        """One kernel launch: min this geom's SDF into `out_signed_dists`, starting from its current value."""
        wp.launch(
            kernel=self.kernel,
            dim=points.size,
            inputs=[
                points,
                scene_per_point,
                *self.launch_params,
                self.scene_offsets_wp,
                self.inv_poses_wp,
                self.enable_inv_poses,
                self.scales_wp,
                self.enable_scales,
                query_mask,
                enable_query_mask,
                enable_normals,
                MAX_DIST,
                out_signed_dists,
                out_normals,
                out_closest_points,
                out_local_coords,
                out_element_indices,
                out_normal_jacobians is not None,
                out_normal_jacobians if out_normal_jacobians is not None else self._unused_mat33,
                *((out_observed,) if isinstance(self, VolumeGeom) else ()),
            ],
            device=self.device_wp,
            record_tape=record_tape,
        )

    def _pad_to(self, arr: wp.array) -> wp.array:
        """Copy `arr` into the front of a capacity-length buffer; the tail is never read."""
        if arr.shape[0] == self.capacity:
            return arr
        out = wp.empty((self.capacity,), dtype=arr.dtype, device=self.device_wp)
        wp.copy(out, arr, count=arr.shape[0])
        return out

    def _offsets_wp(self, scene_offsets: np.ndarray, count: int) -> wp.array:
        offsets = np.asarray(scene_offsets, dtype=np.int32).reshape(-1)
        if offsets.shape[0] != self.num_scenes + 1:
            raise ValueError(f"scene_offsets must have length {self.num_scenes + 1}, got {offsets.shape[0]}.")
        if int(offsets[-1]) != count:
            raise ValueError(f"scene_offsets[-1] must equal the element count {count}, got {int(offsets[-1])}.")
        return wp.from_numpy(offsets, dtype=wp.int32, device=self.device_wp)

    def build(self, device: Device = "cpu") -> "BaseGeom":
        """Allocate the GPU arrays; called by `WarpScene.add`, or directly for scene-less `query_sdf`."""
        device_wp = wp.get_device(str(device))
        self.device_wp = device_wp
        self.num_scenes = self._scene_offsets_np.shape[0] - 1
        # size-1 placeholders for kernel args whose feature flag is off (never read)
        self._unused_vec3 = wp.empty((1,), dtype=wp.vec3, device=device_wp)
        self._unused_mat33 = wp.empty((1,), dtype=wp.mat33, device=device_wp)
        self._unused_int32 = wp.empty((1,), dtype=wp.int32, device=device_wp)
        self.scene_offsets_wp = self._offsets_wp(self._scene_offsets_np, self.count)
        if self.poses_wp is not None:
            self.inv_poses_wp = wp.empty((self.capacity,), dtype=wp.mat44, device=device_wp)
            wp.launch(
                inverse_tf_mat_kernel, dim=self.count, inputs=[self.poses_wp, self.inv_poses_wp], device=device_wp
            )
        else:
            self.inv_poses_wp = wp.zeros((1,), dtype=wp.mat44, device=device_wp)  # disabled dummy
        # Keep capacity-sized scales (ones if unscaled) so updates copy in place without capture-time reallocations.
        self.scales_wp = wp.ones((self.capacity,), dtype=wp.float32, device=device_wp)
        if self._scales is not None:
            wp.copy(self.scales_wp, self._scales, count=self.count)
        if self._params_np is not None:
            self.params_wp = self._pad_to(wp.array(self._params_np, dtype=wp.vec4, device=device_wp))
        return self

    def _update_common(
        self,
        n: int,
        scene_offsets: Optional[np.ndarray],
        poses: Optional[wp.array],
        scales: Optional[wp.array] = None,
    ) -> None:
        if n > self.capacity:
            raise ValueError(f"update: {n} elements exceed capacity {self.capacity}.")
        if scene_offsets is None and n != self.count:
            raise ValueError(f"update: count changed ({self.count} -> {n}); pass scene_offsets.")
        if scene_offsets is not None:
            wp.copy(self.scene_offsets_wp, self._offsets_wp(scene_offsets, n))
            self.count = n
        if poses is not None:
            if not self.enable_inv_poses:
                raise ValueError("update: geom was constructed without poses; the update would be ignored.")
            if n > 0:
                wp.launch(inverse_tf_mat_kernel, dim=n, inputs=[poses, self.inv_poses_wp], device=self.device_wp)
            self.poses_wp = poses  # keep forward poses live so callers can read active centers
        if scales is not None:
            if not self.enable_scales:
                raise ValueError("update: geom was constructed without scales; the update would be ignored.")
            wp.copy(self.scales_wp, scales, count=n)

    def _update_params(
        self,
        params_np: Optional[np.ndarray],
        scene_offsets: Optional[np.ndarray],
        poses: Optional[wp.array],
        scales: Optional[wp.array] = None,
    ) -> None:
        """Shared updater of the four shape classes: swap packed vec4 params + common state."""
        n = self.count if params_np is None else params_np.shape[0]
        self._update_common(n, scene_offsets, poses, scales)
        if params_np is not None and n > 0:
            wp.copy(self.params_wp, wp.array(params_np, dtype=wp.vec4, device=self.device_wp), count=n)


class BoxGeom(BaseGeom):
    """N boxes: `half_extents (N, 3)`, posed/rotated via `poses`."""

    TYPE_ID = 0

    def __init__(
        self,
        half_extents: np.ndarray,
        scene_offsets: np.ndarray,
        poses: Optional[wp.array] = None,
        scales: Optional[wp.array] = None,
        capacity: Optional[int] = None,
    ):
        he = np.asarray(half_extents, dtype=np.float32).reshape(-1, 3)
        super().__init__(he.shape[0], scene_offsets, poses, scales, capacity)
        self._params_np = _pad_params_to_vec4(he)

    def update(
        self,
        half_extents: Optional[np.ndarray] = None,
        scene_offsets: Optional[np.ndarray] = None,
        poses: Optional[wp.array] = None,
        scales: Optional[wp.array] = None,
    ) -> None:
        p = (
            None
            if half_extents is None
            else _pad_params_to_vec4(np.asarray(half_extents, dtype=np.float32).reshape(-1, 3))
        )
        self._update_params(p, scene_offsets, poses, scales)


class SphereGeom(BaseGeom):
    """N spheres: `radii (N,)`, centered via `poses`."""

    TYPE_ID = 1

    def __init__(
        self,
        radii: np.ndarray,
        scene_offsets: np.ndarray,
        poses: Optional[wp.array] = None,
        scales: Optional[wp.array] = None,
        capacity: Optional[int] = None,
    ):
        r = np.asarray(radii, dtype=np.float32).reshape(-1, 1)
        super().__init__(r.shape[0], scene_offsets, poses, scales, capacity)
        self._params_np = _pad_params_to_vec4(r)

    def update(
        self,
        radii: Optional[np.ndarray] = None,
        scene_offsets: Optional[np.ndarray] = None,
        poses: Optional[wp.array] = None,
        scales: Optional[wp.array] = None,
    ) -> None:
        p = None if radii is None else _pad_params_to_vec4(np.asarray(radii, dtype=np.float32).reshape(-1, 1))
        self._update_params(p, scene_offsets, poses, scales)


class PlaneGeom(BaseGeom):
    """N halfspaces (local `z=0` planes, sdf = local z): count = `scene_offsets[-1]`, orientation via `poses`."""

    TYPE_ID = 2

    def __init__(
        self,
        scene_offsets: np.ndarray,
        poses: Optional[wp.array] = None,
        scales: Optional[wp.array] = None,
        capacity: Optional[int] = None,
    ):
        count = int(np.asarray(scene_offsets).reshape(-1)[-1])
        super().__init__(count, scene_offsets, poses, scales, capacity)
        self._params_np = np.zeros((count, 4), dtype=np.float32)

    def update(
        self,
        scene_offsets: Optional[np.ndarray] = None,
        poses: Optional[wp.array] = None,
        scales: Optional[wp.array] = None,
    ) -> None:
        n = self.count if scene_offsets is None else int(np.asarray(scene_offsets).reshape(-1)[-1])
        self._update_common(n, scene_offsets, poses, scales)


class CapsuleGeom(BaseGeom):
    """N capsules: `radii (N,)` + `half_heights (N,)`, axis along local z via `poses`."""

    TYPE_ID = 3

    def __init__(
        self,
        radii: np.ndarray,
        half_heights: np.ndarray,
        scene_offsets: np.ndarray,
        poses: Optional[wp.array] = None,
        scales: Optional[wp.array] = None,
        capacity: Optional[int] = None,
    ):
        cols = np.stack(
            [np.asarray(radii, dtype=np.float32).reshape(-1), np.asarray(half_heights, dtype=np.float32).reshape(-1)],
            axis=1,
        )
        super().__init__(cols.shape[0], scene_offsets, poses, scales, capacity)
        self._params_np = _pad_params_to_vec4(cols)

    def update(
        self,
        radii: Optional[np.ndarray] = None,
        half_heights: Optional[np.ndarray] = None,
        scene_offsets: Optional[np.ndarray] = None,
        poses: Optional[wp.array] = None,
        scales: Optional[wp.array] = None,
    ) -> None:
        if (radii is None) != (half_heights is None):
            raise ValueError("CapsuleGeom.update: pass radii and half_heights together.")
        p = None
        if radii is not None:
            cols = np.stack(
                [
                    np.asarray(radii, dtype=np.float32).reshape(-1),
                    np.asarray(half_heights, dtype=np.float32).reshape(-1),
                ],
                axis=1,
            )
            p = _pad_params_to_vec4(cols)
        self._update_params(p, scene_offsets, poses, scales)


class MeshGeom(BaseGeom):
    """N triangle meshes; `enable_sdf` bakes a disk-cached per-mesh SDF for far queries instead of the exact mesh."""

    def __init__(
        self,
        meshes: MeshList,
        scene_offsets: np.ndarray,
        poses: Optional[wp.array] = None,
        scales: Optional[wp.array] = None,
        enable_sdf: bool = False,
        sdf_voxel_size: float = 0.01,
        sdf_padding: float = 0.1,
        refine_band_voxels: float = REFINE_BAND_VOXELS,
        capacity: Optional[int] = None,
    ):
        super().__init__(len(meshes), scene_offsets, poses, scales, capacity)
        self._raw_meshes = meshes
        self.enable_sdf = enable_sdf
        self.sdf_voxel_size = sdf_voxel_size
        self.sdf_padding = sdf_padding
        self.refine_band_voxels = refine_band_voxels

    @property
    def kernel(self):
        return query_sdf_on_meshes_with_sdf_kernel if self.enable_sdf else query_sdf_on_meshes_kernel

    @property
    def launch_params(self) -> Tuple:
        if self.enable_sdf:
            return (
                self.mesh_ids_wp,
                self.aabb_min_wp,
                self.aabb_max_wp,
                self.volume_ids_wp,
                self.paddings_wp,
                self.refine_band,
            )
        return (self.mesh_ids_wp, self.aabb_min_wp, self.aabb_max_wp)

    @property
    def refine_band(self) -> float:
        return self.refine_band_voxels * self.sdf_voxel_size if self.enable_sdf else 0.0

    def _mesh_arrays(self, meshes: MeshList) -> Tuple[wp.array, ...]:
        """Build (ids, aabb_min, aabb_max[, volume_ids, paddings]); the SDF tier uses the padded volume AABBs."""
        wp_meshes = _to_wp_meshes(meshes, self.device_wp)
        self.meshes = tuple(wp_meshes)  # refs keep the GPU BVHs alive
        ids = wp.array([m.id for m in wp_meshes], dtype=wp.uint64, device=self.device_wp)
        if not self.enable_sdf:
            pts = [m.points.numpy() for m in wp_meshes]
            aabb_min_np = np.stack([p.min(axis=0) for p in pts])
            aabb_max_np = np.stack([p.max(axis=0) for p in pts])
            extra = ()
        else:
            volumes = [
                SdfVolume.from_mesh(m, self.sdf_voxel_size, self.sdf_padding, device=str(self.device_wp), cache=True)
                for m in meshes
            ]
            self.volumes = tuple(volumes)  # refs keep the GPU grids alive
            aabb_min_np = np.stack([v.aabb_min for v in volumes])
            aabb_max_np = np.stack([v.aabb_max for v in volumes])
            extra = (
                wp.array([v.volume.id for v in volumes], dtype=wp.uint64, device=self.device_wp),
                wp.array([float(v.padding) for v in volumes], dtype=wp.float32, device=self.device_wp),
            )
        aabb_min = wp.array(aabb_min_np.astype(np.float32), dtype=wp.vec3, device=self.device_wp)
        aabb_max = wp.array(aabb_max_np.astype(np.float32), dtype=wp.vec3, device=self.device_wp)
        return (ids, aabb_min, aabb_max) + extra

    def build(self, device: Device = "cpu") -> "MeshGeom":
        super().build(device)
        arrays = self._mesh_arrays(self._raw_meshes)
        self.mesh_ids_wp = self._pad_to(arrays[0])
        self.aabb_min_wp = self._pad_to(arrays[1])
        self.aabb_max_wp = self._pad_to(arrays[2])
        if self.enable_sdf:
            self.volume_ids_wp = self._pad_to(arrays[3])
            self.paddings_wp = self._pad_to(arrays[4])
        return self

    def update(
        self,
        meshes: Optional[MeshList] = None,
        scene_offsets: Optional[np.ndarray] = None,
        poses: Optional[wp.array] = None,
        scales: Optional[wp.array] = None,
    ) -> None:
        """`meshes` swaps the mesh ids + AABBs (and rebakes the SDF tier if enabled)."""
        n = self.count if meshes is None else len(meshes)
        self._update_common(n, scene_offsets, poses, scales)
        if meshes is not None and n > 0:
            if self.device_wp.is_cuda:
                wp.synchronize_device(self.device_wp)
            arrays = self._mesh_arrays(meshes)
            for dst, src in zip((self.mesh_ids_wp, self.aabb_min_wp, self.aabb_max_wp), arrays[:3]):
                wp.copy(dst, src, count=n)
            if self.enable_sdf:
                wp.copy(self.volume_ids_wp, arrays[3], count=n)
                wp.copy(self.paddings_wp, arrays[4], count=n)
            self._graph_revision += 1


class VolumeGeom(BaseGeom):
    """N pre-baked `SdfVolume` grids (NanoVDB trilinear sample; conservative extrapolation outside the AABB)."""

    def __init__(
        self,
        sdf_volumes: List[SdfVolume],
        scene_offsets: np.ndarray,
        poses: Optional[wp.array] = None,
        scales: Optional[wp.array] = None,
        capacity: Optional[int] = None,
    ):
        super().__init__(len(sdf_volumes), scene_offsets, poses, scales, capacity)
        self._raw_volumes = sdf_volumes

    @property
    def kernel(self):
        return query_sdf_on_volumes_kernel

    @property
    def launch_params(self) -> Tuple:
        return (self.volume_ids_wp, self.aabb_min_wp, self.aabb_max_wp, self.paddings_wp, self.observations_wp)

    def _volume_arrays(self, sdf_volumes: List[SdfVolume]) -> Tuple[wp.array, ...]:
        self.volumes = tuple(sdf_volumes)  # refs keep the GPU grids alive
        ids = wp.array([v.volume.id for v in sdf_volumes], dtype=wp.uint64, device=self.device_wp)
        aabb_min_np = np.stack([np.asarray(v.aabb_min, dtype=np.float32).reshape(3) for v in sdf_volumes])
        aabb_max_np = np.stack([np.asarray(v.aabb_max, dtype=np.float32).reshape(3) for v in sdf_volumes])
        aabb_min = wp.array(aabb_min_np, dtype=wp.vec3, device=self.device_wp)
        aabb_max = wp.array(aabb_max_np, dtype=wp.vec3, device=self.device_wp)
        paddings = wp.array([float(v.padding) for v in sdf_volumes], dtype=wp.float32, device=self.device_wp)
        observations = []
        for volume in sdf_volumes:
            data = VolumeObservations()
            data.values, data.unknown_is_occupied = volume._fusion, volume._unknown_is_occupied
            observations.append(data)
        return (
            ids,
            aabb_min,
            aabb_max,
            paddings,
            wp.array(observations, dtype=VolumeObservations, device=self.device_wp),
        )

    def build(self, device: Device = "cpu") -> "VolumeGeom":
        super().build(device)
        arrays = self._volume_arrays(self._raw_volumes)
        self.volume_ids_wp = self._pad_to(arrays[0])
        self.aabb_min_wp = self._pad_to(arrays[1])
        self.aabb_max_wp = self._pad_to(arrays[2])
        self.paddings_wp = self._pad_to(arrays[3])
        self.observations_wp = self._pad_to(arrays[4])
        return self

    def update(
        self,
        sdf_volumes: Optional[List[SdfVolume]] = None,
        scene_offsets: Optional[np.ndarray] = None,
        poses: Optional[wp.array] = None,
        scales: Optional[wp.array] = None,
    ) -> None:
        """`sdf_volumes` swaps ids/AABBs/paddings."""
        n = self.count if sdf_volumes is None else len(sdf_volumes)
        self._update_common(n, scene_offsets, poses, scales)
        if sdf_volumes is not None and n > 0:
            if self.device_wp.is_cuda:
                wp.synchronize_device(self.device_wp)
            arrays = self._volume_arrays(sdf_volumes)
            for dst, src in zip(
                (
                    self.volume_ids_wp,
                    self.aabb_min_wp,
                    self.aabb_max_wp,
                    self.paddings_wp,
                    self.observations_wp,
                ),
                arrays,
            ):
                wp.copy(dst, src, count=n)
            self._graph_revision += 1


__all__ = ["BaseGeom", "BoxGeom", "CapsuleGeom", "MeshGeom", "PlaneGeom", "SphereGeom", "VolumeGeom"]
