# pyright: reportOperatorIssue=false
# pyright: reportArgumentType=false
# pyright: reportGeneralTypeIssues=false
"""Mesh, point clouds, and calibrated depth -> signed collision volumes."""

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import trimesh
import warp as wp

from robokit.xform.warp.transforms import inverse_tf_mat_func


SDF_CACHE_DIR = Path.home() / ".robokit" / "sdf_cache"


# --- volume ----------------------------------------------------------------
@dataclass
class SdfVolume:
    """Signed-distance grid with mesh, XYZ, and calibrated-depth builders. XYZ updates add occupied cells. Depth fuses observations and may clear old samples along observed rays. Both refresh the same collision grid in place. Observation weights determine known/unknown status. Each update sets the unknown-space policy.

    Example:
        >>> # xdoctest: +REQUIRES(env:ROBOKIT_CUDA_DOCS)
        >>> sdf = SdfVolume.from_points(np.array([[0.5, 0.5, 0.5]]), np.zeros(3), np.ones(3))
        >>> sdf.update(points=np.array([[0.6, 0.5, 0.5]])) is sdf
        True
    """

    volume: wp.Volume
    aabb_min: np.ndarray
    aabb_max: np.ndarray
    padding: float = 0.0
    _fusion: Optional[wp.array] = field(default=None, init=False, repr=False)
    _unknown_is_occupied: Optional[wp.array] = field(default=None, init=False, repr=False)

    # --- persistence ---
    def save(self, path_stem: str) -> None:
        """Save `.nvdb` + `.json`, plus `.fusion.npy` when observations are present."""
        self.volume.save_to_nvdb(f"{path_stem}.nvdb", codec="zip")
        if self._fusion is not None:
            np.save(f"{path_stem}.fusion.npy", self._fusion.numpy())
        meta = dict(
            aabb_min=np.asarray(self.aabb_min).tolist(),
            aabb_max=np.asarray(self.aabb_max).tolist(),
            padding=float(self.padding),
            fusion=self._fusion is not None,
            unknown_is_occupied=bool(self._unknown_is_occupied.numpy()[0])
            if self._unknown_is_occupied is not None
            else False,
        )
        Path(f"{path_stem}.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, path_stem: str, device: str = "cuda:0") -> "SdfVolume":
        """Load the collision grid and optional fusion history."""
        meta = json.loads(Path(f"{path_stem}.json").read_text())
        with open(f"{path_stem}.nvdb", "rb") as f:
            volume = wp.Volume.load_from_nvdb(f, device=device)
        result = cls(
            volume,
            np.asarray(meta["aabb_min"], np.float32),
            np.asarray(meta["aabb_max"], np.float32),
            float(meta["padding"]),
        )
        if meta.get("fusion", False):
            result._fusion = wp.array(np.load(f"{path_stem}.fusion.npy"), dtype=wp.vec2, device=device)
            result._unknown_is_occupied = wp.full(
                1, bool(meta.get("unknown_is_occupied", False)), dtype=wp.bool, device=device
            )
        return result

    # --- construction ---
    @classmethod
    def from_mesh(
        cls,
        mesh: Union[str, trimesh.Trimesh, wp.Mesh],
        voxel_size: float,
        padding: float = 0.1,
        max_dist: float = 1e6,
        device: str = "cuda:0",
        cache: bool = False,
    ) -> "SdfVolume":
        """Bake a mesh, optionally using a content-addressed disk cache.

        Example:
            >>> # xdoctest: +REQUIRES(env:ROBOKIT_CUDA_DOCS)
            >>> sdf = SdfVolume.from_mesh(trimesh.creation.box(), 0.1)
            >>> sdf.padding
            0.1
        """
        if cache:
            if isinstance(mesh, str):
                raw = Path(mesh).read_bytes()
            elif isinstance(mesh, wp.Mesh):
                raw = mesh.points.numpy().tobytes() + mesh.indices.numpy().tobytes()
            else:
                raw = mesh.vertices.view(np.ndarray).tobytes() + mesh.faces.view(np.ndarray).tobytes()
            key = f"{hashlib.sha1(raw).hexdigest()}_vox{_float_token(voxel_size)}_pad{_float_token(padding)}_v1"
            stem = SDF_CACHE_DIR / key
            if Path(f"{stem}.json").exists():
                return cls.load(str(stem), device)
        if isinstance(mesh, str):
            mesh = trimesh.load(mesh, process=False, force="mesh")  # pyright: ignore[reportAssignmentType]
        if isinstance(mesh, wp.Mesh):
            device_wp = mesh.points.device
            points_np = mesh.points.numpy().astype(np.float32)
            bbox_min = points_np.min(axis=0)
            bbox_max = points_np.max(axis=0)
        else:
            device_wp = wp.get_device(device)
            bbox_min = mesh.bounds[0].astype(np.float32)
            bbox_max = mesh.bounds[1].astype(np.float32)
            mesh = wp.Mesh(
                points=wp.array(mesh.vertices.view(np.ndarray), dtype=wp.vec3, device=device_wp),
                indices=wp.array(np.ravel(mesh.faces.view(np.ndarray)), dtype=int, device=device_wp),
            )
        padded_min = bbox_min - padding
        padded_max = bbox_max + padding
        extent = padded_max - padded_min

        ni = int(np.ceil(extent[0] / voxel_size)) + 1
        nj = int(np.ceil(extent[1] / voxel_size)) + 1
        nk = int(np.ceil(extent[2] / voxel_size)) + 1

        volume = wp.Volume.allocate(
            [0, 0, 0],
            [ni - 1, nj - 1, nk - 1],
            voxel_size,
            bg_value=float(max_dist),
            translation=tuple(padded_min.tolist()),
            device=device_wp,
        )
        wp.launch(
            kernel=_sdf_grid_kernel,
            dim=(ni, nj, nk),
            inputs=[
                mesh.id,
                wp.vec3(float(padded_min[0]), float(padded_min[1]), float(padded_min[2])),
                voxel_size,
                max_dist,
                volume.id,
            ],
            device=device_wp,
        )
        result = cls(volume=volume, aabb_min=padded_min.copy(), aabb_max=padded_max.copy(), padding=float(padding))
        if cache:
            SDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            result.save(str(stem))
        return result

    @classmethod
    def from_points(
        cls,
        points: Union[np.ndarray, wp.array],
        workspace_min: np.ndarray,
        workspace_max: np.ndarray,
        voxel_size: float = 0.02,
        device: str = "cuda:0",
        unknown_is_occupied: bool = False,
    ) -> "SdfVolume":
        """Voxelize XYZ samples; workspace bounds are rounded outward to voxel centers.

        Example:
            >>> # xdoctest: +REQUIRES(env:ROBOKIT_CUDA_DOCS)
            >>> sdf = SdfVolume.from_points(np.empty((0, 3)), np.zeros(3), np.ones(3), 0.1)
            >>> np.allclose(sdf.volume.get_voxel_size(), 0.1)
            True
        """
        workspace_min, workspace_max = np.asarray(workspace_min, np.float32), np.asarray(workspace_max, np.float32)
        if (
            workspace_min.shape != (3,)
            or workspace_max.shape != (3,)
            or not np.isfinite([workspace_min, workspace_max]).all()
            or not np.all(workspace_max > workspace_min)
            or not np.isfinite(voxel_size)
            or voxel_size <= 0
        ):
            raise ValueError("Expected finite increasing workspace bounds and positive voxel_size.")
        dims = tuple((np.ceil((workspace_max - workspace_min) / voxel_size).astype(int) + 1).tolist())
        volume = wp.Volume.allocate(
            [0, 0, 0],
            [d - 1 for d in dims],
            voxel_size,
            bg_value=1e6,
            translation=tuple(workspace_min.tolist()),
            device=device,
        )
        result = cls(volume, workspace_min.copy(), workspace_min + (np.asarray(dims, np.float32) - 1) * voxel_size)
        result._fusion = wp.zeros(dims, dtype=wp.vec2, device=device)
        result._unknown_is_occupied = wp.zeros(1, dtype=wp.bool, device=device)
        return result.update(points=points, unknown_is_occupied=unknown_is_occupied)

    @classmethod
    def from_depth(
        cls,
        depth: Union[np.ndarray, wp.array],
        intrinsics: Union[np.ndarray, wp.mat33],
        T_world_camera: Union[np.ndarray, wp.mat44],
        workspace_min: np.ndarray,
        workspace_max: np.ndarray,
        voxel_size: float = 0.02,
        device: str = "cuda:0",
        trunc: Optional[float] = None,
        max_weight: float = 30.0,
        depth_max: float = 3.0,
        unknown_is_occupied: bool = False,
    ) -> "SdfVolume":
        """Fuse metric camera z-depth with pinhole intrinsics and world-from-camera pose.

        Example:
            >>> # xdoctest: +REQUIRES(env:ROBOKIT_CUDA_DOCS)
            >>> sdf = SdfVolume.from_depth(np.ones((3, 3)), np.eye(3), np.eye(4), np.zeros(3), np.ones(3))
            >>> sdf.update(depth=np.ones((3, 3)), intrinsics=np.eye(3), T_world_camera=np.eye(4)) is sdf
            True
        """
        result = cls.from_points(np.empty((0, 3)), workspace_min, workspace_max, voxel_size, device)
        return result.update(
            depth=depth,
            intrinsics=intrinsics,
            T_world_camera=T_world_camera,
            trunc=trunc,
            max_weight=max_weight,
            depth_max=depth_max,
            unknown_is_occupied=unknown_is_occupied,
        )

    # --- updates ---
    def update(
        self,
        points: Optional[Union[np.ndarray, wp.array]] = None,
        *,
        depth: Optional[Union[np.ndarray, wp.array]] = None,
        intrinsics: Optional[Union[np.ndarray, wp.mat33]] = None,
        T_world_camera: Optional[Union[np.ndarray, wp.mat44]] = None,
        reset: bool = False,
        trunc: Optional[float] = None,
        max_weight: float = 30.0,
        depth_max: float = 3.0,
        unknown_is_occupied: bool = False,
    ) -> "SdfVolume":
        """Fuse XYZ and/or depth into the existing collision grid on device.

        Lifecycle:
            1. Prepare inputs: validate parameters and array metadata; upload NumPy inputs if supplied.
            2. Reset observations: clear fusion history when `reset=True`.
            3. Fuse depth: average calibrated projective distances; observed free rays may clear old obstacles.
            4. Mark points: add occupied cells after depth fusion, preserving unseen observations.
            5. Refresh collision grid: classify observations and unknown space, then write the signed ESDF in place using approximate jump flooding with two refinement passes.

        Args:
            points: Optional `(N, 3)` NumPy XYZ samples or device Warp `vec3` array; invalid/outside points are ignored.
            depth: Optional metric camera z-depth image, NumPy or device Warp 2D `float32` array.
            intrinsics: Pinhole camera matrix, NumPy `(3, 3)` or `wp.mat33`; required with depth.
            T_world_camera: Rigid world-from-camera transform, NumPy `(4, 4)` or `wp.mat44`; required with depth.
            reset: Forget observations before applying these inputs.
            trunc: TSDF truncation distance in meters; defaults to twice the voxel size.
            max_weight: Depth fusion weight cap.
            depth_max: Exclusive maximum valid depth in meters.
            unknown_is_occupied: Treat unobserved cells and space outside the map as occupied until the next update.

        Returns:
            `self`, retaining bounds, resolution and volume ID so captured collision queries remain valid. Options apply only to this call; an update without inputs refreshes the retained observations.

        Raises:
            ValueError: The volume lacks fusion history, or parameters or input array metadata are invalid.
        """
        # --- prepare inputs ---
        if self._fusion is None:
            raise ValueError("Updates require a volume created from points or depth.")
        voxel_size = float(self.volume.get_voxel_size()[0])
        trunc = float(trunc if trunc is not None else 2 * voxel_size)
        if not all(math.isfinite(v) and v > 0 for v in (trunc, max_weight, depth_max)):
            raise ValueError("Expected positive finite trunc, max_weight, depth_max.")
        device = self.volume.device
        if points is not None:
            if not isinstance(points, wp.array):
                if points.ndim != 2 or points.shape[1] != 3:
                    raise ValueError("Expected (N, 3) XYZ points.")
                points = wp.array(points, dtype=wp.vec3, device=device)
            if points.ndim != 1 or points.dtype != wp.vec3 or points.device != device:
                raise ValueError("Expected XYZ points as a vec3 array on the volume device.")
        if depth is not None:
            if not isinstance(depth, wp.array):
                depth = wp.array(depth, dtype=wp.float32, device=device)
            if (
                depth.ndim != 2
                or not depth.size
                or depth.dtype != wp.float32
                or depth.device != device
                or intrinsics is None
                or T_world_camera is None
            ):
                raise ValueError("Expected depth, pinhole intrinsics, and rigid T_world_camera.")
            if not isinstance(intrinsics, wp.mat33):
                intrinsics = wp.mat33(intrinsics)  # pyright: ignore[reportCallIssue]
            if not isinstance(T_world_camera, wp.mat44):
                T_world_camera = wp.mat44(T_world_camera)  # pyright: ignore[reportCallIssue]
        # --- reset observations ---
        if reset:
            self._fusion.zero_()

        # --- fuse depth ---
        if depth is not None:
            assert intrinsics is not None and T_world_camera is not None
            wp.launch(
                _integrate_tsdf_kernel,
                dim=self._fusion.shape,
                inputs=[
                    self.volume.id,
                    depth,
                    intrinsics,
                    T_world_camera,
                    trunc,
                    float(depth_max),
                    float(max_weight),
                    self._fusion,
                ],
                device=device,
            )
        # --- mark points ---
        if points is not None and points.size:
            wp.launch(
                _mark_points_kernel,
                dim=points.size,
                inputs=[self.volume.id, points, voxel_size, self._fusion],
                device=device,
            )
        # --- refresh collision grid ---
        _distance_transform(self._fusion, voxel_size, unknown_is_occupied, self.volume.id)
        # Keep the policy on device so captured queries see changes without rebuilding the scene.
        assert self._unknown_is_occupied is not None
        self._unknown_is_occupied.fill_(unknown_is_occupied)
        return self


# --- internals -------------------------------------------------------------
def _float_token(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def _distance_transform(
    observations: wp.array, voxel_size: float, unknown_is_occupied: bool = False, volume_id: int = 0
) -> wp.array:
    """Device-only signed JFA+2 from fusion observations."""
    if observations.ndim != 3 or min(observations.shape) < 2 or not math.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("Expected a 3D grid with each dimension >= 2 and a positive finite voxel_size.")
    device, shape = observations.device, observations.shape
    sites = wp.empty(shape, dtype=wp.vec3i, device=device)
    scratch = wp.empty_like(sites)
    result = wp.empty(shape, dtype=wp.float32, device=device)
    jumps = []
    jump = 1
    while jump < max(shape):
        jump *= 2
    while jump > 1:
        jump //= 2
        jumps.append(jump)
    for seed in (1, 0):
        for jump in [0] + jumps + [2, 1]:
            wp.launch(
                _jfa_step_kernel, shape, [observations, unknown_is_occupied, seed, jump, sites, scratch], device=device
            )
            sites, scratch = scratch, sites
        wp.launch(
            _jfa_finish_kernel,
            shape,
            [sites, observations, unknown_is_occupied, seed, voxel_size, volume_id, result],
            device=device,
        )
    return result


# --- kernels ---------------------------------------------------------------
@wp.kernel
def _sdf_grid_kernel(
    mesh_id: wp.uint64,
    min_world: wp.vec3,
    voxel_size: float,
    max_dist: float,
    volume_id: wp.uint64,
):
    i, j, k = wp.tid()  # pyright: ignore[reportAssignmentType]
    pos = min_world + wp.vec3(float(i), float(j), float(k)) * voxel_size
    query = wp.mesh_query_point(mesh_id, pos, max_dist)
    distance = max_dist
    if query.result:
        closest = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
        distance = wp.length(closest - pos) * query.sign
    wp.volume_store_f(volume_id, i, j, k, distance)


@wp.kernel
def _jfa_step_kernel(
    observations: wp.array3d(dtype=wp.vec2),
    unknown_is_occupied: bool,
    seed: int,
    step: int,
    sites: wp.array3d(dtype=wp.vec3i),
    output: wp.array3d(dtype=wp.vec3i),
):
    i, j, k = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    if step == 0:
        value = observations[i, j, k]
        occupied = value[0] < 0.0 or (unknown_is_occupied and value[1] == 0.0)
        output[i, j, k] = wp.where(int(occupied) == seed, wp.vec3i(i, j, k), wp.vec3i(-1, -1, -1))  # pyright: ignore[reportCallIssue]
        return
    best = sites[i, j, k]
    best_distance = float(1e20)
    if best[0] >= 0:
        delta = wp.vec3(float(best[0] - i), float(best[1] - j), float(best[2] - k))
        best_distance = wp.dot(delta, delta)  # pyright: ignore[reportCallIssue, reportArgumentType]
    for di in range(-1, 2):
        for dj in range(-1, 2):
            for dk in range(-1, 2):
                x, y, z = i + di * step, j + dj * step, k + dk * step
                if x < 0 or y < 0 or z < 0 or x >= sites.shape[0] or y >= sites.shape[1] or z >= sites.shape[2]:
                    continue
                candidate = sites[x, y, z]
                if candidate[0] >= 0:
                    delta = wp.vec3(float(candidate[0] - i), float(candidate[1] - j), float(candidate[2] - k))
                    distance = wp.dot(delta, delta)  # pyright: ignore[reportCallIssue, reportArgumentType]
                    if distance < best_distance:
                        best, best_distance = candidate, distance
    output[i, j, k] = best


@wp.kernel
def _jfa_finish_kernel(
    sites: wp.array3d(dtype=wp.vec3i),
    observations: wp.array3d(dtype=wp.vec2),
    unknown_is_occupied: bool,
    seed: int,
    voxel_size: float,
    volume_id: wp.uint64,
    output: wp.array3d(dtype=wp.float32),
):
    i, j, k = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    value = observations[i, j, k]
    occupied = value[0] < 0.0 or (unknown_is_occupied and value[1] == 0.0)
    if int(occupied) != seed:
        site = sites[i, j, k]
        distance = wp.length(wp.vec3(float(sites.shape[0]), float(sites.shape[1]), float(sites.shape[2])))  # pyright: ignore[reportCallIssue]
        if site[0] >= 0:
            delta = wp.vec3(float(site[0] - i), float(site[1] - j), float(site[2] - k))
            distance = wp.length(delta) - 0.5  # pyright: ignore[reportCallIssue]
        signed = distance * voxel_size * wp.where(seed == 1, 1.0, -1.0)  # pyright: ignore[reportCallIssue]
        output[i, j, k] = signed
        if volume_id != wp.uint64(0):
            wp.volume_store_f(volume_id, i, j, k, signed)


@wp.kernel
def _mark_points_kernel(
    volume_id: wp.uint64, points: wp.array(dtype=wp.vec3), voxel_size: float, fusion: wp.array3d(dtype=wp.vec2)
):
    point = points[wp.tid()]
    if not wp.isfinite(point[0]) or not wp.isfinite(point[1]) or not wp.isfinite(point[2]):
        return
    minimum = wp.volume_index_to_world(volume_id, wp.vec3(0.0))  # pyright: ignore[reportCallIssue]
    maximum = wp.volume_index_to_world(  # pyright: ignore[reportCallIssue]
        volume_id, wp.vec3(float(fusion.shape[0] - 1), float(fusion.shape[1] - 1), float(fusion.shape[2] - 1))
    )
    if wp.min(point - minimum) < 0.0 or wp.max(point - maximum) > 0.0:
        return
    index = wp.volume_world_to_index(volume_id, point)
    i, j, k = int(wp.rint(index[0])), int(wp.rint(index[1])), int(wp.rint(index[2]))  # pyright: ignore[reportIndexIssue]
    previous = fusion[i, j, k]
    fusion[i, j, k] = wp.vec2(-0.5 * voxel_size, wp.max(previous[1], 1.0))


@wp.kernel
def _integrate_tsdf_kernel(
    volume_id: wp.uint64,
    depth: wp.array2d(dtype=wp.float32),
    intrinsics: wp.mat33,
    T_world_camera: wp.mat44,
    trunc: float,
    depth_max: float,
    max_weight: float,
    fusion: wp.array3d(dtype=wp.vec2),
):
    i, j, k = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    world = wp.volume_index_to_world(volume_id, wp.vec3(float(i), float(j), float(k)))  # pyright: ignore[reportCallIssue]
    camera = wp.transform_point(inverse_tf_mat_func(T_world_camera), world)  # pyright: ignore[reportCallIssue]
    if camera[2] <= 0.0:
        return
    u = intrinsics[0, 0] * camera[0] / camera[2] + intrinsics[0, 2]
    v = intrinsics[1, 1] * camera[1] / camera[2] + intrinsics[1, 2]
    if not wp.isfinite(u) or not wp.isfinite(v):
        return
    if u < -0.5 or v < -0.5 or u >= float(depth.shape[1]) - 0.5 or v >= float(depth.shape[0]) - 0.5:
        return
    d = depth[int(v + 0.5), int(u + 0.5)]
    if not wp.isfinite(d) or d <= 0.0 or d >= depth_max:
        return
    distance = d - camera[2]
    if distance < -trunc:
        return
    previous = fusion[i, j, k]
    mean = (previous[0] * previous[1] + wp.min(distance, trunc)) / (previous[1] + 1.0)
    fusion[i, j, k] = wp.vec2(mean, wp.min(previous[1] + 1.0, max_weight))
