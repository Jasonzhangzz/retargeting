# pyright: reportArgumentType=false
# pyright: reportOptionalMemberAccess=false
"""Batched heterogeneous scene SDF query: one kernel per attached geom, chained through the per-point min."""

from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Union, overload

import warp as wp

from robokit.geom.geoms import MAX_DIST, BaseGeom, VolumeGeom
from robokit.geom.sdf_kernels import build_scene_per_point_kernel, clear_sdf_query_kernel
from robokit.utils.warp_utils import wp_device_type


if TYPE_CHECKING:
    import torch

Device = Optional[Union[str, object]]


class WarpScene:
    """Heterogeneous batched scene SDF query.

    Example:
        >>> import numpy as np
        >>> import torch
        >>> import warp as wp
        >>> from robokit.geom.geoms import SphereGeom
        >>> wp.init()
        >>> scene = WarpScene(num_scenes=1, device="cpu").add(
        ...     SphereGeom(np.array([1.0]), np.array([0, 1], dtype=np.int32))
        ... )
        >>> signed_dists, _, _ = scene.query_sdf_torch(
        ...     torch.tensor([[2.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        ...     torch.tensor([0, 2]),
        ... )
        >>> bool(torch.allclose(signed_dists, torch.tensor([1.0, -0.5])))
        True
        >>> normal_jacobians = wp.empty(1, dtype=wp.mat33, device="cpu")
        >>> _ = scene.query_sdf(
        ...     wp.array([[2.0, 0.0, 0.0]], dtype=wp.vec3, device="cpu"),
        ...     scene_indices=wp.array([0], dtype=wp.int32, device="cpu"),
        ...     out_normal_jacobians=normal_jacobians,
        ... )
        >>> np.allclose(normal_jacobians.numpy()[0], np.diag([0.0, 0.5, 0.5]))
        True
        >>> observed = wp.empty(1, dtype=wp.bool, device="cpu")
        >>> _ = scene.query_sdf(wp.array([[2.0, 0.0, 0.0]], dtype=wp.vec3, device="cpu"),
        ...     scene_indices=wp.array([0], dtype=wp.int32, device="cpu"), out_observed=observed)
        >>> bool(observed.numpy()[0])  # analytical geometry is fully known
        True
    """

    def __init__(self, num_scenes: int, device: Device = "cpu"):
        if num_scenes < 1:
            raise ValueError(f"num_scenes must be >= 1, got {num_scenes}.")
        self._num_scenes = int(num_scenes)
        self._device_wp = wp.get_device(str(device))
        self._geoms: List[BaseGeom] = []
        # size-1 placeholders for kernel args whose feature flag is off (never read)
        self._unused_vec3 = wp.empty((1,), dtype=wp.vec3, device=self._device_wp)
        self._unused_mat33 = wp.empty((1,), dtype=wp.mat33, device=self._device_wp)
        self._unused_int32 = wp.empty((1,), dtype=wp.int32, device=self._device_wp)
        self._buffers: List[Tuple[wp.array, wp.array, wp.array]] = []  # see query_sdf
        self._graph_revision = 0
        # Weak both ways: a scene kept alive by a cycle frees its SDF volumes whenever GC runs, even mid-capture.

    @property
    def device(self) -> wp_device_type:
        return self._device_wp

    @property
    def geoms(self) -> Tuple[BaseGeom, ...]:
        return tuple(self._geoms)

    @property
    def num_scenes(self) -> int:
        return self._num_scenes

    @property
    def num_elements(self) -> int:
        """Total element count across all attached geoms (a MeshGeom of N meshes counts N)."""
        return sum(geom.count for geom in self._geoms)

    @property
    def graph_revision(self) -> int:
        """Monotonic revision of resources referenced by captured graphs."""
        return self._graph_revision + sum(geom.graph_revision for geom in self._geoms)

    def add(self, geom: BaseGeom) -> "WarpScene":
        """Attach a geom: allocates its GPU arrays on this scene's device; update in place via `geom.update(...)`."""
        if geom.device_wp is not None:
            raise ValueError("geom has already been built or added to a scene.")
        geom.build(self._device_wp)
        if geom.num_scenes != self._num_scenes:
            raise ValueError(f"geom has {geom.num_scenes} scenes, scene expects {self._num_scenes}.")
        self._geoms.append(geom)
        self._graph_revision += 1
        return self

    # --- queries ---
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
        """Query the scene SDF (min over all attached geoms) at `points`.

        Args:
            points: Query points, any shape (flattened to n internally).
            scene_offsets: CSR boundaries into the flat points, int32, length num_scenes+1. Pass exactly one of `scene_offsets` / `scene_indices`.
            scene_indices: Scene id of each point, int32, one per point. Skips the derivation launch, so hot loops pass a pre-built map here.
            query_mask: Optional per-point gate, int32, length n. Points with mask 0 skip every geom kernel.
            distance_only: Skip the normal/closest computation (~2x faster) and return only signed distances.
            out_signed_dists: Optional preallocated output; required shape n. Hot loops and CUDA-graph capture pass all `out_*` so no allocation happens per call.
            out_normals: Optional preallocated output, shape n.
            out_closest_points: Optional preallocated output, shape n.
            out_normal_jacobians: Optional preallocated `wp.mat33` output, one per point. Entry (i, j) is the derivative of normal component i with respect to point coordinate j. Mesh face normals have zero derivatives; voxel and primitive normals use analytical derivatives of the active query branch. Derivatives at cell, face, or closest-object switches are discontinuous.
            record_tape: Record the launches on an active `wp.Tape`; False keeps the query off the gradient.
            out_observed: Optional device bool array, one per point. True when every sensor map in the scene has observed the queried voxel; outside a sensor map or masked out is False. Meshes and analytical geometry leave this status unchanged. Unknown status never skips distance checks.

        Returns:
            Flat `(n,)` views: `signed_dists` if `distance_only`, else `(signed_dists, normals, closest_points)`. Points in an empty scene get `MAX_DIST` distance and zero normals.
        """
        if (scene_offsets is None) == (scene_indices is None):
            raise ValueError("query_sdf requires exactly one of scene_offsets or scene_indices.")
        n = points.size
        if out_observed is not None:
            if out_observed.dtype != wp.bool or out_observed.size != n or out_observed.device != self._device_wp:
                raise ValueError("out_observed must be a bool array with one entry per point on the scene device.")
            out_observed = out_observed.reshape((n,))
        if out_normal_jacobians is not None:
            if distance_only:
                raise ValueError("normal Jacobians require distance_only=False.")
            out_normal_jacobians = out_normal_jacobians.reshape((n,))
            out_normal_jacobians.zero_()

        # --- work buffers ---
        # Grow once per size because hot loops and CUDA graphs must not allocate per call.
        if not self._buffers or self._buffers[-1][0].size < n:
            # keep outgrown buffers: graphs captured with them still write them on replay
            self._buffers.append(
                (
                    wp.empty((n,), dtype=wp.int32, device=self._device_wp),
                    wp.empty((n,), dtype=wp.vec3, device=self._device_wp),
                    wp.empty((n,), dtype=wp.int32, device=self._device_wp),
                )
            )
        scene_per_point, local_coords, element_indices = self._buffers[-1]

        # --- outputs ---
        # Flatten to (n,). Signed distances start at MAX_DIST so geometry kernels chain by minimum.
        if out_signed_dists is None:
            out_signed_dists = wp.empty((n,), dtype=wp.float32, device=self._device_wp)
        signed_dists = out_signed_dists.reshape((n,))
        wp.launch(
            clear_sdf_query_kernel,
            n,
            [signed_dists, out_observed, query_mask, MAX_DIST],
            device=self._device_wp,
            record_tape=record_tape,
        )
        if distance_only:
            # skip the normal/closest math (~2x faster)
            normals = closest_points = self._unused_vec3
        else:
            if out_normals is None:
                out_normals = wp.zeros((n,), dtype=wp.vec3, device=self._device_wp)
            if out_closest_points is None:
                out_closest_points = wp.zeros((n,), dtype=wp.vec3, device=self._device_wp)
            normals = out_normals.reshape((n,))
            closest_points = out_closest_points.reshape((n,))

        # No geometries are attached, so there is nothing to launch.
        if len(self._geoms) == 0:
            if distance_only:
                return signed_dists
            normals.zero_()
            closest_points.zero_()
            return signed_dists, normals, closest_points

        # --- point-to-scene mapping ---
        if scene_indices is not None:
            scene_per_point = scene_indices.reshape((n,))
        else:
            wp.launch(
                kernel=build_scene_per_point_kernel,
                dim=n,
                inputs=[scene_offsets, scene_per_point],
                device=self._device_wp,
                record_tape=record_tape,
            )

        # --- geometry queries ---
        # Launch once per geometry, chaining through the per-point minimum.
        points_flat = points.reshape((n,))
        for geom in self._geoms:
            geom._launch(
                points_flat,
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

        # Return the requested results.
        if distance_only:
            return signed_dists
        return signed_dists, normals, closest_points

    def query_sdf_torch(
        self,
        query_points: "torch.Tensor",
        scene_offsets: "torch.Tensor",
        poses: Optional[Dict[BaseGeom, "torch.Tensor"]] = None,
        scales: Optional[Dict[BaseGeom, "torch.Tensor"]] = None,
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """Differentiable scene SDF query (per-point min over all attached geoms).

        Args:
            query_points: Query points, shape (n, 3), differentiable.
            scene_offsets: CSR boundaries into the points, length num_scenes+1.
            poses: Optional per-geom pose overrides, (count, 4, 4) tensors keyed by attached geom. Overridden geoms are queried on these poses and receive gradients; other geoms use their stored (non-differentiable) transforms.
            scales: Optional per-geom scale overrides, (count,) tensors keyed by attached geom. Same semantics as `poses`.

        Returns:
            `(signed_dists, normals, closest_points)` torch tensors, shapes (n,), (n, 3), (n, 3). Gradients flow to `query_points` and to the `poses`/`scales` tensors of whichever geom is closest at each point.
        """
        import torch

        from robokit.xform.warp.torch_wrappers import inverse_tf_mat, transform_points

        if scene_offsets.shape[0] != self._num_scenes + 1:
            raise ValueError(f"scene_offsets must have length {self._num_scenes + 1}, got {scene_offsets.shape[0]}.")
        poses = poses if poses is not None else {}
        scales = scales if scales is not None else {}
        for geom in list(poses) + list(scales):
            if geom not in self._geoms:
                raise ValueError("poses/scales keys must be geoms attached to this scene.")

        # --- outputs ---
        # Track the running per-point minimum; signed distances start at MAX_DIST.
        best_signed_dists = torch.full_like(query_points[..., 0], float(MAX_DIST), requires_grad=False)
        best_normals = torch.zeros_like(query_points, requires_grad=False)
        best_clst = torch.zeros_like(query_points, requires_grad=False)

        # No geometries are attached, so there is nothing to launch.
        if len(self._geoms) == 0:
            return best_signed_dists, best_normals, best_clst

        # --- point-to-scene mapping ---
        n_total = int(query_points.shape[0])
        device_wp = wp.device_from_torch(query_points.device)
        scene_offsets_wp = wp.from_torch(
            scene_offsets.to(torch.int32).contiguous(), dtype=wp.int32, requires_grad=False
        )
        scene_per_point_wp = wp.empty((n_total,), dtype=wp.int32, device=device_wp)
        wp.launch(
            kernel=build_scene_per_point_kernel,
            dim=n_total,
            inputs=[scene_offsets_wp, scene_per_point_wp],
            device=device_wp,
        )
        query_pts_wp = wp.from_torch(query_points.contiguous().view(-1, 3), dtype=wp.vec3, requires_grad=False)

        # --- geometry queries ---
        # Launch and reconstruct differentiably per geometry, keeping the per-point minimum.
        for geom in self._geoms:
            signed_dists = torch.full_like(query_points[..., 0], MAX_DIST, requires_grad=False)
            normals = torch.zeros_like(query_points, requires_grad=False)
            clst_pts = torch.zeros_like(query_points, requires_grad=False)
            clst_pts_local = torch.zeros_like(query_points, requires_grad=False)
            clst_indices = torch.zeros_like(query_points[..., 0], dtype=torch.int32, requires_grad=False)

            # override wins over stored; stored still queries, just without gradients
            override_poses = poses[geom] if geom in poses else None
            override_scales = scales[geom] if geom in scales else None
            enable_pose = override_poses is not None or geom.enable_inv_poses
            enable_scale = override_scales is not None or geom.enable_scales
            if override_poses is not None:
                inv_poses_gather = inverse_tf_mat(override_poses)
                inv_poses_wp = wp.from_torch(inv_poses_gather.contiguous(), dtype=wp.mat44, requires_grad=False)
            else:
                inv_poses_wp = geom.inv_poses_wp
                inv_poses_gather = wp.to_torch(inv_poses_wp).view(-1, 4, 4) if geom.enable_inv_poses else None
            if override_scales is not None:
                scales_gather = override_scales
                scales_wp = wp.from_torch(override_scales.contiguous(), dtype=wp.float32, requires_grad=False)
            else:
                scales_wp = geom.scales_wp
                scales_gather = wp.to_torch(scales_wp).view(-1) if geom.enable_scales else None

            wp.launch(
                kernel=geom.kernel,
                dim=n_total,
                inputs=[
                    query_pts_wp,
                    scene_per_point_wp,
                    *geom.launch_params,
                    geom.scene_offsets_wp,
                    inv_poses_wp,
                    enable_pose,
                    scales_wp,
                    enable_scale,
                    self._unused_int32,
                    False,  # enable_query_mask
                    True,  # enable_normals: torch path needs normals + closest for gradients
                    MAX_DIST,
                    wp.from_torch(signed_dists.view(-1), dtype=wp.float32),
                    wp.from_torch(normals.view(-1, 3), dtype=wp.vec3),
                    wp.from_torch(clst_pts.view(-1, 3), dtype=wp.vec3),
                    wp.from_torch(clst_pts_local.view(-1, 3), dtype=wp.vec3),
                    wp.from_torch(clst_indices.view(-1), dtype=wp.int32),
                    False,
                    self._unused_mat33,
                    *((None,) if isinstance(geom, VolumeGeom) else ()),
                ],
                device=device_wp,
            )

            # differentiable signed distance: sign(d) * ||pts_in_local - clst_in_local|| (* scale)
            clst_indices_long = clst_indices.to(torch.long).view(-1)
            if inv_poses_gather is not None:
                pose_selected = torch.index_select(inv_poses_gather, dim=0, index=clst_indices_long)
                pts_in_local = transform_points(query_points.unsqueeze(-2), pose_selected).squeeze(-2)
            else:
                pts_in_local = query_points
            if scales_gather is not None:
                pts_in_local = pts_in_local / scales_gather[clst_indices_long].view(*query_points.shape[:-1], 1)
            if enable_pose or enable_scale:
                clst_used = clst_pts_local
            else:
                clst_used = clst_pts
            diff_signed_dists = torch.sign(signed_dists) * torch.norm(pts_in_local - clst_used, p=2, dim=-1)
            if scales_gather is not None:
                diff_signed_dists = diff_signed_dists * scales_gather[clst_indices_long].view(query_points.shape[:-1])
            # points this geom never reached stay at MAX_DIST instead of the wrong ||pts - 0|| value
            diff_signed_dists = torch.where(signed_dists < MAX_DIST, diff_signed_dists, signed_dists)

            mask = diff_signed_dists < best_signed_dists
            best_signed_dists = torch.where(mask, diff_signed_dists, best_signed_dists)
            best_normals = torch.where(mask.unsqueeze(-1), normals, best_normals)
            best_clst = torch.where(mask.unsqueeze(-1), clst_pts, best_clst)

        # Return the requested results.
        return best_signed_dists, best_normals, best_clst

    def __repr__(self) -> str:
        geoms = ",".join(type(g).__name__ for g in self._geoms) or "empty"
        return f"WarpScene(num_scenes={self._num_scenes}, geoms=[{geoms}])"


__all__ = ["WarpScene"]
