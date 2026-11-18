# pyright: reportIndexIssue=false
# pyright: reportArgumentType=false
# pyright: reportCallIssue=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportOptionalSubscript=false
# pyright: reportOperatorIssue=false
# pyright: reportMissingImports=false
"""Mask/depth alignment and optional kinematic model calibration."""

from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union, cast

import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F
import warp as wp
from nvdiffrast.torch.ops import _antialias_func as _NativeAntialias
from nvdiffrast.torch.ops import _get_plugin

from robokit.lie.se3 import SE3Var, se3_adjoint, se3_from_matrix, se3_to_matrix
from robokit.lie.se3_kernels import se3_adjoint_kernel
from robokit.opt.var_values import Var, VarKey, VarValues, VectorVar
from robokit.robo import Robot
from robokit.terms.dense.frame_vector_task import (
    _compute_huber_gradient_scale_func,
    _compute_huber_residual_signed_func,
)
from robokit.terms.task import Context, EagerTask
from robokit.utils.warp_utils import wp_device_type, wp_mat66, wp_vec6, wp_vec7
from robokit.xform.warp.torch_wrappers import intr_to_proj_mat


# Reuse CUDA contexts across pyramid levels and solves on each device.
_GLCTX = {}


def _rasterize_context(device: Optional[int] = None) -> "dr.RasterizeCudaContext":
    if device not in _GLCTX:
        _GLCTX[device] = dr.RasterizeCudaContext(device=device)
    return _GLCTX[device]


_OPENCV2GL = torch.tensor([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=torch.float32)


def render_mask(
    glctx: dr.RasterizeCudaContext,
    verts: torch.Tensor,
    faces: torch.Tensor,
    intrinsic: torch.Tensor,
    T_camera_link: torch.Tensor,
    H: int,
    W: int,
) -> torch.Tensor:
    """Render a differentiable binary mask of a mesh seen from a camera.

    Args:
        glctx: nvdiffrast rasterization context.
        verts: Mesh vertices `[V, 3]`.
        faces: Triangle indices `[F, 3]` (int32).
        intrinsic: Camera intrinsic matrix `[3, 3]`.
        T_camera_link: 4x4 SE3 from link-local frame to camera frame.
        H, W: Output image height and width.

    Returns:
        Anti-aliased mask of shape `[H, W]` with values in `[0, 1]`.
    """
    proj = intr_to_proj_mat(intrinsic, H, W)
    T_gl_link = _OPENCV2GL.to(T_camera_link.device) @ T_camera_link
    mvp = proj @ T_gl_link  # [4, 4]
    # homogeneous clip-space vertices
    ones = torch.ones(verts.shape[0], 1, dtype=torch.float32, device=verts.device)
    pos_homo = torch.cat([verts, ones], dim=1)
    pos_clip = (pos_homo @ mvp.t()).unsqueeze(0)

    rast_out, _ = dr.rasterize(glctx, pos_clip, faces, resolution=(H, W))
    vtx_color = torch.ones(1, verts.shape[0], 3, dtype=torch.float32, device=verts.device)
    color, _ = dr.interpolate(vtx_color, rast_out, faces)
    color = _antialias(color, rast_out, pos_clip, faces)
    mask: torch.Tensor = color[0, :, :, 0]
    return torch.flip(mask, dims=[0])


# --- device code ------------------------------------------------------------
# Native edge work contains [x, y, batch/edge flags, float32 alpha], preceded by one counter row.
@wp.kernel
def _accumulate_antialias_kernel(
    work: wp.array2d(dtype=wp.int32),
    alpha: wp.array2d(dtype=wp.float32),
    color: wp.array1d(dtype=wp.float32),
    result: wp.array1d(dtype=wp.float64),
    width: int,
    height: int,
    channels: int,
):
    edge, channel = wp.tid()  # pyright: ignore[reportAssignmentType]
    if edge >= work[0, 0]:
        return
    value = alpha[edge + 1, 3]
    if value == 0.0:
        return
    flags = work[edge + 1, 2]
    first = work[edge + 1, 0] + width * (work[edge + 1, 1] + height * (flags >> 16))
    second = first + wp.where((flags & 4) != 0, width, 1)
    destination = wp.where(value > 0.0, first, second)
    contribution = value * (color[second * channels + channel] - color[first * channels + channel])
    wp.atomic_add(result, destination * channels + channel, wp.float64(contribution))


class _MaskAntialias(_NativeAntialias):
    """Retain native edge analysis and gradients; sum edge colors without float32 atomic-order noise."""

    @staticmethod
    def forward(
        ctx: Any,
        color: torch.Tensor,
        rast: torch.Tensor,
        pos: torch.Tensor,
        tri: torch.Tensor,
        topology: Any,
        boost: float,
    ) -> torch.Tensor:
        _NativeAntialias.forward(ctx, color, rast, pos, tri, topology, boost)
        work = ctx.saved_misc[1]
        result = color.double()
        current = torch.cuda.current_stream(color.device)
        stream = wp.get_stream(str(color.device))
        if stream.cuda_stream != current.cuda_stream:
            stream = wp.stream_from_torch(current)
        with wp.ScopedStream(stream):
            wp.launch(
                _accumulate_antialias_kernel,
                dim=(work.numel() // 4 - 1, color.shape[-1]),
                inputs=[
                    wp.from_torch(work.view(torch.int32).reshape(-1, 4)),
                    wp.from_torch(work.reshape(-1, 4)),
                    wp.from_torch(color.flatten()),
                    wp.from_torch(result.flatten()),
                    color.shape[2],
                    color.shape[1],
                    color.shape[3],
                ],
                device=str(color.device),
            )
        return result.float()


def _antialias(color: torch.Tensor, rast: torch.Tensor, pos: torch.Tensor, tri: torch.Tensor) -> torch.Tensor:
    return _MaskAntialias.apply(color, rast, pos, tri, _get_plugin().antialias_construct_topology_hash(tri), 1.0)


# An offset dof's column is `j_cam`'s row projected onto its screw axis (offset dofs move vertices, not the camera).
@wp.func
def _accumulate_extra_dofs_func(
    j_cam: wp_vec6,
    j_camera_col: wp_vec6,
    r: wp.float32,
    sn: wp.int32,
    v: wp.int32,
    u: wp.int32,
    n_obs: wp.int32,
    screws: wp.array(dtype=wp_vec6, ndim=4),
    pixel_link: wp.array3d(dtype=wp.int32),
    num_extra: wp.int32,
    col_offset: wp.int32,
    optimize_camera: wp.bool,
    JtJ: wp.array3d(dtype=wp.float64),
    Jtr: wp.array3d(dtype=wp.float64),
):
    if num_extra == 0:
        return
    candidate = sn // n_obs
    obs = sn - candidate * n_obs
    link = pixel_link[sn, v, u]
    for d in range(num_extra):
        jd = wp.dot(j_cam, screws[candidate, obs, d, link])
        if jd == wp.float32(0.0):
            continue
        col = col_offset + d
        wp.atomic_add(Jtr, candidate, col, 0, wp.float64(jd * r))
        if optimize_camera:
            for i in range(6):
                wp.atomic_add(JtJ, candidate, i, col, wp.float64(j_camera_col[i] * jd))
                wp.atomic_add(JtJ, candidate, col, i, wp.float64(jd * j_camera_col[i]))
        for e in range(num_extra):
            je = wp.dot(j_cam, screws[candidate, obs, e, link])
            wp.atomic_add(JtJ, candidate, col, col_offset + e, wp.float64(jd * je))


@wp.kernel
def _accumulate_normal_equations_kernel(
    rendered: wp.array3d(dtype=wp.float32),
    target: wp.array3d(dtype=wp.float32),
    observation_weights: wp.array1d(dtype=wp.float32),
    union_gradient: wp.array1d(dtype=wp.float32),
    optimize_camera: wp.bool,
    contour_geometry: wp.bool,
    positions: wp.array(dtype=wp.float32, ndim=4),
    focal: wp.array2d(dtype=wp.float32),
    img_h: wp.int32,
    img_w: wp.int32,
    n_obs: wp.int32,
    adjoint: wp.array(dtype=wp_mat66),
    screws: wp.array(dtype=wp_vec6, ndim=4),
    pixel_link: wp.array3d(dtype=wp.int32),
    num_extra: wp.int32,
    col_offset: wp.int32,
    JtJ: wp.array3d(dtype=wp.float64),
    Jtr: wp.array3d(dtype=wp.float64),
    cost: wp.array1d(dtype=wp.float64),
):
    """Per-pixel residual + analytical Jacobian, accumulated per candidate."""
    sn, v, u = wp.tid()  # pyright: ignore[reportAssignmentType]
    candidate = sn // n_obs
    fx, fy = focal[sn % n_obs, 0], focal[sn % n_obs, 1]

    sw = wp.sqrt(observation_weights[sn])
    r = sw * (rendered[sn, v, u] - target[sn, v, u])
    # Exact matches contribute zero; avoid contending on one atomic for every background pixel.
    if r != wp.float32(0.0):
        wp.atomic_add(cost, candidate, wp.float64(wp.float32(0.5) * r * r))
    if rendered[sn, v, u] > target[sn, v, u]:
        r -= sw * union_gradient[sn]

    left, right = wp.max(u - 1, 0), wp.min(u + 1, img_w - 1)
    top, bottom = wp.max(v - 1, 0), wp.min(v + 1, img_h - 1)
    du = sw * (rendered[sn, v, right] - rendered[sn, v, left]) / wp.float32(wp.max(right - left, 1))
    dv = sw * (rendered[sn, bottom, u] - rendered[sn, top, u]) / wp.float32(wp.max(bottom - top, 1))

    if du == wp.float32(0.0) and dv == wp.float32(0.0):
        return

    X = positions[sn, v, u, 0]
    Y = positions[sn, v, u, 1]
    Z = positions[sn, v, u, 2]
    surface_v, surface_u = v, u
    if Z <= wp.float32(0.0):
        nearest = wp.int32(3)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                y, x = v + dy, u + dx
                distance = dx * dx + dy * dy
                if y >= 0 and y < img_h and x >= 0 and x < img_w and distance < nearest:
                    depth = positions[sn, y, x, 2]
                    if depth > wp.float32(0.0):
                        nearest, surface_v, surface_u = distance, y, x
                        X = wp.float32(positions[sn, y, x, 0])
                        Y = wp.float32(positions[sn, y, x, 1])
                        Z = wp.float32(depth)
        fraction = wp.float32(0.5) if contour_geometry else wp.float32(1.0)
        X += fraction * wp.float32(u - surface_u) * Z / fx
        Y += fraction * wp.float32(v - surface_v) * Z / fy
    elif contour_geometry:
        # Evaluate silhouette motion at the midpoint between covered and uncovered sample centres.
        nearest = wp.int32(3)
        boundary_v, boundary_u = v, u
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                y, x = v + dy, u + dx
                distance = dx * dx + dy * dy
                if y >= 0 and y < img_h and x >= 0 and x < img_w and distance < nearest:
                    if positions[sn, y, x, 2] <= wp.float32(0.0):
                        nearest, boundary_v, boundary_u = distance, y, x
        X += wp.float32(0.5) * wp.float32(boundary_u - u) * Z / fx
        Y += wp.float32(0.5) * wp.float32(boundary_v - v) * Z / fy
    if Z <= wp.float32(0.0):
        return

    inv_Z = wp.float32(1.0) / Z
    inv_Z2 = inv_Z * inv_Z

    j = wp_vec6(
        -(du * fx * inv_Z),
        -(dv * fy * inv_Z),
        (du * fx * X + dv * fy * Y) * inv_Z2,
        du * fx * X * Y * inv_Z2 - dv * (-fy - fy * Y * Y * inv_Z2),
        -(du * (fx + fx * X * X * inv_Z2) + dv * fy * X * Y * inv_Z2),
        du * fx * Y * inv_Z - dv * fy * X * inv_Z,
    )
    jc = wp_vec6()
    if optimize_camera:
        jc = wp.transpose(adjoint[candidate]) * j
        for i in range(6):
            wp.atomic_add(Jtr, candidate, i, 0, wp.float64(jc[i] * r))
            for k in range(6):
                wp.atomic_add(JtJ, candidate, i, k, wp.float64(jc[i] * jc[k]))
    _accumulate_extra_dofs_func(
        j, jc, r, sn, surface_v, surface_u, n_obs, screws, pixel_link, num_extra, col_offset, optimize_camera, JtJ, Jtr
    )


@wp.kernel
def _compute_cost_kernel(
    rendered: wp.array3d(dtype=wp.float32),
    target: wp.array3d(dtype=wp.float32),
    observation_weights: wp.array1d(dtype=wp.float32),
    n_obs: wp.int32,
    cost: wp.array1d(dtype=wp.float64),
):
    sn, v, u = wp.tid()  # pyright: ignore[reportAssignmentType]
    candidate = sn // n_obs
    r = rendered[sn, v, u] - target[sn, v, u]
    if r != wp.float32(0.0):
        wp.atomic_add(cost, candidate, wp.float64(observation_weights[sn] * wp.float32(0.5) * r * r))


# reject depth-image gradients across occlusion boundaries
_DEPTH_FLOW_GRAD_MAX = wp.constant(0.05)


# combine direct depth motion with image flow so depth constrains all six directions
@wp.kernel
def _accumulate_depth_normal_equations_kernel(
    positions: wp.array(dtype=wp.float32, ndim=4),
    rendered: wp.array3d(dtype=wp.float32),
    target_depth: wp.array3d(dtype=wp.float32),
    observation_weights: wp.array1d(dtype=wp.float32),
    focal: wp.array2d(dtype=wp.float32),
    img_h: wp.int32,
    img_w: wp.int32,
    huber_delta: wp.float32,
    weight: wp.float32,
    n_obs: wp.int32,
    adjoint: wp.array(dtype=wp_mat66),
    screws: wp.array(dtype=wp_vec6, ndim=4),
    pixel_link: wp.array3d(dtype=wp.int32),
    num_extra: wp.int32,
    col_offset: wp.int32,
    JtJ: wp.array3d(dtype=wp.float64),
    Jtr: wp.array3d(dtype=wp.float64),
    cost: wp.array1d(dtype=wp.float64),
):
    sn, v, u = wp.tid()  # pyright: ignore[reportAssignmentType]
    candidate = sn // n_obs
    fx, fy = focal[sn % n_obs, 0], focal[sn % n_obs, 1]

    Z = positions[sn, v, u, 2]
    Dt = target_depth[sn, v, u]
    # require rendered and observed depth
    if rendered[sn, v, u] <= wp.float32(0.0) or Z <= wp.float32(0.0) or Dt <= wp.float32(0.0):
        return

    X = positions[sn, v, u, 0]
    Y = positions[sn, v, u, 1]

    # compute depth gradients away from occlusion edges
    du = wp.float32(0.0)
    dv = wp.float32(0.0)
    if u > 0 and u < img_w - 1:
        zp = positions[sn, v, u + 1, 2]
        zm = positions[sn, v, u - 1, 2]
        if zp > wp.float32(0.0) and zm > wp.float32(0.0):
            g = wp.float32(0.5) * (zp - zm)
            if wp.abs(g) <= _DEPTH_FLOW_GRAD_MAX:
                du = g
    if v > 0 and v < img_h - 1:
        zp = positions[sn, v + 1, u, 2]
        zm = positions[sn, v - 1, u, 2]
        if zp > wp.float32(0.0) and zm > wp.float32(0.0):
            g = wp.float32(0.5) * (zp - zm)
            if wp.abs(g) <= _DEPTH_FLOW_GRAD_MAX:
                dv = g

    inv_Z = wp.float32(1.0) / Z
    inv_Z2 = inv_Z * inv_Z
    r = Z - Dt
    sw = wp.sqrt(weight * observation_weights[sn])
    rw = sw * _compute_huber_residual_signed_func(r, huber_delta)
    scale = sw * _compute_huber_gradient_scale_func(r, huber_delta)
    j = wp_vec6(
        scale * (-(du * fx * inv_Z)),
        scale * (-(dv * fy * inv_Z)),
        scale * (wp.float32(1.0) + (du * fx * X + dv * fy * Y) * inv_Z2),
        scale * (Y + du * fx * X * Y * inv_Z2 - dv * (-fy - fy * Y * Y * inv_Z2)),
        scale * (-X - (du * (fx + fx * X * X * inv_Z2) + dv * fy * X * Y * inv_Z2)),
        scale * (du * fx * Y * inv_Z - dv * fy * X * inv_Z),
    )
    jc = wp.transpose(adjoint[candidate]) * j

    wp.atomic_add(cost, candidate, wp.float64(wp.float32(0.5) * rw * rw))
    for i in range(6):
        wp.atomic_add(Jtr, candidate, i, 0, wp.float64(jc[i] * rw))
        for k in range(6):
            wp.atomic_add(JtJ, candidate, i, k, wp.float64(jc[i] * jc[k]))
    _accumulate_extra_dofs_func(j, jc, rw, sn, v, u, n_obs, screws, pixel_link, num_extra, col_offset, True, JtJ, Jtr)


@wp.kernel
def _compute_depth_cost_kernel(
    positions: wp.array(dtype=wp.float32, ndim=4),
    rendered: wp.array3d(dtype=wp.float32),
    target_depth: wp.array3d(dtype=wp.float32),
    observation_weights: wp.array1d(dtype=wp.float32),
    huber_delta: wp.float32,
    weight: wp.float32,
    n_obs: wp.int32,
    cost: wp.array1d(dtype=wp.float64),
):
    sn, v, u = wp.tid()  # pyright: ignore[reportAssignmentType]
    candidate = sn // n_obs
    Z = positions[sn, v, u, 2]
    Dt = target_depth[sn, v, u]
    if rendered[sn, v, u] <= wp.float32(0.0) or Z <= wp.float32(0.0) or Dt <= wp.float32(0.0):
        return
    rw = _compute_huber_residual_signed_func(Z - Dt, huber_delta)
    wp.atomic_add(cost, candidate, wp.float64(weight * observation_weights[sn] * wp.float32(0.5) * rw * rw))


# --- rendering --------------------------------------------------------------
def _precompute_merged_mesh(
    link_vertices: List[torch.Tensor],
    link_faces: List[torch.Tensor],
    T_ref_link: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute merged faces and per-observation base-frame vertices.

    Returns:
        merged_faces: `[F_total, 3]` int32 on CUDA.
        v_base: `[N, V_total, 3]` float32 on CUDA.
    """
    all_faces: List[torch.Tensor] = []
    vert_offset = 0
    for mi in range(len(link_vertices)):
        all_faces.append(link_faces[mi] + vert_offset)
        vert_offset += link_vertices[mi].shape[0]
    merged_faces = torch.cat(all_faces, dim=0)

    v_base = torch.cat(
        [
            v[None] @ T_ref_link[:, i, :3, :3].transpose(-1, -2) + T_ref_link[:, i, None, :3, 3]
            for i, v in enumerate(link_vertices)
        ],
        dim=1,
    )
    return merged_faces, v_base


def _render_batch(
    glctx: dr.RasterizeCudaContext,
    T_cam_base: torch.Tensor,
    intrinsic: torch.Tensor,
    merged_faces: torch.Tensor,
    v_base: torch.Tensor,
    H: int,
    W: int,
    with_positions: bool = True,
    face_link_ids: Optional[torch.Tensor] = None,
    target_link_mask: Optional[torch.Tensor] = None,
    projection: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Batch-render S candidates × N observations with bounded CUDA workspaces.

    Args:
        T_cam_base: `[S, 4, 4]` or `[4, 4]` camera extrinsic(s).
        v_base: Per-observation `[N,V_total,3]` vertices, or per-candidate `[S,N,V_total,3]` with model offsets.
        face_link_ids: Optional 1-based `[F_total]` triangle link IDs; returns `[S*N,H,W]` IDs with background 0.
        target_link_mask: Optional global `[num_links+1]` or per-observation `[N,num_links+1]` visible-link selection.

    Returns:
        Masks `[S*N,H,W]`, optional camera-space positions `[S*N,H,W,3]` and optional per-pixel link IDs.
    """
    if T_cam_base.ndim == 2:
        T_cam_base = T_cam_base.unsqueeze(0)

    S = T_cam_base.shape[0]
    R = T_cam_base[:, :3, :3]  # [S, 3, 3]
    t = T_cam_base[:, :3, 3]  # [S, 3]

    # v_cam[s, n] = v_base[(s,) n] @ R[s].T + t[s]
    eq = "snvi,sji->snvj" if v_base.ndim == 4 else "nvi,sji->snvj"
    v_cam = torch.einsum(eq, v_base, R) + t[:, None, None, :]  # [S, N, V, 3]
    N, V = v_base.shape[-3:-1]
    v_cam = v_cam.reshape(S * N, V, 3)

    if projection is None:
        intrinsics = intrinsic[None] if intrinsic.ndim == 2 else intrinsic
        projection = torch.stack([intr_to_proj_mat(k, H, W) for k in intrinsics]) @ _OPENCV2GL.to(v_cam.device)
    mvp = projection.repeat(S, 1, 1) if len(projection) > 1 else projection
    v_clip = torch.baddbmm(mvp[:, None, :, 3], v_cam, mvp[:, :, :3].transpose(-1, -2).expand(S * N, 3, 4))

    vtx_white = torch.ones(1, V, 1, device=v_cam.device, dtype=torch.float32)
    selection = target_link_mask
    if selection is not None and selection.ndim == 2:
        selection = selection.repeat(S, 1)
    masks, positions, labels = [], [], []
    # Bound both geometry and pixel workspaces without dropping candidates or observations.
    chunk = max(1, min(1024, 2**26 // (H * W)))
    for start in range(0, S * N, chunk):
        end = min(start + chunk, S * N)
        clip = v_clip[start:end]
        rast, _ = dr.rasterize(glctx, clip, merged_faces, resolution=(H, W))
        color, _ = dr.interpolate(vtx_white, rast, merged_faces)
        if face_link_ids is not None:
            tri = rast[..., 3].to(torch.int32)
            link_ids = torch.where(tri > 0, face_link_ids[(tri - 1).clamp(min=0).long()], 0)
            if selection is not None:
                selected = (
                    selection[link_ids.long()]
                    if selection.ndim == 1
                    else torch.gather(selection[start:end], 1, link_ids.flatten(1).long()).reshape_as(link_ids)
                )
                color = selected.float().unsqueeze(-1)
            labels.append(torch.flip(link_ids, dims=[1]).contiguous())
        aa_chunk = max(1, 2**23 // (H * W))
        color = torch.cat(
            [
                _antialias(color[i : i + aa_chunk], rast[i : i + aa_chunk], clip[i : i + aa_chunk], merged_faces)
                for i in range(0, len(color), aa_chunk)
            ]
        )
        masks.append(torch.flip(color[..., 0], dims=[1]))
        if with_positions:
            point, _ = dr.interpolate(v_cam[start:end], rast, merged_faces)
            positions.append(torch.flip(point, dims=[1]))
    result = torch.cat(masks), torch.cat(positions) if with_positions else None
    return (*result, torch.cat(labels)) if face_link_ids is not None else result  # type: ignore[return-value]


class HandEyeContext(Context):
    """Values a `HandEyeCalibrator` gives its task templates at each pyramid level."""

    CAMERA_INTRINSIC = 0
    """The level's camera intrinsic `[3, 3]` or per-observation `[N, 3, 3]`."""
    TARGET_MASKS = 1
    """Target silhouettes `[N, H, W]` of the level's observations; valid depth stands in when only depth is given."""
    TARGET_DEPTHS = 2
    """Target depth maps `[N, H, W]` in meters (0 = invalid), or None when the level has no depth term."""
    TARGET_LINK_MASK = 3
    """Rendered links each target shows, `[L + 1]` or `[N, L + 1]`; None for every link."""
    OBSERVATION_WEIGHTS = 4
    """Objective weight of each observation `[N]`, or None for ones."""
    LINK_POSES = 5
    """Nominal poses `[N, L, 4, 4]` of the rendered links in the reference frame."""
    LINK_VERTICES = 6
    """Visual mesh vertices of each rendered link."""
    LINK_FACES = 7
    """Visual mesh faces of each rendered link."""
    HEIGHT = 8
    """The level's image height."""
    WIDTH = 9
    """The level's image width."""
    BLUR_SIGMA = 10
    """The level's mask blur."""
    PATIENCE = 11
    """Iterations without improvement before the level stops."""
    ROBOT = 12
    """The calibrated robot."""
    Q = 13
    """Joint positions `[N, dofs]` of the level's observations."""
    LINK_INDICES = 14
    """Robot link index of each rendered link."""
    T_MOUNT_BASE = 15
    """Mount poses `[N, 4, 4]` of the level's observations, or None."""


class MaskAlignmentNormalEqTask(EagerTask):
    """Stream mask and depth normal equations without storing pixel Jacobians."""

    num_dofs = 6
    residual_weight = None
    batch_size: int = 0
    # subclasses that also solve model offsets (mount / joint zeros) set these
    num_extra_dofs: int = 0
    extra_col_offset: int = 6
    face_link_ids: Optional[torch.Tensor] = None
    normalize_by_union = False
    _optimize_camera = True
    _contour_geometry = False

    def _vertices(self, var_values: VarValues) -> torch.Tensor:
        """Return render vertices `[N, V, 3]`, or `[S, N, V, 3]` when offsets move them."""
        del var_values
        return self.v_base

    def _extra_screws(self, var_values: VarValues) -> Optional[wp.array]:
        """Return `[S, N, num_extra_dofs, num_links + 1]` offset screws in camera coordinates."""
        del var_values
        return None

    @property
    def residual_dim(self) -> int:
        return 0  # Pixel residuals stream directly into normal equations.

    def compute_weighted_residual(self, var_values: VarValues, *args: Any, **kwargs: Any) -> wp.array:
        raise NotImplementedError("MaskAlignmentNormalEqTask streams residuals directly into normal equations.")

    def __init__(
        self,
        camera_intrinsic: Union[torch.Tensor, Context] = HandEyeContext.CAMERA_INTRINSIC,
        target_masks: Union[torch.Tensor, Context] = HandEyeContext.TARGET_MASKS,
        link_poses: Union[torch.Tensor, Context] = HandEyeContext.LINK_POSES,
        link_vertices: Union[List[torch.Tensor], Context] = HandEyeContext.LINK_VERTICES,
        link_faces: Union[List[torch.Tensor], Context] = HandEyeContext.LINK_FACES,
        height: Union[int, Context] = HandEyeContext.HEIGHT,
        width: Union[int, Context] = HandEyeContext.WIDTH,
        blur_sigma: Union[float, Context] = HandEyeContext.BLUR_SIGMA,
        patience: Union[int, Context] = HandEyeContext.PATIENCE,
        target_depths: Union[torch.Tensor, Context, None] = HandEyeContext.TARGET_DEPTHS,
        target_link_mask: Union[torch.Tensor, Context, None] = HandEyeContext.TARGET_LINK_MASK,
        observation_weights: Union[torch.Tensor, Context, None] = HandEyeContext.OBSERVATION_WEIGHTS,
        depth_weight: float = 100.0,
        depth_huber_delta: float = 0.02,
    ):
        """Configure a template; `resolve` takes the observations the calibrator gives per pyramid level."""
        self.camera_intrinsic, self.target_masks, self.target_depths = camera_intrinsic, target_masks, target_depths
        self.link_poses, self.link_vertices, self.link_faces = link_poses, link_vertices, link_faces
        self.H, self.W, self.blur_sigma, self.patience = height, width, blur_sigma, patience
        self.target_link_mask, self.observation_weights = target_link_mask, observation_weights
        self.depth_weight, self.depth_huber_delta = depth_weight, depth_huber_delta

    def resolve(self, var_values: VarValues, context: Mapping[Context, Any]):
        """Take a level's observations and merge the robot's rendered meshes."""
        super().resolve(var_values, context)
        self.torch_device = self.target_masks.device
        self.N = self.target_masks.shape[0]
        if self.observation_weights is None:
            self.observation_weights = torch.ones(self.N, device=self.torch_device)
        self.glctx = _rasterize_context(self.torch_device.index)
        intrinsics = self.camera_intrinsic.expand(self.N, 3, 3)
        self._wp_focal = wp.from_torch(intrinsics[:, [0, 1], [0, 1]].contiguous())
        render_intrinsics = self.camera_intrinsic[None] if self.camera_intrinsic.ndim == 2 else self.camera_intrinsic
        self._projection = torch.stack(
            [intr_to_proj_mat(k, self.H, self.W) for k in render_intrinsics]
        ) @ _OPENCV2GL.to(self.torch_device)
        self.merged_faces, self.v_base = _precompute_merged_mesh(self.link_vertices, self.link_faces, self.link_poses)
        self.face_link_ids = torch.cat(
            [
                torch.full((faces.shape[0],), i + 1, dtype=torch.int32, device=self.torch_device)
                for i, faces in enumerate(self.link_faces)
            ]
        )

    def _alloc_buffers(self, batch_size: int, device: wp_device_type):
        self.batch_size = batch_size
        S, N, H, W = batch_size, self.N, self.H, self.W
        self.wp_device = device
        dev = self.torch_device
        self._rendered_curr = torch.empty(S * N, H, W, dtype=torch.float32, device=dev)
        self._positions_curr = torch.empty(S * N, H, W, 3, dtype=torch.float32, device=dev)
        self._rendered_prop = torch.empty(S * N, H, W, dtype=torch.float32, device=dev)
        self._positions_prop = torch.empty(S * N, H, W, 3, dtype=torch.float32, device=dev)
        self._wp_rendered_curr = wp.from_torch(self._rendered_curr)
        self._wp_rendered_prop = wp.from_torch(self._rendered_prop)
        target_exp = self.target_masks.repeat(S, 1, 1)
        weights_exp = self.observation_weights.repeat(S).contiguous()
        self._wp_observation_weights = wp.from_torch(weights_exp)
        # alias render buffers when blur is disabled
        if self.blur_sigma < 0.5:
            self._blurred_curr = self._rendered_curr
            self._blurred_prop = self._rendered_prop
            self._target_blurred = target_exp
        else:
            self._blurred_curr = torch.empty(S * N, H, W, dtype=torch.float32, device=dev)
            self._blurred_prop = torch.empty(S * N, H, W, dtype=torch.float32, device=dev)
            self._target_blurred = self._blur(target_exp, self.blur_sigma).contiguous()
        self._wp_blurred_curr = wp.from_torch(self._blurred_curr)
        self._wp_blurred_prop = wp.from_torch(self._blurred_prop)
        self._wp_target_blurred = wp.from_torch(self._target_blurred)
        if self.target_depths is not None:
            depth_exp = self.target_depths.repeat(S, 1, 1)
            self._wp_target_depth = wp.from_torch(depth_exp)
        self._wp_positions_curr = wp.from_torch(self._positions_curr)
        self._wp_positions_prop = wp.from_torch(self._positions_prop)
        self._pixel_link = torch.zeros(S * N, H, W, dtype=torch.int32, device=dev)
        self._pixel_link_prop = torch.empty_like(self._pixel_link) if self.num_extra_dofs else None
        self._wp_pixel_link = wp.from_torch(self._pixel_link)
        self._wp_screws = wp.zeros((max(S, 1), N, max(self.num_extra_dofs, 1), 1), dtype=wp_vec6, device=device)
        # pixel Jacobians are camera-frame; Ad(T) maps them onto the body-frame twist VarValues.integrate applies
        self._wp_adjoint_curr = wp.empty(S, dtype=wp_mat66, device=device)
        self._best_cost = float("inf")
        self._last_improvement = 0
        self._current_var: Optional[SE3Var] = None
        self._best_xyz_wxyz: Optional[np.ndarray] = None
        # mark whether current buffers already contain the accepted render
        self._curr_is_fresh = False
        self._current_weights = torch.ones(S * N, device=dev)
        self._proposed_weights = torch.ones(S * N, device=dev)
        self._union_gradient = torch.zeros(S * N, device=dev)
        self._wp_current_weights = wp.from_torch(self._current_weights)
        self._wp_proposed_weights = wp.from_torch(self._proposed_weights)
        self._wp_union_gradient = wp.from_torch(self._union_gradient)
        self._cost64 = wp.zeros(S, dtype=wp.float64, device=device)
        self._normal64: Optional[wp.array] = None
        self._gradient64: Optional[wp.array] = None

    def _normalize_mask_loss(self, proposed: bool):
        """Normalize image error by mask union, including its contribution to the gradient."""
        if not self.normalize_by_union:
            return
        rendered = self._blurred_prop if proposed else self._blurred_curr
        area = self._target_blurred.sum((-2, -1))
        union = torch.maximum(rendered, self._target_blurred).sum((-2, -1)).clamp(min=1e-8)
        scale = torch.where(area > 0, area / union, 1.0)
        weights = self._proposed_weights if proposed else self._current_weights
        weights.copy_(wp.to_torch(self._wp_observation_weights) * scale)
        if not proposed:
            squared_error = (rendered - self._target_blurred).square().sum((-2, -1))
            self._union_gradient.copy_(torch.where(area > 0, 0.5 * squared_error / union, 0.0))

    def _blur(self, masks: torch.Tensor, sigma: float) -> torch.Tensor:
        if sigma < 0.5:
            return masks
        ks = int(6 * sigma + 1) | 1
        x = torch.arange(ks, device=masks.device, dtype=torch.float32) - ks // 2
        k = torch.exp(-0.5 * (x / sigma) ** 2)
        k = k / k.sum()
        m = masks.unsqueeze(1)
        m = F.conv2d(F.pad(m, (0, 0, ks // 2, ks // 2), mode="replicate"), k.view(1, 1, -1, 1))
        m = F.conv2d(F.pad(m, (ks // 2, ks // 2, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
        return m.squeeze(1)

    def prepare(self, var_values: VarValues, proposed: bool, iter_idx: int):
        """Render the current or proposed camera transform into persistent buffers."""
        var = var_values.get(self.var_key)
        # --- proposal render ---
        if proposed:
            with torch.no_grad():
                T_torch = wp.to_torch(se3_to_matrix(var.xyz_wxyz))
                prop_masks, prop_positions, *link_ids = _render_batch(
                    self.glctx,
                    T_torch,
                    self.camera_intrinsic,
                    self.merged_faces,
                    self._vertices(var_values),
                    self.H,
                    self.W,
                    face_link_ids=self.face_link_ids
                    if self.num_extra_dofs or self.target_link_mask is not None
                    else None,
                    target_link_mask=self.target_link_mask,
                    projection=self._projection,
                )
                self._rendered_prop.copy_(prop_masks)
                self._positions_prop.copy_(prop_positions)
                if self.num_extra_dofs:
                    self._pixel_link_prop.copy_(link_ids[0])
                if self.blur_sigma >= 0.5:
                    self._blurred_prop.copy_(self._blur(self._rendered_prop, self.blur_sigma))
                self._normalize_mask_loss(True)
            return
        if self.batch_size != var.batch_size:
            self._alloc_buffers(var.batch_size, var.device)
        self._current_var = var
        wp.launch(
            se3_adjoint_kernel,
            dim=var.batch_size,
            inputs=[var.xyz_wxyz],
            outputs=[self._wp_adjoint_curr],
            device=self.wp_device,
        )
        # --- current render reuse ---
        # Discard cached renders and stopping state at the start of each solve.
        if iter_idx == 0:
            self._curr_is_fresh = False
            self._best_cost, self._last_improvement, self._best_xyz_wxyz = float("inf"), 0, None
        if self._curr_is_fresh:
            # reuse the accepted proposal render
            self._curr_is_fresh = False
            if self.num_extra_dofs:
                self._wp_screws = self._extra_screws(var_values)
            self._normalize_mask_loss(False)
            return
        self._curr_is_fresh = False
        with torch.no_grad():
            T_torch = wp.to_torch(se3_to_matrix(var.xyz_wxyz))
            rendered, positions, *link_ids = _render_batch(
                self.glctx,
                T_torch,
                self.camera_intrinsic,
                self.merged_faces,
                self._vertices(var_values),
                self.H,
                self.W,
                face_link_ids=self.face_link_ids if self.num_extra_dofs or self.target_link_mask is not None else None,
                target_link_mask=self.target_link_mask,
                projection=self._projection,
            )
            self._rendered_curr.copy_(rendered)
            self._positions_curr.copy_(positions)
            if self.num_extra_dofs:
                self._pixel_link.copy_(link_ids[0])
                self._wp_screws = self._extra_screws(var_values)
            if self.blur_sigma >= 0.5:
                self._blurred_curr.copy_(self._blur(self._rendered_curr, self.blur_sigma))
            self._normalize_mask_loss(False)

    def accumulate_normal_equations(
        self,
        var_values: VarValues,
        *,
        costs: wp.array,
        JtJ: Optional[wp.array] = None,
        Jtr: Optional[wp.array] = None,
    ):
        """Sum pixels in float64 and add one rounded system to the solver."""
        S, N, H, W = self.batch_size, self.N, self.H, self.W
        self._cost64.zero_()
        if JtJ is None:
            # --- cost step ---
            wp.launch(
                _compute_cost_kernel,
                dim=(S * N, H, W),
                inputs=[
                    self._wp_blurred_prop,
                    self._wp_target_blurred,
                    self._wp_proposed_weights if self.normalize_by_union else self._wp_observation_weights,
                    N,
                ],
                outputs=[self._cost64],
                device=self.wp_device,
            )
            if self.target_depths is not None:
                wp.launch(
                    _compute_depth_cost_kernel,
                    dim=(S * N, H, W),
                    inputs=[
                        self._wp_positions_prop,
                        self._wp_rendered_prop,
                        self._wp_target_depth,
                        self._wp_observation_weights,
                        self.depth_huber_delta,
                        self.depth_weight,
                        N,
                    ],
                    outputs=[self._cost64],
                    device=self.wp_device,
                )
            with torch.cuda.stream(cast(torch.cuda.Stream, wp.stream_to_torch(costs.device))):
                wp.to_torch(costs).add_(wp.to_torch(self._cost64))
            return
        # --- build step ---
        if self._normal64 is None or self._normal64.shape != JtJ.shape:
            self._normal64 = wp.zeros(JtJ.shape, dtype=wp.float64, device=self.wp_device)
            self._gradient64 = wp.zeros(Jtr.shape, dtype=wp.float64, device=self.wp_device)
        self._normal64.zero_()
        self._gradient64.zero_()
        wp.launch(
            _accumulate_normal_equations_kernel,
            dim=(S * N, H, W),
            inputs=[
                self._wp_blurred_curr,
                self._wp_target_blurred,
                self._wp_current_weights if self.normalize_by_union else self._wp_observation_weights,
                self._wp_union_gradient,
                self._optimize_camera,
                self._contour_geometry,
                self._wp_positions_curr,
                self._wp_focal,
                H,
                W,
                N,
                self._wp_adjoint_curr,
                self._wp_screws,
                self._wp_pixel_link,
                self.num_extra_dofs,
                self.extra_col_offset,
            ],
            outputs=[self._normal64, self._gradient64, self._cost64],
            device=self.wp_device,
        )
        if self.target_depths is not None:
            wp.launch(
                _accumulate_depth_normal_equations_kernel,
                dim=(S * N, H, W),
                inputs=[
                    self._wp_positions_curr,
                    self._wp_rendered_curr,
                    self._wp_target_depth,
                    self._wp_observation_weights,
                    self._wp_focal,
                    H,
                    W,
                    self.depth_huber_delta,
                    self.depth_weight,
                    N,
                    self._wp_adjoint_curr,
                    self._wp_screws,
                    self._wp_pixel_link,
                    self.num_extra_dofs,
                    self.extra_col_offset,
                ],
                outputs=[self._normal64, self._gradient64, self._cost64],
                device=self.wp_device,
            )
        with torch.cuda.stream(cast(torch.cuda.Stream, wp.stream_to_torch(costs.device))):
            wp.to_torch(JtJ).add_(wp.to_torch(self._normal64))
            wp.to_torch(Jtr).add_(wp.to_torch(self._gradient64))
            wp.to_torch(costs).add_(wp.to_torch(self._cost64))

    def _accept_render(self, accept_mask: wp.array):
        """Retain accepted pixels, positions and link identities for the next linearization."""
        # select accepted proposal buffers without a CPU synchronization
        S, N = self.batch_size, self.N
        # broadcast each candidate decision across its observations
        w = wp.to_torch(accept_mask).to(torch.float32).view(S, 1).expand(S, N).reshape(S * N, 1, 1)
        self._rendered_curr.lerp_(self._rendered_prop, w)
        self._positions_curr.lerp_(self._positions_prop, w.unsqueeze(-1))
        if self.num_extra_dofs:
            self._pixel_link.copy_(torch.where(w.bool(), self._pixel_link_prop, self._pixel_link))
        if self.blur_sigma >= 0.5:
            self._blurred_curr.lerp_(self._blurred_prop, w)
        self._curr_is_fresh = True

    def on_step(self, accept_mask: wp.array, iter_idx: int, costs: wp.array) -> bool:
        """Update accepted render buffers and return whether patience expired."""
        self._accept_render(accept_mask)
        # update early-stopping state
        costs_np = costs.numpy()
        m = float(costs_np.min())
        best_idx = int(np.argmin(costs_np))
        if m < self._best_cost - np.finfo(np.float32).eps * max(1.0, abs(m)):
            self._best_cost = m
            self._best_xyz_wxyz = self._current_var.xyz_wxyz.numpy()[best_idx].copy()
            self._last_improvement = iter_idx
        return iter_idx - self._last_improvement >= self.patience


# --- joint and mount offsets ------------------------------------------------
class MaskAlignmentOffsetTask(MaskAlignmentNormalEqTask):
    """Align masks while fitting joint zeros, absolute q or a link transform."""

    def __init__(
        self,
        offset_joints: Optional[Sequence[int]] = None,
        offset_link: Optional[str] = None,
        offset_link_dofs: Sequence[int] = (0, 1, 2, 3, 4, 5),
        robot: Union[Robot, Context] = HandEyeContext.ROBOT,
        q: Union[torch.Tensor, Context] = HandEyeContext.Q,
        link_indices: Union[np.ndarray, Context] = HandEyeContext.LINK_INDICES,
        T_mount_base: Union[torch.Tensor, Context, None] = HandEyeContext.T_MOUNT_BASE,
        optimize_q: bool = False,
        **kwargs: Any,
    ):
        """Configure free model dofs; optimize_q uses absolute, joint-limited robot states instead of offsets."""
        if offset_joints is None and offset_link is None:
            raise ValueError("pass offset_joints and/or offset_link")
        if not set(offset_link_dofs) <= set(range(6)):
            raise ValueError(f"offset_link_dofs must index the six twist dofs, got {offset_link_dofs}")
        super().__init__(**kwargs)
        self.offset_joints = None if offset_joints is None else np.asarray(offset_joints, dtype=np.int64)
        self.offset_link = offset_link
        self.offset_link_dofs = tuple(offset_link_dofs)
        self.num_extra_dofs = (0 if offset_joints is None else len(self.offset_joints)) + (6 if offset_link else 0)
        self.robot, self.q, self.link_indices, self.T_mount_base = robot, q, link_indices, T_mount_base
        self.optimize_q = optimize_q

    def resolve(self, var_values: VarValues, context: Mapping[Context, Any]):
        """Take the observations, then build the kinematic structure the offsets move."""
        super().resolve(var_values, context)
        spec, device = self.robot.spec, self.q.device
        link_indices = np.asarray(self.link_indices)
        self.link_indices = torch.from_numpy(link_indices).to(device)

        offset, self.vert_slices = 0, []
        for i, verts in enumerate(self.link_vertices):
            self.vert_slices.append((offset, offset + verts.shape[0]))
            offset += verts.shape[0]

        # (rendered link, joint, offset dof, dq_joint / d_offset) for every joint above a link
        dof_of_joint = {}
        if self.offset_joints is not None:
            dof_of_joint = {int(j): d for d, j in enumerate(self.offset_joints)}
        rows: List[List[float]] = []
        for out_idx, link_idx in enumerate(link_indices):
            joint = int(spec.link_parent_joint_indices[link_idx])
            while joint >= 0:
                act = int(spec.actuated_joint_indices[joint])
                mult = 1.0
                if act < 0:
                    act = int(spec.mimic_actuated_joint_indices[joint])
                    mult = float(spec.mimic_multipliers[joint])
                if act in dof_of_joint:
                    rows.append([out_idx + 1, joint, dof_of_joint[act], mult])
                joint = int(spec.parent_joint_indices[joint])
        pairs = np.array(rows, dtype=np.float64).reshape(-1, 4)
        self._pair_link = torch.from_numpy(pairs[:, 0].astype(np.int64)).to(device)
        self._pair_joint = torch.from_numpy(pairs[:, 1].astype(np.int64)).to(device)
        self._pair_dof = torch.from_numpy(pairs[:, 2].astype(np.int64)).to(device)
        self._pair_mult = torch.from_numpy(pairs[:, 3].astype(np.float32)).to(device)

        # child link and local twist of every joint, for the joint screw axes
        child = np.full(spec.num_joints, -1, dtype=np.int64)
        for link_idx, joint in enumerate(spec.link_parent_joint_indices):
            if joint >= 0:
                child[joint] = link_idx
        self._joint_child = torch.from_numpy(child).to(device)
        self._joint_twist = torch.from_numpy(np.asarray(spec.joint_twists, dtype=np.float32)).to(device)

        # the virtual joint: which links hang off offset_link, and its column mask
        self._mount_link_index = None
        if self.offset_link is not None:
            self._mount_link_index = spec.link_names.index(self.offset_link)
            moved = spec.link_ancestor_links_mask[:, self._mount_link_index].copy()
            moved[self._mount_link_index] = True
            self._moved_links = torch.from_numpy(moved).to(device)
            rendered_moved = np.concatenate([[0.0], moved[link_indices].astype(np.float32)])
            self._moved_rendered = torch.from_numpy(rendered_moved).to(device)
            dof_mask = np.zeros(6, dtype=np.float32)
            dof_mask[list(self.offset_link_dofs)] = 1.0
            self._link_dof_mask = torch.from_numpy(dof_mask).to(device)

    def initial_vars(self, num_candidates: int, device: Any) -> Dict[Enum, Var]:
        """Nominal-model variables this task adds to `VarValues` (zero offset, identity mount)."""
        out: Dict[Enum, Var] = {}
        if self.offset_joints is not None:
            if self.optimize_q:
                if not isinstance(self.robot, Robot) or list(self.offset_joints) != list(
                    range(self.robot.num_actuated_joints)
                ):
                    raise ValueError("optimize_q needs a robot and all actuated joints")
                state = self.robot.state(
                    wp.from_numpy(
                        np.tile(self.robot.spec.midrange_q, (num_candidates, 1)).astype(np.float32),
                        dtype=wp.float32,
                        device=device,
                    )
                )
                state.clamp_q_on_integrate = True
                out[VarKey.QPOS_OFFSET] = state
            else:
                out[VarKey.QPOS_OFFSET] = VectorVar(
                    wp.zeros((num_candidates, len(self.offset_joints)), dtype=wp.float32, device=device)
                )
        if self.offset_link is not None:
            identity = np.tile(np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32), (num_candidates, 1))
            out[VarKey.MOUNT_OFFSET] = SE3Var(wp.from_numpy(identity, dtype=wp_vec7, device=device))
        return out

    def _link_poses_for(self, var_values: VarValues) -> torch.Tensor:
        """Forward kinematics at the offset model: `[S, N, num_links, 4, 4]`."""
        q, N = self.q, self.q.shape[0]
        S = var_values.batch_size
        if self.offset_joints is not None:
            variable = var_values.get(VarKey.QPOS_OFFSET)
            if self.optimize_q:
                q = wp.to_torch(variable.q)[:, None].expand(S, N, -1)
            else:
                dq = wp.to_torch(cast(VectorVar, variable).x)  # [S, J]
                q = q[None].repeat(S, 1, 1)
                q[:, :, self.offset_joints] += dq[:, None]
        else:
            q = q[None].expand(S, -1, -1)
        poses = self.robot.forward_kinematics_via_matrix_torch(q.reshape(S * N, -1))
        if self.T_mount_base is not None:
            poses = self.T_mount_base.repeat(S, 1, 1)[:, None] @ poses
        if self.offset_link is not None:
            T_mount = poses[:, self._mount_link_index]
            T_offset = wp.to_torch(se3_to_matrix(cast(SE3Var, var_values.get(VarKey.MOUNT_OFFSET)).xyz_wxyz))
            shift = T_mount @ T_offset.repeat_interleave(N, dim=0) @ torch.linalg.inv(T_mount)
            poses = torch.where(self._moved_links[None, :, None, None], shift[:, None] @ poses, poses)
        return poses.reshape(S, N, *poses.shape[1:])

    def _vertices(self, var_values: VarValues) -> torch.Tensor:
        poses = self._link_poses_for(var_values)[:, :, self.link_indices]
        S, N = poses.shape[:2]
        out = torch.empty(S, N, self.v_base.shape[1], 3, device=self.v_base.device, dtype=torch.float32)
        for i, (start, end) in enumerate(self.vert_slices):
            T = poses[:, :, i]
            out[:, :, start:end] = (
                torch.einsum("snij,vj->snvi", T[..., :3, :3], self.link_vertices[i]) + T[..., None, :3, 3]
            )
        return out

    def _extra_screws(self, var_values: VarValues) -> wp.array:
        poses = self._link_poses_for(var_values)
        S, N = poses.shape[:2]
        D, L = self.num_extra_dofs, len(self.link_vertices)
        screws = torch.zeros(S, N, D, L + 1, 6, device=poses.device)
        if self.offset_joints is not None:
            # each joint's screw axis in the reference frame, read off its child link pose
            joint_poses = se3_from_matrix(wp.from_torch(poses[:, :, self._joint_child].contiguous(), dtype=wp.mat44))
            joint_screws = torch.einsum("...ij,...j->...i", wp.to_torch(se3_adjoint(joint_poses)), self._joint_twist)
            flat = screws.view(S, N, -1, 6)
            cols = self._pair_dof * (L + 1) + self._pair_link
            flat.index_add_(2, cols, joint_screws[:, :, self._pair_joint] * self._pair_mult[:, None])
        if self.offset_link is not None:
            # the virtual joint: the six columns of Ad(T_ref_mount), gated to the mounted subtree
            mount_poses = se3_from_matrix(
                wp.from_torch(poses[:, :, self._mount_link_index].contiguous(), dtype=wp.mat44)
            )
            link_screws = wp.to_torch(se3_adjoint(mount_poses)).transpose(-1, -2) * self._link_dof_mask[:, None]
            screws[:, :, D - 6 :] = link_screws[:, :, :, None] * self._moved_rendered[:, None]
        # the pixel Jacobian lives in camera coordinates, so move the screws there too
        cam = cast(SE3Var, var_values.get(self.var_key))
        camera_adjoint = wp.to_torch(se3_adjoint(cam.xyz_wxyz))[:, None, None, None]
        return wp.from_torch(torch.einsum("...ij,...j->...i", camera_adjoint, screws).contiguous(), dtype=wp_vec6)
