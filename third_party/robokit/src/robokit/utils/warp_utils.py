# pyright: reportGeneralTypeIssues=false
# pyright: reportArgumentType=false
from typing import Any, Optional, Sequence, Union

import numpy as np
import warp as wp
from warp.types import matrix, vector


# wp.Device is public in newer Warp; older versions (e.g. 1.10) only expose wp.context.Device.
wp_device_type = wp.Device if hasattr(wp, "Device") else wp.context.Device


def resolve_device(device: Optional[wp_device_type] = None) -> wp_device_type:
    """Resolve the device; None selects "cuda" (honoring ScopedDevice) when available, else CPU.

    Example:
        >>> resolve_device("cpu").is_cpu
        True
    """
    return wp.get_device(device if device is not None else ("cuda" if wp.is_cuda_available() else "cpu"))


class wp_vec6f(vector(length=6, dtype=wp.float32)):
    pass


class wp_vec7f(vector(length=7, dtype=wp.float32)):
    pass


class wp_mat66f(matrix(shape=(6, 6), dtype=wp.float32)):
    pass


wp_vec6 = wp_vec6f
wp_vec7 = wp_vec7f
wp_mat66 = wp_mat66f


# --- kernels ---------------------------------------------------------------
@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
    float v = x;
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_xor_sync(0xffffffff, v, offset);
    return v;
#else
    return x;
#endif
"""
)
def warp_sum(x: float) -> float:
    """The sum of `x` over the 32 lanes of a CUDA warp; `x` itself on CPU, where a kernel runs one lane per row."""
    ...


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
    float v = x;
    for (int offset = width >> 1; offset > 0; offset >>= 1) v += __shfl_xor_sync(0xffffffff, v, offset);
    return v;
#else
    return x;
#endif
"""
)
def group_sum(x: float, width: int) -> float:
    """The sum of `x` over each aligned group of `width` lanes (a power of two) of a CUDA warp; `x` itself on CPU."""
    ...


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
    float v = x;
    for (int offset = 16; offset > 0; offset >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, offset));
    return v;
#else
    return x;
#endif
"""
)
def warp_max(x: float) -> float:
    """The maximum of `x` over the 32 lanes of a CUDA warp; `x` itself on CPU, where a kernel runs one lane per row."""
    ...


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
    // each shuffle halves the values a lane carries, so lane j ends with its warp's sum of v[j]; warp 0 adds the warps'
    __shared__ float partial[8][32];
    __shared__ float total[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    float x[32];
    #pragma unroll
    for (int k = 0; k < 32; ++k) x[k] = v.c[k];
    #define ROBOKIT_HALVE(H)                                                                       \
        _Pragma("unroll") for (int k = 0; k < H; ++k) {                                            \
            const float keep = lane & H ? x[k + H] : x[k];                                         \
            x[k] = keep + __shfl_xor_sync(0xffffffff, lane & H ? x[k] : x[k + H], H);              \
        }
    ROBOKIT_HALVE(16) ROBOKIT_HALVE(8) ROBOKIT_HALVE(4) ROBOKIT_HALVE(2) ROBOKIT_HALVE(1)
    #undef ROBOKIT_HALVE
    __syncthreads();
    partial[warp][lane] = x[0];
    __syncthreads();
    if (warp == 0) {
        float t = 0.0f;
        for (int w = 0; w < (blockDim.x >> 5); ++w) t += partial[w][lane];
        total[lane] = t;
    }
    __syncthreads();
    return total[lane];
#else
    return v[0];
#endif
"""
)
def row_sum32(v: wp.types.vector(length=32, dtype=wp.float32)) -> float:
    """CUDA row sum of `v[j]` over up to 8 whole warps; each lane gets identical bits for `warp_shuffle`."""
    ...


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
    return __shfl_sync(0xffffffff, x, lane);
#else
    return x;
#endif
"""
)
def warp_shuffle(x: float, lane: int) -> float:
    """`x` of the given lane of the CUDA warp; `x` itself on CPU."""
    ...


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
    __syncthreads();
#endif
"""
)
def row_sync():
    """Wait for every thread of the row; memory written before is visible to every thread after."""
    ...


@wp.kernel
def seed_rand_states_kernel(seed: wp.int32, out_rand_states: wp.array1d(dtype=wp.uint32)):
    """Give each batch element an independent RNG state derived from one seed."""
    tid = wp.tid()
    out_rand_states[tid] = wp.uint32(seed + tid * 1099087573)


@wp.kernel
def _gather_2d_kernel(
    source: wp.array2d(dtype=Any),
    indices: wp.array1d(dtype=wp.int32),
    out: wp.array2d(dtype=Any),
):
    tid, j = wp.tid()  # pyright: ignore[reportAssignmentType]
    out[tid, j] = source[indices[tid], j]


# Warp 1.18's CPU replay finds a kernel by name, so one capture mixing a generic kernel's dtypes runs the wrong one.
@wp.kernel
def _gather_2d_vec7_kernel(
    source: wp.array2d(dtype=wp_vec7f),
    indices: wp.array1d(dtype=wp.int32),
    out: wp.array2d(dtype=wp_vec7f),
):
    tid, j = wp.tid()  # pyright: ignore[reportAssignmentType]
    out[tid, j] = source[indices[tid], j]


@wp.kernel
def _gather_3d_kernel(
    source: wp.array3d(dtype=Any),
    indices: wp.array1d(dtype=wp.int32),
    out: wp.array3d(dtype=Any),
):
    tid, i, j = wp.tid()  # pyright: ignore[reportAssignmentType]
    out[tid, i, j] = source[indices[tid], i, j]


@wp.kernel
def _gather_4d_kernel(
    source: wp.array4d(dtype=Any),
    indices: wp.array1d(dtype=wp.int32),
    out: wp.array4d(dtype=Any),
):
    tid, i, j, k = wp.tid()  # pyright: ignore[reportAssignmentType]
    out[tid, i, j, k] = source[indices[tid], i, j, k]


@wp.kernel
def _repeat_first_dim_1d_kernel(
    source: wp.array1d(dtype=Any),
    repeats: wp.int32,
    out: wp.array1d(dtype=Any),
):
    tid = wp.tid()
    src_idx = tid // int(repeats)  # pyright: ignore[reportOperatorIssue]
    out[tid] = source[src_idx]


@wp.kernel
def _repeat_first_dim_2d_kernel(
    source: wp.array2d(dtype=Any),
    repeats: wp.int32,
    out: wp.array2d(dtype=Any),
):
    tid = wp.tid()
    src_idx = tid // int(repeats)  # pyright: ignore[reportOperatorIssue]
    num_cols = out.shape[1]
    for j in range(num_cols):
        out[tid, j] = source[src_idx, j]


@wp.kernel
def _repeat_first_dim_3d_kernel(
    source: wp.array3d(dtype=Any),
    repeats: wp.int32,
    out: wp.array3d(dtype=Any),
):
    tid = wp.tid()
    src_idx = tid // int(repeats)  # pyright: ignore[reportOperatorIssue]
    dim1 = out.shape[1]
    dim2 = out.shape[2]
    for j in range(dim1):
        for k in range(dim2):
            out[tid, j, k] = source[src_idx, j, k]


@wp.kernel
def _repeat_first_dim_4d_kernel(
    source: wp.array4d(dtype=Any),
    repeats: wp.int32,
    out: wp.array4d(dtype=Any),
):
    tid = wp.tid()
    src_idx = tid // int(repeats)  # pyright: ignore[reportOperatorIssue]
    dim1 = out.shape[1]
    dim2 = out.shape[2]
    dim3 = out.shape[3]
    for i in range(dim1):
        for j in range(dim2):
            for k in range(dim3):
                out[tid, i, j, k] = source[src_idx, i, j, k]


@wp.kernel
def _tile_1d_kernel(
    source: wp.array1d(dtype=Any),
    repeats: wp.int32,
    out: wp.array1d(dtype=Any),
):
    tid = wp.tid()
    src_idx = tid % int(source.shape[0])  # pyright: ignore[reportOperatorIssue]
    out[tid] = source[src_idx]


@wp.kernel
def _tile_2d_kernel(
    source: wp.array2d(dtype=Any),
    repeats: wp.int32,
    out: wp.array2d(dtype=Any),
):
    tid = wp.tid()
    src_idx = tid % int(source.shape[0])  # pyright: ignore[reportOperatorIssue]
    num_cols = out.shape[1]
    for j in range(num_cols):
        out[tid, j] = source[src_idx, j]


@wp.kernel
def _tile_3d_kernel(
    source: wp.array3d(dtype=Any),
    repeats: wp.int32,
    out: wp.array3d(dtype=Any),
):
    tid = wp.tid()
    src_idx = tid % int(source.shape[0])  # pyright: ignore[reportOperatorIssue]
    dim1 = out.shape[1]
    dim2 = out.shape[2]
    for j in range(dim1):
        for k in range(dim2):
            out[tid, j, k] = source[src_idx, j, k]


@wp.kernel
def _tile_4d_kernel(
    source: wp.array4d(dtype=Any),
    repeats: wp.int32,
    out: wp.array4d(dtype=Any),
):
    tid = wp.tid()
    src_idx = tid % int(source.shape[0])  # pyright: ignore[reportOperatorIssue]
    dim1 = out.shape[1]
    dim2 = out.shape[2]
    dim3 = out.shape[3]
    for i in range(dim1):
        for j in range(dim2):
            for k in range(dim3):
                out[tid, i, j, k] = source[src_idx, i, j, k]


@wp.kernel
def _stack_1d_kernel(
    src: wp.array1d(dtype=Any),
    out: wp.array2d(dtype=Any),
    stack_idx: int,
    axis: int,
):
    i = wp.tid()
    if axis == 0:
        out[stack_idx, i] = src[i]
    else:
        out[i, stack_idx] = src[i]


@wp.kernel
def _stack_2d_kernel(
    src: wp.array2d(dtype=Any),
    out: wp.array3d(dtype=Any),
    stack_idx: int,
    axis: int,
):
    i, j = wp.tid()  # pyright: ignore
    if axis == 0:
        out[stack_idx, i, j] = src[i, j]
    elif axis == 1:
        out[i, stack_idx, j] = src[i, j]
    else:
        out[i, j, stack_idx] = src[i, j]


@wp.kernel
def _stack_3d_kernel(
    src: wp.array3d(dtype=Any),
    out: wp.array4d(dtype=Any),
    stack_idx: int,
    axis: int,
):
    i, j, k = wp.tid()  # pyright: ignore
    if axis == 0:
        out[stack_idx, i, j, k] = src[i, j, k]
    elif axis == 1:
        out[i, stack_idx, j, k] = src[i, j, k]
    elif axis == 2:
        out[i, j, stack_idx, k] = src[i, j, k]
    else:
        out[i, j, k, stack_idx] = src[i, j, k]


@wp.kernel
def _masked_copy_2d_kernel(
    mask: wp.array1d(dtype=wp.int32),
    source: wp.array2d(dtype=Any),
    out: wp.array2d(dtype=Any),
):
    tid, j = wp.tid()  # pyright: ignore[reportAssignmentType]
    if mask[tid] == 1:
        out[tid, j] = source[tid, j]


@wp.kernel
def _masked_copy_3d_kernel(
    mask: wp.array1d(dtype=wp.int32),
    source: wp.array3d(dtype=Any),
    out: wp.array3d(dtype=Any),
):
    tid, i, j = wp.tid()  # pyright: ignore[reportAssignmentType]
    if mask[tid] == 1:
        out[tid, i, j] = source[tid, i, j]


@wp.kernel
def _masked_copy_kernel(
    mask: wp.array1d(dtype=wp.int32),
    source: wp.array4d(dtype=Any),
    out: wp.array4d(dtype=Any),
):
    tid, i, j, k = wp.tid()  # pyright: ignore[reportAssignmentType]
    if mask[tid] == 1:
        out[tid, i, j, k] = source[tid, i, j, k]


# Overload the generic kernels for every dtype robokit uses up front: a later overload rehashes the module, making its Warp kernel-cache key depend on which dtypes a process happened to launch first.
for _kernel in [k for k in list(globals().values()) if isinstance(k, wp.Kernel) and k.is_generic]:
    _generic_args = [a for a in _kernel.adj.args if getattr(a.type, "dtype", None) is Any]
    for _dtype in (wp.float32, wp.int32, wp.uint32, wp.vec3f, wp_vec7f):
        wp.overload(_kernel, {a.label: wp.array(dtype=_dtype, ndim=a.type.ndim) for a in _generic_args})


# --- public functions ------------------------------------------------------
def masked_copy(mask: wp.array, src: wp.array, out: wp.array) -> wp.array:
    """
    Masked copy: `out[i] = src[i]` where `mask[i] == 1`, unchanged otherwise.

    Args:
        mask: 1D int32 array of 0/1 flags, shape [N].
        src: Array to copy from, shape [N, ...].
        out: Array to copy into (modified in-place), shape [N, ...].

    Returns:
        out array (modified in-place).

    Example:
        >>> out = wp.zeros(3, dtype=wp.int32)
        >>> masked_copy(wp.array([1, 0, 1], dtype=wp.int32), wp.array([7, 8, 9], dtype=wp.int32), out).numpy().tolist()
        [7, 0, 9]
    """
    batch_size = mask.shape[0]
    device = src.device

    if src.ndim == 1:
        src_2d = src.reshape((src.shape[0], 1))
        dst_2d = out.reshape((out.shape[0], 1))
        wp.launch(_masked_copy_2d_kernel, dim=(batch_size, 1), inputs=[mask, src_2d, dst_2d], device=device)
    elif src.ndim == 2:
        wp.launch(_masked_copy_2d_kernel, dim=(batch_size, src.shape[1]), inputs=[mask, src, out], device=device)
    elif src.ndim == 3:
        wp.launch(
            _masked_copy_3d_kernel, dim=(batch_size, src.shape[1], src.shape[2]), inputs=[mask, src, out], device=device
        )
    elif src.ndim == 4:
        wp.launch(
            _masked_copy_kernel,
            dim=(batch_size, src.shape[1], src.shape[2], src.shape[3]),
            inputs=[mask, src, out],
            device=device,
        )
    else:
        raise ValueError(f"masked_copy() only supports 1D, 2D, 3D, or 4D arrays, got ndim={src.ndim}")

    return out


def gather(source: wp.array, indices: wp.array, out: Optional[wp.array] = None) -> wp.array:
    """Gather first-axis rows by indices, equivalent to np.take(source, indices, axis=0) or source[indices].

    Args:
        source: Source array with shape [N, ...] to gather from.
        indices: 1D int32 array of row indices to gather, shape [M].
        out: Optional pre-allocated destination array with shape [M, ...].

    Returns:
        Gathered array with shape [M, ...] where M = len(indices).

    Example:
        >>> source = wp.array([[1, 2], [3, 4], [5, 6]], dtype=wp.int32)
        >>> gather(source, wp.array([2, 0], dtype=wp.int32)).numpy().tolist()
        [[5, 6], [1, 2]]
    """
    num_gather = indices.shape[0]
    device = source.device

    if out is None:
        out_shape = (num_gather,) + source.shape[1:]
        out = wp.empty(out_shape, dtype=source.dtype, device=device, requires_grad=source.requires_grad)

    if source.ndim == 1:
        source_2d = source.reshape((source.shape[0], 1))
        dest_2d = out.reshape((num_gather, 1))
        kernel = _gather_2d_vec7_kernel if source.dtype == wp_vec7f else _gather_2d_kernel
        wp.launch(kernel, dim=(num_gather, 1), inputs=[source_2d, indices, dest_2d], device=device)
    elif source.ndim == 2:
        wp.launch(_gather_2d_kernel, dim=(num_gather, source.shape[1]), inputs=[source, indices, out], device=device)
    elif source.ndim == 3:
        wp.launch(
            _gather_3d_kernel,
            dim=(num_gather, source.shape[1], source.shape[2]),
            inputs=[source, indices, out],
            device=device,
        )
    elif source.ndim == 4:
        wp.launch(
            _gather_4d_kernel,
            dim=(num_gather, source.shape[1], source.shape[2], source.shape[3]),
            inputs=[source, indices, out],
            device=device,
        )
    else:
        raise ValueError(f"gather() only supports 1D, 2D, 3D, or 4D arrays, got ndim={source.ndim}")

    return out


def repeat(array: wp.array, repeats: int, out: Optional[wp.array] = None) -> wp.array:
    """Repeat each first-axis element `repeats` times contiguously, as np.repeat(array, repeats, axis=0).

    Args:
        array: Source array with shape [N, ...].
        repeats: Number of times to repeat each element.
        out: Optional pre-allocated destination array with shape [N*repeats, ...].

    Example:
        >>> repeat(wp.array([10, 20], dtype=wp.int32), 3).numpy().tolist()
        [10, 10, 10, 20, 20, 20]
    """
    if repeats < 1:
        raise ValueError("repeat requires repeats >= 1")

    device = array.device
    out_shape = (array.shape[0] * repeats,) + array.shape[1:]
    if out is None:
        out = wp.empty(out_shape, dtype=array.dtype, device=device, requires_grad=array.requires_grad)

    if array.ndim == 1:
        wp.launch(
            _repeat_first_dim_1d_kernel,
            dim=out_shape[0],
            inputs=[array, repeats, out],
            device=device,
        )
    elif array.ndim == 2:
        wp.launch(
            _repeat_first_dim_2d_kernel,
            dim=out_shape[0],
            inputs=[array, repeats, out],
            device=device,
        )
    elif array.ndim == 3:
        wp.launch(
            _repeat_first_dim_3d_kernel,
            dim=out_shape[0],
            inputs=[array, repeats, out],
            device=device,
        )
    elif array.ndim == 4:
        wp.launch(
            _repeat_first_dim_4d_kernel,
            dim=out_shape[0],
            inputs=[array, repeats, out],
            device=device,
        )
    else:
        raise ValueError(f"repeat() only supports 1D, 2D, 3D, or 4D arrays, got ndim={array.ndim}")

    return out


def tile(array: wp.array, repeats: int) -> wp.array:
    """Repeat the whole first dimension `repeats` times, as np.tile(array, (repeats, 1, ...)).

    Example:
        >>> tile(wp.array([10, 20], dtype=wp.int32), 3).numpy().tolist()
        [10, 20, 10, 20, 10, 20]
    """
    if repeats < 1:
        raise ValueError("tile requires repeats >= 1")

    device = array.device
    out_shape = (array.shape[0] * repeats,) + array.shape[1:]
    out = wp.empty(out_shape, dtype=array.dtype, device=device, requires_grad=array.requires_grad)

    if array.ndim == 1:
        wp.launch(
            _tile_1d_kernel,
            dim=out_shape[0],
            inputs=[array, repeats, out],
            device=device,
        )
    elif array.ndim == 2:
        wp.launch(
            _tile_2d_kernel,
            dim=out_shape[0],
            inputs=[array, repeats, out],
            device=device,
        )
    elif array.ndim == 3:
        wp.launch(
            _tile_3d_kernel,
            dim=out_shape[0],
            inputs=[array, repeats, out],
            device=device,
        )
    elif array.ndim == 4:
        wp.launch(
            _tile_4d_kernel,
            dim=out_shape[0],
            inputs=[array, repeats, out],
            device=device,
        )
    else:
        raise ValueError(f"tile() only supports 1D, 2D, 3D, or 4D arrays, got ndim={array.ndim}")

    return out


def stack(arrays: Sequence[wp.array], axis: int = 0, out: Optional[wp.array] = None) -> wp.array:
    """Stack equal-shape/dtype arrays on a new axis, like np.stack; support 1D–3D inputs.

    Args:
        arrays: Sequence of arrays with identical shape and dtype.
        axis: Position of the new dimension in the output (default 0).
        out: Optional pre-allocated destination array.

    Example:
        >>> a, b = wp.array([1, 2], dtype=wp.int32), wp.array([3, 4], dtype=wp.int32)
        >>> stack([a, b]).numpy().tolist()
        [[1, 2], [3, 4]]
        >>> stack([a, b], axis=1).numpy().tolist()
        [[1, 3], [2, 4]]
    """
    num_arrays = len(arrays)
    first = arrays[0]
    ndim = first.ndim
    shape = first.shape
    device = first.device
    dtype = first.dtype

    out_shape = list(shape)
    out_shape.insert(axis, num_arrays)
    out_shape = tuple(out_shape)

    if out is None:
        out = wp.empty(out_shape, dtype=dtype, device=device)

    if ndim == 1:
        for idx, arr in enumerate(arrays):
            wp.launch(_stack_1d_kernel, dim=shape[0], inputs=[arr, out, idx, axis], device=device)
    elif ndim == 2:
        for idx, arr in enumerate(arrays):
            wp.launch(_stack_2d_kernel, dim=(shape[0], shape[1]), inputs=[arr, out, idx, axis], device=device)
    elif ndim == 3:
        for idx, arr in enumerate(arrays):
            wp.launch(
                _stack_3d_kernel,
                dim=(shape[0], shape[1], shape[2]),
                inputs=[arr, out, idx, axis],
                device=device,
            )
    else:
        raise ValueError(f"stack() supports 1D, 2D, or 3D input arrays, got ndim={ndim}")

    return out


def pose_array(target: Union[wp.array, np.ndarray], device: Optional[str] = None) -> wp.array:
    """Return `(batch, frames)` `wp_vec7` poses, uploading NumPy `(frames, 7)` or `(batch, frames, 7)` input.

    Example:
        >>> pose_array(np.zeros((2, 7), dtype=np.float32)).shape
        (1, 2)
    """
    if isinstance(target, wp.array):
        return target
    poses = np.ascontiguousarray(target, dtype=np.float32)
    return wp.array(poses[None] if poses.ndim == 2 else poses, dtype=wp_vec7, device=device)


__all__ = [
    "row_sum32",
    "row_sync",
    "group_sum",
    "warp_max",
    "warp_shuffle",
    "warp_sum",
    "pose_array",
    "wp_vec6f",
    "wp_vec7f",
    "wp_vec6",
    "wp_vec7",
    "wp_mat66f",
    "wp_mat66",
    "masked_copy",
    "gather",
    "repeat",
    "tile",
    "stack",
    "wp_device_type",
]
