"""Backend-neutral transform utilities (pure torch/numpy, no warp kernels): coord specs, look-at, 6D rotation."""

import re
from typing import TYPE_CHECKING, List, Literal, Union, overload

import numpy as np
from jaxtyping import Float


if TYPE_CHECKING:
    import torch


COMMON_COORDS = {
    "opencv": "x: right, y: down, z: front",
    "opengl": "x: right, y: up, z: back",
    "sapien": "x: front, y: left, z: up",  # https://sapien.ucsd.edu/docs/latest/tutorial/basic/hello_world.html#viewer
}

# Note it doesn't matter if `right` correspond to [1, 0, 0], the resulted matrix is the same
DIRECTIONS = {
    "right": np.array([1, 0, 0], dtype=np.float32),
    "left": np.array([-1, 0, 0], dtype=np.float32),
    "up": np.array([0, -1, 0], dtype=np.float32),
    "down": np.array([0, 1, 0], dtype=np.float32),
    "front": np.array([0, 0, 1], dtype=np.float32),
    "back": np.array([0, 0, -1], dtype=np.float32),
}


@overload
def coord_conversion(
    src_spec: str, dst_spec: str, check_handness: bool = ..., *, return_tensors: Literal["pt"]
) -> "Float[torch.Tensor, '3 3']": ...
@overload
def coord_conversion(
    src_spec: str, dst_spec: str, check_handness: bool = ..., *, return_tensors: Literal["np"]
) -> Float[np.ndarray, "3 3"]: ...
def coord_conversion(
    src_spec: str, dst_spec: str, check_handness: bool = True, return_tensors: Literal["np", "pt"] = "np"
) -> "Union[Float[np.ndarray, '3 3'], Float[torch.Tensor, '3 3']]":
    """
    Construct a rotation matrix based on given source and destination coordinate specifications.

    Args:
        src_spec: Source coordinate specification, e.g., "x: right, y: down, z: front" or "opencv".
        dst_spec: Destination coordinate specification, e.g., "x: right, y: up, z: back" or "opengl".
        check_handness: If True, checks if the rotation matrix preserves right-handedness.
        return_tensors: Return type of the rotation matrix, either "np" for NumPy array or "pt" for PyTorch tensor.

    Returns:
        A 3x3 rotation matrix converting coordinates from the source to the destination specification.

    Examples:
        >>> coord_conversion("opencv", "opengl")
        array([[ 1.,  0.,  0.],
               [ 0., -1.,  0.],
               [ 0.,  0., -1.]], dtype=float32)
        >>> coord_conversion("x: front, y: left, z: up", "x: left, y: up, z: front")
        array([[0., 1., 0.],
               [0., 0., 1.],
               [1., 0., 0.]], dtype=float32)
        >>> coord_conversion("x: right, y: down, z: front", "x: left, y: up, z: front")
        array([[-1.,  0.,  0.],
               [ 0., -1.,  0.],
               [ 0.,  0.,  1.]], dtype=float32)
        >>> coord_conversion("x: left, y: up, z: front", "x: front, y: left, z: up", return_tensors="pt")
        tensor([[0., 0., 1.],
                [1., 0., 0.],
                [0., 1., 0.]])
    """

    def parse_spec(spec: str) -> List[str]:
        spec = spec.strip().lower()
        if spec in COMMON_COORDS:
            coord = COMMON_COORDS[spec]
        else:
            coord = spec
        # use regex to parse the coordinate specification
        pattern = r"\s*(x|y|z)\s*:\s*(\w+)\s*"
        matches = re.findall(pattern, coord)
        if len(matches) != 3:
            raise ValueError(f"Invalid coordinate specification: '{spec}'.")
        dirs = {axis: direction for axis, direction in matches}
        if set(dirs.keys()) != {"x", "y", "z"}:
            raise ValueError(f"Invalid coordinate specification: '{spec}'.")
        return [dirs["x"], dirs["y"], dirs["z"]]

    src_dirs = parse_spec(src_spec)
    dst_dirs = parse_spec(dst_spec)

    src_basis = np.stack([DIRECTIONS[dir] for dir in src_dirs])
    dst_basis = np.stack([DIRECTIONS[dir] for dir in dst_dirs])

    rot_mat = dst_basis @ src_basis.T

    if check_handness and np.linalg.det(rot_mat) < 0:
        raise RuntimeWarning("The rotation matrix is not right-handed.")

    if return_tensors == "pt":
        import torch

        return torch.from_numpy(rot_mat).float()
    elif return_tensors == "np":
        return rot_mat.astype(np.float32)
    else:
        raise ValueError(f"Invalid return_tensors: '{return_tensors}'")


# Ref: https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/renderer/cameras.py
def look_at_rotation(
    camera_position: "Float[torch.Tensor, '*batch 3']",
    at: "Float[torch.Tensor, '*batch 3']",
    up: "Float[torch.Tensor, '*batch 3']",
) -> "Float[torch.Tensor, '*batch 3 3']":
    """
    Camera-to-world rotation in OpenGL coords (x right, y up, z backward); column-major, unlike pytorch3d.

    Args:
        camera_position: position of the camera in world coordinates
        at: position of the object in world coordinates
        up: vector specifying the up direction in the world coordinate frame.

    Returns:
        R: rotation matrices of shape [..., 3, 3]

    Example:
        >>> import torch
        >>> R = look_at_rotation(torch.tensor([0.0, 0.0, 5.0]), torch.zeros(3), torch.tensor([0.0, 1.0, 0.0]))
        >>> torch.allclose(R, torch.eye(3))  # camera on +z looking at the origin
        True
    """
    import torch
    import torch.nn.functional as F

    dtype, device = camera_position.dtype, camera_position.device
    at, up = torch.broadcast_to(at, camera_position.shape), torch.broadcast_to(up, camera_position.shape)
    z_axis = F.normalize(camera_position - at, eps=1e-5, dim=-1)
    x_axis = F.normalize(torch.cross(up, z_axis, dim=-1), eps=1e-5, dim=-1)
    is_close = torch.isclose(x_axis, torch.tensor(0.0, dtype=dtype, device=device), atol=5e-3)
    is_close = is_close.all(dim=-1, keepdim=True)
    if is_close.any():
        # `up` is nearly parallel to `z_axis`; use the world axis least aligned with `z_axis` as a stand-in `up`.
        eye = torch.eye(3, dtype=dtype, device=device)
        least_aligned = z_axis.abs().argmin(dim=-1)
        alt_up = eye[least_aligned]
        alt_x = F.normalize(torch.cross(alt_up, z_axis, dim=-1), eps=1e-5, dim=-1)
        x_axis = torch.where(is_close, alt_x, x_axis)
    y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=-1), eps=1e-5, dim=-1)
    rot_mat = torch.cat((x_axis[..., None, :], y_axis[..., None, :], z_axis[..., None, :]), dim=-2)
    return rot_mat.transpose(-2, -1)


def rotation_6d_to_matrix(d6: "Float[torch.Tensor, '... 6']") -> "Float[torch.Tensor, '... 3 3']":
    """Convert 6D rotation representation (Zhou et al. 2019) to rotation matrix via Gram-Schmidt.

    Args:
        d6: 6D rotation representation, shape [..., 6].

    Returns:
        Rotation matrices, shape [..., 3, 3].

    Example:
        >>> import torch
        >>> d6 = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        >>> rotation_6d_to_matrix(d6)
        tensor([[1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.]])

    Reference:
        Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
        On the Continuity of Rotation Representations in Neural Networks. CVPR 2019.
    """
    import torch
    import torch.nn.functional as F

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(rot_mat: "Float[torch.Tensor, '... 3 3']") -> "Float[torch.Tensor, '... 6']":
    """Convert rotation matrix to 6D representation (Zhou et al. 2019) by dropping the last row.

    Args:
        rot_mat: Rotation matrices, shape [..., 3, 3].

    Returns:
        6D rotation representation, shape [..., 6].

    Example:
        >>> import torch
        >>> rot_mat = torch.eye(3)
        >>> matrix_to_rotation_6d(rot_mat)
        tensor([1., 0., 0., 0., 1., 0.])
    """
    batch_dim = rot_mat.size()[:-2]
    return rot_mat[..., :2, :].clone().reshape(batch_dim + (6,))


__all__ = [
    "coord_conversion",
    "look_at_rotation",
    "rotation_6d_to_matrix",
    "matrix_to_rotation_6d",
]
