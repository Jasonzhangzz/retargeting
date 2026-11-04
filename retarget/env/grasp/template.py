"""GPU WUJI2 forward kinematics shared by initialization and pose scoring."""

import numpy as np

from scipy.spatial.transform import Rotation

import torch


class WujiContactKinematics:
    """Differentiable four-joint finger chains using the actual MuJoCo model.

    The six root variables are xyz + intrinsic XYZ, as in right.xml. Bodies
    with other semantics are rejected rather than silently assigned a finger.
    """

    def __init__(self, model, bodies, device):
        self.device = torch.device(device)
        if self.device.type not in ("cuda", "cpu"):
            raise ValueError("Wuji kinematics requires a CPU or CUDA Torch device")
        self.dtype = torch.float64
        ids = np.array(
            [
                [model.body(f"right_finger{f}_link{k}").id for k in range(1, 5)]
                for f in range(1, 6)
            ]
        )
        joints = model.body_jntadr[ids]
        if not np.all(model.body_jntnum[ids] == 1):
            raise ValueError("Expected one hinge per Wuji finger body")
        self.position = self.tensor(model.body_pos[ids])
        self.rotation = self.tensor(
            Rotation.from_quat(model.body_quat[ids].reshape(-1, 4)[:, [1, 2, 3, 0]])
            .as_matrix()
            .reshape(5, 4, 3, 3)
        )
        self.pivot = self.tensor(model.jnt_pos[joints])
        self.identity = torch.eye(3, device=self.device, dtype=self.dtype)
        self.skew = self.cross_matrix(self.tensor(model.jnt_axis[joints]))
        self.skew_squared = self.skew @ self.skew
        self.root_skew = self.cross_matrix(self.identity)
        finger, link, palm = [], [], []
        for body in bodies:
            name = model.body(int(body)).name
            if name == "right_palm_link":
                finger.append(0)
                link.append(0)
                palm.append(True)
            elif name.startswith("right_finger") and body in ids:
                f, k = np.argwhere(ids == body)[0]
                finger.append(f)
                link.append(k)
                palm.append(False)
            else:
                raise ValueError(f"Unsupported template contact body: {name}")
        self.finger = torch.tensor(finger, device=self.device)
        self.link = torch.tensor(link, device=self.device)
        self.palm = torch.tensor(palm, device=self.device)

    def tensor(self, value):
        return torch.as_tensor(
            np.asarray(value).copy(), device=self.device, dtype=self.dtype
        )

    @staticmethod
    def cross_matrix(a):
        x, y, z = a.unbind(-1)
        zero = torch.zeros_like(x)
        return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(
            *a.shape[:-1], 3, 3
        )

    def frames(self, q):
        angles = q[6:].reshape(5, 4)
        joint_rotation = (
            self.identity
            + angles.sin()[..., None, None] * self.skew
            + (1 - angles.cos())[..., None, None] * self.skew_squared
        )
        local_rotation = self.rotation @ joint_rotation
        local_position = (
            self.position
            + (
                self.rotation
                @ (self.pivot - (joint_rotation @ self.pivot[..., None])[..., 0])[
                    ..., None
                ]
            )[..., 0]
        )
        positions, rotations = [], []
        p = torch.zeros((5, 3), device=self.device, dtype=self.dtype)
        r = self.identity.expand(5, 3, 3)
        for depth in range(4):
            p = p + (r @ local_position[:, depth, :, None])[..., 0]
            r = r @ local_rotation[:, depth]
            positions.append(p)
            rotations.append(r)
        root_parts = (
            self.identity
            + q[3:6].sin()[:, None, None] * self.root_skew
            + (1 - q[3:6].cos())[:, None, None] * (self.root_skew @ self.root_skew)
        )
        root = root_parts[0] @ root_parts[1] @ root_parts[2]
        p = torch.stack(positions, 1)[self.finger, self.link]
        r = torch.stack(rotations, 1)[self.finger, self.link]
        p = torch.where(self.palm[:, None], torch.zeros_like(p), p)
        r = torch.where(self.palm[:, None, None], self.identity, r)
        return p @ root.T + q[:3], root @ r

    def contacts(self, q, local, normals):
        p, r = self.frames(q)
        return p + (r @ local[..., None])[..., 0], (r @ normals[..., None])[..., 0]
