"""Immutable contact-slot data shared by CEM and GraspADMM."""

from __future__ import annotations

from dataclasses import dataclass

import copy


from types import MappingProxyType


import numpy as np


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return copy.deepcopy(value)


def _plain(value):
    if isinstance(value, (dict, MappingProxyType)):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return copy.deepcopy(value)


def _immutable_array(value, shape, label):
    import torch

    if isinstance(value, torch.Tensor):
        if tuple(value.shape) != shape or not bool(torch.isfinite(value).all()):
            raise ValueError(f"{label} must be finite with shape {shape}")
        return value.detach().clone()
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite with shape {shape}")
    # A bytes-backed array cannot have writeability re-enabled by its consumer.
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(shape)


@dataclass(frozen=True)
class TemplateContacts:
    body_names: tuple[str, ...]
    local_points: np.ndarray
    local_normals: np.ndarray
    finger_qpos: np.ndarray
    required_cbody: tuple[tuple[str, ...], ...]
    provenance: dict

    def __post_init__(self):
        names = tuple(self.body_names)
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("Template requires nonempty contact body names")
        groups = tuple(tuple(group) for group in self.required_cbody)
        if not groups or any(not group or set(group) - set(names) for group in groups):
            raise ValueError("Required contact groups must refer to annotated bodies")
        points = _immutable_array(self.local_points, (len(names), 3), "local_points")
        normals = _immutable_array(self.local_normals, (len(names), 3), "local_normals")
        if not np.allclose(
            np.linalg.norm(normals, axis=-1), 1.0, rtol=1.0e-6, atol=1.0e-6
        ):
            raise ValueError("Template local normals must be unit vectors")
        object.__setattr__(self, "body_names", names)
        object.__setattr__(self, "local_points", points)
        object.__setattr__(self, "local_normals", normals)
        object.__setattr__(
            self,
            "finger_qpos",
            _immutable_array(self.finger_qpos, (20,), "finger_qpos"),
        )
        object.__setattr__(self, "required_cbody", groups)
        object.__setattr__(self, "provenance", _freeze(dict(self.provenance)))

    def to_dict(self):
        return dict(
            body_names=list(self.body_names),
            local_points=self.local_points.tolist(),
            local_normals=self.local_normals.tolist(),
            finger_qpos=self.finger_qpos.tolist(),
            required_cbody=[list(group) for group in self.required_cbody],
            provenance=_plain(self.provenance),
        )
