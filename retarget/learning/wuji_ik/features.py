"""Metres, xyzw quaternions, column-wise rotation 6D and physical joint limits."""
import json
import xml.etree.ElementTree as ET
import numpy as np
import torch
from torch import nn

SIDES = ('right', 'left')
FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')


def joint_names(side):
    result = []
    for finger in FINGERS:
        stem = finger if finger in ('thumb', 'pinky') else finger + '_finger'
        suffixes = ('cmc_flex', 'cmc_abd', 'mcp', 'ip') if finger == 'thumb' else ('mcp_flex', 'mcp_abd', 'pip', 'dip')
        result.extend(side[0] + '_' + stem + '_' + s for s in suffixes)
    return result


def urdf_limits(path, side):
    nodes = {j.get('name'): j for j in ET.parse(path).getroot().findall('joint')}
    limits = []
    for name in joint_names(side):
        j = nodes[name]
        if j.get('type') not in ('revolute', 'prismatic'):
            raise ValueError('Expected actuated joint: ' + name)
        limits.append([float(j.find('limit').get(k)) for k in ('lower', 'upper')])
    a = np.asarray(limits, np.float32)
    if not np.isfinite(a).all() or np.any(a[:, 0] >= a[:, 1]):
        raise ValueError('Invalid physical joint limits')
    return a


def wrist_tx_transform(path):
    from scipy.spatial.transform import Rotation
    nodes = ET.parse(path).getroot().findall('joint')
    node = next(j for j in nodes if j.find('child').get('link') == 'emf_tx_base')
    if node.get('type') != 'fixed' or node.find('parent').get('link') != 'wrist':
        raise ValueError('Expected fixed wrist -> emf_tx_base transform')
    origin = node.find('origin')
    r = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
    t = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
    tf = np.eye(4); tf[:3, :3] = r; tf[:3, 3] = t
    return tf


def wrist_emf_features(position, orientation_xyzw, wrist_T_tx=None):
    """[...,5,3/4] -> [...,5,9]. No sequence-dependent normalization."""
    from scipy.spatial.transform import Rotation
    p = np.asarray(position, dtype=np.float64)
    q = np.asarray(orientation_xyzw, dtype=np.float64)
    if p.shape[-2:] != (5, 3) or q.shape != p.shape[:-1] + (4,):
        raise ValueError('Expected five EMF poses with matching leading dimensions')
    if not np.isfinite(p).all() or not np.isfinite(q).all() or np.any(np.linalg.norm(q, axis=-1) < 1e-8):
        raise ValueError('Invalid EMF position or quaternion')
    r = Rotation.from_quat(q.reshape(-1, 4)).as_matrix().reshape(*p.shape[:-1], 3, 3)
    if wrist_T_tx is not None:
        tf = np.asarray(wrist_T_tx)
        p = np.einsum('ij,...fj->...fi', tf[:3, :3], p) + tf[:3, 3]
        r = tf[:3, :3] @ r
    rotation6d = r[..., :, :2].swapaxes(-1, -2).reshape(*p.shape[:-1], 6)
    return np.concatenate((p / .1, rotation6d), axis=-1).astype(np.float32)


class JointNormalizer(nn.Module):
    """Side 0=right, 1=left. Buffers travel with device/checkpoint."""
    def __init__(self, limits):
        super().__init__()
        a = torch.as_tensor(limits, dtype=torch.float32)
        if a.shape != (2, 20, 2) or not torch.isfinite(a).all() or torch.any(a[..., 0] >= a[..., 1]):
            raise ValueError('Expected physical limits [right/left,20,lower/upper]')
        self.register_buffer('limits', a.clone())

    def bounds(self, q, side):
        bounds = self.limits[side]
        lo, hi = bounds[..., 0], bounds[..., 1]
        while lo.ndim < q.ndim:
            lo, hi = lo.unsqueeze(-2), hi.unsqueeze(-2)
        return lo, hi

    def normalize(self, q, side):
        lo, hi = self.bounds(q, side)
        return 2 * (q - lo) / (hi - lo) - 1

    def denormalize(self, q, side):
        lo, hi = self.bounds(q, side)
        return (q + 1) * .5 * (hi - lo) + lo


def check_joint_order(group, side):
    if json.loads(group.attrs['q_joint_names_json']) != joint_names(side):
        raise ValueError('Cache joint order disagrees with named Hand2 URDF joints')
