"""Anatomical contact exclusions and exact sampled mesh clearance for IK."""

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from retarget.env.grasp.hand import palmar_axis, query_path

FORBIDDEN_CLEARANCE_M = 0.001
MAX_BONE_DEVIATION_DEG = 30.0


def belly_axis(model, body):
    name = model.body(int(body)).name
    endpoint = None
    if name.startswith("right_finger"):
        f, k = int(name[len("right_finger")]), int(name[-1])
        endpoint = (
            model.body(f"right_finger{f}_link{k + 1}").pos
            if k < 4
            else model.site(
                f"right_{('thumb', 'index', 'middle', 'ring', 'pinky')[f - 1]}_tip"
            ).pos
        )
    return palmar_axis(model, body, endpoint)


def contact_mask(model, body, normals, reference):
    """Keep the annotated side and prevent a palmar patch drifting to the rim."""
    belly = belly_axis(model, body)
    allowed = normals @ reference > 0.5
    if reference @ belly > 0.25:
        allowed &= normals @ belly > 0.25
    return allowed


def forbidden_mask(model, body, normals, annotated_bodies, reference_normals):
    """Dorsal skin is excluded unless that body explicitly annotates it.

    Side contacts remain legal. An explicitly dorsal annotation grants only
    its own normal region, rather than opening the entire back of the hand.
    """
    belly = belly_axis(model, body)
    forbidden = normals @ belly < -0.25
    for ref in np.asarray(reference_normals)[np.asarray(annotated_bodies) == body]:
        if ref @ belly < -0.25:
            forbidden &= normals @ ref <= 0.5
    return forbidden


def hand_forbidden_mask(hand, reference_normals):
    mask = np.zeros(len(hand.all_bodies), bool)
    for body in np.unique(hand.all_bodies):
        ids = np.flatnonzero(hand.all_bodies == body)
        with np.load(query_path(hand.model, int(body))) as queries:
            normals = queries["local_normal"]
            if len(ids) != len(normals):
                raise ValueError("Whole-hand queries must retain their asset ordering")
        mask[ids] = forbidden_mask(
            hand.model, int(body), normals, hand.bodies, reference_normals
        )
    return mask


class SurfaceClearance:
    """GPU exact SDF refresh with differentiable local clearance planes.

    Planes are refreshed between LM rounds. Final acceptance always queries
    the mesh again; a satisfied stale plane never certifies a configuration.
    """

    def __init__(self, hand, reference_normals, device, *, frame=None):
        from retarget.env.grasp.template import WujiContactKinematics
        from retarget.env.grasp.audit import GraspAudit

        self.fk = WujiContactKinematics(hand.model, hand.all_bodies, device)
        self.local = self.fk.tensor(hand.all_local)
        self.zero = torch.zeros_like(self.local)
        self.forbidden = torch.as_tensor(
            hand_forbidden_mask(hand, reference_normals), device=device
        )
        self.auditor = GraspAudit(hand.surface, str(device))
        self.rotation = self.fk.tensor(
            np.eye(3)
            if frame is None
            else Rotation.from_euler("XYZ", frame[3:6]).as_matrix()
        )
        self.translation = self.fk.tensor(np.zeros(3) if frame is None else frame[:3])
        self.anchor = torch.zeros_like(self.local)
        self.normal = torch.zeros_like(self.local)
        # Fit with a buffer for float32 mesh queries and changing mesh features.
        self.margin = torch.where(
            self.forbidden, FORBIDDEN_CLEARANCE_M + 0.0005, -0.0002
        )

    def points(self, q):
        p, _ = self.fk.contacts(q, self.local, self.zero)
        return p @ self.rotation.T + self.translation

    @torch.no_grad()
    def refresh(self, q):
        points = self.points(q)
        projected, normals, distance = self.auditor.query(points)
        delta = points - projected
        gradient = (
            distance.sign()[:, None]
            * delta
            / delta.norm(dim=-1, keepdim=True).clamp(min=1.0e-12)
        )
        gradient = torch.where(
            (distance.abs() > 1.0e-7)[:, None], gradient, normals.double()
        )
        self.anchor.copy_(projected)
        self.normal.copy_(gradient)
        return distance

    def residual(self, q):
        distance = ((self.points(q) - self.anchor) * self.normal).sum(-1)
        return (self.margin - distance).clamp(min=0.0)

    @torch.no_grad()
    def diagnostics(self, q):
        distance = self.refresh(q)
        minimum = (
            float(distance[self.forbidden].min())
            if bool(self.forbidden.any())
            else None
        )
        return dict(
            forbidden_region_min_gap_mm=None if minimum is None else minimum * 1000,
            forbidden_region_clearance_mm=FORBIDDEN_CLEARANCE_M * 1000,
            forbidden_region_samples=int(self.forbidden.sum()),
            forbidden_region_valid=minimum is None
            or minimum >= FORBIDDEN_CLEARANCE_M - 1.0e-6,
            whole_hand_penetration_mm=float((-distance.min()).clamp(min=0.0)) * 1000,
        )


def robust_contact(delta):
    """Bound each annotation's influence, so missing contacts cannot buy twist."""
    return delta / torch.sqrt(1.0 + delta.square().sum(-1, keepdim=True))
