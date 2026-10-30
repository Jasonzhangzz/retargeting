"""Contact-aware Shadow -> Wuji template fitting, before Dexonomy initialization.

This changes input construction only. The object pose/contact initializer and
the shared sampling solvers are untouched. All fitting and surface selection
run on CUDA; MuJoCo supplies immutable model geometry.
"""

from retarget.env.grasp.capture import capture_guard
import time

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from retarget.env.grasp.hand import query_path
from retarget.env.grasp.regions import (
    MAX_BONE_DEVIATION_DEG,
    SurfaceClearance,
    contact_mask,
    robust_contact,
)


class WujiContactKinematics:
    """Differentiable four-joint finger chains using the actual MuJoCo model.

    The six root variables are xyz + intrinsic XYZ, as in right.xml. Bodies
    with other semantics are rejected rather than silently assigned a finger.
    """

    def __init__(self, model, bodies, device):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("Template fitting requires CUDA")
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
            Rotation.from_quat(model.body_quat[ids].reshape(-1, 4), scalar_first=True)
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


class BoneReference:
    """All fifteen bone directions in the palm frame, independent of root pose."""

    def __init__(self, model, qpos, device):
        bodies, points = [], []
        for f, name in enumerate(("thumb", "index", "middle", "ring", "pinky"), 1):
            ids = [model.body(f"right_finger{f}_link{k}").id for k in (2, 3, 4)]
            bodies.extend([*ids, ids[-1]])
            points.extend(
                [
                    np.zeros(3),
                    np.zeros(3),
                    np.zeros(3),
                    model.site(f"right_{name}_tip").pos,
                ]
            )
        self.fk = WujiContactKinematics(model, np.array(bodies), device)
        self.local = self.fk.tensor(points)
        self.zero = torch.zeros_like(self.local)
        self.root = self.fk.tensor(np.zeros(6))
        self.target = self.directions(self.fk.tensor(qpos)).detach()
        # Closing a nearly extended chain changes the gesture (e.g. the index
        # in an extension grasp). Curled chains still adapt to object size.
        straight = ((self.target[:, 0] * self.target[:, -1]).sum(-1) > 0.7) & (
            torch.arange(5, device=self.fk.device) != 0
        )
        # Thumb opposition needs freedom even when its own chain is straight.
        self.weights = torch.where(straight, 100.0, 1.0).to(self.fk.dtype)

    def directions(self, q):
        canonical = torch.cat((self.root, q[6:]))
        p, _ = self.fk.contacts(canonical, self.local, self.zero)
        v = p.reshape(5, 4, 3).diff(dim=1)
        return v / v.norm(dim=-1, keepdim=True).clamp(min=1.0e-12)

    def residual(self, q):
        return (self.directions(q) - self.target).flatten()

    def style_residual(self, q):
        return (
            (self.directions(q) - self.target) * self.weights[:, None, None]
        ).flatten()

    def accepts(self, q):
        return (
            (self.directions(q) * self.target).sum(-1)
            >= np.cos(np.deg2rad(MAX_BONE_DEVIATION_DEG))
        ).all()

    def maximum_angle(self, q):
        cosine = (self.directions(q) * self.target).sum(-1).clamp(-1.0, 1.0)
        return float(torch.rad2deg(torch.acos(cosine)).max())


def skin_reference(hand, reference_normals=None):
    normals = np.array(
        hand.normals if reference_normals is None else reference_normals, copy=True
    )
    if normals.shape != hand.normals.shape or not np.isfinite(normals).all():
        raise ValueError("Expected one finite reference normal per annotated contact")
    lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
    if np.any(lengths < 1.0e-12):
        raise ValueError("Zero contact reference normal")
    return normals / lengths


def contact_targets(shadow, posed, contacts, template_q, opening=None):
    """Preserve annotated contact layout in the aligned Wuji palm frame.

    For cylinders/spheres, resize the opposed opening to their diameter. This
    is a Wuji input adaptation, not an object pose or surface-contact result:
    both of those are still obtained from the original Dexonomy matcher.
    """
    points, normals = [], []
    for name, local in contacts:
        body = shadow.body(name).id
        rotation = posed.xmat[body].reshape(3, 3)
        points.append(posed.xpos[body] + rotation @ local[:3])
        normals.append(rotation @ local[3:])
    root = Rotation.from_euler("XYZ", template_q[3:6])
    points = root.inv().apply(np.asarray(points) - template_q[:3])
    normals = root.inv().apply(normals)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    info = dict(requested_opening_m=opening, opening_resized=False)
    if opening is not None and len(points) > 1:
        # The leading normal axis identifies opposed surfaces without relying
        # on contact ordering or naming a particular grasp type/finger.
        _, vectors = np.linalg.eigh(normals.T @ normals)
        axis = vectors[:, -1]
        dots = normals @ axis
        positive, negative = dots > 0.5, dots < -0.5
        if positive.any() and negative.any():
            coordinate = points @ axis
            a, b = coordinate[positive].mean(), coordinate[negative].mean()
            gap = b - a
            if gap > 0.005:
                points += ((coordinate - (a + b) / 2) * (opening / gap - 1))[
                    :, None
                ] * axis
                info.update(opening_resized=True, source_opening_m=float(gap))
    return points, normals, info


def fit_contact_template(
    hand,
    shadow,
    posed,
    contacts,
    template_q,
    *,
    device,
    opening=None,
    iterations=45,
    surface_rounds=3,
    object_contacts=None,
    joint_prior=None,
    reference_qpos=None,
    reference_normals=None,
):
    """Fit the root + joints and choose real query points, with joint limits.

    Root and joints move together, with bounded bone-direction changes.
    Contacts are optional robust targets; object-aware calls additionally
    enforce whole-hand clearance through refreshed mesh-distance planes.
    """
    begun = time.perf_counter()
    from retarget.env.grasp.initialization import template_contact_layout

    before = template_contact_layout(hand, template_q)
    if object_contacts is None:
        target_p, target_n, opening_info = contact_targets(
            shadow, posed, contacts, template_q, opening
        )
        target_source = "Shadow annotated contact layout"
    else:
        # These are the unedited surface contacts/normals returned by the
        # original matcher, expressed in the current fixed object frame.
        root = Rotation.from_euler("XYZ", template_q[3:6])
        target_p = root.inv().apply(object_contacts[0] - template_q[:3])
        target_n = root.inv().apply(object_contacts[1])
        opening_info = dict(requested_opening_m=None, opening_resized=False)
        target_source = "original Dexonomy object contacts and inward normals"
    fk = WujiContactKinematics(hand.model, hand.bodies, device)
    tp, tn = fk.tensor(target_p), fk.tensor(target_n)
    local, normals = fk.tensor(hand.local), fk.tensor(hand.normals)
    start = fk.tensor(np.r_[np.zeros(6), template_q[6:]])
    q = start.clone()
    joint_ids = np.array(
        [np.flatnonzero(hand.model.jnt_qposadr == i)[0] for i in range(26)]
    )
    low, high = (fk.tensor(v) for v in hand.model.jnt_range[joint_ids].T)
    prior = fk.tensor(np.r_[np.full(3, 100.0), np.ones(23)]) * 0.03
    posture = start.clone()
    reference_qpos = np.array(
        template_q if reference_qpos is None else reference_qpos, copy=True
    )
    posture[6:] = fk.tensor(reference_qpos[6:])
    center_weight = 0.03 * (joint_prior or 0.0)
    center = (low[6:] + high[6:]) / 2
    bones = BoneReference(hand.model, reference_qpos, device)
    region_normals = skin_reference(hand, reference_normals)
    from retarget.env.grasp.input_geometry import WujiSelfCollision

    capsules = WujiSelfCollision(hand.model, device)
    clearance = (
        SurfaceClearance(hand, region_normals, device, frame=template_q)
        if object_contacts is not None
        else None
    )
    if clearance is not None:
        clearance.refresh(q)
    eye = torch.eye(26, device=fk.device, dtype=fk.dtype)
    damping = fk.tensor(0.01)

    def residual(value):
        p, n = fk.contacts(value, local, normals)
        delta = (p - tp) / 0.01
        reach = 1.0 / (1.0 + delta.square().sum(-1))
        parts = [
            robust_contact(delta).flatten(),
            ((n - tn) * reach[:, None]).flatten(),
            prior * (value - posture),
            center_weight * (value[6:] - center),
            2.0 * bones.style_residual(value),
            (capsules.depths(value) - 0.0003).clamp(min=0.0) * 5000,
        ]
        if clearance is not None:
            parts.append(clearance.residual(value) * 5000)
        return torch.cat(parts)

    def step():
        r = residual(q)
        jac = torch.func.jacfwd(residual)(q)
        delta = torch.linalg.solve_ex(jac.T @ jac + damping * eye, -jac.T @ r)[0]
        trial = (q + delta.clamp(-0.2, 0.2)).clamp(low, high)
        accept = (residual(trial).square().sum() < r.square().sum()) & bones.accepts(
            trial
        )
        q.copy_(torch.where(accept, trial, q))
        damping.copy_(
            torch.where(accept, damping * 0.5, damping * 4).clamp(1.0e-5, 1.0e6)
        )

    # Batched padded query surfaces keep discrete selection on the GPU too.
    surfaces = []
    for body in hand.bodies:
        name = hand.model.body(int(body)).name
        with np.load(query_path(hand.model, name)) as queries:
            surfaces.append((queries["local_pos"], queries["local_normal"]))
    maximum = max(len(p) for p, _ in surfaces)
    pool_p, pool_n = np.zeros((2, len(surfaces), maximum, 3))
    valid = np.zeros((len(surfaces), maximum), bool)
    for i, (p, n) in enumerate(surfaces):
        pool_p[i, : len(p)], pool_n[i, : len(n)] = p, n
        valid[i, : len(p)] = contact_mask(
            hand.model, int(hand.bodies[i]), n, region_normals[i]
        )
        patch = getattr(hand, "tip_patches", {}).get(int(hand.bodies[i]))
        if patch is not None:
            valid[i, : len(p)] &= patch["mask"]
    if not valid.any(axis=1).all():
        raise ValueError("No query samples in an annotated contact region")
    pool_p, pool_n = fk.tensor(pool_p), fk.tensor(pool_n)
    valid = torch.tensor(valid, device=fk.device)
    rows = torch.arange(len(surfaces), device=fk.device)

    stream = torch.cuda.Stream(device=fk.device)
    stream.wait_stream(torch.cuda.current_stream(fk.device))
    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
    torch.cuda.current_stream(fk.device).wait_stream(stream)
    # One fitting iteration is captured once; rounds change tensor contents,
    # not pointers. This graph is separate from the task rollout graph.
    graph = torch.cuda.CUDAGraph()
    with capture_guard(), torch.cuda.graph(graph, stream=stream):
        step()
    torch.cuda.current_stream(fk.device).wait_stream(stream)
    with torch.no_grad():
        for cycle in range(surface_rounds):
            if clearance is not None:
                clearance.refresh(q)
            damping.fill_(0.01)
            for _ in range(iterations):
                graph.replay()
            p, rotation = fk.frames(q)
            pp = pool_p @ rotation.transpose(-1, -2) + p[:, None]
            nn = pool_n @ rotation.transpose(-1, -2)
            score = ((pp - tp[:, None]) / 0.01).square().sum(-1) + (
                nn - tn[:, None]
            ).square().sum(-1)
            selected = score.masked_fill(~valid, torch.inf).argmin(-1)
            local.copy_(pool_p[rows, selected])
            normals.copy_(pool_n[rows, selected])
        # Fit the last selected surfaces too, rather than returning an unfitted
        # discrete change from the final round.
        damping.fill_(0.01)
        if clearance is not None:
            clearance.refresh(q)
        for _ in range(iterations):
            graph.replay()
        fitted_p, fitted_n = fk.contacts(q, local, normals)
        normal_error = torch.rad2deg(torch.acos((fitted_n * tn).sum(-1).clamp(-1, 1)))
        position_error = torch.linalg.vector_norm(fitted_p - tp, dim=-1) * 1000
        values = q.cpu().numpy()
        hand.local, hand.normals = (
            local.cpu().numpy().copy(),
            normals.cpu().numpy().copy(),
        )
    root = Rotation.from_euler("XYZ", template_q[3:6])
    fitted = np.r_[
        template_q[:3] + root.apply(values[:3]),
        (root * Rotation.from_euler("XYZ", values[3:6])).as_euler("XYZ"),
        values[6:],
    ]
    return fitted, dict(
        method="CUDA_contact_layout_IK_and_actual_query_selection",
        joint_center_prior_weight=center_weight,
        bone_direction_weight=2.0,
        contacts_required=False,
        bone_direction_max_deviation_deg=bones.maximum_angle(q),
        **(clearance.diagnostics(q) if clearance is not None else {}),
        bone_finger_weights=bones.weights.cpu().tolist(),
        reference_qpos=reference_qpos.tolist(),
        reference_local_normals=region_normals.tolist(),
        contact_region_normal_dot=np.einsum(
            "ij,ij->i", hand.normals, region_normals
        ).tolist(),
        bone_direction_rms=float(bones.residual(q).square().mean().sqrt()),
        device=str(fk.device),
        seconds=time.perf_counter() - begun,
        target_source=target_source,
        iterations_per_round=iterations,
        surface_rounds=surface_rounds,
        object_pose_optimized=False,
        shared_solver_modified=False,
        surface_query_indices=selected.cpu().tolist(),
        contact_position_error_mm=position_error.cpu().tolist(),
        contact_normal_error_deg=normal_error.cpu().tolist(),
        before=before,
        after=template_contact_layout(hand, fitted),
        **opening_info,
    )
