"""Grasp candidate decoding, FK and full-objective batch evaluation."""

from types import SimpleNamespace
import math
import numpy as np
import torch
import warp as wp
from scipy.spatial.transform import Rotation
from retarget.task.dex import cost as fa
from retarget.task.grasp.cost import (
    SearchState,
    CostData,
    force_priorities,
    clear_cost,
    contact_cost,
    style_cost,
    smooth_cost,
    wrist_reference_cost,
    object_cost,
    capsule_cost,
    palm_clearance_masked,
    finish_cost,
    prepare_ranking,
)


@wp.func
def rotation_matrix(v: wp.vec3):
    angle = wp.length(v)
    axis = wp.vec3(1.0, 0.0, 0.0)
    if angle > 1.0e-10:
        axis = v / angle
    return wp.quat_to_matrix(wp.quat_from_axis_angle(axis, angle))


@wp.func
def xyz_angles(r: wp.mat33):
    y = wp.asin(wp.clamp(r[0, 2], -1.0, 1.0))
    x = wp.atan2(-r[1, 2], r[2, 2])
    z = wp.atan2(-r[0, 1], r[0, 0])
    if wp.abs(wp.cos(y)) < 1.0e-6:
        x = wp.atan2(r[2, 1], r[1, 1])
        z = 0.0
    return wp.vec3(x, y, z)


@wp.kernel(enable_backward=False)
def scatter(s: SearchState):
    world = wp.tid()
    for j in range(s.seed_qpos.shape[0]):
        s.qpos[world, j] = s.seed_qpos[j]
    angles = xyz_angles(
        s.root_rotation
        @ rotation_matrix(wp.vec3(s.x[world, 3], s.x[world, 4], s.x[world, 5]))
    )
    for j in range(26):
        value = s.x[world, j]
        if j >= 3 and j < 6:
            value = angles[j - 3]
        s.qpos[world, s.hand_ids[j]] = value


def initialize_runtime(runtime):
    """Compile geometry/FK without allocating the full executor population."""
    from retarget.env.mjwarp import KinematicModel, allocate_output, kinematics
    from retarget.env.surface import HandSamples

    fk = KinematicModel(runtime.model, runtime.device_model)
    device = fk.device
    geometry = fa._finger_joint_geometry(
        runtime.model, HandSamples.from_model(runtime.model), runtime.cfg.sequence.hand
    )
    loss_model = fa.LossModel()
    names = (
        "finger_joint_body",
        "finger_joint_start",
        "finger_joint_end",
        "finger_joint_radius",
    )
    for name, value, dtype in zip(names, geometry[:4], (int, wp.vec3, wp.vec3, float)):
        setattr(loss_model, name, wp.array(value, dtype=dtype, device=device))
    loss_model.finger_joint_pair_left = wp.array(
        geometry[4][:, 0], dtype=int, device=device
    )
    loss_model.finger_joint_pair_right = wp.array(
        geometry[4][:, 1], dtype=int, device=device
    )
    loss_model.finger_joint_pair_count = len(geometry[4])
    loss_model.object_body = runtime.index.object_body
    pose = allocate_output(fk, 1, requires_grad=False)
    qpos = wp.array(runtime.qpos[None], dtype=float, device=device)
    kinematics(fk, qpos, pose)
    job = fa.LossJob()
    job.model = loss_model
    state = fa.LossState()
    state.body_position, state.body_matrix, state.qpos = (
        pose.body_position,
        pose.body_matrix,
        qpos,
    )
    state.qvel = wp.zeros((1, runtime.model.nv), dtype=float, device=device)
    job.state = state
    workspace = SimpleNamespace(job=job)
    loss = SimpleNamespace(immediate=[workspace])
    backend = SimpleNamespace(device=device, kinematics=fk, steps=(), loss=loss)
    runtime.backend = backend
    runtime.static_pose = pose
    runtime.output_workspace = workspace
    runtime.output_step = 0
    runtime.prediction_horizon_seconds = 0.0
    runtime.grasp_cost_profile = {}


def _mesh_arrays(model, body):
    from retarget.env.grasp.hand import rigid_mesh_arrays

    return rigid_mesh_arrays(model, body)


def pad_chart(model, body, points, normals):
    """Wuji palmar face and longitudinal centreline in the link's local frame.

    Flexion axis x bone direction points out of the palmar face. The model
    profile resolves Hand 2's MCP abduction axis and different palm frame.
    Keep the front hemisphere including its rounded rim;
    full-body samples are still used for object penetration.
    """
    from retarget.env.grasp.hand import palmar_axis

    name = model.body(int(body)).name
    endpoint = None
    belly = palmar_axis(model, body, endpoint) if name == "right_palm_link" else None
    if name.startswith("right_finger"):
        finger, link = int(name[len("right_finger")]), int(name[-1])
        endpoint = (
            model.body(f"right_finger{finger}_link{link + 1}").pos.copy()
            if link < 4
            else model.site(
                (
                    "right_thumb_tip",
                    "right_index_tip",
                    "right_middle_tip",
                    "right_ring_tip",
                    "right_pinky_tip",
                )[finger - 1]
            ).pos.copy()
        )
        belly = palmar_axis(model, body, endpoint)
    mask = normals @ belly > 0.25
    if not mask.any():
        raise ValueError(f"No palmar surface samples: {name}")
    return mask, belly, endpoint


def enclosing_capsule_radii(model, bodies, starts, ends, radii):
    """Enclose every mesh vertex, rather than only the collision-cloud quantile.

    A capsule is convex: enclosing all vertices also encloses every triangle.
    Disjoint enclosing capsules therefore prevent inter-finger mesh crossings.
    The original capsules remain unchanged for the independent legacy audit.
    """
    result = np.asarray(radii, dtype=np.float32).copy()
    for i, body in enumerate(bodies):
        vertices, _ = _mesh_arrays(model, int(body))
        axis = ends[i] - starts[i]
        t = np.clip(
            (vertices - starts[i]) @ axis / max(float(axis @ axis), 1.0e-20), 0.0, 1.0
        )
        distance = np.linalg.norm(vertices - starts[i] - t[:, None] * axis, axis=1)
        result[i] = max(result[i], float(distance.max()) + 1.0e-6)
    return result


class GraspProblem:
    def __init__(self, runtime):
        self.runtime = runtime
        self.backend = runtime.backend
        self.device = self.backend.device
        self._initialize_search()
        from retarget.dynamics.explicit import ExplicitPoseModel

        self.explicit = ExplicitPoseModel(self)
        self.cost.pose_cost, self.cost.pose_enabled = self.explicit.cost, 1
        self.mean = wp.to_torch(self.state.seed).clone()
        self.sigma = wp.to_torch(self.state.initial_sigma).clone()
        self.low = wp.to_torch(self.state.low)
        self.high = wp.to_torch(self.state.high)

    def project(self, x):
        return torch.maximum(self.low, torch.minimum(self.high, x))

    def evaluate(self, values):
        from retarget.solver.base import Evaluation

        costs = []
        ranking = []
        with wp.ScopedStream(
            wp.stream_from_torch(torch.cuda.current_stream(values.device))
        ):
            for chunk in values.split(self.state.population):
                x = wp.to_torch(self.state.x)
                x[: len(chunk)] = chunk
                if len(chunk) < len(x):
                    x[len(chunk) :] = x[0]
                wp.launch(
                    scatter,
                    self.state.population,
                    inputs=[self.state],
                    device=self.device,
                )
                self._score()
                wp.launch(
                    prepare_ranking,
                    self.state.population,
                    inputs=[self.cost, self.state],
                    device=self.device,
                )
                costs.append(wp.to_torch(self.state.scores)[: len(chunk)].clone())
                ranking.append(
                    wp.to_torch(self.state.selection_scores)[: len(chunk)].clone()
                )
        cost = torch.cat(costs)
        return Evaluation(
            cost, cost if bool(torch.isfinite(cost).any()) else torch.cat(ranking)
        )

    def _initialize_search(self):
        """Allocate the numerical search independently of the CUDA scheduler."""
        from retarget.env.mjwarp import allocate_output
        from retarget.env.grasp.hand import query_path

        self.explicit = None
        runtime = self.runtime
        args = runtime.args
        n = args.worlds
        s = self.state = SearchState()
        s.population = n
        array = lambda value, dtype=float: wp.array(
            np.asarray(value), dtype=dtype, device=self.device
        )
        self.array = array
        q = runtime.qpos[np.asarray(runtime.index.hand_qpos)].copy()
        seed = q.copy()
        seed[3:6] = 0.0
        preload = bool(getattr(args, "explicit_pose", False))
        if preload:
            seed = np.r_[seed, np.zeros(20)]
        s.root_rotation = wp.mat33(Rotation.from_euler("XYZ", q[3:6]).as_matrix())
        s.seed = array(seed)
        s.seed_qpos = array(runtime.qpos)
        s.hand_ids = array(runtime.index.hand_qpos, int)
        low, high = np.full(26, -np.inf), np.full(26, np.inf)
        for j, address in enumerate(runtime.index.hand_qpos):
            joint = np.flatnonzero(runtime.model.jnt_qposadr == address)[0]
            if j < 3 or j >= 6:
                if runtime.model.jnt_limited[joint]:
                    low[j], high[j] = runtime.model.jnt_range[joint]
            if j >= 6:
                low[j], high[j] = max(low[j], q[j] - 0.25), min(high[j], q[j] + 0.25)
            elif j < 3:
                low[j], high[j] = max(low[j], q[j] - 0.02), min(high[j], q[j] + 0.02)
        if np.any(low > high):
            raise ValueError(
                "Initialization is too far outside joint limits for the CARD per-frame cap"
            )
        if preload:
            limit = getattr(args, "explicit_preload_limit", 0.5)
            low, high = np.r_[low, np.full(20, -limit)], np.r_[high, np.full(20, limit)]
        s.low, s.high = array(low), array(high)
        sigma = np.r_[
            np.full(3, getattr(args, "translation_noise_mm", 2.0) / 1000),
            np.full(3, getattr(args, "rotation_noise", 0.016)),
            np.full(20, getattr(args, "joint_noise", 0.05)),
        ]
        if preload:
            sigma = np.r_[sigma, np.full(20, limit / 3.0)]
        if not np.isfinite(sigma).all() or np.any(sigma <= 0):
            raise ValueError("CARD initial sampling noise must be finite and positive")
        s.initial_sigma = array(sigma)
        s.x = wp.zeros((n, len(seed)), dtype=float, device=self.device)
        s.qpos = wp.empty((n, runtime.model.nq), dtype=float, device=self.device)
        s.scores = wp.empty(n, dtype=float, device=self.device)
        s.selection_scores = wp.empty(n, dtype=float, device=self.device)
        self.pose = allocate_output(
            self.backend.kinematics, n, requires_grad=False
        )
        wp.synchronize_device(self.device)
        self.cost = d = CostData()
        d.position, d.rotation = self.pose.body_position, self.pose.body_matrix
        d.seed_position = array(runtime.static_pose.body_position.numpy()[0], wp.vec3)
        d.seed_rotation = array(runtime.static_pose.body_matrix.numpy()[0], wp.mat33)
        auditor, hand, _ = runtime.audit_context
        d.object_mesh, d.object_body = auditor.mesh.id, runtime.index.object_body
        d.palm_body = runtime.model.body("right_palm_link").id
        from retarget.task.grasp.wrist_prior import warp_prior

        reference_path = runtime.experiment / "wrist_pose_reference.json"
        self.wrist_reference = None
        if reference_path.exists():
            import json

            self.wrist_reference = json.loads(reference_path.read_text())
        d.wrist_prior = warp_prior(self.wrist_reference)
        d.wrist_prior_cost = wp.zeros(n, dtype=float, device=self.device)
        pv, pf = _mesh_arrays(runtime.model, d.palm_body)
        self.palm_vertices, self.palm_faces = (
            array(pv, wp.vec3),
            array(pf.reshape(-1), int),
        )
        self.palm_mesh = wp.Mesh(points=self.palm_vertices, indices=self.palm_faces)
        d.palm_mesh = self.palm_mesh.id
        surface, normals, bodies, pads, starts, ends = [], [], [], [], {}, {}
        from retarget.env.grasp.regions import forbidden_mask

        with np.load(runtime.experiment / "card_input.npz") as supplied:
            region_normals = (
                supplied["region_reference_normals"].copy()
                if "region_reference_normals" in supplied.files
                else None
            )
            d.object_axis_enabled = int("object_axis_palm" in supplied.files)
            if d.object_axis_enabled:
                d.object_axis_palm = wp.vec3(supplied["object_axis_palm"])
                d.object_axis_cosine = float(
                    np.cos(np.deg2rad(float(supplied["object_axis_limit_deg"])))
                )
            self.tip_patches = (
                {
                    int(b): dict(
                        center=center.copy(),
                        radius_m=float(radius),
                        weight=float(weight),
                    )
                    for b, center, radius, weight in zip(
                        supplied["tip_patch_bodies"],
                        supplied["tip_patch_centers"],
                        supplied["tip_patch_radii"],
                        supplied["tip_patch_weights"],
                    )
                }
                if "tip_patch_bodies" in supplied.files
                else {}
            )
        d.region_enabled = int(region_normals is not None)
        forbidden = []
        self.pad_charts = {}
        self.surfaces = {}
        for body in np.unique(hand.all_bodies):
            name = runtime.model.body(int(body)).name
            with np.load(query_path(runtime.model, name)) as asset:
                p, normal = asset["local_pos"], asset["local_normal"]
            normal = normal / np.maximum(
                np.linalg.norm(normal, axis=-1, keepdims=True), 1.0e-12
            )
            starts[body], ends[body] = len(surface), len(surface) + len(p)
            mask, belly, endpoint = pad_chart(runtime.model, body, p, normal)
            patch = self.tip_patches.get(int(body))
            if patch is not None:
                mask &= (normal @ belly > 0.75) & (
                    np.linalg.norm(p - patch["center"], axis=1)
                    <= patch["radius_m"] + 1.0e-7
                )
                if not mask.any():
                    raise ValueError(f"No query samples in the central tip pad: {name}")
            self.pad_charts[int(body)] = mask, belly, endpoint
            surface.extend(p)
            normals.extend(normal)
            bodies.extend([body] * len(p))
            pads.extend(mask)
            forbidden.extend(
                forbidden_mask(runtime.model, body, normal, hand.bodies, region_normals)
                if region_normals is not None
                else np.zeros(len(p), bool)
            )
            self.surfaces[int(body)] = (p, normal)
        d.sample_body, d.sample_local, d.sample_normal = (
            array(bodies, int),
            array(surface, wp.vec3),
            array(normals, wp.vec3),
        )
        d.sample_pad = array(pads, int)
        d.sample_forbidden = array(forbidden, int)
        d.forbidden_gap = wp.full(s.population, wp.inf, dtype=float, device=self.device)
        groups, inverse = np.unique(hand.bodies, return_inverse=True)
        d.group_body, d.group_start, d.group_end = (
            array(groups, int),
            array([starts[b] for b in groups], int),
            array([ends[b] for b in groups], int),
        )
        self.annotation_group = array(inverse, int)
        self.annotation_bodies = hand.bodies.copy()
        self.annotation_count = len(hand.bodies)
        # Static reference input is persisted independently of any closing/preload reference.
        with np.load(runtime.experiment / "card_input.npz") as supplied:
            anchors, outward = (
                supplied["object_points"].copy(),
                supplied["object_normals"].copy(),
            )
            forces = (
                supplied["contact_forces"].copy()
                if "contact_forces" in supplied
                else None
            )
        force_priorities(
            forces, self.annotation_count
        )  # validate persisted optional input
        if (
            anchors.shape != (self.annotation_count, 3)
            or outward.shape != anchors.shape
            or not np.isfinite(anchors).all()
            or not np.isfinite(outward).all()
        ):
            raise ValueError(
                "CARD requires finite Nx3 object contact positions and normals"
            )
        group_forces = (
            None
            if forces is None
            else np.bincount(inverse, weights=forces, minlength=len(groups))
        )
        preferences, target, push, alpha, previous, center_weights = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        seed_pos, seed_rot = d.seed_position.numpy(), d.seed_rotation.numpy()
        for b in groups:
            ids = np.flatnonzero(hand.bodies == b)
            weights = np.ones(len(ids)) if forces is None else forces[ids]
            if weights.sum() <= 0:
                weights = np.ones(len(ids))
            preference = np.average(hand.local[ids], axis=0, weights=weights)
            anchor = np.average(anchors[ids], axis=0, weights=weights)
            direction = -np.average(outward[ids], axis=0, weights=weights)
            if np.linalg.norm(direction) < 1.0e-8:
                raise ValueError("Semantic contact has cancelling/zero push normals")
            direction /= np.linalg.norm(direction)
            name = runtime.model.body(int(b)).name
            a = 0.15 if name.endswith(("link3", "link4")) else 0.0
            p, _ = self.surfaces[int(b)]
            mask, belly, endpoint = self.pad_charts[int(b)]
            patch = self.tip_patches.get(int(b))
            if patch is not None:
                preference, a = patch["center"], patch["weight"]
            elif endpoint is not None:
                u = np.clip(
                    np.dot(preference, endpoint) / np.dot(endpoint, endpoint), 0.0, 1.0
                )
                # Discrete palmar centreline at the remapped fractional bone coordinate.
                score = np.sum((p - u * endpoint) ** 2, axis=-1)
                preference = p[np.argmin(np.where(mask, score, np.inf))]
            world = p @ seed_rot[b].T + seed_pos[b]
            # Object is fixed at the input pose (normally identity).
            obj = d.object_body
            anchor_world = seed_rot[obj] @ anchor + seed_pos[obj]
            scores = np.sum((world - anchor_world) ** 2, axis=1) + a * np.sum(
                (p - preference) ** 2, axis=1
            )
            previous.append(p[np.argmin(np.where(mask, scores, np.inf))])
            preferences.append(preference)
            target.append(anchor)
            push.append(direction)
            alpha.append(a)
            center_weights.append(0.0 if patch is None else patch["weight"])
        d.preference, d.previous_local = (
            array(preferences, wp.vec3),
            array(previous, wp.vec3),
        )
        d.anchor, d.push, d.alpha = (
            array(target, wp.vec3),
            array(push, wp.vec3),
            array(alpha),
        )
        d.center_weight = array(center_weights)
        d.rho = array(force_priorities(group_forces, len(groups)))
        self.force_source = (
            "uniform_no_force_measurement"
            if forces is None
            else "provided_contact_force_magnitudes"
        )
        d.selected = wp.zeros((n, len(groups)), dtype=int, device=self.device)
        d.parts = wp.zeros((n, 4), dtype=float, device=self.device)
        d.penetration, d.palm_gap = (
            wp.zeros(n, device=self.device),
            wp.empty(n, device=self.device),
        )
        d.self_gap = wp.empty(n, dtype=float, device=self.device)
        d.self_clearance = getattr(args, "self_clearance_mm", 0.5)
        if not math.isfinite(d.self_clearance) or d.self_clearance < 0:
            raise ValueError("CEM self clearance must be finite and nonnegative")
        style_bodies, apex, pad_normal, axis = [], [], [], []
        for finger in range(1, 6):
            for link in (2, 3, 4):
                body = runtime.model.body(f"right_finger{finger}_link{link}").id
                p, normal = self.surfaces[body]
                mask, belly, endpoint = self.pad_charts[body]
                axis.append(endpoint / max(np.linalg.norm(endpoint), 1.0e-12))
                # The apex stays on the palmar pad; its belly axis is independent of cap normals.
                landmark = (
                    self.tip_patches[body]["center"]
                    if body in self.tip_patches
                    else endpoint
                )
                score = np.sum((p - landmark) ** 2, axis=-1)
                nearest = np.argmin(np.where(mask, score, np.inf))
                style_bodies.append(body)
                apex.append(p[nearest])
                pad_normal.append(belly)
        d.style_body, d.style_apex = array(style_bodies, int), array(apex, wp.vec3)
        d.style_normal, d.style_axis = array(pad_normal, wp.vec3), array(axis, wp.vec3)
        m = runtime.output_workspace.job.model
        d.capsule_body, d.capsule_start, d.capsule_end, d.capsule_radius = (
            m.finger_joint_body,
            m.finger_joint_start,
            m.finger_joint_end,
            m.finger_joint_radius,
        )
        old_radii = m.finger_joint_radius.numpy()
        enclosing_radii = enclosing_capsule_radii(
            runtime.model,
            m.finger_joint_body.numpy(),
            m.finger_joint_start.numpy(),
            m.finger_joint_end.numpy(),
            old_radii,
        )
        d.capsule_radius = array(enclosing_radii)
        self.maximum_capsule_radius_increase_mm = (
            float(np.max(enclosing_radii - old_radii)) * 1000.0
        )
        d.pair_left, d.pair_right = m.finger_joint_pair_left, m.finger_joint_pair_right
        from retarget.env.grasp.hand import model_hand, body_boxes

        if model_hand(runtime.model) == "wuji2":
            center, half = body_boxes(runtime.model, d.capsule_body.numpy())
            d.box_center, d.box_half = array(center, wp.vec3), array(half, wp.vec3)
            d.box_certificate = 1
        else:
            # Warp's struct adjoint traverses optional array members too.
            d.box_center, d.box_half = (
                array(np.empty((0, 3)), wp.vec3),
                array(np.empty((0, 3)), wp.vec3),
            )
        # Wuji has an extra abduction base before the semantic proximal link.
        # Both belong to the palm/proximal articulation, so its connected cap
        # must not be tested against the palm as a nonadjacent collision pair.
        # They remain in the inter-finger soft capsule penalty.
        cb = d.capsule_body.numpy()
        parent = runtime.model.body_parentid[cb]
        grandparent = runtime.model.body_parentid[parent]
        self.palm_capsules = array(
            np.flatnonzero((parent != d.palm_body) & (grandparent != d.palm_body)), int
        )
        runtime.card_contact_local = wp.zeros(
            self.annotation_count, dtype=wp.vec3, device=self.device
        )
        runtime.card_contact_normal = wp.zeros_like(runtime.card_contact_local)

    def _score(self, worlds=None):
        from retarget.env.mjwarp import kinematics

        d, s = self.cost, self.state
        worlds = s.population if worlds is None else worlds
        view = s.qpos if worlds == s.population else s.qpos[:worlds]
        kinematics(self.backend.kinematics, view, self.pose)
        wp.launch(clear_cost, worlds, inputs=[d], device=self.device)
        wp.launch(
            contact_cost, (worlds, len(d.group_body)), inputs=[d], device=self.device
        )
        wp.launch(style_cost, (worlds, 15), inputs=[d], device=self.device)
        wp.launch(smooth_cost, worlds, inputs=[d, s], device=self.device)
        wp.launch(wrist_reference_cost, worlds, inputs=[d], device=self.device)
        wp.launch(
            object_cost, (worlds, len(d.sample_body)), inputs=[d], device=self.device
        )
        wp.launch(
            capsule_cost, (worlds, len(d.pair_left)), inputs=[d], device=self.device
        )
        wp.launch(
            palm_clearance_masked,
            (worlds, len(self.palm_capsules)),
            inputs=[d, self.palm_capsules],
            device=self.device,
        )
        predict = self.explicit is not None and True
        d.pose_enabled = int(predict)
        if predict:
            self.explicit.forward(worlds)
        wp.launch(finish_cost, worlds, inputs=[d, s], device=self.device)
