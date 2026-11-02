"""Dex tracking terms evaluated on a static grasp's predicted trajectory.

The loss formulas are the existing dex functions. Geometry supplies signed
mesh gaps rather than simulation constraint rows; it does not create forces.
Template query identities and object-local anchors stay fixed during search.
"""

from collections.abc import Mapping
import math
from pathlib import Path
import numpy as np
import warp as wp

from retarget.task.dex import cost as dex
from retarget.task.grasp.cost import CostData, SearchState, object_axis_cosine


PART_NAMES = ("penetration", "approach", "maintain", "object_tracking")
HAND_WEIGHT_NAMES = ("action_ref_base", "action_ref_base_rot", "action_ref_finger")
METRIC_NAMES = (
    "maximum_penetration_mm",
    "mean_template_distance_mm",
    "max_template_distance_mm",
    "minimum_body_gap_mm",
    "maximum_guided_gap_mm",
)
# Source: third_party/DexForge/forceaware/configs/forceaware.yaml. Velocity
# weights are the defaults of retarget.config.dex.LossWeights.
DEFAULT_WEIGHTS = dict(
    obj_pos=1.0, obj_rot=0.3, obj_vel=0.1, obj_ang_vel=0.03,
    reference_pose_scale=100.0, terminal=2.0, contact_robustness=40.0,
    marked_query_pos=1.0, soft_penetration=15.0, penetration=75.0,
    action_ref_base=0.0, action_ref_base_rot=0.0, action_ref_finger=0.0,
)
DEFAULT_CONTACT = dict(
    target_phi=-0.004, penetration_limit=0.008,
    marked_query_dist_mm=10.0, marked_query_temp_mm=1.0,
)


def configuration(args, cfg=None):
    """Use prepared dex values when present, with explicit grasp overrides."""
    weights, contact = DEFAULT_WEIGHTS.copy(), DEFAULT_CONTACT.copy()
    for target, source in ((weights, getattr(cfg, "loss", None)),
                           (contact, getattr(cfg, "contact", None))):
        if source is not None:
            for key in target:
                # Adding the hand reference is opt-in for this profile. An
                # older prepared FA configuration may have nonzero action
                # weights without containing a contact-aware grasp reference.
                if key in HAND_WEIGHT_NAMES:
                    continue
                if hasattr(source, key):
                    target[key] = getattr(source, key)
    for name, target, extra in (("tracking_weights", weights, set()),
                                ("tracking_contact", contact, {"soft_penetration_limit"})):
        supplied = getattr(args, name, {})
        if not isinstance(supplied, Mapping):
            raise ValueError(f"{name} must be a mapping")
        unknown = set(supplied) - set(target) - extra
        if unknown:
            raise ValueError(f"Unknown {name}: {sorted(unknown)}")
        for key, value in supplied.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name}.{key} must be finite")
            target[key] = float(value)
    contact.setdefault("soft_penetration_limit", -contact["target_phi"])
    if any(value < 0 for value in weights.values()):
        raise ValueError("Tracking weights must be nonnegative")
    if (contact["target_phi"] > 0 or contact["soft_penetration_limit"] < 0
            or contact["penetration_limit"] < contact["soft_penetration_limit"]
            or contact["marked_query_dist_mm"] < 0
            or contact["marked_query_temp_mm"] <= 0):
        raise ValueError("Invalid tracking contact distances/temperature")
    return weights, contact


def load_hand_reference(experiment, hand_dimension, enabled):
    """Read an explicit solved reference; a template/seed is never a fallback."""
    if not enabled:
        return None
    path = Path(experiment) / "card_input.npz"
    if not path.is_file():
        raise ValueError("hand_tracking requires card_input.npz with hand_reference_qpos")
    with np.load(path, allow_pickle=False) as supplied:
        if "hand_reference_qpos" not in supplied:
            raise ValueError("hand_tracking requires explicit contact-aware hand_reference_qpos")
        if ("hand_reference_source" not in supplied
                or np.asarray(supplied["hand_reference_source"]).shape != ()
                or str(supplied["hand_reference_source"].item()) != "dexforge_contactaware"):
            raise ValueError("hand_reference_qpos must originate from dexforge_contactaware")
        reference = np.asarray(supplied["hand_reference_qpos"], dtype=np.float32)
    if reference.shape != (hand_dimension,) or not np.isfinite(reference).all():
        raise ValueError(f"hand_reference_qpos must contain {hand_dimension} finite hand coordinates")
    return reference.copy()


def load_reference_contact_mask(experiment, contact_count):
    """A released ContactAware slot must not remain a required tracking target."""
    with np.load(Path(experiment) / "card_input.npz", allow_pickle=False) as supplied:
        mask = (np.asarray(supplied["reference_contact_mask"], dtype=np.float32)
                if "reference_contact_mask" in supplied else np.ones(contact_count, np.float32))
    if (mask.shape != (contact_count,) or not np.isfinite(mask).all()
            or not np.isin(mask, [0.0, 1.0]).all() or not mask.any()):
        raise ValueError("reference_contact_mask must contain one binary value per contact and an active slot")
    return mask.copy()


@wp.kernel(enable_backward=False)
def measure_gaps(
    mesh: wp.uint64, object_body: int, bodies: wp.array(dtype=int),
    starts: wp.array(dtype=int), ends: wp.array(dtype=int),
    samples: wp.array(dtype=wp.vec3), positions: wp.array2d(dtype=wp.vec3),
    rotations: wp.array2d(dtype=wp.mat33),
    distance: wp.array2d(dtype=float), active: wp.array2d(dtype=int),
):
    """Per-link minimum signed surface gap, in metres, in the object frame.

    Keep separated distances instead of clipping them to the physics contact
    activation distance, so the maintenance term can penalize lost contact.
    """
    w, row = wp.tid()
    body = bodies[row]
    best = float(wp.inf)
    for i in range(starts[row], ends[row]):
        world = positions[w, body] + rotations[w, body] @ samples[i]
        local = wp.transpose(rotations[w, object_body]) @ (world - positions[w, object_body])
        query = wp.mesh_query_point(mesh, local, 1.0e6)
        if query.result:
            closest = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
            best = wp.min(best, query.sign * wp.length(local - closest))
    distance[w, row] = best
    active[w, row] = 1


@wp.kernel(enable_backward=False)
def measure_guidance_gaps(
    mesh: wp.uint64, object_body: int, bodies: wp.array(dtype=int),
    starts: wp.array(dtype=int), ends: wp.array(dtype=int),
    samples: wp.array(dtype=wp.vec3), positions: wp.array2d(dtype=wp.vec3),
    rotations: wp.array2d(dtype=wp.mat33), row_offset: int,
    distance: wp.array2d(dtype=float), active: wp.array2d(dtype=int),
):
    """Measure each fixed semantic patch, independently of other link skin."""
    w, contact = wp.tid()
    body = bodies[contact]
    best = float(wp.inf)
    for i in range(starts[contact], ends[contact]):
        world = positions[w, body] + rotations[w, body] @ samples[i]
        local = wp.transpose(rotations[w, object_body]) @ (world - positions[w, object_body])
        query = wp.mesh_query_point(mesh, local, 1.0e6)
        if query.result:
            closest = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
            best = wp.min(best, query.sign * wp.length(local - closest))
    distance[w, row_offset + contact] = best
    active[w, row_offset + contact] = 1


@wp.kernel(enable_backward=False)
def map_velocity(
    velocity: wp.array2d(dtype=float), hand_dof: wp.array(dtype=int),
    object_dof: int, out: wp.array2d(dtype=float),
):
    w = wp.tid()
    for j in range(out.shape[1]):
        out[w, j] = 0.0
    for j in range(6):
        out[w, object_dof + j] = velocity[w, j]
    for j in range(hand_dof.shape[0]):
        out[w, hand_dof[j]] = velocity[w, j + 6]


@wp.kernel(enable_backward=False)
def score_state(
    job: dex.LossJob, step: int,
    parts: wp.array3d(dtype=float), metrics: wp.array3d(dtype=float),
):
    w = wp.tid()
    approach, maintain = float(0.0), float(0.0)
    mean_distance, max_distance = float(0.0), float(0.0)
    maximum_guided_gap = float(-wp.inf)
    active_count = int(0)
    for i in range(job.model.guidance_count):
        approach += dex._marked_query_loss(job, w, i)
        maintain += dex._physical_contact_loss(job, w, i)
        if job.target.contact_mask[i] > 0.0:
            distance = wp.length(dex._query_world(job, w, i) - dex._query_target_world(job, w, i)) * 1000.0
            mean_distance += distance
            active_count += 1
            max_distance = wp.max(max_distance, distance)
            gap, point = dex._contact(job, w, i)
            maximum_guided_gap = wp.max(maximum_guided_gap, gap * 1000.0)
    if active_count > 0:
        mean_distance /= float(active_count)
    else:
        maximum_guided_gap = 0.0
    minimum_gap = float(wp.inf)
    for row in range(job.model.output_count):
        gap, point = dex._collision_output(job, w, row)
        minimum_gap = wp.min(minimum_gap, gap * 1000.0)
    parts[w, step, 0] = dex._penetration_loss(job, w)
    parts[w, step, 1] = approach
    parts[w, step, 2] = maintain
    parts[w, step, 3] = 1.0e6 * job.weights.reference_pose_scale * (
        dex._object_reference_squared(job, w) + dex._object_velocity_squared(job, w)
    )
    metrics[w, step, 0] = wp.max(-minimum_gap, 0.0)
    metrics[w, step, 1] = mean_distance
    metrics[w, step, 2] = max_distance
    metrics[w, step, 3] = minimum_gap
    metrics[w, step, 4] = maximum_guided_gap


@wp.kernel(enable_backward=False)
def score_hand_tracking(job: dex.LossJob, step: int, parts: wp.array3d(dtype=float)):
    w = wp.tid()
    parts[w, step, 4] = 1.0e6 * dex._hand_reference_squared(job, w)


@wp.kernel(enable_backward=False)
def aggregate(
    steps: wp.array3d(dtype=float), terminal: float,
    parts: wp.array2d(dtype=float), cost: wp.array(dtype=float),
):
    w = wp.tid()
    total = float(0.0)
    final = steps.shape[1] - 1
    denominator = float(final) + terminal
    for column in range(steps.shape[2]):
        value = float(0.0)
        for step in range(steps.shape[1]):
            weight = float(1.0)
            if step == final:
                weight = terminal
            value += weight * steps[w, step, column] / denominator
        parts[w, column] = value
        total += value
    cost[w] = total


@wp.kernel(enable_backward=False)
def synchronize_ranking_parts(
    parts: wp.array2d(dtype=float), cost: wp.array(dtype=float),
    legacy_parts: wp.array2d(dtype=float),
):
    """Preserve the four-column feasibility dispatch and reject any NaN term."""
    w = wp.tid()
    for column in range(legacy_parts.shape[1]):
        legacy_parts[w, column] = parts[w, column]
    if not wp.isfinite(cost[w]):
        legacy_parts[w, 0] = float(wp.inf)


@wp.kernel(enable_backward=False)
def finish_tracking(d: CostData, s: SearchState, total: wp.array(dtype=float)):
    """Original q0 geometric constraints, without the CARD soft objective."""
    w = wp.tid()
    value = total[w]
    if (d.penetration[w] > 3.0 or d.palm_gap[w] < 6.0
            or d.self_gap[w] < d.self_clearance or not wp.isfinite(value)):
        value = float(wp.inf)
    if d.region_enabled != 0 and d.forbidden_gap[w] < 1.0:
        value = float(wp.inf)
    if d.object_axis_enabled != 0 and object_axis_cosine(d, w) < d.object_axis_cosine:
        value = float(wp.inf)
    s.scores[w] = value


class TrackingObjective:
    def __init__(self, node):
        self.node = node
        runtime, d, s, explicit = node.runtime, node.cost, node.state, node.explicit
        weights, contact = configuration(runtime.args, runtime.cfg)
        self.weights, self.contact = weights, contact
        self.hand_tracking_enabled = any(weights[name] > 0 for name in HAND_WEIGHT_NAMES)
        self.part_names = PART_NAMES + (("hand_tracking",) if self.hand_tracking_enabled else ())
        hand_reference = load_hand_reference(
            runtime.experiment, len(runtime.index.hand_qpos), self.hand_tracking_enabled
        )
        self.parts = wp.zeros((s.population, len(self.part_names)), device=node.device)
        self.step_parts = wp.zeros((s.population, explicit.horizon + 1, len(self.part_names)), device=node.device)
        self.metrics = wp.zeros((s.population, explicit.horizon + 1, 5), device=node.device)
        self.cost = wp.zeros(s.population, device=node.device)
        array = node.array
        model = dex.LossModel()
        model.object_body = runtime.index.object_body
        model.object_qpos = runtime.index.object_qpos
        model.action_dim = len(runtime.index.hand_qpos)
        model.hand_qpos = array(runtime.index.hand_qpos, int)
        joint = np.flatnonzero(runtime.model.jnt_qposadr == model.object_qpos)[0]
        model.object_dof = int(runtime.model.jnt_dofadr[joint])
        self.hand_dof = array([
            runtime.model.jnt_dofadr[np.flatnonzero(runtime.model.jnt_qposadr == address)[0]]
            for address in runtime.index.hand_qpos
        ], int)
        bodies = np.unique(d.sample_body.numpy())
        source = d.sample_body.numpy()
        self.bodies = array(bodies, int)
        self.starts = array([np.flatnonzero(source == b)[0] for b in bodies], int)
        self.ends = array([np.flatnonzero(source == b)[-1] + 1 for b in bodies], int)
        model.output_count = len(bodies)
        hand = runtime.audit_context[1]
        model.guidance_count = len(hand.bodies)
        if model.guidance_count < 1:
            raise ValueError("Tracking grasp requires template contact points")
        if (getattr(runtime.args, "contact_mapping", "nearest") == "semantic_patch"
                and not hasattr(hand, "semantic_contact_masks")):
            # A restored runtime must use the original masks rather than
            # silently reverting semantic contact maintenance to whole links.
            with np.load(runtime.experiment / "card_input.npz") as supplied:
                if "semantic_patch_masks" not in supplied:
                    raise ValueError("Semantic tracking requires persisted semantic contact masks")
                offsets = supplied["semantic_patch_offsets"]
                packed = supplied["semantic_patch_masks"]
                if len(offsets) != model.guidance_count + 1:
                    raise ValueError("Persisted semantic masks and contact count disagree")
                hand.semantic_contact_masks = [
                    packed[offsets[i]:offsets[i + 1]].astype(bool).copy()
                    for i in range(model.guidance_count)
                ]
                hand.semantic_patch_labels = supplied["semantic_patch_labels"].tolist()
        self.semantic = hasattr(hand, "semantic_contact_masks")
        rows = model.output_count
        if self.semantic:
            from retarget.env.grasp.hand import query_path

            points, starts, ends = [], [], []
            for i, body in enumerate(hand.bodies):
                with np.load(query_path(runtime.model, int(body))) as queries:
                    local_points = queries["local_pos"].copy()
                mask = np.asarray(hand.semantic_contact_masks[i], dtype=bool)
                if mask.shape != (len(local_points),) or not mask.any():
                    raise ValueError("Every semantic contact requires a nonempty matching query mask")
                starts.append(len(points))
                points.extend(local_points[mask])
                ends.append(len(points))
            self.patch_bodies = array(hand.bodies, int)
            self.patch_starts, self.patch_ends = array(starts, int), array(ends, int)
            self.patch_points = array(points, wp.vec3)
            rows += model.guidance_count
        # The first output_count rows measure the entire hand for penetration.
        # Optional additional rows feed only the guided patch maintenance term.
        model.output_contact_rows = array(np.arange(rows), int)
        model.contact_thresholds = array(np.full(rows, 0.01))
        target = dex.LossTarget()
        target.action = array(
            hand_reference if hand_reference is not None else np.zeros(model.action_dim)
        )
        target.query_body = array(hand.bodies, int)
        target.query_local = array(hand.local, wp.vec3)
        with np.load(runtime.experiment / "card_input.npz") as supplied:
            anchors = supplied["object_points"].copy()
        if anchors.shape != (model.guidance_count, 3) or not np.isfinite(anchors).all():
            raise ValueError("One finite object-local anchor is required per template point")
        target.query_position_object = array(anchors, wp.vec3)
        target.contact_position_object = target.query_position_object
        target.contact_output = array(
            model.output_count + np.arange(model.guidance_count)
            if self.semantic else [np.flatnonzero(bodies == b)[0] for b in hand.bodies], int
        )
        self.reference_contact_mask = load_reference_contact_mask(runtime.experiment, model.guidance_count)
        target.query_mask = target.contact_mask = array(self.reference_contact_mask)
        target.query_weight = target.contact_weight = array(np.ones(model.guidance_count))
        target.object_qpos = array(runtime.qpos[model.object_qpos:model.object_qpos + 7])
        target.object_velocity_world = array(np.zeros(6))
        loss_weights = dex.LossWeights()
        for name, key in (("object_position", "obj_pos"), ("object_rotation", "obj_rot"),
                          ("object_velocity", "obj_vel"), ("object_angular_velocity", "obj_ang_vel"),
                          ("marked_query", "marked_query_pos"),
                          ("action_base_position", "action_ref_base"),
                          ("action_base_rotation", "action_ref_base_rot"),
                          ("action_finger", "action_ref_finger")):
            setattr(loss_weights, name, weights[key])
        for name in ("reference_pose_scale", "contact_robustness", "soft_penetration", "penetration"):
            setattr(loss_weights, name, weights[name])
        for name in ("target_phi", "soft_penetration_limit", "penetration_limit"):
            setattr(loss_weights, name, contact[name])
        loss_weights.marked_distance_mm = contact["marked_query_dist_mm"]
        loss_weights.marked_temperature_mm = contact["marked_query_temp_mm"]
        state = dex.LossState()
        shape = (s.population, rows)
        state.contact_distance = wp.empty(shape, device=node.device)
        state.contact_active = wp.empty(shape, dtype=int, device=node.device)
        state.contact_position = wp.zeros(shape, dtype=wp.vec3, device=node.device)
        state.qvel = wp.zeros((s.population, runtime.model.nv), device=node.device)
        auxiliary = dex.LossAuxiliary()
        auxiliary.marked_query_weight = wp.empty((s.population, model.guidance_count), device=node.device)
        self.job = dex.LossJob()
        self.job.model, self.job.target, self.job.state = model, target, state
        self.job.weights, self.job.auxiliary = loss_weights, auxiliary
        # Reuse one terminal FK output; intermediate states already have one.
        self.terminal_pose = node.backend.kinematics.allocate(s.population)
        self.metadata = dict(
            objective="dex_tracking", part_names=list(self.part_names), metric_names=list(METRIC_NAMES),
            weights=weights, contact=contact,
            formula_source="retarget/task/dex/cost.py",
            fallback_defaults_source="third_party/DexForge/forceaware/configs/forceaware.yaml",
            velocity_defaults_source="retarget/config/dex.py:LossWeights",
            configuration_precedence="prepared cfg.loss/cfg.contact, then task.options tracking overrides; fallback constants for absent fields",
            hand_tracking_enabled=self.hand_tracking_enabled,
            hand_tracking_weight_source="explicit task.options.tracking_weights action_ref_* only; absent/zero keeps the four-term profile",
            hand_reference_source="dexforge_contactaware" if self.hand_tracking_enabled else None,
            hand_reference_storage="card_input.npz:hand_reference_qpos" if self.hand_tracking_enabled else None,
            hand_reference_qpos=hand_reference.tolist() if hand_reference is not None else None,
            hand_reference_coordinates="scene hand qpos order: wrist xyz, intrinsic XYZ angles, then finger joints",
            hand_tracking_formula="1e6 * dex._hand_reference_squared; scalar scene joint-coordinate differences",
            aggregation="sum(step_weight * step_parts) / (horizon + terminal_weight)",
            scored_states="q0 through qH, qH has terminal weight; q0 velocity is zero",
            contact_distance_source=(
                "minimum signed object-mesh gap on each fixed matched semantic patch; no activation clipping"
                if self.semantic else
                "minimum signed object-mesh gap of all samples on each guided link; no activation clipping"
            ),
            penetration_distance_source="minimum signed object-mesh gap of every full hand link cloud",
            semantic_patch_labels=getattr(hand, "semantic_patch_labels", None),
            active_contact_mask=self.reference_contact_mask.tolist(),
            active_contact_count=int(self.reference_contact_mask.sum()),
            contact_metric_scope="active reference contact slots only",
            approach_source="fixed template hand.local and fixed object-local initialization anchors",
            reference="initial object pose and zero world linear/angular velocity",
            angular_velocity="explicit body-local omega rotated into world by dex loss",
            hard_constraints="original q0 geometric constraints only",
        )

    def evaluate(self, worlds):
        node, job = self.node, self.job
        for k, qpos in enumerate(node.explicit.qpos):
            if k < node.explicit.horizon:
                pose = node.explicit.stages[k].pose
            else:
                pose = self.terminal_pose
                node.backend.kinematics.evaluate(qpos, pose)
            job.state.qpos = qpos
            job.state.body_position, job.state.body_matrix = pose.body_position, pose.body_matrix
            if k == 0:
                job.state.qvel.zero_()
            else:
                wp.launch(map_velocity, worlds, inputs=[node.explicit.stages[k - 1].velocity,
                    self.hand_dof, job.model.object_dof, job.state.qvel], device=node.device)
            wp.launch(measure_gaps, (worlds, job.model.output_count), inputs=[node.cost.object_mesh,
                job.model.object_body, self.bodies, self.starts, self.ends, node.cost.sample_local,
                pose.body_position, pose.body_matrix, job.state.contact_distance, job.state.contact_active],
                device=node.device)
            if self.semantic:
                wp.launch(measure_guidance_gaps, (worlds, job.model.guidance_count), inputs=[
                    node.cost.object_mesh, job.model.object_body, self.patch_bodies,
                    self.patch_starts, self.patch_ends, self.patch_points,
                    pose.body_position, pose.body_matrix, job.model.output_count,
                    job.state.contact_distance, job.state.contact_active,
                ], device=node.device)
            wp.launch(dex._prepare_auxiliary, (worlds, job.model.guidance_count), inputs=[job], device=node.device)
            wp.launch(score_state, worlds, inputs=[job, k, self.step_parts, self.metrics], device=node.device)
            if self.hand_tracking_enabled:
                wp.launch(score_hand_tracking, worlds, inputs=[job, k, self.step_parts], device=node.device)
        wp.launch(aggregate, worlds, inputs=[self.step_parts, self.weights["terminal"], self.parts, self.cost],
                  device=node.device)
        wp.launch(synchronize_ranking_parts, worlds, inputs=[self.parts, self.cost, node.cost.parts],
                  device=node.device)
