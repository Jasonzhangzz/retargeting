"""CARD CEM with optional explicit-contact pose prediction and refinement.

Distances inside the objective are millimetres; configurations/meshes use metres.
Rotation search uses a local SO(3) chart about the initialization, not Euler noise.
The compliant contact model predicts motion and supplies posture/control gradients;
the full ComFree executor and the ADMM solver are separate validation paths.
"""
from types import SimpleNamespace
from time import perf_counter
from retarget.grasp_capture import capture_guard
import math

import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation
from forceaware import mpc_loss as fa


DEFAULTS = dict(population=1024, iterations=16, elite_fraction=.05,
                temperature=.1, translation_sigma_mm=2., rotation_sigma=.016,
                joint_sigma=.05, final_noise=.2)
COST_PROFILE = dict(contact=40., direction_price=2., direction_radius_mm=10.,
    correspondence=.15, anti_slip=2.5, fingertip=3., pad_normal=60., bone=20.,
    joint_smooth=.1, translation_smooth=math.sqrt(125.), rotation_smooth=10.,
    self_collision=1.5, self_margin_mm=6., object_collision=14.,
    object_allowance_mm=2., hard_penetration_mm=3., self_clearance_mm=.5)


def force_priorities(forces, count):
    """Optional forces are magnitudes in N, in semantic-contact order."""
    if forces is None:
        return np.ones(count, np.float32)
    values = np.asarray(forces, dtype=float)
    if values.shape != (count,) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('Contact forces must be finite, nonnegative, one per contact')
    maximum = values.max(initial=0.)
    return (1.+.5*values/maximum).astype(np.float32) if maximum > 0 else np.ones(count, np.float32)


@wp.struct
class CemState:
    x: wp.array2d(dtype=float)
    qpos: wp.array2d(dtype=float)
    seed_qpos: wp.array(dtype=float)
    hand_ids: wp.array(dtype=int)
    seed: wp.array(dtype=float)
    low: wp.array(dtype=float)
    high: wp.array(dtype=float)
    initial_sigma: wp.array(dtype=float)
    mean: wp.array(dtype=float)
    best: wp.array(dtype=float)
    scores: wp.array(dtype=float)
    selection_scores: wp.array(dtype=float)
    rank: wp.array(dtype=int)
    best_score: wp.array(dtype=float)
    history: wp.array(dtype=float)
    restoration_history: wp.array(dtype=int)
    root_rotation: wp.mat33
    population: int
    iterations: int
    elites: int
    random_seed: int
    temperature: float
    final_noise: float


@wp.func
def rotation_matrix(v: wp.vec3):
    angle = wp.length(v)
    axis = wp.vec3(1., 0., 0.)
    if angle > 1.e-10:
        axis = v/angle
    return wp.quat_to_matrix(wp.quat_from_axis_angle(axis, angle))


@wp.func
def xyz_angles(r: wp.mat33):
    y = wp.asin(wp.clamp(r[0, 2], -1., 1.))
    x = wp.atan2(-r[1, 2], r[2, 2])
    z = wp.atan2(-r[0, 1], r[0, 0])
    if wp.abs(wp.cos(y)) < 1.e-6:
        x = wp.atan2(r[2, 1], r[1, 1])
        z = 0.
    return wp.vec3(x, y, z)


@wp.kernel(enable_backward=False)
def initialize(s: CemState):
    j = wp.tid()
    s.mean[j] = s.seed[j]
    s.best[j] = s.seed[j]
    if j == 0:
        s.best_score[0] = float(wp.inf)


@wp.kernel(enable_backward=False)
def sample(s: CemState, generation: int):
    world = wp.tid()
    # i=0,...,K-1 as in Eq.13; the distribution after K updates has gamma*sigma0.
    scale = wp.pow(s.final_noise, float(generation)/float(s.iterations))
    for j in range(s.seed.shape[0]):
        rng = wp.rand_init(s.random_seed+generation*104729, world*26+j)
        value = s.mean[j]+s.initial_sigma[j]*scale*wp.randn(rng)
        if j >= 26:
            # Acquire contacts geometrically before optimizing servo forces.
            value = 0.
        if world == 0:
            value = s.best[j]
        elif world == 1 and s.population > 2:
            value = s.mean[j]
        s.x[world, j] = wp.clamp(value, s.low[j], s.high[j])
    delta = wp.vec3(s.x[world, 0]-s.seed[0], s.x[world, 1]-s.seed[1], s.x[world, 2]-s.seed[2])
    if wp.length(delta) > .02:
        delta = .02*wp.normalize(delta)
        for j in range(3):
            s.x[world, j] = s.seed[j]+delta[j]


@wp.kernel(enable_backward=False)
def scatter(s: CemState, restore: int):
    world = wp.tid()
    for j in range(s.seed_qpos.shape[0]):
        s.qpos[world, j] = s.seed_qpos[j]
    if restore == 1:
        for j in range(s.seed.shape[0]):
            s.x[world, j] = s.best[j]
    angles = xyz_angles(s.root_rotation@rotation_matrix(wp.vec3(s.x[world, 3], s.x[world, 4], s.x[world, 5])))
    for j in range(26):
        value = s.x[world, j]
        if j >= 3 and j < 6:
            value = angles[j-3]
        s.qpos[world, s.hand_ids[j]] = value


@wp.kernel(enable_backward=False)
def rank_candidates(s: CemState):
    world = wp.tid()
    rank = int(0)
    for other in range(s.population):
        if s.selection_scores[other] < s.selection_scores[world] or (s.selection_scores[other] == s.selection_scores[world] and other < world):
            rank += 1
    s.rank[world] = rank


@wp.kernel(enable_backward=False)
def update_distribution(s: CemState):
    j = wp.tid()
    count = int(0)
    total = float(0.)
    minimum = float(wp.inf)
    winner = int(0)
    for w in range(s.population):
        if s.rank[w] == 0:
            winner = w
        if s.rank[w] < s.elites and wp.isfinite(s.selection_scores[w]):
            count += 1
            total += s.selection_scores[w]
            minimum = wp.min(minimum, s.selection_scores[w])
    if count > 0:
        average = total/float(count)
        variance = float(0.)
        for w in range(s.population):
            if s.rank[w] < s.elites and wp.isfinite(s.selection_scores[w]):
                variance += wp.pow(s.selection_scores[w]-average, 2.)/float(count)
        std = wp.max(wp.sqrt(variance), 1.e-12)
        weighted = float(0.)
        norm = float(0.)
        for w in range(s.population):
            if s.rank[w] < s.elites and wp.isfinite(s.selection_scores[w]):
                weight = wp.exp(-(s.selection_scores[w]-minimum)/(std*s.temperature))
                weighted += weight*s.x[w, j]
                norm += weight
        s.mean[j] = weighted/norm
        if s.scores[winner] < s.best_score[0]:
            s.best[j] = s.x[winner, j]


@wp.kernel(enable_backward=False)
def commit_score(s: CemState, generation: int):
    for w in range(s.population):
        if s.rank[w] == 0:
            s.best_score[0] = wp.min(s.best_score[0], s.scores[w])
    s.history[generation] = s.best_score[0]


@wp.struct
class CostData:
    position: wp.array2d(dtype=wp.vec3)
    rotation: wp.array2d(dtype=wp.mat33)
    seed_position: wp.array(dtype=wp.vec3)
    seed_rotation: wp.array(dtype=wp.mat33)
    # All hand-surface samples, grouped by semantic link.
    sample_body: wp.array(dtype=int)
    sample_local: wp.array(dtype=wp.vec3)
    sample_normal: wp.array(dtype=wp.vec3)
    sample_pad: wp.array(dtype=int)
    group_body: wp.array(dtype=int)
    group_start: wp.array(dtype=int)
    group_end: wp.array(dtype=int)
    preference: wp.array(dtype=wp.vec3)
    previous_local: wp.array(dtype=wp.vec3)
    anchor: wp.array(dtype=wp.vec3)
    push: wp.array(dtype=wp.vec3)
    rho: wp.array(dtype=float)
    alpha: wp.array(dtype=float)
    selected: wp.array2d(dtype=int)
    parts: wp.array2d(dtype=float)
    pose_cost: wp.array(dtype=float)
    pose_enabled: int
    region_enabled: int
    sample_forbidden: wp.array(dtype=int)
    forbidden_gap: wp.array(dtype=float)
    penetration: wp.array(dtype=float)
    palm_gap: wp.array(dtype=float)
    self_gap: wp.array(dtype=float)
    self_clearance: float
    object_mesh: wp.uint64
    object_body: int
    palm_mesh: wp.uint64
    palm_body: int
    style_body: wp.array(dtype=int)
    style_apex: wp.array(dtype=wp.vec3)
    style_normal: wp.array(dtype=wp.vec3)
    style_axis: wp.array(dtype=wp.vec3)
    capsule_body: wp.array(dtype=int)
    capsule_start: wp.array(dtype=wp.vec3)
    capsule_end: wp.array(dtype=wp.vec3)
    capsule_radius: wp.array(dtype=float)
    box_center: wp.array(dtype=wp.vec3)
    box_half: wp.array(dtype=wp.vec3)
    box_certificate: int
    pair_left: wp.array(dtype=int)
    pair_right: wp.array(dtype=int)


@wp.func
def world_point(d: CostData, w: int, b: int, local: wp.vec3):
    return d.position[w, b]+d.rotation[w, b]@local


@wp.func
def seed_point(d: CostData, b: int, local: wp.vec3):
    return d.seed_position[b]+d.seed_rotation[b]@local


@wp.kernel(enable_backward=False)
def clear_cost(d: CostData):
    w = wp.tid()
    for k in range(4):
        d.parts[w, k] = 0.
    d.penetration[w] = 0.
    d.palm_gap[w] = float(wp.inf)
    d.self_gap[w] = float(wp.inf)
    if d.region_enabled != 0:
        d.forbidden_gap[w] = float(wp.inf)


@wp.kernel(enable_backward=False)
def contact_cost(d: CostData):
    w, group = wp.tid()
    body = d.group_body[group]
    obj = d.object_body
    anchor = world_point(d, w, obj, d.anchor[group])
    push = d.rotation[w, obj]@d.push[group]
    pref = world_point(d, w, body, d.preference[group])
    best = int(d.group_start[group])
    best_value = float(wp.inf)
    for i in range(d.group_start[group], d.group_end[group]):
        if d.sample_pad[i] != 0:
            point = world_point(d, w, body, d.sample_local[i])
            value = wp.length_sq(point-anchor)+d.alpha[group]*wp.length_sq(point-pref)
            if value < best_value:
                best_value = value
                best = i
    d.selected[w, group] = best
    point = world_point(d, w, body, d.sample_local[best])
    normal = d.rotation[w, body]@d.sample_normal[best]
    # deltaT maps the previous selected pad point into the current link pose.
    previous = world_point(d, w, body, d.previous_local[group])
    reach = wp.length_sq((point-anchor)*1000.)
    cone = 2.*100.*(1.-wp.clamp(wp.dot(normal, push), -1., 1.))
    fix = wp.length_sq((point-previous)*1000.)
    value = d.rho[group]*d.rho[group]*(1600.*(reach+cone)+6.25*fix)
    wp.atomic_add(d.parts, w, 0, value)


@wp.kernel(enable_backward=False)
def style_cost(d: CostData):
    w, i = wp.tid()
    body = d.style_body[i]
    normal = d.style_normal[i]
    axis = d.style_axis[i]
    value = 400.*(1.-wp.clamp(wp.dot(d.rotation[w, body]@axis, d.seed_rotation[body]@axis), -1., 1.))
    # Only distal regions carry apex and belly-normal style terms.
    if i % 3 == 2:
        delta = (world_point(d, w, body, d.style_apex[i])-seed_point(d, body, d.style_apex[i]))*1000.
        value += 9.*wp.length_sq(delta)
        value += 3600.*(1.-wp.clamp(wp.dot(d.rotation[w, body]@normal, d.seed_rotation[body]@normal), -1., 1.))
    wp.atomic_add(d.parts, w, 1, value)


@wp.kernel(enable_backward=False)
def smooth_cost(d: CostData, s: CemState):
    w = wp.tid()
    value = float(0.)
    for j in range(s.seed.shape[0]):
        delta = s.x[w, j]-s.seed[j]
        if j < 3:
            value += 125.*delta*delta*1.e6
        elif j < 6:
            value += 100.*delta*delta
        elif j >= 26:
            value += .01*delta*delta
        else:
            value += .01*delta*delta
    d.parts[w, 2] = value


@wp.kernel(enable_backward=False)
def object_cost(d: CostData):
    w, i = wp.tid()
    point = world_point(d, w, d.sample_body[i], d.sample_local[i])
    local = wp.transpose(d.rotation[w, d.object_body])@(point-d.position[w, d.object_body])
    hit = wp.mesh_query_point(d.object_mesh, local, 1.e6)
    depth = float(wp.inf)
    if hit.result:
        nearest = wp.mesh_eval_position(d.object_mesh, hit.face, hit.u, hit.v)
        depth = wp.max(-hit.sign*wp.length(local-nearest)*1000., 0.)
        if d.region_enabled != 0 and d.sample_forbidden[i] != 0:
            gap = hit.sign*wp.length(local-nearest)*1000.
            wp.atomic_min(d.forbidden_gap, w, gap)
            wp.atomic_add(d.parts, w, 3, 2500.*wp.pow(wp.max(1.-gap, 0.), 2.))
    wp.atomic_max(d.penetration, w, depth)
    wp.atomic_add(d.parts, w, 3, 196.*wp.pow(wp.max(depth-2., 0.), 2.))


@wp.func
def matrix_axis(r: wp.mat33, i: int):
    return wp.vec3(r[0, i], r[1, i], r[2, i])


@wp.func
def box_separation(d: CostData, w: int, a: int, b: int):
    """SAT projection gap: a positive value certifies full mesh separation."""
    ba, bb = d.capsule_body[a], d.capsule_body[b]
    ra, rb = d.rotation[w, ba], d.rotation[w, bb]
    delta = world_point(d, w, ba, d.box_center[a])-world_point(d, w, bb, d.box_center[b])
    gap = float(-wp.inf)
    for k in range(15):
        axis = wp.vec3(0.)
        if k < 3:
            axis = matrix_axis(ra, k)
        elif k < 6:
            axis = matrix_axis(rb, k-3)
        else:
            axis = wp.cross(matrix_axis(ra, (k-6)//3), matrix_axis(rb, (k-6)%3))
        length = wp.length(axis)
        if length > 1.e-6:
            axis = axis/length
            radius = float(0.)
            for j in range(3):
                radius += wp.abs(wp.dot(axis, matrix_axis(ra, j)))*d.box_half[a][j]
                radius += wp.abs(wp.dot(axis, matrix_axis(rb, j)))*d.box_half[b][j]
            gap = wp.max(gap, wp.abs(wp.dot(delta, axis))-radius)
    return gap*1000.


@wp.kernel(enable_backward=False)
def capsule_cost(d: CostData):
    w, i = wp.tid()
    a, b = d.pair_left[i], d.pair_right[i]
    p0 = world_point(d, w, d.capsule_body[a], d.capsule_start[a])
    p1 = world_point(d, w, d.capsule_body[a], d.capsule_end[a])
    q0 = world_point(d, w, d.capsule_body[b], d.capsule_start[b])
    q1 = world_point(d, w, d.capsule_body[b], d.capsule_end[b])
    gap = (wp.sqrt(fa._segment_distance_squared(p0, p1, q0, q1)+1.e-16)-d.capsule_radius[a]-d.capsule_radius[b])*1000.
    hard_gap = gap
    if d.box_certificate != 0:
        hard_gap = wp.max(hard_gap, box_separation(d, w, a, b))
    wp.atomic_min(d.self_gap, w, hard_gap)
    wp.atomic_add(d.parts, w, 3, 2.25*wp.pow(wp.max(6.-gap, 0.), 2.))


@wp.kernel(enable_backward=False)
def finish_cost(d: CostData, s: CemState):
    w = wp.tid()
    value = float(0.)
    for k in range(4):
        value += d.parts[w, k]
    if d.pose_enabled != 0:
        value += d.pose_cost[w]
    if (d.penetration[w] > 3. or d.palm_gap[w] < 6.
            or d.self_gap[w] < d.self_clearance or not wp.isfinite(value)):
        value = float(wp.inf)
    if d.region_enabled != 0 and d.forbidden_gap[w] < 1.:
        value = float(wp.inf)
    s.scores[w] = value


@wp.kernel(enable_backward=False)
def prepare_ranking(d: CostData, s: CemState, generation: int):
    """Restore feasibility only when the whole population violates hard limits.

    Invalid poses can guide the search mean, but can never become the returned
    best: update_distribution/commit_score retain the original hard scores.
    """
    w = wp.tid()
    feasible = bool(False)
    for i in range(s.population):
        feasible = feasible or wp.isfinite(s.scores[i])
    value = s.scores[w]
    if not feasible:
        value = (wp.pow(wp.max(d.penetration[w]-3., 0.), 2.)
                 +wp.pow(wp.max(6.-d.palm_gap[w], 0.), 2.)
                 +wp.pow(wp.max(d.self_clearance-d.self_gap[w], 0.), 2.))
        if d.region_enabled != 0:
            value += wp.pow(wp.max(1.-d.forbidden_gap[w], 0.), 2.)
        for k in range(4):
            if not wp.isfinite(d.parts[w, k]):
                value = float(wp.inf)
        if d.pose_enabled != 0 and not wp.isfinite(d.pose_cost[w]):
            value = float(wp.inf)
    s.selection_scores[w] = value
    if w == 0:
        s.restoration_history[generation] = int(not feasible)


@wp.kernel(enable_backward=False)
def export_contacts(d: CostData, annotation_group: wp.array(dtype=int),
                    local: wp.array(dtype=wp.vec3), normal: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    sample_index = d.selected[0, annotation_group[i]]
    local[i] = d.sample_local[sample_index]
    normal[i] = d.sample_normal[sample_index]


def initialize_runtime(runtime):
    """Compile geometry/FK without allocating the full executor population."""
    from comfree_warp.native_adjoint.fast_step import compile_fast_step
    from comfree_warp.native_adjoint.kinematics import allocate_output, kinematics
    compiled = compile_fast_step(runtime.model, runtime.device_model)
    device = compiled.base.device
    geometry = fa._finger_joint_geometry(runtime.model, compiled.base.collision, runtime.cfg.sequence.hand)
    loss_model = fa.LossModel()
    names = ('finger_joint_body', 'finger_joint_start', 'finger_joint_end', 'finger_joint_radius')
    for name, value, dtype in zip(names, geometry[:4], (int, wp.vec3, wp.vec3, float)):
        setattr(loss_model, name, wp.array(value, dtype=dtype, device=device))
    loss_model.finger_joint_pair_left = wp.array(geometry[4][:, 0], dtype=int, device=device)
    loss_model.finger_joint_pair_right = wp.array(geometry[4][:, 1], dtype=int, device=device)
    loss_model.finger_joint_pair_count = len(geometry[4])
    loss_model.object_body = runtime.index.object_body
    pose = allocate_output(compiled.base.kinematics, 1, requires_grad=False)
    qpos = wp.array(runtime.qpos[None], dtype=float, device=device)
    kinematics(compiled.base.kinematics, qpos, pose)
    job = fa.LossJob()
    job.model = loss_model
    state = fa.LossState()
    state.body_position, state.body_matrix, state.qpos = pose.body_position, pose.body_matrix, qpos
    state.qvel = wp.zeros((1, runtime.model.nv), dtype=float, device=device)
    job.state = state
    workspace = SimpleNamespace(job=job)
    loss = SimpleNamespace(immediate=[workspace])
    backend = SimpleNamespace(device=device, compiled=compiled, steps=(), loss=loss)
    runtime.optimizer = SimpleNamespace(backend=backend, compiled=compiled, loss=loss, last_solver_diag={})
    runtime.static_pose = pose
    runtime.output_workspace = workspace
    runtime.output_step = 0
    runtime.prediction_horizon_seconds = 0.
    runtime.grasp_cost_profile = COST_PROFILE.copy()


def _mesh_arrays(model, body):
    from retarget.grasp_hand import rigid_mesh_arrays
    return rigid_mesh_arrays(model, body)


def pad_chart(model, body, points, normals):
    """Wuji palmar face and longitudinal centreline in the link's local frame.

    Flexion axis x bone direction points out of the palmar face. The model
    profile resolves Hand 2's MCP abduction axis and different palm frame.
    Keep the front hemisphere including its rounded rim;
    full-body samples are still used for object penetration.
    """
    from retarget.grasp_hand import palmar_axis
    name = model.body(int(body)).name
    endpoint = None
    belly = palmar_axis(model, body, endpoint) if name == 'right_palm_link' else None
    if name.startswith('right_finger'):
        finger, link = int(name[len('right_finger')]), int(name[-1])
        endpoint = (model.body(f'right_finger{finger}_link{link+1}').pos.copy() if link < 4
                    else model.site(('right_thumb_tip', 'right_index_tip', 'right_middle_tip',
                                     'right_ring_tip', 'right_pinky_tip')[finger-1]).pos.copy())
        belly = palmar_axis(model, body, endpoint)
    mask = normals@belly > .25
    if not mask.any():
        raise ValueError(f'No palmar surface samples: {name}')
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
        axis = ends[i]-starts[i]
        t = np.clip((vertices-starts[i])@axis/max(float(axis@axis), 1.e-20), 0., 1.)
        distance = np.linalg.norm(vertices-starts[i]-t[:, None]*axis, axis=1)
        result[i] = max(result[i], float(distance.max())+1.e-6)
    return result


class CardCemNode:
    """GPU scheduling protocol shared with the AdamW scene nodes."""
    def __init__(self, runtime, *, capture_standalone=True):
        self.started = perf_counter()
        self.runtime = runtime
        self.backend = runtime.optimizer.backend
        self.device = self.backend.device
        self.stream = wp.Stream(self.device)
        self.start_event, self.end_event, self.done = [wp.Event(self.device, enable_timing=True) for _ in range(3)]
        self.selected = wp.zeros(1, dtype=int, device=self.device)
        self._initialize_search()
        self.explicit = None
        if getattr(runtime.args, 'explicit_pose', False):
            from retarget.grasp_explicit import ExplicitPoseModel
            self.explicit = ExplicitPoseModel(self)
            self.cost.pose_cost, self.cost.pose_enabled = self.explicit.cost, 1
            runtime.prediction_horizon_seconds = self.explicit.horizon*self.explicit.parameters.dt
        # Warm FK, optional explicit prediction and its VJP before capture.
        refinement_steps = self.explicit.refinement_steps if self.explicit is not None else 0
        if self.explicit is not None:
            self.explicit.refinement_steps = min(refinement_steps, 1)
        with wp.ScopedStream(self.stream):
            self._compute(generations=1)
        wp.synchronize_stream(self.stream)
        if self.explicit is not None:
            self.explicit.refinement_steps = refinement_steps
        self.audit = None
        from retarget.grasp_device_audit import GraspDeviceAudit
        self.audit = GraspDeviceAudit(runtime, self.selected, self.stream, capture_standalone=False)
        self.graph = None
        if capture_standalone:
            with capture_guard(), wp.ScopedStream(self.stream), wp.ScopedCapture(stream=self.stream) as capture:
                self._compute()
            self.graph = capture.graph
        self.prepare_seconds = perf_counter()-self.started

    def _initialize_search(self):
        """Allocate the numerical search independently of the CUDA scheduler."""
        from comfree_warp.native_adjoint.kinematics import allocate_output
        from retarget.grasp_hand import query_path
        self.explicit = None
        runtime = self.runtime
        args = runtime.args
        self.iterations = getattr(args, 'cem_iterations', DEFAULTS['iterations'])
        n = getattr(args, 'cem_population', DEFAULTS['population'])
        fraction = getattr(args, 'cem_elite_fraction', .05)
        if n < 2 or self.iterations < 1 or not 0 < fraction <= 1:
            raise ValueError('CARD requires population >=2, iterations >=1, and elite fraction in (0,1]')
        s = self.state = CemState()
        s.population, s.iterations = n, self.iterations
        s.elites = max(1, math.ceil(n*fraction))
        s.temperature = getattr(args, 'cem_temperature', .1)
        s.final_noise = getattr(args, 'cem_final_noise', .2)
        s.random_seed = args.seed
        if not math.isfinite(s.temperature) or s.temperature <= 0 or not 0 < s.final_noise <= 1:
            raise ValueError('CARD requires a positive temperature and final noise ratio in (0,1]')
        array = lambda value, dtype=float: wp.array(np.asarray(value), dtype=dtype, device=self.device)
        self.array = array
        q = runtime.qpos[np.asarray(runtime.index.hand_qpos)].copy()
        seed = q.copy(); seed[3:6] = 0.
        preload = bool(getattr(args, 'explicit_pose', False))
        if preload:
            seed = np.r_[seed, np.zeros(20)]
        s.root_rotation = wp.mat33(Rotation.from_euler('XYZ', q[3:6]).as_matrix())
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
                low[j], high[j] = max(low[j], q[j]-.25), min(high[j], q[j]+.25)
            elif j < 3:
                low[j], high[j] = max(low[j], q[j]-.02), min(high[j], q[j]+.02)
        if np.any(low > high):
            raise ValueError('Initialization is too far outside joint limits for the CARD per-frame cap')
        if preload:
            limit = getattr(args, 'explicit_preload_limit', .5)
            low, high = np.r_[low, np.full(20, -limit)], np.r_[high, np.full(20, limit)]
        s.low, s.high = array(low), array(high)
        sigma = np.r_[np.full(3, getattr(args, 'cem_translation_sigma_mm', 2.)/1000),
                      np.full(3, getattr(args, 'cem_rotation_sigma', .016)),
                      np.full(20, getattr(args, 'cem_joint_sigma', .05))]
        if preload:
            sigma = np.r_[sigma, np.full(20, limit/3.)]
        if not np.isfinite(sigma).all() or np.any(sigma <= 0):
            raise ValueError('CARD initial sampling noise must be finite and positive')
        s.initial_sigma = array(sigma)
        for name in ('mean', 'best'):
            setattr(s, name, wp.zeros(len(seed), dtype=float, device=self.device))
        s.x = wp.zeros((n, len(seed)), dtype=float, device=self.device)
        s.qpos = wp.empty((n, runtime.model.nq), dtype=float, device=self.device)
        s.scores = wp.empty(n, dtype=float, device=self.device)
        s.selection_scores = wp.empty(n, dtype=float, device=self.device)
        s.rank = wp.empty(n, dtype=int, device=self.device)
        s.best_score = wp.empty(1, dtype=float, device=self.device)
        s.history = wp.empty(self.iterations, dtype=float, device=self.device)
        s.restoration_history = wp.zeros(self.iterations, dtype=int, device=self.device)
        self.pose = allocate_output(self.backend.compiled.base.kinematics, n, requires_grad=False)
        wp.synchronize_device(self.device)
        self.cost = d = CostData()
        d.position, d.rotation = self.pose.body_position, self.pose.body_matrix
        d.seed_position = array(runtime.static_pose.body_position.numpy()[0], wp.vec3)
        d.seed_rotation = array(runtime.static_pose.body_matrix.numpy()[0], wp.mat33)
        auditor, hand, _ = runtime.audit_context
        d.object_mesh, d.object_body = auditor.mesh.id, runtime.index.object_body
        d.palm_body = runtime.model.body('right_palm_link').id
        pv, pf = _mesh_arrays(runtime.model, d.palm_body)
        self.palm_vertices, self.palm_faces = array(pv, wp.vec3), array(pf.reshape(-1), int)
        self.palm_mesh = wp.Mesh(points=self.palm_vertices, indices=self.palm_faces)
        d.palm_mesh = self.palm_mesh.id
        surface, normals, bodies, pads, starts, ends = [], [], [], [], {}, {}
        from retarget.grasp_regions import forbidden_mask
        with np.load(runtime.experiment/'card_input.npz') as supplied:
            region_normals = supplied['region_reference_normals'].copy() if 'region_reference_normals' in supplied.files else None
        d.region_enabled = int(region_normals is not None)
        forbidden = []
        self.pad_charts = {}
        self.surfaces = {}
        for body in np.unique(hand.all_bodies):
            name = runtime.model.body(int(body)).name
            with np.load(query_path(runtime.model, name)) as asset:
                p, normal = asset['local_pos'], asset['local_normal']
            normal = normal/np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1.e-12)
            starts[body], ends[body] = len(surface), len(surface)+len(p)
            mask, belly, endpoint = pad_chart(runtime.model, body, p, normal)
            self.pad_charts[int(body)] = mask, belly, endpoint
            surface.extend(p); normals.extend(normal); bodies.extend([body]*len(p)); pads.extend(mask)
            forbidden.extend(forbidden_mask(runtime.model, body, normal, hand.bodies, region_normals)
                             if region_normals is not None else np.zeros(len(p), bool))
            self.surfaces[int(body)] = (p, normal)
        d.sample_body, d.sample_local, d.sample_normal = array(bodies, int), array(surface, wp.vec3), array(normals, wp.vec3)
        d.sample_pad = array(pads, int)
        d.sample_forbidden = array(forbidden, int)
        d.forbidden_gap = wp.full(s.population, wp.inf, dtype=float, device=self.device)
        groups, inverse = np.unique(hand.bodies, return_inverse=True)
        d.group_body, d.group_start, d.group_end = array(groups, int), array([starts[b] for b in groups], int), array([ends[b] for b in groups], int)
        self.annotation_group = array(inverse, int)
        self.annotation_bodies = hand.bodies.copy()
        self.annotation_count = len(hand.bodies)
        # Static reference input is persisted independently of any closing/preload reference.
        with np.load(runtime.experiment/'card_input.npz') as supplied:
            anchors, outward = supplied['object_points'].copy(), supplied['object_normals'].copy()
            forces = supplied['contact_forces'].copy() if 'contact_forces' in supplied else None
        force_priorities(forces, self.annotation_count)  # validate persisted optional input
        if (anchors.shape != (self.annotation_count, 3) or outward.shape != anchors.shape
                or not np.isfinite(anchors).all() or not np.isfinite(outward).all()):
            raise ValueError('CARD requires finite Nx3 object contact positions and normals')
        group_forces = None if forces is None else np.bincount(inverse, weights=forces, minlength=len(groups))
        preferences, target, push, alpha, previous = [], [], [], [], []
        seed_pos, seed_rot = d.seed_position.numpy(), d.seed_rotation.numpy()
        for b in groups:
            ids = np.flatnonzero(hand.bodies == b)
            weights = np.ones(len(ids)) if forces is None else forces[ids]
            if weights.sum() <= 0:
                weights = np.ones(len(ids))
            preference = np.average(hand.local[ids], axis=0, weights=weights)
            anchor = np.average(anchors[ids], axis=0, weights=weights)
            direction = -np.average(outward[ids], axis=0, weights=weights)
            if np.linalg.norm(direction) < 1.e-8:
                raise ValueError('Semantic contact has cancelling/zero push normals')
            direction /= np.linalg.norm(direction)
            name = runtime.model.body(int(b)).name
            a = .15 if name.endswith(('link3', 'link4')) else 0.
            p, _ = self.surfaces[int(b)]
            mask, belly, endpoint = self.pad_charts[int(b)]
            if endpoint is not None:
                u = np.clip(np.dot(preference, endpoint)/np.dot(endpoint, endpoint), 0., 1.)
                # Discrete palmar centreline at the remapped fractional bone coordinate.
                score = np.sum((p-u*endpoint)**2, axis=-1)
                preference = p[np.argmin(np.where(mask, score, np.inf))]
            world = p@seed_rot[b].T+seed_pos[b]
            # Object is fixed at the input pose (normally identity).
            obj = d.object_body
            anchor_world = seed_rot[obj]@anchor+seed_pos[obj]
            scores = np.sum((world-anchor_world)**2, axis=1)+a*np.sum((p-preference)**2, axis=1)
            previous.append(p[np.argmin(np.where(mask, scores, np.inf))])
            preferences.append(preference); target.append(anchor); push.append(direction); alpha.append(a)
        d.preference, d.previous_local = array(preferences, wp.vec3), array(previous, wp.vec3)
        d.anchor, d.push, d.alpha = array(target, wp.vec3), array(push, wp.vec3), array(alpha)
        d.rho = array(force_priorities(group_forces, len(groups)))
        self.force_source = 'uniform_no_force_measurement' if forces is None else 'provided_contact_force_magnitudes'
        d.selected = wp.zeros((n, len(groups)), dtype=int, device=self.device)
        d.parts = wp.zeros((n, 4), dtype=float, device=self.device)
        d.penetration, d.palm_gap = wp.zeros(n, device=self.device), wp.empty(n, device=self.device)
        d.self_gap = wp.empty(n, dtype=float, device=self.device)
        d.self_clearance = getattr(args, 'cem_self_clearance_mm', .5)
        if not math.isfinite(d.self_clearance) or d.self_clearance < 0:
            raise ValueError('CEM self clearance must be finite and nonnegative')
        style_bodies, apex, pad_normal, axis = [], [], [], []
        for finger in range(1, 6):
            for link in (2, 3, 4):
                body = runtime.model.body(f'right_finger{finger}_link{link}').id
                p, normal = self.surfaces[body]
                mask, belly, endpoint = self.pad_charts[body]
                axis.append(endpoint/max(np.linalg.norm(endpoint), 1.e-12))
                # The apex stays on the palmar pad; its belly axis is independent of cap normals.
                score = np.sum((p-endpoint)**2, axis=-1)
                nearest = np.argmin(np.where(mask, score, np.inf))
                style_bodies.append(body); apex.append(p[nearest]); pad_normal.append(belly)
        d.style_body, d.style_apex = array(style_bodies, int), array(apex, wp.vec3)
        d.style_normal, d.style_axis = array(pad_normal, wp.vec3), array(axis, wp.vec3)
        m = runtime.output_workspace.job.model
        d.capsule_body, d.capsule_start, d.capsule_end, d.capsule_radius = m.finger_joint_body, m.finger_joint_start, m.finger_joint_end, m.finger_joint_radius
        old_radii = m.finger_joint_radius.numpy()
        enclosing_radii = enclosing_capsule_radii(runtime.model, m.finger_joint_body.numpy(),
            m.finger_joint_start.numpy(), m.finger_joint_end.numpy(), old_radii)
        d.capsule_radius = array(enclosing_radii)
        self.maximum_capsule_radius_increase_mm = float(np.max(enclosing_radii-old_radii))*1000.
        d.pair_left, d.pair_right = m.finger_joint_pair_left, m.finger_joint_pair_right
        from retarget.grasp_hand import model_hand, body_boxes
        if model_hand(runtime.model) == 'wuji2':
            center, half = body_boxes(runtime.model, d.capsule_body.numpy())
            d.box_center, d.box_half = array(center, wp.vec3), array(half, wp.vec3)
            d.box_certificate = 1
        # Wuji has an extra abduction base before the semantic proximal link.
        # Both belong to the palm/proximal articulation, so its connected cap
        # must not be tested against the palm as a nonadjacent collision pair.
        # They remain in the inter-finger soft capsule penalty.
        cb = d.capsule_body.numpy()
        parent = runtime.model.body_parentid[cb]
        grandparent = runtime.model.body_parentid[parent]
        self.palm_capsules = array(np.flatnonzero((parent != d.palm_body)&
                                                   (grandparent != d.palm_body)), int)
        runtime.card_contact_local = wp.zeros(self.annotation_count, dtype=wp.vec3, device=self.device)
        runtime.card_contact_normal = wp.zeros_like(runtime.card_contact_local)
    def _score(self, worlds=None):
        from comfree_warp.native_adjoint.kinematics import kinematics
        d, s = self.cost, self.state
        worlds = s.population if worlds is None else worlds
        view = s.qpos if worlds == s.population else s.qpos[:worlds]
        kinematics(self.backend.compiled.base.kinematics, view, self.pose)
        wp.launch(clear_cost, worlds, inputs=[d], device=self.device)
        wp.launch(contact_cost, (worlds, len(d.group_body)), inputs=[d], device=self.device)
        wp.launch(style_cost, (worlds, 15), inputs=[d], device=self.device)
        wp.launch(smooth_cost, worlds, inputs=[d, s], device=self.device)
        wp.launch(object_cost, (worlds, len(d.sample_body)), inputs=[d], device=self.device)
        wp.launch(capsule_cost, (worlds, len(d.pair_left)), inputs=[d], device=self.device)
        wp.launch(palm_clearance_masked, (worlds, len(self.palm_capsules)), inputs=[d, self.palm_capsules], device=self.device)
        predict = self.explicit is not None and worlds == 1 and getattr(self, 'scoring_prediction', True)
        d.pose_enabled = int(predict)
        if predict:
            self.explicit.forward(worlds)
        wp.launch(finish_cost, worlds, inputs=[d, s], device=self.device)

    def _compute_steps(self, generations=None):
        from comfree_warp.native_adjoint.kinematics import kinematics
        s, r = self.state, self.runtime
        wp.launch(initialize, len(s.seed), inputs=[s], device=self.device)
        self.selected.zero_()
        if self.explicit is not None:
            self.explicit.accepted.zero_()
            self.cost.pose_enabled = 0
            self.scoring_prediction = False
        yield
        for generation in range(self.iterations if generations is None else generations):
            wp.launch(sample, s.population, inputs=[s, generation], device=self.device)
            wp.launch(scatter, s.population, inputs=[s, 0], device=self.device)
            self._score()
            wp.launch(prepare_ranking, s.population, inputs=[self.cost, s, generation], device=self.device)
            wp.launch(rank_candidates, s.population, inputs=[s], device=self.device)
            wp.launch(update_distribution, len(s.seed), inputs=[s], device=self.device)
            wp.launch(commit_score, 1, inputs=[s, generation], device=self.device)
            yield
        wp.launch(scatter, s.population, inputs=[s, 1], device=self.device)
        if self.explicit is not None:
            self.cost.pose_enabled = 1
            self.scoring_prediction = True
            from retarget.grasp_explicit import gradient_proposal, accept_proposal
            e = self.explicit
            wp.copy(e.initial_x, s.best)
            self._score(worlds=1)
            # CEM history ranks geometry; refinement starts from its complete
            # gravity/contact score, never compares unlike objectives.
            wp.copy(s.best_score, s.scores, count=1)
            wp.copy(e.before, s.best_score)
            wp.copy(e.pose_before, e.cost, count=1)
            for iteration in range(e.refinement_steps):
                e.backward()
                wp.copy(e.base, s.best)
                # Every line-search proposal is evaluated against all five
                # costs and the unchanged geometry hard rejection rules.
                for backtrack in range(e.settings['line_search_steps']):
                    wp.launch(gradient_proposal, 1, inputs=[s.x, s.x.grad, e.base,
                        s.low, s.high, s.seed, s.initial_sigma,
                        e.settings['refinement_step_size']*(.5**backtrack),
                        26 if iteration < (e.refinement_steps+1)//2 else 0], device=self.device)
                    # FK and geometric constraints read qpos; the explicit
                    # predictor reads x. Both must evaluate this proposal.
                    wp.launch(scatter, 1, inputs=[s, 0], device=self.device)
                    self._score(worlds=1)
                    wp.launch(accept_proposal, 1, inputs=[s.x, s.scores, s.best, s.best_score,
                        e.accepted, e.history, iteration*e.settings['line_search_steps']+backtrack], device=self.device)
                wp.launch(scatter, s.population, inputs=[s, 1], device=self.device)
                yield
        self._score(worlds=1)
        if self.explicit is not None and self.explicit.refinement_steps > 0:
            # Retain the pose gradient at the final exported configuration.
            self.explicit.backward()
        wp.launch(export_contacts, self.annotation_count, inputs=[self.cost, self.annotation_group,
            r.card_contact_local, r.card_contact_normal], device=self.device)
        wp.copy(r.output_workspace.job.state.qpos, s.qpos, count=r.model.nq)
        kinematics(self.backend.compiled.base.kinematics, r.output_workspace.job.state.qpos, r.static_pose)
        yield

    def _compute(self, generations=None):
        for _ in self._compute_steps(generations):
            pass

    def launch(self):
        import torch
        if self.graph is None:
            raise RuntimeError('Node belongs to a combined CUDA graph')
        with wp.ScopedStream(self.stream), torch.cuda.stream(self.audit.torch_stream):
            wp.record_event(self.start_event)
            wp.capture_launch(self.graph)
            wp.record_event(self.end_event)
            self.audit.compute()
            wp.record_event(self.done)

    def export(self):
        from forceaware.optimizer import Plan
        r, s, d = self.runtime, self.state, self.cost
        qpos = s.qpos.numpy()[0].copy()
        historical_score = float(s.best_score.numpy()[0])
        final_score = float(s.scores.numpy()[0])
        success = bool(np.isfinite(historical_score) and np.isfinite(final_score))
        seconds = wp.get_event_elapsed_time(self.start_event, self.end_event, synchronize=False)/1000.
        parts = dict(zip(('contact', 'style', 'smooth', 'collision'), map(float, d.parts.numpy()[0])))
        if self.explicit is not None:
            parts['object_pose'] = float(self.explicit.cost.numpy()[0])
        loss = sum(parts.values())
        command = qpos[np.asarray(r.index.hand_qpos)].copy()
        if self.explicit is not None:
            command = self.explicit.hold.numpy()[0, np.asarray(r.index.hand_qpos)].copy()
        unclipped_command = command.copy()
        if self.explicit is not None:
            unclipped_command = qpos[np.asarray(r.index.hand_qpos)].copy()
            unclipped_command[6:] += s.best.numpy()[26:]
            unclipped_command -= self.explicit.stages[0].gravity.numpy()[0]/self.explicit.parameters.stiffness.numpy()
        controls = np.asarray(r.index.hand_ctrl)
        limits = r.model.actuator_ctrlrange[controls]
        limited = r.model.actuator_ctrllimited[controls].astype(bool)
        command[limited] = np.clip(command[limited], limits[limited, 0], limits[limited, 1])
        zero = np.zeros(r.model.nv, np.float32)
        plan = Plan(raw=s.best.numpy(), initial_qpos=qpos.copy(), next_qpos=qpos.copy(),
            next_qvel=zero, action=command, qpos_sequence=qpos[None], qvel_sequence=zero[None],
            control_sequence=command[None], loss=loss, drift=0., grad_norm=0., seconds=seconds,
            diag=dict(selected=0, scores=[loss], static_configuration=True, optimizer_success=success))
        groups = self.annotation_group.numpy()
        anchors = d.anchor.numpy()[groups]
        snapshot = dict(z=anchors[None], dual=np.zeros_like(anchors[None]),
            hand_local_points=r.card_contact_local.numpy(), hand_local_normals=r.card_contact_normal.numpy(),
            admm_active=np.array(False), static_configuration=np.array(True))
        diag = dict(solver='card_annealed_cem', evaluation_backend='native_warp_FK_mesh_SDF',
            gradient_source='explicit_model_reverse_AD' if self.explicit is not None else 'none',
            contact_admm_enabled=False, dynamics_rollouts=1 if self.explicit is not None else 0,
            full_executor_rollouts=0,
            scheduling='independent_cuda_graph_nodes', device_resident_iterations=True,
            host_readbacks_during_replay=0, host_syncs_during_replay=0,
            optimization_variables=('wrist xyz, local SO(3), 20 finger joints, 20 servo preload offsets; fixed object'
                if self.explicit is not None else 'wrist xyz, local SO(3), 20 finger joints; fixed object'),
            optimizer_success=success, termination=('static_solution' if success else
                'final_configuration_failed_hard_constraints' if np.isfinite(historical_score) else
                'no_valid_candidate_initialization_saved'),
            final_configuration_score=final_score if np.isfinite(final_score) else None,
            evaluation_worlds=s.population,
            cem_population=s.population, cem_elites=s.elites, cem_iterations=self.iterations,
            cem_temperature=s.temperature, cem_final_noise=s.final_noise,
            cem_ranking='contact_acquisition_geometry',
            cem_initial_sigma=s.initial_sigma.numpy().tolist(),
            cem_best_loss_history=[float(v) if np.isfinite(v) else None for v in s.history.numpy()],
            cem_restoration_generations=int(np.count_nonzero(s.restoration_history.numpy())),
            cost_groups=parts, cost_profile=dict(COST_PROFILE, self_clearance_mm=d.self_clearance), contact_force_source=self.force_source,
            contact_surface='palmar query samples; local belly-normal dot > 0.25',
            selected_surface_indices=d.selected.numpy()[0].tolist(),
            contact_group_body_ids=d.group_body.numpy().tolist(), contact_rho=d.rho.numpy().tolist(),
            hold_action_clipped_to_actuator_limits=bool(np.any(limited &
                ((unclipped_command < limits[:, 0]) | (unclipped_command > limits[:, 1])))),
            max_object_penetration_mm=float(d.penetration.numpy()[0]),
            minimum_interfinger_gap_mm=float(d.self_gap.numpy()[0]),
            self_clearance_mm=d.self_clearance,
            inter_finger_hard_geometry=('enclosing capsule or enclosing OBB separating-axis certificate'
                if d.box_certificate else 'capsules enclosing every triangle vertex of each finger mesh'),
            inter_finger_capsules=len(d.capsule_body), inter_finger_capsule_pairs=len(d.pair_left),
            maximum_capsule_radius_increase_mm=self.maximum_capsule_radius_increase_mm,
            minimum_palm_finger_gap_mm=float(d.palm_gap.numpy()[0]),
            palm_clearance_measurement='9 samples per middle/distal capsule against palm mesh SDF; adjacent base/proximal excluded',
            hard_constraints_satisfied=success, output_time_seconds=0.,
            prediction_horizon_seconds=r.prediction_horizon_seconds,
            graph_prepare_seconds=self.prepare_seconds, gpu_seconds=seconds)
        if hasattr(self, 'dispatch_timing'):
            diag['node_dispatch'] = self.dispatch_timing
        if d.region_enabled:
            diag['forbidden_region_min_gap_mm'] = float(d.forbidden_gap.numpy()[0])
            diag['forbidden_region_clearance_mm'] = 1.
        r.optimizer.last_solver_diag = diag
        r.predicted_terminal_qpos = qpos.copy()
        if self.explicit is not None:
            r.predicted_terminal_qpos, diag['explicit_prediction'] = self.explicit.export()
            diag['predicted_hold_feasible'] = diag['explicit_prediction']['predicted_hold_feasible']
            if success and not diag['predicted_hold_feasible']:
                diag['termination'] = 'predicted_motion_exceeds_tolerance'
            diag['explicit_model_steps_per_candidate'] = self.explicit.horizon
            diag['prediction_backend'] = 'projected_compliant_contact_dynamics'
            snapshot['explicit_predicted_qpos'] = np.stack([value.numpy()[0] for value in self.explicit.qpos])
            gradient = s.x.grad.numpy()[0]
            valid_gradient = bool(self.explicit.refinement_steps > 0 and np.isfinite(gradient).all())
            snapshot['explicit_pose_gradient_valid'] = np.array(valid_gradient)
            snapshot['explicit_pose_gradient'] = gradient.copy() if valid_gradient else np.zeros(len(s.seed), np.float32)
            snapshot['explicit_preload_offsets'] = s.best.numpy()[26:].copy()
            snapshot['explicit_refinement_initial_x'] = self.explicit.initial_x.numpy().copy()
        audit, arrays = self.audit.export()
        audit.update(optimizer_success=success, optimizer_hard_constraints_satisfied=success)
        audit['feasible'] = bool(audit['feasible'] and success)
        if self.explicit is not None:
            audit['predicted_hold_feasible'] = diag['predicted_hold_feasible']
            audit['feasible'] = bool(audit['feasible'] and diag['predicted_hold_feasible'])
        r.device_audit_result = audit, arrays
        return plan, snapshot, qpos


@wp.kernel(enable_backward=False)
def palm_clearance_masked(d: CostData, indices: wp.array(dtype=int)):
    w, slot = wp.tid()
    i = indices[slot]
    body = d.capsule_body[i]
    for k in range(9):
        local = wp.lerp(d.capsule_start[i], d.capsule_end[i], float(k)/8.)
        point = world_point(d, w, body, local)
        p = wp.transpose(d.rotation[w, d.palm_body])@(point-d.position[w, d.palm_body])
        hit = wp.mesh_query_point(d.palm_mesh, p, 1.e6)
        gap = float(-wp.inf)
        if hit.result:
            nearest = wp.mesh_eval_position(d.palm_mesh, hit.face, hit.u, hit.v)
            gap = (hit.sign*wp.length(p-nearest)-d.capsule_radius[i])*1000.
        wp.atomic_min(d.palm_gap, w, gap)
