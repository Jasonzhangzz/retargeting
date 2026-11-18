"""One float64 Laplacian over skeleton and Moxian material nodes, without CEM.

The pinned EMF skeleton graph is extended by anatomical material attachments
and boosted cross-semantic contact edges. One L*V-D residual mixes all edge
types; contact gains are applied after neighborhood normalization. Direction,
thumb, temporal and antireverse losses retain the reference formulas.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np

from third_party.emf_wuji_reference.objectives import FrameObjective, direction_loss, thumb_relative_loss
from third_party.emf_wuji_reference.targets import prepare_targets

CONTACT_GRAPH_ALGORITHM = 'moxian_discrete_contact_graph_slsqp_v8'


@dataclass(frozen=True)
class ContactSLSQPSettings:
    lambda_lap: float = 150.
    lambda_warm: float = 1.
    lambda_pose: float = .1
    lambda_smooth: float = .05
    lambda_thumb_rel: float = .5
    # Negative PIP/DIP bends are valid Wuji2 configurations (for example
    # thumb-to-pinky opposition), so reverse bending is never penalized by
    # the realtime default. The parameter remains available for controlled
    # ablations, but the hard joint limits and collision gate still apply.
    lambda_antirev: float = 0.
    lambda_contact_edge: float = 15000.
    # Direction of an active contacted distal skeleton segment.  This is a
    # separate high-priority term from the all-hand Studio direction prior and
    # does not use tactile surface normals.
    # Distal pose is deliberately weak: glove and Wuji2 distal links have
    # different lengths, so matching its wrist-frame pose creates folded
    # multi-solutions. Contact geometry remains the strong distal signal.
    lambda_contact_distal_direction: float = 0.
    lambda_patch_chamfer: float = 1800.
    lambda_patch_normal: float = 80.
    lambda_contact_edge_length: float = 5000.
    lambda_contact_edge_direction: float = 700.
    # Active tactile edges should stay in zero-gap contact.  Collision
    # recovery enforces the inequality (no penetration); this equality-like
    # term pulls the selected contact witnesses back to that boundary.
    lambda_contact_gap: float = 10000.
    contact_gap_target_m: float = 0.0
    contact_gap_tolerance_m: float = .0005
    lambda_contact_anchor_position: float = 500.
    contact_anchor_position_scale_m: float = .01
    pinky_distal_positive_lower_rad: float = 0.0
    lambda_gesture_prior: float = .02
    lambda_cem_pose_prior: float = .5
    lambda_gesture_orientation: float = .08
    # Configuration-space regularization resolves redundant finger poses.
    # Middle and distal joints are weighted more strongly because they can
    # fold while preserving nearly the same fingertip position.
    lambda_configuration_prior: float = 12.0
    middle_configuration_gain: float = 2.5
    distal_configuration_gain: float = .2
    lambda_fold_direction: float = 3.0
    fold_direction_margin_rad: float = .035
    # PIP/DIP on the little finger have a single anatomical flexion branch in
    # Wuji2.  Keep both joints on the positive flexion side; otherwise a
    # thumb-pinky contact can be satisfied by an inside-out distal fold.
    lambda_pinky_distal_branch: float = 10.0
    pinky_distal_margin_rad: float = .04
    # Fixed Wuji Studio EMF retarget direction term.  The reference uses
    # lambda_warm=1 in stage 1 and lambda_pose=.1 in stage 2; the contact
    # solver keeps the stage independent value below so the wrist-frame bone
    # direction cannot disappear when the contact topology changes.
    lambda_studio_direction: float = 2.5
    distal_skeleton_weight: float = .12
    active_distal_skeleton_weight: float = .04
    distal_direction_gain: float = .05
    lambda_velocity: float = .2
    lambda_acceleration: float = 1.0
    active_tactile_gain: float = 12.
    inactive_patch_weight: float = .001
    penetration_contact_gain: float = 18.
    chamfer_temperature: float = 0.0015
    # CEM's event cost deliberately accepts a small contact gap (2 mm) and
    # scores the contact witness, normal and tangential placement instead of
    # requiring identical mesh points.  The direct point-cloud profile uses
    # the same tolerance-aware robust kernel.  It is kept explicit so a
    # replay can reproduce the exact contact interpretation.
    contact_tolerance_m: float = .002
    contact_witness_weight: float = 8.0
    patch_sample_count: int = 4
    release_antireverse_for_thumb_pinky_distal: bool = True
    # Optional local contact-priority profile.  The production default is
    # neutral (all multipliers are one), so existing realtime behavior is
    # unchanged.  When enabled, rows belonging to active semantic links are
    # strengthened while the rest of the hand keeps an explicit floor.
    active_contact_skeleton_gain: float = 1.
    active_contact_material_gain: float = 1.
    active_contact_edge_gain: float = 1.
    inactive_skeleton_floor: float = 1.
    # Experimental discrete graph selector.  It is intentionally separate
    # from the continuous objective: candidate edges are scored once per
    # frame, then held fixed during SLSQP so the optimizer does not chase a
    # changing nearest-contact topology.
    contact_graph_top_k: int = 3
    contact_graph_hysteresis: float = .35
    contact_graph_min_degree: int = 0
    contact_graph_relative_score: float = .8
    contact_graph_source_distance_max_m: float = .015
    max_joint_velocity_rad_s: float = 2.5
    topology_change_velocity_rad_s: float = 1.2
    first_stage1_maxeval: int = 50
    first_stage2_maxeval: int = 80
    later_stage1_maxeval: int = 1
    later_stage2_maxeval: int = 10
    contact_stage2_maxeval: int = 80
    ftol_abs: float = 1e-4
    self_penetration_mm: float = .5
    object_penetration_mm: float = 3.
    separation_rounds: int = 6
    separation_maxeval: int = 60
    separation_tolerance_mm: float = .01

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value < 0:
                raise ValueError('Invalid SLSQP setting '+name)
            if (name.endswith('maxeval') or name in ('separation_rounds', 'contact_graph_top_k')) and (not isinstance(value, int) or value < 1):
                raise ValueError('Positive integer evaluation budget required')
            if name == 'contact_graph_min_degree' and (not isinstance(value, int) or value < 0):
                raise ValueError('Contact graph minimum degree must be nonnegative')
            if name == 'contact_graph_relative_score' and not 0 < value <= 1:
                raise ValueError('Contact graph relative score must be in (0, 1]')
            if name == 'contact_graph_source_distance_max_m' and value <= 0:
                raise ValueError('Contact graph source distance gate must be positive')
        if self.ftol_abs <= 0: raise ValueError('ftol_abs must be positive')
        if self.max_joint_velocity_rad_s <= 0 or self.topology_change_velocity_rad_s <= 0:
            raise ValueError('Joint velocity limits must be positive')
        if (self.chamfer_temperature <= 0 or self.contact_tolerance_m < 0
                or self.contact_witness_weight < 0 or self.patch_sample_count < 4 or self.inactive_patch_weight < 0):
            raise ValueError('Chamfer temperature and patch sample count must be positive')

    def payload(self):
        return dict(**asdict(self), algorithm=CONTACT_GRAPH_ALGORITHM, position_units='m', angle_units='rad',
                    contact_reduction='skeleton_row_sum_material_row_mean', coefficient_policy='direct_not_squared',
                    wrist_policy='fixed_source_wrist', postprocessing='none',
                    tactile_objective='cem_tolerant_witness_plus_bidirectional_soft_chamfer', normal_tracking=True, absolute_contact_tracking=False,
                    graph_edge_gain='explicit_patch_edge_length_and_direction', graph_normalization='none',
                    skeleton_orientation_policy='weak_gesture_prior_only', temporal_policy='velocity_and_acceleration')


def semantic_joints(link):
    """Fixed anatomical attachment, with indices in the native 21-point hand."""
    if link == 'palm': return (0, 5, 9, 13, 17)
    try:
        finger, segment = link.split('_')
        f = ('thumb', 'index', 'middle', 'ring', 'pinky').index(finger)
        k = {'proximal': 0, 'intermediate': 1, 'distal': 2}[segment]
        if f == 0:
            if segment == 'intermediate': raise ValueError('thumb has no intermediate patch')
            k = 1 if segment == 'proximal' else 2
        return (1+4*f+k, 2+4*f+k)
    except (ValueError, KeyError) as exc:
        raise ValueError('Unknown contact semantic link: '+str(link)) from exc


def distal_skeleton_edges():
    return np.asarray([semantic_joints(link) for link in (
        'thumb_distal', 'index_distal', 'middle_distal', 'ring_distal', 'pinky_distal')], dtype=np.int64)


def distal_skeleton_endpoints():
    return np.unique(distal_skeleton_edges().reshape(-1))


def _softmin_weights(dist2, temperature):
    values = -np.asarray(dist2, dtype=np.float64) / max(float(temperature) ** 2, 1e-16)
    values -= np.max(values, axis=-1, keepdims=True)
    # Keep the true soft-min tail.  The former -60 clip created a finite
    # weight floor whose derivative was zero, so the Chamfer gradient no
    # longer matched its smooth cost when one sample became the nearest
    # witness.  Underflow at -745 is harmless and preserves differentiability.
    weights = np.exp(np.clip(values, -745., 0.))
    return weights / np.maximum(weights.sum(axis=-1, keepdims=True), 1e-16)


def _cem_contact_residual(distance, tolerance, temperature):
    """CEM-style nonnegative contact residual with a soft clearance band.

    CEM's event term is zero while a witness is within its 2 mm contact
    allowance and grows only after that allowance.  ``logaddexp`` gives the
    same hinge a finite derivative at the allowance, which is important for
    SLSQP when the nearest surface witness changes between samples.
    """
    temperature = max(float(temperature), 1e-7)
    z = (np.asarray(distance, dtype=np.float64) - float(tolerance)) / temperature
    # logaddexp remains stable for large positive z; only the sigmoid needs a
    # clipped argument to avoid overflow.  Clipping the residual itself would
    # create a flat cost exactly where the contact is far outside tolerance.
    z_clip = np.clip(z, -60., 60.)
    residual = temperature * np.logaddexp(0., z)
    sigmoid = 1. / (1. + np.exp(-z_clip))
    return residual, sigmoid


class ContactObjective:
    def __init__(self, skeleton, contacts, normals, edges, q_ref, *, contact_links=None, settings=None, fast=False,
                 patch_envelopes=None, patch_samples=None, edge_targets=None, patch_weights=None,
                 previous_q=None, previous2_q=None, patch_provider=None, active_contact_links=None,
                 include_laplacian=False, contact_only=False, configuration_q=None,
                 contact_gap_edges=None, contact_gap_target_m=None,
                 contact_anchor_position_weight=0.0):
        self.settings = settings or ContactSLSQPSettings()
        self.fast=bool(fast)
        self.q_ref = None if q_ref is None else np.asarray(q_ref, dtype=np.float64).copy()
        self.previous_q = None if previous_q is None else np.asarray(previous_q, dtype=np.float64).copy()
        self.previous2_q = None if previous2_q is None else np.asarray(previous2_q, dtype=np.float64).copy()
        self.configuration_q = None if configuration_q is None else np.asarray(configuration_q, dtype=np.float64).copy()
        if self.configuration_q is not None and self.q_ref is not None and self.configuration_q.shape != self.q_ref.shape:
            raise ValueError('configuration_q shape must match q_ref')
        self.contact_gap_edges = np.asarray(contact_gap_edges if contact_gap_edges is not None else [], dtype=np.int64).reshape(-1, 2)
        self.contact_gap_target_m = float(self.settings.contact_gap_target_m if contact_gap_target_m is None else contact_gap_target_m)
        self.contact_anchor_position_weight = float(contact_anchor_position_weight)
        self.patch_provider = patch_provider
        self.active_contact_links = set(str(x) for x in (active_contact_links or ()))
        self.include_laplacian = bool(include_laplacian)
        self.contact_only = bool(contact_only)
        self.envelope_mode = patch_envelopes is not None
        self.patch_envelopes = list(patch_envelopes or [])
        self.patch_samples = list(patch_samples or [])
        self.edge_targets = list(edge_targets or [])
        self.patch_weights = np.asarray(patch_weights if patch_weights is not None else [], dtype=np.float64)
        if self.envelope_mode:
            if len(self.patch_envelopes) != len(self.patch_samples):
                raise ValueError('Patch envelope/sample count mismatch')
            if len(self.patch_weights) != len(self.patch_envelopes):
                self.patch_weights = np.ones(len(self.patch_envelopes), dtype=np.float64)
            if not np.isfinite(self.patch_weights).all() or np.any(self.patch_weights <= 0):
                raise ValueError('Patch weights must be finite and positive')
            self.contacts = np.asarray(contacts, dtype=np.float64).reshape(-1, 3)
            self.normals = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
            self.edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
            self.contact_links = list(contact_links or [])
            self.source_nodes = np.asarray(skeleton, dtype=np.float64).reshape(-1, 3)
            studio_targets = prepare_targets(self.source_nodes, thumb_free=True, thumb_rel_skip_proximal=2)
            self.studio_direction_edges = np.asarray(studio_targets['edges'], dtype=np.int64)
            self.studio_directions = np.asarray(studio_targets['directions'], dtype=np.float64)
            distal_edges = distal_skeleton_edges()
            self.studio_distal_mask = np.asarray([any(np.array_equal(edge, d) for d in distal_edges)
                                                  for edge in self.studio_direction_edges], dtype=bool)
            if self.include_laplacian:
                params = {**asdict(self.settings), 'thumb_rel_full_rad': .35,
                          'thumb_rel_off_rad': .5, 'edge_eps': 1e-8}
                antirev_indices = [i for i in range(20) if i % 4 in (2, 3)]
                base_mask = ~self.studio_distal_mask
                base_weights = np.asarray(studio_targets['row_weights'], dtype=np.float64).copy()
                base_weights[distal_skeleton_endpoints()] *= float(self.settings.distal_skeleton_weight)
                self.base = FrameObjective(self.source_nodes, self.studio_direction_edges[base_mask],
                    directions=self.studio_directions[base_mask], laplacian=studio_targets['L'],
                    differential=studio_targets['D'], weights=base_weights,
                    thumb_triplets=studio_targets['thumb_triplets'], params=params,
                    q_ref=self.q_ref, antirev_indices=antirev_indices)
            self.row_weights = np.ones(len(self.source_nodes), dtype=np.float64)
            self.laplacian = np.zeros((len(self.source_nodes), len(self.source_nodes)), dtype=np.float64)
            self.differential = np.zeros_like(self.source_nodes)
            self.adjacency = np.zeros_like(self.laplacian)
            self.scale = 1.
            self.attachments = []
            return
        targets = prepare_targets(skeleton, thumb_free=True, thumb_rel_skip_proximal=2)
        distal_edges = distal_skeleton_edges()
        direction_edges = np.asarray(targets['edges'], dtype=np.int64)
        direction_mask = np.asarray([any(np.array_equal(edge, d) for d in distal_edges)
                                     for edge in direction_edges], dtype=bool)
        self.studio_direction_edges = direction_edges
        self.studio_directions = np.asarray(targets['directions'], dtype=np.float64)
        self.studio_distal_mask = direction_mask
        self.contacts = np.asarray(contacts, float).reshape(-1, 3)
        self.normals = np.asarray(normals, float).reshape(-1, 3)
        self.edges = np.asarray(edges, int).reshape(-1, 2)
        if (self.contacts.shape != self.normals.shape or not np.isfinite(self.contacts).all()
                or not np.isfinite(self.normals).all()
                or (len(self.normals) and not np.allclose(np.linalg.norm(self.normals, axis=1), 1, atol=1e-6))
                or (self.edges.size and (self.edges.min() < 0 or self.edges.max() >= len(self.contacts)))):
            raise ValueError('Invalid patch graph targets')
        self.graph_enabled = bool(len(self.edges) and self.settings.lambda_contact_edge > 0)
        self.contact_links = list(contact_links or [])
        if self.graph_enabled and len(self.contact_links) != len(self.contacts):
            raise ValueError('Semantic link required for every contact graph node')
        # One adjacency and one Laplacian for [21 skeleton nodes, material nodes].
        # Retain the reference directed skeleton neighborhood exactly. Normalize
        # the contact neighborhood BEFORE boosting; renormalizing afterwards
        # would cancel a uniform increase of contact-edge weights.
        self.scale = self.settings.lambda_lap or self.settings.lambda_contact_edge or 1.
        skeletal_gain = np.sqrt(self.settings.lambda_lap/self.scale)
        contact_gain = np.sqrt(self.settings.lambda_contact_edge/self.scale)
        count = 21+len(self.contacts) if self.graph_enabled else 21
        self.adjacency = np.zeros((count, count))
        self.adjacency[:21, :21] = skeletal_gain*(np.eye(21)-targets['L'])
        self.attachments = []
        self.row_weights = targets['row_weights'].copy()
        self.row_weights[distal_skeleton_endpoints()] *= float(self.settings.distal_skeleton_weight)
        for link in self.contact_links:
            if str(link).endswith('_distal'):
                try:
                    _, endpoint = semantic_joints(link)
                    self.row_weights[endpoint] *= float(self.settings.active_distal_skeleton_weight / max(self.settings.distal_skeleton_weight, 1e-12))
                except ValueError:
                    pass
        self.source_nodes = targets['S']
        if self.graph_enabled:
            self.source_nodes = np.concatenate((targets['S'], self.contacts))
            self.row_weights = np.r_[self.row_weights, np.full(len(self.contacts), 1/len(self.contacts))]
            active_joints = set()
            active_links = set(self.contact_links)
            for link in active_links:
                active_joints.update(semantic_joints(link))
            # Keep the native thumb-free rows free even in a contact profile;
            # all other active finger rows receive the local gain.  Nonactive
            # rows retain a floor rather than disappearing from the graph.
            if active_links:
                floor = float(self.settings.inactive_skeleton_floor)
                gain = float(self.settings.active_contact_skeleton_gain)
                for j in range(21):
                    if j in active_joints and targets['row_weights'][j] > 0:
                        self.row_weights[j] *= gain
                    elif targets['row_weights'][j] > 0:
                        self.row_weights[j] *= floor
                material_gain = float(self.settings.active_contact_material_gain)
                self.row_weights[21:] *= material_gain
            for i, link in enumerate(self.contact_links):
                joints = semantic_joints(link)
                for j in joints:
                    weight = skeletal_gain/len(joints)
                    if link in active_links:
                        weight *= float(self.settings.active_contact_skeleton_gain)
                    if str(link).endswith('_distal'):
                        weight *= float(self.settings.active_distal_skeleton_weight if link in active_links
                                        else self.settings.distal_skeleton_weight)
                    self.adjacency[j, 21+i] = self.adjacency[21+i, j] = weight
                    self.attachments.append(dict(nodes=[int(j), 21+i], weight=float(weight), kind='material_attachment'))
            neighborhood = np.zeros((len(self.contacts), len(self.contacts)))
            for i, j in self.edges:
                if i == j or self.contact_links[i] == self.contact_links[j]:
                    raise ValueError('Contact edges require different semantic links')
                neighborhood[i, j] = neighborhood[j, i] = 1.
            degree = neighborhood.sum(axis=1)
            edge_gain = contact_gain * float(self.settings.active_contact_edge_gain)
            self.adjacency[21:, 21:] = edge_gain*neighborhood/np.maximum(degree[:, None], 1.)
        self.laplacian = np.diag(self.adjacency.sum(axis=1))-self.adjacency
        if not self.graph_enabled: self.laplacian = skeletal_gain*targets['L']
        self.differential = self.laplacian @ self.source_nodes
        if not self.graph_enabled: self.differential = skeletal_gain*targets['D']
        self.contact_direction_edges = []
        if self.settings.lambda_contact_distal_direction > 0 and self.contact_links:
            seen = set()
            for link in self.contact_links:
                if not str(link).endswith('_distal') or link in seen:
                    continue
                seen.add(link)
                a, b = semantic_joints(link)
                vector = self.source_nodes[b] - self.source_nodes[a]
                length = float(np.linalg.norm(vector))
                if length > 1e-8:
                    self.contact_direction_edges.append((a, b, vector/length))
        self.contact_direction_edges = tuple(self.contact_direction_edges)
        params = {**asdict(self.settings), 'lambda_lap': self.scale,
                  'thumb_rel_full_rad': .35, 'thumb_rel_off_rad': .5, 'edge_eps': 1e-8}
        antirev_indices = [i for i in range(20) if i % 4 in (2, 3)]
        if (self.settings.release_antireverse_for_thumb_pinky_distal and self.graph_enabled
                and any({self.contact_links[i], self.contact_links[j]} == {'thumb_distal', 'pinky_distal'}
                        for i, j in self.edges)):
            # q is grouped as four joints per finger: thumb, index, middle,
            # ring, pinky. Release only the distal bend pair for this explicit
            # opposition contact; all other fingers retain antireverse.
            antirev_indices = [i for i in antirev_indices if i // 4 not in (0, 4)]
        self.base = FrameObjective(self.source_nodes, direction_edges[~direction_mask], directions=targets['directions'][~direction_mask],
            laplacian=self.laplacian, differential=self.differential, weights=self.row_weights,
            thumb_triplets=targets['thumb_triplets'], params=params, q_ref=q_ref,
            antirev_indices=antirev_indices)

    def nodes(self, geometry):
        if self.envelope_mode:
            p, jp = geometry[:2]
            return p, jp
        p, jp, c, _, jc, _ = geometry
        return (np.concatenate((p, c)), np.concatenate((jp, jc))) if self.graph_enabled else (p, jp)

    def residual(self, geometry):
        if self.envelope_mode:
            return self.nodes(geometry)[0] - self.source_nodes
        return self.laplacian @ self.nodes(geometry)[0]-self.differential

    def _sample_geometry(self, q):
        if self.patch_provider is not None:
            return self.patch_provider.patch_geometry(q, self.patch_samples)
        result = []
        for item in self.patch_samples:
            result.append(dict(points=np.asarray(item['points'], float),
                               jacobians=np.asarray(item['jacobians'], float),
                               normals=np.asarray(item.get('normals', np.zeros_like(item['points'])), float),
                               normal_jacobians=np.asarray(item.get('normal_jacobians', np.zeros((len(item['points']), 3, len(q)))), float)))
        return result

    def _configuration_regularizer(self, q):
        """Keep redundant joints on the current glove-consistent branch."""
        s = self.settings
        grad = np.zeros_like(q, dtype=np.float64)
        terms = {'configuration_prior': 0., 'fold_direction': 0., 'pinky_distal_branch': 0.}
        if self.configuration_q is None:
            return 0., grad, terms
        weights = np.ones(len(q), dtype=np.float64)
        indices = np.arange(len(q))
        weights[indices % 4 == 2] *= float(s.middle_configuration_gain)
        weights[indices % 4 == 3] *= float(s.distal_configuration_gain)
        delta = q - self.configuration_q
        if s.lambda_configuration_prior > 0:
            terms['configuration_prior'] = .5 * s.lambda_configuration_prior * float(np.sum(weights * delta * delta))
            grad += s.lambda_configuration_prior * weights * delta
        if s.lambda_fold_direction > 0:
            signs = np.sign(self.configuration_q)
            valid = np.abs(self.configuration_q) >= float(s.fold_direction_margin_rad)
            violation = np.maximum(float(s.fold_direction_margin_rad) - q * signs, 0.)
            violation[~valid] = 0.
            terms['fold_direction'] = .5 * s.lambda_fold_direction * float(np.sum(weights * violation * violation))
            grad -= s.lambda_fold_direction * weights * violation * signs
        if s.lambda_pinky_distal_branch > 0 and len(q) >= 20:
            indices = np.asarray([18, 19], dtype=np.int64)
            violation = np.maximum(float(s.pinky_distal_margin_rad) - q[indices], 0.)
            branch_weights = weights[indices]
            terms['pinky_distal_branch'] = .5 * s.lambda_pinky_distal_branch * float(np.sum(branch_weights * violation * violation))
            grad[indices] -= s.lambda_pinky_distal_branch * branch_weights * violation
        return float(sum(terms.values())), grad, terms

    def _contact_gap_regularizer(self, geometry):
        """Pull active mapped contact witnesses to the nonpenetrating boundary."""
        terms = {'contact_gap': 0.}
        grad = np.zeros(20, dtype=np.float64)
        if not len(self.contact_gap_edges) or self.settings.lambda_contact_gap <= 0:
            return 0., grad, terms
        contacts = np.asarray(geometry[2], dtype=np.float64)
        jacobians = np.asarray(geometry[4], dtype=np.float64)
        for i, j in self.contact_gap_edges:
            if i >= len(contacts) or j >= len(contacts):
                continue
            delta = contacts[j] - contacts[i]
            length = float(np.linalg.norm(delta))
            if length <= 1e-9:
                continue
            residual = length - self.contact_gap_target_m
            terms['contact_gap'] += .5 * self.settings.lambda_contact_gap * residual * residual
            direction = delta / length
            grad += self.settings.lambda_contact_gap * residual * (jacobians[j] - jacobians[i]).T @ direction
        return float(terms['contact_gap']), grad, terms

    def _contact_anchor_regularizer(self, geometry):
        terms = {'contact_anchor_position': 0.}
        grad = np.zeros(20, dtype=np.float64)
        if self.contact_anchor_position_weight <= 0 or not len(self.contacts):
            return 0., grad, terms
        points = np.asarray(geometry[2], dtype=np.float64)
        jacobians = np.asarray(geometry[4], dtype=np.float64)
        scale = max(float(self.settings.contact_anchor_position_scale_m), 1e-4)
        count = min(len(points), len(self.contacts))
        residual = (points[:count] - self.contacts[:count]) / scale
        terms['contact_anchor_position'] = .5 * self.contact_anchor_position_weight * float(np.sum(residual * residual))
        grad += self.contact_anchor_position_weight * np.einsum('nik,ni->k', jacobians[:count], residual) / scale
        return float(terms['contact_anchor_position']), grad, terms

    def _contact_graph_direction_regularizer(self, geometry):
        terms = {'contact_graph_direction': 0.}
        grad = np.zeros(20, dtype=np.float64)
        if self.contact_anchor_position_weight <= 0 or not len(self.contact_gap_edges):
            return 0., grad, terms
        points = np.asarray(geometry[2], dtype=np.float64)
        jacobians = np.asarray(geometry[4], dtype=np.float64)
        valid_edges = []; directions = []
        for i, j in self.contact_gap_edges:
            if i >= len(points) or j >= len(points) or i >= len(self.contacts) or j >= len(self.contacts):
                continue
            source_delta = self.contacts[j] - self.contacts[i]
            length = float(np.linalg.norm(source_delta))
            if length < .001:
                continue
            valid_edges.append((int(i), int(j))); directions.append(source_delta / length)
        if not valid_edges:
            return 0., grad, terms
        c, g = direction_loss(points, jacobians, np.asarray(valid_edges, dtype=np.int64),
                              np.asarray(directions, dtype=np.float64),
                              self.settings.lambda_contact_edge_direction, 1e-8)
        terms['contact_graph_direction'] = float(c)
        grad += g
        return float(c), grad, terms

    def _evaluate_envelope(self, q, geometry, stage):
        s = self.settings
        grad = np.zeros_like(q, dtype=np.float64)
        base_terms = {}
        if self.include_laplacian:
            base_cost, base_grad, base_terms = self.base.evaluate(q, geometry[0], geometry[1], stage)
            grad += base_grad
        terms = dict(base_terms)
        terms.update({'patch_chamfer': 0., 'patch_witness': 0., 'patch_normal': 0., 'penetration_contact': 0.,
                 'contact_edge_length': 0., 'contact_edge_direction': 0., 'contact_gap': 0.,
                 'gesture_prior': 0., 'gesture_orientation': 0., 'studio_direction': 0., 'smooth': 0.,
                 'contact_distal_direction': 0., 'velocity': 0., 'acceleration': 0., 'antireverse': 0.,
                 'configuration_prior': 0., 'fold_direction': 0., 'pinky_distal_branch': 0.})
        samples = self._sample_geometry(q)
        active_threshold = max(float(s.inactive_patch_weight) * 1.5, 1e-8)
        has_active_patch = bool(np.any(self.patch_weights > active_threshold))
        total_weight = (float(np.sum(self.patch_weights[self.patch_weights > active_threshold]))
                        if has_active_patch else 1.)
        centers = []; center_jac = []
        temp = float(s.chamfer_temperature)
        for index, (envelope, robot) in enumerate(zip(self.patch_envelopes, samples)):
            ep = np.asarray(envelope.get('points', envelope.get('source_points')), dtype=np.float64).reshape(-1, 3)
            rp = np.asarray(robot['points'], dtype=np.float64).reshape(-1, 3)
            rj = np.asarray(robot['jacobians'], dtype=np.float64).reshape(len(rp), 3, len(q))
            if not len(ep) or not len(rp):
                centers.append(np.zeros(3)); center_jac.append(np.zeros((3, len(q)))); continue
            scale = max(float(envelope.get('scale', .01)), 1e-4)
            # CEM's event contact term is expressed in metres with a fixed
            # clearance band.  Direct point-cloud patches must use that same
            # metric; dividing by a 3 mm patch extent magnifies ordinary
            # cross-asset shape differences into a dominant 1e5+ objective.
            distance_norm = 1.0 if envelope.get('fallback') in ('direct_point_cloud', 'cem_event_witness') else scale**2
            patch_weight = float(self.patch_weights[index]) / total_weight
            point_weights = np.asarray(envelope.get('point_weights', np.ones(len(ep))), dtype=np.float64).reshape(-1)
            if len(point_weights) != len(ep) or not np.isfinite(point_weights).all() or not np.any(point_weights > 1e-12):
                point_weights = np.ones(len(ep), dtype=np.float64)
            point_weights = np.maximum(point_weights, 0.)
            point_weight_sum = max(float(point_weights.sum()), 1e-12)
            d2_er = ((ep[:, None, :] - rp[None, :, :]) ** 2).sum(-1)
            w_er = _softmin_weights(d2_er, temp)
            d2_re = ((rp[:, None, :] - ep[None, :, :]) ** 2).sum(-1)
            w_re = _softmin_weights(d2_re, temp)
            # The CEM contact event has a tolerance band around its witness;
            # use the same band for point-cloud witnesses.  This preserves a
            # useful contact signal when glove and robot meshes have different
            # link lengths or sample layouts, instead of pulling every robot
            # sample onto a glove taxel.
            d_er = np.sqrt(d2_er + 1e-12)
            rho_er, sig_er = _cem_contact_residual(d_er, s.contact_tolerance_m, temp)
            d_re = np.sqrt(d2_re + 1e-12)
            rho_re, sig_re = _cem_contact_residual(d_re, s.contact_tolerance_m, temp)
            er_cost = float(np.sum(w_er * rho_er**2) / max(len(ep), 1) / distance_norm)
            # Weight the source witness distribution by tactile response, then
            # renormalize each robot sample.  It is a soft correspondence, not
            # a taxel-to-taxel assignment.
            w_re_weighted = w_re * point_weights[None, :]
            w_re_weighted /= np.maximum(w_re_weighted.sum(axis=1, keepdims=True), 1e-12)
            re_cost = float(np.sum(w_re_weighted * rho_re**2) / max(len(rp), 1) / distance_norm)
            # A point inside a convex envelope is still a valid contact witness.
            # Increase its Chamfer weight instead of filtering it out.
            planes = np.asarray(envelope.get('planes', []), dtype=np.float64).reshape(-1, 4)
            penetration = np.zeros(len(rp), dtype=np.float64)
            if len(planes):
                values = rp @ planes[:, :3].T + planes[:, 3]
                penetration = np.maximum(-np.max(values, axis=1), 0.)
            boost = 1. + float(s.penetration_contact_gain) * np.clip(penetration / scale, 0., 1.)
            re_cost_boosted = float(np.mean(boost * np.sum(w_re_weighted * rho_re**2, axis=1)) / distance_norm)
            terms['patch_chamfer'] += patch_weight * (er_cost + re_cost_boosted)
            # Source -> robot gradient and robot -> source gradient.
            for j in range(len(rp)):
                unit_er = (rp[j] - ep) / np.maximum(d_er[:, j, None], 1e-6)
                er_values = rho_er[:, j] ** 2
                er_mean = np.sum(w_er * (rho_er ** 2), axis=1)
                er_pair = (2. * rho_er[:, j] * sig_er[:, j])[:, None] * unit_er
                er_pair -= ((er_values - er_mean) / max(temp**2, 1e-12))[:, None] * (2. * (rp[j] - ep))
                grad_er = np.sum(w_er[:, j, None] * er_pair, axis=0)
                grad_er *= 1. / max(len(ep), 1) / distance_norm
                unit_re = (rp[j] - ep) / np.maximum(d_re[j, :, None], 1e-6)
                re_values = rho_re[j] ** 2
                re_mean = float(np.sum(w_re_weighted[j] * re_values))
                re_pair = (2. * rho_re[j] * sig_re[j])[:, None] * unit_re
                re_pair -= ((re_values - re_mean) / max(temp**2, 1e-12))[:, None] * (2. * (rp[j] - ep))
                grad_re = boost[j] * np.sum(w_re_weighted[j, :, None] * re_pair, axis=0)
                grad_re /= max(len(rp), 1) * distance_norm
                target_grad = grad_er + grad_re
                grad += patch_weight * s.lambda_patch_chamfer * (rj[j].T @ target_grad)
            # CEM represents an event with one stable witness.  A soft witness
            # at the tactile cloud centroid prevents sparse or differently
            # shaped links from collapsing toward a taxel extremity while
            # leaving the cloud Chamfer term responsible for local placement.
            centroid = np.sum(ep * point_weights[:, None], axis=0) / point_weight_sum
            witness_target = np.asarray(envelope.get('witness', centroid), dtype=np.float64).reshape(3)
            d2_witness = ((rp - witness_target[None, :]) ** 2).sum(-1)
            if envelope.get('fallback') in ('direct_point_cloud', 'cem_event_witness') and envelope.get('witness') is not None:
                # The first active robot sample is the mapped MuJoCo contact
                # anchor.  CEM's event cost is evaluated at that witness, so
                # do not let a different patch sample hide a large anchor
                # separation behind a soft minimum.
                witness_weights = np.zeros(len(rp), dtype=np.float64)
                witness_weights[0] = 1.
            else:
                witness_weights = _softmin_weights(d2_witness[None, :], temp)[0]
            witness_distance = np.sqrt(d2_witness + 1e-12)
            witness_rho, witness_sig = _cem_contact_residual(witness_distance, s.contact_tolerance_m, temp)
            witness_cost = float(np.sum(witness_weights * witness_rho**2) / distance_norm)
            terms['patch_witness'] += patch_weight * s.contact_witness_weight * witness_cost
            witness_values = witness_rho**2
            witness_mean = float(np.sum(witness_weights * witness_values))
            for j in range(len(rp)):
                unit_witness = (rp[j] - centroid) / max(float(witness_distance[j]), 1e-6)
                witness_pair = 2. * witness_rho[j] * witness_sig[j] * unit_witness
                witness_pair -= ((witness_values[j] - witness_mean) / max(temp**2, 1e-12)) * (2. * (rp[j] - centroid))
                witness_grad = (s.contact_witness_weight * witness_weights[j] * witness_pair / distance_norm)
                grad += patch_weight * s.lambda_patch_chamfer * (rj[j].T @ witness_grad)
            normals = np.asarray(robot.get('normals', []), dtype=np.float64).reshape(-1, 3)
            nj = np.asarray(robot.get('normal_jacobians', []), dtype=np.float64).reshape(-1, 3, len(q))
            target_normal = np.asarray(envelope.get('normal', [0., 1., 0.]), dtype=np.float64)
            if len(normals) == len(rp) and len(nj) == len(rp):
                normal_error = 1. - np.clip(normals @ target_normal, -1., 1.)
                ncost = float(np.mean(normal_error))
                terms['patch_normal'] += patch_weight * ncost
                for j in range(len(normals)):
                    grad += patch_weight * s.lambda_patch_normal * (-nj[j].T @ target_normal)
            center = rp.mean(0); cj = rj.mean(0)
            centers.append(center); center_jac.append(cj)
            terms['penetration_contact'] += patch_weight * float(np.mean(boost - 1.))
        # The explicit patch terms have separate coefficients; the diagnostic
        # terms above are unscaled so each contribution remains inspectable.
        terms['patch_chamfer'] *= s.lambda_patch_chamfer
        terms['penetration_contact'] *= s.lambda_patch_chamfer
        terms['patch_normal'] *= s.lambda_patch_normal
        if len(centers):
            for edge in self.edge_targets:
                i, j = int(edge['i']), int(edge['j'])
                if i >= len(centers) or j >= len(centers): continue
                delta = centers[j] - centers[i]; length = float(np.linalg.norm(delta))
                if length <= 1e-9: continue
                target_length = float(edge['length']); target_direction = np.asarray(edge['direction'], float)
                dl = length - target_length
                terms['contact_edge_length'] += .5 * dl * dl
                unit = delta / length; dr = unit - target_direction
                terms['contact_edge_direction'] += .5 * float(dr @ dr)
                dlength = delta / length
                direction_grad = (dr - unit * float(unit @ dr)) / length
                gj = center_jac[j]; gi = center_jac[i]
                grad += s.lambda_contact_edge_length * dl * ((gj - gi).T @ dlength)
                grad += s.lambda_contact_edge_direction * ((gj - gi).T @ direction_grad)
        terms['contact_edge_length'] *= s.lambda_contact_edge_length
        terms['contact_edge_direction'] *= s.lambda_contact_edge_direction
        # Keep the complete skeleton graph in the objective.  Contact links get
        # a much larger local multiplier, while every non-contact link retains
        # a weak orientation/position prior so the rest of the hand is not
        # silently removed when a tactile patch appears.
        p, jp = geometry[:2]
        count = min(len(p), len(self.source_nodes))
        if self.contact_only and s.lambda_cem_pose_prior > 0 and count:
            residual = (p[:count] - self.source_nodes[:count]) / .05
            distal_nodes = distal_skeleton_endpoints()
            distal_nodes = distal_nodes[distal_nodes < count]
            if len(distal_nodes):
                residual_weights = np.ones(count, dtype=np.float64)
                residual_weights[distal_nodes] = float(s.distal_skeleton_weight)
            else:
                residual_weights = np.ones(count, dtype=np.float64)
            terms['cem_pose_prior'] = .5 * s.lambda_cem_pose_prior * float(np.sum(residual_weights[:, None] * residual * residual))
            grad += s.lambda_cem_pose_prior * np.einsum('nik,ni->k', jp[:count], residual_weights[:, None] * residual) / .05
        if (not self.include_laplacian) and (not self.contact_only) and s.lambda_gesture_prior > 0 and count:
            residual = (p[:count] - self.source_nodes[:count]) / .05
            node_weights = np.ones(count, dtype=np.float64)
            active_gain = 1. + float(s.active_tactile_gain)
            for link in self.active_contact_links:
                if link == 'palm':
                    node_weights[:] = np.maximum(node_weights, active_gain)
                else:
                    try:
                        node_weights[list(semantic_joints(link))] = active_gain
                    except ValueError:
                        pass
            for node in distal_skeleton_endpoints():
                if node < count:
                    node_weights[node] *= float(s.active_distal_skeleton_weight if any(
                        str(link).endswith('_distal') and node in semantic_joints(link)
                        for link in self.active_contact_links) else s.distal_skeleton_weight)
            terms['gesture_prior'] = .5 * s.lambda_gesture_prior * float(np.sum(node_weights[:, None] * residual * residual))
            grad += s.lambda_gesture_prior * np.einsum('nik,ni->k', jp[:count], node_weights[:, None] * residual) / .05
            if s.lambda_gesture_orientation > 0:
                link_names = ('thumb_proximal', 'thumb_distal',
                              'index_proximal', 'index_intermediate', 'index_distal',
                              'middle_proximal', 'middle_intermediate', 'middle_distal',
                              'ring_proximal', 'ring_intermediate', 'ring_distal',
                              'pinky_proximal', 'pinky_intermediate', 'pinky_distal')
                for link in link_names:
                    a, b = semantic_joints(link)
                    if b >= count or a >= count:
                        continue
                    src_delta = self.source_nodes[b] - self.source_nodes[a]
                    dst_delta = p[b] - p[a]
                    src_len = float(np.linalg.norm(src_delta)); dst_len = float(np.linalg.norm(dst_delta))
                    if src_len <= 1e-9 or dst_len <= 1e-9:
                        continue
                    src_unit = src_delta / src_len; dst_unit = dst_delta / dst_len
                    residual_dir = dst_unit - src_unit
                    multiplier = active_gain if link in self.active_contact_links else 1.
                    if str(link).endswith('_distal'):
                        multiplier *= float(s.distal_direction_gain)
                    coefficient = s.lambda_gesture_orientation * multiplier
                    terms['gesture_orientation'] += .5 * coefficient * float(residual_dir @ residual_dir)
                    projected = (residual_dir - dst_unit * float(dst_unit @ residual_dir)) / dst_len
                    grad += coefficient * (jp[b] - jp[a]).T @ projected
        # This is the Wuji Studio / emf_wuji_retarget wrist-frame direction
        # objective: all non-thumb finger-chain edges are compared as unit
        # vectors.  It is deliberately evaluated in both SLSQP stages with a
        # fixed coefficient, instead of inheriting the contact topology or a
        # per-frame tactile weight.
        if (not self.contact_only) and s.lambda_studio_direction > 0 and hasattr(self, 'studio_direction_edges'):
            normal_mask = ~getattr(self, 'studio_distal_mask', np.zeros(len(self.studio_direction_edges), dtype=bool))
            c, g = direction_loss(p, jp, self.studio_direction_edges[normal_mask],
                                  self.studio_directions[normal_mask], s.lambda_studio_direction, 1e-8)
            if np.any(~normal_mask) and s.distal_direction_gain > 0:
                c_distal, g_distal = direction_loss(p, jp, self.studio_direction_edges[~normal_mask],
                                                    self.studio_directions[~normal_mask],
                                                    s.lambda_studio_direction * s.distal_direction_gain, 1e-8)
                c += c_distal; g += g_distal
            terms['studio_direction'] = float(c)
            grad += g
        if (not self.contact_only) and s.lambda_contact_distal_direction > 0 and self.active_contact_links:
            active_distal = [link for link in self.active_contact_links if str(link).endswith('_distal')]
            if active_distal:
                edges = np.asarray([semantic_joints(link) for link in sorted(active_distal)], dtype=np.int64)
                vectors = self.source_nodes[edges[:, 1]] - self.source_nodes[edges[:, 0]]
                lengths = np.linalg.norm(vectors, axis=1)
                keep = lengths > 1e-8
                if np.any(keep):
                    c, g = direction_loss(p, jp, edges[keep], vectors[keep] / lengths[keep, None],
                                          s.lambda_contact_distal_direction, 1e-8)
                    terms['contact_distal_direction'] = float(c)
                    grad += g
        if (not self.include_laplacian) and (not self.contact_only) and s.lambda_smooth > 0:
            change = q - self.q_ref
            terms['smooth'] = .5 * s.lambda_smooth * float(change @ change)
            grad += s.lambda_smooth * change
        if (not self.include_laplacian) and (not self.contact_only) and s.lambda_antirev > 0:
            reverse = np.minimum(q[np.asarray([i for i in range(min(len(q), 20)) if i % 4 in (2, 3)], dtype=np.int64)], 0.)
            terms['antireverse'] = .5 * s.lambda_antirev * float(reverse @ reverse)
            indices = [i for i in range(min(len(q), 20)) if i % 4 in (2, 3)]
            grad[indices] += s.lambda_antirev * reverse
        if self.previous_q is not None:
            velocity = q - self.previous_q
            terms['velocity'] = .5 * s.lambda_velocity * float(velocity @ velocity)
            grad += s.lambda_velocity * velocity
        if self.previous_q is not None and self.previous2_q is not None:
            acceleration = q - 2. * self.previous_q + self.previous2_q
            terms['acceleration'] = .5 * s.lambda_acceleration * float(acceleration @ acceleration)
            grad += s.lambda_acceleration * acceleration
        config_cost, config_grad, config_terms = self._configuration_regularizer(q)
        grad += config_grad
        terms.update(config_terms)
        gap_cost, gap_grad, gap_terms = self._contact_gap_regularizer(geometry)
        grad += gap_grad
        if self.contact_gap_edges.size:
            terms.update(gap_terms)
        anchor_cost, anchor_grad, anchor_terms = self._contact_anchor_regularizer(geometry)
        grad += anchor_grad
        if self.contact_anchor_position_weight > 0:
            terms.update(anchor_terms)
        graph_direction_cost, graph_direction_grad, graph_direction_terms = self._contact_graph_direction_regularizer(geometry)
        if self.contact_anchor_position_weight > 0:
            grad += graph_direction_grad
            terms.update(graph_direction_terms)
        total = float(sum(terms.values()))
        if not np.isfinite(total) or not np.isfinite(grad).all():
            raise ValueError('Nonfinite envelope contact objective')
        return total, grad, terms

    def evaluate(self, q, geometry, stage=2):
        if self.envelope_mode:
            return self._evaluate_envelope(q, geometry, stage)
        p, jp = self.nodes(geometry)
        if not self.fast:total, grad, terms = self.base.evaluate(q, p, jp, stage)
        else:
            b=self.base;s=self.settings;grad=np.zeros_like(q);terms={}
            if stage==2:
                residual=self.laplacian@p-self.differential;weighted=self.row_weights[:,None]*residual
                terms['laplacian']=float(.5*self.scale*np.sum(weighted*residual))
                grad+=self.scale*np.einsum('nik,ni->k',jp,self.laplacian.T@weighted)
            edges=b.edges;delta=p[edges[:,1]]-p[edges[:,0]];length=np.linalg.norm(delta,axis=1);valid=length>b.edge_eps
            unit=delta[valid]/length[valid,None];residual=unit-b.directions[valid]
            weight=s.lambda_warm if stage==1 else s.lambda_pose
            terms['direction']=float(.5*weight*np.sum(residual**2))
            projected=(residual-unit*np.sum(unit*residual,axis=1)[:,None])/length[valid,None]
            grad+=weight*np.einsum('nik,ni->k',(jp[edges[:,1]]-jp[edges[:,0]])[valid],projected)
            change=q-b.q_ref;terms['smooth']=float(.5*s.lambda_smooth*np.dot(change,change));grad+=s.lambda_smooth*change
            terms['thumb_relative'],g=thumb_relative_loss(p,jp,b.thumb_targets,b.edge_eps);grad+=g
            reverse=np.minimum(q[b.antirev_indices],0.)
            terms['antireverse']=float(.5*s.lambda_antirev*np.dot(reverse,reverse));grad[b.antirev_indices]+=s.lambda_antirev*reverse
            total=float(sum(terms.values()))
        if self.contact_direction_edges and self.settings.lambda_contact_distal_direction > 0:
            edge_array=np.asarray([[a,b] for a,b,_ in self.contact_direction_edges],dtype=np.int64)
            directions=np.asarray([d for _,_,d in self.contact_direction_edges],dtype=np.float64)
            c,g=direction_loss(p,jp,edge_array,directions,
                               self.settings.lambda_contact_distal_direction, self.base.edge_eps)
            total += c; grad += g; terms['contact_distal_direction']=float(c)
        if not self.envelope_mode and self.previous_q is not None:
            velocity = q - self.previous_q
            c = .5 * self.settings.lambda_velocity * float(velocity @ velocity)
            total += c; grad += self.settings.lambda_velocity * velocity; terms['velocity'] = c
        if not self.envelope_mode and self.previous_q is not None and self.previous2_q is not None:
            acceleration = q - 2. * self.previous_q + self.previous2_q
            c = .5 * self.settings.lambda_acceleration * float(acceleration @ acceleration)
            total += c; grad += self.settings.lambda_acceleration * acceleration; terms['acceleration'] = c
        config_cost, config_grad, config_terms = self._configuration_regularizer(q)
        if self.configuration_q is not None:
            total += config_cost; grad += config_grad; terms.update(config_terms)
        gap_cost, gap_grad, gap_terms = self._contact_gap_regularizer(geometry)
        if self.contact_gap_edges.size:
            total += gap_cost; grad += gap_grad; terms.update(gap_terms)
        anchor_cost, anchor_grad, anchor_terms = self._contact_anchor_regularizer(geometry)
        if self.contact_anchor_position_weight > 0:
            total += anchor_cost; grad += anchor_grad; terms.update(anchor_terms)
        graph_direction_cost, graph_direction_grad, graph_direction_terms = self._contact_graph_direction_regularizer(geometry)
        if self.contact_anchor_position_weight > 0:
            total += graph_direction_cost; grad += graph_direction_grad; terms.update(graph_direction_terms)
        if not np.isfinite(total) or not np.isfinite(grad).all():
            raise ValueError('Nonfinite contact objective')
        return total, grad, terms


class JointProjectionObjective:
    """Smallest joint correction satisfying zero signed surface gap.

    Tactile priorities have already selected the frame's pose. Separating
    projection repairs that pose without letting a competing contact cost
    keep the initial point infeasible in the constrained line search.
    """
    def __init__(self, reference):
        self.reference = np.asarray(reference, dtype=np.float64).copy()

    def evaluate(self, q, geometry, stage=2):
        delta = q-self.reference; cost = float(.5*delta @ delta)
        return cost, delta, dict(separation_projection=cost)


class SeparationRestorationObjective:
    """Reduce negative signed gaps even when zero gap is outside this frame's box.

    This is an infeasibility recovery objective, not a relaxation of the final
    collision gate. Witnesses and their analytic derivatives include both
    moving links. The independent dense audit decides whether to retain a step.
    """
    def __init__(self, constraints, reference):
        self.constraints = constraints
        self.reference = np.asarray(reference, dtype=np.float64).copy()

    def evaluate(self, q, geometry, stage=2):
        gap, jac = self.constraints.evaluate(q)
        penetration = np.maximum(-gap, 0.)*1000.
        delta = q-self.reference
        cost = float(.5*np.dot(penetration,penetration)+.00005*np.dot(delta, delta))
        gradient = -1000.*(penetration @ jac)+.0001*delta
        return cost, gradient, dict(penetration_restoration=cost)
