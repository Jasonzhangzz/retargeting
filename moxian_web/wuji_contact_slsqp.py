"""Bounded 20-DoF NLopt SLSQP for Moxian material-patch retargeting.

MuJoCo supplies native FK and analytic geometric Jacobians; no dynamics,
finite differences, CEM objective, population solver or MANO model is used.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
import weakref

import mujoco
import nlopt
import numpy as np
from scipy.spatial.transform import Rotation

from .wuji_contact_cost import (CONTACT_GRAPH_ALGORITHM, ContactObjective, ContactSLSQPSettings,
                               semantic_joints, distal_skeleton_endpoints,
                               JointProjectionObjective, SeparationRestorationObjective)
from .wuji_contact_source import MoxianContactSource, closest_triangle, robot_mapping, unit
from .wuji_surface import WujiSurface


def palm_basis(joints):
    """Right-handed anatomical palm axes, independent of each asset's root axes."""
    lateral = unit(joints[5]-joints[17])
    forward = joints[[5, 9, 13, 17]].mean(0)-joints[0]
    normal = unit(np.cross(lateral, forward))
    forward = np.cross(normal, lateral)
    return np.stack((lateral, forward, normal), axis=1)


class NativeContactKinematics:
    """Local robot-wrist FK. Root coordinates are fixed at zero for derivatives."""
    def __init__(self, surface):
        self.surface = surface; self.model = surface.model; self.data = mujoco.MjData(self.model)
        self.body_ids = np.asarray(surface.body_ids)
        self.names = [self.model.joint(i).name for i in range(6, 26)]
        self.lower = self.model.jnt_range[6:26, 0].copy(); self.upper = self.model.jnt_range[6:26, 1].copy()
        self.sites = [self.model.site('right_'+f+'_tip') for f in ('thumb', 'index', 'middle', 'ring', 'pinky')]
        self.contacts = []; self._last_q = None; self._last_geometry = None
        self._last_patch_q = None; self._last_patch_descriptors = None; self._last_patch_geometry = None
        self.palm_basis = palm_basis(self.forward(np.zeros(20))[0])

    def set_contacts(self, contacts):
        self.contacts = contacts; self._last_q = None; self._last_patch_q = None

    def forward(self, q):
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (20,) or not np.isfinite(q).all(): raise ValueError('20 finite robot angles required')
        if self._last_q is not None and np.array_equal(q, self._last_q): return self._last_geometry
        self.data.qpos[:] = np.r_[np.zeros(6), q]
        mujoco.mj_kinematics(self.model, self.data); mujoco.mj_comPos(self.model, self.data)
        rotations = self.data.xmat.reshape(-1, 3, 3)
        points = [self.data.xpos[self.body_ids[0]].copy()]; bodies = [int(self.body_ids[0])]
        for finger, site in enumerate(self.sites):
            for k in (2, 3, 4):
                body = int(self.body_ids[4*finger+k]); points.append(self.data.xpos[body].copy()); bodies.append(body)
            body = int(site.bodyid[0]); points.append(self.data.site_xpos[site.id].copy()); bodies.append(body)
        contact_points = []; contact_normals = []
        for a in self.contacts:
            body = int(self.body_ids[a['robot_owner']]); r = rotations[body]
            contact_points.append(self.data.xpos[body]+r @ np.asarray(a['robot_local_m']))
            contact_normals.append(r @ np.asarray(a['robot_local_normal']))
        jp = np.empty((21, 3, 20)); jc = np.empty((len(self.contacts), 3, 20)); jn = np.empty_like(jc)
        linear = np.zeros((3, self.model.nv)); angular = np.zeros_like(linear)
        for i, (point, body) in enumerate(zip(points, bodies)):
            mujoco.mj_jac(self.model, self.data, linear, angular, point, body); jp[i] = linear[:, 6:]
        for i, (a, point, normal) in enumerate(zip(self.contacts, contact_points, contact_normals)):
            body = int(self.body_ids[a['robot_owner']])
            mujoco.mj_jac(self.model, self.data, linear, angular, point, body)
            jc[i] = linear[:, 6:]; jn[i] = np.cross(angular[:, 6:].T, normal).T
        result = (np.asarray(points), jp, np.asarray(contact_points).reshape(-1, 3),
                  np.asarray(contact_normals).reshape(-1, 3), jc, jn)
        self._last_q = q.copy(); self._last_geometry = result
        return result

    def patch_geometry(self, q, descriptors):
        """Evaluate fixed local patch samples and analytic FK Jacobians."""
        q = np.asarray(q, dtype=np.float64)
        if (self._last_patch_q is not None and np.array_equal(q, self._last_patch_q)
                and self._last_patch_descriptors is descriptors):
            return self._last_patch_geometry
        self.forward(q)
        rotations = self.data.xmat.reshape(-1, 3, 3)
        linear = np.zeros((3, self.model.nv)); angular = np.zeros_like(linear)
        result = []
        for descriptor in descriptors:
            owner = int(descriptor['owner']); body = int(self.body_ids[owner]); r = rotations[body]
            local_points = np.asarray(descriptor['local_points'], dtype=np.float64).reshape(-1, 3)
            local_normals = np.asarray(descriptor['local_normals'], dtype=np.float64).reshape(-1, 3)
            points = self.data.xpos[body] + local_points @ r.T
            normals = local_normals @ r.T
            jacobians = np.empty((len(points), 3, 20)); normal_jacobians = np.empty_like(jacobians)
            for i, (point, normal) in enumerate(zip(points, normals)):
                mujoco.mj_jac(self.model, self.data, linear, angular, point, body)
                jacobians[i] = linear[:, 6:]
                normal_jacobians[i] = np.cross(angular[:, 6:].T, normal).T
            result.append(dict(points=points, normals=normals, jacobians=jacobians,
                               normal_jacobians=normal_jacobians))
        self._last_patch_q = q.copy(); self._last_patch_descriptors = descriptors; self._last_patch_geometry = result
        return result


class ContactGapConstraints:
    """Upper-bound active contact witness distances during tracking.

    Separation constraints provide the lower bound ``distance >= 0``.  This
    companion constraint provides the upper bound ``distance <= tolerance``
    only for tactile contact pairs, so a penetration repair cannot leave a
    visibly separated pair behind.
    """
    def __init__(self, fk, edges, tolerance_m):
        self.fk = fk
        self.edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        self.tolerance_m = float(max(tolerance_m, 0.))
        self.a = np.zeros(len(self.edges), dtype=np.int64)

    def evaluate(self, q):
        geometry = self.fk.forward(q)
        points = np.asarray(geometry[2], dtype=np.float64)
        jacobians = np.asarray(geometry[4], dtype=np.float64)
        values = np.empty(len(self.edges), dtype=np.float64)
        jac = np.zeros((len(self.edges), len(q)), dtype=np.float64)
        for row, (i, j) in enumerate(self.edges):
            delta = points[j] - points[i]
            distance = float(np.linalg.norm(delta))
            if distance <= 1e-9:
                values[row] = self.tolerance_m
                continue
            direction = delta / distance
            values[row] = self.tolerance_m - distance
            jac[row] = -(jacobians[j] - jacobians[i]).T @ direction
        return values, jac


class CombinedContactConstraints:
    """Concatenate lower-bound separation and upper-bound contact gaps."""
    def __init__(self, separation, contact):
        self.separation = separation
        self.contact = contact
        self.a = np.zeros(len(getattr(separation, 'a', ())) + len(getattr(contact, 'a', ())), dtype=np.int64)

    def evaluate(self, q):
        gaps = []; jacobians = []
        if self.separation is not None:
            gap, jac = self.separation.evaluate(q); gaps.append(gap); jacobians.append(jac)
        if self.contact is not None and len(self.contact.a):
            gap, jac = self.contact.evaluate(q); gaps.append(gap); jacobians.append(jac)
        if not gaps:
            return np.empty(0, dtype=np.float64), np.empty((0, len(q)), dtype=np.float64)
        return np.concatenate(gaps), np.vstack(jacobians)


class ContactSLSQPRetarget:
    def __init__(self, topology, cache=None, device='cpu', settings=None, rebuild=False, object_sdf=None,
                 realtime=False, objective_profile='contact_graph'):
        self.settings = settings or ContactSLSQPSettings(); self.device = str(device)
        self.realtime=bool(realtime)
        if objective_profile != 'contact_graph':
            raise ValueError('Unknown SLSQP objective profile: '+str(objective_profile))
        self.objective_profile = str(objective_profile)
        self.algorithm = CONTACT_GRAPH_ALGORITHM
        self.runtime_settings={**self.settings.payload(), 'algorithm':self.algorithm, 'execution_profile':'realtime_stream' if self.realtime else 'offline_reference',
            'objective_profile':self.objective_profile,
            'cost_implementation':'vectorized_reference' if self.realtime else 'pinned_ordered',
            'frame_budget_ms':25 if self.realtime else None,
            'max_joint_velocity_rad_s':float(self.settings.max_joint_velocity_rad_s) if self.realtime else None,
            'topology_change_velocity_rad_s':float(self.settings.topology_change_velocity_rad_s) if self.realtime else None,
            'slsqp_stage_maxtime_ms':12 if self.realtime else None,'first_frame_budget_ms':50 if self.realtime else None,
            'penetration_recovery':'reserved_restoration_and_certified_continuation_v1' if self.realtime else 'zero_gap_projection',
            'recovery_rounds_per_frame':1 if self.realtime else self.settings.separation_rounds}
        self.runtime_settings.update(tactile_objective='discrete_contact_graph_selection',
                                     graph_selection='warm_start_edge_score_with_hysteresis',
                                     graph_selection_top_k=int(self.settings.contact_graph_top_k),
                                     graph_selection_hysteresis=float(self.settings.contact_graph_hysteresis))
        self.source = MoxianContactSource(topology); self.source_topology = topology
        self.surface = WujiSurface(); self.model = self.surface.model
        self.mapping = robot_mapping(self.source, self.surface, cache, rebuild,normal_limit_deg=60.)
        if not self.mapping['ready']: raise ValueError('Unmapped Moxian SIDs: '+str(self.mapping['failed_sids']))
        self.fk = NativeContactKinematics(self.surface)
        records = self.mapping['records']
        self.owners = np.asarray([r['owner'] for r in records]); self.local = np.asarray([r['local_position_m'] for r in records])
        self.normal = np.asarray([r['local_normal'] for r in records])
        self.valid_sids=np.asarray([r.get('normal_valid',True) for r in records],bool)
        self.source.correspondence_valid_sids=self.valid_sids
        self.pads = {}
        for owner in np.unique(self.owners): self.pads[int(owner)] = self.surface.contact_surface(int(owner))
        self.patch_sample_cache = {}
        # This reusable checker has no dependency on CEM cost or optimization.
        from retarget.baseline.card.collision import WujiMeshCollision
        self.mesh_collision = WujiMeshCollision(self.surface, self.device)
        self.audit_local = self.mesh_collision.samples[True][0].numpy()
        self.audit_owners = self.mesh_collision.samples[True][1].numpy()
        self.external_sdf = object_sdf
        from .wuji_contact_collision import NativeSeparation
        self.separation = NativeSeparation(self.fk, self.mesh_collision, object_sdf)
        self.topology_version = hashlib.sha256(json.dumps([self.mapping['version'], self.runtime_settings,
            object_sdf.version if object_sdf else None], sort_keys=True).encode()).hexdigest()[:24]
        self.reset()

    def reset(self):
        self.previous = None; self.previous2 = None; self.previous_timestamp = None; self.last_output = None
        self.previous_stream_epoch = None
        self.recovery_radius=1.
        self.recovery_stalls=0;self.last_feasible_q=None;self.escape_reference=None
        self.previous_contact_signature=None
        self.previous_contact_graph=None
        self.session_id = uuid.uuid4().hex

    def topology(self):
        return {**self.surface.topology(self.mapping), 'algorithm': self.algorithm, 'solver_backend': 'slsqp',
            'topology_version': self.topology_version, 'settings': self.runtime_settings,
            'source_surface': 'moxian_original_skn', 'object_sdf': self.external_sdf.descriptor() if self.external_sdf else None,
            'retarget_graph': dict(type='unified_weighted_laplacian', skeleton_nodes=21,
                material_attachment='semantic_segment_endpoints_or_palm',
                contact_edge_gain=('discrete_hysteretic_graph_selection'),
                skeleton_orientation='reference_graph',
                temporal_regularization='velocity_and_acceleration'),
            'contact_graph': dict(nodes='one_per_active_sensor_patch', edges=('selected_hysteretic_candidates'),
                                  within_patch_edges=False, within_semantic_link_edges=False, max_nodes=20, max_edges=175,
                                  envelope_policy='local_oriented_box_no_wrist_exclude_rank_lt_2')}

    def _gesture_seed(self, target, start):
        """Take one damped Gauss-Newton step toward the glove gesture."""
        q = np.asarray(start, dtype=np.float64).copy()
        p, jp = self.fk.forward(q)[:2]
        count = min(len(p), len(target))
        if count == 0:
            return q
        residual = p[:count] - np.asarray(target[:count], dtype=np.float64)
        j = jp[:count]
        residual_weights = np.ones(count, dtype=np.float64)
        distal = distal_skeleton_endpoints()
        residual_weights[distal[distal < count]] = float(self.settings.distal_skeleton_weight)
        weighted = residual * np.sqrt(residual_weights[:, None])
        weighted_j = j * np.sqrt(residual_weights[:, None, None])
        h = weighted_j.reshape(-1, 20).T @ weighted_j.reshape(-1, 20) + 1e-5 * np.eye(20)
        try:
            delta = np.linalg.solve(h, -weighted_j.reshape(-1, 20).T @ weighted.reshape(-1))
        except np.linalg.LinAlgError:
            return q
        return np.clip(q + np.clip(delta, -.25, .25), self.fk.lower, self.fk.upper)

    def _select_contact_graph(self, anchors, candidate_edges, source_contacts, robot_contacts):
        """Select a fixed contact graph before the continuous SLSQP solve.

        The old edge profile passed every pair of active patches into one
        Laplacian.  That makes a wrong pair pull two fingers sideways just as
        strongly as a geometrically plausible pair.  This experimental
        profile scores the discrete candidates once from the warm start,
        keeps a small hysteretic set, and then freezes it for both SLSQP
        stages.  The objective therefore remains smooth during a solve.
        """
        raw = np.asarray(candidate_edges, dtype=np.int64).reshape(-1, 2)
        if self.objective_profile != 'contact_graph' or len(raw) == 0:
            return raw, dict(candidate_count=int(len(raw)), selected_count=int(len(raw)),
                             selected_edges=[list(map(int, e)) for e in raw], scores=[])
        source_contacts = np.asarray(source_contacts, dtype=np.float64).reshape(-1, 3)
        robot_contacts = np.asarray(robot_contacts, dtype=np.float64).reshape(-1, 3)
        candidates = []
        previous = set(self.previous_contact_graph or ())
        def segment_priority(link):
            link = str(link)
            if link.endswith('_distal'): return 1.0
            if link.endswith('_intermediate'): return .65
            if link.endswith('_proximal'): return .4
            return .15 if link == 'palm' else .5
        # If several tiles on one finger fire together, retain the most
        # distal/high-response representative.  Otherwise every cross-product
        # edge becomes a valid graph candidate and a proximal tile can win by
        # a few millimetres of warm-start geometry.
        representative = {}
        for index, anchor in enumerate(anchors):
            link = str(anchor.get('semantic_link', ''))
            finger = link.split('_', 1)[0]
            if finger in ('palm', ''): continue
            force = float(max(0., anchor.get('force', 0.)))
            rank = segment_priority(link) * (0.5 + force)
            if finger not in representative or rank > representative[finger][0]:
                representative[finger] = (rank, index)
        for i, j in raw:
            i = int(i); j = int(j)
            if i >= len(anchors) or j >= len(anchors) or i >= len(source_contacts) or j >= len(source_contacts):
                continue
            # Palm patches are retained as unary graph nodes, but a palm
            # patch should not create a false finger-to-finger pinch edge.
            if anchors[i].get('semantic_link') == 'palm' or anchors[j].get('semantic_link') == 'palm':
                continue
            keep_pair = True
            for node in (i, j):
                link = str(anchors[node].get('semantic_link', ''))
                finger = link.split('_', 1)[0]
                if finger in representative and representative[finger][1] != node:
                    if segment_priority(link) < segment_priority(anchors[representative[finger][1]].get('semantic_link')):
                        keep_pair = False
            if not keep_pair:
                continue
            ds = source_contacts[j] - source_contacts[i]
            dr = robot_contacts[j] - robot_contacts[i]
            ls = float(np.linalg.norm(ds)); lr = float(np.linalg.norm(dr))
            if ls <= 1e-8 or lr <= 1e-8:
                continue
            # Simultaneous tactile responses are not sufficient evidence that
            # two links touch each other.  The glove-side representatives must
            # already be within a small neighborhood; otherwise this is likely
            # two independent contacts on an object and must stay unary.
            if ls > float(self.settings.contact_graph_source_distance_max_m):
                continue
            cosine = float(np.clip(np.dot(ds, dr) / (ls * lr), -1., 1.))
            length_error = abs(np.log(max(lr, 1e-5) / max(ls, 1e-5)))
            # Direction is more informative for a pinch than absolute length,
            # since the two hands have different link scales.
            mismatch = .65 * length_error + 1.35 * (1. - cosine)
            force_i = float(max(0., anchors[i].get('force', 0.)))
            force_j = float(max(0., anchors[j].get('force', 0.)))
            force_ratio = min(force_i, force_j) / max(max(force_i, force_j), 1e-8)
            tactile_confidence = .25 + .75 * np.sqrt(np.clip(force_ratio, 0., 1.))
            link_priority = np.sqrt(segment_priority(anchors[i].get('semantic_link'))
                                    * segment_priority(anchors[j].get('semantic_link')))
            key = tuple(sorted((int(anchors[i].get('patch', i)), int(anchors[j].get('patch', j)))))
            score = float(np.exp(-mismatch) * tactile_confidence * (0.35 + 0.65 * link_priority))
            if key in previous:
                score += float(self.settings.contact_graph_hysteresis)
            candidates.append(dict(edge=(i, j), key=key, score=score,
                                   direction_cosine=cosine, length_error=length_error,
                                   mismatch=mismatch))
        if not candidates:
            return np.empty((0, 2), dtype=np.int64), dict(candidate_count=0, selected_count=0,
                             selected_edges=[], scores=[])
        candidates.sort(key=lambda row: (-row['score'], row['key']))
        best_score = float(candidates[0]['score'])
        relative_floor = best_score * float(self.settings.contact_graph_relative_score)
        candidates = [row for row in candidates if row['score'] >= relative_floor]
        top_k = int(self.settings.contact_graph_top_k)
        selected = []
        selected_keys = set()
        # Preserve at least one compatible edge for each active non-palm node
        # whenever the candidate graph permits it.
        minimum_degree = int(self.settings.contact_graph_min_degree)
        for node in range(len(anchors)):
            if anchors[node].get('semantic_link') == 'palm':
                continue
            incident = [row for row in candidates if node in row['edge']]
            for row in incident[:minimum_degree]:
                if row['key'] not in selected_keys:
                    selected.append(row); selected_keys.add(row['key'])
        for row in candidates:
            if len(selected) >= top_k and row['key'] not in selected_keys:
                continue
            if row['key'] not in selected_keys:
                selected.append(row); selected_keys.add(row['key'])
        selected.sort(key=lambda row: (-row['score'], row['key']))
        self.previous_contact_graph = tuple(row['key'] for row in selected)
        diag = dict(candidate_count=int(len(candidates)), selected_count=int(len(selected)),
                    selected_edges=[list(map(int, row['edge'])) for row in selected],
                    scores=[dict(patches=list(row['key']), score=float(row['score']),
                                 direction_cosine=float(row['direction_cosine']),
                                 length_error=float(row['length_error'])) for row in selected])
        return np.asarray([row['edge'] for row in selected], dtype=np.int64).reshape(-1, 2), diag

    def material_contacts(self, source):
        anchors = []
        for anchor in source['anchors']:
            a = dict(anchor); ids = np.asarray(a['sids']); response = source['weights']['response'][ids]
            if response.sum() <= 1e-12: response = np.ones(len(ids))
            owner = max(np.unique(self.owners[ids]), key=lambda o: float(response[self.owners[ids] == o].sum()))
            selected = self.owners[ids] == owner
            centre = np.average(self.local[ids[selected]], axis=0, weights=response[selected])
            desired=unit(np.average(self.normal[ids[selected]],axis=0,weights=response[selected]))
            pad = self.pads[int(owner)]
            eligible=np.flatnonzero(pad['normals']@desired>=np.cos(np.deg2rad(60.)))
            if not len(eligible):raise ValueError('Mapped patch has no normal-compatible native face')
            local,index,bc=closest_triangle(pad['triangles'][eligible],centre);face=int(eligible[index])
            a.update(robot_owner=int(owner), robot_link=self.model.body(self.surface.body_ids[owner]).name,
                robot_local_m=local.tolist(), robot_local_normal=pad['normals'][face].tolist(),
                robot_face_id=int(self.surface.face_offsets[owner]+pad['face_ids'][face]), robot_barycentric=bc.tolist())
            a['projection_normal_error_deg']=float(np.degrees(np.arccos(np.clip(pad['normals'][face]@desired,-1,1))))
            anchors.append(a)
        return anchors

    def _solve(self, objective, start, stage, budget, constraints=None):
        lower=getattr(self,'solve_lower',self.fk.lower);upper=getattr(self,'solve_upper',self.fk.upper)
        start = np.clip(start, lower, upper)
        opt = nlopt.opt(nlopt.LD_SLSQP, 20)
        optimizer=weakref.proxy(opt)
        opt.set_lower_bounds(lower); opt.set_upper_bounds(upper)
        opt.set_ftol_abs(self.settings.ftol_abs); opt.set_maxeval(budget)
        if getattr(self,'realtime',False):
            # Recovery owns a reserved solve slice even when source extraction
            # or dense auditing consumed the nominal tracking-frame budget.
            active_contact = (isinstance(objective, ContactObjective)
                              and (bool(getattr(objective.contact_gap_edges, 'size', 0))
                                   or objective.contact_anchor_position_weight > 0
                                   or (objective.envelope_mode and bool(objective.patch_envelopes))))
            tracking_maxtime = .012 if stage == 2 and active_contact else .008
            opt.set_maxtime(.006 if isinstance(objective,(SeparationRestorationObjective,JointProjectionObjective))
                            else max(.0005,min(tracking_maxtime,self.deadline-time.perf_counter())))
        evaluations = 0; callback_error = None; initial_cost = None; last_cost = None; invalid_trial = False
        best_q=start.copy();best_cost=float('inf')
        def callback(q, gradient):
            nonlocal evaluations, callback_error, initial_cost, last_cost, invalid_trial,best_q,best_cost
            if not np.isfinite(q).all():
                invalid_trial = True; optimizer.force_stop()
                if gradient.size: gradient[:] = 0.
                return 1e30
            try:
                value, jac, _ = objective.evaluate(q, self.fk.forward(q), stage)
                if gradient.size: gradient[:] = jac
                evaluations += 1; last_cost = value
                if initial_cost is None: initial_cost = value
                constraint_ok = True
                if constraints is not None:
                    try:
                        gap, _ = constraints.evaluate(q)
                        constraint_ok = bool(np.all(gap >= -1e-4))
                    except Exception:
                        constraint_ok = False
                if constraint_ok and value < best_cost and np.all(q>=lower-1e-9) and np.all(q<=upper+1e-9):
                    best_q=q.copy();best_cost=value
                return value
            except Exception as exc:
                callback_error = exc; raise
        opt.set_min_objective(callback)
        if constraints is not None:
            def nonpenetration(result, q, gradient):
                nonlocal callback_error, invalid_trial
                if not np.isfinite(q).all():
                    invalid_trial = True; optimizer.force_stop(); result[:] = 1e6
                    if gradient.size: gradient[:] = 0.
                    return
                try:
                    gap, jac = constraints.evaluate(q)
                    result[:] = -1000*gap
                    if gradient.size: gradient[:] = -1000*jac
                except Exception as exc:
                    callback_error = exc; raise
            opt.add_inequality_mconstraint(nonpenetration, np.full(len(constraints.a), 1e-4))
        fallback = False; error = None
        try:
            q = opt.optimize(start.copy()); status = int(opt.last_optimize_result())
        except Exception as exc:
            if callback_error is not None: raise callback_error
            status = int(opt.last_optimize_result()); q = start.copy(); fallback = True; error = type(exc).__name__
        if status <= 0:
            q = best_q.copy() if np.isfinite(best_cost) else start.copy(); fallback = True
        elif np.isfinite(best_cost) and last_cost is not None and best_cost < last_cost - 1e-10:
            # SLSQP status 6 can return its last line-search trial even when a
            # better feasible iterate was already evaluated. Never publish
            # that uphill trial into the temporal warm-start state.
            q = best_q.copy()
        # A limited restoration solve may have already found a useful descent
        # step. Returning its penetrating start again would lock every frame.
        retained_descent=(isinstance(objective,SeparationRestorationObjective)
                          and initial_cost is not None and best_cost<initial_cost-1e-12)
        if retained_descent:q=best_q.copy()
        if constraints is not None and len(constraints.a) and not np.isfinite(best_cost):
            # No iterate satisfied both nonpenetration and the active contact
            # upper bound. Do not publish an uphill infeasible line-search
            # trial as if it had maintained contact.
            q = start.copy(); fallback = True; error = 'NoFeasibleConstraintIterate'
        if invalid_trial: q = start.copy(); fallback = True; error = 'NonfiniteOptimizerTrial'
        if not np.isfinite(q).all(): raise ValueError('Nonfinite SLSQP output')
        return np.clip(q, lower, upper), dict(stage=stage, status=status, maxeval=budget,
            evaluations=evaluations, fallback=fallback, error=error, converged=status in (1, 2, 3, 4),
            initial_cost=initial_cost, last_callback_cost=last_cost,retained_descent=bool(retained_descent))

    def audit(self, rotation, origin, frame):
        import torch
        p = self.fk.data.xpos @ rotation.T+origin
        r = np.einsum('ij,bjk->bik', rotation, self.fk.data.xmat.reshape(-1, 3, 3))
        tp = torch.as_tensor(p[None].astype(np.float32), device=self.device)
        tr = torch.as_tensor(r[None].astype(np.float32), device=self.device)
        depth, valid = self.mesh_collision.query(tp, tr, dense=True)
        report = self.mesh_collision.report(depth[0], valid[0])
        object_depth = 0.
        if self.external_sdf:
            if frame.get('object_transform') is not None:
                transform = np.asarray(frame['object_transform'], float)
                if (transform.shape != (4, 4) or not np.isfinite(transform).all()
                        or not np.allclose(transform[3], [0, 0, 0, 1])
                        or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5)
                        or np.linalg.det(transform[:3, :3]) < .99):
                    raise ValueError('Invalid tracked object transform')
                self.external_sdf.transform = transform
            ids = np.asarray(self.surface.body_ids)[self.audit_owners]
            samples = p[ids]+np.einsum('nij,nj->ni', r[ids], self.audit_local)
            value = self.external_sdf.penetration(torch.as_tensor(samples[None].astype(np.float32), device=self.device))
            object_depth = float(value.max())*1000
        passed = (report['query_valid'] and report['max_penetration_mm'] <= self.settings.self_penetration_mm
                  and object_depth <= self.settings.object_penetration_mm)
        return bool(passed), dict(source='native_mesh_final_audit', dynamics=False, self_mesh=report,
            self_hard_limit_mm=self.settings.self_penetration_mm, object_max_penetration_mm=object_depth,
            object_hard_limit_mm=self.settings.object_penetration_mm, object_hard_gate_enabled=self.external_sdf is not None,
            max_penetration_mm=max(object_depth, report['max_penetration_mm']),
            continuous_intersection_proof=False), p, r

    def step(self, frame, contact_enabled=True):
        began = time.perf_counter(); source = self.source.read(frame, contact_enabled,valid_sids=self.valid_sids,
            build_envelopes=False); stamp = float(frame['timestamp_ms'])
        epoch=frame.get('input_stream_epoch')
        reset_reason=None
        if self.previous_timestamp is not None:
            if stamp<=self.previous_timestamp:reset_reason='timestamp_rollback'
            elif stamp-self.previous_timestamp>250 and (self.realtime or epoch is None):reset_reason='source_gap'
            elif epoch is not None:
                if self.previous_stream_epoch is not None and epoch!=self.previous_stream_epoch:reset_reason='input_stream_interrupted'
        if reset_reason:
            # A stream gap must reset the optimizer/recovery bookkeeping, but
            # restarting the hand at the joint-limit midpoint creates a large
            # visible jump when the first post-gap frame arrives. Preserve the
            # last bounded pose as a continuity seed and let the normal
            # realtime velocity box limit the first resumed step. Rollbacks
            # and stream-epoch changes still use a clean midpoint reset.
            continuity_seed = None
            if reset_reason == 'source_gap':
                # The latest finite raw candidate is the temporal state. A
                # historical feasible pose can be many degrees away after a
                # rejected collision frame and would reintroduce a jump.
                candidate = self.previous if self.previous is not None else self.last_feasible_q
                if candidate is not None and np.isfinite(candidate).all():
                    continuity_seed = np.asarray(candidate, dtype=np.float64).copy()
            self.reset()
            if continuity_seed is not None:
                self.previous = continuity_seed
                self.previous2 = None
                self.previous_timestamp = stamp - 1000./30.
                self.previous_stream_epoch = epoch
                self.last_feasible_q = continuity_seed.copy()
                reset_reason = 'source_gap_continuity_seed'
        anchors = self.material_contacts(source); self.fk.set_contacts(anchors)
        contact_signature = tuple(sorted((str(a.get('patch')), str(a.get('semantic_link'))) for a in anchors))
        topology_changed = (self.previous_contact_signature is not None
                            and contact_signature != self.previous_contact_signature)
        # The local native Hand2.2 mesh and SDK reference URDF use different
        # wrist axes. Align their anatomical palm bases, NOT SDK/MANO constants.
        # The root is observed here and never an optimization variable.
        rotation = palm_basis(source['joints']) @ self.fk.palm_basis.T
        origin = source['joints'][0]
        skeleton = (source['joints']-origin) @ rotation
        contacts = (np.array([a['position_m'] for a in anchors]).reshape(-1, 3)-origin) @ rotation
        normals = np.array([a['direction'] for a in anchors]).reshape(-1, 3) @ rotation
        first = self.previous is None
        warm_stamp=self.previous_timestamp if not first else None
        start = (self.fk.lower+self.fk.upper)*.5 if first else self.previous.copy()
        self.solve_lower=self.fk.lower.copy();self.solve_upper=self.fk.upper.copy()
        # Wuji2's little-finger PIP/DIP axes have a single useful flexion
        # branch for retargeting.  Keep the alternative negative branch
        # out of the discrete graph solver; otherwise a contact edge can
        # be satisfied by folding the distal link through the palm.
        pinky = np.asarray([18, 19], dtype=np.int64)
        self.solve_lower[pinky] = np.maximum(self.solve_lower[pinky], self.settings.pinky_distal_positive_lower_rad)
        if self.realtime:
            self.deadline=began+(.050 if first else .025)
            if not first:
                dt=min(max((stamp-warm_stamp)/1000,1/120),1/30)
                velocity_limit = (self.settings.topology_change_velocity_rad_s
                                  if topology_changed or reset_reason == 'source_gap_continuity_seed'
                                  else self.settings.max_joint_velocity_rad_s)
                step_limit=velocity_limit*dt
                self.solve_lower=np.maximum(self.solve_lower,start-step_limit);self.solve_upper=np.minimum(self.solve_upper,start+step_limit)
        # Compute a bounded current-frame gesture configuration on every
        # frame.  The previous solution remains the warm start, while this
        # one-step seed tells the objective which redundant joint branch is
        # consistent with the glove skeleton.
        gesture_reference = self._gesture_seed(skeleton, start)
        if first:
            start = gesture_reference
        # Select and freeze a discrete contact graph before the continuous
        # objective is constructed.  Nearest-contact topology therefore cannot
        # switch inside an SLSQP line search.
        selected_graph_edges, graph_selection = self._select_contact_graph(
            anchors, source.get('edges', []), contacts, self.fk.forward(start)[2])
        contact_gap_edges = np.asarray([
            (int(i), int(j)) for i, j in selected_graph_edges
            if int(i) < len(anchors) and int(j) < len(anchors)
            and anchors[int(i)].get('semantic_link') != 'palm'
            and anchors[int(j)].get('semantic_link') != 'palm'
        ], dtype=np.int64).reshape(-1, 2)
        objective = ContactObjective(skeleton, contacts, normals,
            selected_graph_edges, start.copy(),
            contact_links=[a['semantic_link'] for a in anchors], settings=self.settings, fast=self.realtime,
            previous_q=self.previous, previous2_q=self.previous2,
            configuration_q=gesture_reference, contact_gap_edges=contact_gap_edges,
            contact_anchor_position_weight=self.settings.lambda_contact_anchor_position)
        envelopes = []; patch_samples = []; patch_weights = []; excluded_envelopes = []
        constraints=self.separation.refresh(start,rotation,origin,per_pair=1) if self.realtime else None
        initial_constraint_count=0 if constraints is None else len(constraints.a)
        contact_constraints = (ContactGapConstraints(self.fk, contact_gap_edges, self.settings.contact_gap_tolerance_m)
                               if len(contact_gap_edges) else None)
        tracking_constraints = (CombinedContactConstraints(constraints, contact_constraints)
                                if contact_constraints is not None else constraints)
        restoring=bool(self.realtime and constraints is not None and
                       np.min(constraints.evaluate(start)[0]) < -self.settings.separation_tolerance_mm/1000)
        prepared = time.perf_counter(); s = self.settings
        if restoring:
            # Give an infeasible warm start a restoration frame. A strict
            # zero-gap constraint may have no solution inside the motion box;
            # tracking must not pull back against incremental separation.
            q1=start.copy();q=start.copy()
            stage1=dict(stage=1,skipped=True,reason='penetration_restoration',fallback=False,converged=False)
            stage2=dict(stage=2,skipped=True,reason='penetration_restoration',fallback=False,converged=False)
        else:
            q1, stage1 = self._solve(objective, start, 1, s.first_stage1_maxeval if first else s.later_stage1_maxeval)
            graph_edge_count = len(selected_graph_edges)
            stage2_budget = s.first_stage2_maxeval if first else s.contact_stage2_maxeval if graph_edge_count else s.later_stage2_maxeval
            q, stage2 = self._solve(objective, q1, 2, stage2_budget, tracking_constraints)
            if (tracking_constraints is not constraints and stage2.get('error') == 'NoFeasibleConstraintIterate'):
                # The hard contact upper bound can be temporarily infeasible
                # when a different-length link enters contact inside the
                # realtime velocity box.  Keep the soft zero-gap objective
                # active and relax only this upper bound for this frame;
                # nonpenetration remains a hard constraint.
                q, relaxed = self._solve(objective, q1, 2, stage2_budget, constraints)
                relaxed['contact_constraint_relaxed'] = True
                stage2 = relaxed
        geometry = self.fk.forward(q); _, _, terms = objective.evaluate(q, geometry)
        solved = time.perf_counter(); collision_ok, collision, p, r = self.audit(rotation, origin, frame)
        repair = []; used_current_frame_restart = False
        for iteration in range(1 if self.realtime else s.separation_rounds):
            if collision['max_penetration_mm'] <= s.separation_tolerance_mm: break
            if self.realtime and not restoring and time.perf_counter()>=self.deadline:break
            # One deepest witness per directed link pair avoids contradictory
            # tangent planes inside curved intersecting meshes. Refreshing the
            # dense witnesses resolves the remaining surface sequentially.
            if not (restoring and iteration==0):constraints = self.separation.refresh(q, rotation, origin, per_pair=1)
            if constraints is None: break
            initial_penetration = collision['max_penetration_mm']
            if self.realtime:
                recovery=SeparationRestorationObjective(constraints,q)
                saved_lower,saved_upper=self.solve_lower,self.solve_upper
                recovery_velocity = (self.settings.topology_change_velocity_rad_s
                                     if topology_changed else self.settings.max_joint_velocity_rad_s)
                radius=recovery_velocity/30*self.recovery_radius
                # Keep recovery inside the same frame-level velocity box as
                # tracking. Centering this box on the already moved q would
                # add a second trust step and produce visible jumps when a
                # contact topology change also causes penetration.
                self.solve_lower=np.maximum(saved_lower,start-radius)
                self.solve_upper=np.minimum(saved_upper,start+radius)
                try:repaired,trace=self._solve(recovery,q,2,s.separation_maxeval)
                finally:self.solve_lower,self.solve_upper=saved_lower,saved_upper
                trace['trust_radius_rad']=radius
            else:
                repaired, trace = self._solve(JointProjectionObjective(q), q, 2, s.separation_maxeval, constraints)
            self.fk.forward(repaired)
            repaired_ok, repaired_collision, repaired_p, repaired_r = self.audit(rotation, origin, frame)
            backtracks = 0
            if repaired_collision['max_penetration_mm'] >= initial_penetration-1e-5:
                # A fixed witness can cross another curved facet. Find a
                # smaller independently audited step before refreshing it.
                direction = repaired-q
                for scale in ((.5,.25) if self.realtime else (.5, .25, .125, .0625)):
                    if np.linalg.norm(direction)*scale < 1e-8: break
                    trial = q+scale*direction; self.fk.forward(trial)
                    ok, report, pp, rr = self.audit(rotation, origin, frame); backtracks += 1
                    if report['max_penetration_mm'] < repaired_collision['max_penetration_mm']:
                        repaired = trial; repaired_ok = ok; repaired_collision = report; repaired_p = pp; repaired_r = rr
                    if self.realtime and report['max_penetration_mm'] < initial_penetration-1e-5:break
            # Local recovery must improve the independent dense audit. A
            # certified escape continuation below also measures progress
            # toward its fixed clear endpoint, retaining infeasible status.
            accepted = repaired_collision['max_penetration_mm'] < initial_penetration-1e-5
            if self.realtime:
                self.recovery_stalls=0 if accepted else self.recovery_stalls+1
                if not accepted and (self.escape_reference is not None or self.recovery_stalls>=2):
                    # Deeply intersecting curved meshes can trap a local
                    # witness projection. Optimize a bounded step toward an
                    # independently re-audited clear pose, then audit the step
                    # itself. Never substitute a historical output wholesale.
                    reference=self.escape_reference if self.escape_reference is not None else self.last_feasible_q
                    if reference is None:reference=np.clip(np.zeros(20),self.fk.lower,self.fk.upper)
                    self.fk.forward(reference)
                    reference_ok,reference_collision,_,_=self.audit(rotation,origin,frame)
                    guide=dict(reference='certified_continuation' if self.escape_reference is not None else 'last_feasible' if self.last_feasible_q is not None else 'native_open',
                               certified=bool(reference_ok and reference_collision['max_penetration_mm']<=s.separation_tolerance_mm))
                    if guide['certified']:
                        delta=reference-q;alpha=1.
                        positive=delta>1e-12;negative=delta < -1e-12
                        if np.any(positive):alpha=min(alpha,float(np.min((self.solve_upper[positive]-q[positive])/delta[positive])))
                        if np.any(negative):alpha=min(alpha,float(np.min((self.solve_lower[negative]-q[negative])/delta[negative])))
                        goal=q+max(0.,alpha)*delta
                        guided,guided_trace=self._solve(JointProjectionObjective(goal),q,2,s.separation_maxeval)
                        self.fk.forward(guided)
                        guided_ok,guided_collision,guided_p,guided_r=self.audit(rotation,origin,frame)
                        guide.update(solve=guided_trace,after_mm=guided_collision['max_penetration_mm'])
                        progress=np.linalg.norm(guided-reference)<np.linalg.norm(q-reference)-1e-8
                        guide.update(distance_to_reference_rad=float(np.linalg.norm(guided-reference)),
                            depth_increased=bool(guided_collision['max_penetration_mm']>initial_penetration+1e-5))
                        # A local penetration minimum can require a temporary
                        # increase in depth. Keep the candidate explicitly
                        # infeasible while continuing bounded SLSQP steps to
                        # the fixed, freshly certified clear endpoint.
                        if progress:
                            repaired=guided;repaired_ok=guided_ok;repaired_collision=guided_collision
                            repaired_p=guided_p;repaired_r=guided_r;accepted=True;self.recovery_stalls=0
                            self.escape_reference=reference.copy()
                    else:self.escape_reference=None
                    trace['escape_guide']=guide
            if self.realtime:self.recovery_radius=min(1.,self.recovery_radius*1.5) if accepted else max(.0005,self.recovery_radius*.25)
            if not self.realtime and not accepted and not used_current_frame_restart and np.linalg.norm(q1-q) > 1e-6:
                # Restart the same projection from this frame's direction
                # solve to escape a deeply intersecting local branch. This is
                # a new bounded solve, never a historical output substitution.
                used_current_frame_restart = True
                alternate, restart = self._solve(JointProjectionObjective(q), q1, 2, s.separation_maxeval, constraints)
                self.fk.forward(alternate)
                alt_ok, alt_collision, alt_p, alt_r = self.audit(rotation, origin, frame)
                restart['after_mm'] = alt_collision['max_penetration_mm']; trace['current_frame_restart'] = restart
                if not restart['fallback'] and alt_collision['max_penetration_mm'] < initial_penetration-1e-5:
                    repaired = alternate; repaired_ok = alt_ok; repaired_collision = alt_collision
                    repaired_p = alt_p; repaired_r = alt_r; accepted = True
            trace.update(constraints=len(constraints.a), objective='negative_gap_restoration' if self.realtime else 'minimum_joint_correction', backtracks=backtracks, accepted=accepted, before_mm=initial_penetration,
                         after_mm=repaired_collision['max_penetration_mm'])
            repair.append(trace)
            if accepted:
                q = repaired; collision_ok = repaired_ok; collision = repaired_collision; p = repaired_p; r = repaired_r
            else:
                self.fk.forward(q)
                break
        geometry = self.fk.forward(q); _, _, terms = objective.evaluate(q, geometry)
        audited = time.perf_counter()
        separated = collision['max_penetration_mm'] <= s.separation_tolerance_mm
        collision.update(separation_target_mm=0., separation_tolerance_mm=s.separation_tolerance_mm,
                         separation_satisfied=bool(separated))
        feasible = bool(collision_ok and separated and not stage1['fallback'] and not stage2['fallback'])
        if feasible:self.last_feasible_q=q.copy()
        if separated:self.escape_reference=None;self.recovery_stalls=0;self.recovery_radius=1.
        # A finite bounded raw candidate is the next initial guess even if the
        # final collision gate rejects it. Feasibility is reported separately;
        # rejecting contact never sends the next frame back to the midrange.
        self.previous2 = None if self.previous is None else self.previous.copy()
        self.previous = q.copy()
        # Timestamp records every consumed frame, even if its candidate fails.
        self.previous_timestamp = stamp
        self.previous_stream_epoch = epoch
        self.previous_contact_signature = contact_signature
        body_p = p[self.surface.body_ids]; body_r = r[self.surface.body_ids]
        robot_taxels = body_p[self.owners]+np.einsum('nij,nj->ni', body_r[self.owners], self.local)
        robot_normals = np.einsum('nij,nj->ni', body_r[self.owners], self.normal)
        world_contacts = geometry[2] @ rotation.T+origin; world_normals = geometry[3] @ rotation.T
        errors = []; angles = []; finger_errors = []
        for i, a in enumerate(anchors):
            error = float(np.linalg.norm(world_contacts[i]-a['position_m'])*1000)
            angle = float(np.degrees(np.arccos(np.clip(world_normals[i] @ a['direction'], -1, 1))))
            a.update(robot_position_m=world_contacts[i].tolist(), robot_normal=world_normals[i].tolist(),
                     error_mm=error, normal_error_deg=angle)
            errors.append(error); angles.append(angle)
            if a.get('semantic_link') != 'palm':
                finger_errors.append(error)
        distal_angles = []
        for anchor in anchors:
            link = str(anchor.get('semantic_link', ''))
            if not link.endswith('_distal'):
                continue
            try:
                a_idx, b_idx = semantic_joints(link)
                source_vec = skeleton[b_idx] - skeleton[a_idx]
                robot_vec = geometry[0][b_idx] - geometry[0][a_idx]
                sn = np.linalg.norm(source_vec); rn = np.linalg.norm(robot_vec)
                if sn > 1e-9 and rn > 1e-9:
                    distal_angles.append(float(np.degrees(np.arccos(np.clip((source_vec @ robot_vec)/(sn*rn), -1., 1.)))))
            except ValueError:
                pass
        edge_report = []; edge_errors = []
        contact_gap_errors = []
        for i, j in contact_gap_edges:
            if int(i) < len(world_contacts) and int(j) < len(world_contacts):
                contact_gap_errors.append(float(np.linalg.norm(world_contacts[int(j)]-world_contacts[int(i)])*1000.))
        # Keep the old contact-only metric for comparable diagnostics; it is
        # not an additional loss. Optimization uses the single augmented graph.
        contact_adjacency = np.zeros((len(anchors), len(anchors)))
        reported_graph_edges = (selected_graph_edges)
        for i, j in reported_graph_edges: contact_adjacency[i, j] = contact_adjacency[j, i] = 1.
        degree = contact_adjacency.sum(axis=1); active_rows = degree > 0
        contact_laplacian = np.diag(active_rows.astype(float))-contact_adjacency/np.maximum(degree[:, None], 1.)
        laplacian_error = (contact_laplacian @ (geometry[2]-contacts))[active_rows]
        unified_error = objective.residual(geometry)
        unified_rms = float(np.sqrt(np.sum(objective.row_weights[:, None]*unified_error**2)/objective.row_weights.sum())*1000)
        graph_edges = [dict(nodes=[int(i), int(j)], weight=float(objective.adjacency[i, j]),
                           kind='skeleton' if i < 21 and j < 21 else 'contact' if i >= 21 and j >= 21 else 'material_attachment')
                       for i, j in zip(*np.nonzero(objective.adjacency))]
        graph_source = objective.source_nodes @ rotation.T+origin
        graph_robot = objective.nodes(geometry)[0] @ rotation.T+origin
        graph_type = 'unified_weighted_laplacian'
        graph_material_count = len(anchors) if objective.graph_enabled else 0
        graph_rows = objective.row_weights
        envelope_graph_edges = []
        graph_coefficient = objective.scale
        for i, j in reported_graph_edges:
            if (self.objective_profile == 'chamfer'
                    and (anchors[i].get('semantic_link') == 'palm' or anchors[j].get('semantic_link') == 'palm')):
                continue
            source_edge = np.asarray(anchors[j]['position_m'])-anchors[i]['position_m']
            robot_edge = world_contacts[j]-world_contacts[i]
            error = float(np.linalg.norm(robot_edge-source_edge)*1000); edge_errors.append(error)
            edge_report.append(dict(nodes=[int(i), int(j)], patches=[anchors[i]['patch'], anchors[j]['patch']],
                semantic_links=[anchors[i]['semantic_link'], anchors[j]['semantic_link']],
                source_vector_m=source_edge.tolist(), robot_vector_m=robot_edge.tolist(), error_mm=error))
        euler = Rotation.from_matrix(rotation).as_euler('XYZ'); qpos = np.r_[origin, euler, q]
        ready = bool(source['weights']['ready'] and feasible)
        visual=lambda value:value if self.realtime else value.tolist()
        timing = dict(extract_prepare=(prepared-began)*1000, slsqp=(solved-prepared)*1000,
                      final_audit=(audited-solved)*1000, total=(time.perf_counter()-began)*1000)
        output = dict(status='ready' if feasible else 'infeasible', ready=feasible, solution_feasible=feasible, display_available=True,
            timestamp_ms=stamp, source_timestamp_ms=stamp, tactile_timestamp_ms=frame.get('tactile_timestamp_ms'),
            source_model_hash=frame.get('model_hash'), hand_side='right', source_hand_side=self.source_topology['hand_side'],
            source_surface_backend='moxian', coordinate_frame='shared_virtual_world', units='m',
            solver_backend='slsqp', solver_profile=self.objective_profile, algorithm=self.algorithm, settings=self.runtime_settings, solver_session_id=self.session_id,
            topology_version=self.topology_version, mapping_version=self.mapping['version'], mapping_ready=self.mapping['ready'],
            failed_sids=self.mapping['failed_sids'], geometry_version=frame['geometry_version'], source_binding_version=frame['binding_version'],
            postprocessing='none', qpos=qpos.tolist(), raw_qpos=qpos.tolist(), initial_qpos=np.r_[origin, euler, start].tolist(),
            joint_names=self.fk.names, joint_positions_rad=q.tolist(), wrist_position_m=origin.tolist(), wrist_rotation=rotation.tolist(),
            link_positions_m=visual(body_p), link_rotations=visual(body_r), taxel_positions_m=visual(robot_taxels), taxel_normals=visual(robot_normals),
            source_taxel_positions_m=visual(source['taxels']), source_taxel_normals=visual(source['normals']),
            source_vertices_m=visual(source['vertices']), source_joints_m=visual(source['joints']),
            contact_anchors=anchors, contact_edges=edge_report, contact_ready=ready,
            patch_envelopes=[{k:v for k,v in e.items() if k not in ('source_points',)} for e in envelopes],
            excluded_patch_envelopes=excluded_envelopes,
            retarget_graph=dict(type=graph_type, policy=('discrete_hysteretic_contact_graph_laplacian'), skeleton_node_count=21,
                material_node_count=graph_material_count,
                all_material_node_count=(graph_material_count),
                all_source_nodes_m=(graph_source.tolist()),
                all_robot_nodes_m=(graph_robot.tolist()),
                all_row_weights=(graph_rows.tolist()),
                source_nodes_m=graph_source.tolist(), robot_nodes_m=graph_robot.tolist(), edges=graph_edges,
                row_weights=graph_rows.tolist(), coefficient=graph_coefficient,
                laplacian_rmse_mm=unified_rms),
            envelope_graph=None,
            contact_mask=source['weights']['mask'].tolist() if ready else [None]*460,
            observed_contact_mask=source['weights']['observed_mask'].tolist(), self_contact_pairs=[],
            contact_weights=source['weights']['response'].tolist(), collision=collision,
            sdf=self.external_sdf.metadata() if self.external_sdf else dict(type='none', ready=False),
            taxel_projection=dict(policy=self.mapping['policy'], mapped_sids=int(self.valid_sids.sum()),
                normal_rejected_sids=self.mapping.get('normal_rejected_sids',[]),normal_limit_deg=60.,
                position_and_normal=True, physical_registration_verified=False),
            source_contact_geometry=dict(source='Moxian native SKN', material_binding_version=frame['binding_version'],
                personalized_moxian=bool(frame.get('calibrated')), registration=self.source_topology.get('region_binding_source'),
                sensor_mounting_status=(frame.get('sensor_mounting') or {}).get('status', 'unknown')),
            solver=dict(backend='nlopt.LD_SLSQP', stages=[stage1, stage2], separation_recovery=repair, timing_ms=timing,
                execution_mode='penetration_restoration' if restoring else 'tracking',
                escape_continuation_active=self.escape_reference is not None,
                initial_guess=('gesture_seed') if first else 'previous_raw_solution',reset_reason=reset_reason,
                warm_start_source_timestamp_ms=warm_stamp,initial_collision_constraints=initial_constraint_count,
                contact_gap_constraints=int(len(contact_gap_edges)),
                converged=stage1['converged'] and stage2['converged'], feasible=feasible, used_history_fallback=False,
                contact_graph_selection=graph_selection,
                contact_topology_changed=bool(topology_changed),
                failure_reasons=([] if collision_ok else ['collision_audit_failed'])+
                    ([] if separated else ['nonpenetration_constraint_unsatisfied'])+
                    (['optimizer_failed'] if stage1['fallback'] or stage2['fallback'] else [])),
            quality=dict(active_taxels=int(source['weights']['mask'].sum()), active_patches=len(anchors), active_anchors=len(anchors),
                normal_rejected_active_taxels=int((source['weights']['observed_mask'] & ~self.valid_sids).sum()),
                contact_rmse_mm=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
                finger_contact_rmse_mm=float(np.sqrt(np.mean(np.square(finger_errors)))) if finger_errors else None,
                patch_center_rmse_mm=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
                contact_max_error_mm=max(errors) if errors else None, contact_normal_mean_deg=float(np.mean(angles)) if angles else None,
                contact_edge_rmse_mm=float(np.sqrt(np.mean(np.square(edge_errors)))) if edge_errors else None,
                contact_distance_rmse_mm=float(np.sqrt(np.mean(np.square(contact_gap_errors)))) if contact_gap_errors else None,
                contact_distance_max_mm=max(contact_gap_errors) if contact_gap_errors else None,
                contact_laplacian_rmse_mm=float(np.sqrt(np.mean(np.sum(laplacian_error**2, axis=1)))*1000) if len(laplacian_error) else None,
                unified_laplacian_rmse_mm=unified_rms,
                patch_chamfer_rmse_mm=None,
                patch_witness_cost=None,
                patch_normal_error_deg=None,
                contact_edge_length_error_mm=None,
                contact_edge_direction_error_deg=None,
                contact_distal_direction_error_deg=float(np.mean(distal_angles)) if distal_angles else None,
                gesture_prior_cost=float(terms.get('gesture_prior', 0.)), velocity_cost=float(terms.get('velocity', 0.)),
                configuration_prior_cost=float(terms.get('configuration_prior', 0.)),
                fold_direction_cost=float(terms.get('fold_direction', 0.)),
                pinky_distal_branch_cost=float(terms.get('pinky_distal_branch', 0.)),
                contact_gap_cost=float(terms.get('contact_gap', 0.)),
                contact_anchor_position_cost=float(terms.get('contact_anchor_position', 0.)),
                contact_graph_direction_cost=float(terms.get('contact_graph_direction', 0.)),
                gesture_orientation_cost=float(terms.get('gesture_orientation', 0.)),
                studio_direction_cost=float(terms.get('studio_direction', 0.)),
                acceleration_cost=float(terms.get('acceleration', 0.)), active_patch_weights=[float(x) for x in patch_weights],
                contact_graph_candidate_count=int(graph_selection.get('candidate_count', 0)),
                contact_graph_selected_count=int(graph_selection.get('selected_count', 0)),
                contact_graph_selected_edges=graph_selection.get('selected_edges', []),
                envelope_fallbacks=[e.get('fallback') for e in envelopes if e.get('fallback')],
                contact_cost_policy=('discrete_hysteretic_contact_graph_laplacian'),
                cost_terms=terms, processing_ms=timing['total'], source_quality=frame.get('quality'),
                joint_bounds_saturated=int(((q-self.fk.lower < .01) | (self.fk.upper-q < .01)).sum())))
        timing['report'] = (time.perf_counter()-audited)*1000
        timing['total'] = (time.perf_counter()-began)*1000
        output['quality']['processing_ms'] = timing['total']
        self.last_output = output
        return output
