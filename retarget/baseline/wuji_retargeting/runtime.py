"""Calibrated Moxian skeleton -> unchanged original robot retarget session."""
import hashlib
import json
import time
import uuid

import numpy as np
from scipy.spatial.transform import Rotation

from retarget.baseline.wuji_retargeting.reference import REFERENCE_ROOT
from moxian_web.wuji_surface import WujiSurface, mirror_matrix
from retarget.baseline.wuji_retargeting.source import MoxianSkeleton, INPUT_REVISION
from retarget_reference import RetargetSession
from retarget_reference.kinematics import (
    HandKinematics, _quaternion_product, _quaternion_rotate, _sincos,
)

ALGORITHM = 'wuji_retargeting_moxian_skeleton_v2'
DISPLAY_REVISION = 'inverse_original_preprocess_v3'


def link_poses(kinematics, q):
    """Expose all URDF links for rendering, in the reference FK operation order.

    The solver uses its untouched HandKinematics.forward; this helper never
    contributes a value, gradient, seed or filter state to the solver.
    """
    rotations = {kinematics.root: np.array([0., 0., 0., 1.])}
    positions = {kinematics.root: np.zeros(3)}
    for joint in kinematics._ordered:
        parent = rotations[joint['parent']]
        positions[joint['child']] = _quaternion_rotate(parent, joint['xyz']) + positions[joint['parent']]
        rotation = _quaternion_product(parent, joint['quaternion'])
        index = joint['q_index']
        if index is not None:
            sine, cosine = _sincos(float(q[index])*.5)
            rotation = _quaternion_product(rotation, np.r_[joint['axis']*sine, cosine])
        rotations[joint['child']] = rotation
    return positions, rotations


def render_links(kinematics):
    children = {j['name']: j['child'] for j in kinematics.joints}
    return [kinematics.landmark_names[0]] + [children[n] for n in kinematics.joint_names]


class WujiReferenceRetarget:
    def __init__(self, topology, cache=None, device='cpu', rebuild=False, object_sdf=None):
        self.source_topology = topology
        self.side = topology['hand_side']
        self.source = MoxianSkeleton(topology)
        self.robot = RetargetSession.for_hand('WujiHand2', self.side)
        self.kinematics = self.robot.kinematics
        self.surface = WujiSurface()
        self.link_names = render_links(self.kinematics)
        self.mirror = mirror_matrix(topology)
        self.native_mirror = np.diag([-1., 1., 1.]) if self.side == 'left' else np.eye(3)
        self.render_local = self.surface.local.copy()
        if self.side == 'left':
            # Current web assets contain the right mesh. Mirror that mesh at
            # rest into each *original left URDF* link frame. This changes only
            # drawing geometry; left angles/FK keep the upstream left model.
            right = HandKinematics('WujiHand2', 'right')
            rp, rr = link_poses(right, np.zeros(20))
            lp, lr = link_poses(self.kinematics, np.zeros(20))
            for owner, (rn, ln) in enumerate(zip(render_links(right), self.link_names)):
                ids = self.surface.owners == owner
                right_world = self.render_local[ids] @ Rotation.from_quat(rr[rn]).as_matrix().T + rp[rn]
                left_local = (right_world @ self.native_mirror.T-lp[ln]) @ Rotation.from_quat(lr[ln]).as_matrix()
                self.render_local[ids] = left_local @ self.native_mirror.T
        manifest = json.loads((REFERENCE_ROOT/'SOURCE.json').read_text())
        self.reference_sha256 = manifest['tree_sha256']
        self.asset_sha256 = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            self.kinematics.urdf_path,)}
        version = hashlib.sha256(json.dumps([ALGORITHM, DISPLAY_REVISION, self.reference_sha256,
            INPUT_REVISION, topology['geometry_version'], topology['binding_version'], topology.get('model_hash'), self.side], sort_keys=True).encode()).hexdigest()[:24]
        self.mapping = dict(ready=True, failed_sids=[], records=[None]*460, version=version,
            geometry_version=topology['geometry_version'], binding_version=topology['binding_version'],
            mirror_matrix=self.mirror.tolist(), policy='wuji_reference_no_tactile_mapping')
        self.topology_version = version
        self.session_id = uuid.uuid4().hex
        self.external_sdf = None
        self.frames_computed = 0
        self._cached_key = None
        self._cached_q = None

    def reset(self):
        self.robot.reset()
        self.frames_computed = 0
        self._cached_key = None
        self._cached_q = None
        self.session_id = uuid.uuid4().hex

    def topology(self):
        return dict(self.surface.topology(self.mapping), local_vertices_m=self.render_local.tolist(),
            link_names=self.link_names, algorithm=ALGORITHM, solver_backend='wuji-retargeting',
            topology_version=self.topology_version, reference_sha256=self.reference_sha256,
            asset_sha256=self.asset_sha256, native_hand_side=self.side, display_revision=DISPLAY_REVISION,
            input_revision=INPUT_REVISION, source_model_hash=self.source_topology.get('model_hash'),
            source_skeleton_indices=self.source.indices)

    def step(self, frame, contact_enabled=True):
        began = time.perf_counter()
        points, source_rotation, source_joints, source_vertices = self.source.read(frame)
        # Only duplicate GUI snapshots are suppressed. New timestamps, including
        # unchanged poses, advance upstream warm/filter state exactly once.
        key = (frame['timestamp_ms'], points.astype(np.float32).tobytes())
        started = time.perf_counter()
        if key != self._cached_key:
            q = self.robot.step(points)
            self._cached_q = q.copy()
            self._cached_key = key
            self.frames_computed += 1
        else:
            q = self._cached_q.copy()
        solved = time.perf_counter()
        # Everything below this line is output/visualization only. In
        # particular no collision repair, angle clipping or second filter.
        positions, quaternions = link_poses(self.kinematics, q)
        native_p = np.array([positions[n] for n in self.link_names])
        native_r = Rotation.from_quat([quaternions[n] for n in self.link_names]).as_matrix()
        joints = source_joints @ self.mirror.T
        vertices = source_vertices @ self.mirror.T
        origin = points[0] @ self.mirror.T
        rotation = self.mirror @ source_rotation @ self.native_mirror
        body_p = native_p @ self.native_mirror.T @ rotation.T + origin
        body_r = rotation @ self.native_mirror @ native_r @ self.native_mirror
        robot = self.robot
        trace = robot.last_trace
        stages = [{k: v for k, v in stage.items() if k not in ('evaluations', 'output')}
                  | dict(evaluations=len(stage['evaluations'])) for stage in trace['stages']]
        zeros = np.zeros((460, 3)).tolist()
        elapsed = (time.perf_counter()-began)*1000
        return dict(ready=True, display_available=True, status='ready', algorithm=ALGORITHM,
            solver_backend='wuji-retargeting', solver_session_id=self.session_id,
            mapping_version=self.mapping['version'], topology_version=self.topology_version,
            geometry_version=frame['geometry_version'], source_binding_version=frame['binding_version'],
            source_timestamp_ms=frame['timestamp_ms'], timestamp_ms=frame['timestamp_ms'],
            source_display_timestamp_ms=frame['timestamp_ms'],
            source_hand_side=self.side, hand_side='right', coordinate_frame='shared_virtual_world', units='m',
            joint_names=self.kinematics.joint_names, joint_positions_rad=q.tolist(),
            raw_joint_positions_rad=trace['unfiltered'],
            qpos=np.r_[origin, Rotation.from_matrix(rotation).as_euler('XYZ'), q].tolist(),
            wrist_position_m=origin.tolist(), wrist_rotation=rotation.tolist(),
            link_positions_m=body_p.tolist(), link_rotations=body_r.tolist(),
            source_vertices_m=vertices.tolist(), source_joints_m=joints.tolist(),
            source_taxel_positions_m=zeros, taxel_positions_m=zeros,
            source_taxel_normals=zeros, taxel_normals=zeros,
            contact_ready=False, contact_mask=[None]*460, contact_weights=[0.]*460,
            contact_anchors=[], contact_edges=[], self_contact_pairs=[],
            solution_feasible=None, contact_constraints_satisfied=None,
            postprocessing='upstream_float32_lowpass_alpha_0.2',
            input_revision=INPUT_REVISION, source_model_hash=frame.get('model_hash'),
            calibration_fingerprint=frame.get('calibration_fingerprint'),
            source_skeleton_indices=self.source.indices, retarget_input_keypoints_m=points.tolist(),
            reference_sha256=self.reference_sha256, asset_sha256=self.asset_sha256,
            display_revision=DISPLAY_REVISION,
            solver=dict(backend='original_retarget_reference.RetargetSession', compute_device='cpu', stages=stages,
                first=trace['first'], frames_computed=self.frames_computed,
                initial_guess='native_midrange' if trace['first'] else 'previous_unfiltered_float32',
                filter_alpha=float(robot.alpha), input='current_calibrated_moxian_skeleton',
                object_constraints_enabled=False, tactile_objective_enabled=False),
            quality=dict(processing_ms=elapsed, reference_robot_ms=(solved-started)*1000,
                source_quality=frame.get('quality'),
                joint_bounds_saturated=int(((q-self.kinematics.lower<.01)|(self.kinematics.upper-q<.01)).sum()),
                source_display_offset_ms=0.),
            virtual_only=True)
