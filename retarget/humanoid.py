"""G1 retargeting with MJWarp dynamics and a shared hand/foot ADMM block.

SPIDER source provenance and deliberate objective changes are documented in
data/spider/README.md. Actions are actual MuJoCo actuator controls; the floating
base and object are never optimization variables or reset during a rollout.
"""
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import time

import mujoco
import numpy as np


ROLES = ('left_hand', 'right_hand', 'left_foot', 'right_foot')
KEYPOINTS = ('pelvis', 'torso_link',
             'left_shoulder_pitch_link', 'right_shoulder_pitch_link',
             'left_elbow_link', 'right_elbow_link',
             'left_wrist_yaw_link', 'right_wrist_yaw_link',
             'left_hip_pitch_link', 'right_hip_pitch_link',
             'left_knee_link', 'right_knee_link',
             'left_ankle_roll_link', 'right_ankle_roll_link')


@dataclass(frozen=True)
class Weights:
    root_position: float = 10.0
    root_rotation: float = 5.0
    joints: float = 0.3
    robot_velocity: float = 0.002
    keypoint_position: float = 100.0
    keypoint_rotation: float = 0.1
    upright: float = 10.0
    balance: float = 5.0
    foot_slip: float = 0.5
    action_reference: float = 0.01
    action_smoothness: float = 0.05
    penetration: float = 100.0
    contact_wrench: float = 5.0
    hand_position: float = 50.0


@dataclass(frozen=True)
class Settings:
    # Old saved configs omit this field; keep their Python API interpretation.
    # run_humanoid.py explicitly selects the new SPIDER task by default.
    cost_mode: str = 'legacy'
    mjwarp_fd_step: float = .01
    mjwarp_fd_batch: int = 64
    mjwarp_contact_capacity: int = 100
    mjwarp_constraint_capacity: int = 350
    contact_lm_iterations: int = 3
    contact_sweeps: int = 1
    horizon: int = 8
    admm_iterations: int = 1
    action_iterations: int = 2
    rho: float = 1.0
    contact_scale: float = 0.05
    anchor_weight: float = 0.05
    hand_contact_threshold: float = 0.04
    foot_contact_threshold: float = 0.035
    substeps: int = 2
    primal_tolerance: float = 0.002
    dual_tolerance: float = 0.002
    knot_stride: int = 3
    multistarts: int = 8
    multistart_noise_std: float | None = None
    adamw_lr: float = 0.3
    adamw_weight_decay: float = 0.01
    adamw_persist: bool = False
    control_scale: float = 0.05
    contact_softness: float = 100.0
    terminal_weight: float = 2.0
    contact_radius: float = 0.005
    foot_contact_radius: float = 0.15
    contact_samples: int = 8
    contact_qp_iterations: int = 48
    seed: int = 0
    device: str = 'cuda:0'
    balance_xy_scale: float = 0.15
    balance_z_scale: float = 0.08
    balance_velocity_scale: float = 0.5
    balance_height_margin: float = 0.08
    balance_height_scale: float = 0.05
    upright_scale: float = 0.25
    contact_realization_weight: float = 1.0
    wrench_robust_scale: float = 1.0
    trajectory_cache: bool = True
    action_solver: str = 'adamw'
    line_search_trials: int = 6
    line_search_contraction: float = .5
    line_search_armijo: float = 1.e-4
    mppi_samples: int = 64
    mppi_noise_std: float = .05
    mppi_temperature: float = 1.
    mppi_update: str = 'safeguarded'
    mppi_single_rollout: bool = False
    acados_difference: str = 'forward'
    acados_fd_step: float = .01
    acados_trust_radius: float = .15
    acados_damping: float = .01
    acados_globalization: str = 'cauchy'
    acados_initial_iterations: int = 0
    sample_lm_samples: int = 64
    sample_lm_rank: int = 32
    sample_lm_noise_std: float = .05
    sample_lm_fd_step: float = .02
    sample_lm_trust_radius: float = .03
    sample_lm_damping: float = .01
    sample_lm_reuse_samples: bool = False


def rotation_error(current, reference):
    """SO(3) difference in radians, insensitive to quaternion sign."""
    quat_current = np.empty(4)
    quat_reference = np.empty(4)
    mujoco.mju_mat2Quat(quat_current, np.asarray(current).reshape(9))
    mujoco.mju_mat2Quat(quat_reference, np.asarray(reference).reshape(9))
    result = np.empty(3)
    mujoco.mju_subQuat(result, quat_current, quat_reference)
    return result


def project_box(points, halfsize):
    """Euclidean projection onto the box *surface*, including interior points."""
    points = np.asarray(points, dtype=float)
    flat = points.reshape(-1, 3)
    projected = np.clip(flat, -halfsize, halfsize)
    inside = np.all(np.abs(flat) <= halfsize, axis=1)
    rows = np.flatnonzero(inside)
    axes = np.argmin(halfsize - np.abs(flat[rows]), axis=1)
    projected[rows, axes] = np.where(flat[rows, axes] >= 0, 1., -1.) * halfsize[axes]
    return projected.reshape(points.shape)


def project_ground(points, height):
    result = np.array(points, dtype=float, copy=True)
    result[..., 2] = height
    return result



class ObjectSurface:
    """Project in the object's collision-geom frame, without pose tracking."""

    def __init__(self, model, geom_id):
        self.kind = model.geom_type[geom_id]
        self.size = model.geom_size[geom_id].copy()
        self.mesh = None
        if self.kind == mujoco.mjtGeom.mjGEOM_MESH:
            import trimesh
            mesh_id = model.geom_dataid[geom_id]
            va, vn = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
            fa, fn = model.mesh_faceadr[mesh_id], model.mesh_facenum[mesh_id]
            self.mesh = trimesh.Trimesh(model.mesh_vert[va:va+vn],
                                       model.mesh_face[fa:fa+fn], process=False).convex_hull
        elif self.kind not in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_SPHERE):
            raise ValueError('Object collision must be box, sphere or mesh')

    def project(self, points):
        points = np.asarray(points, dtype=float)
        if self.kind == mujoco.mjtGeom.mjGEOM_BOX:
            return project_box(points, self.size)
        if self.kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            norm = np.linalg.norm(points, axis=-1, keepdims=True)
            direction = points / np.maximum(norm, 1e-15)
            direction = np.where(norm < 1e-15, np.array([1., 0., 0.]), direction)
            return direction * self.size[0]
        import trimesh
        shape = points.shape
        return trimesh.proximity.closest_point_naive(self.mesh, points.reshape(-1, 3))[0].reshape(shape)

    def normal(self, points, projected):
        if self.kind == mujoco.mjtGeom.mjGEOM_BOX:
            axis = np.argmin(np.abs(np.abs(projected) - self.size), axis=-1)
            normal = np.zeros_like(projected)
            flat = normal.reshape(-1, 3)
            rows = np.arange(len(flat))
            cols = axis.reshape(-1)
            flat[rows, cols] = np.where(projected.reshape(-1, 3)[rows, cols] >= 0, 1., -1.)
            outside = np.any(np.abs(points) > self.size, axis=-1, keepdims=True)
            difference = points - projected
            exterior = difference / np.maximum(np.linalg.norm(difference, axis=-1, keepdims=True), 1e-15)
            return np.where(outside, exterior, normal)
        if self.kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            return projected / self.size[0]
        import trimesh
        face = trimesh.proximity.closest_point_naive(self.mesh, points.reshape(-1, 3))[2]
        normal = self.mesh.face_normals[face].reshape(points.shape)
        difference = points - projected
        length = np.linalg.norm(difference, axis=-1, keepdims=True)
        outside = (np.sum(difference*normal, axis=-1, keepdims=True) >= 0) & (length > 1e-12)
        return np.where(outside, difference/np.maximum(length, 1e-15), normal)



def _id(model, kind, name):
    index = mujoco.mj_name2id(model, kind, name)
    if index < 0:
        raise ValueError(f'Model is missing {name!r}')
    return index


class HumanoidProblem:
    def __init__(self, scene, reference, *, ref_dt=1/30, settings=Settings(),
                 weights=None, contact_mask=None):
        if not np.isfinite(settings.mjwarp_fd_step) or settings.mjwarp_fd_step<=0:
            raise ValueError('MJWarp finite difference step must be finite and positive')
        if min(settings.mjwarp_fd_batch, settings.mjwarp_contact_capacity, settings.mjwarp_constraint_capacity)<1:
            raise ValueError('MJWarp batch/contact/constraint capacities must be positive')
        if settings.cost_mode not in ('legacy','spider'):
            raise ValueError('Unknown humanoid cost mode')
        if min(settings.contact_lm_iterations,settings.contact_sweeps)<1:
            raise ValueError('Contact LM iterations and sweeps must be positive')
        if settings.cost_mode=='spider':
            settings=replace(settings,terminal_weight=1.)
            if weights is not None and any(asdict(weights).values()):
                raise ValueError('Legacy weights do not configure the SPIDER pose norm')
            weights=Weights(**{name:0. for name in Weights.__dataclass_fields__})
        if not np.isfinite(settings.wrench_robust_scale) or settings.wrench_robust_scale<=0:
            raise ValueError('Wrench robust scale must be finite and positive')
        if settings.multistart_noise_std is not None and (not np.isfinite(settings.multistart_noise_std) or settings.multistart_noise_std<0):
            raise ValueError('Multistart noise must be finite and nonnegative')
        self.scene = Path(scene).resolve()
        self.reference_path = Path(reference).resolve()
        self.settings, self.weights, self.ref_dt = settings, weights, ref_dt
        self.model = model = mujoco.MjModel.from_xml_path(str(self.scene))
        model.opt.timestep = ref_dt / settings.substeps
        # Match spider.simulators.mjwp.setup_mj_model for both humanoid modes.
        model.opt.iterations=5
        model.opt.ls_iterations=10
        model.opt.o_solref[:]=[.02,1.]
        model.opt.o_solimp[:]=[.9,.95,.001,.5,2.]
        # CPU model supplies reference kinematics and visualization only.
        # Optimization and execution both use MJWarp implicitfast.
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        with np.load(reference, allow_pickle=False) as arrays:
            self.qref = np.array(arrays['qpos'], dtype=float)
            self.vref = np.array(arrays['qvel'], dtype=float)
            self.aref = np.array(arrays['ctrl'], dtype=float)
        for key, array, width in [('qpos', self.qref, model.nq),
                                  ('qvel', self.vref, model.nv),
                                  ('ctrl', self.aref, model.nu)]:
            if array.ndim != 2 or array.shape != (len(self.qref), width):
                raise ValueError(f'{key} must have shape (T,{width}), got {array.shape}')
            if not np.isfinite(array).all():
                raise ValueError(f'Nonfinite reference {key}')
        if len(self.qref) < 2:
            raise ValueError('At least two reference frames are required')
        self.tracking_vref=self.vref.copy()
        self.reference_velocity_source='provided'
        # Some released kinematic clips store zero placeholder velocities.
        # Preserve their initial simulator state, but do not interpret those
        # placeholders as a command to stop a moving reference trajectory.
        if not np.any(np.abs(self.vref)>1e-9):
            for i in range(len(self.qref)):
                left,right=max(0,i-1),min(len(self.qref)-1,i+1)
                mujoco.mj_differentiatePos(model,self.tracking_vref[i],(right-left)*ref_dt,self.qref[left],self.qref[right])
            self.reference_velocity_source='manifold_derivative_of_qpos; initial_state_preserves_provided_velocity'
        # This adapter deliberately validates G1 topology rather than silently
        # interpreting a bimanual hand-only scene as a floating humanoid.
        self.root = _id(model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
        self.torso = _id(model, mujoco.mjtObj.mjOBJ_BODY, 'torso_link')
        self.bodies = np.array([_id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in KEYPOINTS])
        self.object_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'largebox_geom')
        self.surface = ObjectSurface(model, self.object_geom) if self.object_geom >= 0 else None
        if settings.multistart_noise_std is None:
            settings=replace(settings,multistart_noise_std=.05 if self.surface else .003)
            self.settings=settings
        if weights is None:self.weights=Weights(joints=3.) if self.surface is None else Weights()
        self.hand_geoms = np.array([_id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ('lh', 'rh')]) if self.surface else np.array([], dtype=int)
        if self.surface and np.any(model.geom_type[self.hand_geoms] != mujoco.mjtGeom.mjGEOM_SPHERE):
            raise ValueError('G1 adapter requires spherical lh/rh collision geoms')
        self.foot_geoms = []
        for short, side in [('l', 'left'), ('r', 'right')]:
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f'{short}f0') >= 0:
                names = [f'{short}f{i}' for i in range(4)]
            else:
                names = [f'{side}_foot{i}_collision' for i in range(1, 8)]
            ids = np.array([_id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in names])
            if any(model.geom_type[i] not in (mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CAPSULE) for i in ids):
                raise ValueError('Foot collision geoms must be spheres or capsules')
            self.foot_geoms.append(ids)
        floor = _id(model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        if model.geom_type[floor] != mujoco.mjtGeom.mjGEOM_PLANE:
            raise ValueError('floor must be a horizontal plane')
        self.floor = floor
        robot_joints = [j for j in range(model.njnt)
                        if self._robot_body(model.jnt_bodyid[j])]
        if model.jnt_type[robot_joints[0]] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError('Expected a free floating humanoid root')
        self.robot_dofs = np.concatenate([
            np.arange(model.jnt_dofadr[j], model.jnt_dofadr[j] +
                      (6 if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else 1))
            for j in robot_joints])
        if np.any(model.jnt_type[robot_joints[1:]] != mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError('G1 adapter expects hinge limb joints')
        self.joint_qpos = np.array([model.jnt_qposadr[j] for j in robot_joints[1:]])
        if np.any(model.actuator_trntype != mujoco.mjtTrn.mjTRN_JOINT):
            raise ValueError('Only real robot joint actuators are supported')
        if any(not self._robot_body(model.jnt_bodyid[j]) for j in model.actuator_trnid[:, 0]):
            raise ValueError('Object/virtual guidance actuators must be removed')
        self.lower = np.where(model.actuator_ctrllimited, model.actuator_ctrlrange[:, 0], -np.inf)
        self.upper = np.where(model.actuator_ctrllimited, model.actuator_ctrlrange[:, 1], np.inf)
        self.aref = np.clip(self.aref, self.lower, self.upper)
        self.data = mujoco.MjData(model)
        self.reference = []
        for q, v in zip(self.qref, self.vref):
            self.data.qpos[:] = q
            mujoco.mj_normalizeQuat(model, self.data.qpos)
            self.data.qvel[:] = v
            mujoco.mj_forward(model, self.data)
            self.reference.append(self.features(self.data))
        self.ground_height = float(self.data.geom_xpos[floor, 2])
        self.contact_friction = []
        role_geoms = (list(self.hand_geoms) if self.surface else [-1,-1])+[int(ids[0]) for ids in self.foot_geoms]
        for role,geom in enumerate(role_geoms):
            target = self.object_geom if role < 2 else floor
            mu = 1.
            for pair in range(model.npair):
                if {int(model.pair_geom1[pair]),int(model.pair_geom2[pair])} == {geom,target}:
                    mu = float(model.pair_friction[pair,0])
                    break
            self.contact_friction.append(mu)
        self.object_floor_friction=1.
        if self.surface:
            for pair in range(model.npair):
                if {int(model.pair_geom1[pair]),int(model.pair_geom2[pair])}=={self.object_geom,floor}:
                    self.object_floor_friction=float(model.pair_friction[pair,0])
                    break
        if not np.allclose(self.data.geom_xmat[floor].reshape(3, 3)[:, 2], [0, 0, 1]):
            raise ValueError('Only horizontal ground is supported')
        anchors, masks = [], []
        for feature in self.reference:
            x = feature['contacts']
            z = x.copy()
            mask = np.zeros(4, dtype=bool)
            if self.surface is not None:
                z[:2] = self.surface.project(x[:2])
                mask[:2] = np.linalg.norm(z[:2] - x[:2], axis=1) < settings.hand_contact_threshold
            z[2:] = project_ground(x[2:], self.ground_height)
            # Negative signed gap is penetration/contact, never flight. The
            # released kinematic AMASS clip starts several cm below the plane.
            mask[2:] = x[2:, 2] - self.ground_height < settings.foot_contact_threshold
            anchors.append(z)
            masks.append(mask)
        self.anchors, self.mask = np.array(anchors), np.array(masks)
        self.stance_anchors = self.anchors.copy()
        for role in (2,3):
            for i in range(1,len(self.mask)):
                if self.mask[i,role] and self.mask[i-1,role]:
                    self.stance_anchors[i,role] = self.stance_anchors[i-1,role]
        # Required external centroidal wrench from the demonstrated whole-body
        # motion. No object position/velocity tracking objective is introduced.
        d = mujoco.MjData(model)
        linear, angular = [], []
        object_linear, object_angular = [], []
        for q, v in zip(self.qref,self.tracking_vref):
            d.qpos[:],d.qvel[:] = q,v
            mujoco.mj_forward(model,d)
            mujoco.mj_subtreeVel(model,d)
            linear.append(d.subtree_linvel[self.root].copy())
            angular.append(d.subtree_angmom[self.root].copy())
            if self.surface:
                body=model.geom_bodyid[self.object_geom]
                object_linear.append(d.subtree_linvel[body].copy())
                object_angular.append(d.subtree_angmom[body].copy())
        mass = model.body_subtreemass[self.root]
        acceleration = np.gradient(np.array(linear),ref_dt,axis=0)
        momentum_rate = np.gradient(np.array(angular),ref_dt,axis=0)
        self.reference_wrench = np.stack((mass*(acceleration-model.opt.gravity),momentum_rate),axis=1)
        self.reference_object_wrench=np.zeros_like(self.reference_wrench)
        if self.surface:
            mass=model.body_subtreemass[model.geom_bodyid[self.object_geom]]
            self.reference_object_wrench=np.stack((mass*(np.gradient(object_linear,ref_dt,axis=0)-model.opt.gravity),
                np.gradient(object_angular,ref_dt,axis=0)),axis=1)
        if contact_mask is not None:
            explicit = np.load(contact_mask, allow_pickle=False)
            if explicit.shape != self.mask.shape or not np.isin(explicit, [0, 1]).all():
                raise ValueError(f'Contact mask must be binary with shape {self.mask.shape}')
            self.mask = explicit.astype(bool)
            if self.surface is None and self.mask[:, :2].any():
                raise ValueError('Hand-object contacts requested in a scene without an object')
            self.stance_anchors=self.anchors.copy()
            for role in (2,3):
                for i in range(1,len(self.mask)):
                    if self.mask[i,role] and self.mask[i-1,role]:
                        self.stance_anchors[i,role]=self.stance_anchors[i-1,role]

    def _robot_body(self, body):
        while body:
            if body == self.root:
                return True
            body = self.model.body_parentid[body]
        return False

    def features(self, data):
        model = self.model
        contacts = np.zeros((4, 3))
        hand_penetration = np.zeros(2)
        if self.surface is not None:
            origin = data.geom_xpos[self.object_geom]
            rot = data.geom_xmat[self.object_geom].reshape(3, 3)
            centers = (data.geom_xpos[self.hand_geoms] - origin) @ rot
            nearest = self.surface.project(centers)
            normal = self.surface.normal(centers, nearest)
            # Actual hand collision surface, not the palm site/sphere center.
            contacts[:2] = centers - model.geom_size[self.hand_geoms, :1] * normal
            gap = np.sum((centers - nearest) * normal, axis=1) - model.geom_size[self.hand_geoms, 0]
            hand_penetration = np.maximum(-gap, 0.)
        foot_corners = []
        for ids in self.foot_geoms:
            samples = []
            for geom in ids:
                center = data.geom_xpos[geom]
                offset = np.array([0., 0., -model.geom_size[geom, 0]])
                if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_CAPSULE:
                    axis = data.geom_xmat[geom].reshape(3, 3)[:, 2]
                    samples.extend([center + sign * model.geom_size[geom, 1] * axis + offset for sign in [-1, 1]])
                else:
                    samples.append(center + offset)
            foot_corners.append(samples)
        foot_corners = np.array(foot_corners)
        # A tilted stance foot can bear load at its toe/heel. Its cloud centroid
        # is not a physical ground contact and must not be driven to z=ground.
        for foot in range(2):
            height = foot_corners[foot,:,2].min()
            support = foot_corners[foot,:,2] <= height+0.001
            contacts[foot+2] = foot_corners[foot,support].mean(axis=0)
        foot_velocity = []
        for foot, ids in enumerate(self.foot_geoms):
            jp = np.zeros((3, model.nv))
            jr = np.zeros_like(jp)
            body = model.geom_bodyid[ids[0]]
            mujoco.mj_jac(model, data, jp, jr, contacts[foot+2], body)
            foot_velocity.append(jp @ data.qvel)
        return dict(contacts=contacts, keypoints=data.xpos[self.bodies].copy(),
                    orientations=data.xmat[self.bodies].reshape(-1, 3, 3).copy(),
                    torso_up=data.xmat[self.torso].reshape(3, 3)[:, 2].copy(),
                    com=data.subtree_com[self.root].copy(),
                    foot_corners=foot_corners, foot_velocity=np.array(foot_velocity),
                    penetration=np.r_[hand_penetration,
                        np.maximum(data.geom_xpos[self.floor, 2]-foot_corners[:, :, 2], 0.).ravel()])

    def initial_state(self, frame):
        data = mujoco.MjData(self.model)
        data.qpos[:] = self.qref[frame]
        mujoco.mj_normalizeQuat(self.model, data.qpos)
        data.qvel[:] = self.vref[frame]
        mujoco.mj_forward(self.model, data)
        return data


from retarget.humanoid_admm import ProjectionADMM


def run(problem, output, *, start_frame=0, frames=None, viewer=False, initial_state=None, solver_factory=None):
    """Receding horizon: solve, execute first action, shift the shared ADMM state."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    def status(state, **details):
        temporary=output/'status.tmp.json'
        temporary.write_text(json.dumps(dict(state=state, **details),indent=2)+'\n')
        temporary.replace(output/'status.json')
    p = problem
    available = len(p.qref) - start_frame - 1
    frames = available if frames is None else min(frames, available)
    if frames < 1:
        raise ValueError('No reference transitions available')
    if initial_state is None:state=p.initial_state(start_frame)
    else:
        state=mujoco.MjData(p.model)
        mujoco.mj_copyData(state,p.model,initial_state)
    setup_started=time.perf_counter()
    status('initializing',completed_frames=0,requested_frames=frames)
    try:
        solver = (solver_factory or ProjectionADMM)(p)
    except BaseException as error:
        status('failed',completed_frames=0,requested_frames=frames,error=str(error))
        raise
    setup_seconds=time.perf_counter()-setup_started
    qpos, qvel, actions, zs, us, xs, masks, indices = [state.qpos.copy()], [state.qvel.copy()], [], [], [], [], [], []
    dense_actions,contact_forces,realized_wrenches=[],[],[]
    previous = p.aref[start_frame].copy()
    converged_windows = 0
    visualizer = None
    if viewer:
        from mujoco import viewer as mj_viewer
        visualizer = mj_viewer.launch_passive(p.model, state)
    started = time.perf_counter()
    failure=None
    status('running',completed_frames=0,requested_frames=frames)
    try:
        with (output / 'iterations.jsonl').open('w') as log:
            for window in range(frames):
                first = start_frame + window + 1
                targets = np.arange(first, min(first + p.settings.horizon, len(p.qref)))
                rollout, history = solver.solve(state, targets, previous)
                # This is the predicted first step from the exact measured state,
                # including actuator and solver warm-start state, with no reset.
                if visualizer is None:
                    mujoco.mj_copyData(state, p.model, rollout['next_state'])
                else:
                    with visualizer.lock():
                        mujoco.mj_copyData(state, p.model, rollout['next_state'])
                previous = solver.actions[0].copy()
                qpos.append(state.qpos.copy())
                qvel.append(state.qvel.copy())
                actions.append(previous)
                dense_actions.append(rollout['controls_dense'])
                contact_forces.append(solver.contact_forces[0])
                realized_wrenches.append(rollout['realized_wrench_dense'])
                zs.append(solver.z[0].copy())
                us.append(solver.dual[0].copy())
                xs.append(rollout['contacts'][0])
                masks.append(p.mask[first])
                indices.append(first)
                record = dict(window=window, reference_frame=first, solve_seconds=rollout['solve_seconds'],iterations=history)
                log.write(json.dumps(record) + '\n')
                log.flush()
                status('running',completed_frames=len(actions),requested_frames=frames,
                       seconds=time.perf_counter()-started)
                final = history[-1]
                converged_windows += int(final['converged'])
                print(f'[humanoid {window+1}/{frames}] primal={final["primal_max_m"]*1000:.2f}mm '
                      f'dual={final["dual_max_rho_m"]*1000:.2f} rho-mm '
                      f'cost={final["action_cost_after"]:.4g} '
                      f'h={final["executed_root_height_m"]:.3f}m tilt={final["executed_torso_tilt_deg"]:.1f}deg', flush=True)
                if visualizer is not None:
                    if not visualizer.is_running():
                        break
                    visualizer.sync()
    except BaseException as error:
        failure=f'{type(error).__name__}: {error}'
        raise
    finally:
        if visualizer is not None:
            visualizer.close()
        np.savez_compressed(output / 'rollout.npz', qpos=np.array(qpos), qvel=np.array(qvel),
                            ctrl=np.array(actions), z=np.array(zs), dual=np.array(us),
                            x=np.array(xs), contact_mask=np.array(masks),
                            reference_frames=np.array(indices),
                            time=np.arange(len(qpos))*p.ref_dt,
                            roles=np.array(ROLES))
        np.savez_compressed(output / 'execution.npz',ctrl_substeps=np.array(dense_actions),
                            contact_force_auxiliary=np.array(contact_forces),
                            normalized_mjwarp_contact_wrench=np.array(realized_wrenches),
                            physics_dt=p.ref_dt/p.settings.substeps)
        spider=p.settings.cost_mode=='spider'
        audit = dict(scene=str(p.scene), reference=str(p.reference_path), ref_dt=p.ref_dt,
                     start_frame=start_frame,
                     settings=asdict(p.settings), weights=asdict(p.weights),
                     solver_setup_seconds=setup_seconds,
                     reference_velocity_source=p.reference_velocity_source,
                     objective_version='spider_pose_mjwarp_admm_v9' if spider else 'whole_body_mjwarp_admm_v9',
                     task_cost=('norm(weight * manifold_qpos_difference, 2); post-step mean; terminal scale=1; velocity scale=0' if spider else 'legacy whole-body weighted squares'),
                     spider_pose_weights=(dict(root_position=1. if p.surface else 3.,root_rotation=1.,joints=1.,object_position=1.,object_rotation=.3) if spider else None),
                     keypoint_frame='no separate keypoint cost' if spider else 'pelvis-relative positions; separate absolute pelvis and COM tracking',
                     balance_model='no separate balance/upright cost' if spider else 'reference-relative normalized COM xy/z, DCM xy, vertical COM velocity and root-height collapse; no static hull during flight',
                     object_pose_cost=bool(spider and p.surface), object_velocity_cost=False,
                     hand_position_cost='no separate hand world cost; SPIDER root/joint tracking and contact ADMM' if spider else 'reference box world surface anchors plus real sphere radius during hand contact; independent of predicted pelvis and box pose',
                     object_wrench_target='no centroidal wrench task; sampled ComFree force LM targets SPIDER object pose' if spider else 'reference COM acceleration and angular-momentum rate; feedforward only, no object pose/velocity tracking',
                     wrench_penalty='disabled' if spider else ('pseudo_huber' if p.surface else 'quadratic'),
                     action_variables='real_joint_actuator_controls',
                     dynamics='MJWarp full floating-base and free object dynamics',
                     gradient=solver.gradient_source,finite_difference_gradients=(p.settings.action_solver in ('adamw','adamw-ls','acados-gn') or
                         (p.settings.action_solver=='sample-lm' and (not p.settings.sample_lm_reuse_samples or p.settings.action_iterations>1))),
                     action_solver=solver.action_solver_name,integrator='MJWarp_implicitfast',
                     contact_frames=['object_body_local', 'object_body_local', 'world', 'world'],
                     contact_query='actual box-facing sphere surface points and lowest toe/heel support points in each predicted state',
                     horizon_seconds=p.settings.horizon*p.settings.knot_stride*p.ref_dt,
                     actual_contact_forces_from_mjwarp_physics=True,
                     auxiliary_contact_forces_injected=False,
                     contact_realization_cost='real MJWarp contact response and action finite differences; no separate wrench loss' if spider else 'normalized robot/object COM wrench from MJWarp generalized constraint forces versus auxiliary contact forces; both contact and action blocks',
                     actual_contact_wrench_scale='robot force: m*g, torque: m*g*0.5m; object force: m*g, torque: m*g*0.3m',
                     joint_friction_model='original MJWarp frictionloss constraints and implicitfast damping',
                     collision_model='original MJCF collision shapes and explicit contact pairs',
                     contact_mask_source='reference_geometry_or_explicit_npy',
                     contact_mask_active_counts=p.mask.sum(axis=0).tolist(),
                     optimization_updates_per_window=p.settings.admm_iterations*p.settings.action_iterations,
                     control_update_period_seconds=p.ref_dt,
                     selection_policy='spider_pose_without_ADMM' if spider else 'physical_plus_primal_contact_penalty_without_dual',
                     update_order=['hands_and_feet_z', 'action', 'dual'],
                     seconds=time.perf_counter()-started, executed_frames=len(actions),
                     completed_requested_frames=len(actions) == frames,
                     converged_windows=converged_windows,
                     convergence_guarantee=False)
        if xs:
            error = np.linalg.norm(np.array(xs) - np.array(zs), axis=-1)
            active = np.array(masks)
            audit['executed_contact_error_max_m'] = float(error[active].max()) if active.any() else 0.
            audit['executed_contact_error_rms_m'] = float(np.sqrt(np.mean(error[active]**2))) if active.any() else 0.
            audit['contact_consensus_within_tolerance'] = bool(
                audit['executed_contact_error_max_m'] < p.settings.primal_tolerance)
            audit['root_height_min_m'] = float(np.min(np.array(qpos)[:, 2]))
        (output / 'config.json').write_text(json.dumps(audit, indent=2) + '\n')
        status('failed' if failure else 'completed' if len(actions)==frames else 'stopped',
               completed_frames=len(actions),requested_frames=frames,
               seconds=time.perf_counter()-started,error=failure)
    return output
