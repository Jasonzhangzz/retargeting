"""Single-key DexForge ContactAware adapter for fixed primitive objects.

The original elastic trust-region SQP and Clarabel QP are called directly.
This adapter supplies Wuji FK, actual mesh queries and semantic contact slots;
it does not require or synthesize MANO demonstrations.  The object remains in
its fixed input frame.  Template angles are only the numerical starting state.
"""

from dataclasses import asdict, dataclass
import copy
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from retarget.config.run import ROOT
from retarget.env.load_support import skew
from retarget.env.grasp.audit import GraspAudit
from retarget.env.grasp.hand import query_path
from retarget.env.grasp.input_geometry import WujiSelfCollision


COMMIT = "fa6d671948ef03d18514062748b94e445b63cd49"
UPSTREAM_ROOT = ROOT / "third_party/DexForge"


@dataclass(frozen=True)
class Settings:
    max_iterations: int = 100
    contact_tolerance_mm: float = 2.0
    penetration_tolerance_mm: float = 2.0
    self_penetration_tolerance_mm: float = 0.2
    normal_cosine: float = 0.5
    required_fingers: object = None
    anchor_weight: float = 100.0
    normal_weight: float = 1.0
    shape_prior_weight: float = 0.001
    force_closure_weight: float = 1.0
    friction: object = None
    force_limit: float = 20.0
    trust_translation_mm: float = 4.0
    trust_rotation_rad: float = 0.05
    trust_joint_rad: float = 0.08
    qp_max_iterations: int = 10000
    bone_max_deviation_deg: object = None
    bone_style_weight: float = 0.0
    pose_rotation_limit_deg: float = 15.0
    pose_rotation_weight: float = 1.0
    pose_center_limit_mm: float = 15.0
    pose_center_weight: float = 1.0


def validate_settings(args):
    supplied = getattr(args, "contactaware", {})
    if not isinstance(supplied, dict):
        raise ValueError("contactaware must be a mapping")
    try:
        settings = Settings(**supplied)
    except TypeError as error:
        raise ValueError(f"Invalid contactaware settings: {error}") from error
    for name, value in asdict(settings).items():
        if name in ("required_fingers", "friction", "bone_max_deviation_deg") and value is None:
            continue
        if name == "required_fingers":
            if not isinstance(value, (list, tuple)) or not value or set(value) - set(FINGER_NAMES):
                raise ValueError("contactaware.required_fingers must name physical fingers")
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise ValueError(f"contactaware.{name} must be finite and nonnegative")
    for name in ("max_iterations", "qp_max_iterations"):
        value = getattr(settings, name)
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"contactaware.{name} must be a positive integer")
    if not 0 <= settings.normal_cosine < 1:
        raise ValueError("contactaware.normal_cosine must be in [0, 1)")
    if not 0 < settings.pose_rotation_limit_deg < 180 or (
            settings.bone_max_deviation_deg is not None and not 0 < settings.bone_max_deviation_deg < 180):
        raise ValueError("Grasp intent angular limits must be between 0 and 180 degrees")
    if settings.pose_center_limit_mm <= 0:
        raise ValueError("Grasp intent center limit must be positive")
    if min(settings.contact_tolerance_mm, settings.penetration_tolerance_mm,
           settings.trust_translation_mm, settings.trust_rotation_rad,
           settings.trust_joint_rad, settings.force_limit) <= 0:
        raise ValueError("ContactAware tolerances, trust and force limit must be positive")
    return settings


def require_upstream():
    manifest = UPSTREAM_ROOT / "CONTACTAWARE_UPSTREAM.json"
    if not manifest.is_file():
        raise ValueError("Pinned DexForge ContactAware source is missing")
    source = json.loads(manifest.read_text())
    if source.get("commit") != COMMIT:
        raise ValueError("Unexpected ContactAware source revision")
    for relative in (
        "contactaware/contactaware/solver/grasp_sqp.py",
        "contactaware/contactaware/solver/qp.py",
        "contactaware/contactaware/solver/sparse_qp.py",
        "contactaware/contactaware/contact/wrench_residual.py",
        "contactaware/contactaware/initialization/multi_contact_keys.py",
    ):
        path = UPSTREAM_ROOT / relative
        expected = source["sha256"].get(relative)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"ContactAware source integrity check failed: {relative}")
    package = str(UPSTREAM_ROOT / "contactaware")
    if package not in sys.path:
        sys.path.insert(0, package)
    try:
        import clarabel  # noqa: F401
        from contactaware.solver.grasp_sqp import solve_joint_grasp
        from contactaware.solver.sparse_qp import make_joint_contact_qp
    except ImportError as error:
        raise ValueError("ContactAware requires the pinned source and clarabel dependency") from error
    return source


FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")


def _finger(body_name):
    return FINGER_NAMES[int(body_name[len("right_finger")]) - 1] if body_name.startswith("right_finger") else "palm"


def signed_distance_normal(points, projected, face_normals, distance):
    """Outward feature normal; triangle choice does not choose an edge normal."""
    face = np.asarray(face_normals, float)
    face = face / np.maximum(np.linalg.norm(face, axis=1, keepdims=True), 1e-12)
    delta = np.asarray(points, float)-np.asarray(projected, float)
    lengths = np.linalg.norm(delta, axis=1)
    result = np.sign(distance)[:, None]*delta/np.maximum(lengths[:, None], 1e-12)
    near_surface = (np.abs(distance) < 1e-7) | (lengths < 1e-7)
    result[near_surface] = face[near_surface]
    return result


class _Problem:
    def __init__(self, hand, seed, args, settings):
        from contactaware.contact.wrench_residual import ContactWrenchResidual
        from contactaware.initialization.multi_contact_keys import TopologyReference

        self.hand, self.settings = hand, settings
        self.enforce_intent = not getattr(args, 'ca_as_seed', False)
        self.device = getattr(args, "init_device", "cuda:0")
        self.audit = GraspAudit(hand.surface, self.device)
        self.radius = max(self.audit.radius, 0.01)
        self.count = len(hand.bodies)
        self.dimension = 26 + self.count * 3
        self.seed = np.asarray(seed.hand_qpos, dtype=float).copy()
        self.scale = np.r_[np.full(3, self.radius), np.ones(23), np.full(self.count * 3, self.radius)]
        self.initial_anchors, _, _ = self.query(np.asarray(seed.object_points))
        self.reference = self.initial_anchors.copy()
        self.topology = TopologyReference.from_points(self.initial_anchors)
        friction = args.friction if settings.friction is None else settings.friction
        self.wrench = ContactWrenchResidual(np.zeros(3), self.radius, friction, settings.force_limit)
        self.contact_fingers = np.asarray([_finger(hand.model.body(int(b)).name) for b in hand.bodies])
        annotated = tuple(name for name in FINGER_NAMES if np.any(self.contact_fingers == name))
        self.required = annotated if settings.required_fingers is None else tuple(settings.required_fingers)
        if set(self.required) - set(annotated):
            raise ValueError("A required ContactAware finger has no template contact patch")
        self.groups_by_body = tuple(np.flatnonzero(hand.all_bodies == body) for body in np.unique(hand.all_bodies))
        self.working = np.empty(0, int)
        self.capsules = WujiSelfCollision(hand.model, self.device)
        self.self_depths = lambda q: self.capsules.depths(q)
        self.self_jacobian = torch.func.jacfwd(self.self_depths)
        from retarget.env.grasp.regions import hand_forbidden_mask
        supplied_forbidden = getattr(hand, "contactaware_forbidden_mask", None)
        if supplied_forbidden is None:
            self.forbidden = hand_forbidden_mask(hand, getattr(hand, "contact_reference_normals", hand.normals))
        else:
            supplied_forbidden = np.asarray(supplied_forbidden)
            if supplied_forbidden.dtype != np.bool_ or supplied_forbidden.shape != (len(hand.all_bodies),):
                raise ValueError("ContactAware precomputed forbidden mask must be boolean in full hand-query order")
            self.forbidden = supplied_forbidden.copy()
        self.patch_radii = []
        self.query_indices = []
        for i, body in enumerate(hand.bodies):
            with np.load(query_path(hand.model, int(body))) as asset:
                points = asset["local_pos"]
            mask = np.asarray(hand.semantic_contact_masks[i], bool)
            if mask.shape != (len(points),) or not mask.any():
                raise ValueError("ContactAware requires nonempty semantic query masks")
            distance = np.linalg.norm(points - hand.local[i], axis=1)
            selected = int(distance.argmin())
            if distance[selected] > 1e-7 or not mask[selected]:
                raise ValueError("ContactAware point is outside its fixed semantic patch")
            self.query_indices.append(selected)
            self.patch_radii.append(max(float(np.sqrt(np.mean(np.sum((points[mask] - points[mask].mean(0))**2, axis=1)))), 0.001))
        self.patch_radii = np.asarray(self.patch_radii)
        self.bones = None
        if settings.bone_max_deviation_deg is not None or settings.bone_style_weight:
            from .template import BoneReference
            self.bones = BoneReference(hand.model, hand.ik_reference_qpos, self.device)
            self.bone_residual_jacobian = torch.func.jacfwd(self.bones.style_residual)
            self.bone_cosines = lambda q: (self.bones.directions(q)*self.bones.target).sum(-1).flatten()
            self.bone_cosine_jacobian = torch.func.jacfwd(self.bone_cosines)
        self.cache_x = self.cache = None

    def query(self, points):
        tensor = torch.as_tensor(np.asarray(points).copy(), dtype=torch.float64, device=self.device)
        nearest, face_normal, phi = self.audit.query(tensor)
        nearest, normal, phi = nearest.cpu().numpy().astype(float), face_normal.cpu().numpy().astype(float), phi.cpu().numpy().astype(float)
        gradient = signed_distance_normal(points, nearest, normal, phi)
        return nearest, gradient, (phi, gradient)

    def normal_jacobian(self, points):
        # Mesh edge/vertex normals vary with the material point. This finite
        # difference is only over the guided points, never the whole-hand cloud.
        epsilon = 1e-6
        offsets = np.r_[np.eye(3), -np.eye(3)]*epsilon
        samples = (np.asarray(points)[:, None, :]+offsets[None]).reshape(-1, 3)
        _, normals, _ = self.query(samples)
        normals = normals.reshape(len(points), 6, 3)
        return ((normals[:, :3]-normals[:, 3:])/(2*epsilon)).swapaxes(1, 2)

    def coordinates(self, vector):
        return self.seed + vector[:26] * self.scale[:26], self.initial_anchors + self.radius * vector[26:].reshape(-1, 3)

    def project_initial(self, vector):
        result = np.asarray(vector, dtype=float).copy()
        _, anchors = self.coordinates(result)
        projected, _, _ = self.query(anchors)
        result[26:] = ((projected - self.initial_anchors) / self.radius).ravel()
        return result

    def tangent_rows(self, vector):
        _, anchors = self.coordinates(vector)
        _, normals, _ = self.query(anchors)
        rows = np.zeros((self.count, self.dimension))
        for i, normal in enumerate(normals):
            rows[i, 26 + 3*i:26 + 3*i + 3] = normal
        return rows

    def bounds(self):
        joints = [np.flatnonzero(self.hand.model.jnt_qposadr == i)[0] for i in range(26)]
        lower, upper = self.hand.model.jnt_range[joints].T
        return (np.r_[(lower-self.seed)/self.scale[:26], np.full(self.count*3, -np.inf)],
                np.r_[(upper-self.seed)/self.scale[:26], np.full(self.count*3, np.inf)])

    def trust(self):
        s = self.settings
        return np.r_[np.full(3, s.trust_translation_mm/1000), np.full(3, s.trust_rotation_rad),
                     np.full(20, s.trust_joint_rad), np.full(self.count*3, s.trust_translation_mm/1000)] / self.scale

    def evaluate(self, vector):
        x = np.asarray(vector, float)
        if self.cache_x is not None and np.array_equal(x, self.cache_x):
            return self.cache
        q, anchors = self.coordinates(x)
        hand = self.hand
        hand.set(q)
        points, jp, jr = hand.points(hand.bodies, hand.local, jacobian=True)
        normal = np.einsum("nij,nj->ni", hand.data.xmat[hand.bodies].reshape(-1,3,3), hand.normals)
        normal_jac = -skew(normal) @ jr
        all_points, all_jac, _ = hand.points(hand.all_bodies, hand.all_local, jacobian=True)
        _, anchor_normals, anchor_query = self.query(anchors)
        _, object_normals, _ = self.query(points)
        _, _, object_query = self.query(all_points)
        phi, gradient = object_query
        arows = np.zeros((self.count,3,self.dimension))
        arows[:,:,26:] = self.radius*np.eye(self.count*3).reshape(self.count,3,-1)
        prows, nrows = np.zeros_like(arows), np.zeros_like(arows)
        prows[:,:,:26], nrows[:,:,:26] = jp*self.scale[:26], normal_jac*self.scale[:26]
        object_normal_rows = np.zeros_like(arows)
        object_normal_rows[:, :, :26] = (self.normal_jacobian(points)@jp)*self.scale[:26]
        delta, drows = points-anchors, prows-arows
        gap = np.linalg.norm(delta, axis=1)
        facing = -np.sum(normal*object_normals,axis=1)
        depth_tensor = self.capsules.fk.tensor(q)
        self_depth = self.self_depths(depth_tensor).detach().cpu().numpy()
        self_rows = self.self_jacobian(depth_tensor).detach().cpu().numpy()
        s, r = self.settings, self.radius
        residuals = [np.sqrt(s.anchor_weight)*delta.ravel()/r,
            np.sqrt(s.anchor_weight)*s.normal_weight*((normal+object_normals)*self.patch_radii[:,None]).ravel()/r,
            np.sqrt(s.shape_prior_weight)*x[:26],
            (anchors.mean(0)-self.reference.mean(0))/r,
            ((anchors-anchors.mean(0)-self.reference+self.reference.mean(0))/np.sqrt(self.count)/r).ravel()]
        jacobians = [np.sqrt(s.anchor_weight)*drows.reshape(-1,self.dimension)/r,
            np.sqrt(s.anchor_weight)*s.normal_weight*((nrows+object_normal_rows)*self.patch_radii[:,None,None]).reshape(-1,self.dimension)/r,
            np.pad(np.sqrt(s.shape_prior_weight)*np.eye(26),((0,0),(0,self.count*3))),
            arows.mean(0)/r,
            ((arows-arows.mean(0))/np.sqrt(self.count)/r).reshape(-1,self.dimension)]
        if self.bones is not None and s.bone_style_weight:
            residuals.append(np.sqrt(s.bone_style_weight)*self.bones.style_residual(depth_tensor).detach().cpu().numpy())
            bj = self.bone_residual_jacobian(depth_tensor).detach().cpu().numpy()*self.scale[:26]
            jacobians.append(np.pad(np.sqrt(s.bone_style_weight)*bj, ((0,0),(0,self.count*3))))
        if hasattr(hand, "pose_intent_qpos"):
            from .intent import rotation_residual
            rr = rotation_residual(q, hand.pose_intent_qpos)
            rj = np.zeros((9, self.dimension))
            for axis in range(3, 6):
                offset = np.zeros(26); offset[axis] = 1e-6
                rj[:, axis] = (rotation_residual(q+offset, hand.pose_intent_qpos)-
                    rotation_residual(q-offset, hand.pose_intent_qpos))/(2e-6)*self.scale[axis]
            residuals.append(np.sqrt(s.pose_rotation_weight)*rr)
            jacobians.append(np.sqrt(s.pose_rotation_weight)*rj)
        if hasattr(hand, "pose_intent_center"):
            from .intent import palm_object_center
            cr = (palm_object_center(hand,q)-hand.pose_intent_center)/r
            cj = np.zeros((3,self.dimension))
            for axis in range(6):
                offset=np.zeros(26); offset[axis]=1e-6
                cj[:,axis]=(palm_object_center(hand,q+offset)-palm_object_center(hand,q-offset))/(2e-6*r)*self.scale[axis]
            residuals.append(np.sqrt(s.pose_center_weight)*cr)
            jacobians.append(np.sqrt(s.pose_center_weight)*cj)
        if s.force_closure_weight:
            # DexForge actual_contact_costs uses the real robot material
            # points, not independently movable object anchors. Their actual
            # feature normals carry the same guided-point derivative.
            wrench, wrench_rows = self.wrench(points,-object_normals,prows,-object_normal_rows)
            residuals.append(np.sqrt(s.force_closure_weight)*wrench)
            jacobians.append(np.sqrt(s.force_closure_weight)*wrench_rows)
        checked = np.union1d([ids[np.argmin(phi[ids])] for ids in self.groups_by_body], self.working)
        collision_rows = np.einsum("ni,nij->nj", gradient[checked],all_jac[checked])*self.scale[:26]/r
        # A 0.1 mm buffer makes the independent physical acceptance stricter
        # than the normalized SQP stopping tolerance.
        pen_allow = max(s.penetration_tolerance_mm-0.1,0)/1000
        values = [(phi[checked]+pen_allow)/r]
        rows = [np.pad(collision_rows,((0,0),(0,self.count*3)))]
        group = [np.zeros(len(checked),int)]
        if self.forbidden.any():
            forbidden_ids = np.flatnonzero(self.forbidden)
            i = forbidden_ids[np.argmin(phi[forbidden_ids])]
            values.append(np.array([(phi[i]-.0011)/r]))
            fr = gradient[i]@all_jac[i]*self.scale[:26]/r
            rows.append(np.r_[fr,np.zeros(self.count*3)][None]);group.append(np.zeros(1,int))
        values.append((s.self_penetration_tolerance_mm/1000-self_depth)/r)
        rows.append(np.pad(-self_rows*self.scale[:26]/r,((0,0),(0,self.count*3))))
        group.append(np.ones(len(self_depth),int))
        for name in self.required:
            ids = (hand.cylinder_middle_slots if name == "middle" and hasattr(hand, "cylinder_middle_slots")
                   else np.flatnonzero(self.contact_fingers==name))
            i=ids[np.argmin(gap[ids])]
            unit=delta[i]/max(gap[i],1e-12)
            values.append(np.array([(max(s.contact_tolerance_mm-.1,0)/1000-gap[i])/r,
                                    facing[i]-s.normal_cosine-.01]))
            rows.append(np.stack([-unit@drows[i]/r,
                -object_normals[i]@nrows[i]-normal[i]@object_normal_rows[i]]))
            group.append(np.full(2,2,int))
        topology, topology_rows = self.topology.linearize(anchors,arows,r)
        values.append(topology);rows.append(topology_rows);group.append(np.full(len(topology),3,int))
        if self.enforce_intent and self.bones is not None and s.bone_max_deviation_deg is not None:
            cosines = self.bone_cosines(depth_tensor).detach().cpu().numpy()
            bj = self.bone_cosine_jacobian(depth_tensor).detach().cpu().numpy()*self.scale[:26]
            values.append(cosines-np.cos(np.deg2rad(s.bone_max_deviation_deg)))
            rows.append(np.pad(bj,((0,0),(0,self.count*3)))); group.append(np.full(len(cosines),4,int))
        if self.enforce_intent and hasattr(hand, "pose_intent_qpos"):
            # ||R-Rref||_F² = 4(1-cos(theta)); smooth at zero rotation.
            values.append(np.array([4*(1-np.cos(np.deg2rad(s.pose_rotation_limit_deg)))-rr@rr]))
            rows.append((-2*rr@rj)[None]); group.append(np.array([4]))
        if self.enforce_intent and hasattr(hand, "cylinder_middle_slots"):
            ids = hand.cylinder_middle_slots
            band = hand.cylinder_middle_half_band
            # Constrain both real material points and anchors: no phantom
            # centered anchor while the middle finger still grips an end cap.
            for locations, derivatives in ((points,prows),(anchors,arows)):
                z, dz = locations[ids,2], derivatives[ids,2]
                values.extend(((band-z)/r,(band+z)/r))
                rows.extend((-dz/r,dz/r)); group.extend((np.full(len(ids),4,int),)*2)
        if self.enforce_intent and hasattr(hand, "pose_intent_center"):
            values.append(np.array([(s.pose_center_limit_mm/1000/r)**2-cr@cr]))
            rows.append((-2*cr@cj)[None]); group.append(np.array([4]))
        result = dict(q=q,anchors=anchors,object_normals=object_normals,anchor_normals=anchor_normals,
            points=points,normal=normal,
            gap=gap,facing=facing,phi=phi,self_depth=self_depth,
            residual=np.concatenate(residuals),jacobian=np.vstack(jacobians),
            constraints=np.concatenate(values),rows=np.vstack(rows),groups=np.concatenate(group),
            checked=checked,anchor_surface_gap=anchor_query[0])
        self.cache_x,self.cache=x.copy(),result
        return result

    def prepare_iteration(self, vector):
        value=self.evaluate(vector)
        self.working=np.union1d(self.working,value["checked"])
        self.cache_x=None

    def diagnostics(self, vector):
        v=self.evaluate(vector);s=self.settings
        per_finger={}
        for name in self.required:
            ids=np.flatnonzero(self.contact_fingers==name)
            good=(v["gap"][ids]*1000<=s.contact_tolerance_mm)&(v["facing"][ids]>=s.normal_cosine)
            per_finger[name]=dict(contact=bool(good.any()),contact_count=int(good.sum()),
                minimum_error_mm=float(v["gap"][ids].min()*1000),maximum_facing=float(v["facing"][ids].max()))
        pen=float(np.maximum(-v["phi"],0).max()*1000)
        self_pen=float(v["self_depth"].max(initial=0)*1000)
        forbidden=float(v["phi"][self.forbidden].min()*1000) if self.forbidden.any() else None
        joints=np.array([np.flatnonzero(self.hand.model.jnt_qposadr==i)[0] for i in range(26)])
        lower,upper=self.hand.model.jnt_range[joints].T
        joint_violation=float(np.maximum(np.r_[lower-v["q"],v["q"]-upper],0).max())
        active=(v["gap"]*1000<=s.contact_tolerance_mm)&(v["facing"]>=s.normal_cosine)
        violation=float(np.maximum(-v["constraints"],0).max(initial=0))
        success=all(item["contact"] for item in per_finger.values()) and pen<=s.penetration_tolerance_mm and self_pen<=s.self_penetration_tolerance_mm and (forbidden is None or forbidden>=1.0) and joint_violation<1e-7 and violation<=1e-4/self.radius
        from .intent import intent_diagnostics, ANGLE_NUMERIC_TOL_DEG, POSITION_NUMERIC_TOL_M
        intent = intent_diagnostics(self.hand, v["q"], v["points"])
        if "nominal_rotation_error_deg" in intent:
            intent["nominal_rotation_limit_deg"] = s.pose_rotation_limit_deg
            intent["nominal_rotation_valid"] = intent["nominal_rotation_error_deg"] <= s.pose_rotation_limit_deg+ANGLE_NUMERIC_TOL_DEG
            success = success and intent["nominal_rotation_valid"]
        if "nominal_center_error_mm" in intent:
            intent["nominal_center_limit_mm"] = s.pose_center_limit_mm
            intent["nominal_center_valid"] = intent["nominal_center_error_mm"] <= s.pose_center_limit_mm+POSITION_NUMERIC_TOL_M*1000
            success = success and intent["nominal_center_valid"]
        if self.bones is not None:
            intent["bone_max_deviation_deg"] = self.bones.maximum_angle(self.bones.fk.tensor(v["q"]))
            if s.bone_max_deviation_deg is not None:
                intent["bone_deviation_limit_deg"] = s.bone_max_deviation_deg
                intent["bone_deviation_valid"] = intent["bone_max_deviation_deg"] <= s.bone_max_deviation_deg+ANGLE_NUMERIC_TOL_DEG
                success = success and intent["bone_deviation_valid"]
        if hasattr(self.hand, "cylinder_middle_slots"):
            ids = self.hand.cylinder_middle_slots
            centered_contact = bool(np.any(active[ids] &
                (np.abs(v["points"][ids,2]) <= self.hand.cylinder_middle_half_band+POSITION_NUMERIC_TOL_M)))
            intent["middle_center_contact_active"] = centered_contact
            success = success and centered_contact and intent["middle_in_center_band"]
        return dict(success=bool(success),required_finger_contact=per_finger,
            grasp_intent=intent,
            per_patch=[dict(label=self.hand.semantic_patch_labels[i],query_index=self.query_indices[i],
                contact_error_mm=float(v["gap"][i]*1000),facing_cosine=float(v["facing"][i]),
                anchor_surface_gap_mm=float(v["anchor_surface_gap"][i]*1000),
                active=bool(active[i]),release_reason=None if active[i] else (
                    "contact_distance" if v["gap"][i]*1000>s.contact_tolerance_mm else "surface_facing")) for i in range(self.count)],
            whole_hand_penetration_mm=pen,self_penetration_mm=self_pen,
            forbidden_region_min_gap_mm=forbidden,joint_violation_rad=joint_violation,
            semantic_patch_violation_count=0,active_contact_mask=active.tolist(),
            max_normalized_constraint_violation=violation)


class _Objective:
    def __init__(self, problem):self.problem=problem
    def __call__(self,x):return self.problem.evaluate(x)["residual"]
    def jacobian(self,x):return self.problem.evaluate(x)["jacobian"]


class _Constraints:
    def __init__(self,problem):self.problem=problem
    def __call__(self,x):return self.problem.evaluate(x)["constraints"]
    def jacobian(self,x):return self.problem.evaluate(x)["rows"]
    def groups(self,x):return self.problem.evaluate(x)["groups"]
    def diagnostics(self,x):return [self.problem.diagnostics(x)]
    def transfer_phase_initials(self,x):return np.asarray(x),[]


def refine_grasp_reference(hand, seed_guess, args, *, primitive=None):
    """Solve contact anchors and Wuji IK, returning a separately marked q_ref."""
    if primitive is not None and hasattr(args, "intent_template"):
        from .intent import configure_intent
        configure_intent(hand, args.intent_template, primitive, args)
    source=require_upstream();settings=validate_settings(args)
    from contactaware.solver.grasp_sqp import solve_joint_grasp,ShrinkingTrustRegion
    from contactaware.solver.qp import QPSolverOptions
    from contactaware.solver.sparse_qp import make_joint_contact_qp
    problem=_Problem(hand,seed_guess,args,settings)
    lower,upper=problem.bounds()
    solution,report=solve_joint_grasp(_Objective(problem),_Constraints(problem),problem,
        np.zeros(problem.dimension),lower,upper,problem.trust(),
        solver=make_joint_contact_qp(QPSolverOptions(1e-9,0.,1e-7,settings.qp_max_iterations)),
        regularization=1e-6,tolerance=1e-5,feasibility_tolerance=1e-4/problem.radius,
        max_iterations=settings.max_iterations,
        accept=lambda x:problem.diagnostics(x)["success"],
        update_trust=ShrinkingTrustRegion(.25,.5,1/64),prepare_iteration=problem.prepare_iteration)
    # Match the controller's intrinsic XYZ principal branch, so q_ref and the
    # scattered candidate represent the same angles as well as the same pose.
    q,_=problem.coordinates(solution)
    q[3:6]=Rotation.from_euler("XYZ",q[3:6]).as_euler("XYZ")
    solution[:26]=(q-problem.seed)/problem.scale[:26]
    value=problem.evaluate(solution);diagnostics=problem.diagnostics(solution)
    result=copy.deepcopy(seed_guess)
    for attribute in ("hand_reference_qpos", "contactaware_reference_qpos"):
        if hasattr(result,attribute):
            delattr(result,attribute)
    result.hand_qpos=value["q"].copy()
    result.object_points=value["anchors"].copy()
    result.object_normals=value["anchor_normals"].copy()
    result.contact_mask=np.asarray(diagnostics["active_contact_mask"],dtype=bool)
    metadata=dict(diagnostics,source_commit=source["commit"],
        method="DexForge solve_joint_grasp elastic trust-region SQP + Clarabel; primitive geometry/FK adapter",
        object_pose_fixed=True,template_used_as="semantic patches and numerical seed only",
        variables="26 Wuji qpos and independent object-surface anchors",
        force_closure_geometry="actual hand query points and nearest object outward normals; no phantom-anchor wrench",
        object_normal_jacobian_policy="signed-distance feature normal; centered 1e-6m finite difference at guided material points only",
        guided_finger_bone_angle_constraint=settings.bone_max_deviation_deg is not None,
        settings=asdict(settings),solver=report,
        initial_qpos=np.asarray(seed_guess.hand_qpos).tolist(),reference_qpos=result.hand_qpos.tolist())
    # Independent diagnostic on contacts which actually met contact/facing
    # tolerances. It does not certify actuator-limited or dynamic support.
    active=result.contact_mask
    if active.sum()>=3:
        from contactaware.contact.force_margin import make_force_margin_solver
        solve_margin=make_force_margin_solver(center=np.zeros(3),radius=problem.radius,
            friction=args.friction if settings.friction is None else settings.friction)
        margin=solve_margin(value["points"][active],-value["object_normals"][active])
        metadata["active_actual_contact_force_margin"]=dict(status=margin.status,
            reserve=margin.reserve,wrench_rank=margin.wrench_rank,
            maximum_balance_error=margin.max_balance_error,
            scope="normalized friction-cone geometric diagnostic, without actuator or dynamic hold certification")
    result.metadata.update(contactaware=metadata,initialization_valid=diagnostics["success"],
        contacts_optional=False,initialization_policy="dexforge_contactaware_strict_required_fingers",
        method="DexForge ContactAware primitive contact and hand IK reference",
        initial_penetration_mm=diagnostics["whole_hand_penetration_mm"],
        initial_self_penetration_mm=diagnostics["self_penetration_mm"],
        forbidden_region_min_gap_mm=diagnostics["forbidden_region_min_gap_mm"],
        matching=dict(gap_mm=[p["contact_error_mm"] for p in diagnostics["per_patch"]],
            contact_mask=result.contact_mask.tolist(),
            source="optimized CA material point to refined object anchor"),
        failure_reasons=[] if diagnostics["success"] else ["contactaware_reference_infeasible"])
    if diagnostics["success"]:
        result.contactaware_reference_qpos=result.hand_qpos.copy()
        result.hand_reference_qpos=result.hand_qpos.copy()
    return result
