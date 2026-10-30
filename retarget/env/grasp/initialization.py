"""Input bridge from the original Dexonomy initializer to static Wuji inputs.

Sampling, contact-frame construction, pose optimization and surface queries run
in the external checkout. Local code packs hand/model inputs, converts frames
and validates candidates without moving original poses or surface contacts.
Upstream contact losses remain diagnostic. Shape-first inputs use geometric
clearance and bone references for acceptance, allowing detached fingers.
The downstream evolution/filter pipeline is not run.
"""

from dataclasses import dataclass
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

MIN_SHARP_EDGE_CLEARANCE_M = 0.0001
# Stable bit order for numeric archives; append new reasons, never reorder.
INITIALIZATION_FAILURES = (
    "nonfinite_hand_configuration",
    "hand_joint_limits",
    "contact_skin_region_mismatch",
    "contact_normal_mismatch",
    "contact_position_mismatch",
    "hand_object_penetration",
    "finger_self_collision",
    "contact_on_sharp_edge",
    "contact_force_cones_do_not_balance_load",
    "actuator_preload_infeasible",
    "preload_unavailable",
    "upstream_initialization_failed",
    "forbidden_region_contact",
    "hand_shape_mismatch",
    "object_orientation_mismatch",
    "missing_size_one_reference",
    "tip_patch_center_mismatch",
)


def initialization_failure_reasons(info):
    """Concrete input failures, distinct from a measured gravity-test drop."""
    reasons = list(info.get("ik_failure_reasons", []))
    optional = info.get("contacts_optional", False)
    matching = info.get("matching", {})
    if not optional and matching.get(
        "normal_squared_residual_max", 0.0
    ) >= matching.get("normal_squared_residual_threshold", 0.3):
        reasons.append("contact_normal_mismatch")
    gap = max(matching.get("gap_mm", [0.0])) / 1000
    if not optional and gap * gap >= matching.get(
        "position_squared_residual_threshold_m2", 0.001
    ):
        reasons.append("contact_position_mismatch")
    if (
        not optional
        and gap * 1000 > info.get("contact_tolerance_mm", float("inf"))
        and "contact_position_mismatch" not in reasons
    ):
        reasons.append("contact_position_mismatch")
    tolerance = info.get("penetration_tolerance_mm", 2.0)
    if info.get("initial_penetration_mm", 0.0) > tolerance:
        reasons.append("hand_object_penetration")
    if info.get("initial_self_penetration_mm", 0.0) > tolerance:
        reasons.append("finger_self_collision")
    clearance = info.get("minimum_sharp_edge_clearance_mm")
    if (
        not optional
        and clearance is not None
        and clearance < MIN_SHARP_EDGE_CLEARANCE_M * 1000
    ):
        reasons.append("contact_on_sharp_edge")
    if (
        not optional
        and info.get("load_conditioning_required")
        and not (info.get("load_conditioning") or {}).get("feasible")
    ):
        load = info.get("load_conditioning") or {}
        if load.get("force_cone_diagnostic", {}).get("static_load_excluded"):
            reasons.append("contact_force_cones_do_not_balance_load")
        else:
            reasons.append(
                "actuator_preload_infeasible" if load else "preload_unavailable"
            )
    return reasons


def validate_initialization(hand, guess, args, reference_normals):
    """Final local acceptance; never move upstream poses or surface anchors."""
    info, q = guess.metadata, guess.hand_qpos
    failures = []
    if getattr(hand, "wrist_reference_missing", False):
        failures.append("missing_size_one_reference")
    if not np.isfinite(q).all():
        failures.append("nonfinite_hand_configuration")
    joints = [np.flatnonzero(hand.model.jnt_qposadr == i)[0] for i in range(26)]
    low, high = hand.model.jnt_range[joints].T
    if np.any(q < low - 1.0e-7) or np.any(q > high + 1.0e-7):
        failures.append("hand_joint_limits")
    dots = np.einsum("ij,ij->i", hand.normals, reference_normals)
    if not np.isfinite(dots).all() or np.any(dots <= 0.5):
        failures.append("contact_skin_region_mismatch")
    if getattr(hand, "tip_patches", None):
        patches = []
        for i, body in enumerate(hand.bodies):
            patch = hand.tip_patches.get(int(body))
            if patch is not None:
                distance = float(np.linalg.norm(hand.local[i] - patch["center"]))
                patches.append(
                    dict(
                        body=hand.model.body(int(body)).name,
                        center=patch["center"].tolist(),
                        offset_mm=distance * 1000,
                        radius_mm=patch["radius_m"] * 1000,
                    )
                )
                if (
                    distance > patch["radius_m"] + 1.0e-7
                    and "tip_patch_center_mismatch" not in failures
                ):
                    failures.append("tip_patch_center_mismatch")
        info["tip_patch_contacts"] = patches
    if hasattr(hand, "object_orientation") and np.isfinite(q).all():
        from retarget.env.grasp.orientation import orientation_angles

        angle = float(orientation_angles(hand, q))
        limit = hand.object_orientation["maximum_angle_deg"]
        info["object_orientation"] = dict(
            hand.object_orientation, angle_deg=angle, valid=angle <= limit + 1.0e-5
        )
        if angle > limit + 1.0e-5:
            failures.append("object_orientation_mismatch")
    if hasattr(hand, "wrist_pose_reference") and np.isfinite(q).all():
        from retarget.task.grasp.wrist_prior import prior_diagnostics

        diagnostics = prior_diagnostics(hand, q)
        info["wrist_pose_prior"] = dict(
            hand.wrist_pose_reference,
            **{k: float(v[0]) for k, v in diagnostics.items()},
        )
    info.update(
        contact_tolerance_mm=args.contact_tolerance_mm,
        contact_region_normal_dot=dots.tolist(),
        ik_failure_reasons=failures,
    )
    if hasattr(hand, "ik_reference_qpos") and np.isfinite(q).all():
        from retarget.env.grasp.template import BoneReference
        from retarget.env.grasp.regions import SurfaceClearance, MAX_BONE_DEVIATION_DEG

        bones = BoneReference(hand.model, hand.ik_reference_qpos, args.init_device)
        regions = SurfaceClearance(hand, reference_normals, args.init_device)
        constraints = regions.diagnostics(regions.fk.tensor(q))
        angle = bones.maximum_angle(bones.fk.tensor(q))
        info.update(
            constraints,
            contacts_optional=True,
            initialization_policy="shape_first",
            bone_direction_max_deviation_deg=angle,
            initial_penetration_mm=constraints["whole_hand_penetration_mm"],
        )
        if not constraints["forbidden_region_valid"]:
            failures.append("forbidden_region_contact")
        if angle > MAX_BONE_DEVIATION_DEG + 1.0e-5:
            failures.append("hand_shape_mismatch")
    failures = initialization_failure_reasons(info)
    info["initialization_valid"] = bool(
        info.get("initialization_valid", True) and not failures
    )
    info["failure_reasons"] = failures or (
        [] if info["initialization_valid"] else ["upstream_initialization_failed"]
    )
    return info["initialization_valid"]


def pose_in_hand(q):
    """Object pose in Wuji root coordinates: xyz followed by quaternion wxyz."""
    rotation = Rotation.from_euler("XYZ", q[3:6]).as_matrix().T
    return np.r_[
        -rotation @ q[:3], Rotation.from_matrix(rotation).as_quat(scalar_first=True)
    ]


@dataclass
class GraspInitialization:
    hand_qpos: np.ndarray
    object_points: np.ndarray
    object_normals: np.ndarray  # outward, in the fixed object frame
    raw: dict  # unmodified original outputs, in the source world frame
    metadata: dict
    contact_forces: np.ndarray | None = (
        None  # optional magnitudes in N, annotation order
    )


def matching_diagnostics(
    loss_dist, loss_normal, *, normal_threshold, position_threshold
):
    """Per-contact upstream residuals, not an average hiding one bad finger.

    Upstream compares *squared* position and normal residuals to its YAML
    thresholds, despite the units given in that YAML's comments.
    """
    distance = np.asarray(loss_dist)
    normal = np.asarray(loss_normal)
    valid = (normal.max(-1) < normal_threshold) & (
        distance.max(-1) < position_threshold
    )
    return valid, dict(
        normal_squared_residual=normal.tolist(),
        normal_angle_deg=np.degrees(np.arccos(np.clip(1 - normal / 2, -1, 1))).tolist(),
        gap_mm=(1000 * np.sqrt(np.maximum(distance, 0))).tolist(),
        normal_squared_residual_max=normal.max(-1).tolist(),
        matching_position_cost=(1000 * distance.mean(-1)).tolist(),
        matching_normal_cost=normal.mean(-1).tolist(),
        normal_squared_residual_threshold=normal_threshold,
        position_squared_residual_threshold_m2=position_threshold,
        loss_filter_valid=valid.tolist(),
    )


def template_contact_layout(hand, template_q):
    """Diagnose whether two annotated surfaces actually face one another."""
    hand.set(template_q)
    points = hand.points(hand.bodies, hand.local)
    normals = np.einsum(
        "nij,nj->ni", hand.data.xmat[hand.bodies].reshape(-1, 3, 3), hand.normals
    )
    # Row i points toward column j when this signed projection is positive.
    toward = np.einsum("ijk,ik->ij", points[None] - points[:, None], normals) * 1000
    return dict(
        body_names=[hand.model.body(int(b)).name for b in hand.bodies],
        normal_toward_other_contact_mm=toward.tolist(),
        normal_pair_dot=(normals @ normals.T).tolist(),
    )


def from_dexonomy_result(result, metadata=None):
    """Convert one unbatched original result; never query/project the surface.

    Original poses are xyz+wxyz. Both poses map local coordinates to world.
    Dexonomy obj_cn_w points inward; the forward model expects outward normals.
    Object dimensions have already been applied to obj_cp_w by the initializer.
    """
    raw = {key: np.asarray(value).copy() for key, value in result.items()}
    obj_pose, hand_pose = raw["obj_pose"], raw["grasp_pose"]
    obj_rot = Rotation.from_quat(obj_pose[3:], scalar_first=True)
    hand_rot = Rotation.from_quat(hand_pose[3:], scalar_first=True)
    rotation = obj_rot.inv() * hand_rot
    translation = obj_rot.inv().apply(hand_pose[:3] - obj_pose[:3])
    q = np.r_[translation, rotation.as_euler("XYZ"), raw["grasp_qpos"]]
    points = obj_rot.inv().apply(raw["obj_cp_w"] - obj_pose[:3])
    normals = -obj_rot.inv().apply(raw["obj_cn_w"])
    if (
        q.shape != (26,)
        or points.ndim != 2
        or points.shape[1] != 3
        or normals.shape != points.shape
    ):
        raise ValueError(
            "Dexonomy result must contain a Wuji pose (20 finger joints) and Nx3 object contacts"
        )
    if not all(np.isfinite(v).all() for v in (q, points, normals)):
        raise ValueError("Nonfinite Dexonomy initialization")
    info = dict(metadata or {})
    info.update(
        object_pose_in_hand_xyz_wxyz=pose_in_hand(q).tolist(),
        input_frame="world -> fixed object by inverse obj_pose; no reprojection",
        input_normal_convention="Dexonomy inward -> forward-model outward by sign reversal",
    )
    return GraspInitialization(q, points, normals, raw, info)


def upstream_modules(checkout):
    from retarget.initialization.dexonomy.op import gen_init
    from retarget.initialization.dexonomy.data import obj_loader, hand_loader
    from retarget.initialization.dexonomy.util import warp_util

    return gen_init, obj_loader, hand_loader, warp_util


def matching_mesh(surface):
    """Exact input triangles, with independent normal patches at sharp edges."""
    import trimesh

    vertices, inverse = np.unique(
        surface.triangles.reshape(-1, 3), axis=0, return_inverse=True
    )
    mesh = trimesh.Trimesh(
        vertices=vertices, faces=inverse.reshape(-1, 3), process=False
    )
    return trimesh.graph.smooth_shade(mesh, angle=np.pi / 4, facet_minarea=None)


def sharp_edge_clearance(surface, points, device):
    """GPU distance to >45 degree mesh edges, for input selection only."""
    import torch

    triangles = surface.triangles
    vertices, inverse = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    neighbors = {}
    for face_id, face in enumerate(faces):
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            neighbors.setdefault(tuple(sorted((int(a), int(b)))), []).append(face_id)
    edges = [
        edge
        for edge, ids in neighbors.items()
        if len(ids) == 1
        or any(
            surface.normals[ids[0]] @ surface.normals[i] < np.cos(np.pi / 4)
            for i in ids[1:]
        )
    ]
    p = torch.as_tensor(points, device=device, dtype=torch.float64)
    if not edges:
        return torch.full(p.shape[:-1], torch.inf, device=device, dtype=p.dtype)
    segment = torch.as_tensor(vertices[np.array(edges)], device=device, dtype=p.dtype)
    clearance = torch.full(p.shape[:-1], torch.inf, device=device, dtype=p.dtype)
    for chunk in segment.split(1024):
        a = chunk[:, 0]
        v = chunk[:, 1] - a
        offset = p[..., None, :] - a
        t = ((offset * v).sum(-1) / v.square().sum(-1)).clamp(0.0, 1.0)
        clearance = torch.minimum(
            clearance,
            torch.linalg.vector_norm(offset - t[..., None] * v, dim=-1).amin(-1),
        )
    return clearance


def sample_original_poses(obj_loader, mesh, points, inplane, seed):
    """Run the original sampler with deterministic Trimesh random streams.

    Recent Trimesh samplers use default_rng, independent of np.random.seed.
    Only supply their RNG seed; upstream sampling/pose math is unchanged.
    """
    import trimesh

    even, ordinary = trimesh.sample.sample_surface_even, trimesh.sample.sample_surface

    def seeded_even(*a, **kw):
        if kw.get("seed") is None:
            kw["seed"] = seed
        return even(*a, **kw)

    def seeded_ordinary(*a, **kw):
        if kw.get("seed") is None:
            kw["seed"] = seed + 1
        return ordinary(*a, **kw)

    try:
        trimesh.sample.sample_surface_even = seeded_even
        trimesh.sample.sample_surface = seeded_ordinary
        return obj_loader.sample_init_pose(mesh, points, inplane)
    finally:
        trimesh.sample.sample_surface_even = even
        trimesh.sample.sample_surface = ordinary


def prepare_matcher_inputs(hand, template_q, args, rng, modules):
    """Pack the mapped Wuji template for the original loader and object sampler."""
    import torch

    _, obj_loader, hand_loader, _ = modules
    device = torch.device(getattr(args, "init_device", "cuda"))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "The original Dexonomy Warp query requires CUDA; no rewritten CPU fallback is used"
        )

    # The template root is the identity. Its finger angles and local contacts
    # are unchanged; the upstream matcher solves the entire relative root pose.
    canonical_q = template_q.copy()
    canonical_q[:6] = 0
    hand.set(canonical_q)
    points = hand.points(hand.bodies, hand.local)
    normals = np.einsum(
        "nij,nj->ni", hand.data.xmat[hand.bodies].reshape(-1, 3, 3), hand.normals
    )
    template = dict(
        grasp_qpos=np.r_[np.zeros(3), 1.0, np.zeros(3), canonical_q[6:]].astype(
            np.float32
        ),
        hand_cpn_w=np.c_[points, normals].astype(np.float32),
        n_evo=np.int64(0),
    )

    # _load_data uses skeleton FK only to supply later collision filters. No
    # skeleton filtering is invoked here, so an empty skeleton is sufficient.
    loader = object.__new__(hand_loader.HandTemplateLoader)
    loader.skt_data, loader.skt_bid = np.empty((0, 6), np.float32), np.empty(0, int)

    def forward_kinematics(finger_q):
        hand.set(np.r_[np.zeros(6), finger_q])
        return hand.data.xmat.reshape(-1, 3, 3).astype(
            np.float32
        ), hand.data.xpos.astype(np.float32)

    fk = SimpleNamespace(forward_kinematics=forward_kinematics)
    seed = int(rng.integers(2**31))
    previous_state = np.random.get_state()
    try:
        np.random.seed(seed)
        with tempfile.TemporaryDirectory(prefix="wuji-dexonomy-") as directory:
            path = Path(directory) / "mapped_template.npy"
            np.save(path, template)
            prepared = loader._load_data(fk, str(path))
    finally:
        np.random.set_state(previous_state)

    # Reconstruct the exact centered, scaled input mesh. Preserve sharp normal
    # boundaries: welding a cylinder's cap rim to its side tilts side vertex
    # normals toward the cap (even far along a large triangle). The upstream
    # query interpolates these normals, so that artifact changes its objective.
    mesh = matching_mesh(hand.surface)
    rotations, translations = sample_original_poses(
        obj_loader, mesh, args.init_points, args.init_inplane, seed
    )
    count = len(translations)

    def tensor(value):
        return torch.as_tensor(
            np.asarray(value).copy(), dtype=torch.float32, device=device
        )

    tmpl_sample = {
        key: tensor(prepared[key])[None, None]
        .expand(1, count, *np.shape(prepared[key]))
        .contiguous()
        for key in (
            "grasp_qpos",
            "rot_c2h",
            "trans_c2h",
            "hand_cpn_c",
            "hand_skt_h",
            "n_evo",
        )
    }
    obj_sample = dict(
        rot_o2c_init=tensor(rotations)[None],
        trans_o2c_init=tensor(translations)[None],
        obj_scale=tensor([[1.0, 1.0, 1.0]]),
        pose_w2so=tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]),
        ext_center=tensor([mesh.center_mass]),
        ext_wrench=tensor(np.zeros((1, 6))),
    )
    return tmpl_sample, obj_sample, mesh, seed


def pose_seed_inputs(tmpl, obj, hand_qpos):
    """Express a fitted relative hand pose in the upstream contact frame.

    This only adds an initializer candidate. HandObjMatcher itself performs
    the surface query and constructs every exported pose/contact field.
    """
    import torch

    q = np.asarray(hand_qpos)
    if q.shape[-1:] != (26,) or q.ndim not in (1, 2) or not np.isfinite(q).all():
        raise ValueError("Pose seeds must contain 26 finite Wuji coordinates each")
    q = q.reshape(-1, 26)
    seeded_template = {
        key: value[:, :1].expand(1, len(q), *value.shape[2:]).clone()
        for key, value in tmpl.items()
    }
    seeded_object = {key: value.clone() for key, value in obj.items()}
    device = obj["obj_scale"].device
    rotation = torch.as_tensor(
        Rotation.from_euler("XYZ", q[:, 3:6]).as_matrix(),
        device=device,
        dtype=torch.float32,
    )
    contact_rotation = rotation @ tmpl["rot_c2h"][0, 0].T
    translation = torch.as_tensor(q[:, :3], device=device, dtype=torch.float32)
    contact_translation = (
        translation - (contact_rotation @ tmpl["trans_c2h"][0, 0, :, None])[..., 0]
    )
    seeded_object["rot_o2c_init"] = contact_rotation[None]
    seeded_object["trans_o2c_init"] = contact_translation[None]
    return seeded_template, seeded_object


def global_alignment(hand, template_q, args, rng, *, pose_seed=None):
    """Run original sampling + contact frame + HandObjMatcher, then bridge outputs."""
    begun = time.perf_counter()
    import torch

    modules = upstream_modules(args.dexonomy_root)
    gen_init, _, _, warp_util = modules
    tmpl, obj, mesh, seed = prepare_matcher_inputs(hand, template_q, args, rng, modules)
    device = obj["obj_scale"].device
    environment = warp_util.WarpCollisionEnv(device)
    environment.load(None, [("wuji-static-object", mesh)])
    matcher = gen_init.HandObjMatcher(
        SimpleNamespace(opt_step=args.init_iterations), device, environment
    )
    result = matcher.forward(tmpl, obj)
    sampled_count = len(result["loss_dist"])
    extra_poses = [] if pose_seed is None else [pose_seed]
    if hasattr(hand, "wrist_pose_reference"):
        from retarget.task.grasp.wrist_prior import reference_pose_seeds

        extra_poses.extend(reference_pose_seeds(hand, template_q))
    if hasattr(hand, "object_orientation"):
        from retarget.env.grasp.orientation import aligned_disk_seeds

        extra_poses.extend(aligned_disk_seeds(hand, template_q))
    if extra_poses:
        # Rerandomizing all root poses loses the already fitted envelope.
        # One upstream evaluation (opt_step=1) preserves it as an additional
        # candidate. Original random sampling/optimization remains unchanged.
        seeded_template, seeded_object = pose_seed_inputs(
            tmpl, obj, np.asarray(extra_poses)
        )
        evaluator = gen_init.HandObjMatcher(
            SimpleNamespace(opt_step=1), device, environment
        )
        seeded = evaluator.forward(seeded_template, seeded_object)
        result = {
            key: torch.cat((value, seeded[key]), dim=0) for key, value in result.items()
        }
    arrays = {key: value.detach().cpu().numpy().copy() for key, value in result.items()}
    objective = (1000 * arrays["loss_dist"] + arrays["loss_normal"]).mean(-1)
    selection_objective = objective.copy()
    if hasattr(hand, "wrist_pose_reference"):
        from retarget.task.grasp.wrist_prior import prior_diagnostics

        rotations = Rotation.from_quat(
            arrays["obj_pose"][:, 3:], scalar_first=True
        ).inv() * Rotation.from_quat(arrays["grasp_pose"][:, 3:], scalar_first=True)
        configurations = np.zeros((len(objective), 26))
        configurations[:, :3] = (
            Rotation.from_quat(arrays["obj_pose"][:, 3:], scalar_first=True)
            .inv()
            .apply(arrays["grasp_pose"][:, :3] - arrays["obj_pose"][:, :3])
        )
        configurations[:, 3:6] = rotations.as_euler("XYZ")
        selection_objective += prior_diagnostics(hand, configurations)["cost"] / 1.0e5
    import yaml

    loss_config = yaml.safe_load(
        (Path(args.dexonomy_root) / "dexonomy/config/op/init.yaml").read_text()
    )["filter"]["loss"]
    matching_valid, _ = matching_diagnostics(
        arrays["loss_dist"],
        arrays["loss_normal"],
        normal_threshold=loss_config["normal_thre"],
        position_threshold=loss_config["pos_thre"],
    )
    layout = template_contact_layout(hand, template_q)
    # Matching loss alone can prefer a cylinder penetrating an unannotated
    # finger/palm. Validate the complete mapped hand cloud before choosing an
    # input pose. Original matcher poses/contacts are never edited/reprojected.
    hand.set(np.r_[np.zeros(6), arrays["grasp_qpos"][0]])
    cloud = torch.as_tensor(
        hand.points(hand.all_bodies, hand.all_local), dtype=torch.float32, device=device
    )
    from retarget.env.grasp.audit import GraspAudit

    geometry = GraspAudit(hand.surface, str(device))
    shape_first = hasattr(hand, "ik_reference_qpos")
    forbidden = None
    if shape_first:
        from retarget.env.grasp.regions import (
            hand_forbidden_mask,
            FORBIDDEN_CLEARANCE_M,
        )

        forbidden = torch.as_tensor(
            hand_forbidden_mask(hand, hand.contact_reference_normals), device=device
        )
    penetration = []
    forbidden_gaps = []
    with torch.no_grad():
        poses = result["grasp_pose"].detach()
        for start in range(0, len(poses), 64):
            pose = poses[start : start + 64]
            rotation = gen_init.torch_quaternion_to_matrix(pose[:, 3:])
            points = cloud[None] @ rotation.transpose(-1, -2) + pose[
                :, :3, None
            ].transpose(-1, -2)
            _, _, distance = geometry.query(points.reshape(-1, 3))
            depth = (-distance.reshape(len(pose), -1)).clamp(min=0.0)
            penetration.append(depth.amax(-1).cpu().numpy() * 1000)
            if shape_first:
                forbidden_gaps.append(
                    distance.reshape(len(pose), -1)[:, forbidden].amin(-1).cpu().numpy()
                    * 1000
                )
    penetration = np.concatenate(penetration)
    tolerance = getattr(args, "penetration_tolerance_mm", 2.0)
    from retarget.env.grasp.input_geometry import WujiSelfCollision

    capsules = WujiSelfCollision(hand.model, str(device))
    # All original pose candidates have identical joints: root transforms
    # cannot change inter-finger collision. Check this once, on CUDA.
    with torch.no_grad():
        self_penetration = (
            float(
                capsules.depths(
                    capsules.fk.tensor(np.r_[np.zeros(6), arrays["grasp_qpos"][0]])
                ).max()
            )
            * 1000
        )
    contact_tolerance = getattr(args, "contact_tolerance_mm", 2.0)
    local_contact_valid = arrays["loss_dist"].max(-1) <= (contact_tolerance / 1000) ** 2
    valid = (
        matching_valid
        & local_contact_valid
        & (penetration <= tolerance)
        & (self_penetration <= tolerance)
    )
    if shape_first:
        forbidden_gaps = np.concatenate(forbidden_gaps)
        valid = (
            (penetration <= tolerance)
            & (self_penetration <= tolerance)
            & (forbidden_gaps >= FORBIDDEN_CLEARANCE_M * 1000 - 1.0e-3)
        )
    orientation = None
    if hasattr(hand, "object_orientation"):
        from retarget.env.grasp.orientation import orientation_angles

        rotations = Rotation.from_quat(
            arrays["obj_pose"][:, 3:], scalar_first=True
        ).inv() * Rotation.from_quat(arrays["grasp_pose"][:, 3:], scalar_first=True)
        configurations = np.zeros((len(objective), 26))
        configurations[:, 3:6] = rotations.as_euler("XYZ")
        orientation = orientation_angles(hand, configurations)
        valid &= orientation <= hand.object_orientation["maximum_angle_deg"] + 1.0e-5
    # Prefer candidates passing *both* checks. Keep a failed candidate only for
    # downstream refinement/diagnostics, explicitly marked as invalid. Unlike
    # upstream valid_index_padding, padding never implies feasibility here.
    order = np.lexsort(
        (selection_objective, np.maximum(penetration - tolerance, 0.0), ~valid)
    )
    if shape_first:
        order = np.lexsort(
            (
                selection_objective,
                np.maximum(penetration - tolerance, 0.0),
                np.maximum(FORBIDDEN_CLEARANCE_M * 1000 - forbidden_gaps, 0.0),
                ~valid,
            )
        )
    if orientation is not None:
        # Infeasible fallback diagnostics must also preserve the disk's face
        # direction, rather than hiding a geometric failure with a tilted disk.
        order = np.lexsort(
            (
                selection_objective,
                np.maximum(penetration - tolerance, 0.0),
                np.maximum(FORBIDDEN_CLEARANCE_M * 1000 - forbidden_gaps, 0.0)
                if shape_first
                else np.zeros(len(objective)),
                np.maximum(
                    orientation - hand.object_orientation["maximum_angle_deg"], 0.0
                ),
                ~valid,
            )
        )
    loads = {}
    edge_clearance = None
    if getattr(args, "load_conditioning", False) and not shape_first:
        from retarget.env.grasp.load import load_command

        object_pose = result["obj_pose"].detach()
        object_rotation = gen_init.torch_quaternion_to_matrix(object_pose[:, 3:])
        object_points = torch.bmm(
            result["obj_cp_w"].detach() - object_pose[:, None, :3], object_rotation
        )
        edge_clearance = (
            sharp_edge_clearance(hand.surface, object_points, device)
            .amin(-1)
            .cpu()
            .numpy()
        )
        # A seam has several valid nearest normals. Passing the upstream
        # normal test on one side of an exact rim is not a robust input.
        # Reject its 0.1 mm neighborhood, retaining matching-loss ordering on
        # regular patches. No original surface point or pose is moved.
        valid &= edge_clearance >= MIN_SHARP_EDGE_CLEARANCE_M
        # A point just below a cylinder rim can change its nearest normal to
        # the end cap as preload forms. Favor interior patches without moving
        # a single original matcher pose or surface anchor.
        if getattr(args, "init_selection", "matching") == "interior":
            order = np.lexsort(
                (
                    objective,
                    np.maximum(0.003 - edge_clearance, 0.0),
                    np.maximum(penetration - tolerance, 0.0),
                    ~valid,
                )
            )
        else:
            order = np.lexsort(
                (objective, np.maximum(penetration - tolerance, 0.0), ~valid)
            )
        radius = float(
            np.linalg.norm(hand.surface.triangles.reshape(-1, 3), axis=-1).max()
        )
        accepted = []
        # Keep the original matcher results intact. Prefer original candidates
        # whose load can also be realized by these bounded position servos.
        candidates = order[valid[order]]
        if len(candidates):
            from retarget.env.grasp.load import load_commands_batch

            inputs = [
                from_dexonomy_result(
                    {key: value[index] for key, value in arrays.items()}
                )
                for index in candidates
            ]
            commands, infos = load_commands_batch(
                hand.model,
                hand.bodies,
                hand.local,
                np.stack([guess.hand_qpos for guess in inputs]),
                np.stack([guess.object_points for guess in inputs]),
                np.stack([guess.object_normals for guess in inputs]),
                device=str(device),
                mass=args.object_mass,
                friction=args.friction,
                radius=radius,
            )
            for index, command, info in zip(candidates, commands, infos):
                info["command"] = command.tolist()
                loads[int(index)] = info
                if info["feasible"] and len(accepted) < args.starts:
                    accepted.append(int(index))
        order = np.array(accepted + [int(i) for i in order if i not in accepted])
    selected = order[: args.starts]
    elapsed = time.perf_counter() - begun
    metadata = dict(
        method="Dexonomy_original_HandObjMatcher",
        implementation="retained_Dexonomy_input_initializer",
        source_files=[
            "dexonomy/data/obj_loader.py:sample_init_pose",
            "dexonomy/data/hand_loader.py:HandTemplateLoader._load_data",
            "dexonomy/op/gen_init.py:HandObjMatcher.forward",
            "dexonomy/util/warp_util.py:WarpCollisionEnv.query_point",
        ],
        fixed_hand_joint_angles=True,
        sampled_poses=len(objective),
        random_sampled_poses=sampled_count,
        fitted_pose_seed_candidates=int(pose_seed is not None),
        orientation_seed_candidates=(48 if hasattr(hand, "object_orientation") else 0),
        wrist_reference_seed_candidates=(
            7 if hasattr(hand, "wrist_pose_reference") else 0
        ),
        iterations=args.init_iterations,
        device=str(device),
        seconds=elapsed,
        contact_frame_seed=seed,
        object_surface_seed=seed,
        candidate_selection=(
            "shape-first geometry and dorsal clearance; contact matching and preload are optional"
            if shape_first
            else "upstream per-contact loss validity and GPU whole-hand penetration; "
            "load-conditioned inputs reject sharp-edge clearance <0.1 mm and require actuator-feasible preload; "
            "interior policy additionally prefers 3 mm clearance; "
            "invalid fallback is diagnostic only; original poses and contacts are not edited"
        ),
        geometry_valid_candidates=int(
            np.sum((penetration <= tolerance) & (self_penetration <= tolerance))
        ),
        matching_valid_candidates=int(matching_valid.sum()),
        initialization_valid_candidates=int(valid.sum()),
        candidate_minimum_max_normal_squared_residual=float(
            arrays["loss_normal"].max(-1).min()
        ),
        template_contact_layout=layout,
        penetration_tolerance_mm=tolerance,
        contact_tolerance_mm=contact_tolerance,
        contacts_optional=shape_first,
        initialization_policy="shape_first" if shape_first else "contact_matching",
        initial_self_penetration_mm=self_penetration,
        template_input="Shadow contacts mapped to Wuji query surface; Wuji IK finger pose",
        mesh_normal_policy="split patches at 45 degree sharp edges; triangle geometry unchanged",
        object_scale_already_baked_into_mesh=True,
    )
    metadata.update(
        load_checked_candidates=len(loads),
        load_conditioning_required=bool(
            getattr(args, "load_conditioning", False) and not shape_first
        ),
        load_conditioning_requested=bool(getattr(args, "load_conditioning", False)),
        sharp_edge_clearance_threshold_mm=(
            MIN_SHARP_EDGE_CLEARANCE_M * 1000 if edge_clearance is not None else None
        ),
        sharp_edge_valid_candidates=(
            int(np.sum(edge_clearance >= MIN_SHARP_EDGE_CLEARANCE_M))
            if edge_clearance is not None
            else None
        ),
        candidate_policy=getattr(args, "init_selection", "matching"),
        load_feasible_candidates=sum(info["feasible"] for info in loads.values()),
    )
    return [
        from_dexonomy_result(
            {key: value[index] for key, value in arrays.items()},
            dict(
                metadata,
                candidate_index=int(index),
                objective=float(objective[index]),
                selection_objective=float(selection_objective[index]),
                initial_penetration_mm=float(penetration[index]),
                **(
                    dict(
                        forbidden_region_min_gap_mm=float(forbidden_gaps[index]),
                        forbidden_region_clearance_mm=FORBIDDEN_CLEARANCE_M * 1000,
                    )
                    if shape_first
                    else {}
                ),
                initialization_valid=bool(
                    valid[index]
                    and (
                        shape_first
                        or not getattr(args, "load_conditioning", False)
                        or loads.get(int(index), {}).get("feasible", False)
                    )
                ),
                load_conditioning=loads.get(int(index)),
                minimum_sharp_edge_clearance_mm=(
                    None
                    if edge_clearance is None or not np.isfinite(edge_clearance[index])
                    else float(edge_clearance[index] * 1000)
                ),
                matching=matching_diagnostics(
                    arrays["loss_dist"][index],
                    arrays["loss_normal"][index],
                    normal_threshold=loss_config["normal_thre"],
                    position_threshold=loss_config["pos_thre"],
                )[1],
                initialization_geometry_valid=bool(
                    penetration[index] <= tolerance and self_penetration <= tolerance
                ),
            ),
        )
        for index in selected
    ]
