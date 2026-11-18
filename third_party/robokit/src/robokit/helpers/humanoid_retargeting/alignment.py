"""Fit retargeting `scale_table` and per-link `position_offset` by closed-form weighted least squares (<1 ms).

Alignment procedure:
1. Pose neutral, betas=0 SMPL-X and the robot in the same physical configuration, grounded at Z-up z=0.
2. Match human keypoints to fixed robot FK, capturing stature differences and preserving ground contact.
3. Solve the linear `compute_targets` mapping for joint scales and link offsets, without iterations or learning rates.

Typical use (see `scripts/humanoid_retargeting/alignment/`)::

    pairs = pose_pairs(g1_alignment)
    aligned, report = align_config(g1, pairs, robot, body_model_path)
"""

import dataclasses
import pprint
from copy import deepcopy
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation as R

from robokit.helpers.humanoid_retargeting.config import HumanoidRetargeterOnlineConfig
from robokit.robo import Robot
from robokit.smplx import SMPLX_JOINT_NAMES, BodyModelSpecTensors, body_lbs_warp, from_amass_dict
from robokit.utils.warp_utils import wp_vec7


PoseJointOverrides = Dict[str, Dict[str, float]]
"""`{pose_name: {robot_joint: position}}`."""

PosePairs = Dict[str, Tuple[np.ndarray, Dict[str, float]]]
"""`{pose_name: ([63] SMPL-X body_pose axis-angle, {robot_joint: position})}`."""

# human joints whose mapped robot links get extra loss weight (they matter most downstream)
EE_HUMAN_JOINTS = ("left_wrist", "right_wrist", "left_foot", "right_foot")


def _bp(*edits: Tuple[int, List[float]]) -> np.ndarray:
    bp = np.zeros(63, dtype=np.float32)
    for start, vals in edits:
        bp[start : start + 3] = vals
    return bp


# SMPL-X pose indices: hips 0/3, knees 9/12, shoulders 45/48, elbows 51/54; presets add robot poses.
ALIGNMENT_POSES: Dict[str, np.ndarray] = {
    "T-pose": _bp(),
    "A-pose": _bp((45, [0, 0, -0.6]), (48, [0, 0, 0.6])),
    "Arms-down": _bp((45, [0, 0, -1.57]), (48, [0, 0, 1.57])),
    "Arms-forward": _bp((45, [0, -1.57, 0]), (48, [0, 1.57, 0])),
    "Squat": _bp(
        (0, [-1, 0, 0]), (3, [-1, 0, 0]), (9, [1.5, 0, 0]), (12, [1.5, 0, 0]), (45, [0, 0, -0.6]), (48, [0, 0, 0.6])
    ),
    "Arms-up": _bp((45, [0, 0, 0.6]), (48, [0, 0, -0.6])),
    "Half-T": _bp((45, [0, 0, 0.35]), (48, [0, 0, -0.35])),
    "Walking": _bp((0, [-0.5, 0, 0]), (3, [0.3, 0, 0]), (9, [0.8, 0, 0]), (45, [0, 0.4, -0.8]), (48, [0, 0.4, 0.8])),
    "Lunge": _bp((0, [-1.2, 0, 0]), (3, [0.4, 0, 0]), (9, [1.3, 0, 0]), (45, [0, 0, -0.6]), (48, [0, 0, 0.6])),
}

# Human skeleton edges over mapped joints; solve matches SCALED segments onto robot segments, preserving proportions.
_SKELETON_EDGES: Tuple[Tuple[str, str], ...] = (
    ("pelvis", "left_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_foot"),
    ("pelvis", "right_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_foot"),
    ("pelvis", "spine3"),
    ("spine3", "left_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("spine3", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
)

# Store (label, joint a, joint b, floor degrees); flag angle > floor + 15°. G1-derived floors are heuristic elsewhere.
BONE_PAIRS: Tuple[Tuple[str, str, str, float], ...] = (
    ("L uarm", "left_shoulder", "left_elbow", 26.0),
    ("L farm", "left_elbow", "left_wrist", 6.0),
    ("R uarm", "right_shoulder", "right_elbow", 19.0),
    ("R farm", "right_elbow", "right_wrist", 6.0),
    ("L thigh", "left_hip", "left_knee", 9.0),
    ("L shin", "left_knee", "left_foot", 9.0),
    ("R thigh", "right_hip", "right_knee", 9.0),
    ("R shin", "right_knee", "right_foot", 9.0),
)


@dataclasses.dataclass(frozen=True)
class HumanPose:
    """Posed SMPL-X alignment body in the runtime frame: Z-up, lowest mesh vertex on z=0."""

    positions: Dict[str, np.ndarray]
    """`{human_joint: [3]}` world joint positions."""
    quaternions: Dict[str, np.ndarray]
    """`{human_joint: [4]}` world joint rotations, wxyz."""
    vertices: np.ndarray
    """`[V, 3]` posed mesh vertices."""
    faces: np.ndarray
    """`[F, 3]` mesh faces."""


@dataclasses.dataclass(frozen=True)
class AlignmentPose:
    """One prepared pose pair: grounded human keypoints + grounded robot FK."""

    label: str
    human_positions: Dict[str, np.ndarray]
    human_quaternions: Dict[str, np.ndarray]
    robot_positions: Dict[str, np.ndarray]
    """`{robot_link: [3]}` world link positions with the foot soles on z=0."""
    robot_base: np.ndarray
    """`[7]` grounded floating-base transform (xyz + wxyz)."""


def human_pose(body_model_path: str, body_pose_aa: np.ndarray) -> HumanPose:
    """Pose the neutral SMPL-X body via Warp LBS, Z-up, pelvis at xy=0 so root scale can't absorb an xy offset."""
    from robokit.helpers.humanoid_retargeting.loaders.smplx import _get_spec

    spec = _get_spec(body_model_path, "neutral")
    poses = np.zeros((1, 165), dtype=np.float32)
    poses[0, :3] = (np.pi / 2.0, 0.0, 0.0)  # SMPL-X native Y-up -> runtime Z-up
    poses[0, 3:66] = np.asarray(body_pose_aa, dtype=np.float32)
    data = {"poses": poses, "trans": np.zeros((1, 3), np.float32), "betas": np.zeros(spec.num_betas, np.float32)}
    out = body_lbs_warp(
        BodyModelSpecTensors(spec=spec, device="cpu"), from_amass_dict(data, spec), return_landmarks=False
    )
    T = out["T_world_joint"].numpy()[0]  # [J, 7]: xyz + wxyz
    verts = out["vertices"].numpy()[0].copy()
    names = SMPLX_JOINT_NAMES[:22]
    shift = np.array([T[0, 0], T[0, 1], verts[:, 2].min()], dtype=np.float32)  # pelvis xy, lowest vertex z
    verts -= shift
    positions = {n: (T[i, :3] - shift).astype(np.float32) for i, n in enumerate(names)}
    quaternions = {n: T[i, 3:].astype(np.float32).copy() for i, n in enumerate(names)}
    return HumanPose(positions=positions, quaternions=quaternions, vertices=verts, faces=spec.faces)


_link_points_cache: Dict[int, Dict[str, np.ndarray]] = {}


def ground_robot(
    robot: Robot,
    root_xy: np.ndarray,
    root_quat_wxyz: np.ndarray,
    joint_overrides: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """FK once, shift robot so its lowest vertex sits at z=0 (needs `load_meshes=True`); returns `(links, base)`."""
    q = np.zeros(robot.spec.num_actuated_joints, dtype=np.float32)
    if joint_overrides:
        name_to_idx = {name: i for i, name in enumerate(robot.spec.actuated_joint_names)}
        unknown = [j for j in joint_overrides if j not in name_to_idx]
        if unknown:
            raise ValueError(f"joint_overrides not in this robot's actuated joints: {unknown}")
        for joint_name, value in joint_overrides.items():
            q[name_to_idx[joint_name]] = value

    base7 = np.concatenate([root_xy, [0.0], root_quat_wxyz]).astype(np.float32).reshape(1, 7)
    T_world_base = wp.from_numpy(base7, dtype=wp_vec7, device="cpu")
    state = robot.forward_kinematics(robot.state(q=q.reshape(1, -1), T_world_base=T_world_base))
    T = state.T_world_link.numpy()[0]  # (num_links, 7): xyz + wxyz

    if id(robot) not in _link_points_cache:  # per-link mesh vertices; collision scene, else visual
        points = {}
        for name in robot.spec.link_names:
            scene = robot.spec.link_collision_geometries[name]
            if scene.is_empty:
                scene = robot.spec.link_visual_geometries[name]
            if not scene.is_empty:
                points[name] = np.asarray(scene.dump(concatenate=True).vertices, dtype=np.float32)
        _link_points_cache[id(robot)] = points

    sole_z = np.inf
    for i, name in enumerate(robot.spec.link_names):
        verts = _link_points_cache[id(robot)].get(name)
        if verts is None:
            continue
        rot = R.from_quat(T[i, [4, 5, 6, 3]])  # wxyz -> xyzw
        sole_z = min(sole_z, float((rot.apply(verts) + T[i, :3])[:, 2].min()))

    dz = np.array([0.0, 0.0, -sole_z], dtype=np.float32)
    positions = {name: T[i, :3].astype(np.float32) + dz for i, name in enumerate(robot.spec.link_names)}
    return positions, base7[0] + np.concatenate([dz, np.zeros(4, np.float32)])


def prepare_pose(
    config: HumanoidRetargeterOnlineConfig,
    robot: Robot,
    human: HumanPose,
    joint_overrides: Optional[Dict[str, float]] = None,
    label: str = "",
) -> AlignmentPose:
    """Pair a posed human with the robot: base quat = human quat * link's `rotation_offset`, z grounds the sole."""
    root = config.human_root_name
    root_rot_offset = np.array([1, 0, 0, 0], dtype=np.float32)
    for m in config.link_mapping.values():
        if m.human_joint == root and np.linalg.norm(m.rotation_offset) > 1e-6:
            root_rot_offset = m.rotation_offset / np.linalg.norm(m.rotation_offset)
    # as_quat() is already non-canonical; the explicit kwarg predates scipy 1.11 and breaks on 3.8
    q_root = (
        R.from_quat(human.quaternions[root][[1, 2, 3, 0]]) * R.from_quat(root_rot_offset[[1, 2, 3, 0]])
    ).as_quat()[[3, 0, 1, 2]]
    robot_positions, base = ground_robot(robot, human.positions[root][:2], q_root.astype(np.float32), joint_overrides)
    return AlignmentPose(
        label=label,
        human_positions=human.positions,
        human_quaternions=human.quaternions,
        robot_positions=robot_positions,
        robot_base=base,
    )


def compute_targets(
    config: HumanoidRetargeterOnlineConfig,
    positions: Dict[str, np.ndarray],
    quaternions: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Mirror `compute_targets_kernel` in NumPy at `human_height == human_height_assumption` (height factor 1).

    Examples:
        >>> import numpy as np
        >>> from robokit.helpers.humanoid_retargeting import HumanoidRetargeterOnlineConfig, LinkMapping
        >>> config = HumanoidRetargeterOnlineConfig(
        ...     link_mapping={"torso": LinkMapping("spine", np.zeros(3), np.zeros(4))},
        ...     scale_table={"spine": 0.5, "pelvis": 1.0},
        ... )
        >>> pos = {"pelvis": np.zeros(3, np.float32), "spine": np.array([0.0, 0.0, 1.0], np.float32)}
        >>> quat = {"spine": np.array([1.0, 0, 0, 0], np.float32)}
        >>> compute_targets(config, pos, quat)["torso"].tolist()
        [0.0, 0.0, 0.5]
    """
    root = positions[config.human_root_name]
    root_scale = config.scale_table.get(config.human_root_name, 1.0)
    targets = {}
    for robot_link, m in config.link_mapping.items():
        if m.human_joint not in positions:
            continue
        scale = config.scale_table.get(m.human_joint, 1.0)
        pos = root * root_scale + (positions[m.human_joint] - root) * scale
        if np.linalg.norm(m.position_offset) > 1e-6:
            quat = quaternions[m.human_joint]
            rot = R.from_quat(quat[[1, 2, 3, 0]])
            if np.linalg.norm(m.rotation_offset) > 1e-6:
                off = m.rotation_offset / np.linalg.norm(m.rotation_offset)
                rot = rot * R.from_quat(off[[1, 2, 3, 0]])
            pos = pos + rot.apply(m.position_offset)
        targets[robot_link] = (pos - [0.0, 0.0, config.ground_height]).astype(np.float32)
    return targets


def bone_audit(pose: AlignmentPose, config: HumanoidRetargeterOnlineConfig) -> List[Tuple[str, float, float]]:
    """Human-vs-robot limb mismatch: `(label, angle_deg, floor_deg)` per bone; over `floor+15` flags a mismatch."""
    hj2rl = {m.human_joint: rl for rl, m in config.link_mapping.items()}
    out = []
    for label, a, b, floor in BONE_PAIRS:
        if a not in hj2rl or b not in hj2rl:
            continue
        hd = pose.human_positions[b] - pose.human_positions[a]
        rd = pose.robot_positions[hj2rl[b]] - pose.robot_positions[hj2rl[a]]
        cosang = np.dot(hd / np.linalg.norm(hd), rd / np.linalg.norm(rd))
        out.append((label, float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))), floor))
    return out


def _symmetric_pairs(names: Sequence[str]) -> List[Tuple[str, str]]:
    pairs = []
    for name in names:
        if "left" in name and name.replace("left", "right") in names:
            pairs.append((name, name.replace("left", "right")))
    return pairs


def solve_alignment(
    config: HumanoidRetargeterOnlineConfig,
    poses: Sequence[AlignmentPose],
    ee_weight: float = 3.0,
    bone_weight: float = 3.0,
    scale_reg: float = 0.01,
    offset_reg: float = 0.1,
    symmetry_reg: float = 0.5,
) -> HumanoidRetargeterOnlineConfig:
    """Fit scales and link offsets in one weighted `lstsq`; return a new config without modifying the input.

    Objective terms:
    - Keypoints: weighted robot/human position residuals.
    - Segments: `bone_weight` matches skeleton edges to robot links, preventing unreachable scale/offset trades.
    - Regularizers: offsets toward 0, scales toward 1, and left/right symmetry.

    Examples:
        >>> import numpy as np
        >>> from robokit.helpers.humanoid_retargeting import HumanoidRetargeterOnlineConfig, LinkMapping
        >>> config = HumanoidRetargeterOnlineConfig(
        ...     link_mapping={"torso": LinkMapping("spine", np.zeros(3), np.zeros(4))},
        ... )
        >>> human = {"pelvis": np.zeros(3, np.float32), "spine": np.array([0.0, 0.0, 1.0], np.float32)}
        >>> quat = {n: np.array([1.0, 0, 0, 0], np.float32) for n in human}
        >>> pose = AlignmentPose("T", human, quat, {"torso": np.array([0.0, 0.0, 0.5])}, np.zeros(7))
        >>> aligned = solve_alignment(config, [pose], offset_reg=1e6)  # pin offsets to 0
        >>> round(aligned.scale_table["spine"], 2)
        0.5
    """
    mapping = [(rl, m) for rl, m in config.link_mapping.items() if m.human_joint in poses[0].human_positions]
    links = [rl for rl, _ in mapping]
    joints = list(dict.fromkeys([m.human_joint for _, m in mapping] + [config.human_root_name]))
    n_cols = len(joints) + 3 * len(links)
    root_col = joints.index(config.human_root_name)

    position_weights, _ = config.task("target").weights_for([rl for rl, _ in mapping])
    raw = {rl: max(weight, 1.0) for (rl, _), weight in zip(mapping, position_weights)}
    mean_w = sum(raw.values()) / len(raw)
    weights = {rl: raw[rl] / mean_w * (ee_weight if m.human_joint in EE_HUMAN_JOINTS else 1.0) for rl, m in mapping}

    rows: List[np.ndarray] = []
    rhs: List[np.ndarray] = []

    def add(row: np.ndarray, b: np.ndarray, w: float):
        rows.append(np.sqrt(w) * row)
        rhs.append(np.sqrt(w) * b)

    joint_link = {}
    for rl, m in mapping:
        joint_link.setdefault(m.human_joint, rl)
    for pose in poses:
        root = pose.human_positions[config.human_root_name]
        link_rows: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for i, (rl, m) in enumerate(mapping):
            if rl not in pose.robot_positions:
                continue
            quat = pose.human_quaternions[m.human_joint]
            rot = R.from_quat(quat[[1, 2, 3, 0]])
            if np.linalg.norm(m.rotation_offset) > 1e-6:
                off = m.rotation_offset / np.linalg.norm(m.rotation_offset)
                rot = rot * R.from_quat(off[[1, 2, 3, 0]])
            row = np.zeros((3, n_cols))
            row[:, root_col] += root  # scaled root
            row[:, joints.index(m.human_joint)] += pose.human_positions[m.human_joint] - root  # scaled relative
            row[:, len(joints) + 3 * i : len(joints) + 3 * i + 3] = rot.as_matrix()  # rotated offset
            b = np.asarray(pose.robot_positions[rl], dtype=np.float64) + [0.0, 0.0, config.ground_height]
            link_rows[rl] = (row, b)
            add(row, b, weights[rl])
        for ja, jb in _SKELETON_EDGES:  # scaled human segment == robot link segment
            ra, rb = joint_link.get(ja), joint_link.get(jb)
            if ra in link_rows and rb in link_rows:
                add(link_rows[rb][0] - link_rows[ra][0], link_rows[rb][1] - link_rows[ra][1], bone_weight)

    for j in range(len(joints)):  # scale -> 1
        row = np.zeros((1, n_cols))
        row[0, j] = 1.0
        add(row, np.ones(1), scale_reg)
    for i in range(len(links)):  # offset -> 0
        row = np.zeros((3, n_cols))
        row[:, len(joints) + 3 * i : len(joints) + 3 * i + 3] = np.eye(3)
        add(row, np.zeros(3), offset_reg)
    for a, b in _symmetric_pairs(joints):  # scale_left == scale_right
        row = np.zeros((1, n_cols))
        row[0, joints.index(a)], row[0, joints.index(b)] = 1.0, -1.0
        add(row, np.zeros(1), symmetry_reg)
    for a, b in _symmetric_pairs(links):  # offset_left == -offset_right
        row = np.zeros((3, n_cols))
        row[:, len(joints) + 3 * links.index(a) : len(joints) + 3 * links.index(a) + 3] = np.eye(3)
        row[:, len(joints) + 3 * links.index(b) : len(joints) + 3 * links.index(b) + 3] = np.eye(3)
        add(row, np.zeros(3), symmetry_reg)

    x = np.linalg.lstsq(np.vstack(rows), np.concatenate(rhs), rcond=None)[0]
    scale_table = {name: float(x[j]) for j, name in enumerate(joints)}
    offsets = {rl: x[len(joints) + 3 * i : len(joints) + 3 * i + 3].astype(np.float32) for i, rl in enumerate(links)}
    link_mapping = {
        rl: dataclasses.replace(m, position_offset=offsets.get(rl, m.position_offset))
        for rl, m in config.link_mapping.items()
    }
    return dataclasses.replace(config, scale_table=scale_table, link_mapping=link_mapping)


def pose_error_mm(config: HumanoidRetargeterOnlineConfig, pose: AlignmentPose) -> float:
    """Mean target-vs-robot-link distance (mm) of one pose under a config's mapping."""
    targets = compute_targets(config, pose.human_positions, pose.human_quaternions)
    errs = [np.linalg.norm(t - pose.robot_positions[rl]) for rl, t in targets.items() if rl in pose.robot_positions]
    return float(np.mean(errs)) * 1000.0


def measure_height(body_model_path: str) -> float:
    """T-pose mesh height (m) of the alignment body - the `human_height_assumption` to export."""
    return float(np.ptp(human_pose(body_model_path, np.zeros(63, np.float32)).vertices[:, 2]))


def align_config(
    config: HumanoidRetargeterOnlineConfig,
    pairs: PosePairs,
    robot: Robot,
    body_model_path: str,
) -> Tuple[HumanoidRetargeterOnlineConfig, List[Tuple[str, float, float]]]:
    """One-call alignment: prepare pairs, solve, measure height; returns `(config, [(name, before_mm, after_mm)])`."""
    poses = [
        prepare_pose(config, robot, human_pose(body_model_path, bp), ov, label=name) for name, (bp, ov) in pairs.items()
    ]
    height = measure_height(body_model_path)
    aligned = dataclasses.replace(solve_alignment(config, poses), human_height_assumption=round(height, 4))
    report = [(p.label, pose_error_mm(config, p), pose_error_mm(aligned, p)) for p in poses]
    return aligned, report


def ik_seed_pairs(
    config: HumanoidRetargeterOnlineConfig,
    robot: Robot,
    body_model_path: str,
    pairs: PosePairs,
    device: str = "cuda:0",
) -> PosePairs:
    """Seed each pair's joints from a single-frame IK of its SMPL-X pose, using a static preset tracking all links."""
    from robokit.helpers.humanoid_retargeting.online import HumanoidRetargeterOnline
    from robokit.opt.lm_optimizer import LMOptimizerConfig
    from robokit.opt.population_solver import PopulationSolverConfig, Stage

    position_weights, _ = config.task("target").weights_for(list(config.link_mapping))
    tracking_weights = {rl: max(weight, 20.0) for rl, weight in zip(config.link_mapping, position_weights)}
    ik_config = dataclasses.replace(
        config,
        velocity_clamp_scale=0.0,
        solver=PopulationSolverConfig(
            stages=[
                Stage(10, 64, LMOptimizerConfig(lm_lambda=10.0)),
                Stage(20, 8, LMOptimizerConfig(lm_lambda=1.0)),
                Stage(40, 1, LMOptimizerConfig(lm_lambda=0.3)),
            ],
        ),
        terms=deepcopy(config.terms),
    )
    ik_config.task("target").position_weight = tracking_weights
    ik_config.task("smoothness").weight = 0.0
    ik_config.task("smoothness").base_weight = 0.0
    ik_config.task("rest").weight = 0.0
    ik_config.task("velocity_limit").weight = 0.0
    ik_config.task("position_limit").warmup_frames = 0
    helper = HumanoidRetargeterOnline(ik_config, device=device, robot=robot)
    helper.warmup(1)
    joint_names = list(robot.spec.actuated_joint_names)
    out = {}
    for name, (bp, _) in pairs.items():
        human = human_pose(body_model_path, bp)
        helper.reset()
        frame = np.stack([np.concatenate([human.positions[j], human.quaternions[j]]) for j in helper.human_joint_names])
        qpos = helper.solve_numpy(frame[None])[0]
        out[name] = (bp, {j: round(float(qpos[7 + i]), 4) for i, j in enumerate(joint_names)})
    return out


def pose_pairs(joint_overrides: PoseJointOverrides) -> PosePairs:
    """Combine shared human poses with a robot's joint overrides."""
    return {
        name: (body_pose.copy(), dict(joint_overrides.get(name, {}))) for name, body_pose in ALIGNMENT_POSES.items()
    }


def emit_alignment_preset(robot_name: str, pairs: PosePairs) -> str:
    """Render a robot's alignment pose joint overrides as Python source."""
    overrides = {
        name: {joint: round(float(value), 4) for joint, value in joints.items()} for name, (_, joints) in pairs.items()
    }
    return (
        f'"""{robot_name.upper()} alignment pose joint overrides."""\n\n'
        "from typing import Dict\n\n\n"
        f"{robot_name}: Dict[str, Dict[str, float]] = {pprint.pformat(overrides, width=118, sort_dicts=False)}\n"
    )


def unaligned(config: HumanoidRetargeterOnlineConfig) -> HumanoidRetargeterOnlineConfig:
    """`config` reset to identity mapping (scale 1, zero offsets) as baseline; `rotation_offset` unchanged."""
    link_mapping = {
        rl: dataclasses.replace(m, position_offset=np.zeros(3, dtype=np.float32))
        for rl, m in config.link_mapping.items()
    }
    return dataclasses.replace(config, scale_table={}, link_mapping=link_mapping)


def emit_preset(robot_name: str, config: HumanoidRetargeterOnlineConfig) -> str:
    """Render `<robot_name>_aligned.py` preset source from an aligned config; rest inherited from the base preset."""
    offsets = "".join(
        f'    "{rl}": np.array({[round(float(x), 4) for x in m.position_offset]}, dtype=np.float32),\n'
        for rl, m in config.link_mapping.items()
    )
    scales = "".join(f'        "{k}": {round(float(v), 4)},\n' for k, v in config.scale_table.items())
    return (
        f'"""{robot_name} aligned preset: learns scale_table + offsets from the base `{robot_name}` preset."""\n\n'
        "import dataclasses\n\n"
        "import numpy as np\n\n"
        f"from robokit.helpers.humanoid_retargeting.presets.{robot_name} import {robot_name}\n\n\n"
        f"_OFFSETS = {{\n{offsets}}}\n\n\n"
        f"{robot_name}_aligned = dataclasses.replace(\n"
        f"    {robot_name},\n"
        f"    human_height_assumption={round(float(config.human_height_assumption), 4)},\n"
        f"    scale_table={{\n{scales}    }},\n"
        "    link_mapping={\n"
        f"        link: dataclasses.replace({robot_name}.link_mapping[link], position_offset=offset)\n"
        "        for link, offset in _OFFSETS.items()\n"
        "    },\n"
        ")\n"
    )
