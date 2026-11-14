"""Legged IK gallery: floating-base humanoids and a quadruped, balance-aware."""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import viser
import warp as wp
from robot_descriptions.loaders.yourdfpy import load_robot_description

from robokit.assets.robots.humanoids import berkeley_humanoid_lite, fourier_gr3, unitree_g1, unitree_h1_2
from robokit.helpers.ik import IKContext, IKSolver, IKSolverConfig
from robokit.lie.se3 import se3_compose
from robokit.opt.lm_optimizer import LMOptimizerConfig
from robokit.opt.population_solver import PopulationSolverConfig, Stage
from robokit.opt.sampler import RobertsSamplerConfig
from robokit.robo import Robot, RobotState
from robokit.terms.dense.com_position_task import ComPositionTask
from robokit.terms.dense.configuration_task import ConfigurationTask
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.position_task import PositionTask
from robokit.terms.dense.rotation_task import RotationTask
from robokit.terms.dense.smoothness_task import SmoothnessTask
from robokit.utils.visualize_utils import ViserBatchUrdf
from robokit.utils.warp_utils import stack, wp_vec7


# Legged gallery fields:
# - links: task frames, each with one draggable gizmo.
# - foot_links: feet pinned at z=0.
# - foot_ori_weight: flat humanoid soles; zero for quadruped point feet.
ROBOT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "g1 (humanoid)": {
        "kind": "path",
        "ref": str(unitree_g1.URDF_PATH),
        "links": ["left_rubber_hand_link", "right_rubber_hand_link"],
        "foot_links": ["left_ankle_roll_link", "right_ankle_roll_link"],
        "foot_ori_weight": 2.0,
    },
    "h1_2 (humanoid)": {
        "kind": "path",
        "ref": str(unitree_h1_2.URDF_PATH),
        "links": ["left_wrist_yaw_link", "right_wrist_yaw_link"],
        "foot_links": ["left_ankle_roll_link", "right_ankle_roll_link"],
        "foot_ori_weight": 2.0,
    },
    "gr3 (humanoid)": {
        "kind": "path",
        "ref": str(fourier_gr3.URDF_PATH),
        "links": ["left_end_effector_link", "right_end_effector_link"],
        "foot_links": ["left_foot_roll_link", "right_foot_roll_link"],
        "foot_ori_weight": 2.0,
    },
    "bhl (humanoid)": {
        "kind": "path",
        "ref": str(berkeley_humanoid_lite.URDF_PATH),
        "links": ["arm_left_hand_link", "arm_right_hand_link"],
        "foot_links": ["leg_left_ankle_roll", "leg_right_ankle_roll"],
        "foot_ori_weight": 2.0,
    },
    # Quadruped: task one front paw; the other three feet pin the body so that leg reaches.
    "go2 (quad)": {
        "kind": "urdf",
        "ref": "go2_description",
        "links": ["FL_foot"],
        "foot_links": ["FR_foot", "RL_foot", "RR_foot"],
        "foot_ori_weight": 0.0,
    },
}

GRID_COLS = 6
GRID_SPACING = 1.5

# balance terms held fixed at build time (vector weights don't map to a single set_weight scalar)
FOOT_POS_WEIGHT = 5.0
COM_WEIGHT = [1.0, 1.0, 0.0]
BASE_UPRIGHT_WEIGHT = [0.0, 0.0, 0.0, 100.0, 100.0, 0.0]  # lock base roll/pitch; free xyz+yaw

# (gui label, term name in IKSolverConfig, default, min, max, step)
WEIGHT_SLIDERS: List[Tuple[str, str, float, float, float, float]] = [
    ("Position", "position_task_0", 1.5, 0.0, 20.0, 0.1),
    ("Rotation", "rotation_task_0", 0.3, 0.0, 20.0, 0.1),
    ("PositionLimit", "position_limit_0", 50.0, 0.0, 200.0, 0.5),
    ("Rest", "configuration_task_0", 0.05, 0.0, 5.0, 0.05),  # anchors null-space joints to the q=0 home pose
    ("Smoothness", "smoothness_task_0", 0.03, 0.0, 5.0, 0.01),
]


def load_robot(spec: Dict[str, Any]) -> Robot:
    kind, ref = spec["kind"], spec["ref"]
    src: Any = load_robot_description(ref) if kind == "urdf" else ref
    return Robot.load(src, load_meshes=True)


@dataclass
class Scene:
    robot_name: str
    batch_size: int
    robot: Robot
    ik: IKSolver
    batch_urdf: ViserBatchUrdf
    transform_controls: List[Any]
    T_world_base_grid: np.ndarray
    grid_se3: wp.array
    per_robot_offset: np.ndarray  # (batch, 3) deterministic per-instance target offset
    rest_state: RobotState
    out_state: RobotState
    prev_state: Optional[RobotState] = None


def build_scene(
    server: viser.ViserServer,
    robot_name: str,
    batch_size: int,
    weights: Dict[str, float],
    device: str,
    prior: Optional[Scene],
) -> Scene:
    if prior is not None:
        prior.batch_urdf.remove()
        for ctrl in prior.transform_controls:
            ctrl.remove()

    spec = ROBOT_CONFIGS[robot_name]
    robot = load_robot(spec)
    links = spec["links"]
    foot_indices = [robot.spec.link_names.index(n) for n in spec["foot_links"]]

    # Lift the base so the lowest link rests at z=0: probe FK, raise by its depth, re-run FK for lifted targets
    q0 = wp.zeros((1, robot.spec.num_actuated_joints), dtype=wp.float32, device=device)
    probe = robot.forward_kinematics(robot.state(q=q0))
    base_height = -float(probe.T_world_link.numpy()[0, :, 2].min())
    home_base_np = np.array([[0.0, 0.0, base_height, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    home_base = wp.from_numpy(home_base_np, dtype=wp_vec7, device=device)
    home_state = robot.forward_kinematics(robot.state(q=q0, T_world_base=home_base))
    foot_targets = stack([home_state.get_T_world_link(i) for i in foot_indices], axis=1)
    com_target = robot.compute_center_of_mass(home_state).com_world.numpy()[0]

    # --- IK ---
    config = (
        IKSolverConfig(
            enable_T_world_base=True,
            solver=PopulationSolverConfig(
                stages=[
                    Stage(10, 64, LMOptimizerConfig(lm_lambda=10.0)),
                    Stage(16, 4, LMOptimizerConfig(lm_lambda=1.0)),
                    Stage(0, 1),
                ],
                cuda_graph_mode="full",
            ),
            sampler=RobertsSamplerConfig(keep_first=True, scale=0.2),
        )
        .add(
            PositionTask(
                frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=weights["position_task_0"]
            )
        )
        .add(
            RotationTask(
                frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=weights["rotation_task_0"]
            )
        )
        .add(PositionTask(robot, foot_indices, foot_targets, weight=FOOT_POS_WEIGHT, fixed_target=True))
        .add(RotationTask(robot, foot_indices, foot_targets, weight=spec["foot_ori_weight"], fixed_target=True))
        .add(ComPositionTask(robot, target_com_position=com_target, weight=COM_WEIGHT))
        .add(PositionLimit(weight=weights["position_limit_0"]))
        .add(
            ConfigurationTask(
                rest_q=IKContext.REST_Q,
                T_world_base_rest=IKContext.REST_BASE,
                weight=weights["configuration_task_0"],
                base_weight=BASE_UPRIGHT_WEIGHT,
            )
        )
        .add(SmoothnessTask(prev_var=IKContext.PREVIOUS_STATE, weight=weights["smoothness_task_0"], base_weight=0.0))
    )
    ik = IKSolver(config, robot=robot, target_links=links, device=device)
    ik.warmup(batch_size=batch_size)

    # Grid is XY-only (floor lift lives in the base height).
    cols = min(GRID_COLS, batch_size)
    num_rows = (batch_size + cols - 1) // cols
    grid = np.zeros((batch_size, 7), dtype=np.float32)
    grid[:, 3] = 1.0
    for i in range(batch_size):
        row, col = i // cols, i % cols
        grid[i, 0] = -col * GRID_SPACING
        grid[i, 1] = (row - (num_rows - 1) / 2) * GRID_SPACING

    # deterministic per-instance target offset (like 03_batch) so the instances differ; position only
    per_robot_offset = np.zeros((batch_size, 3), dtype=np.float32)
    if batch_size > 1:
        idx = np.arange(batch_size)
        per_robot_offset[:, 0] = 0.05 * (idx % cols - (cols - 1) / 2.0)
        per_robot_offset[:, 1] = 0.05 * (idx // cols - (num_rows - 1) / 2.0)

    T_world_link = home_state.T_world_link.numpy()[0]  # (num_links, 7), base lifted

    # --- viewer ---
    # One draggable gizmo per tasked frame is the target (like 01_basic); no extra target axes.
    transform_controls: List[Any] = []
    for link, link_idx in zip(links, ik.target_link_indices):
        node = link.replace("/", "_")
        transform_controls.append(
            server.scene.add_transform_controls(
                f"/target/{node}", scale=0.15, position=T_world_link[link_idx, :3], wxyz=T_world_link[link_idx, 3:7]
            )
        )

    batch_urdf = ViserBatchUrdf(target=server, robot=robot, batch_size=batch_size, root_node_name="/robots")

    # Floating-base states: rest/init base sit at the lifted home pose; out_state holds the solve's base.
    home_base_tiled = np.tile(home_base_np, (batch_size, 1))
    identity_tiled = np.tile([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], (batch_size, 1)).astype(np.float32)
    rest_state = robot.state(
        q=wp.zeros((batch_size, ik.num_joints), dtype=wp.float32, device=device),
        T_world_base=wp.from_numpy(home_base_tiled, dtype=wp_vec7, device=device),
    )
    out_state = robot.state(
        q=wp.empty((batch_size, ik.num_joints), dtype=wp.float32, device=device),  # type: ignore[arg-type]
        T_world_base=wp.from_numpy(identity_tiled, dtype=wp_vec7, device=device),
    )

    return Scene(
        robot_name=robot_name,
        batch_size=batch_size,
        robot=robot,
        ik=ik,
        batch_urdf=batch_urdf,
        transform_controls=transform_controls,
        T_world_base_grid=grid,
        grid_se3=wp.from_numpy(grid, dtype=wp_vec7, device=device),
        per_robot_offset=per_robot_offset,
        rest_state=rest_state,
        out_state=out_state,
        prev_state=None,
    )


def main():
    wp.init()
    device = "cuda" if wp.is_cuda_available() else "cpu"

    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=20, height=20, cell_size=1.0)

    robot_dropdown = server.gui.add_dropdown("Robot", options=list(ROBOT_CONFIGS), initial_value="g1 (humanoid)")
    instances_slider = server.gui.add_slider("Instances", min=1, max=16, step=1, initial_value=1)
    weight_handles: Dict[str, Any] = {}
    with server.gui.add_folder("Cost weights"):
        for label, name, default, lo, hi, step in WEIGHT_SLIDERS:
            weight_handles[name] = server.gui.add_slider(
                f"w: {label}", min=lo, max=hi, step=step, initial_value=default
            )
    # seed spread (live): joint range as a fraction of travel, base range in meters on free base axes
    sample_range_slider = server.gui.add_slider("Sample range", min=0.0, max=1.0, step=0.01, initial_value=1.0)
    base_sample_range_slider = server.gui.add_slider(
        "Base sample range", min=0.0, max=0.5, step=0.01, initial_value=0.1
    )
    timing = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    scene: Optional[Scene] = None

    def push_weight(name: str):
        # Push one slider into the live IK only when it moves (not every frame).
        if scene is not None:
            scene.ik.set_weight(name, float(weight_handles[name].value))

    for name, handle in weight_handles.items():
        handle.on_update(lambda _, n=name: push_weight(n))

    while True:
        # Rebuild whenever the live selection differs from what's built (loop reads widgets directly).
        robot_name = robot_dropdown.value
        batch_size = int(instances_slider.value)
        if scene is None or robot_name != scene.robot_name or batch_size != scene.batch_size:
            weights = {name: float(handle.value) for name, handle in weight_handles.items()}
            scene = build_scene(server, robot_name, batch_size, weights, device, scene)

        assert scene is not None
        start = time.time()

        # --- IK: solve ---
        # Weights come from slider callbacks, not every frame; gizmos use the lifted frame plus a per-instance offset
        positions: List[np.ndarray] = []
        wxyzs: List[np.ndarray] = []
        for ctrl in scene.transform_controls:
            wxyz = np.asarray(ctrl.wxyz, dtype=np.float32)
            wxyz /= np.linalg.norm(wxyz) + 1e-12
            positions.append(np.asarray(ctrl.position, dtype=np.float32)[None] + scene.per_robot_offset)
            wxyzs.append(np.tile(wxyz, (scene.batch_size, 1)))

        init_state = scene.prev_state if scene.prev_state is not None else scene.rest_state
        scene.ik.solve_numpy(
            np.stack([np.concatenate([p, q], axis=-1) for p, q in zip(positions, wxyzs)], axis=-2),
            init_state=init_state,
            rest_state=scene.rest_state,
            prev_state=scene.prev_state,
            out_state=scene.out_state,
            init_sample_range=float(sample_range_slider.value),
            base_init_sample_range=(base_sample_range_slider.value, base_sample_range_slider.value, 0.0),
        )
        wp.synchronize()

        elapsed_ms = (time.time() - start) * 1000.0
        timing.value = 0.99 * timing.value + 0.01 * elapsed_ms

        # --- viewer: render ---
        # Compose each grid cell with the solved floating base.
        render_base = se3_compose(scene.grid_se3, scene.out_state.T_world_base)
        q_np = scene.out_state.q.numpy()
        scene.batch_urdf.update_cfg(q_np, T_world_base=render_base.numpy())

        scene.prev_state = scene.robot.state(
            q=wp.clone(scene.out_state.q),
            T_world_base=wp.clone(scene.out_state.T_world_base),
        )

        time.sleep(0.001)


if __name__ == "__main__":
    main()
