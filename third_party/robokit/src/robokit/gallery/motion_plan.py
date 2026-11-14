# pyright: reportMissingImports=false
"""Interactive motion planning gallery."""

import dataclasses
import os
import threading
import time
import weakref
import xml.etree.ElementTree as ET
from copy import deepcopy
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, cast


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import trimesh
import tyro
import viser
import warp as wp
import yourdfpy
from viser.extras import ViserUrdf

from robokit.assets import fetch
from robokit.geom import BoxGeom, MeshGeom, WarpScene
from robokit.helpers.motion_plan import MotionPlanner, presets
from robokit.robo import Robot
from robokit.utils.warp_utils import wp_device_type


wp.config.quiet = True
wp.config.verify_cuda = False
wp.init()

_BOX_CAPACITY = 512


def load_panda_urdf_fixed(finger_q: float = 0.0) -> yourdfpy.URDF:
    root = fetch(["robots/robot_description/arms/franka_panda/**"])
    urdf_path = root / "robots/robot_description/arms/franka_panda/panda_v2.urdf"
    franka_root = urdf_path.parent / "franka_description"
    xml_tree = ET.parse(urdf_path)
    for mesh in xml_tree.findall(".//mesh"):
        filename = mesh.get("filename")
        assert filename is not None
        mesh_path = franka_root.parent / filename
        if not mesh_path.exists():
            matches = sorted(franka_root.glob(f"meshes/**/*{Path(filename).stem}.*"))
            assert matches
            mesh_path = matches[0]
        mesh.set("filename", str(mesh_path))
    for joint in xml_tree.findall('.//joint[@type="prismatic"]'):
        joint.set("type", "fixed")
        origin = joint.find("origin")
        axis = joint.find("axis")
        if finger_q and origin is not None and axis is not None:
            xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            xyz += finger_q * np.fromstring(axis.get("xyz", "1 0 0"), sep=" ")
            origin.set("xyz", " ".join(f"{v:.6f}" for v in xyz))
        for tag in ("axis", "limit", "dynamics", "mimic"):
            child = joint.find(tag)
            if child is not None:
                joint.remove(child)
    urdf = yourdfpy.URDF.load(StringIO(ET.tostring(xml_tree.getroot(), encoding="unicode")))
    assert urdf.validate()
    return urdf


def obb_to_trimesh(dims: List[float], T_world_obstacle: List[float]) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=dims)
    matrix = trimesh.transformations.quaternion_matrix(T_world_obstacle[3:7])
    matrix[:3, 3] = T_world_obstacle[:3]
    mesh.apply_transform(matrix)
    return mesh


def cylinder_to_trimesh(radius: float, height: float, T_world_obstacle: List[float]) -> trimesh.Trimesh:
    mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=16)
    matrix = trimesh.transformations.quaternion_matrix(T_world_obstacle[3:7])
    matrix[:3, 3] = T_world_obstacle[:3]
    mesh.apply_transform(matrix)
    return mesh


def mesh_to_trimesh(mesh_dict: dict) -> trimesh.Trimesh:
    mesh = trimesh.Trimesh(
        vertices=np.array(mesh_dict["vertices"], dtype=np.float64),
        faces=np.array(mesh_dict["faces"], dtype=np.int32),
    )
    if "pose" in mesh_dict:
        pose = mesh_dict["pose"]
        matrix = trimesh.transformations.quaternion_matrix(pose[3:7])
        matrix[:3, 3] = pose[:3]
        mesh.apply_transform(matrix)
    return mesh


def obstacles_to_warp_meshes(obstacles: dict, device: wp_device_type) -> List[wp.Mesh]:
    meshes = [obb_to_trimesh(x["dims"], x["pose"]) for x in obstacles.get("cuboid", {}).values()]
    meshes += [cylinder_to_trimesh(x["radius"], x["height"], x["pose"]) for x in obstacles.get("cylinder", {}).values()]
    meshes += [mesh_to_trimesh(x) for x in obstacles.get("mesh", {}).values()]
    return [
        wp.Mesh(
            points=wp.array(np.asarray(mesh.vertices, dtype=np.float32), dtype=wp.vec3, device=device),
            indices=wp.array(np.asarray(mesh.faces, dtype=np.int32).ravel(), dtype=int, device=device),
        )
        for mesh in meshes
    ]


def obstacles_to_box_data(obstacles: dict) -> tuple:
    if obstacles.get("mesh"):
        raise ValueError("Primitive collision does not support mesh obstacles.")
    half_extents = []
    T_world_obstacle = []
    for kind in ("cuboid", "cylinder"):
        for obstacle in obstacles.get(kind, {}).values():
            if kind == "cuboid":
                half_extents.append(np.asarray(obstacle["dims"], dtype=np.float32) / 2.0)
            else:
                half_extents.append(
                    np.array([obstacle["radius"], obstacle["radius"], obstacle["height"] / 2.0], dtype=np.float32)
                )
            matrix = trimesh.transformations.quaternion_matrix(obstacle["pose"][3:7])
            matrix[:3, 3] = obstacle["pose"][:3]
            T_world_obstacle.append(matrix.astype(np.float32))
    n = len(half_extents)
    half = np.stack(half_extents) if n else np.zeros((0, 3), dtype=np.float32)
    matrices = np.stack(T_world_obstacle) if n else np.zeros((0, 4, 4), dtype=np.float32)
    return half, matrices, np.array([0, n], dtype=np.int32)


def merge_warp_meshes(meshes: List[wp.Mesh], device: wp_device_type) -> wp.Mesh:
    vertices = []
    faces = []
    offset = 0
    for mesh in meshes:
        points = np.asarray(mesh.points.numpy(), dtype=np.float32)
        vertices.append(points)
        faces.append(np.asarray(mesh.indices.numpy(), dtype=np.int32).reshape(-1, 3) + offset)
        offset += len(points)
    return wp.Mesh(
        points=wp.array(np.concatenate(vertices), dtype=wp.vec3, device=device),
        indices=wp.array(np.concatenate(faces).ravel(), dtype=int, device=device),
    )


def build_obstacle_scene(obstacles: dict, device: wp_device_type, use_box_collision: bool) -> tuple:
    meshes = obstacles_to_warp_meshes(obstacles, device)
    scene = WarpScene(1, device)
    if use_box_collision:
        half, T_world_obstacle, offsets = obstacles_to_box_data(obstacles)
        if len(half) == 0:
            box = BoxGeom(
                np.zeros((1, 3), dtype=np.float32),
                np.array([0, 1], dtype=np.int32),
                poses=wp.from_numpy(np.eye(4, dtype=np.float32)[None], dtype=wp.mat44, device=device),
                capacity=_BOX_CAPACITY,
            )
            scene.add(box)
            box.update(half, offsets, poses=wp.from_numpy(T_world_obstacle, dtype=wp.mat44, device=device))
        else:
            scene.add(
                BoxGeom(
                    half,
                    offsets,
                    poses=wp.from_numpy(T_world_obstacle, dtype=wp.mat44, device=device),
                    capacity=_BOX_CAPACITY,
                )
            )
    elif meshes:
        scene.add(MeshGeom([merge_warp_meshes(meshes, device)], np.array([0, 1], dtype=np.int32), enable_sdf=True))
    return scene, meshes


def update_obstacle_scene(scene: WarpScene, obstacles: dict, device: wp_device_type) -> List[wp.Mesh]:
    meshes = obstacles_to_warp_meshes(obstacles, device)
    if not scene.geoms:
        if meshes:
            raise ValueError("Cannot add obstacles to a scene built empty.")
        return meshes
    geom = scene.geoms[0]
    if isinstance(geom, BoxGeom):
        half, T_world_obstacle, offsets = obstacles_to_box_data(obstacles)
        geom.update(half, offsets, poses=wp.from_numpy(T_world_obstacle, dtype=wp.mat44, device=device))
    else:
        assert isinstance(geom, MeshGeom)
        merged = [merge_warp_meshes(meshes, device)] if meshes else []
        geom.update(merged, np.array([0, len(merged)], dtype=np.int32))
    return meshes


@dataclass
class PlanSummary:
    success: bool
    total_time: float
    position_error: float  # mm
    max_penetration_mm: float
    cspace_path_length: float
    max_jerk: float


def load_planner(
    device: wp_device_type,
    num_timesteps: int = 32,
    dt: float = 0.25,
    max_iter: int = 30,
    use_cuda_graph: bool = False,
    solver: str = "lm",
    num_candidates: int = 8,
    scene: Optional[WarpScene] = None,
    finger_q: float = 0.04,
) -> MotionPlanner:
    override = os.environ.get("MP_SPHERES")
    spheres_path = (
        Path(override)
        if override and override != "robokit"
        else fetch(["robots/collision_spheres/franka_panda/**"])
        / "robots/collision_spheres/franka_panda/collision_spheres.yaml"
    )
    robot = Robot.load(
        load_panda_urdf_fixed(finger_q=finger_q),
        load_collision_spheres=True,
        collision_spheres_path=spheres_path,
    )
    base = {
        "lm": presets.offline_pose_lm,
        "lbfgs": presets.offline_pose_lbfgs,
        "mppi": presets.online_pose_mppi,
    }[solver]
    if solver == "lbfgs":
        config = presets.build_offline_pose_goal_config(
            num_timesteps=num_timesteps,
            dt=dt,
            max_iter=max_iter,
            num_candidates=num_candidates,
            use_cuda_graph=use_cuda_graph,
        )
        return MotionPlanner(robot=robot, target_links="panda_hand", config=config, device=str(device), scene=scene)
    stage = base.stages[0]
    trajectory_optimizer = dataclasses.replace(
        stage,
        solver=dataclasses.replace(
            stage.solver,
            stages=[dataclasses.replace(stage.solver.stages[0], iters=max_iter, num_candidates=num_candidates)],
        ),
        terms=deepcopy(stage.terms),
    )
    if solver == "mppi":
        trajectory_optimizer.task("collision").sweep_steps = 0  # the gallery also supports mesh and empty scenes
    config = dataclasses.replace(
        base,
        num_timesteps=num_timesteps,
        dt=dt,
        cuda_graph="stages" if use_cuda_graph else "none",
        goal_ik=presets.build_offline_ik_config(num_candidates, "full" if use_cuda_graph else "none"),
        stages=(trajectory_optimizer,),
    )
    return MotionPlanner(robot=robot, target_links="panda_hand", config=config, device=str(device), scene=scene)


SOLVER_CONFIGS: "dict[str, dict]" = {
    "lm-1seed": {"solver": "lm", "num_candidates": 1},
    "lm-4seed": {"solver": "lm", "num_candidates": 4},
    "lbfgs-1seed": {"solver": "lbfgs", "num_candidates": 1, "max_iter": 100},
    "lbfgs-4seed": {"solver": "lbfgs", "num_candidates": 4, "max_iter": 80},
    "lbfgs-8seed": {"solver": "lbfgs", "num_candidates": 8, "max_iter": 80},
    "mppi-4seed": {"solver": "mppi", "num_candidates": 4, "max_iter": 50},
}
DATASET_NAMES = ("demo", "motion_benchmaker", "mpinets")


@dataclass
class Config:
    dataset: str = "demo"
    scene_name: str = ""
    problem_idx: int = 0
    port: int = 8080
    gpu: int = 0
    num_timesteps: int = 32
    dt: float = 0.25
    max_iter: int = 50
    num_viz_frames: int = 100
    fps: float = 30.0


class Gallery:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.server = viser.ViserServer(host="0.0.0.0", port=cfg.port)
        self.urdf = load_panda_urdf_fixed()

        self._datasets: Dict[str, Dict[str, list]] = {}
        self._planners: dict = {}
        self.q_traj: np.ndarray = np.zeros((1, 7), dtype=np.float32)
        self.robot_vis: ViserUrdf = ViserUrdf(self.server, self.urdf, root_node_name="/robot")
        self._lock = threading.Lock()
        self._loading = False
        self._suppress_reload = False

        self.cur_dataset = cfg.dataset
        problems = self._get_dataset(self.cur_dataset)
        scene_names = list(problems.keys())
        self.cur_scene = cfg.scene_name if cfg.scene_name else scene_names[0]
        self.cur_problem_idx = cfg.problem_idx

        self._build_gui(scene_names, len(problems[self.cur_scene]))
        self._reload_problem()

    def _get_dataset(self, name: str) -> "dict[str, list]":
        if name not in self._datasets:
            from robometrics.datasets import demo_raw, motion_benchmaker_raw, mpinets_raw

            fns = {"demo": demo_raw, "motion_benchmaker": motion_benchmaker_raw, "mpinets": mpinets_raw}
            self._datasets[name] = fns[name]()
        return self._datasets[name]

    def _run_solver(self, config_name: str, problem: dict) -> "tuple[np.ndarray, PlanSummary]":
        cfg = self.cfg
        preset = SOLVER_CONFIGS[config_name]
        use_box_collision = self.cur_dataset != "mpinets"
        state_key = (config_name, use_box_collision)
        if state_key not in self._planners:
            device = wp.get_device(f"cuda:{cfg.gpu}")
            scene, _ = build_obstacle_scene(problem["obstacles"], device, use_box_collision)
            planner = load_planner(
                device=device,
                num_timesteps=cfg.num_timesteps,
                dt=cfg.dt,
                max_iter=preset.get("max_iter", cfg.max_iter),
                use_cuda_graph=True,
                solver=preset["solver"],
                num_candidates=preset["num_candidates"],
                scene=scene,
            )
            self._planners[state_key] = (planner, scene)
        planner, scene = self._planners[state_key]
        start_q = np.asarray(problem["start"], dtype=np.float32)
        goal_position = np.asarray(problem["goal_pose"]["position_xyz"], dtype=np.float32)
        goal_wxyz = np.asarray(problem["goal_pose"]["quaternion_wxyz"], dtype=np.float32)
        update_obstacle_scene(scene, problem["obstacles"], planner.device)
        seed_q = None
        if problem.get("goal_ik") is not None and len(problem["goal_ik"]) > 0:
            seed_q = np.asarray(problem["goal_ik"][0], dtype=np.float32)

        start_time = time.perf_counter()
        result = planner.solve_offline_numpy(
            [start_q, {"panda_hand": np.concatenate([goal_position, goal_wxyz])}], seed_q=seed_q
        )
        solve_time = time.perf_counter() - start_time
        metrics = planner.compute_metrics_numpy(
            result.q_traj, T_world_target=np.concatenate([goal_position, goal_wxyz])[None, None]
        )
        q_traj = result.q_traj[0]
        assert result.motion_time is not None
        dt = float(result.motion_time[0]) / float(q_traj.shape[0] - 1)
        diffs = np.diff(q_traj, axis=0)
        jerk = np.diff(np.diff(diffs / dt, axis=0) / dt, axis=0) / dt
        return q_traj, PlanSummary(
            success=bool(metrics.success[0]),
            total_time=solve_time,
            position_error=float(metrics.position_error_m[0]) * 1000.0,
            max_penetration_mm=float(metrics.penetration_mm[0, 1:-1].max(initial=0.0)),
            cspace_path_length=float(np.sum(np.linalg.norm(diffs, axis=-1))),
            max_jerk=float(np.max(np.abs(jerk))) if jerk.size else 0.0,
        )

    def _build_gui(self, scene_names: list, num_problems: int):
        # Callbacks reach the gallery weakly: the server holds them and the gallery holds the server.
        gallery = weakref.ref(self)
        with self.server.gui.add_folder("Solver", expand_by_default=True):
            self.robokit_dd = self.server.gui.add_dropdown(
                "RoboKit", options=tuple(SOLVER_CONFIGS.keys()), initial_value="lbfgs-1seed"
            )

            @self.robokit_dd.on_update
            def _(_: viser.GuiEvent):
                cast(Gallery, gallery())._reload_problem()

        with self.server.gui.add_folder("Problem", expand_by_default=True):
            self.dataset_dd = self.server.gui.add_dropdown(
                "Dataset", options=DATASET_NAMES, initial_value=self.cur_dataset
            )
            self.scene_dd = self.server.gui.add_dropdown("Scene", options=scene_names, initial_value=self.cur_scene)
            prob_opts = tuple(str(i) for i in range(num_problems))
            self.problem_dd = self.server.gui.add_dropdown(
                "Problem", options=prob_opts, initial_value=str(self.cur_problem_idx)
            )
            self.status_text = self.server.gui.add_text("Status", initial_value="Loading...", disabled=True)

            @self.dataset_dd.on_update
            def _(event: viser.GuiEvent):
                self = cast(Gallery, gallery())
                self.cur_dataset = str(event.target.value)
                problems = self._get_dataset(self.cur_dataset)
                snames = list(problems.keys())
                self._suppress_reload = True
                self.scene_dd.options = snames
                self.cur_scene = snames[0]
                self.scene_dd.value = self.cur_scene
                n = len(problems[self.cur_scene])
                self.problem_dd.options = tuple(str(i) for i in range(n))
                self.cur_problem_idx = 0
                self.problem_dd.value = "0"
                self._suppress_reload = False
                self._reload_problem()

            @self.scene_dd.on_update
            def _(event: viser.GuiEvent):
                self = cast(Gallery, gallery())
                self.cur_scene = str(event.target.value)
                if self._suppress_reload:
                    return
                problems = self._get_dataset(self.cur_dataset)
                n = len(problems[self.cur_scene])
                self._suppress_reload = True
                self.problem_dd.options = tuple(str(i) for i in range(n))
                self.cur_problem_idx = 0
                self.problem_dd.value = "0"
                self._suppress_reload = False
                self._reload_problem()

            @self.problem_dd.on_update
            def _(event: viser.GuiEvent):
                self = cast(Gallery, gallery())
                self.cur_problem_idx = int(event.target.value)
                if not self._suppress_reload:
                    self._reload_problem()

        self.metrics_md = self.server.gui.add_markdown("**Metrics**: --")
        self.timestep_slider = self.server.gui.add_slider(
            "Timestep", min=0, max=self.cfg.num_viz_frames - 1, step=1, initial_value=0
        )
        self.playing_checkbox = self.server.gui.add_checkbox("Playing", initial_value=True)

    def _reload_problem(self):
        with self._lock:
            if self._loading:
                return
            self._loading = True

        config_name = str(self.robokit_dd.value)
        self.status_text.value = f"Loading {self.cur_dataset}/{self.cur_scene}/#{self.cur_problem_idx}..."
        problems = self._get_dataset(self.cur_dataset)
        scene_problems = problems[self.cur_scene]
        idx = min(self.cur_problem_idx, len(scene_problems) - 1)
        problem = scene_problems[idx]

        print(f"\n{self.cur_dataset}/{self.cur_scene} #{idx}  {config_name}...", end=" ", flush=True)
        q_traj, metrics = self._run_solver(config_name, problem)
        print(f"{'OK' if metrics.success else 'FAIL'} {metrics.total_time:.3f}s err={metrics.position_error:.1f}mm")

        num_viz_frames = self.cfg.num_viz_frames
        if q_traj.shape[0] != num_viz_frames:
            old_t = np.linspace(0, 1, q_traj.shape[0])
            new_t = np.linspace(0, 1, num_viz_frames)
            q_traj = np.stack([np.interp(new_t, old_t, q_traj[:, j]) for j in range(q_traj.shape[1])], axis=-1)

        with self.server.atomic():
            self.server.scene.remove_by_name("/obstacles")
            for name, cuboid in problem["obstacles"].get("cuboid", {}).items():
                mesh = obb_to_trimesh(cuboid["dims"], cuboid["pose"])
                mesh.visual.face_colors = (180, 180, 180, 255)
                self.server.scene.add_mesh_trimesh(f"/obstacles/{name}", mesh)
            for name, cyl in problem["obstacles"].get("cylinder", {}).items():
                mesh = cylinder_to_trimesh(cyl["radius"], cyl["height"], cyl["pose"])
                mesh.visual.face_colors = (180, 180, 180, 255)
                self.server.scene.add_mesh_trimesh(f"/obstacles/{name}", mesh)
            for name, mesh_obs in problem["obstacles"].get("mesh", {}).items():
                mesh = mesh_to_trimesh(mesh_obs)
                mesh.visual.face_colors = (180, 180, 180, 255)
                self.server.scene.add_mesh_trimesh(f"/obstacles/{name}", mesh)
            goal_pos = problem["goal_pose"]["position_xyz"]
            goal_wxyz = problem["goal_pose"]["quaternion_wxyz"]
            self.server.scene.add_frame(
                "/goal", position=tuple(goal_pos), wxyz=tuple(goal_wxyz), axes_length=0.08, axes_radius=0.004
            )
            self.robot_vis.update_cfg(q_traj[0])
            self.q_traj = q_traj

        self.metrics_md.content = (
            f"**{config_name}** - T: {metrics.total_time * 1000:.0f}ms, Pos: {metrics.position_error:.1f}mm, "
            f"Pen: {metrics.max_penetration_mm:.1f}mm, Path: {metrics.cspace_path_length:.2f}, "
            f"Jerk: {metrics.max_jerk:.1f}"
        )
        self.timestep_slider.value = 0
        self.status_text.value = f"{self.cur_dataset} / {self.cur_scene} / #{idx}"
        self._loading = False

    def run(self):
        num_viz_frames = self.cfg.num_viz_frames
        while True:
            with self._lock:
                if not self._loading:
                    if self.playing_checkbox.value:
                        self.timestep_slider.value = (self.timestep_slider.value + 1) % num_viz_frames
                    t = int(self.timestep_slider.value)
                    with self.server.atomic():
                        self.robot_vis.update_cfg(self.q_traj[t])
            time.sleep(1.0 / self.cfg.fps)


if __name__ == "__main__":
    Gallery(tyro.cli(Config)).run()
