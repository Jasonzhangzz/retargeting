"""Terrain contact retargeting example: mocap climbing clip -> G1 over scaled boxes (OmniRetarget-style)."""

import glob
import time

import numpy as np
import trimesh
import viser
import warp as wp

from robokit.assets.datasets import climb
from robokit.geom import MeshGeom, PlaneGeom, WarpScene
from robokit.helpers.humanoid_retargeting import HumanoidRetargeterOnline, build_interaction_mesh_frame
from robokit.helpers.humanoid_retargeting.loaders import load_mocap_clip
from robokit.helpers.humanoid_retargeting.presets.g1_mocap import G1_HEIGHT, g1_mocap
from robokit.utils.visualize_utils import ViserBatchUrdf
from robokit.utils.warp_utils import wp_vec7


CLIP = "mocap_climb_seq_0"
FPS = 30.0


def main():
    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"

    # --- assets: fetch only this climbing sequence (mocap .npy + its box terrain) ---
    ref = climb.sequence_dir(CLIP)
    preset = g1_mocap
    robot = preset.load_robot(load_meshes=True)
    T_world_human, _, interaction_scale = load_mocap_clip(
        glob.glob(str(ref / "*.npy"))[0], preset.human_joint_names, robot_height=G1_HEIGHT
    )
    terrain_path = str(ref / "multi_boxes.obj")
    terrain = trimesh.load(terrain_path, force="mesh")
    assert isinstance(terrain, trimesh.Trimesh)
    terrain.apply_scale(interaction_scale)
    interaction_points, _ = trimesh.sample.sample_surface_even(terrain, 100, seed=42)
    scene = (
        WarpScene(1, device)
        .add(MeshGeom([terrain], np.array([0, 1], dtype=np.int32)))
        .add(PlaneGeom(np.array([0, 1], dtype=np.int32)))
    )
    retargeter = HumanoidRetargeterOnline(preset, robot=robot, scene=scene, device=device)
    key_indices = [retargeter.human_joint_names.index(entry.human_joint) for entry in preset.link_mapping.values()]
    T_world_terrain = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    retargeter.interaction_task.set_frame(
        *build_interaction_mesh_frame(T_world_human[0, key_indices, :3], interaction_points, T_world_terrain)
    )
    retargeter.warmup(1)
    root = T_world_human[0, retargeter.human_joint_names.index(preset.human_root_name), :3]
    face_xy = interaction_points.mean(0)[:2]
    yaw = np.arctan2(face_xy[1] - root[1], face_xy[0] - root[0])
    T_world_base = np.array([*root, np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], dtype=np.float32)
    retargeter.reset(wp.from_numpy(T_world_base[None], dtype=wp_vec7, device=device))

    # --- retargeting: interaction-mesh solve; the terrain is static and welded into the scene (~1 min on GPU) ---
    t0 = time.time()
    qpos = []
    for frame in T_world_human:
        retargeter.interaction_task.set_frame(
            *build_interaction_mesh_frame(frame[key_indices, :3], interaction_points, T_world_terrain)
        )
        qpos.append(retargeter.solve_numpy(frame[None])[0])
    qpos = np.stack(qpos)
    print(f"solved {qpos.shape[0]} frames in {time.time() - t0:.1f}s")
    ndof = robot.spec.num_actuated_joints

    # --- viewer: robot + static terrain (scaled to the clip like the solver does) ---
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=10, height=10)
    robot_viz = ViserBatchUrdf(server, robot, batch_size=1, root_node_name="/robot")
    server.scene.add_mesh_trimesh("/terrain", terrain)
    frame_slider = server.gui.add_slider("frame", min=0, max=qpos.shape[0] - 1, step=1, initial_value=0)
    playing = server.gui.add_checkbox("play", True)
    while True:
        if playing.value:
            frame_slider.value = (frame_slider.value + 1) % qpos.shape[0]
        i = frame_slider.value
        robot_viz.update_cfg(qpos[i : i + 1, 7 : 7 + ndof], T_world_base=qpos[i : i + 1, :7])
        time.sleep(1.0 / FPS)


if __name__ == "__main__":
    main()
