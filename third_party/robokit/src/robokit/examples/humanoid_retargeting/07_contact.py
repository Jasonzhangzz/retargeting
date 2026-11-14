"""Contact retargeting example: OMOMO human+box clip -> G1 with contacts preserved (OmniRetarget-style)."""

import time

import numpy as np
import trimesh
import viser
import warp as wp

from robokit.assets.datasets import omomo
from robokit.geom import MeshGeom, PlaneGeom, WarpScene
from robokit.helpers.humanoid_retargeting import HumanoidRetargeterOnline, build_interaction_mesh_frame
from robokit.helpers.humanoid_retargeting.loaders import load_omomo_clip
from robokit.helpers.humanoid_retargeting.presets.g1_smplh import G1_HEIGHT, g1_smplh
from robokit.utils.visualize_utils import ViserBatchUrdf
from robokit.utils.warp_utils import wp_vec7
from robokit.xform.numpy import quaternion_to_matrix


CLIP = "sub3_largebox_003"
FPS = 30.0


def main():
    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"

    # --- assets: fetch only this clip, its object mesh, and the height table ---
    preset = g1_smplh
    robot = preset.load_robot(load_meshes=True)
    T_world_human, T_world_object_vec7, _, interaction_scale = load_omomo_clip(
        str(omomo.clip_path(CLIP)),
        str(omomo.height_dict_path()),
        CLIP.split("_", maxsplit=1)[0],
        preset.human_joint_names,
        robot_height=G1_HEIGHT,
    )
    box_path = str(omomo.mesh_path(CLIP.split("_")[1]))
    box = trimesh.load(box_path, force="mesh")
    assert isinstance(box, trimesh.Trimesh)
    box.apply_scale(interaction_scale)
    interaction_points, _ = trimesh.sample.sample_surface_even(box, 100, seed=42)
    T_world_object = np.tile(np.eye(4, dtype=np.float32), (len(T_world_object_vec7), 1, 1))
    T_world_object[:, :3, :3] = quaternion_to_matrix(T_world_object_vec7[:, 3:])
    T_world_object[:, :3, 3] = T_world_object_vec7[:, :3]
    identity = wp.from_numpy(np.eye(4, dtype=np.float32)[None], dtype=wp.mat44, device=device)
    object_geom = MeshGeom([box], np.array([0, 1], dtype=np.int32), poses=identity)
    scene = WarpScene(1, device).add(object_geom).add(PlaneGeom(np.array([0, 1], dtype=np.int32)))
    retargeter = HumanoidRetargeterOnline(preset, robot=robot, scene=scene, device=device)
    key_indices = [retargeter.human_joint_names.index(entry.human_joint) for entry in preset.link_mapping.values()]
    object_geom.update(poses=wp.from_numpy(T_world_object[:1], dtype=wp.mat44, device=device))
    retargeter.interaction_task.set_frame(
        *build_interaction_mesh_frame(T_world_human[0, key_indices, :3], interaction_points, T_world_object_vec7[0])
    )
    retargeter.warmup(1)
    root = T_world_human[0, retargeter.human_joint_names.index(preset.human_root_name), :3]
    yaw = np.arctan2(T_world_object_vec7[0, 1] - root[1], T_world_object_vec7[0, 0] - root[0])
    T_world_base = np.array([*root, np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], dtype=np.float32)
    retargeter.reset(wp.from_numpy(T_world_base[None], dtype=wp_vec7, device=device))

    # --- retargeting: interaction-mesh solve over the whole clip (~20 s on a modern GPU) ---
    t0 = time.time()
    qpos = []
    for i in range(len(T_world_human)):
        object_geom.update(poses=wp.from_numpy(T_world_object[i : i + 1], dtype=wp.mat44, device=device))
        retargeter.interaction_task.set_frame(
            *build_interaction_mesh_frame(T_world_human[i, key_indices, :3], interaction_points, T_world_object_vec7[i])
        )
        qpos.append(retargeter.solve_numpy(T_world_human[i : i + 1])[0])
    qpos = np.stack(qpos)
    print(f"solved {qpos.shape[0]} frames in {time.time() - t0:.1f}s")
    ndof = robot.spec.num_actuated_joints

    # --- viewer: robot + moving box playback ---
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=6, height=6)
    robot_viz = ViserBatchUrdf(server, robot, batch_size=1, root_node_name="/robot")
    box_handle = server.scene.add_mesh_trimesh("/object", box)
    frame_slider = server.gui.add_slider("frame", min=0, max=qpos.shape[0] - 1, step=1, initial_value=0)
    playing = server.gui.add_checkbox("play", True)
    while True:
        if playing.value:
            frame_slider.value = (frame_slider.value + 1) % qpos.shape[0]
        i = frame_slider.value
        with server.atomic():
            robot_viz.update_cfg(qpos[i : i + 1, 7 : 7 + ndof], T_world_base=qpos[i : i + 1, :7])
            box_handle.position = T_world_object_vec7[i, :3]
            box_handle.wxyz = T_world_object_vec7[i, 3:]
        time.sleep(1.0 / FPS)


if __name__ == "__main__":
    main()
