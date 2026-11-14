"""Humanoid retargeting gallery: pick a dataset, sequence, and robot preset; retarget and scrub the result."""

import argparse
import importlib
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import viser
import warp as wp
import yourdfpy
from viser.extras import ViserUrdf

from robokit.assets import fetch
from robokit.assets.body_models import get_body_models_dir
from robokit.helpers.humanoid_retargeting import HumanoidRetargeterOnline, HumanoidRetargeterOnlineConfig
from robokit.helpers.humanoid_retargeting.loaders import get_smplx_motion, load_smplx_file


PRESETS = [
    "g1_aligned",
    "g1_collide",
    "gr3_aligned",
    "h1_2_aligned",
    "bhl_aligned",
    "h1_aligned",
    "t1_aligned",
    "k1_aligned",
    "pm01_aligned",
    "talos_aligned",
    "adam_lite_aligned",
    "tienkung_aligned",
    "toddy_aligned",
]
DATASET = "sfu"  # other AMASS datasets appear once fetched
FPS = 30


def load_preset(name: str) -> HumanoidRetargeterOnlineConfig:
    module = importlib.import_module(f"robokit.helpers.humanoid_retargeting.presets.{name}")
    return getattr(module, name)


def list_sequences(folder: Path) -> List[Path]:
    return sorted(p for p in folder.rglob("*.npz") if not p.name.endswith("_stagei.npz") and p.name != "shape.npz")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=PRESETS, default=PRESETS[0])
    parser.add_argument("--sequence", default="", help="sequence file name, e.g. 0005_JumpRope001_poses.npz")
    args = parser.parse_args()

    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    body_models = str(get_body_models_dir())
    datasets_root = fetch([f"datasets/{DATASET}/**"]) / "datasets"
    datasets = sorted(d.name for d in datasets_root.iterdir() if d.is_dir() and d.name in ("accad", "dancedb", "sfu"))

    # --- viewer: source panel + playback controls ---
    server = viser.ViserServer()
    sequences = list_sequences(datasets_root / DATASET)
    with server.gui.add_folder("Source"):
        dataset_dd = server.gui.add_dropdown("Dataset", tuple(datasets), initial_value=DATASET)
        seq_names = tuple(s.name for s in sequences)
        seq_dd = server.gui.add_dropdown("Sequence", seq_names, initial_value=args.sequence or seq_names[0])
        robot_dd = server.gui.add_dropdown("Robot", tuple(PRESETS), initial_value=args.robot)
    frame_slider = server.gui.add_slider("frame", min=0, max=1, step=1, initial_value=0)
    playing = server.gui.add_checkbox("play", True)
    timing = server.gui.add_number("Retarget (ms/frame)", 0.001, disabled=True)
    human_y = server.gui.add_slider("Human Y Offset", min=-3.0, max=3.0, step=0.05, initial_value=1.0)
    server.scene.add_grid("/ground", width=10.0, height=10.0, cell_size=0.5)

    dirty = [True]

    def mark_dirty(_) -> None:
        dirty[0] = True

    # Callbacks only set flags; the main loop serializes GUI writes to avoid stale state from worker callbacks.
    dataset_dd.on_update(mark_dirty)
    seq_dd.on_update(mark_dirty)
    robot_dd.on_update(mark_dirty)

    helpers: Dict[str, HumanoidRetargeterOnline] = {}
    robot_viz = None
    robot_root = None
    human_mesh = None
    human_vertices = np.zeros((1, 1, 3), dtype=np.float32)
    qpos = np.zeros((1, 7), dtype=np.float32)
    num_frames = 1
    built = ("", "", "")
    while True:
        if dirty[0]:
            dirty[0] = False
            # Snapshot dropdowns before rebuilding; resolve stale choices there to avoid mismatched helpers.
            selection = (dataset_dd.value, seq_dd.value, robot_dd.value)
            seqs = list_sequences(datasets_root / selection[0])
            names = tuple(s.name for s in seqs)
            if names and selection != built:
                if seq_dd.options != names:
                    seq_dd.options = names  # dataset switched: rebuild the sequence list
                if selection[1] not in names:
                    seq_dd.value = names[0]  # heal a stale sequence from a mid-switch client
                    selection = (selection[0], names[0], selection[2])
                seq_path, robot_name = seqs[names.index(selection[1])], selection[2]

                # --- retargeting: precompute the whole clip with the selected preset ---
                config = load_preset(robot_name)
                assert config.urdf_path is not None
                motion = load_smplx_file(str(seq_path), body_models)
                if robot_name not in helpers:
                    helpers[robot_name] = HumanoidRetargeterOnline(config, device=device, robot=config.load_robot())
                helper = helpers[robot_name]
                T_world_human, fps = get_smplx_motion(motion, helper.human_joint_names, tgt_fps=FPS)
                human_heights = np.array([motion.human_height], dtype=np.float32)
                print(f"retargeting {seq_path.name} with {robot_name} ({len(T_world_human)} frames)...")
                helper.warmup(1)
                helper.reset()
                helper.solve_numpy(T_world_human[:1], human_heights=human_heights)
                helper.reset()
                t0 = time.time()
                qpos = np.stack(
                    [helper.solve_numpy(frame[None], human_heights=human_heights)[0] for frame in T_world_human]
                )
                timing.value = (time.time() - t0) * 1000 / len(T_world_human)

                # --- viewer: rebuild the scene for the new robot/clip ---
                verts = np.ascontiguousarray(motion.vertices).astype(np.float32)
                if fps < motion.mocap_frame_rate and motion.mocap_frame_rate % fps == 0:
                    verts = verts[:: int(motion.mocap_frame_rate // fps)]
                if robot_viz is not None:
                    server.scene.remove_by_name("/robot")
                if human_mesh is not None:
                    human_mesh.remove()
                robot_root = server.scene.add_frame("/robot", show_axes=False)
                robot_viz = ViserUrdf(server, yourdfpy.URDF.load(str(config.urdf_path)), root_node_name="/robot")
                human_vertices = verts
                human_mesh = server.scene.add_mesh_simple(
                    "/human_mesh",
                    vertices=human_vertices[0] + np.array([0.0, human_y.value, 0.0], dtype=np.float32),
                    faces=motion.faces.astype(np.int32),
                    opacity=0.5,
                    color=(150, 200, 255),
                )
                num_frames = len(T_world_human)
                frame_slider.value = 0
                frame_slider.max = num_frames - 1
                built = selection

        # --- viewer: playback (clamp: the slider may hold a stale value from a longer clip) ---
        idx = min(frame_slider.value, num_frames - 1)
        if playing.value:
            idx = (idx + 1) % num_frames
            frame_slider.value = idx
        if robot_root is not None and robot_viz is not None and human_mesh is not None:
            with server.atomic():
                robot_root.position = qpos[idx, :3]
                robot_root.wxyz = qpos[idx, 3:7]
                robot_viz.update_cfg(qpos[idx, 7:])
                human_mesh.vertices = human_vertices[min(idx, len(human_vertices) - 1)] + np.array(
                    [0.0, human_y.value, 0.0], dtype=np.float32
                )
        time.sleep(1.0 / FPS)


if __name__ == "__main__":
    main()
