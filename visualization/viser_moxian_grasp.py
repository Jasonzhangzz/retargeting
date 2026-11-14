#!/usr/bin/env python3
"""Viser viewer for posed Moxian hand meshes and Dexonomy object meshes."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import trimesh
import viser
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "viser_assets"
POSES = np.load(ASSETS / "poses_24_samples.npz", allow_pickle=False)
SAMPLES = [0, 1333, 2666, 3999]
NODE_ORDER = [
    "wrist", "thumb_metacarpal", "thumb_proximal", "thumb_distal", "thumb_tip",
    "index_metacarpal", "index_proximal", "index_intermediate", "index_distal", "index_tip",
    "middle_metacarpal", "middle_proximal", "middle_intermediate", "middle_distal", "middle_tip",
    "ring_metacarpal", "ring_proximal", "ring_intermediate", "ring_distal", "ring_tip",
    "little_metacarpal", "little_proximal", "little_intermediate", "little_distal", "little_tip",
]
EDGES = []
for chain in [
    ["wrist", "thumb_metacarpal", "thumb_proximal", "thumb_distal", "thumb_tip"],
    ["wrist", "index_metacarpal", "index_proximal", "index_intermediate", "index_distal", "index_tip"],
    ["wrist", "middle_metacarpal", "middle_proximal", "middle_intermediate", "middle_distal", "middle_tip"],
    ["wrist", "ring_metacarpal", "ring_proximal", "ring_intermediate", "ring_distal", "ring_tip"],
    ["wrist", "little_metacarpal", "little_proximal", "little_intermediate", "little_distal", "little_tip"],
]:
    EDGES.extend(zip(chain[:-1], chain[1:]))
NODE_INDEX = {name: i for i, name in enumerate(NODE_ORDER)}
LINKS = ["wrist", "thumb_metacarpal", "thumb_proximal", "thumb_distal"] + [
    f"{finger}_{part}" for finger in ("index", "middle", "ring", "little")
    for part in ("metacarpal", "proximal", "intermediate", "distal")
]
COLORS = [(180, 225, 255), (255, 130, 205), (190, 145, 255), (255, 210, 115), (145, 240, 170)]


def xyzw_to_wxyz(q):
    return np.asarray([q[3], q[0], q[1], q[2]], dtype=np.float32)


def quat_mul_xyzw(a, b):
    ax, ay, az, aw = np.asarray(a)
    bx, by, bz, bw = np.asarray(b)
    return np.asarray([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], dtype=np.float32)


def mesh_arrays(path: Path):
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return np.asarray(mesh.vertices, np.float32), np.asarray(mesh.faces, np.int32)


def main():
    hand_root = Path(".venv-moxian-ik/lib/python3.13/site-packages/qnbot_sdk/glove/_private/algorithms/skeleton/qnbot-hand-dynamic_openxr_hand-0.3.4-linux-x86_64/share/qnbot_hand/hands/dynamic_openxr_hand/resources/meshes/right/visual")
    if not hand_root.exists():
        hand_root = Path(__file__).resolve().parents[1] / hand_root
    server = viser.ViserServer(host="0.0.0.0", port=8766, label="Moxian Grasp Mesh Viewer")
    server.scene.add_grid("/grid", width=0.5, height=0.5, cell_size=0.01, plane="xy")
    wrist_frame = server.scene.add_frame("/wrist_axes", axes_length=0.04, axes_radius=0.0015, origin_radius=0.003)
    server.gui.add_markdown("## Moxian Grasp\n24_Tip_Pinch · posed hand mesh + Dexonomy object mesh")
    sample_gui = server.gui.add_slider("Sample", 0, len(SAMPLES) - 1, 1, 0)
    show_hand = server.gui.add_checkbox("Show hand mesh", True)
    show_object = server.gui.add_checkbox("Show object mesh", True)
    show_skeleton = server.gui.add_checkbox("Show skeleton", True)
    show_labels = server.gui.add_checkbox("Show labels", False)
    reset = server.gui.add_button("Reset camera")

    hand_handles = {}
    for link in LINKS:
        path = hand_root / f"{link}.obj"
        if not path.exists():
            continue
        vertices, faces = mesh_arrays(path)
        hand_handles[link] = server.scene.add_mesh_simple(
            f"/hand/{link}", vertices, faces, color=(185, 205, 225), material="toon3", side="double"
        )

    labels = {
        name: server.scene.add_label(f"/labels/{name}", name, font_scene_height=0.008, visible=False)
        for name in NODE_ORDER
    }
    line_handle = None
    point_handle = None
    object_handles = []
    for sample_i in SAMPLES:
        object_mesh = mesh_arrays(ASSETS / "objects" / f"object_{sample_i}.obj")
        object_handles.append(server.scene.add_mesh_simple(
            f"/object/{sample_i}", object_mesh[0], object_mesh[1], color=(240, 160, 70),
            opacity=0.82, material="standard", side="double", visible=False
        ))

    def update(sample_slot: int):
        nonlocal line_handle, point_handle
        slot = int(sample_slot)
        src_i = int(POSES["indices"][slot])
        local_pos = POSES["moxian_skeleton_positions"][slot]
        local_quat = POSES["moxian_skeleton_quaternions_xyzw"][slot]
        root_qpos = POSES["moxian_qpos"][slot]
        root_pos = root_qpos[:3]
        root_quat = Rotation.from_rotvec(root_qpos[3:6]).as_quat().astype(np.float32)
        root_R = Rotation.from_quat(root_quat).as_matrix().astype(np.float32)
        pos = local_pos @ root_R.T + root_pos
        quat_xyzw = np.asarray([quat_mul_xyzw(root_quat, q) for q in local_quat], dtype=np.float32)
        wrist_frame.position = root_pos
        wrist_frame.wxyz = xyzw_to_wxyz(root_quat)
        for link, handle in hand_handles.items():
            idx = NODE_INDEX[link]
            handle.position = pos[idx]
            handle.wxyz = xyzw_to_wxyz(quat_xyzw[idx])
            handle.visible = bool(show_hand.value)
        for name, handle in labels.items():
            idx = NODE_INDEX[name]
            handle.position = pos[idx]
            handle.visible = bool(show_labels.value)
        for i, handle in enumerate(object_handles):
            handle.visible = bool(show_object.value) and i == slot
            if i == slot:
                pose = POSES["object_pose_world"][slot]
                scale = POSES["object_scale"][slot]
                handle.position = pose[:3]
                handle.wxyz = np.asarray([pose[3], pose[4], pose[5], pose[6]], np.float32)
                handle.scale = tuple(np.asarray(scale, np.float32).tolist())
        if line_handle is not None:
            line_handle.remove()
        if point_handle is not None:
            point_handle.remove()
        edge_points = np.asarray([[pos[NODE_INDEX[a]], pos[NODE_INDEX[b]]] for a, b in EDGES], np.float32)
        edge_colors = np.tile(np.asarray([[95, 220, 245]], np.uint8), (len(EDGES), 2, 1))
        line_handle = server.scene.add_line_segments("/skeleton/edges", edge_points, edge_colors, thickness=0.002, visible=bool(show_skeleton.value))
        point_colors = np.tile(np.asarray([[220, 245, 255]], np.uint8), (len(pos), 1))
        point_handle = server.scene.add_point_cloud("/skeleton/joints", pos, point_colors, point_size=0.004, point_shape="circle", visible=bool(show_skeleton.value))
        case = POSES['source_case_name'][slot]
        case = case.decode() if hasattr(case, 'decode') else str(case)
        server.gui.add_markdown(f"**Sample {src_i}** · case `{case}` · object scale `{POSES['object_scale'][slot].round(4).tolist()}`")

    @sample_gui.on_update
    def _(_event): update(int(sample_gui.value))

    @show_hand.on_update
    def _(_event): update(int(sample_gui.value))

    @show_object.on_update
    def _(_event): update(int(sample_gui.value))

    @show_skeleton.on_update
    def _(_event): update(int(sample_gui.value))

    @show_labels.on_update
    def _(_event): update(int(sample_gui.value))

    @reset.on_click
    def _(_event):
        # The client-side camera can be reset by reconnecting; keep this button
        # as a visible control without mutating the selected sample.
        update(int(sample_gui.value))

    update(0)
    print("Viser URL: http://127.0.0.1:8766", flush=True)
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
