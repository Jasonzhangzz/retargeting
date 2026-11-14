"""Offline humanoid retargeting example: optimize the whole trajectory at once, then scrub it."""

import time

import numpy as np
import viser
import warp as wp
import yourdfpy
from viser.extras import ViserUrdf

from robokit.helpers.humanoid_retargeting import HumanoidRetargeterOffline
from robokit.helpers.humanoid_retargeting.loaders import fetch_smplx_clip
from robokit.helpers.humanoid_retargeting.presets.g1_offline import g1_offline
from robokit.helpers.humanoid_retargeting.presets.g1_offline_mapping import g1_offline_mapping


MOTION = "datasets/sfu/0007/0007_Walking001_poses.npz"


def main():
    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"

    # --- retargeting: IK warm-start + one LM solve over the whole trajectory ---
    assert g1_offline_mapping.urdf_path is not None
    retargeter = HumanoidRetargeterOffline(g1_offline_mapping, g1_offline, device=device)
    smplx, T_world_human, fps = fetch_smplx_clip(MOTION, retargeter.human_joint_names)
    t0 = time.time()
    qpos = retargeter.solve_numpy(T_world_human[None], human_heights=np.array([smplx.human_height], dtype=np.float32))[
        0
    ]
    print(f"solved {len(T_world_human)} frames in {time.time() - t0:.1f}s")

    # --- viewer: scrub the solved trajectory ---
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=4, height=4)
    base_frame = server.scene.add_frame("/robot", show_axes=False)
    urdf_vis = ViserUrdf(server, yourdfpy.URDF.load(str(g1_offline_mapping.urdf_path)), root_node_name="/robot")
    frame_slider = server.gui.add_slider("frame", min=0, max=len(T_world_human) - 1, step=1, initial_value=0)
    playing = server.gui.add_checkbox("play", True)
    while True:
        if playing.value:
            frame_slider.value = (frame_slider.value + 1) % len(T_world_human)
        q = qpos[frame_slider.value]
        with server.atomic():
            base_frame.position, base_frame.wxyz = q[:3], q[3:7]
            urdf_vis.update_cfg(q[7:])
        time.sleep(1.0 / fps)


if __name__ == "__main__":
    main()
