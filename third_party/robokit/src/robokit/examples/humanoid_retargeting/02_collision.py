"""Floor-collision humanoid retargeting example: penalize foot-ground penetration."""

import dataclasses
import time
from copy import deepcopy

import numpy as np
import viser
import warp as wp
import yourdfpy
from viser.extras import ViserUrdf

from robokit.geom import PlaneGeom, WarpScene
from robokit.helpers.humanoid_retargeting import HumanoidRetargeterOnline
from robokit.helpers.humanoid_retargeting.loaders import fetch_smplx_clip
from robokit.helpers.humanoid_retargeting.presets.g1_aligned import g1_aligned


MOTION = "datasets/sfu/0005/0005_2FeetJump001_poses.npz"
GROUND_WEIGHT = 1000.0  # set to 0.0 to watch the feet dive through the floor on landing


def main():
    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"

    # --- retargeting: caller owns the floor scene; its config weight enables the collision term ---
    config = dataclasses.replace(g1_aligned, terms=deepcopy(g1_aligned.terms))
    config.task("collision").weight = GROUND_WEIGHT
    config.task("collision").margin = 0.0
    assert config.urdf_path is not None
    robot = config.load_robot()
    scene = WarpScene(1, device).add(PlaneGeom(np.array([0, 1], dtype=np.int32)))
    helper = HumanoidRetargeterOnline(config, device=device, robot=robot, scene=scene)

    # --- motion: SMPL-X clip -> ordered per-frame human transforms ---
    smplx, T_world_human, fps = fetch_smplx_clip(MOTION, helper.human_joint_names)
    human_heights = np.array([smplx.human_height], dtype=np.float32)
    helper.warmup(1)
    helper.solve_numpy(T_world_human[None, 0], human_heights)  # warm up the CUDA graph
    helper.reset()

    # --- viewer ---
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=4, height=4)
    base = server.scene.add_frame("/robot", show_axes=False)
    urdf_vis = ViserUrdf(server, yourdfpy.URDF.load(str(config.urdf_path)), root_node_name="/robot")
    keypoints = [
        server.scene.add_icosphere(f"/human/{name}", radius=0.02, color=(255, 60, 60))
        for name in helper.human_joint_names
    ]
    timing = server.gui.add_number("Retarget (ms)", 0.001, disabled=True)

    i = 0
    while True:
        # --- retargeting: one frame per loop ---
        t0 = time.time()
        qpos = helper.solve_numpy(T_world_human[None, i], human_heights)[0]
        timing.value = 0.99 * timing.value + 0.01 * (time.time() - t0) * 1000

        # --- viewer: visualize the result ---
        with server.atomic():
            base.position, base.wxyz = qpos[:3], qpos[3:7]
            urdf_vis.update_cfg(qpos[7:])
            for j, handle in enumerate(keypoints):
                handle.position = T_world_human[i, j, :3]
        i = (i + 1) % len(T_world_human)
        if i == 0:
            helper.reset()
        time.sleep(max(0.0, 1.0 / fps - (time.time() - t0)))


if __name__ == "__main__":
    main()
