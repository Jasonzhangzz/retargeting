"""Retained Dexonomy template contact-frame loader (CC BY-NC 4.0).

The caller supplies FK and skeleton arrays. The upstream background loader,
viewer and task pipeline are intentionally not part of this input adapter.
See source.json and licenses/Dexonomy-NOTICE.txt.
"""

from __future__ import annotations

import numpy as np
from transforms3d import quaternions as tq

from retarget.initialization.dexonomy.util.np_util import (
    np_normal_to_rot,
    np_transform_points,
    np_inv_transform_points,
)


class HandTemplateLoader:
    def _load_data(self, vis_env, data_path: str):
        hand_data = np.load(data_path, allow_pickle=True).item()
        hand_data["grasp_qpos"] = hand_data["grasp_qpos"].squeeze()
        xmat, xpos = vis_env.forward_kinematics(hand_data["grasp_qpos"][7:])
        skt_xmat, skt_xpos = xmat[self.skt_bid], xpos[self.skt_bid]
        hand_data["hand_skt_h"] = (
            np.reshape(self.skt_data, (-1, 2, 3)) @ skt_xmat.transpose(0, 2, 1)
            + skt_xpos[:, None, :]
        ).reshape(-1, 6)

        h_r = tq.quat2mat(hand_data["grasp_qpos"][3:7]).astype(np.float32)
        h_t = hand_data["grasp_qpos"][:3]
        h_cpn_h = np_inv_transform_points(hand_data["hand_cpn_w"], h_r, h_t)
        init_idx = np.random.randint(low=0, high=len(h_cpn_h))
        rot_c2h = np_normal_to_rot(h_cpn_h[init_idx, 3:].reshape(1, 3)).squeeze(0).T
        trans_c2h = -rot_c2h @ h_cpn_h[init_idx, :3]
        hand_data["rot_c2h"] = rot_c2h
        hand_data["trans_c2h"] = trans_c2h
        hand_data["hand_cpn_c"] = np_transform_points(h_cpn_h, rot_c2h, trans_c2h)
        hand_data["hand_path"] = data_path
        return hand_data
