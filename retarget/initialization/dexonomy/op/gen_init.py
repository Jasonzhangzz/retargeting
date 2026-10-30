"""Retained Dexonomy HandObjMatcher input initialization (CC BY-NC 4.0)."""
import numpy as np
import torch
from retarget.initialization.dexonomy.util.warp_util import WarpCollisionEnv
from retarget.initialization.dexonomy.util.torch_util import (torch_transform_points,torch_quaternion_to_matrix,torch_matrix_to_quaternion,torch_normalize_vector,torch_multiply_pose,torch_inv_transform_points)
class HandObjMatcher:

    def __init__(self, cfg: dict, device: torch.device, coll_env: WarpCollisionEnv):
        self.cfg = cfg
        self.device = device
        self.coll_env = coll_env

    def forward(self, tmpl_sample: dict, obj_sample: dict):
        opt_q = torch_matrix_to_quaternion(obj_sample["rot_o2c_init"])
        opt_t = obj_sample["trans_o2c_init"]
        opt_q.requires_grad_()
        opt_t.requires_grad_()
        optimizer = torch.optim.Adam(
            [{"params": opt_q, "lr": 1e-2}, {"params": opt_t, "lr": 1e-2}]
        )

        b, h = opt_t.shape[:-1]
        obj_scale = obj_sample["obj_scale"].view(b, 1, 1, 3)
        h_cpn_c = tmpl_sample["hand_cpn_c"]

        for i in range(self.cfg.opt_step):
            optimizer.zero_grad()
            opt_r = torch_quaternion_to_matrix(opt_q)
            h_cpn_o = torch_transform_points(
                h_cpn_c, opt_r, opt_t.view(b, h, 1, 3), 1 / obj_scale
            )
            o_cp_o, o_cn_o = self.coll_env.query_point(h_cpn_o[..., :3])
            loss_dist = ((obj_scale * (o_cp_o - h_cpn_o[..., :3])) ** 2).sum(dim=-1)
            loss_normal = ((o_cn_o - h_cpn_o[..., 3:]) ** 2).sum(dim=-1)  # [b,h,n]
            loss = (1000 * loss_dist + loss_normal).mean(dim=-1).mean(dim=-1).sum()
            if i == self.cfg.opt_step - 1:
                break

            loss.backward()
            optimizer.step()

        # Post processing
        quat_so2c = torch_normalize_vector(opt_q)
        rot_so2c = torch_quaternion_to_matrix(quat_so2c)
        trans_so2c = opt_t.view(b, h, 1, 3) * obj_scale

        pose_w2so = obj_sample["pose_w2so"]
        rot_w2so = torch_quaternion_to_matrix(pose_w2so[..., 3:]).view(b, 1, 3, 3)
        trans_w2so = pose_w2so[..., :3].view(b, 1, 1, 3)

        rot_w2c, trans_w2c = torch_multiply_pose(
            rot_w2so, trans_w2so, rot_so2c, trans_so2c
        )

        h_cpn_w = torch_transform_points(h_cpn_o, rot_w2so, trans_w2so, obj_scale)
        o_cp_w = torch_transform_points(o_cp_o, rot_w2so, trans_w2so, obj_scale)
        o_cn_w = torch_transform_points(o_cn_o, rot_w2so)

        rot_c2h = tmpl_sample["rot_c2h"]
        trans_c2h = tmpl_sample["trans_c2h"].unsqueeze(-2)
        rot_w2h, trans_w2h = torch_multiply_pose(rot_w2c, trans_w2c, rot_c2h, trans_c2h)
        quat_w2h = torch_matrix_to_quaternion(rot_w2h)
        trans_w2h = trans_w2h.squeeze(-2)

        ret_dict = {
            "loss_normal": loss_normal.view(b * h, -1),
            "loss_dist": loss_dist.view(b * h, -1),
            "grasp_pose": torch.cat([trans_w2h, quat_w2h], dim=-1).view(b * h, 7),
            "hand_cpn_w": h_cpn_w.view(b * h, -1, 6),
            "hand_skt_h": tmpl_sample["hand_skt_h"].view(b * h, -1, 3),
            "grasp_qpos": tmpl_sample["grasp_qpos"][..., 7:].view(b * h, -1),
            "n_evo": tmpl_sample["n_evo"].view(b * h),
            "obj_cp_w": o_cp_w.view(b * h, -1, 3),
            "obj_cn_w": o_cn_w.view(b * h, -1, 3),
            "obj_scale": obj_scale.repeat(1, h, 1, 1).view(b * h, 3),
            "obj_id": torch.arange(b * h, device=self.device) // h,
            "obj_pose": pose_w2so.repeat_interleave(h, dim=0),
            "ext_center": obj_sample["ext_center"].repeat_interleave(h, dim=0),
            "ext_wrench": obj_sample["ext_wrench"].repeat_interleave(h, dim=0),
        }
        return ret_dict
