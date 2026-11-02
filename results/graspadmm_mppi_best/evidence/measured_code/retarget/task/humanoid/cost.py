"""Unmodified SPIDER reward bodies, commit 44717007de41cbef7565dff7ff9f4453557a2d3d.
Copyright Meta Platforms, Inc.; CC BY-NC 4.0, see licenses/SPIDER.txt.
"""
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from retarget.env.humanoid.config import PhysicsConfig as Config
    from retarget.env.humanoid.physics import MJWPEnv
import torch
import warp as wp
from .math import quat_sub

def _weight_diff_qpos(config: Config) -> torch.Tensor:
    w = torch.ones(config.nv, device=config.device)
    if config.embodiment_type == "bimanual":
        half_dof = int(config.nu // 2)
        w[:3] = config.base_pos_rew_scale
        w[3:6] = config.base_rot_rew_scale
        w[6:half_dof] = config.joint_rew_scale
        w[half_dof : half_dof + 3] = config.base_pos_rew_scale
        w[half_dof + 3 : half_dof + 6] = config.base_rot_rew_scale
        w[half_dof + 6 : config.nu] = config.joint_rew_scale
        # object: weights live in nv-space (6-dim per freejoint), regardless of nq_obj
        w[-12:-9] = config.pos_rew_scale
        w[-9:-6] = config.rot_rew_scale
        w[-6:-3] = config.pos_rew_scale
        w[-3:] = config.rot_rew_scale
    elif config.embodiment_type in ["right", "left"]:
        w[:3] = config.base_pos_rew_scale
        w[3:6] = config.base_rot_rew_scale
        w[6 : config.nu] = config.joint_rew_scale
        w[-6:-3] = config.pos_rew_scale
        w[-3:] = config.rot_rew_scale
    elif config.embodiment_type in ["humanoid"]:  # humanoid robot
        # robot pos and rot
        w[:3] = config.pos_rew_scale
        w[3:6] = config.rot_rew_scale
        # robot joint
        w[6:] = config.joint_rew_scale
    elif config.embodiment_type in ["humanoid_object"]:
        # robot pos and rot
        w[:3] = config.base_pos_rew_scale
        w[3:6] = config.base_rot_rew_scale
        # robot joint
        w[6:-6] = config.joint_rew_scale
        # object pos and rot
        w[-6:-3] = config.pos_rew_scale
        w[-3:] = config.rot_rew_scale
    else:
        raise ValueError(f"Invalid embodiment_type: {config.embodiment_type}")
    return w


def _diff_qpos(
    config: Config, qpos_sim: torch.Tensor, qpos_ref: torch.Tensor
) -> torch.Tensor:
    """Compute the difference between qpos_sim and qpos_ref
    TODO: replace with mujoco built-in function, not sure how to call warp internal function yet.
    """
    batch_size = qpos_sim.shape[0]
    qpos_diff = torch.zeros((batch_size, config.nv), device=config.device)
    if config.embodiment_type == "bimanual":
        if config.nq_obj == 12:
            qpos_diff[:, :-12] = qpos_sim[:, :-12] - qpos_ref[:, :-12]
            qpos_diff[:, -12:-9] = qpos_sim[:, -12:-9] - qpos_ref[:, -12:-9]
            qpos_diff[:, -9:-6] = qpos_sim[:, -9:-6] - qpos_ref[:, -9:-6]
            qpos_diff[:, -6:-3] = qpos_sim[:, -6:-3] - qpos_ref[:, -6:-3]
            qpos_diff[:, -3:] = qpos_sim[:, -3:] - qpos_ref[:, -3:]
            return qpos_diff
        # joint
        qpos_diff[:, :-12] = qpos_sim[:, :-14] - qpos_ref[:, :-14]
        # position
        qpos_diff[:, -12:-9] = qpos_sim[:, -14:-11] - qpos_ref[:, -14:-11]
        qpos_diff[:, -6:-3] = qpos_sim[:, -7:-4] - qpos_ref[:, -7:-4]
        # rotation
        qpos_diff[:, -9:-6] = quat_sub(qpos_sim[:, -11:-7], qpos_ref[:, -11:-7])
        qpos_diff[:, -3:] = quat_sub(qpos_sim[:, -4:], qpos_ref[:, -4:])
    elif config.embodiment_type in ["right", "left"]:
        if config.nq_obj == 6:
            qpos_diff[:, :-6] = qpos_sim[:, :-6] - qpos_ref[:, :-6]
            qpos_diff[:, -6:-3] = qpos_sim[:, -6:-3] - qpos_ref[:, -6:-3]
            qpos_diff[:, -3:] = qpos_sim[:, -3:] - qpos_ref[:, -3:]
            return qpos_diff
        # joint
        qpos_diff[:, :-6] = qpos_sim[:, :-7] - qpos_ref[:, :-7]
        # position
        qpos_diff[:, -6:-3] = qpos_sim[:, -7:-4] - qpos_ref[:, -7:-4]
        # rotation
        qpos_diff[:, -3:] = quat_sub(qpos_sim[:, -4:], qpos_ref[:, -4:])
    elif config.embodiment_type in ["humanoid"]:
        # joint
        qpos_diff[:, 6:] = qpos_sim[:, 7:] - qpos_ref[:, 7:]
        # position
        qpos_diff[:, :3] = qpos_sim[:, :3] - qpos_ref[:, :3]
        # rotation
        qpos_diff[:, 3:6] = quat_sub(qpos_sim[:, 3:7], qpos_ref[:, 3:7])
    elif config.embodiment_type in ["humanoid_object"]:
        qpos_humanoid = qpos_sim[:, :-7]
        qpos_object = qpos_sim[:, -7:]
        qpos_ref_humanoid = qpos_ref[:, :-7]
        qpos_ref_object = qpos_ref[:, -7:]
        # position
        qpos_diff[:, :3] = qpos_humanoid[:, :3] - qpos_ref_humanoid[:, :3]
        # rotation
        qpos_diff[:, 3:6] = quat_sub(qpos_humanoid[:, 3:7], qpos_ref_humanoid[:, 3:7])
        # joint
        qpos_diff[:, 6:-6] = qpos_humanoid[:, 7:] - qpos_ref_humanoid[:, 7:]
        # object
        qpos_diff[:, -6:-3] = qpos_object[:, :3] - qpos_ref_object[:, :3]
        qpos_diff[:, -3:] = quat_sub(qpos_object[:, 3:7], qpos_ref_object[:, 3:7])
    else:
        raise ValueError(f"Invalid embodiment_type: {config.embodiment_type}")
    return qpos_diff


def get_reward(
    config: Config,
    env: MJWPEnv,
    ref: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Non-terminal step reward for MJWP batched worlds.
    ref is a tuple: (qpos_ref, qvel_ref, ctrl_ref, contact_ref, contact_pos_ref)
    Returns (N,)

    TODO: move reward computation to task-specific module
    """
    qpos_ref, qvel_ref, ctrl_ref, contact_ref, contact_pos_ref = ref
    qpos_sim = wp.to_torch(env.data_wp.qpos)
    qvel_sim = wp.to_torch(env.data_wp.qvel)

    # weighted qpos tracking
    qpos_diff = _diff_qpos(
        config, qpos_sim, qpos_ref.unsqueeze(0).repeat(qpos_sim.shape[0], 1)
    )
    qpos_weight = _weight_diff_qpos(config)
    delta_qpos = qpos_diff * qpos_weight
    qpos_dist = torch.norm(delta_qpos, p=2, dim=1)
    qvel_dist = torch.norm(qvel_sim - qvel_ref, p=2, dim=1)

    qpos_rew = -qpos_dist * 1.0
    qvel_rew = -config.vel_rew_scale * qvel_dist * 1.0

    # contact reward
    if config.contact_rew_scale > 0.0 and len(config.contact_site_ids) > 0:
        site_xpos_torch = wp.to_torch(env.data_wp.site_xpos)
        contact_pos = site_xpos_torch[:, config.contact_site_ids]
        contact_dist = torch.norm(contact_pos - contact_pos_ref, p=2, dim=-1)
        contact_dist_masked = contact_dist * contact_ref.unsqueeze(0)
        contact_rew = -contact_dist_masked.sum(dim=1)
    else:
        contact_rew = 0.0

    reward = qpos_rew + qvel_rew + contact_rew

    info = {
        "qpos_dist": qpos_dist,
        "qvel_dist": qvel_dist,
        "qpos_rew": qpos_rew,
        "qvel_rew": qvel_rew,
    }
    return reward, info


def get_terminal_reward(
    config: Config,
    env: MJWPEnv,
    ref_slice: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Terminal reward focusing on object tracking."""
    # return config.terminal_rew_scale * get_reward(config, env, ref_slice)
    # qpos_ref, qvel_ref, ctrl_ref, contact_ref, _ = ref_slice
    # qpos_sim = wp.to_torch(env.data_wp.qpos)
    # qpos_weight = torch.zeros(qpos_sim.shape[1], device=config.device)
    # if config.embodiment_type == "bimanual":
    #     qpos_weight[-14:-11] = config.pos_rew_scale
    #     qpos_weight[-11:-7] = config.rot_rew_scale
    #     qpos_weight[-7:-4] = config.pos_rew_scale
    #     qpos_weight[-4:] = config.rot_rew_scale
    # elif config.embodiment_type in ["right", "left"]:
    #     qpos_weight[-7:-4] = config.pos_rew_scale
    #     qpos_weight[-4:] = config.rot_rew_scale
    # elif config.embodiment_type in ["CMU", "DanceDB"]:
    #     qpos_weight[:3] = config.pos_rew_scale
    #     qpos_weight[3:7] = config.rot_rew_scale
    # else:
    #     raise ValueError(f"Invalid embodiment_type: {config.embodiment_type}")
    # delta_qpos = (qpos_sim - qpos_ref) * qpos_weight
    # cost_object = config.terminal_rew_scale * torch.sum(delta_qpos**2, dim=1)

    rew, info = get_reward(config, env, ref_slice)
    terminal_rew = config.terminal_rew_scale * rew
    return terminal_rew, info


def get_terminate(
    config: Config, env: MJWPEnv, ref_slice: tuple[torch.Tensor, ...]
) -> torch.Tensor:
    # compute object position and orientation error, compare to thereshold
    qpos_sim = wp.to_torch(env.data_wp.qpos)
    qpos_ref, qvel_ref, ctrl_ref, contact_ref, _contact_pos_ref = ref_slice
    if config.embodiment_type == "bimanual":
        if config.nq_obj == 12:
            right_obj_pos = qpos_sim[:, -12:-9]
            right_obj_pos_ref = qpos_ref[-12:-9].unsqueeze(0)
            right_obj_pos_error = torch.norm(
                right_obj_pos - right_obj_pos_ref, p=2, dim=1
            )
            right_obj_rot = qpos_sim[:, -9:-6]
            right_obj_rot_ref = qpos_ref[-9:-6].unsqueeze(0)
            right_obj_rot_error = torch.norm(
                right_obj_rot - right_obj_rot_ref, p=2, dim=1
            )
            left_obj_pos = qpos_sim[:, -6:-3]
            left_obj_pos_ref = qpos_ref[-6:-3].unsqueeze(0)
            left_obj_pos_error = torch.norm(left_obj_pos - left_obj_pos_ref, p=2, dim=1)
            left_obj_rot = qpos_sim[:, -3:]
            left_obj_rot_ref = qpos_ref[-3:].unsqueeze(0)
            left_obj_rot_error = torch.norm(left_obj_rot - left_obj_rot_ref, p=2, dim=1)
            if torch.all(right_obj_pos_ref.abs() < 1e-4):
                right_obj_pos_error *= 0.0
                right_obj_rot_error *= 0.0
            if torch.all(left_obj_pos_ref.abs() < 1e-4):
                left_obj_pos_error *= 0.0
                left_obj_rot_error *= 0.0
            terminate = (
                (left_obj_pos_error > config.object_pos_threshold)
                | (right_obj_pos_error > config.object_pos_threshold)
                | (left_obj_rot_error > config.object_rot_threshold)
                | (right_obj_rot_error > config.object_rot_threshold)
            )
            return terminate
        left_obj_pos = qpos_sim[:, -14:-11]
        left_obj_pos_ref = qpos_ref[-14:-11].unsqueeze(0)
        left_obj_pos_error = torch.norm(left_obj_pos - left_obj_pos_ref, p=2, dim=1)
        left_obj_quat = qpos_sim[:, -11:-7]
        left_obj_quat_ref = qpos_ref[-11:-7].unsqueeze(0)
        left_obj_quat_error = torch.norm(
            quat_sub(left_obj_quat, left_obj_quat_ref.repeat(qpos_sim.shape[0], 1)),
            p=2,
            dim=1,
        )
        right_obj_pos = qpos_sim[:, -7:-4]
        right_obj_pos_ref = qpos_ref[-7:-4].unsqueeze(0)
        right_obj_pos_error = torch.norm(right_obj_pos - right_obj_pos_ref, p=2, dim=1)
        right_obj_quat = qpos_sim[:, -4:]
        right_obj_quat_ref = qpos_ref[-4:].unsqueeze(0)
        right_obj_quat_error = torch.norm(
            quat_sub(right_obj_quat, right_obj_quat_ref.repeat(qpos_sim.shape[0], 1)),
            p=2,
            dim=1,
        )
        # special case: only have left object
        if torch.all(right_obj_pos_ref.abs() < 1e-4):
            right_obj_pos_error *= 0.0
            right_obj_quat_error *= 0.0
        # special case: only have right object
        if torch.all(left_obj_pos_ref.abs() < 1e-4):
            left_obj_pos_error *= 0.0
            left_obj_quat_error *= 0.0
        terminate = (
            (left_obj_pos_error > config.object_pos_threshold)
            | (right_obj_pos_error > config.object_pos_threshold)
            | (left_obj_quat_error > config.object_rot_threshold)
            | (right_obj_quat_error > config.object_rot_threshold)
        )
    elif config.embodiment_type in ["right", "left"]:
        if config.nq_obj == 6:
            obj_pos = qpos_sim[:, -6:-3]
            obj_pos_ref = qpos_ref[-6:-3].unsqueeze(0)
            obj_pos_error = torch.norm(obj_pos - obj_pos_ref, p=2, dim=1)
            obj_rot = qpos_sim[:, -3:]
            obj_rot_ref = qpos_ref[-3:].unsqueeze(0)
            obj_rot_error = torch.norm(obj_rot - obj_rot_ref, p=2, dim=1)
            terminate = (obj_pos_error > config.object_pos_threshold) | (
                obj_rot_error > config.object_rot_threshold
            )
            return terminate
        obj_pos = qpos_sim[:, -7:-4]
        obj_pos_ref = qpos_ref[-7:-4].unsqueeze(0)
        obj_pos_error = torch.norm(obj_pos - obj_pos_ref, p=2, dim=1)
        obj_quat = qpos_sim[:, -4:]
        obj_quat_ref = qpos_ref[-4:].unsqueeze(0)
        obj_quat_error = torch.norm(
            quat_sub(obj_quat, obj_quat_ref.repeat(qpos_sim.shape[0], 1)), p=2, dim=1
        )
        terminate = (obj_pos_error > config.object_pos_threshold) | (
            obj_quat_error > config.object_rot_threshold
        )
    elif config.embodiment_type in ["humanoid", "humanoid_object"]:
        base_pos = qpos_sim[:, :3]
        base_pos_ref = qpos_ref[:3].unsqueeze(0)
        base_pos_error = torch.norm(base_pos - base_pos_ref, p=2, dim=1)
        base_quat = qpos_sim[:, 3:7]
        base_quat_ref = qpos_ref[3:7].unsqueeze(0)
        base_quat_error = torch.norm(
            quat_sub(base_quat, base_quat_ref.repeat(qpos_sim.shape[0], 1)), p=2, dim=1
        )
        terminate = (base_pos_error > config.base_pos_threshold) | (
            base_quat_error > config.base_rot_threshold
        )
    else:
        raise ValueError(f"Invalid embodiment_type: {config.embodiment_type}")
    return terminate
