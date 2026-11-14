"""`Entity`: a robot in a `MjWarpSim`, its state as torch tensors and its writes."""

import re
from typing import Any, List, Optional, Sequence

import mujoco  # pyright: ignore[reportMissingImports]
import numpy as np
import torch


def _find(names: Sequence[str], keys: Sequence[str]) -> List[int]:
    """The indices of `names` that fully match each regex of `keys`, in the keys' order."""
    return [i for key in keys for i, name in enumerate(names) if re.fullmatch(key, name)]


class EntityData:
    """An entity's state as torch tensors read from `sim.data.torch`, so `torch.compile` works.

    Example:
        >>> # xdoctest: +REQUIRES(module:mujoco_warp)
        >>> from robokit.sim import MjWarpSim
        >>> xml = '<mujoco><worldbody><body name="r/b" pos="0 0 .5"><freejoint name="r/root"/><geom size=".1"/></body>'
        >>> xml += '</worldbody></mujoco>'
        >>> data = MjWarpSim(mujoco.MjModel.from_xml_string(xml), nworld=2).build_entity("r/").data
        >>> data.root_link_pos_w[:, 2].tolist(), data.projected_gravity_b[:, 2].tolist()
        ([0.5, 0.5], [-1.0, -1.0])
    """

    def __init__(self, sim: Any, entity: "Entity", default_qpos: np.ndarray, soft_joint_pos_limit_factor: float):
        mjm = sim.mjm
        self._views = sim.data.torch
        device = entity.device
        joints = entity.joint_ids
        self._root_body = entity.root_body
        self.root_q_adr = int(mjm.jnt_qposadr[entity.root_joint]) if entity.root_joint >= 0 else 0
        self.root_v_adr = int(mjm.jnt_dofadr[entity.root_joint]) if entity.root_joint >= 0 else 0
        self.joint_q_adr = torch.tensor(mjm.jnt_qposadr[joints], device=device)
        self.joint_v_adr = torch.tensor(mjm.jnt_dofadr[joints], device=device)
        self._body_ids = torch.tensor(entity.body_ids, device=device)
        self._site_ids = torch.tensor(entity.site_ids, device=device)
        self._ctrl_ids = torch.tensor(entity.actuator_ids, device=device)
        self._site_bodies = torch.tensor(mjm.site_bodyid[entity.site_ids], device=device)
        self._site_roots = torch.tensor(mjm.body_rootid[mjm.site_bodyid[entity.site_ids]], device=device)
        self._body_roots = torch.tensor(mjm.body_rootid[entity.body_ids], device=device)
        root_pose = default_qpos[self.root_q_adr : self.root_q_adr + 7] if entity.root_joint >= 0 else np.zeros(7)
        self.default_root_state = torch.zeros(13, device=device)  # position, wxyz quaternion, linear, angular velocity
        self.default_root_state[:7] = torch.tensor(root_pose, device=device)
        self.default_joint_pos = torch.tensor(default_qpos[mjm.jnt_qposadr[joints]], device=device, dtype=torch.float32)
        lo, hi = mjm.jnt_range[joints].T
        mid = (lo + hi) / 2
        half = soft_joint_pos_limit_factor / 2 * (hi - lo)
        limits = np.stack([mid - half, mid + half], 1)
        self.soft_joint_pos_limits = torch.tensor(limits, device=device, dtype=torch.float32)  # (joints, 2)
        self.encoder_bias = torch.zeros(sim.nworld, len(joints), device=device)  # rad, in `joint_pos_biased`

    @property
    def projected_gravity_b(self) -> torch.Tensor:
        """(nworld, 3): gravity's direction in the root's frame."""
        return -self._views.xmat[:, self._root_body, 2]

    @property
    def heading_w(self) -> torch.Tensor:
        """(nworld,): the yaw of the root's x axis, rad."""
        R_world_root = self._views.xmat[:, self._root_body]
        return torch.atan2(R_world_root[:, 1, 0], R_world_root[:, 0, 0])

    @property
    def root_link_vel_w(self) -> torch.Tensor:
        """(nworld, 6): the root's linear and angular velocity in the world frame, from the free joint's."""
        R_world_root = self._views.xmat[:, self._root_body]
        ang_vel_w = torch.einsum("nij,nj->ni", R_world_root, self.root_link_ang_vel_b)
        return torch.cat([self._views.qvel[:, self.root_v_adr : self.root_v_adr + 3], ang_vel_w], 1)

    @property
    def root_link_lin_vel_b(self) -> torch.Tensor:
        """(nworld, 3): the root's linear velocity in its frame, from the free joint's."""
        R_world_root = self._views.xmat[:, self._root_body]
        return torch.einsum("nji,nj->ni", R_world_root, self._views.qvel[:, self.root_v_adr : self.root_v_adr + 3])

    @property
    def root_link_ang_vel_b(self) -> torch.Tensor:
        """(nworld, 3): the root's angular velocity in its frame, the free joint's."""
        return self._views.qvel[:, self.root_v_adr + 3 : self.root_v_adr + 6]

    @property
    def root_link_pos_w(self) -> torch.Tensor:
        """(nworld, 3): the root's position."""
        return self._views.xpos[:, self._root_body]

    @property
    def body_link_pos_w(self) -> torch.Tensor:
        """(nworld, bodies, 3): each body's position."""
        return self._views.xpos[:, self._body_ids]

    @property
    def body_link_mat_w(self) -> torch.Tensor:
        """(nworld, bodies, 3, 3): each body's orientation."""
        return self._views.xmat[:, self._body_ids]

    @property
    def body_link_lin_vel_w(self) -> torch.Tensor:
        """(nworld, bodies, 3): each body origin's linear velocity, from its `cvel` about its tree's center of mass."""
        cvel = self._views.cvel[:, self._body_ids]
        arm = self.body_link_pos_w - self._views.subtree_com[:, self._body_roots]
        return cvel[..., 3:] + torch.cross(cvel[..., :3], arm, dim=-1)

    @property
    def body_link_ang_vel_w(self) -> torch.Tensor:
        """(nworld, bodies, 3): each body's angular velocity in the world frame."""
        return self._views.cvel[:, self._body_ids, :3]

    @property
    def site_pos_w(self) -> torch.Tensor:
        """(nworld, sites, 3)."""
        return self._views.site_xpos[:, self._site_ids]

    @property
    def site_mat_w(self) -> torch.Tensor:
        """(nworld, sites, 3, 3): each site's orientation."""
        return self._views.site_xmat[:, self._site_ids]

    @property
    def site_lin_vel_w(self) -> torch.Tensor:
        """(nworld, sites, 3): each site's linear velocity, from its body's `cvel` about its tree's center of mass."""
        cvel = self._views.cvel[:, self._site_bodies]
        arm = self.site_pos_w - self._views.subtree_com[:, self._site_roots]
        return cvel[..., 3:] + torch.cross(cvel[..., :3], arm, dim=-1)

    @property
    def joint_pos_target(self) -> torch.Tensor:
        """(nworld, actuated joints): the position targets in `ctrl`, in joint order."""
        return self._views.ctrl[:, self._ctrl_ids]

    @property
    def joint_pos(self) -> torch.Tensor:
        """(nworld, joints)."""
        return self._views.qpos[:, self.joint_q_adr]

    @property
    def joint_pos_biased(self) -> torch.Tensor:
        """(nworld, joints): as the encoders read them, with `encoder_bias`."""
        return self.joint_pos + self.encoder_bias

    @property
    def joint_vel(self) -> torch.Tensor:
        """(nworld, joints)."""
        return self._views.qvel[:, self.joint_v_adr]

    @property
    def qfrc_actuator(self) -> torch.Tensor:
        """(nworld, joints): the actuators' generalized forces at the joints."""
        return self._views.qfrc_actuator[:, self.joint_v_adr]


class Entity:
    """A robot in a `MjWarpSim`: the bodies, sites, joints named `prefix*`, actuators and `data`.

    Args:
        prefix: The prefix of its names, which `joint_names`, `body_names` and `site_names` drop.

    Example:
        >>> # xdoctest: +REQUIRES(module:mujoco_warp)
        >>> import mujoco
        >>> from robokit.sim import MjWarpSim
        >>> xml = '<mujoco><compiler angle="radian"/><worldbody><body name="r/base"><freejoint name="r/root"/>'
        >>> xml += '<geom size=".1"/><site name="r/tip"/><body><joint name="r/a" range="-1 1"/><geom size=".1"/>'
        >>> xml += '</body></body></worldbody><actuator><position joint="r/a" kp="10"/></actuator></mujoco>'
        >>> robot = MjWarpSim(mujoco.MjModel.from_xml_string(xml), nworld=2).build_entity(
        ...     "r/", soft_joint_pos_limit_factor=0.5)
        >>> robot.joint_names, robot.find_sites(["tip"]), robot.data.soft_joint_pos_limits.tolist()
        (['a'], [0], [[-0.5, 0.5]])
        >>> mask = torch.tensor([True, False], device=robot.device)
        >>> robot.write_root_link_velocity_to_sim(torch.ones(2, 6, device=robot.device), mask)
        >>> robot.data.root_link_vel_w[:, 0].tolist()
        [1.0, 0.0]
    """

    def __init__(self, sim: Any, prefix: str, default_qpos: np.ndarray, soft_joint_pos_limit_factor: float = 1.0):
        mjm = sim.mjm
        self.device = sim.data.torch.qpos.device
        free = mujoco.mjtJoint.mjJNT_FREE
        ours = [j for j in range(mjm.njnt) if mjm.joint(j).name.startswith(prefix)]
        self.joint_ids = [j for j in ours if mjm.jnt_type[j] != free]  # hinges and slides, in model order
        roots = [j for j in ours if mjm.jnt_type[j] == free]
        self.root_joint = roots[0] if roots else -1  # its free joint, if any, is the root
        self.body_ids = [b for b in range(mjm.nbody) if mjm.body(b).name.startswith(prefix)]
        self.root_body = int(mjm.jnt_bodyid[self.root_joint]) if roots else self.body_ids[0]
        self.site_ids = [s for s in range(mjm.nsite) if mjm.site(s).name.startswith(prefix)]
        self.joint_names = [mjm.joint(j).name.removeprefix(prefix) for j in self.joint_ids]
        self.body_names = [mjm.body(b).name.removeprefix(prefix) for b in self.body_ids]
        self.site_names = [mjm.site(s).name.removeprefix(prefix) for s in self.site_ids]
        acting = {}  # joint: its actuator
        for a in range(mjm.nu):
            if mjm.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT and mjm.actuator_trnid[a, 0] in self.joint_ids:
                acting[int(mjm.actuator_trnid[a, 0])] = a
        actuated = [i for i, j in enumerate(self.joint_ids) if j in acting]  # joints with an actuator, in joint order
        self.actuator_ids = [acting[self.joint_ids[i]] for i in actuated]  # their actuators, in that order
        self.actuated_ids = torch.tensor(actuated, device=self.device)
        self._views = sim.data.torch  # writes to them are masked, with no host syncs
        self._ctrl_ids = torch.tensor(self.actuator_ids, device=self.device)
        self.data = EntityData(sim, self, default_qpos, soft_joint_pos_limit_factor)

    def find_bodies(self, keys: Sequence[str]) -> List[int]:
        """Indices into `body_names` (and `data`'s body axes) of the bodies matching each regex of `keys`, in order.

        Example:
            >>> # xdoctest: +REQUIRES(module:mujoco_warp)
            >>> from robokit.sim import MjWarpSim
            >>> xml = '<mujoco><worldbody><body name="r/base"><geom size=".1"/><body name="r/foot"><geom size=".1"/>'
            >>> xml += '</body></body></worldbody></mujoco>'
            >>> MjWarpSim(mujoco.MjModel.from_xml_string(xml), nworld=1).build_entity("r/").find_bodies(["foot", "b.*"])
            [1, 0]
        """
        return _find(self.body_names, keys)

    def find_sites(self, keys: Sequence[str]) -> List[int]:
        """Indices into `site_names` (and `data`'s site axes) of the sites matching each regex of `keys`, in order.

        Example:
            >>> # xdoctest: +REQUIRES(module:mujoco_warp)
            >>> from robokit.sim import MjWarpSim
            >>> xml = '<mujoco><worldbody><body name="r/b"><geom size=".1"/><site name="r/heel"/><site name="r/toe"/>'
            >>> xml += '</body></worldbody></mujoco>'
            >>> MjWarpSim(mujoco.MjModel.from_xml_string(xml), nworld=1).build_entity("r/").find_sites(["toe", "h.*"])
            [1, 0]
        """
        return _find(self.site_names, keys)

    def set_joint_position_target(self, target: torch.Tensor):
        """Write `target` (nworld, actuated joints), in joint order, to their actuators' `ctrl`.

        Example:
            >>> # xdoctest: +REQUIRES(module:mujoco_warp)
            >>> from robokit.sim import MjWarpSim
            >>> xml = '<mujoco><worldbody><body name="r/b"><joint name="r/a"/><joint name="r/c" axis="1 0 0"/>'
            >>> xml += '<geom size=".1"/></body></worldbody><actuator><position joint="r/c"/></actuator></mujoco>'
            >>> robot = MjWarpSim(mujoco.MjModel.from_xml_string(xml), nworld=2).build_entity("r/")
            >>> robot.set_joint_position_target(torch.tensor([[0.5], [-0.5]], device=robot.device))  # c only
            >>> robot.data.joint_pos_target.tolist()
            [[0.5], [-0.5]]
        """
        self._views.ctrl[:, self._ctrl_ids] = target

    def write_root_link_pose_to_sim(self, pose: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """Set the root's pose (nworld, 7: position, wxyz quaternion) where `mask` (nworld,) is set, else all.

        Example:
            >>> # xdoctest: +REQUIRES(module:mujoco_warp)
            >>> from robokit.sim import MjWarpSim
            >>> xml = '<mujoco><worldbody><body name="r/b"><freejoint name="r/root"/><geom size=".1"/></body>'
            >>> xml += '</worldbody></mujoco>'
            >>> robot = (sim := MjWarpSim(mujoco.MjModel.from_xml_string(xml), nworld=2)).build_entity("r/")
            >>> T_world_root = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]], device=robot.device).repeat(2, 1)
            >>> robot.write_root_link_pose_to_sim(T_world_root, torch.tensor([True, False], device=robot.device))
            >>> sim.kinematics()  # xpos from the new qpos
            >>> robot.data.root_link_pos_w[:, 2].tolist()
            [1.0, 0.0]
        """
        q = self.data.root_q_adr
        old = self._views.qpos[:, q : q + 7]
        self._views.qpos[:, q : q + 7] = pose if mask is None else torch.where(mask[:, None], pose, old)

    def write_root_link_velocity_to_sim(self, velocity: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """Set the root's world-frame velocity (nworld, 6: linear, angular) where `mask` (nworld,) is set, else all.

        Example:
            >>> # xdoctest: +REQUIRES(module:mujoco_warp)
            >>> from robokit.sim import MjWarpSim
            >>> xml = '<mujoco><worldbody><body name="r/b" quat="1 0 0 1"><freejoint name="r/root"/>'
            >>> xml += '<geom size=".1"/></body></worldbody></mujoco>'  # yawed 90 deg
            >>> robot = MjWarpSim(mujoco.MjModel.from_xml_string(xml), nworld=2).build_entity("r/")
            >>> robot.write_root_link_velocity_to_sim(torch.tensor([[0.0, 0, 0, 1, 0, 0]] * 2, device=robot.device))
            >>> expected = torch.tensor([0.0, -1, 0], device=robot.device)  # about world x is about the root's -y
            >>> torch.allclose(robot.data.root_link_ang_vel_b, expected, atol=1e-6)
            True
        """
        q = self.data.root_q_adr
        v = self.data.root_v_adr
        q_wxyz_world_root = self._views.qpos[:, q + 3 : q + 7]  # current, unlike `xmat` right after a pose write
        twice_cross = 2 * torch.cross(q_wxyz_world_root[:, 1:], velocity[:, 3:], dim=-1)
        rotated = q_wxyz_world_root[:, :1] * twice_cross
        ang_vel_b = velocity[:, 3:] - rotated + torch.cross(q_wxyz_world_root[:, 1:], twice_cross, dim=-1)  # body frame
        qvel = torch.cat([velocity[:, :3], ang_vel_b], 1)
        old = self._views.qvel[:, v : v + 6]
        self._views.qvel[:, v : v + 6] = qvel if mask is None else torch.where(mask[:, None], qvel, old)

    def write_joint_state_to_sim(
        self, position: torch.Tensor, velocity: torch.Tensor, mask: Optional[torch.Tensor] = None
    ):
        """Set the joints' positions and velocities (nworld, joints) where `mask` (nworld,) is set, else all.

        Example:
            >>> # xdoctest: +REQUIRES(module:mujoco_warp)
            >>> from robokit.sim import MjWarpSim
            >>> xml = '<mujoco><worldbody><body name="r/b"><joint name="r/a"/><geom size=".1"/></body>'
            >>> xml += '</worldbody></mujoco>'
            >>> robot = MjWarpSim(mujoco.MjModel.from_xml_string(xml), nworld=2).build_entity("r/")
            >>> one = torch.ones(1, device=robot.device)  # broadcast to every world
            >>> robot.write_joint_state_to_sim(0.5 * one, 0 * one, torch.tensor([False, True], device=robot.device))
            >>> robot.data.joint_pos.tolist()
            [[0.0], [0.5]]
        """
        position = position.expand_as(self.data.joint_pos)
        velocity = velocity.expand_as(self.data.joint_vel)
        if mask is not None:
            position = torch.where(mask[:, None], position, self.data.joint_pos)
            velocity = torch.where(mask[:, None], velocity, self.data.joint_vel)
        self._views.qpos[:, self.data.joint_q_adr] = position
        self._views.qvel[:, self.data.joint_v_adr] = velocity
