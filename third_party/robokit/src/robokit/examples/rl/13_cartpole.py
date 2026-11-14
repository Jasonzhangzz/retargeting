# /// script
# requires-python = ">=3.10"
# dependencies = ["robokit[mjwarp,pt-cu126]"]
#
# [tool.uv.sources]
# robokit = { path = "../../../..", editable = true }
# ///
"""Cartpole: a cart learns to swing up (or balance) a pole with PPO on 4096 MuJoCo Warp worlds, then is scored.

Example usage:
    uv run src/robokit/examples/rl/13_cartpole.py swingup --minutes 1
    uv run src/robokit/examples/rl/13_cartpole.py balance
    uv run src/robokit/examples/rl/13_cartpole.py swingup --coarse 0.5  # coarse physics mixed in over the first half
"""

import math
from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple

import mujoco  # pyright: ignore[reportMissingImports]
import torch
import tyro

from robokit.helpers.rl import Env, ppo
from robokit.sim import CoarseConfig, MjWarpSim, mjscene


@dataclass(frozen=True)
class CartpoleConfig:
    """The cartpole task and its training: the start, the worlds and the learner's time."""

    home: Tuple[float, float]  # initial qpos: the cart's position (m) and the pole's angle (rad, 0 upright)
    reset_offset: Tuple[float, float]  # half-widths of the uniform offsets added to `home` on reset (m, rad)
    worlds: int = 4096  # MjWarpSim worlds
    minutes: float = 1.0  # wall time to train, compilation included
    coarse: float = 0.0  # the share of PPO's time that mixes in coarse physics (`coarse_physics`)
    control_dt: float = 0.05  # s per policy action; physics steps are `mjm.opt.timestep`
    episode: int = 1000  # steps per episode
    # Coarse physics (`CoarseConfig`): 25 ms physics steps (nothing collides)
    coarse_physics: CoarseConfig = CoarseConfig(timestep=0.025)


SWINGUP = CartpoleConfig(home=(0.0, math.pi), reset_offset=(0.0, 0.034))  # from hanging down
BALANCE = CartpoleConfig(home=(0.0, 0.0), reset_offset=(0.1, 0.034))  # from near upright


class Cartpole(Env):
    """The swingup (or balance) on `config.worlds` worlds, in the full physics or the `coarse` one."""

    def __init__(self, config: CartpoleConfig, coarse: Optional[CoarseConfig] = None):
        n = config.worlds
        self.n = n
        self.config = config

        # --- build scene ---
        # a 1 kg cart on a 3.6 m slider 1 m up, a 1 m 0.1 kg pole, a 10 N motor; 10 ms steps, no contacts
        scene_spec = mjscene.build_scene_spec(floor=False, timestep=0.01)
        scene_spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        cart = scene_spec.worldbody.add_body(name="cartpole/cart", pos=[0.0, 0.0, 1.0])
        cart.add_joint(
            name="cartpole/slider",
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=[1.0, 0.0, 0.0],
            limited=mujoco.mjtLimited.mjLIMITED_TRUE,
            range=[-1.8, 1.8],
            solref_limit=[0.08, 1.0],
            damping=5e-4,
        )
        cart.add_geom(name="cart", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.2, 0.15, 0.1], mass=1.0)
        pole = cart.add_body(name="cartpole/pole")
        pole.add_joint(name="cartpole/hinge", axis=[0.0, 1.0, 0.0], damping=2e-6)
        pole.add_geom(
            name="pole", type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[0, 0, 0, 0, 0, 1], size=[0.045, 0, 0], mass=0.1
        )
        scene_spec.add_actuator(
            name="slide",
            target="cartpole/slider",
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            gear=[10.0, 0, 0, 0, 0, 0],
            ctrllimited=mujoco.mjtLimited.mjLIMITED_TRUE,
            ctrlrange=[-1.0, 1.0],
        )
        if coarse is not None:
            scene_spec = MjWarpSim.build_coarse_spec(scene_spec, coarse)

        # --- build simulator ---
        mjm = scene_spec.compile()
        self.mjm = mjm
        substeps = round(config.control_dt / mjm.opt.timestep)
        assert abs(substeps * mjm.opt.timestep - config.control_dt) < 1e-9, (
            "the control step must be a timestep multiple"
        )
        sim = MjWarpSim(mjm, n, substeps=substeps)
        self.sim = sim
        sim.set_torch_stream()
        cartpole = sim.build_entity("cartpole/")  # joints: the slider, then the hinge
        self.cartpole = cartpole
        self.device = cartpole.device

        # --- set up actions ---
        self.bound = torch.ones(1, device=self.device)  # the motor's control range

        # --- set up rewards ---
        self.names = ["smooth_reward"]

        # --- set up resets ---
        self.home = torch.tensor(config.home, device=self.device)
        self.reset_offset = torch.tensor(config.reset_offset, device=self.device)

        # --- set up observations ---
        self.obs_dims = (5, 5)

        # --- initialize state ---
        self.action = torch.zeros(n, 1, device=self.device)
        self.reward = torch.zeros(n, device=self.device)
        self.steps = torch.zeros(n, dtype=torch.long, device=self.device)
        self.reset(torch.ones(n, dtype=torch.bool, device=self.device))

    def curriculum(self, step: int):
        """Spread episodes that are in step, as after a full reset, over their length."""
        spread = torch.randint_like(self.steps, self.config.episode)
        self.steps.copy_(torch.where((self.steps == self.steps[0]).all(), spread, self.steps))

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Push the cart with `action` x 10 N, as MuJoCo clamps the control to [-1, 1]; step; score."""
        self.action.copy_(action)
        self.sim.data.torch.ctrl.copy_(action)
        self.sim.step()
        return self.score()

    @torch.compile
    def score(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The smooth reward: upright x centered x small control x small pole velocity; no terminations."""
        cartpole = self.cartpole.data
        self.steps += 1
        cart_position = cartpole.joint_pos[:, 0]
        pole_angle = cartpole.joint_pos[:, 1]
        pole_speed = cartpole.joint_vel[:, 1]
        control = self.action[:, 0]
        centered = (1 + 0.1 ** ((cart_position / 2.0) ** 2)) / 2  # Gaussian: 0.1 at the margin
        slow = (1 + 0.1 ** ((pole_speed / 5.0) ** 2)) / 2
        smooth = (pole_angle.cos() + 1) / 2 * centered * (4 + (1 - control**2).clamp(min=0.0)) / 5 * slow
        self.reward.copy_(smooth * self.config.control_dt)
        timeout = self.steps >= self.config.episode
        return self.reward, torch.zeros_like(timeout), timeout, smooth.mean()[None]

    @torch.compile
    def reset(self, mask: torch.Tensor):
        """Each joint at home plus a uniform offset, a uniform velocity in +-0.01; masked writes."""
        offset = self.reset_offset * (2 * torch.rand(self.n, 2, device=self.device) - 1)
        velocity = 0.01 * (2 * torch.rand(self.n, 2, device=self.device) - 1)
        self.cartpole.write_joint_state_to_sim(self.home + offset, velocity, mask)
        warmstart = self.sim.data.torch.qacc_warmstart
        warmstart.copy_(torch.where(mask[:, None], 0.0, warmstart))
        self.steps.masked_fill_(mask, 0)

    @torch.compile
    def obs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cart position, the pole angle's cosine and sine, cart and pole velocities, for actor and critic alike."""
        cartpole = self.cartpole.data
        cart_position = cartpole.joint_pos[:, 0]
        pole_angle = cartpole.joint_pos[:, 1]
        cart_speed = cartpole.joint_vel[:, 0]
        pole_speed = cartpole.joint_vel[:, 1]
        obs = torch.stack([cart_position, pole_angle.cos(), pole_angle.sin(), cart_speed, pole_speed], 1)
        return obs, obs


if __name__ == "__main__":
    config = tyro.cli(tyro.extras.subcommand_type_from_defaults(dict(swingup=SWINGUP, balance=BALANCE)))
    env = Cartpole(config)  # the full physics, where the policy is scored
    coarse_env = Cartpole(config, coarse=config.coarse_physics) if config.coarse else None
    learner = ppo.PPO(env.obs_dims, len(env.bound), ppo.PPOConfig(hidden=(64, 64), horizon=32, normalize=False))
    actor = learner.train(env, config.minutes, coarse_env, config.coarse)
    # the return per episode at each step's reward (at most 50)
    print(ppo.evaluate(env, actor, dict(episode_reward=partial(torch.mul, env.reward, config.episode)), config.episode))
