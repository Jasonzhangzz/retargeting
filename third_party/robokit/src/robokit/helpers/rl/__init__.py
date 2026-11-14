"""PPO (`ppo`) and FastSAC (`fastsac`) for batched GPU envs that implement `Env`; examples/rl holds such tasks."""

from typing import List, Protocol, Tuple

import torch


class Env(Protocol):
    """What solvers call: GPU tensors, no host syncs (FastSAC graphs a step), on the current (not default) stream."""

    n: int  # worlds
    obs_dims: Tuple[int, int]  # the actor's and the critic's observation sizes
    bound: torch.Tensor  # (number of actions,): FastSAC's tanh spans [-bound, bound]
    names: List[str]  # the reward terms

    def obs(self) -> Tuple[torch.Tensor, torch.Tensor]: ...  # the actor's and the critic's
    def reset(self, mask: torch.Tensor): ...  # the masked worlds
    def curriculum(self, step: int): ...  # to `step` env steps; called outside CUDA graphs, which freeze Python ints
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Act; return the reward, terminated and truncated (n,), and each reward term's mean over worlds."""
        ...


from robokit.helpers.rl.fastsac import FastSACConfig  # noqa: E402  (after `Env`, which the solvers import)
from robokit.helpers.rl.ppo import PPOConfig  # noqa: E402


__all__ = ["Env", "FastSACConfig", "PPOConfig"]
