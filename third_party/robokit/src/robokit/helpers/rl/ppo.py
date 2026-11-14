"""PPO with GAE, a KL-adaptive learning rate and observation normalizers, the whole update as one CUDA graph.

PPO: https://arxiv.org/abs/1707.06347. GAE: https://arxiv.org/abs/1506.02438.
"""

import math
import random
import time
from dataclasses import dataclass
from itertools import chain
from typing import Callable, Dict, List, Literal, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.adam import Adam
from torch.optim.optimizer import Optimizer

from robokit.helpers.rl import Env


@dataclass(frozen=True)
class PPOConfig:
    """PPO hyperparameters for locomotion and manipulation tasks.

    Example:
        >>> # xdoctest: +REQUIRES(module:torch)
        >>> PPOConfig(entropy=0.005).horizon
        24
    """

    gamma: float = 0.99
    """Discount."""
    lam: float = 0.95
    """GAE lambda."""
    clip: float = 0.2
    """Clip of the probability ratio and of the value's change."""
    value_coef: float = 1.0
    """Value loss weight."""
    entropy: float = 0.01
    """Entropy bonus weight."""
    lr: float = 1e-3
    """Initial learning rate."""
    lr_range: Tuple[float, float] = (1e-5, 1e-2)
    """Bounds of the adaptive learning rate."""
    lr_factor: float = 1.5
    """The rate divides by this when the KL exceeds `2 kl`, multiplies by it when the KL falls under `kl / 2`."""
    kl: float = 0.01
    """Target KL divergence per minibatch."""
    epochs: int = 5
    """Passes over each rollout."""
    minibatches: int = 4
    """Minibatches per pass."""
    reshuffle: bool = False
    """A new minibatch split each pass, not one split per update."""
    max_grad: float = 1.0
    """Gradient-norm clip."""
    grad_clip: Literal["network", "joint"] = "network"
    """Clip each network's gradient norm (shared parameters count as the actor's), or one norm over both."""
    horizon: int = 24
    """Env steps per world per rollout."""
    hidden: Tuple[int, ...] = (512, 256, 128)
    """Hidden widths of the actor's and the critic's ELU MLPs."""
    init_std: float = 1.0
    """Initial action std."""
    std_range: Tuple[float, float] = (1e-6, 1e6)
    """Clamp of the action std."""
    std: Literal["global", "state"] = "global"
    """`global`: a learned std per action; `state`: a softplus std per sample, actions tanh-squashed."""
    squash_entropy: bool = False
    """With `std="state"`, the squashed entropy (tanh log-Jacobian at a sample added): saturation loses entropy."""
    normalize: bool = True
    """Empirical normalization of both networks' observations (off for cartpole)."""


class Normalizer(nn.Module):
    """Empirical normalization: running mean and variance over every sample seen; 0.01 added to the std.

    Example:
        >>> # xdoctest: +REQUIRES(module:torch)
        >>> (norm := Normalizer(2)).update(torch.tensor([[1.0, 2.0], [3.0, 6.0]]))
        >>> norm(torch.tensor([[2.0, 4.0]])).tolist()
        [[0.0, 0.0]]
    """

    def __init__(self, dim: int):
        super().__init__()
        for name, value in (("_mean", 0.0), ("_var", 1.0), ("_std", 1.0)):
            self.register_buffer(name, torch.full((1, dim), value))
        self.register_buffer("count", torch.zeros((), dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._mean) / (self._std + 0.01)  # pyright: ignore[reportOperatorIssue]

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        self.count += x.shape[0]
        rate, mean = x.shape[0] / self.count, x.mean(0, keepdim=True)
        delta = mean - self._mean
        self._mean += rate * delta
        self._var += rate * (x.var(0, unbiased=False, keepdim=True) - self._var + delta * (mean - self._mean))
        self._std.copy_(self._var.sqrt())  # pyright: ignore[reportCallIssue]


class Model(nn.Module):
    """A normalizer, an ELU MLP and a scalar std per action.

    Args:
        state_std: the MLP also outputs each std; `forward` is the deterministic action, tanh of the mean.
        normalize: an empirical normalizer, else the identity.

    Example:
        >>> # xdoctest: +REQUIRES(module:torch)
        >>> tuple(Model(48, 12, hidden=(64, 64), std=1.0)(torch.zeros(5, 48)).shape)
        (5, 12)
        >>> tuple(Model(48, 12, hidden=(64, 64), std=1.0, state_std=True).dist(torch.zeros(5, 48), (0.0, 9.0))[1].shape)
        (5, 12)
    """

    def __init__(
        self,
        obs_dim: int,
        out_dim: int,
        hidden: Tuple[int, ...],
        std: float = 0.0,
        state_std: bool = False,
        normalize: bool = True,
    ):
        super().__init__()
        dims, self.state_std = (obs_dim, *hidden), state_std
        self.obs_normalizer = Normalizer(obs_dim) if normalize else nn.Identity()
        layers = [layer for a, b in zip(dims, dims[1:]) for layer in (nn.Linear(a, b), nn.ELU())]
        self.mlp = nn.Sequential(*layers, nn.Linear(dims[-1], out_dim * (1 + state_std)))
        self.std_shift = math.log(math.expm1(std)) if state_std else 0.0  # softplus(shift) = std initially
        if std and not state_std:  # the actor's
            self.distribution = nn.Module()
            self.distribution.std_param = nn.Parameter(torch.full((out_dim,), std))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        out = self.mlp(self.obs_normalizer(obs))
        return out.chunk(2, -1)[0].tanh() if self.state_std else out

    def dist(self, obs: torch.Tensor, std_range: Tuple[float, float]) -> Tuple[torch.Tensor, torch.Tensor]:
        """The actor's Gaussian (before any tanh): its mean and std, the std per action or per sample and action."""
        out = self.mlp(self.obs_normalizer(obs))
        if self.state_std:
            mean, raw = out.chunk(2, -1)
            return mean, (F.softplus(raw + self.std_shift) + 1e-3).clamp(*std_range)  # a 0.001 floor
        return out, self.distribution.std_param.clamp(*std_range)  # pyright: ignore[reportCallIssue]


def _log_density(z: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return -0.5 * (z * z).sum(-1) - std.log().sum(-1) - 0.5 * z.shape[-1] * math.log(2 * math.pi)


@torch.compile
def _loss(actor: nn.Module, critic: nn.Module, batch: dict, lr: torch.Tensor, config: PPOConfig) -> torch.Tensor:
    """The PPO loss (clipped surrogate, clipped value loss, entropy bonus); adapts `lr` in place from the KL."""
    (mean, std), std0 = actor.dist(batch["obs_a"], config.std_range), batch["std"]  # pyright: ignore[reportCallIssue]
    value = critic(batch["obs_c"]).squeeze(-1)
    # Tanh-squashed actions use their pre-tanh samples: the squash's Jacobian cancels in the ratio and the KL.
    logp = _log_density((batch["act"] - mean) / std, std)
    ratio, adv, clip = (logp - batch["logp"]).exp(), batch["adv"], config.clip
    surrogate = torch.max(-adv * ratio, -adv * ratio.clamp(1 - clip, 1 + clip)).mean()
    clipped = batch["val"] + (value - batch["val"]).clamp(-clip, clip)
    value_loss = torch.max((value - batch["ret"]) ** 2, (clipped - batch["ret"]) ** 2).mean()
    entropy = (0.5 + 0.5 * math.log(2 * math.pi) + std.log()).sum(-1).mean()
    if config.squash_entropy:  # at the rollout's sample, moved with the new mean and std: log(1 - tanh(x) ** 2)
        x = mean + std * (batch["act"] - batch["mean"]) / std0
        entropy = entropy + (2 * (math.log(2) - x - F.softplus(-2 * x))).sum(-1).mean()
    kl = ((std / std0).log() + (std0**2 + (batch["mean"] - mean) ** 2) / (2 * std**2) - 0.5).sum(-1).mean()
    down, up = kl > 2 * config.kl, (kl < config.kl / 2) & (kl > 0)
    lower = (lr / config.lr_factor).clamp(min=config.lr_range[0])
    upper = (lr * config.lr_factor).clamp(max=config.lr_range[1])
    lr.copy_(torch.where(down, lower, torch.where(up, upper, lr)))
    return surrogate + config.value_coef * value_loss - config.entropy * entropy


def _update(
    actor: nn.Module,
    critic: nn.Module,
    opt: Optimizer,
    params: List[List[nn.Parameter]],
    batch: dict,
    last: torch.Tensor,
    config: PPOConfig,
):
    """GAE, PPO epochs at a KL-adaptive learning rate, then the normalizers' update; `params`: per network."""
    vals, done, adv, carry = batch["val"], batch["done"].float(), torch.zeros_like(batch["val"]), 0.0
    gamma, lam = config.gamma, config.lam
    for t in reversed(range(len(vals))):
        nxt = last if t == len(vals) - 1 else vals[t + 1]
        carry = batch["rew"][t] + (1 - done[t]) * gamma * nxt - vals[t] + (1 - done[t]) * gamma * lam * carry
        adv[t] = carry
    batch["ret"] = adv + vals
    batch["adv"] = (adv - adv.mean()) / (adv.std() + 1e-8)
    flat = {k: v.flatten(0, 1) for k, v in batch.items()}
    lr, size = opt.param_groups[0]["lr"], len(flat["val"]) // config.minibatches  # lr is a GPU tensor
    for epoch in range(config.epochs):
        if epoch == 0 or config.reshuffle:
            perm = torch.randperm(size * config.minibatches, device=last.device)
        for i in range(config.minibatches):
            idx = perm[i * size : (i + 1) * size]
            loss = _loss(actor, critic, {k: v[idx] for k, v in flat.items()}, lr, config)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            for group in [params[0] + params[1]] if config.grad_clip == "joint" else params:
                nn.utils.clip_grad_norm_(group, config.max_grad)
            opt.step()
    if config.normalize:  # a normalizer the two share once, with the actor's observations
        actor.obs_normalizer.update(flat["obs_a"])  # pyright: ignore[reportCallIssue]
        if critic.obs_normalizer is not actor.obs_normalizer:
            critic.obs_normalizer.update(flat["obs_c"])  # pyright: ignore[reportCallIssue]


class PPO:
    """An actor and critic for `obs_dims` (actor's, critic's) observations and `actions` actions, trained by `update`.

    Args:
        actor: Replaces the actor `Model`: a graph-safe module with `dist`, `forward` (the mean action) and `state_std`.
        critic: Likewise replaces the critic `Model`; with `config.normalize`, both need `obs_normalizer`, maybe shared.

    Example:
        >>> # xdoctest: +REQUIRES(env:ROBOKIT_CUDA_DOCS)
        >>> from types import SimpleNamespace
        >>> o, f = torch.zeros(64, 3, device="cuda"), torch.zeros(64, dtype=torch.bool, device="cuda")
        >>> env = SimpleNamespace(n=64, obs_dims=(3, 3), bound=torch.ones(2, device="cuda"), names=["a2"],
        ...     obs=lambda: (o, o), step=lambda a: (-(a**2).sum(1), f, f, (a**2).mean()[None]), reset=id, curriculum=id)
        >>> learner = PPO(env.obs_dims, 2)
        >>> for _ in range(10):  # a bandit: actions near 0 pay
        ...     learner.update(env)
        >>> tuple(learner.actor(o).shape)
        (64, 2)
    """

    def __init__(
        self,
        obs_dims: Tuple[int, int],
        actions: int,
        config: PPOConfig = PPOConfig(),
        device: str = "cuda",
        actor: Optional[nn.Module] = None,
        critic: Optional[nn.Module] = None,
    ):
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True  # faster matmuls
        self.config, self.env, self.it, self.graph = config, None, 0, torch.cuda.CUDAGraph()
        state_std, normalize = config.std == "state", config.normalize
        if actor is None:
            actor = Model(obs_dims[0], actions, config.hidden, config.init_std, state_std, normalize)
        if critic is None:
            critic = Model(obs_dims[1], 1, config.hidden, normalize=normalize)
        self.actor, self.critic = actor.to(device), critic.to(device)
        shared = {id(p) for p in self.actor.parameters()}  # the critic's own: a shared parameter is the actor's
        self.params = [list(self.actor.parameters()), [p for p in self.critic.parameters() if id(p) not in shared]]
        lr = torch.tensor(config.lr, device=device)  # a tensor rate lets fused Adam adapt it without a host sync
        self.opt = Adam(chain(*self.params), lr=lr, fused=True, capturable=True)

    def update(self, env: Env):
        """Set `env`'s curriculum to this update's env step, roll out `horizon` steps from where it is, then update.

        Args:
            env: The caller's pick for this update (e.g. coarse, then fine physics); the caller also owns the loop.
        """
        actor, critic, config = self.actor, self.critic, self.config
        if env is not self.env:  # a new env: start from its observations
            self.env, self.obs = env, env.obs()
        env.curriculum(self.it * config.horizon)
        (obs_a, obs_c), steps, self.reward = self.obs, [], torch.zeros(())
        with torch.no_grad():  # rollout
            for _ in range(config.horizon):
                (mean, std), val = actor.dist(obs_a, config.std_range), critic(obs_c).squeeze(-1)  # pyright: ignore
                z = torch.randn_like(mean)
                act = mean + std * z
                reward, terminated, timeout, _ = env.step(act.tanh() if actor.state_std else act)
                self.reward = self.reward + reward.mean() / config.horizon  # its mean reward per step, a GPU scalar
                done = terminated | timeout
                logp, std = _log_density(z, std), std.expand_as(mean)
                steps.append(dict(obs_a=obs_a, obs_c=obs_c, act=act, logp=logp, mean=mean, std=std, val=val))
                steps[-1].update(rew=reward + config.gamma * val * timeout, done=done)  # bootstrap: V(s_t)
                env.reset(done)
                obs_a, obs_c = self.obs = env.obs()
            batch, last = {k: torch.stack([s[k] for s in steps]) for k in steps[0]}, critic(obs_c).squeeze(-1)
        # eager while torch.compile warms up, then the whole update is one CUDA graph over fixed inputs
        if self.it == 3:
            self.static = (*batch.values(), last)
            with torch.cuda.graph(self.graph):
                _update(actor, critic, self.opt, self.params, dict(batch), last, config)  # a copy: it adds keys
        if self.it < 3:
            _update(actor, critic, self.opt, self.params, batch, last, config)
        else:
            for a, b in zip(self.static, (*batch.values(), last)):
                a.copy_(b)
            self.graph.replay()
        self.it += 1

    def train(self, env: Env, minutes: float, coarse_env: Optional[Env] = None, coarse: float = 0.0) -> nn.Module:
        """Update for `minutes` of wall time, then return the actor.

        Args:
            env: What every other update draws.
            coarse_env: Over the first `coarse` fraction, an update draws it with a chance falling linearly from 1 to 0.

        Example:
            >>> # xdoctest: +REQUIRES(env:ROBOKIT_CUDA_DOCS)
            >>> from types import SimpleNamespace
            >>> o, f = torch.zeros(64, 3, device="cuda"), torch.zeros(64, dtype=torch.bool, device="cuda")
            >>> env = SimpleNamespace(n=64, obs_dims=(3, 3), bound=torch.ones(2, device="cuda"), names=["a2"],
            ...     obs=lambda: (o, o), step=lambda a: (-(a**2).sum(1), f, f, (a**2).mean()[None]),
            ...     reset=id, curriculum=id)
            >>> actor = PPO(env.obs_dims, 2).train(env, minutes=0.01)  # a few updates
            >>> tuple(actor(o).shape)
            (64, 2)
        """
        start, draw, mix = time.perf_counter(), random.Random(0).random, coarse * minutes * 60  # s of mixing
        while (elapsed := time.perf_counter() - start) < minutes * 60:
            self.update(coarse_env if coarse_env is not None and draw() * mix >= elapsed else env)
        return self.actor


@torch.no_grad()
def evaluate(env: Env, actor: Callable, metrics: Dict[str, Callable] = {}, steps: int = 1000) -> Dict[str, float]:
    """Score a deterministic `actor` over `steps` env steps from a full reset at the first curriculum stage.

    Returns:
        Each reward term's and `metrics()`'s mean per step (metrics read after resets), and terminations per world.

    Example:
        >>> # xdoctest: +REQUIRES(env:ROBOKIT_CUDA_DOCS)
        >>> from types import SimpleNamespace
        >>> o, f = torch.zeros(64, 3, device="cuda"), torch.zeros(64, dtype=torch.bool, device="cuda")
        >>> env = SimpleNamespace(n=64, obs_dims=(3, 3), bound=torch.ones(2, device="cuda"), names=["a2"],
        ...     obs=lambda: (o, o), step=lambda a: (-(a**2).sum(1), f, f, (a**2).mean()[None]), reset=id, curriculum=id)
        >>> evaluate(env, lambda obs: obs[:, :2], dict(ones=lambda: o[:, 0] + 1), steps=10)
        {'a2': 0.0, 'ones': 1.0, 'terminations': 0.0}
    """
    torch.manual_seed(0)
    env.curriculum(0)
    env.reset(torch.ones(env.n, dtype=torch.bool, device=env.bound.device))
    obs, sums, ends = env.obs(), 0.0, 0.0
    for _ in range(steps):
        _, terminated, timeout, means = env.step(actor(obs[0]))
        ends = ends + terminated.float().mean()
        env.reset(terminated | timeout)
        obs = env.obs()
        sums = sums + torch.cat([means, *(f().mean()[None] for f in metrics.values())])
    score = dict(zip([*env.names, *metrics], (sums / steps).tolist()), terminations=float(ends))
    return {k: round(v, 4) for k, v in score.items()}
