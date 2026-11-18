# pyright: reportCallIssue=false, reportPrivateImportUsage=false
"""FastSAC (SAC, C51 critics, G1 settings) with a 1024-step replay per world and no symmetry.

FastSAC: https://arxiv.org/abs/2512.01996. SAC: https://arxiv.org/abs/1801.01290. C51: https://arxiv.org/abs/1707.06887.
"""

import copy
import math
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import warp as wp
from torch import nn
from torch.optim.adamw import AdamW

from robokit.helpers.rl import Env
from robokit.helpers.rl.ppo import Normalizer


@dataclass(frozen=True)
class FastSACConfig:
    """FastSAC settings for the G1, with a 1024-step replay per world.

    Example:
        >>> # xdoctest: +REQUIRES(module:torch)
        >>> FastSACConfig().batch
        8192
    """

    actor_width: int = 512
    """Width of the actor's first hidden layer; the next two halve it."""
    critic_width: int = 768
    """Width of each critic's first hidden layer; the next two halve it."""
    atoms: int = 101
    """C51 atoms per critic, evenly on [-v_max, v_max]."""
    v_max: float = 20.0
    """Largest return the critics represent."""
    batch: int = 8192
    """Samples per update."""
    updates: int = 8
    """Updates per env step."""
    actor_every: int = 4
    """Critic updates per actor update."""
    gamma: float = 0.97
    """Discount."""
    tau: float = 0.125
    """Target critics' step toward the critics per update."""
    lr: float = 3e-4
    """AdamW learning rate of the actor, the critics and the temperature."""
    weight_decay: float = 0.001
    """AdamW weight decay of the actor and the critics."""
    alpha: float = 0.001
    """Initial entropy temperature."""
    log_std_range: Tuple[float, float] = (-5.0, 0.0)
    """Range of the actor's log-std."""
    buffer: int = 1024
    """Replay env steps per world."""
    start: int = 10
    """Env steps before learning, which fill the replay."""


def _mlp(dim: int, hidden: int, out: int) -> nn.Sequential:
    dims, layers = (dim, hidden, hidden // 2, hidden // 4), []
    for a, b in zip(dims, dims[1:]):
        layers += [nn.Linear(a, b), nn.LayerNorm(b), nn.SiLU()]
    return nn.Sequential(*layers, nn.Linear(dims[-1], out))


class Actor(nn.Module):
    """A tanh-squashed Gaussian actor with its observation normalizer; actions in [-1, 1] times `bound`.

    Example:
        >>> # xdoctest: +REQUIRES(module:torch)
        >>> [tuple(x.shape) for x in Actor(48, torch.ones(12), FastSACConfig()).sample(torch.zeros(5, 48))]
        [(5, 12), (5,)]
    """

    def __init__(self, dim: int, bound: torch.Tensor, config: FastSACConfig):
        super().__init__()
        self.n, self.obs_normalizer, self.log_std_range = len(bound), Normalizer(dim), config.log_std_range
        self.net = _mlp(dim, config.actor_width, 2 * self.n)
        nn.init.normal_(self.net[-1].weight, 0.0, 0.0)  # pyright: ignore[reportArgumentType]
        nn.init.zeros_(self.net[-1].bias)  # pyright: ignore[reportArgumentType]
        self.register_buffer("bound", bound)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """The deterministic action, in the env's units."""
        return self.net(self.obs_normalizer(obs))[..., : self.n].tanh() * self.bound

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """A squashed Gaussian action in [-1, 1] and its log-probability in the env's units."""
        mean, log_std = self.net(self.obs_normalizer(obs)).float().chunk(2, -1)
        low, high = self.log_std_range
        log_std = low + 0.5 * (high - low) * (log_std.tanh() + 1)
        z = torch.randn_like(mean)
        u = (mean + log_std.exp() * z).tanh()
        logp = -0.5 * z**2 - log_std - 0.5 * math.log(2 * math.pi) - (1 - u**2 + 1e-6).log() - self.bound.log()
        return u, logp.sum(-1)


def train(env: Env, minutes: float, config: FastSACConfig = FastSACConfig()) -> Actor:
    """Train an actor with FastSAC on `env` for `minutes` of wall time; returns it, deterministic in the env's units.

    Example:
        >>> # xdoctest: +REQUIRES(env:ROBOKIT_CUDA_DOCS)
        >>> from types import SimpleNamespace
        >>> o, f = torch.zeros(64, 3, device="cuda"), torch.zeros(64, dtype=torch.bool, device="cuda")
        >>> env = SimpleNamespace(n=64, obs_dims=(3, 3), bound=torch.ones(2, device="cuda"), names=["a2"],
        ...     obs=lambda: (o, o), step=lambda a: (-(a**2).sum(1), f, f, (a**2).mean()[None]), reset=id, curriculum=id)
        >>> torch.cuda.set_stream(torch.cuda.Stream())  # envs run on a stream other than the default, as graphs need
        >>> tuple(train(env, minutes=0.2)(o).shape)  # a bandit: actions near 0 pay
        (64, 2)
    """
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    n, (da, dc), k, obs = env.n, env.obs_dims, config.batch // env.n, env.obs()
    device = env.bound.device
    gamma, v_max, atoms = config.gamma, config.v_max, config.atoms
    actor, norm_c = Actor(da, env.bound, config).to(device), Normalizer(dc).to(device)
    critic = nn.ModuleList(_mlp(dc + actor.n, config.critic_width, atoms) for _ in range(2)).to(device)  # two C51s
    target, support = copy.deepcopy(critic).requires_grad_(False), torch.linspace(-v_max, v_max, atoms, device=device)
    log_alpha = torch.tensor(math.log(config.alpha), device=device, requires_grad=True)
    opt_a = AdamW(actor.parameters(), config.lr, weight_decay=config.weight_decay, capturable=True)
    opt_c = AdamW(critic.parameters(), config.lr, weight_decay=config.weight_decay, capturable=True)
    opt_t = AdamW([log_alpha], config.lr, betas=(0.9, 0.95), capturable=True)
    dims = dict(a=da, c=dc, act=actor.n, r=0, a2=da, c2=dc, boot=0)
    buf = {key: torch.zeros(config.buffer, n, *([d] if d else []), device=device) for key, d in dims.items()}
    filled = torch.zeros((), device=device)  # replay size
    pos = torch.zeros(1, dtype=torch.long, device=device)  # next slot
    side, worlds = torch.cuda.Stream(priority=-1), torch.arange(n, device=device)  # learning, the longer branch, first

    def q(net: nn.ModuleList, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:  # both critics' logits
        x = torch.cat([obs, act], -1)
        return torch.stack([m(x) for m in net]).float()

    @torch.compile
    def update(b: Dict[str, torch.Tensor], policy: bool):
        """One critic step and the temperature's, every `actor_every`-th an actor step, then the targets."""
        xc, xc2 = norm_c(b["c"]), norm_c(b["c2"])
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            u2, logp2 = actor.sample(b["a2"])
            reward = b["r"] - gamma * b["boot"] * log_alpha.exp() * logp2
            # the distribution of reward + gamma * Z, Z ~ the target critics, projected onto the support
            x = (reward[:, None] + gamma * b["boot"][:, None] * support).clamp(-v_max, v_max)
            x = (x + v_max) / (2 * v_max / (atoms - 1))  # in atoms
            lo, hi = x.floor(), x.ceil()
            p = q(target, xc2, u2).softmax(-1)
            dist = torch.zeros_like(p).scatter_add_(-1, lo.long().expand_as(p), p * (hi - x + (lo == hi)))  # whole atom
            dist.scatter_add_(-1, hi.long().expand_as(p), p * (x - lo))
        with torch.autocast("cuda", torch.bfloat16):
            loss = -(dist * q(critic, xc, b["act"]).log_softmax(-1)).sum(-1).mean(-1).sum()
        opt_c.zero_grad(set_to_none=True)
        loss.backward()
        opt_c.step()
        opt_t.zero_grad(set_to_none=True)
        (-log_alpha.exp() * logp2.mean()).backward()  # target entropy 0
        opt_t.step()
        if policy:
            with torch.autocast("cuda", torch.bfloat16):
                u, logp = actor.sample(b["a"])
                value = (q(critic, xc, u).softmax(-1) * support).sum(-1)
                loss = (log_alpha.exp().detach() * logp - value.mean(0)).mean()
            opt_a.zero_grad(set_to_none=True)
            loss.backward()
            opt_a.step()
        torch._foreach_lerp_(list(target.parameters()), list(critic.parameters()), config.tau)

    @torch.compile
    def act(obs: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", torch.bfloat16):
            return actor.sample(obs)[0]

    @torch.no_grad()
    def step(learning: bool):
        """Act, then physics and resets alongside learning on the replay so far, then the replay write; in place."""
        obs_a, obs_c = obs
        actor.obs_normalizer.update(obs_a)
        norm_c.update(obs_c)
        u = act(obs_a)
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.enable_grad():  # pyright: ignore[reportArgumentType]
            if learning:  # sample `updates` batches at once (a batch per world), then update on each in turn
                idx = (torch.rand(config.updates * k, n, device=device) * filled).long()
                batch = {key: v[idx, worlds].flatten(0, 1) for key, v in buf.items()}
                for i in range(config.updates):
                    minibatch = {key: v[i * k * n : (i + 1) * k * n] for key, v in batch.items()}
                    update(minibatch, i % config.actor_every == 1)
        reward, terminated, timeout, _ = env.step(u * actor.bound)  # pyright: ignore[reportOperatorIssue]
        final_a, final_c = env.obs()  # where a timed-out episode ended, for its bootstrap
        env.reset(terminated | timeout)
        obs2 = env.obs()
        torch.cuda.current_stream().wait_stream(side)
        new = dict(a=obs_a, c=obs_c, act=u, r=reward, a2=final_a, c2=final_c, boot=(~terminated).float())
        for key, value in new.items():
            buf[key].index_copy_(0, pos, value[None])
        pos.add_(1).remainder_(config.buffer)
        filled.add_(1).clamp_(max=config.buffer)
        for o, o2 in zip(obs, obs2):
            o.copy_(o2)

    graph, it, start = torch.cuda.CUDAGraph(), 0, time.perf_counter()
    while time.perf_counter() - start < minutes * 60:
        env.curriculum(it)
        if it == config.start + 3:  # eager while the replay fills and torch.compile warms up, then one graph per step
            with torch.cuda.graph(graph):  # on torch's capture stream, which Warp then also records on
                with wp.ScopedStream(wp.stream_from_torch(), sync_enter=False, sync_exit=False):
                    step(True)
        step(it >= config.start) if it < config.start + 3 else graph.replay()
        it += 1
    return actor
