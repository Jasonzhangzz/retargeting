"""Real replay -> multi-step world model -> differentiable imagination training.

Uses Dex-Supermimic's state integrator, replay masks, networks and normalizers.
Contact forces are continuous regression targets, not Bernoulli contacts.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dex_supermimic.rl.dex_supermimic.buffer import ReplayBuffer
from dex_supermimic.rl.dex_supermimic.common import RunningNorm
from dex_supermimic.rl.dex_supermimic.networks import Actor, Critic, FailureHead, WorldModel
from dex_supermimic.rl.dex_supermimic.state import difference

from .contact import AUX_DIM
from .environment import CONTEXT_DIM, GraspHoldEnv, control, make_env_cfg, observation, rewards

FORMAT = 5
FEATURE_SCHEMA = "grasp_hold_features:1"


@dataclass
class TrainConfig:
    manifest: str = ""
    task: str = "hand2-grasp-hold"
    device: str = "cuda:3"
    num_envs: int = 644
    seed: int = 42
    max_iterations: int = 8000
    max_hours: float = 12.
    buffer_steps: int = 1000
    warmup_steps: int = 64
    wm_window: int = 8
    wm_batches: int = 10
    wm_batch: int = 2048
    wm_hidden: tuple = (2048, 2048, 2048)
    wm_only_iters: int = 500
    pi_batch: int = 1024
    horizon: int = 32
    critic_epochs: int = 5
    critic_minibatches: int = 4
    lr: float = 2e-4
    gamma: float = .99
    lam: float = .95
    failure_clip: float = .1
    save_interval: int = 1000
    validation_repeats: int = 2
    test_repeats: int = 10
    log_interval: int = 50
    log_dir: str = "logs/dex_supermimic/hand2-grasp-hold"
    checkpoint: str | None = None
    case_ids: tuple | None = None
    amp: bool = True
    final_evaluation: bool = True


class GraspWorldModel(WorldModel):
    def __init__(self, cfg):
        super().__init__(58 + CONTEXT_DIM, AUX_DIM, cfg.wm_hidden, cfg.amp)
        self.aux_norm = RunningNorm(AUX_DIM - 1, eps=1e-3)
        with torch.no_grad():
            self.net[-1].bias[52:] = 0

    def predict(self, x, target, ctx):
        x1, raw = super().forward(x, target, torch.cat((x, ctx), -1))
        encoded = raw[..., :-1] * self.aux_norm.std() + self.aux_norm.mean
        continuous = torch.expm1(encoded.clamp(-12, 8)).clamp_min(0)
        return x1, torch.cat((continuous, raw[..., -1:].sigmoid()), -1), raw


def critic_features(x, ctx, previous, aux):
    return torch.cat((x, ctx, previous, torch.log1p(aux[..., :12].clamp_min(0))), -1)


def advance(ctx):
    return torch.cat((ctx[..., :-1], ctx[..., -1:] + 1 / 110), -1)


def lambda_returns(rewards_, values, discounts, lam):
    result = []
    value = values[-1]
    for i in range(len(rewards_) - 1, -1, -1):
        value = rewards_[i] + discounts[i] * ((1 - lam) * values[i + 1] + lam * value)
        result.append(value)
    return torch.stack(result[::-1])


class GraspRunner:
    def __init__(self, cfg):
        self.cfg = cfg
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        self.env = env = GraspHoldEnv(make_env_cfg(cfg.num_envs, manifest=cfg.manifest,
                                                 case_ids=cfg.case_ids), cfg.device, cfg.seed)
        self.device = torch.device(cfg.device)
        self.obs_dim = env.observe().shape[-1]
        self.actor = Actor(self.obs_dim).to(self.device)
        self.wm = GraspWorldModel(cfg).to(self.device)
        self.critic = Critic(58 + CONTEXT_DIM + 20 + 12, amp=cfg.amp).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)
        self.failure = FailureHead(CONTEXT_DIM).to(self.device)
        self.modules = dict(actor=self.actor, wm=self.wm, critic=self.critic,
                            target_critic=self.target_critic, failure=self.failure)
        self.optimizers = {name: torch.optim.Adam(module.parameters(), lr=cfg.lr)
                           for name, module in self.modules.items() if name != "target_critic"}
        # Dataset seed splits provide the reference holdout. Splitting worlds here
        # would omit entire cases from dynamics training with static variants.
        self.buffer = ReplayBuffer(cfg.buffer_steps, cfg.num_envs, self.device, holdout_fraction=0)
        self.iteration = 0
        self.collected = 0
        self.elapsed_before = 0.
        self.folder = Path(cfg.log_dir)
        self.folder.mkdir(parents=True, exist_ok=True)
        # TensorBoard's supported no-TensorFlow mode avoids importing a second
        # GPU framework just to write PyTorch scalar event files.
        import sys
        import types
        sys.modules.setdefault("tensorboard.compat.notf", types.ModuleType("tensorboard.compat.notf"))
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(str(self.folder / "tensorboard"))
        self.started = time.monotonic()
        self.best_score = (-1., -1.)
        if cfg.checkpoint:
            self.load(cfg.checkpoint)
        (self.folder / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
        import hashlib
        self.source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in Path(__file__).parent.glob("*.py")}
        (self.folder / "source_hashes.json").write_text(json.dumps(self.source_hashes, indent=2))

    @torch.no_grad()
    def collect(self, count):
        stats = []
        for _ in range(count):
            obs = self.env.observe()
            self.actor.norm.update(obs)
            action = (self.actor(obs) + .1 * torch.randn((self.env.num_envs, 20), device=self.device)).clamp(-1, 1)
            tr = self.env.step(action)
            valid = tr["valid"]
            if valid.any():
                self.wm.norm.update(self.wm.inputs(torch.cat((tr["x"][valid], tr["context"][valid]), -1), tr["u"][valid]))
                self.wm.delta_norm.update(difference(tr["x"][valid], tr["x1"][valid]))
                self.wm.aux_norm.update(torch.log1p(tr["aux"][valid, :-1].clamp_min(0)))
                self.failure.norm.update(self.failure.features(tr["x1"][valid], tr["context"][valid]))
                self.critic.norm.update(critic_features(tr["x"][valid], tr["context"][valid], tr["previous"][valid], tr["old_aux"][valid]))
            replay = {key: tr[key] for key in ("x", "x1", "context", "previous", "previous_target", "action", "u", "aux", "old_aux", "done", "terminated", "valid")}
            for key, value in replay.items():
                if value.is_floating_point():
                    replay[key] = torch.where(valid.reshape(-1, *([1] * (value.ndim - 1))), value, torch.zeros_like(value))
            self.buffer.add(replay)
            # Same pure function must reproduce simulator-side rewards exactly.
            parity = (rewards(tr["x1"], tr["context"], tr["aux"], tr["action"], tr["previous"])[0] - tr["reward"]).abs().max()
            stats.append(torch.stack((tr["reward"].mean(), tr["terminated"].float().mean(), parity)))
            self.collected += self.env.num_envs
            self.env.reset(tr["done"].nonzero(as_tuple=True)[0], perturb=True)
        return torch.stack(stats).mean(0)

    def fit_world_model(self):
        c = self.cfg
        totals = []
        for _ in range(c.wm_batches):
            w = self.buffer.sample(c.wm_batch, c.wm_window, failure_frac=.25, failure_horizon=16)
            x = w["x"][:, 0]
            state_losses, contact_losses, failure_losses = [], [], []
            for j in range(c.wm_window):
                ctx = w["context"][:, j]
                x1, aux, raw = self.wm.predict(x, w["u"][:, j], ctx)
                scale = torch.maximum(self.wm.delta_norm.std(), self.wm.error_floor)
                state_losses.append((difference(x1, w["x1"][:, j]) / scale).square().mean(-1))
                target = (torch.log1p(w["aux"][:, j, :-1].clamp_min(0)) - self.wm.aux_norm.mean) / self.wm.aux_norm.std()
                continuous = F.smooth_l1_loss(raw[:, :-1], target, reduction="none").mean(-1)
                binary = F.binary_cross_entropy_with_logits(raw[:, -1], w["aux"][:, j, -1], reduction="none")
                contact_losses.append(continuous + binary)
                failure_losses.append(F.binary_cross_entropy_with_logits(
                    self.failure(self.failure.features(x1.detach(), ctx)), w["terminated"][:, j].float(), reduction="none"))
                x = x1
            mask = w["mask"] * w["importance"]
            state = (torch.stack(state_losses, 1) * mask).sum() / mask.sum().clamp_min(1)
            contact = (torch.stack(contact_losses, 1) * mask).sum() / mask.sum().clamp_min(1)
            fail = (torch.stack(failure_losses, 1) * mask).sum() / mask.sum().clamp_min(1)
            loss = state + .1 * contact + fail
            self.optimizers["wm"].zero_grad(set_to_none=True)
            self.optimizers["failure"].zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.wm.parameters(), 100.)
            nn.utils.clip_grad_norm_(self.failure.parameters(), 100.)
            self.optimizers["wm"].step()
            self.optimizers["failure"].step()
            totals.append(torch.stack((state.detach(), contact.detach(), fail.detach())))
        return torch.stack(totals).mean(0)

    def improve_policy(self):
        c = self.cfg
        w = self.buffer.sample(c.pi_batch, 1)
        x, ctx = w["x"][:, 0], w["context"][:, 0]
        previous, target = w["previous"][:, 0], w["previous_target"][:, 0]
        aux = w["old_aux"][:, 0].detach()
        features, rs, values, discounts = [], [], [], []
        frozen = (self.wm, self.failure, self.critic)
        for module in frozen:
            module.requires_grad_(False)
        try:
            for _ in range(c.horizon):
                obs = observation(x, ctx, previous, aux)
                feature = critic_features(x, ctx, previous, aux)
                features.append(feature)
                values.append(self.target_critic(feature))
                a = self.actor(obs)
                target = control(a, target, ctx, self.env.lower, self.env.upper)
                x1, aux1, _ = self.wm.predict(x, target, ctx)
                r, _ = rewards(x1, ctx, aux1, a, previous)
                terminal = self.failure(self.failure.features(x1, ctx)).sigmoid().clamp(max=c.failure_clip)
                rs.append(r)
                discounts.append(c.gamma * (1 - terminal))
                x, aux, previous, ctx = x1, aux1, a, advance(ctx)
            values.append(self.target_critic(critic_features(x, ctx, previous, aux)))
            returns = lambda_returns(rs, values, discounts, c.lam)
            actor_loss = -returns[0].mean()
            self.optimizers["actor"].zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_grad = nn.utils.clip_grad_norm_(self.actor.parameters(), 100.)
            if not torch.isfinite(actor_grad):
                raise RuntimeError("Nonfinite imagination actor gradient")
            self.optimizers["actor"].step()
        finally:
            for module in frozen:
                module.requires_grad_(True)
        inputs = torch.stack(features).detach().flatten(0, 1)
        targets = returns.detach().flatten()
        self.critic.norm.update(inputs)
        critic_losses = []
        for _ in range(c.critic_epochs):
            order = torch.randperm(len(inputs), device=self.device)
            for ids in order.chunk(c.critic_minibatches):
                loss = F.mse_loss(self.critic(inputs[ids]), targets[ids])
                self.optimizers["critic"].zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 100.)
                self.optimizers["critic"].step()
                critic_losses.append(loss.detach())
        with torch.no_grad():
            for source, target in zip(self.critic.parameters(), self.target_critic.parameters()):
                target.lerp_(source, .005)
            self.target_critic.norm.load_state_dict(self.critic.norm.state_dict())
        return float(actor_loss.detach()), float(torch.stack(critic_losses).mean()), float(actor_grad)

    def elapsed(self):
        return self.elapsed_before + time.monotonic() - self.started

    def save(self, name):
        path = self.folder / name
        tmp = path.with_suffix(".tmp")
        torch.save(dict(format_version=FORMAT, feature_schema=FEATURE_SCHEMA, config=asdict(self.cfg),
                        dataset_signature=self.env.dataset.signature, iteration=self.iteration,
                        adapter_source_sha256=self.source_hashes,
                        collected=self.collected, elapsed=self.elapsed(), best_score=self.best_score,
                        obs_dim=self.obs_dim, modules={k: v.state_dict() for k, v in self.modules.items()},
                        optimizers={k: v.state_dict() for k, v in self.optimizers.items()},
                        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(self.device),
                        numpy_rng=self.env.rng.bit_generator.state), tmp)
        tmp.replace(path)
        return path

    def load(self, path):
        d = torch.load(path, map_location=self.device, weights_only=False)
        if d["format_version"] != FORMAT or d["feature_schema"] != FEATURE_SCHEMA:
            raise ValueError("Incompatible grasp checkpoint format")
        if d["dataset_signature"] != self.env.dataset.signature:
            raise ValueError("Checkpoint and grasp cache signatures differ")
        for key, module in self.modules.items():
            module.load_state_dict(d["modules"][key])
        for key, optimizer in self.optimizers.items():
            optimizer.load_state_dict(d["optimizers"][key])
        self.iteration, self.collected = d["iteration"], d["collected"]
        self.elapsed_before = d["elapsed"]
        self.best_score = tuple(d["best_score"])
        torch.set_rng_state(d["torch_rng"].cpu())
        torch.cuda.set_rng_state(d["cuda_rng"].cpu(), self.device)
        self.env.rng.bit_generator.state = d["numpy_rng"]

    def learn(self):
        from .evaluate import evaluate
        from tqdm import tqdm
        import sys
        c = self.cfg
        progress = tqdm(total=c.max_iterations, initial=self.iteration, desc="hand2-grasp-hold",
                        unit="iter", file=sys.stderr, ncols=110, mininterval=30, miniters=1)
        status = dict(state="warming_up", pid=__import__("os").getpid(), task=c.task)
        def report(**extra):
            status.update(iteration=self.iteration, elapsed_seconds=self.elapsed(), collected_steps=self.collected, **extra)
            tmp = self.folder / "status.tmp"
            tmp.write_text(json.dumps(status, indent=2))
            tmp.replace(self.folder / "status.json")
        report()
        try:
            self.collect(c.warmup_steps)
            while self.iteration < c.max_iterations and self.elapsed() < c.max_hours * 3600:
                self.env.force_scale = min(1., self.iteration / 2000)
                real = self.collect(2 + self.iteration % 2)
                world = self.fit_world_model()
                policy = self.improve_policy() if self.iteration >= c.wm_only_iters else (0., 0., 0.)
                self.iteration += 1
                progress.update(1)
                if self.iteration % c.log_interval == 0 or self.iteration == 1:
                    stats = dict(iteration=self.iteration, elapsed=self.elapsed(), real_reward=float(real[0]),
                                 real_failure=float(real[1]), reward_parity=float(real[2]),
                                 wm_state=float(world[0]), wm_contact=float(world[1]), wm_failure=float(world[2]),
                                 actor_loss=policy[0], critic_loss=policy[1], actor_grad=policy[2])
                    print(json.dumps(stats), flush=True)
                    with (self.folder / "metrics.jsonl").open("a") as out:
                        out.write(json.dumps(stats) + "\n")
                    for k, v in stats.items():
                        self.writer.add_scalar(k, v, self.iteration)
                    self.writer.flush()
                    progress.set_postfix(phase="wm" if self.iteration <= c.wm_only_iters else "actor",
                                         wm=f"{float(world[0]):.3g}", reward=f"{float(real[0]):.3g}", refresh=False)
                    report(state="training", latest=stats)
                if self.iteration % c.save_interval == 0:
                    self.save(f"model_{self.iteration}.pt")
                    validation = evaluate(self.env, self.actor, split="validation", repeats=c.validation_repeats,
                                          output=self.folder / f"validation_{self.iteration}")
                    score = (validation["minimum_case_success_rate"], validation["macro_success_rate"])
                    if score >= self.best_score:
                        self.best_score = score
                        self.save("best.pt")
                    # Evaluation reset every world outside replay collection.
                    # Cut the previous sequence before collecting new references.
                    self.buffer.data["done"][(self.buffer.ptr - 1) % self.buffer.capacity].fill_(True)
                    self.env.reset(split="train", perturb=True)
            final = self.save("last.pt")
            if not (self.folder / "best.pt").exists():
                self.save("best.pt")
            report(state="evaluating", checkpoint=str(final))
            if c.final_evaluation:
                selected = torch.load(self.folder / "best.pt", map_location=self.device, weights_only=False)
                self.actor.load_state_dict(selected["modules"]["actor"])
                evaluate(self.env, None, split="test", repeats=c.test_repeats, output=self.folder / "baseline")
                result = evaluate(self.env, self.actor, split="test", repeats=c.test_repeats, output=self.folder / "evaluation")
                report(state="completed", evaluation=result)
            else:
                report(state="completed_short_run")
        except BaseException as error:
            self.save("interrupted.pt")
            report(state="failed", error=repr(error))
            raise
        finally:
            progress.close()
            self.writer.close()
            self.env.close()


def parse_args(argv=None):
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    initial = argparse.ArgumentParser(add_help=False)
    initial.add_argument("--config", type=Path)
    known, _ = initial.parse_known_args(argv)
    loaded = json.loads(known.config.read_text()) if known.config else {}
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path)
    defaults = TrainConfig(**loaded)
    for name, value in asdict(defaults).items():
        flag = "--" + name.replace("_", "-")
        if name == "wm_hidden":
            p.add_argument(flag, type=int, nargs="+", default=value)
        elif name == "case_ids":
            p.add_argument(flag, type=int, nargs="+", default=None)
        elif isinstance(value, bool):
            p.add_argument(flag, action=argparse.BooleanOptionalAction, default=value)
        else:
            p.add_argument(flag, type=type(value) if value is not None else str, default=value)
    args = vars(p.parse_args(argv))
    args.pop("config")
    if "log_dir" not in loaded and not any(a == "--log-dir" or a.startswith("--log-dir=") for a in argv):
        from datetime import datetime
        args["log_dir"] = str(Path(args["checkpoint"]).resolve().parent if args["checkpoint"] else
                              Path(args["log_dir"]) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    if not args["manifest"]:
        args["manifest"] = make_env_cfg()["manifest"]
    cfg = TrainConfig(**args)
    if cfg.task != "hand2-grasp-hold" or cfg.warmup_steps < cfg.wm_window:
        p.error("Require hand2-grasp-hold and warmup_steps >= wm_window")
    return cfg


def main(argv=None):
    cfg = parse_args(argv)
    try:
        runner = GraspRunner(cfg)
        runner.learn()
    except (torch.OutOfMemoryError, RuntimeError) as error:
        oom = isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower() or "CUDA_ERROR_OUT_OF_MEMORY" in str(error)
        if cfg.num_envs != 644 or not oom:
            raise
        failure = repr(error)
    else:
        return
    # Warp owns graph allocations outside Torch's allocator. Replace this
    # process so all allocations are released before trying two worlds/case.
    # A training OOM resumes the saved optimizer state and cumulative budget.
    import os
    import sys
    folder = Path(cfg.log_dir).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder / "interrupted.pt"
    if checkpoint.exists():
        cfg.checkpoint = str(checkpoint)
    cfg.num_envs = 322
    fallback = folder / "config-322-fallback.json"
    fallback.write_text(json.dumps(asdict(cfg), indent=2))
    (folder / "oom-fallback.json").write_text(json.dumps(dict(
        error=failure, previous_num_envs=644, num_envs=322,
        checkpoint=cfg.checkpoint), indent=2))
    print("[grasp] CUDA memory exhausted; restarting with 322 worlds (all 161 cases)", flush=True)
    os.execv(sys.executable, [sys.executable, "-B", "-m", "retarget.integrations.supermimic.train",
                            "--config", str(fallback)])


if __name__ == "__main__":
    main()
