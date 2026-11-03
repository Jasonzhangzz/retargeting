"""Two-stage denoising pose autoencoders and conditional flow matching.

The encoder sees only the observation. Neither clean targets nor corruption
metadata are available to the velocity network at inference. All coordinates
are joint-limit-normalized finger angles; the wrist is outside this model.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .manifold import device_name, seed_everything, tensors, write_json

ARCHITECTURE = "pose_conditioned_flow_v2"
FLOW_METHODS = ("diffusion_vq", "diffusion_vae")


@dataclass
class FlowConfig:
    steps: int = 10000
    batch_size: int = 256
    lr: float = 1e-3
    validation_every: int = 250
    patience: int = 12
    min_steps: int = 2500
    latent_dim: int = 16
    codebook_size: int = 128
    hidden_dim: int = 256
    flow_steps: int = 32
    max_rounds: int = 5
    sigma_min: float = .01
    sigma_max: float = .35
    reconstruction_weight: float = 1.
    cross_size_weight: float = 1.
    consistency_weight: float = .1
    contrastive_weight: float = .1
    contrastive_temperature: float = .1
    kl_weight: float = 1e-4
    codebook_weight: float = 1.
    commitment_weight: float = .25

    def validate(self):
        for key in ("steps", "batch_size", "validation_every", "patience", "latent_dim",
                    "codebook_size", "hidden_dim", "flow_steps", "max_rounds"):
            value = getattr(self, key)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.batch_size < 4 or self.batch_size % 2 or not isinstance(self.min_steps, int) or self.min_steps < 0:
            raise ValueError("Need even batch_size >= 4 and nonnegative integer min_steps")
        if not (0 < self.sigma_min <= self.sigma_max < math.inf):
            raise ValueError("Require 0 < sigma_min <= sigma_max, both finite")
        for key in ("lr", "contrastive_temperature"):
            if not 0 < getattr(self, key) < math.inf:
                raise ValueError(f"{key} must be positive and finite")
        for key in ("reconstruction_weight", "cross_size_weight", "consistency_weight",
                    "contrastive_weight", "kl_weight", "codebook_weight", "commitment_weight"):
            if not 0 <= getattr(self, key) < math.inf:
                raise ValueError(f"{key} must be finite and nonnegative")


def finger_corruption(clean, generator, *, max_rounds=5, sigma_min=.01, sigma_max=.35,
                      finger_counts=None, rounds=None):
    """Sum independent Gaussian increments on independently selected fingers.

    Optional counts/rounds make boundary cases and inspection reproducible.
    Metadata is for diagnostics only, never a network condition.
    """
    if clean.ndim != 2 or clean.shape[1] != 20 or not torch.isfinite(clean).all():
        raise ValueError("Expected finite (N,20) poses")
    if not isinstance(max_rounds, int) or max_rounds < 1 or not 0 < sigma_min <= sigma_max < math.inf:
        raise ValueError("Invalid Gaussian corruption configuration")
    n, device = len(clean), clean.device
    counts = (torch.randint(0, 6, (n,), device=device, generator=generator)
              if finger_counts is None else torch.as_tensor(finger_counts, device=device))
    if counts.shape != (n,) or ((counts < 0) | (counts > 5) | (counts != counts.long())).any():
        raise ValueError("finger_counts must be integer (N,) in [0,5]")
    ranks = torch.rand(n, 5, device=device, generator=generator).argsort(1).argsort(1)
    mask = ranks < counts[:, None]
    nr = (torch.randint(1, max_rounds + 1, (n, 5), device=device, generator=generator)
          if rounds is None else torch.as_tensor(rounds, device=device))
    if nr.shape != (n, 5) or ((nr < 1) | (nr > max_rounds) | (nr != nr.long())).any():
        raise ValueError("rounds must be integer (N,5) in [1,max_rounds]")
    nr = nr.long() * mask
    sigma = sigma_min + (sigma_max - sigma_min) * torch.rand(
        n, device=device, generator=generator, dtype=clean.dtype)
    active = torch.arange(max_rounds, device=device)[:, None, None] < nr[None]
    increments = torch.randn(max_rounds, n, 5, 4, device=device, dtype=clean.dtype,
                             generator=generator) * sigma[None, :, None, None]
    increments = increments * active[..., None]
    noise = increments.sum(0).reshape(n, 20)
    return clean + noise, dict(mask=mask, rounds=nr, sigma_step=sigma, noise=noise,
                              increments=increments.reshape(max_rounds, n, 20),
                              sigma_fingers=sigma[:, None] * nr.sqrt())


def mlp(inputs, output, hidden=256, layers=2):
    modules = []
    for _ in range(layers):
        modules.extend((nn.Linear(inputs, hidden), nn.SiLU()))
        inputs = hidden
    return nn.Sequential(*modules, nn.Linear(inputs, output))


class PoseAutoencoder(nn.Module):
    def __init__(self, kind, primitives, log_center, log_scale, config):
        super().__init__()
        if kind not in ("vq", "vae"):
            raise ValueError("latent kind must be vq or vae")
        self.kind, self.config = kind, config
        self.log_center, self.log_scale = float(log_center), float(log_scale)
        self.encoder = mlp(20, config.latent_dim * (2 if kind == "vae" else 1), config.hidden_dim)
        self.primitive_embedding = nn.Embedding(primitives, 8)
        self.decoder = mlp(config.latent_dim + 8 + 1, 20, config.hidden_dim)
        if kind == "vq":
            self.codebook = nn.Embedding(config.codebook_size, config.latent_dim)
            nn.init.normal_(self.codebook.weight, std=.1)

    def encode(self, observed, *, sample=False, generator=None):
        raw = self.encoder(observed)
        if self.kind == "vae":
            mean, logvar = raw.chunk(2, -1)
            logvar = logvar.clamp(-12, 8)
            latent = mean
            if sample:
                latent = mean + (.5 * logvar).exp() * torch.randn(
                    mean.shape, dtype=mean.dtype, device=mean.device, generator=generator)
            return latent, dict(features=mean, mean=mean, logvar=logvar,
                                kl=.5 * (mean.square() + logvar.exp() - 1 - logvar).mean())
        distances = (raw.square().sum(-1, keepdim=True) + self.codebook.weight.square().sum(-1)
                     - 2 * raw @ self.codebook.weight.T)
        indices = distances.argmin(-1)
        quantized = self.codebook(indices)
        latent = raw + (quantized - raw).detach()
        return latent, dict(features=raw, indices=indices,
                            codebook=F.mse_loss(quantized, raw.detach()),
                            commitment=F.mse_loss(raw, quantized.detach()))

    def decode(self, latent, primitives, sizes):
        size_feature = ((sizes.log() - self.log_center) / self.log_scale)[:, None]
        return self.decoder(torch.cat((latent, self.primitive_embedding(primitives), size_feature), -1))


def supervised_contrastive(features, labels, temperature=.1):
    """Supervised contrast on encoder features, with self matches excluded."""
    features = F.normalize(features, dim=-1)
    logits = features @ features.T / temperature
    eye = torch.eye(len(features), device=features.device, dtype=torch.bool)
    positives = labels[:, None].eq(labels[None]) & ~eye
    logits = logits.masked_fill(eye, -torch.inf)
    log_probability = logits - torch.logsumexp(logits, -1, keepdim=True)
    count = positives.sum(-1)
    loss = -log_probability.masked_fill(~positives, 0).sum(-1) / count.clamp_min(1)
    return loss[count > 0].mean() if (count > 0).any() else features.sum() * 0


class VelocityNetwork(nn.Module):
    def __init__(self, types, primitives, log_center, log_scale, config):
        super().__init__()
        self.type_embedding = nn.Embedding(types, 32)
        self.primitive_embedding = nn.Embedding(primitives, 8)
        self.log_center, self.log_scale = float(log_center), float(log_scale)
        self.register_buffer("frequencies", torch.exp(torch.linspace(0, math.log(1000), 16)))
        self.net = mlp(20 + config.latent_dim + 32 + 8 + 1 + 32, 20, config.hidden_dim, layers=3)

    def forward(self, pose, times, latent, types, primitives, sizes):
        angles = times[:, None] * self.frequencies
        size_features = ((sizes.log() - self.log_center) / self.log_scale)[:, None]
        conditions = (pose, latent, self.type_embedding(types), self.primitive_embedding(primitives),
                      size_features, angles.sin(), angles.cos())
        return self.net(torch.cat(conditions, -1))


class PoseFlowModel(nn.Module):
    def __init__(self, kind, data, config):
        super().__init__()
        self.kind, self.config = kind, config
        self.method = f"diffusion_{kind}"
        records = data.meta["groups"]
        self.type_names = sorted({r["template"] for r in records})
        self.primitive_names = sorted({r["primitive"] for r in records})
        self.register_buffer("group_types", torch.tensor([self.type_names.index(r["template"]) for r in records]))
        self.register_buffer("group_primitives", torch.tensor(
            [self.primitive_names.index(r["primitive"]) for r in records]))
        self.autoencoder = PoseAutoencoder(kind, len(self.primitive_names), data.log_center, data.log_scale, config)
        self.velocity = VelocityNetwork(len(self.type_names), len(self.primitive_names),
                                        data.log_center, data.log_scale, config)

    def freeze_autoencoder(self):
        self.autoencoder.requires_grad_(False)
        for parameter in self.autoencoder.parameters():
            parameter.grad = None
        self.autoencoder.eval()

    def train(self, mode=True):
        super().train(mode)
        if not any(p.requires_grad for p in self.autoencoder.parameters()):
            self.autoencoder.eval()
        return self

    def corrupt(self, clean, generator):
        return finger_corruption(clean, generator, max_rounds=self.config.max_rounds,
                                 sigma_min=self.config.sigma_min, sigma_max=self.config.sigma_max)

    def autoencoder_objective(self, clean, groups, sizes, partners, generator):
        observed, _ = self.corrupt(clean, generator)
        latent, aux = self.autoencoder.encode(observed, sample=True, generator=generator)
        clean_latent, clean_aux = self.autoencoder.encode(clean)
        primitives = self.group_primitives[groups]
        reconstruction = self.autoencoder.decode(latent, primitives, sizes)
        clean_reconstruction = self.autoencoder.decode(clean_latent, primitives, sizes)
        cross = self.autoencoder.decode(latent, primitives[partners], sizes[partners])
        losses = dict(reconstruction=.5 * (F.mse_loss(reconstruction, clean)
                                           + F.mse_loss(clean_reconstruction, clean)),
                      cross_size=F.mse_loss(cross, clean[partners]),
                      consistency=F.mse_loss(aux["features"], clean_aux["features"]),
                      contrastive=supervised_contrastive(
                          torch.cat((aux["features"], clean_aux["features"])),
                          self.group_types[groups].repeat(2), self.config.contrastive_temperature))
        for name in (("kl",) if self.kind == "vae" else ("codebook", "commitment")):
            losses[name] = .5 * (aux[name] + clean_aux[name])
        return sum(getattr(self.config, f"{name}_weight") * loss for name, loss in losses.items()), losses

    def flow_objective(self, clean, groups, sizes, generator):
        observed, _ = self.corrupt(clean, generator)
        with torch.no_grad():
            latent, _ = self.autoencoder.encode(observed)
        times = torch.rand(len(clean), device=clean.device, generator=generator)
        current = torch.lerp(observed, clean, times[:, None])
        prediction = self.velocity(current, times, latent, self.group_types[groups],
                                   self.group_primitives[groups], sizes)
        return F.mse_loss(prediction, clean - observed)

    @torch.no_grad()
    def denoise(self, observed, groups, sizes, *, steps=None):
        steps = self.config.flow_steps if steps is None else steps
        if not isinstance(steps, int) or steps < 1:
            raise ValueError("Integration steps must be positive")
        latent, _ = self.autoencoder.encode(observed)
        types, primitives = self.group_types[groups], self.group_primitives[groups]
        x, dt = observed.clone(), 1. / steps
        times = torch.zeros(len(x), device=x.device, dtype=x.dtype)
        for i in range(steps):
            times.fill_(i * dt)
            first = self.velocity(x, times, latent, types, primitives, sizes)
            times.fill_((i + 1) * dt)
            second = self.velocity(x + dt * first, times, latent, types, primitives, sizes)
            x = x + .5 * dt * (first + second)
        return x


def dataset_digest(data):
    digest = hashlib.sha256()
    for name in ("dataset.json", "dataset.npz"):
        digest.update((data.directory / name).read_bytes())
    return digest.hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_registry(data):
    path = data.directory / "models.json"
    return json.loads(path.read_text()) if path.exists() else {}


def register_model(data, method, seed, score):
    registry = read_registry(data)
    registry.update(architecture=ARCHITECTURE)
    if method == "diffusion_vq" or "default_diffusion" not in registry:
        registry["default_diffusion"] = method
    registry.setdefault("runs", {})[f"{method}_seed{seed}"] = dict(validation_mse=score)
    candidates = {int(key.rsplit("seed", 1)[1]): value["validation_mse"]
                  for key, value in registry["runs"].items() if key.startswith(method + "_seed")}
    registry.setdefault("selected_seeds", {})[method] = min(candidates, key=candidates.get)
    write_json(data.directory / "models.json", registry)


def load_flow_model(data, method, seed=None, device="auto", *, stage="flow", path=None):
    device = device_name(device)
    if method not in FLOW_METHODS or stage not in ("flow", "autoencoder"):
        raise ValueError("Unknown flow model or training stage")
    seed = data.model_seed(method, seed)
    path = Path(path) if path is not None else data.directory / "models" / f"{method}_seed{seed}" / (
        "best.pt" if stage == "flow" else "autoencoder.pt")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if (checkpoint.get("architecture") != ARCHITECTURE or checkpoint.get("stage") != stage
            or checkpoint.get("method") != method or checkpoint.get("dataset_sha256") != dataset_digest(data)):
        raise ValueError("Incompatible architecture, stage, latent kind or dataset in checkpoint")
    config = FlowConfig(**checkpoint["config"])
    config.validate()
    model = PoseFlowModel(method.removeprefix("diffusion_"), data, config).to(device)
    if checkpoint.get("condition_schema") != dict(types=model.type_names, primitives=model.primitive_names):
        raise ValueError("Incompatible grasp/primitive condition schema")
    model.load_state_dict(checkpoint["state"])
    if stage == "flow":
        model.freeze_autoencoder()
    model.eval()
    return model, checkpoint


def draw_pairs(table, counts, unique, selected, generator):
    rows = torch.searchsorted(unique, selected)
    a = (torch.rand(len(rows), device=table.device, generator=generator) * counts[rows]).long()
    offset = 1 + (torch.rand(len(rows), device=table.device, generator=generator) * (counts[rows] - 1)).long()
    b = (a + offset) % counts[rows]
    ids = torch.stack((table[rows, a], table[rows, b]), -1).flatten()
    partners = torch.arange(len(ids), device=ids.device).reshape(-1, 2).flip(-1).flatten()
    return ids, partners


def sample_autoencoder_pairs(table, counts, unique, group_types, batch_size, generator):
    """Shared by training and sampling diagnostics; guarantee negative types."""
    selected = unique[torch.randint(len(unique), (batch_size // 2,), device=table.device, generator=generator)]
    first = torch.randint(len(unique), (), device=table.device, generator=generator)
    other_types = unique[group_types[unique] != group_types[unique[first]]]
    selected[0] = unique[first]
    selected[1] = other_types[torch.randint(len(other_types), (), device=table.device, generator=generator)]
    return draw_pairs(table, counts, unique, selected, generator)


def pair_table(groups, sizes):
    unique = groups.unique()
    table = torch.zeros(len(unique), max(int((groups == g).sum()) for g in unique),
                        device=groups.device, dtype=torch.long)
    counts = torch.zeros(len(unique), device=groups.device, dtype=torch.long)
    for row, group in enumerate(unique):
        ids = torch.where(groups == group)[0]
        if len(ids) < 2 or len(torch.unique(sizes[ids])) != len(ids):
            raise ValueError("Training pairs require two or more distinct size nodes per group")
        table[row, :len(ids)], counts[row] = ids, len(ids)
    return table, counts, unique


def macro_mse(prediction, target, groups):
    errors = (prediction - target).square().mean(-1)
    return torch.stack([errors[groups == group].mean() for group in groups.unique()]).mean()


def validation_cases(x, config, seed=20261103):
    generator = torch.Generator(device=x.device).manual_seed(seed)
    cases = []
    # Keep low-noise checks and test the configured broad-noise endpoint too.
    for sigma in sorted({.03, .1, config.sigma_max}):
        observed, metadata = finger_corruption(x, generator, max_rounds=config.max_rounds,
                                               sigma_min=sigma, sigma_max=sigma)
        cases.append((observed, metadata))
    return cases


def train_flow(data, kind="vq", seed=7, config=None, device="auto", stage="all", autoencoder_path=None):
    config = config or FlowConfig()
    config.validate()
    if kind not in ("vq", "vae") or stage not in ("all", "autoencoder", "flow"):
        raise ValueError("Unknown latent kind or stage")
    seed_everything(seed)
    device, method = device_name(device), f"diffusion_{kind}"
    directory = data.directory / "models" / f"{method}_seed{seed}"
    requested_stages = ("autoencoder", "flow") if stage == "all" else (stage,)
    for part in requested_stages:
        target = directory / ("best.pt" if part == "flow" else "autoencoder.pt")
        if target.exists():
            raise FileExistsError(f"Training output already exists: {target}")
    if stage == "flow" and autoencoder_path is None:
        raise ValueError("Independent flow training requires --autoencoder-checkpoint")
    model = PoseFlowModel(kind, data, config).to(device)
    if stage == "flow":
        encoder_model, _ = load_flow_model(data, method, seed, device, stage="autoencoder", path=autoencoder_path)
        # All hyperparameters determining tensor shapes must agree.
        for key in ("latent_dim", "hidden_dim", "codebook_size"):
            if getattr(encoder_model.config, key) != getattr(config, key):
                raise ValueError(f"Autoencoder checkpoint has incompatible {key}")
        model.autoencoder.load_state_dict(encoder_model.autoencoder.state_dict())
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "config.json", dict(architecture=ARCHITECTURE, method=method, seed=seed,
                                                device=device, **asdict(config)))
    train_ids, valid_ids = np.flatnonzero(data.mask("train")), np.flatnonzero(data.mask("validation"))
    x, groups, sizes = tensors(data, train_ids, device)
    vx, vg, vs = tensors(data, valid_ids, device)
    if not len(vx) or len(model.group_types[groups].unique()) < 2:
        raise ValueError("Need validation poses and at least two training grasp types")
    table, counts, unique = pair_table(groups, sizes)
    fixed_cases = validation_cases(vx, config)
    # Persist corruption truth so validation can be reproduced without training.
    np.savez_compressed(directory / "validation_noise.npz", indices=valid_ids,
                        observed=np.stack([a.cpu().numpy() for a, _ in fixed_cases]),
                        mask=np.stack([b["mask"].cpu().numpy() for _, b in fixed_cases]),
                        rounds=np.stack([b["rounds"].cpu().numpy() for _, b in fixed_cases]),
                        noise=np.stack([b["noise"].cpu().numpy() for _, b in fixed_cases]),
                        sigma_step=np.stack([b["sigma_step"].cpu().numpy() for _, b in fixed_cases]))
    result_path = directory / "training.json"
    result = (json.loads(result_path.read_text()) if result_path.exists()
              else dict(method=method, seed=seed, stages={}))
    for part in requested_stages:
        generator = torch.Generator(device=device).manual_seed(seed + (100000 if part == "flow" else 0))
        if part == "flow":
            if stage == "all":
                autoencoder_path = directory / "autoencoder.pt"
                best_ae, _ = load_flow_model(data, method, seed, device, stage="autoencoder", path=autoencoder_path)
                model.autoencoder.load_state_dict(best_ae.autoencoder.state_dict())
            model.freeze_autoencoder()
        parameters = list(model.autoencoder.parameters() if part == "autoencoder" else model.velocity.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=config.lr, weight_decay=1e-5)
        frozen = {k: v.detach().clone() for k, v in model.autoencoder.state_dict().items()} if part == "flow" else {}
        best, best_step, stale, history = math.inf, 0, 0, []
        started = time.perf_counter()
        target_path = directory / ("best.pt" if part == "flow" else "autoencoder.pt")
        for step in range(1, config.steps + 1):
            model.train()
            if part == "autoencoder":
                ids, partners = sample_autoencoder_pairs(table, counts, unique, model.group_types,
                                                        config.batch_size, generator)
                loss, losses = model.autoencoder_objective(x[ids], groups[ids], sizes[ids], partners, generator)
            else:
                ids = torch.randint(len(x), (config.batch_size,), device=device, generator=generator)
                loss = model.flow_objective(x[ids], groups[ids], sizes[ids], generator)
                losses = dict(flow=loss)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite {part} loss at step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 5.)
            optimizer.step()
            if step % config.validation_every and step != config.steps:
                continue
            model.eval()
            with torch.no_grad():
                scores = []
                for observed, _ in fixed_cases:
                    if part == "autoencoder":
                        latent, _ = model.autoencoder.encode(observed)
                        prediction = model.autoencoder.decode(latent, model.group_primitives[vg], vs)
                    else:
                        prediction = model.denoise(observed, vg, vs)
                    scores.append(macro_mse(prediction, vx, vg))
                score = float(torch.stack(scores).mean())
            if not math.isfinite(score):
                raise RuntimeError(f"Nonfinite validation score at {part} step {step}")
            history.append(dict(step=step, loss=float(loss.detach()), validation_mse=score,
                                components={k: float(v.detach()) for k, v in losses.items()}))
            if score < best:
                best, best_step, stale = score, step, 0
                checkpoint = dict(architecture=ARCHITECTURE, stage=part, method=method, seed=seed,
                                  config=asdict(config), state=model.state_dict(), dataset_sha256=dataset_digest(data),
                                  condition_schema=dict(types=model.type_names, primitives=model.primitive_names),
                                  best_step=step, validation_mse=score,
                                  autoencoder_sha256=file_digest(autoencoder_path) if part == "flow" else None)
                temporary = target_path.with_suffix(".tmp")
                torch.save(checkpoint, temporary)
                temporary.replace(target_path)
            else:
                stale += 1
            print(f"{method} seed={seed} {part} step={step}: loss={float(loss.detach()):.6f} val={score:.7f}", flush=True)
            write_json(directory / f"{part}_history.json", dict(best_step=best_step, validation_mse=best,
                       completed_steps=step, elapsed_s=time.perf_counter() - started, history=history))
            if step >= config.min_steps and stale >= config.patience:
                break
        if part == "flow":
            if any(not torch.equal(value, model.autoencoder.state_dict()[key]) for key, value in frozen.items()):
                raise RuntimeError("Frozen autoencoder changed during flow training")
            register_model(data, method, seed, best)
        result["stages"][part] = dict(best_step=best_step, validation_mse=best, completed_steps=step,
                                       elapsed_s=time.perf_counter() - started,
                                       frozen_autoencoder_unchanged=True if part == "flow" else None)
    write_json(directory / "training.json", result)
    return result
