"""Reproducible grasp denoising experiments; independent of production solvers.

All projections and residuals use joint-limit-normalized finger coordinates.
The 128 dictionary columns are overcomplete atoms, not independent PCs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import math
import time

import numpy as np
from scipy.interpolate import PchipInterpolator
import torch
from torch import nn
from torch.nn import functional as F

METHODS = ("pca", "interpolation", "dictionary", "diffusion")
SPLITS = {"train": 0, "validation": 1, "test": 2}
VERSION = 1


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def fingerprint(source):
    source = Path(source)
    return {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in ("manifest.json", "poses.npz", "scene.xml", "metrics.json.gz")}


def device_name(request="auto"):
    if request == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if request.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return request


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)


def split_size_nodes(groups, sizes, seed=7):
    """Split unique original size nodes, keeping endpoints as training anchors."""
    groups, sizes = np.asarray(groups), np.asarray(sizes)
    result = np.zeros(len(groups), dtype=np.int64)
    rng = np.random.default_rng(seed)
    for group in np.unique(groups):
        nodes = np.unique(sizes[groups == group])
        if len(nodes) < 5:
            raise ValueError(f"Group {group} has {len(nodes)} size nodes; need at least five")
        interior = nodes[1:-1].copy()
        rng.shuffle(interior)
        count = max(1, int(round(len(nodes) * .15)))
        for label, selected in ((1, interior[:count]), (2, interior[count:2 * count])):
            result[(groups == group) & np.isin(sizes, selected)] = label
    return result


@dataclass
class Subspace:
    mean: np.ndarray
    basis: np.ndarray
    eigenvalues: np.ndarray
    numerical_rank: int
    retained_fraction: float
    truncation_rmse: float

    @classmethod
    def fit(cls, x, variance=.95):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or len(x) < 2 or x.shape[1] != 20 or not np.isfinite(x).all():
            raise ValueError("PCA requires at least two finite 20D training poses")
        mean = x.mean(0)
        _, singular, vt = np.linalg.svd(x - mean, full_matrices=False)
        tolerance = max(x.shape) * np.finfo(float).eps * max(float(singular[0]), 1.)
        rank = int((singular > tolerance).sum())
        values = singular ** 2 / (len(x) - 1)
        total = values.sum()
        k = 0 if rank == 0 else min(int(np.searchsorted(np.cumsum(values) / total, variance)) + 1,
                                     rank, 19, len(x) - 1)
        basis = vt[:k].T.copy()
        reconstruction = mean + (x - mean) @ basis @ basis.T
        return cls(mean, basis, values, rank, float(values[:k].sum() / total) if total else 1.,
                   float(np.sqrt(np.mean((x - reconstruction) ** 2))))

    def project(self, x):
        x = np.asarray(x)
        return self.mean + (x - self.mean) @ self.basis @ self.basis.T

    def decompose(self, x):
        principal = self.project(x)
        return principal, np.asarray(x) - principal

    def record(self):
        result = asdict(self)
        for name in ("mean", "basis", "eigenvalues"):
            result[name] = result[name].tolist()
        return result

    @classmethod
    def from_record(cls, record):
        record = dict(record)
        for name in ("mean", "basis", "eigenvalues"):
            record[name] = np.asarray(record[name], dtype=float)
        # JSON [] loses the leading dimension when rank is zero.
        record["basis"] = record["basis"].reshape(20, -1)
        return cls(**record)


class SizeCurve:
    """Training-node-only shape-preserving interpolation; no extrapolation."""
    def __init__(self, sizes, poses, subspace=None):
        sizes, poses = np.asarray(sizes), np.asarray(poses)
        self.sizes = np.unique(sizes)
        if len(self.sizes) < 2 or not np.isfinite(poses).all():
            raise ValueError("A size curve needs at least two finite nodes")
        self.poses = np.stack([poses[sizes == size].mean(0) for size in self.sizes])
        self.subspace = subspace
        # Componentwise PCHIP in 20D need not preserve a multi-PC subspace.
        # Interpolate PC coefficients instead, so tightening cannot change r_perp.
        coefficients = (self.poses-subspace.mean) @ subspace.basis if subspace else self.poses
        self.constant = coefficients.shape[1] == 0
        self.interpolator = None if self.constant else PchipInterpolator(
            self.sizes, coefficients, axis=0, extrapolate=False)

    def __call__(self, size):
        size = np.asarray(size)
        if not np.isfinite(size).all():
            raise ValueError("Size must be finite")
        if self.constant:
            return np.broadcast_to(self.subspace.mean, (*size.shape, 20)).copy()
        value = self.interpolator(np.clip(size, self.sizes[0], self.sizes[-1]))
        return self.subspace.mean + value @ self.subspace.basis.T if self.subspace else value

    def tighten(self, clean, residual, source, fraction=.1):
        if not np.isfinite(fraction) or not 0 <= fraction < 1:
            raise ValueError("Tightening fraction must lie in [0,1)")
        if not self.sizes[0] <= source <= self.sizes[-1]:
            raise ValueError("Source size lies outside the training range")
        target = float(np.clip(source * (1 - fraction), self.sizes[0], self.sizes[-1]))
        delta = self(target) - self(source)
        return np.asarray(clean) + delta + residual, target, delta


def collision_depths(model, data):
    import mujoco
    mujoco.mj_kinematics(model, data)
    mujoco.mj_collision(model, data)
    obj = model.body("grasp_object").id
    depths = [0., 0.]
    for contact in data.contact:
        bodies = model.geom_bodyid[contact.geom]
        if np.all(bodies == obj):
            continue
        # These packed scenes contain only the fixed object and the hand.
        slot = 0 if obj in bodies else 1
        depths[slot] = max(depths[slot], -float(contact.dist))
    return depths


class ExperimentData:
    def __init__(self, directory, check_source=True):
        self.directory = Path(directory).resolve()
        self.meta = json.loads((self.directory / "dataset.json").read_text())
        if self.meta["version"] != VERSION:
            raise ValueError("Unsupported dataset version")
        self.source = Path(self.meta["source"])
        if not self.source.is_absolute():
            self.source = (self.directory / self.source).resolve()
        if check_source and fingerprint(self.source) != self.meta["source_sha256"]:
            raise ValueError("Source snapshot changed; prepare a new experiment")
        with np.load(self.directory / "dataset.npz", allow_pickle=False) as values:
            for key in values.files:
                setattr(self, key, values[key].copy())
        self.subspaces = [Subspace.from_record(x) for x in self.meta["subspaces"]]
        self.x = self.normalize(self.qpos[:, 6:])
        self.log_center = float(self.meta["log_size_center"])
        self.log_scale = float(self.meta["log_size_scale"])
        self.snapshot = None

    def model_seed(self, method, requested=None):
        """Use the retained validation-selected model in compact result bundles."""
        return int(requested if requested is not None else self.meta.get("selected_seeds", {}).get(method, 7))

    def normalize(self, angles):
        return (np.asarray(angles) - self.mid) / self.half

    def denormalize(self, x):
        return self.mid + self.half * np.asarray(x)

    def mask(self, split):
        return self.split == SPLITS[split]

    def project(self, x, groups):
        x, groups = np.asarray(x), np.asarray(groups)
        result = np.empty_like(x, dtype=float)
        for group in np.unique(groups):
            mask = groups == group
            result[mask] = self.subspaces[int(group)].project(x[mask])
        return result

    def curve(self, group, poses=None):
        mask = (self.group == group) & self.mask("train")
        if poses is None:
            poses = self.subspaces[group].project(self.x[mask])
        return SizeCurve(self.size[mask], poses, self.subspaces[group])

    def get_snapshot(self):
        if self.snapshot is None:
            from retarget.viewer.snapshot import GraspSnapshot
            self.snapshot = GraspSnapshot(self.source)
        return self.snapshot


def prepare(source, output, seed=7):
    from retarget.viewer.snapshot import GraspSnapshot
    import mujoco
    output = Path(output).resolve()
    if (output / "dataset.json").exists():
        raise FileExistsError("Dataset already exists; use a new experiment directory")
    snapshot = GraspSnapshot(Path(source).resolve())
    rows = snapshot.manifest["cases"]
    names = sorted({(e["template"], e["primitive"]["kind"]) for e in rows},
                   key=lambda x: (int(x[0].split("_")[0]), x[1]))
    labels = {name: i for i, name in enumerate(names)}
    first = snapshot.model(0)
    if first.nq != 26 or first.njnt != 26 or not np.all(first.jnt_limited[6:]):
        raise ValueError("Expected a 26-coordinate hand with 20 limited finger hinges")
    low, high = first.jnt_range[6:].T.copy()
    kept, groups, sizes, quality, rejected = [], [], [], [], []
    for index, row in enumerate(rows):
        q = snapshot.poses["qpos"][index]
        reason = None
        if q.shape != (26,) or not np.isfinite(q).all():
            reason = "nonfinite_or_wrong_shape"
        elif np.any(q[6:] < low - 1e-6) or np.any(q[6:] > high + 1e-6):
            reason = "finger_joint_limits"
        if reason is None:
            model = snapshot.model(index)
            if not np.allclose(model.jnt_range[6:], first.jnt_range[6:]):
                raise ValueError("Joint limits differ across snapshot cases")
            data = mujoco.MjData(model)
            data.qpos[:] = q
            depth = collision_depths(model, data)
            if depth[1] > 1.01e-5:
                reason = "self_penetration"
        if reason:
            rejected.append(dict(source_index=index, case=row["name"], reason=reason))
            continue
        d = snapshot.diagnostics(index)
        kept.append(index)
        groups.append(labels[(row["template"], row["primitive"]["kind"])])
        sizes.append(row["primitive"]["size_scale"])
        quality.append(dict(source_index=index, case=row["name"], coverage=d.get("coverage"),
                            projected_closure=d.get("projected_closure"),
                            hand_object_penetration_m=depth[0], self_penetration_m=depth[1]))
        if (index + 1) % 100 == 0:
            print(f"prepare: checked {index + 1}/{len(rows)} poses", flush=True)
    groups, sizes, kept = np.asarray(groups), np.asarray(sizes, float), np.asarray(kept)
    if not np.isfinite(sizes).all() or np.any(sizes <= 0):
        raise ValueError("Primitive size scales must be positive and finite")
    counts = [int((groups == g).sum()) for g in range(len(names))]
    if min(counts) < 5:
        raise ValueError(f"Insufficient valid nodes for a group: {list(zip(names, counts))}")
    split = split_size_nodes(groups, sizes, seed)
    mid, half = (low + high) / 2, (high - low) / 2
    qpos = snapshot.poses["qpos"][kept]
    x = (qpos[:, 6:] - mid) / half
    subspaces = [Subspace.fit(x[(groups == g) & (split == 0)]) for g in range(len(names))]
    train_log = np.log(sizes[split == 0])
    metadata = dict(version=VERSION, source=str(Path(source).resolve()), seed=seed,
                    source_sha256=fingerprint(source), groups=[dict(template=t, primitive=k) for t, k in names],
                    subspaces=[s.record() for s in subspaces], quality=quality, rejected=rejected,
                    log_size_center=float(train_log.mean()), log_size_scale=max(float(train_log.std()), .1),
                    split_policy="unique original size nodes; train endpoints; 15% validation and test each",
                    dense_interpolated_poses_used=False,
                    clean_definition="kinematically_valid_reference_not_verified_physical_grasp",
                    counts={name: int((split == value).sum()) for name, value in SPLITS.items()})
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "dataset.npz", qpos=qpos, group=groups, size=sizes, split=split,
                        source_index=kept, mid=mid, half=half)
    write_json(output / "dataset.json", metadata)
    print(f"Prepared {len(kept)} poses / {len(names)} groups; rejected {len(rejected)}; {metadata['counts']}", flush=True)
    return metadata


@dataclass
class TrainConfig:
    steps: int = 3000
    batch_size: int = 256
    lr: float = 1e-3
    validation_every: int = 250
    patience: int = 6
    diffusion_steps: int = 1000
    ddim_steps: int = 50
    sparsity_weight: float = 1e-3
    consistency_weight: float = 1e-2


def alpha_schedule(steps=1000):
    # Index zero is exactly clean, unlike the first noisy DDPM timestep.
    beta = np.linspace(1e-4, .02, steps, dtype=np.float64)
    return np.r_[1., np.cumprod(1 - beta)]


def sigma_timestep(sigma, schedule):
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("Noise sigma must be finite and nonnegative")
    if sigma == 0:
        return 0
    sigmas = np.sqrt((1 - schedule) / schedule)
    return int(np.argmin(np.abs(sigmas - sigma)))


class ConditionalNetwork(nn.Module):
    def __init__(self, groups, log_center, log_scale, output):
        super().__init__()
        self.embedding = nn.Embedding(groups, 32)
        self.log_center, self.log_scale = float(log_center), float(log_scale)
        self.register_buffer("frequencies", torch.exp(torch.linspace(0, math.log(1000), 16)))
        self.net = nn.Sequential(nn.Linear(20 + 32 + 1 + 32, 256), nn.SiLU(),
                                 nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, output))

    def forward(self, x, groups, sizes, times):
        angle = times[:, None] * self.frequencies[None]
        time_features = torch.cat((angle.sin(), angle.cos()), -1)
        size_features = ((sizes.log() - self.log_center) / self.log_scale)[:, None]
        return self.net(torch.cat((x, self.embedding(groups), size_features, time_features), -1))


class DenoisingModel(nn.Module):
    def __init__(self, method, data, config):
        super().__init__()
        if method not in ("dictionary", "diffusion"):
            raise ValueError(f"Not a trainable model: {method}")
        self.method, self.config = method, config
        self.encoder = ConditionalNetwork(len(data.subspaces), data.log_center, data.log_scale,
                                          128 if method == "dictionary" else 20)
        means = np.stack([s.mean for s in data.subspaces])
        projections = np.stack([s.basis @ s.basis.T for s in data.subspaces])
        self.register_buffer("means", torch.tensor(means, dtype=torch.float32))
        self.register_buffer("projections", torch.tensor(projections, dtype=torch.float32))
        self.register_buffer("alphas", torch.tensor(alpha_schedule(config.diffusion_steps), dtype=torch.float32))
        if method == "dictionary":
            self.dictionary = nn.Parameter(F.normalize(torch.randn(20, 128), dim=0))

    def project(self, x, groups):
        return self.means[groups] + torch.bmm(self.projections[groups],
                                            (x - self.means[groups])[..., None])[..., 0]

    def coefficients(self, x, groups, sizes, times):
        return F.softshrink(self.encoder(x, groups, sizes, times), lambd=.01)

    def dictionary_pose(self, coefficients, groups):
        delta = coefficients @ F.normalize(self.dictionary, dim=0).T
        return self.project(self.means[groups] + delta, groups)

    def objective(self, clean, groups, sizes, generator):
        clean = self.project(clean, groups)
        count = len(clean)
        steps = self.config.diffusion_steps
        t = torch.randint(1, steps + 1, (count,), device=clean.device, generator=generator)
        if self.method == "dictionary":
            t = torch.where(torch.rand(count, device=clean.device, generator=generator) < .2, 0, t)
        eps = torch.randn(clean.shape, device=clean.device, generator=generator)
        a = self.alphas[t, None]
        noisy = a.sqrt() * clean + (1 - a).sqrt() * eps
        times = t.float() / steps
        if self.method == "diffusion":
            return F.mse_loss(self.encoder(noisy, groups, sizes, times), eps)
        code = self.coefficients(noisy, groups, sizes, times)
        clean_code = self.coefficients(clean, groups, sizes, torch.zeros_like(times))
        prediction = self.dictionary_pose(code, groups)
        return (F.mse_loss(prediction, clean)
                + self.config.sparsity_weight * code.abs().mean()
                + self.config.consistency_weight * F.mse_loss(code, clean_code)
                + .2 * F.mse_loss(self.dictionary_pose(clean_code, groups), clean))

    @torch.no_grad()
    def denoise(self, observed, groups, sizes, timestep):
        """Observed = x_clean + sigma(t)*epsilon, in normalized joint units.

        Convert to VP q_t exactly once. Reverse diffusion starts at that t,
        rather than adding fresh noise or sampling an unrelated grasp.
        """
        t = int(timestep)
        if not 0 <= t <= self.config.diffusion_steps:
            raise ValueError("Timestep outside trained schedule")
        if t == 0 and self.method == "diffusion":
            return self.project(observed, groups)
        x = self.alphas[t].sqrt() * observed
        times = torch.full((len(x),), t / self.config.diffusion_steps, device=x.device)
        if self.method == "dictionary":
            return self.dictionary_pose(self.coefficients(x, groups, sizes, times), groups)
        grid = np.unique(np.rint(np.linspace(0, t, min(t, self.config.ddim_steps) + 1)).astype(int))[::-1]
        for current, previous in zip(grid[:-1], grid[1:]):
            times.fill_(float(current) / self.config.diffusion_steps)
            eps = self.encoder(x, groups, sizes, times)
            a, a_prev = self.alphas[int(current)], self.alphas[int(previous)]
            x0 = (x - (1 - a).sqrt() * eps) / a.sqrt()
            x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
        return self.project(x, groups)


def tensors(data, indices, device):
    return (torch.as_tensor(data.x[indices], dtype=torch.float32, device=device),
            torch.as_tensor(data.group[indices], dtype=torch.long, device=device),
            torch.as_tensor(data.size[indices], dtype=torch.float32, device=device))


def train(data, method, seed=7, config=None, device="auto"):
    config = config or TrainConfig()
    if config.steps < 1 or config.batch_size < 1 or config.validation_every < 1 or config.patience < 1:
        raise ValueError("Training counts must be positive")
    if config.diffusion_steps < 1 or config.ddim_steps < 1 or not np.isfinite(config.lr) or config.lr <= 0:
        raise ValueError("Diffusion counts and finite learning rate must be positive")
    seed_everything(seed)
    device = device_name(device)
    directory = data.directory / "models" / f"{method}_seed{seed}"
    if (directory / "best.pt").exists():
        raise FileExistsError(f"Model already trained: {directory}; use a new seed or experiment")
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "config.json", dict(method=method, seed=seed, device=device, **asdict(config)))
    model = DenoisingModel(method, data, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-5)
    generator = torch.Generator(device=device).manual_seed(seed)
    train_ids, valid_ids = np.flatnonzero(data.mask("train")), np.flatnonzero(data.mask("validation"))
    x, g, s = tensors(data, train_ids, device)
    vx, vg, vs = tensors(data, valid_ids, device)
    valid_generator = torch.Generator(device=device).manual_seed(20261103)
    fixed_noise = torch.randn(vx.shape, device=device, generator=valid_generator)
    schedule = alpha_schedule(config.diffusion_steps)
    valid_times = [sigma_timestep(sigma, schedule) for sigma in (.03, .1, .3)]
    best, best_step, stale, history = math.inf, 0, 0, []
    started = time.perf_counter()
    for step in range(1, config.steps + 1):
        model.train()
        ids = torch.randint(len(x), (config.batch_size,), device=device, generator=generator)
        loss = model.objective(x[ids], g[ids], s[ids], generator)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite training loss at {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        if method == "dictionary":
            with torch.no_grad():
                model.dictionary.copy_(F.normalize(model.dictionary, dim=0))
        if step % config.validation_every != 0 and step != config.steps:
            continue
        model.eval()
        errors = []
        with torch.no_grad():
            for t in valid_times:
                sigma = math.sqrt((1 - schedule[t]) / schedule[t])
                prediction = model.denoise(vx + sigma * fixed_noise, vg, vs, t)
                errors.append(float(((prediction - vx) ** 2).mean()))
        score = float(np.mean(errors))
        history.append(dict(step=step, loss=float(loss.detach()), validation_mse=score))
        if score < best:
            best, best_step, stale = score, step, 0
            torch.save(dict(state=model.state_dict(), config=asdict(config), method=method, seed=seed,
                            dataset_sha256=hashlib.sha256((data.directory / "dataset.json").read_bytes()).hexdigest(),
                            best_step=step, validation_mse=score), directory / "best.pt")
        else:
            stale += 1
        print(f"{method} seed={seed} step={step}: loss={float(loss.detach()):.5f} val={score:.6f}", flush=True)
        if stale >= config.patience:
            break
    write_json(directory / "history.json", dict(best_step=best_step, validation_mse=best,
                                                elapsed_s=time.perf_counter() - started, history=history))
    model, _ = load_model(data, method, seed, device)
    export_codes(data, model, directory)
    return dict(method=method, seed=seed, best_step=best_step, validation_mse=best)


def load_model(data, method, seed=None, device="auto"):
    device = device_name(device)
    seed = data.model_seed(method, seed)
    path = data.directory / "models" / f"{method}_seed{seed}" / "best.pt"
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    digest = hashlib.sha256((data.directory / "dataset.json").read_bytes()).hexdigest()
    if checkpoint["dataset_sha256"] != digest:
        raise ValueError("Checkpoint belongs to a different dataset")
    config = TrainConfig(**checkpoint["config"])
    model = DenoisingModel(method, data, config).to(device)
    model.load_state_dict(checkpoint["state"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def export_codes(data, model, directory):
    if model.method != "dictionary":
        return
    device = model.means.device
    codes, groups, sizes, references = [], [], [], []
    # Training nodes only; never export test predictions as learned anchors.
    for group in range(len(data.subspaces)):
        ids = np.flatnonzero((data.group == group) & data.mask("train"))
        x, g, s = tensors(data, ids, device)
        code = model.coefficients(x, g, s, torch.zeros_like(s)).cpu().numpy()
        codes.extend(code); groups.extend([group] * len(ids)); sizes.extend(data.size[ids])
        ref = int(np.argmin(np.abs(data.size[ids] - 1.)))
        references.append(dict(group=group, size=float(data.size[ids][ref]), code=code[ref].tolist()))
    dictionary = F.normalize(model.dictionary, dim=0).cpu().numpy()
    np.savez_compressed(Path(directory) / "dictionary_codes.npz", dictionary=dictionary,
                        codes=np.asarray(codes), groups=groups, sizes=sizes)
    write_json(Path(directory) / "reference_codes.json", dict(dictionary_shape=list(dictionary.shape),
               rank=int(np.linalg.matrix_rank(dictionary)), references=references,
               meaning="128 redundant atoms in normalized 20D; codes vary with size"))


class Predictor:
    def __init__(self, data, method, seed=None, device="auto"):
        if method not in METHODS:
            raise ValueError(f"Unknown method: {method}")
        self.data, self.method = data, method
        self.device = device_name(device)
        self.model = None
        self.schedule = alpha_schedule()
        if method in ("dictionary", "diffusion"):
            self.model, _ = load_model(data, method, seed, self.device)
            self.schedule = alpha_schedule(self.model.config.diffusion_steps)
        self.curves = []
        for group in range(len(data.subspaces)):
            ids = np.flatnonzero((data.group == group) & data.mask("train"))
            poses = self.predict(data.x[ids], data.group[ids], data.size[ids], 0.) if self.model else None
            self.curves.append(data.curve(group, poses))

    def predict(self, observed, groups, sizes, sigma):
        observed = np.asarray(observed, float)
        groups, sizes = np.asarray(groups, int), np.asarray(sizes, float)
        if observed.ndim != 2 or observed.shape[1] != 20 or not np.isfinite(observed).all():
            raise ValueError("Expected finite (N,20) observed poses")
        if groups.shape != (len(observed),) or sizes.shape != (len(observed),) or not np.isfinite(sizes).all() or np.any(sizes <= 0):
            raise ValueError("Conditions must match batch and sizes must be positive")
        if np.any(groups < 0) or np.any(groups >= len(self.data.subspaces)):
            raise ValueError("Unknown grasp group")
        if not np.isfinite(sigma) or sigma < 0:
            raise ValueError("Noise sigma must be finite and nonnegative")
        if self.method == "pca":
            return self.data.project(observed, groups)
        if self.method == "interpolation":
            return np.stack([self.curves[int(g)](s) for g, s in zip(groups, sizes)])
        t = sigma_timestep(sigma, self.schedule)
        with torch.no_grad():
            x = torch.as_tensor(observed, dtype=torch.float32, device=self.device)
            g = torch.as_tensor(groups, dtype=torch.long, device=self.device)
            s = torch.as_tensor(sizes, dtype=torch.float32, device=self.device)
            return self.model.denoise(x, g, s, t).cpu().numpy().astype(float)


def subspace_stability(data, seed):
    from scipy.linalg import subspace_angles
    rng = np.random.default_rng(seed)
    rows = []
    for group, subspace in enumerate(data.subspaces):
        x = data.x[(data.group == group) & data.mask("train")]
        # Bootstrap only training nodes. Different selected ranks are explicit.
        alternate = Subspace.fit(x[rng.integers(0, len(x), len(x))])
        angle = subspace_angles(subspace.basis, alternate.basis)
        rows.append(dict(group=group, rank=subspace.basis.shape[1], bootstrap_rank=alternate.basis.shape[1],
                         max_angle_deg=float(np.rad2deg(angle).max()) if len(angle) else 0.,
                         projector_distance=float(np.linalg.norm(subspace.basis @ subspace.basis.T
                                                        - alternate.basis @ alternate.basis.T))))
    return rows


def evaluate(data, seeds=(None,), noise_seeds=(11, 23, 37), sigmas=(0., .03, .1, .3), device="auto"):
    """Select by validation only, then report every method on untouched test nodes."""
    if not seeds or not noise_seeds or not sigmas:
        raise ValueError("Evaluation needs model seeds, noise seeds and noise levels")
    records, timings, stability, code_stability, tightening = [], [], [], [], []
    half = data.half
    for requested_seed in seeds:
        for method in METHODS:
            model_seed = data.model_seed(method, requested_seed)
            predictor = Predictor(data, method, model_seed, device)
            for split in ("validation", "test"):
                ids = np.flatnonzero(data.mask(split))
                clean, groups, sizes = data.x[ids], data.group[ids], data.size[ids]
                projected = data.project(clean, groups)
                # One warmed timing per method / split, on the actual device.
                predictor.predict(clean[:2], groups[:2], sizes[:2], .1)
                for sigma_request in sigmas:
                    t = sigma_timestep(sigma_request, predictor.schedule)
                    sigma = math.sqrt((1 - predictor.schedule[t]) / predictor.schedule[t])
                    predictions = []
                    for noise_seed in noise_seeds:
                        rng = np.random.default_rng(noise_seed)
                        observed = clean + sigma * rng.standard_normal(clean.shape)
                        if predictor.device.startswith("cuda"):
                            torch.cuda.synchronize()
                        started = time.perf_counter()
                        result = predictor.predict(observed, groups, sizes, sigma)
                        if predictor.device.startswith("cuda"):
                            torch.cuda.synchronize()
                        elapsed = time.perf_counter() - started
                        timings.append(dict(method=method, model_seed=model_seed, split=split,
                                            sigma=sigma, batch=len(ids), elapsed_s=elapsed))
                        predictions.append(result)
                        for j, index in enumerate(ids):
                            records.append(dict(method=method, model_seed=model_seed, noise_seed=noise_seed,
                                split=split, index=int(index), group=int(groups[j]), sigma_requested=sigma_request,
                                sigma_actual=sigma, rmse_normalized=float(np.sqrt(np.mean((result[j]-clean[j])**2))),
                                rmse_rad=float(np.sqrt(np.mean(((result[j]-clean[j])*half)**2))),
                                projected_rmse_normalized=float(np.sqrt(np.mean((result[j]-projected[j])**2)))))
                    predictions = np.stack(predictions)
                    stability.append(dict(method=method, model_seed=model_seed, split=split, sigma=sigma,
                                          prediction_std_normalized=float(predictions.std(0).mean())))
                    if method == "dictionary":
                        model = predictor.model
                        code_predictions = []
                        for noise_seed in noise_seeds:
                            observed = clean + sigma * np.random.default_rng(noise_seed).standard_normal(clean.shape)
                            with torch.no_grad():
                                tx = torch.as_tensor(observed, device=predictor.device, dtype=torch.float32)
                                tg = torch.as_tensor(groups, device=predictor.device, dtype=torch.long)
                                ts = torch.as_tensor(sizes, device=predictor.device, dtype=torch.float32)
                                tt = torch.full_like(ts, t / model.config.diffusion_steps)
                                code_predictions.append(model.coefficients(model.alphas[t].sqrt()*tx, tg, ts, tt).cpu().numpy())
                        a, b = code_predictions[0], code_predictions[-1]
                        cosine = (a*b).sum(-1) / np.maximum(np.linalg.norm(a, axis=-1)*np.linalg.norm(b, axis=-1), 1e-12)
                        code_stability.append(dict(model_seed=model_seed, split=split, sigma=sigma,
                                                   mean_code_cosine=float(cosine.mean()),
                                                   code_std=float(np.std(code_predictions, axis=0).mean())))
                # Geometry: one reproducible random test pose per class, all methods.
                if split == "test":
                    rng = np.random.default_rng(20261103)
                    chosen = [int(rng.choice(ids[groups == group])) for group in np.unique(groups)]
                    for index in chosen:
                        group = int(data.group[index])
                        x = data.x[index]
                        t = sigma_timestep(.1, predictor.schedule)
                        sigma = math.sqrt((1-predictor.schedule[t])/predictor.schedule[t])
                        observed = x + sigma * np.random.default_rng(11 + index).standard_normal(20)
                        prediction = predictor.predict(observed[None], [group], [data.size[index]], sigma)[0]
                        _, residual = data.subspaces[group].decompose(observed)
                        tightened, target, _ = predictor.curves[group].tighten(prediction, residual, data.size[index])
                        q_before = data.qpos[index].copy(); q_before[6:] = data.denormalize(observed)
                        q_after = data.qpos[index].copy(); q_after[6:] = data.denormalize(tightened)
                        checker = PoseChecker(data, index)
                        tightening.append(dict(method=method, model_seed=model_seed, index=index, group=group,
                            source_size=float(data.size[index]), target_size=target,
                            before=checker.inspect(q_before)[0], after=checker.inspect(q_after)[0]))
            print(f"evaluated {method} seed={model_seed}", flush=True)
    summaries = []
    for method in METHODS:
        for split in ("validation", "test"):
            rows = [r for r in records if r["method"] == method and r["split"] == split]
            # Macro-average classes, so denser type grids cannot dominate selection.
            macro = [np.mean([r["rmse_normalized"] for r in rows if r["group"] == g])
                     for g in range(len(data.subspaces))]
            summaries.append(dict(method=method, split=split, macro_rmse_normalized=float(np.mean(macro)),
                                  mean_rmse_rad=float(np.mean([r["rmse_rad"] for r in rows])),
                                  projected_rmse_normalized=float(np.mean([r["projected_rmse_normalized"] for r in rows]))))
    validation = [r for r in summaries if r["split"] == "validation"]
    selected = min(validation, key=lambda r: (r["macro_rmse_normalized"], METHODS.index(r["method"])))
    result = dict(selected_method=selected["method"], selection_split="validation", model_seeds=list(seeds),
                  selected_seeds={m: data.model_seed(m) for m in ("dictionary", "diffusion")},
                  noise_seeds=list(noise_seeds), sigmas_requested=list(sigmas), summaries=summaries,
                  records=records, timings=timings, prediction_stability=stability, code_stability=code_stability,
                  subspace_stability={str(seed): subspace_stability(data, seed) for seed in noise_seeds},
                  tightening_geometry=tightening,
                  conditioning="known grasp type, primitive and source size; no unseen-object claim",
                  geometry_definition="fixed source-size object; no clipping or geometry repair",
                  limitations=["Residual contains noise and model error as well as possible geometry bias",
                               "Interpolated sizes are not independent solved training examples",
                               "Reference posture validity does not establish force or stable grasp"])
    write_json(data.directory / "evaluation.json", result)
    return result


class PoseChecker:
    """Independent MuJoCo FK, exact object-mesh distance and native collisions."""
    def __init__(self, data, index):
        from retarget.env.geometry import ObjectMeshSurface
        import mujoco
        self.experiment, self.index = data, index
        snapshot = data.get_snapshot()
        self.source_index = int(data.source_index[index])
        self.model = snapshot.model(self.source_index)
        self.state = mujoco.MjData(self.model)
        self.surface = ObjectMeshSurface.from_model(self.model, self.model.body("grasp_object").id)
        self.template = snapshot.template(self.source_index)
        self.bodies = np.array([self.model.body(n).id for n in self.template["body_names"]])
        self.local = np.asarray(self.template["local_points"])
        self.normals = np.asarray(self.template["local_normals"])

    def inspect(self, q):
        from retarget.env.geometry import closest_triangles
        import mujoco
        q = np.asarray(q)
        if q.shape != (26,) or not np.isfinite(q).all():
            raise ValueError("Pose checker needs a finite 26D pose")
        self.state.qpos[:] = q
        depth = collision_depths(self.model, self.state)
        rotation = self.state.xmat[self.bodies].reshape(-1, 3, 3)
        points = self.state.xpos[self.bodies] + np.einsum("nij,nj->ni", rotation, self.local)
        normals = np.einsum("nij,nj->ni", rotation, self.normals)
        obj = self.model.body("grasp_object").id
        obj_rotation = self.state.xmat[obj].reshape(3, 3)
        local_points = (points-self.state.xpos[obj]) @ obj_rotation
        near = closest_triangles(local_points, self.surface.triangles)
        distance = np.linalg.norm(near-local_points[:, None], axis=-1)
        triangle = distance.argmin(-1)
        nearest = near[np.arange(len(points)), triangle]
        outward = self.surface.normals[triangle]
        signed = np.sign(((local_points-nearest)*outward).sum(-1)) * distance.min(-1)
        # At mesh edges the nearest-face normal gives a local signed-distance proxy.
        normal_dot = ((normals @ obj_rotation)*-outward).sum(-1)
        covered = (signed >= -.002) & (signed <= .002) & (normal_dot >= np.cos(np.deg2rad(30)))
        tips = np.array([self.state.site_xpos[self.model.site(f"right_{name}_tip").id].copy()
                         for name in ("thumb", "index", "middle", "ring", "pinky")])
        fingers = sorted({int(name.split("finger")[1].split("_")[0])-1
                          for name in self.template["body_names"] if "finger" in name})
        pairs = [np.linalg.norm(tips[i]-tips[j]) for i in fingers for j in fingers if i < j]
        angle = self.experiment.denormalize(np.ones(20))
        low = self.experiment.denormalize(-np.ones(20))
        violation = np.maximum(low-q[6:], q[6:]-angle).clip(0)
        # RMS pair distance of required template contacts measures enclosure;
        # fingertip distances are reported separately and may move differently.
        lengths = np.linalg.norm(points[:, None]-points[None], axis=-1)
        upper = lengths[np.triu_indices(len(points), 1)]
        metrics = dict(max_joint_violation_rad=float(violation.max()),
                       hand_object_penetration_m=depth[0], self_penetration_m=depth[1],
                       contact_gap_mean_m=float(signed.mean()), contact_absolute_distance_mean_m=float(distance.min(-1).mean()),
                       contact_gap_min_m=float(signed.min()), contact_gap_max_m=float(signed.max()),
                       template_point_coverage=float(covered.mean()),
                       contact_enclosure_rms_m=float(np.sqrt(np.mean(upper**2))) if len(upper) else 0.,
                       active_fingertip_pair_mean_m=float(np.mean(pairs)) if pairs else 0.,
                       contact_pair_distance_mean_m=float(upper.mean()) if len(upper) else 0.)
        joints = [self.state.xpos[self.model.body("right_palm_link").id].copy()]
        for i in range(1, 6):
            joints.extend(self.state.xpos[self.model.body(f"right_finger{i}_link{k}").id].copy() for k in (2, 3, 4))
            joints.append(tips[i-1])
        return metrics, dict(joints=np.asarray(joints).tolist(), contacts=points.tolist(),
                             projections=(nearest @ obj_rotation.T+self.state.xpos[obj]).tolist())
