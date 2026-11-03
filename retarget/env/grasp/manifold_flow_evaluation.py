"""Paired held-out-size evaluation for both pose-flow backends and baselines."""
from __future__ import annotations

import json

import numpy as np
import torch
from torch.nn import functional as F

from .manifold import ExperimentData, PoseChecker, Predictor, tensors, write_json
from .manifold_flow import (
    FLOW_METHODS,
    file_digest,
    finger_corruption,
    load_flow_model,
    read_registry,
)


def validate_baseline(data, baseline):
    """Comparisons require identical original nodes, normalization and splits."""
    for name in ("qpos", "source_index", "group", "size", "split", "mid", "half"):
        if not np.array_equal(getattr(data, name), getattr(baseline, name)):
            raise ValueError(f"Baseline experiment differs in {name}; paired comparison is invalid")
    if data.meta["groups"] != baseline.meta["groups"] or data.meta["source_sha256"] != baseline.meta["source_sha256"]:
        raise ValueError("Baseline condition schema or source snapshot differs")
    if baseline.resolve_method("diffusion") != "diffusion":
        raise ValueError("Baseline must contain the legacy DDPM diffusion")


@torch.no_grad()
def latent_metrics(model, clean, observed, groups, sizes):
    z, aux = model.autoencoder.encode(observed)
    cz, caux = model.autoencoder.encode(clean)
    labels = model.group_types[groups]
    distances = torch.cdist(z, z)
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    same_type = labels[:, None] == labels[None]
    within = same_type & (sizes[:, None] != sizes[None]) & ~eye
    between = ~same_type
    result = dict(noisy_clean_latent_mse=float((z - cz).square().mean()),
                  noisy_clean_feature_cosine=float(F.cosine_similarity(aux["features"], caux["features"]).mean()),
                  cross_size_same_type_distance=float(distances[within].mean()) if within.any() else None,
                  different_type_distance=float(distances[between].mean()) if between.any() else None,
                  feature_std=float(aux["features"].std(0, unbiased=False).mean()))
    if model.kind == "vq":
        indices = torch.cat((aux["indices"], caux["indices"]))
        counts = torch.bincount(indices, minlength=model.config.codebook_size).float()
        probabilities = counts[counts > 0] / counts.sum()
        result.update(codebook_used=int((counts > 0).sum()),
                      codebook_size=model.config.codebook_size,
                      codebook_perplexity=float((-(probabilities * probabilities.log()).sum()).exp()),
                      clean_noisy_code_agreement=float((aux["indices"] == caux["indices"]).float().mean()))
    else:
        result["kl"] = float(aux["kl"])
    return result


def summarize(records):
    results = []
    keys = sorted({(r["method"], r["model_seed"], r["split"]) for r in records})
    for method, model_seed, split in keys:
        rows = [r for r in records if (r["method"], r["model_seed"], r["split"]) == (method, model_seed, split)]
        groups = sorted({r["group"] for r in rows})
        def macro(subset, metric, groups=groups):
            values = [np.mean([r[metric] for r in subset if r["group"] == g])
                      for g in groups if any(r["group"] == g for r in subset)]
            return float(np.mean(values)) if values else None
        by_noise = {}
        for sigma in sorted({r["sigma_step"] for r in rows}):
            subset = [r for r in rows if r["sigma_step"] == sigma]
            by_noise[str(sigma)] = {metric: macro(subset, metric) for metric in ("rmse_normalized", "rmse_rad")}
        by_count = {str(n): macro([r for r in rows if r["noisy_fingers"] == n], "rmse_normalized")
                    for n in range(6)}
        results.append(dict(method=method, model_seed=model_seed, split=split,
                            macro_rmse_normalized=macro(rows, "rmse_normalized"),
                            mean_rmse_rad=macro(rows, "rmse_rad"), by_noise=by_noise,
                            by_noisy_fingers=by_count))
    return results


def evaluate_flow(data, baseline_directory, seeds=None, noise_seeds=(11, 23, 37), device="auto", *,
                  geometry=True, sigmas=None):
    """Write per-seed validation/test results; selection uses validation only.

    Partial-finger noise is shared by every method. A legacy scalar sigma is
    the RMS of the known per-finger standard deviations, a documented
    approximation required by its isotropic-noise API.
    """
    if not noise_seeds or len(set(noise_seeds)) != len(noise_seeds):
        raise ValueError("Need unique, nonempty noise seeds")
    if sigmas is not None and (not sigmas or len(set(sigmas)) != len(sigmas) or any(not np.isfinite(s) or s < 0 for s in sigmas)):
        raise ValueError("Need unique, finite nonnegative per-round sigmas")
    registry = read_registry(data)
    specifications = []
    for method in FLOW_METHODS:
        available = sorted(int(key.rsplit("seed", 1)[1]) for key in registry.get("runs", {})
                           if key.startswith(method + "_seed"))
        if not available:
            continue
        wanted = tuple(seeds) if seeds is not None else available
        for seed in wanted:
            if seed not in available:
                raise FileNotFoundError(f"No completed {method} seed {seed}")
            specifications.append((method, int(seed)))
    if not specifications:
        raise ValueError("No trained flow checkpoints available")
    baseline = ExperimentData(baseline_directory)
    validate_baseline(data, baseline)
    baseline_predictors = {m: Predictor(baseline, m, device=device) for m in ("pca", "interpolation", "diffusion")}
    loaded = {(m, seed): load_flow_model(data, m, seed, device)[0] for m, seed in specifications}
    if sigmas is None:
        sigmas = tuple(sorted({0., .03, .1, *(model.config.sigma_max for model in loaded.values())}))
    device = next(iter(loaded.values())).group_types.device
    max_rounds = {m.config.max_rounds for m in loaded.values()}
    if len(max_rounds) != 1:
        raise ValueError("Paired evaluation requires the same max_rounds across models")
    cases, noise_export = {}, {}
    for split in ("validation", "test"):
        ids = np.flatnonzero(data.mask(split))
        x, groups, sizes = tensors(data, ids, device)
        split_cases = []
        for seed in noise_seeds:
            generator = torch.Generator(device=device).manual_seed(seed)
            for sigma in sigmas:
                observed, info = finger_corruption(x, generator, max_rounds=next(iter(max_rounds)),
                                                  sigma_min=sigma or .03, sigma_max=sigma or .03,
                                                  finger_counts=torch.zeros(len(x), device=device) if sigma == 0 else None)
                split_cases.append((seed, sigma, observed, info))
        cases[split] = ids, x, groups, sizes, split_cases
        noise_export.update({f"{split}_indices": ids,
                             f"{split}_observed": np.stack([c[2].cpu().numpy() for c in split_cases]),
                             f"{split}_mask": np.stack([c[3]["mask"].cpu().numpy() for c in split_cases]),
                             f"{split}_rounds": np.stack([c[3]["rounds"].cpu().numpy() for c in split_cases]),
                             f"{split}_noise": np.stack([c[3]["noise"].cpu().numpy() for c in split_cases]),
                             f"{split}_noise_seed": [c[0] for c in split_cases],
                             f"{split}_sigma_step": [c[1] for c in split_cases]})
    np.savez_compressed(data.directory / "evaluation_noise.npz", **noise_export)
    records, latents, geometry_records = [], [], []
    for split, (ids, x, groups, sizes, split_cases) in cases.items():
        references = x.cpu().numpy()
        chosen = [int(np.flatnonzero(data.group[ids] == group)[0]) for group in np.unique(data.group[ids])]
        checkers = {j: PoseChecker(data, int(ids[j])) for j in chosen} if geometry and split == "test" else {}
        for noise_seed, sigma, observed, info in split_cases:
            obs = observed.cpu().numpy()
            rms_sigma = info["sigma_fingers"].square().mean(-1).sqrt().cpu().numpy()
            predictions = []
            for name, predictor in baseline_predictors.items():
                if name == "diffusion":
                    # Each sample can have a different effective isotropic sigma.
                    pred = np.vstack([predictor.predict(obs[j:j+1], data.group[ids[j:j+1]],
                                                       data.size[ids[j:j+1]], float(rms_sigma[j]))
                                      for j in range(len(ids))])
                else:
                    pred = predictor.predict(obs, data.group[ids], data.size[ids], sigma)
                predictions.append(("legacy_diffusion" if name == "diffusion" else name,
                                    baseline.model_seed(name) if name == "diffusion" else 0, pred))
            for (method, seed), model in loaded.items():
                predictions.append((method, seed, model.denoise(observed, groups, sizes).cpu().numpy()))
                if noise_seed == noise_seeds[0]:
                    latents.append(dict(method=method, model_seed=seed, split=split, sigma_step=sigma,
                                        **latent_metrics(model, x, observed, groups, sizes)))
            for method, model_seed, prediction in predictions:
                delta = prediction - references
                rmse = np.sqrt(np.mean(delta ** 2, -1))
                radians = np.sqrt(np.mean((delta * data.half) ** 2, -1))
                for j, index in enumerate(ids):
                    records.append(dict(method=method, model_seed=model_seed, split=split,
                                        index=int(index), group=int(data.group[index]), noise_seed=noise_seed,
                                        sigma_step=sigma, noisy_fingers=int(info["mask"][j].sum()),
                                        rmse_normalized=float(rmse[j]), rmse_rad=float(radians[j])))
                if sigma == .05 and noise_seed == noise_seeds[0]:
                    for j, checker in checkers.items():
                        q = data.qpos[ids[j]].copy()
                        q[6:] = data.denormalize(prediction[j])
                        geometry_records.append(dict(method=method, model_seed=model_seed, index=int(ids[j]),
                                                      metrics=checker.inspect(q)[0]))
    summaries = summarize(records)
    validation = [r for r in summaries if r["split"] == "validation"]
    selected_seeds = {}
    for method in FLOW_METHODS:
        candidates = [r for r in validation if r["method"] == method]
        if candidates:
            best = min(candidates, key=lambda r: (r["macro_rmse_normalized"], r["model_seed"]))
            selected_seeds[method] = best["model_seed"]
    selected = min((r for r in validation if r["method"] in FLOW_METHODS), key=lambda r: r["macro_rmse_normalized"])
    # Evaluation scores have a different unit from training MSE: keep the two
    # registries separate and update only the selected seed IDs.
    registry["selected_seeds"] = {**registry.get("selected_seeds", {}), **selected_seeds}
    write_json(data.directory / "models.json", registry)
    training = []
    for method, seed in specifications:
        path = data.directory / "models" / f"{method}_seed{seed}" / "training.json"
        training.append(json.loads(path.read_text()))
    result = dict(architecture="pose_conditioned_flow_v2", selection_split="validation",
                  selected_method=selected["method"], selected_seeds=selected_seeds,
                  best_comparison_method=min(validation, key=lambda r: r["macro_rmse_normalized"])["method"],
                  summaries=summaries, records=records, latent_metrics=latents, geometry=geometry_records,
                  completed_training_runs=training, baseline=str(baseline.directory),
                  baseline_checkpoint_sha256=file_digest(baseline.directory / "models" /
                      f"diffusion_seed{baseline.model_seed('diffusion')}" / "best.pt"),
                  noise_seeds=list(noise_seeds), max_rounds=next(iter(max_rounds)),
                  sigma_steps=list(sigmas),
                  conditioning="known grasp type, primitive and size; no noise metadata for flow",
                  limitations=["Legacy DDPM receives an RMS isotropic approximation for partial-finger noise",
                               "These splits measure interpolation of sizes within known grasp types",
                               "Pose and collision metrics do not establish dynamic grasp stability"])
    write_json(data.directory / "evaluation.json", result)
    lines = ["# Pose flow comparison", "", "Selection uses validation only. Errors are macro-averaged over grasp groups.",
             "", "| Model | Seed | Validation RMSE | Test RMSE | Test RMSE (rad) |", "|---|---:|---:|---:|---:|"]
    for row in validation:
        test = next(r for r in summaries if r["method"] == row["method"] and r["model_seed"] == row["model_seed"] and r["split"] == "test")
        lines.append(f"| {row['method']} | {row['model_seed']} | {row['macro_rmse_normalized']:.6f} | "
                     f"{test['macro_rmse_normalized']:.6f} | {test['mean_rmse_rad']:.6f} |")
    lines.extend(("", f"Selected flow: {selected['method']}, seed {selected['model_seed']}.", "",
                  "See evaluation.json for latent distances, codebook usage, corruption buckets and geometry diagnostics."))
    (data.directory / "comparison.md").write_text("\n".join(lines) + "\n")
    return result
