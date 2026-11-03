"""Repeated Gaussian corruption, paired denoisers, and exact noise add-back.

Reuses the trained experiment and the inspection-webpages skill shell. Geometry
is shared across stages rather than duplicating millions of mesh vertices.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import torch

from .manifold import (PoseChecker, Predictor, collision_depths,
                       sigma_timestep, write_json)
from .manifold_report import FK_JS, visual_model
from .manifold_flow import finger_corruption

ROOT = Path(__file__).resolve().parents[3]
METHODS = ("pca", "diffusion")
STAGES = ("clean", "tight", "noise", "removed", "perpendicular")


def corruption_path(clean, sigma, rounds, seed):
    """Independent Gaussian increments; no clipping or fresh reverse noise."""
    if not np.isfinite(sigma) or sigma <= 0 or rounds < 1:
        raise ValueError("Positive finite per-round sigma and at least one round required")
    rng = np.random.default_rng(seed)
    increments = sigma * rng.standard_normal((rounds, 20))
    noise = np.vstack([np.zeros(20), np.cumsum(increments, axis=0)])
    return np.asarray(clean) + noise, noise, increments


def finger_corruption_path(clean, sigma, rounds, seed):
    """The training corruption, exported as a cumulative per-round path."""
    _, info = finger_corruption(torch.as_tensor(np.asarray(clean)[None], dtype=torch.float64),
                               torch.Generator().manual_seed(seed), max_rounds=rounds,
                               sigma_min=sigma, sigma_max=sigma)
    increments = info['increments'][:, 0].numpy()
    noise = np.vstack([np.zeros(20), np.cumsum(increments, axis=0)])
    metadata = dict(finger_mask=info['mask'][0].tolist(), finger_rounds=info['rounds'][0].tolist())
    return np.asarray(clean) + noise, noise, increments, metadata


def compose_stages(clean, delta, noise, observed, perpendicular):
    tight = np.asarray(clean) + delta
    return dict(clean=np.asarray(clean), tight=tight, noise=tight+noise,
                removed=tight+(observed-clean), perpendicular=tight+perpendicular)


def quick_metrics(checker, q, reference, half):
    """Native collisions and contact enclosure; no force/stability claims."""
    model, state = checker.model, checker.state
    state.qpos[:] = q
    depths = collision_depths(model, state)
    rotation = state.xmat[checker.bodies].reshape(-1, 3, 3)
    points = state.xpos[checker.bodies] + np.einsum("nij,nj->ni", rotation, checker.local)
    lengths = np.linalg.norm(points[:, None]-points[None], axis=-1)
    upper = lengths[np.triu_indices(len(points), 1)]
    x = checker.experiment.normalize(q[6:])
    return dict(rmse=float(np.sqrt(np.mean(((q[6:]-reference[6:])/half)**2))),
                enclosure_mm=float(np.sqrt(np.mean(upper**2))*1000) if len(upper) else 0.,
                object_mm=depths[0]*1000, self_mm=depths[1]*1000,
                violation_rad=float((np.maximum(np.abs(x)-1, 0)*half).max()))


def build_inspector(data, destination, *, sigma=.05, rounds=5, seeds=(11, 23, 37),
                    fractions=(0., .1, .2, .35, .5, .65, .8, .9), device="auto", viser_url="http://127.0.0.1:8090",
                    method="diffusion", noise_mode="fingers"):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Noise seeds must be nonempty and unique")
    if not fractions or len(set(fractions)) != len(fractions) or any(not 0 <= f < 1 for f in fractions):
        raise ValueError("Unique tightening fractions must lie in [0,1)")
    if method not in ('diffusion', 'diffusion_vq', 'diffusion_vae') or noise_mode not in ('fingers', 'all'):
        raise ValueError("Unknown diffusion method or noise mode")
    if not np.isfinite(sigma) or sigma <= 0 or not isinstance(rounds, int) or rounds < 1:
        raise ValueError("Need positive finite per-round sigma and positive integer rounds")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    predictors = dict(pca=Predictor(data, 'pca', device=device), diffusion=Predictor(data, method, device=device))
    flow = predictors['diffusion'].is_flow
    models, meshes, cases, exported, records = {}, {}, [], [], []
    max_addback_error = 0.
    for group, info in enumerate(data.meta["groups"]):
        ids = np.flatnonzero(data.group == group)
        # The standard pose is the saved 1x node, not a random held-out pose.
        index = int(ids[np.argmin(abs(data.size[ids]-1.))])
        source = float(data.size[index])
        checker = PoseChecker(data, index)
        models[str(group)] = visual_model(checker.model, meshes, str(group))
        source_case = data.get_snapshot().manifest["cases"][int(data.source_index[index])]
        reference = data.qpos[index].copy()
        x = data.x[index]

        def qpos(value):
            q = reference.copy()
            q[6:] = data.denormalize(value)
            return q

        curves = {}
        for fraction in fractions:
            _, target, delta = predictors['pca'].curves[group].tighten(x, np.zeros(20), source, fraction)
            # Both methods use the same cached, training-node-only PCA size curve.
            curves[str(fraction)] = dict(target=target, delta=delta.tolist())
        item = dict(group=group, label=info['template'], primitive=info['primitive'],
                    index=index, source_index=int(data.source_index[index]), source_size=source,
                    split=("train", "validation", "test")[int(data.split[index])],
                    size_axis=source_case['primitive']['size_axis'],
                    dimensions_m=source_case['primitive']['dimensions_m'],
                    truth=reference.tolist(), curves=curves,
                    truth_metrics=quick_metrics(checker, reference, reference, data.half), paths=[])
        for seed in seeds:
            effective_seed = seed+int(data.source_index[index])
            if noise_mode == 'fingers':
                observed, noises, increments, noise_info = finger_corruption_path(x, sigma, rounds, effective_seed)
            else:
                observed, noises, increments = corruption_path(x, sigma, rounds, effective_seed)
                noise_info = dict(finger_mask=[True] * 5, finger_rounds=[rounds] * 5)
            path = dict(seed=seed, effective_seed=seed+int(data.source_index[index]),
                        increments=increments.tolist(), rounds=[], **noise_info)
            for r, (obs, noise) in enumerate(zip(observed, noises)):
                finger_sigmas = sigma * np.sqrt(np.minimum(r, noise_info['finger_rounds']))
                total_sigma = float(np.sqrt(np.mean(finger_sigmas ** 2)))
                if flow:
                    timing = dict(integration_start=0., integration_end=1.,
                                  integration_steps=predictors['diffusion'].model.config.flow_steps)
                else:
                    t = sigma_timestep(total_sigma, predictors['diffusion'].schedule)
                    a = predictors['diffusion'].schedule[t]
                    timing = dict(timestep=t, model_sigma=math.sqrt((1-a)/a))
                _, perpendicular = data.subspaces[group].decompose(obs)
                round_data = dict(round=r, sigma=total_sigma, **timing,
                                  sigma_fingers=finger_sigmas.tolist(), noise=noise.tolist(),
                                  observed=qpos(obs).tolist(), perpendicular=perpendicular.tolist(),
                                  observed_metrics=quick_metrics(checker, qpos(obs), reference, data.half), methods={})
                for method_key, predictor in predictors.items():
                    clean = predictor.predict(obs[None], [group], [source], total_sigma)[0]
                    result = dict(clean=qpos(clean).tolist(), fractions={})
                    for fraction in fractions:
                        delta = np.asarray(curves[str(fraction)]['delta'])
                        values = compose_stages(clean, delta, noise, obs, perpendicular)
                        metrics = {}
                        poses = []
                        for stage, value in values.items():
                            q = qpos(value)
                            metrics[stage] = quick_metrics(checker, q, reference, data.half)
                            poses.append(q)
                        max_addback_error = max(max_addback_error, float(np.max(abs(values['noise']-values['tight']-noise))))
                        result['fractions'][str(fraction)] = metrics
                        exported.append(np.stack(poses))
                        records.append((group, index, seed, r, method_key, fraction))
                    round_data['methods'][method_key] = result
                path['rounds'].append(round_data)
            item['paths'].append(path)
        cases.append(item)
        print(f"inspector: {info['template']} / {len(cases)} of {len(data.subspaces)}", flush=True)
    payload = dict(version=1, sigma_step=sigma, rounds=rounds, seeds=list(seeds),
                   fractions=list(fractions), cases=cases, models=models, meshes=meshes,
                   cached_poses=len(data.qpos),
                   mid=data.mid.tolist(), half=data.half.tolist(), methods=list(METHODS),
                   diffusion_seed=data.model_seed(method), diffusion_method=predictors['diffusion'].method,
                   flow=flow, noise_mode=noise_mode, viser_url=viser_url,
                   source=str(data.source), experiment=str(data.directory))
    write_json(destination/'experiment.json', {k: v for k, v in payload.items() if k not in ('meshes', 'models')})
    np.savez_compressed(destination/'poses.npz', qpos=np.stack(exported), stages=np.array(STAGES),
                        group=[r[0] for r in records], index=[r[1] for r in records],
                        seed=[r[2] for r in records], round=[r[3] for r in records],
                        method=[r[4] for r in records], fraction=[r[5] for r in records])
    # Generate the accepted local skill shell, then use shared mesh/FK rendering.
    skill = ROOT/'skills/inspection-webpages'
    spec = importlib.util.spec_from_file_location('inspection_webpages', skill/'scripts/build_page.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    contract = dict(title='Grasp pose · 高斯加噪与去噪收紧对照',
                    eyebrow='WUJI2 / DENOISING INSPECTOR',
                    subtitle=f'{len(cases)} 类标准缓存姿态；同一加噪输入分别经过 PCA 与 {predictors["diffusion"].method}，再沿尺寸曲线收紧并加回原始噪声。',
                    views=[dict(label=s) for s in ('标准缓存姿态', '多轮高斯加噪', 'PCA', 'Diffusion')],
                    cases=[dict(id='shared', variants=[dict(id='initial', figures=[dict(data=[]) for _ in range(4)])])])
    output = destination/'index.html'
    builder.build(contract, output)
    html = output.read_text()
    # Keep the skill's styles, accessible shell, panel placement and Plotly bundle.
    start = html.index("<script>\n'use strict';")
    encoded = json.dumps(payload, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    for char, escaped in (('&', r'\u0026'), ('<', r'\u003c'), ('>', r'\u003e')):
        encoded = encoded.replace(char, escaped)
    app = (Path(__file__).with_name('denoising_inspector.js')).read_text()
    html = html[:start] + '<script>\nconst DATA=' + encoded + ';\n' + FK_JS + '\n' + app + '\n</script></body></html>'
    output.write_text(html)
    (destination/'index.data.json').unlink()  # experiment.json is the full domain data.
    verification = dict(groups=len(cases), seeds=list(seeds), rounds=rounds,
                        comparisons=len(records), gaussian_addback_max_error=max_addback_error,
                        poses_shape=list(np.stack(exported).shape), stages=list(STAGES),
                        diffusion_seed=data.model_seed(method), diffusion_method=predictors['diffusion'].method,
                        flow=flow, noise_mode=noise_mode,
                        sigma_step=sigma, source_sha256=data.meta['source_sha256'],
                        checkpoint_sha256=hashlib.sha256((data.directory/'models'/f'{predictors["diffusion"].method}_seed{data.model_seed(method)}'/'best.pt').read_bytes()).hexdigest())
    write_json(destination/'verification.json', verification)
    return output
