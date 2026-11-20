"""Streaming supervised MLP training, resumable optimizer state and held-out audits."""
import json
import math
from pathlib import Path
import signal
import time

import numpy as np
import torch
from torch.nn import functional as F

from .common import atomic_json, atomic_save, fingerprint, sha256
from .data import batches, read_manifest
from .model import ConfigurationSDF, load_model


def tensors(batch, device):
    return {k: torch.as_tensor(v, device=device) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, folder, manifest, split, device, batch_size=8192, max_queries=200000):
    model.eval()
    errors, truths, predictions, links, collisions = [], [], [], [], []
    for batch in batches(folder, manifest, split, batch_size, max_queries=max_queries):
        b = tensors(batch, device)
        prediction = model(b['q'], b['points'], b['links']).cpu().numpy()
        predictions.append(prediction)
        errors.append(prediction - batch['sdf'])
        truths.append(batch['sdf'])
        links.append(batch['links'])
        collisions.append(batch['collision'])
    if not errors:
        raise ValueError('No evaluation queries')
    error, truth, prediction, link, collision = map(np.concatenate, (errors, truths, predictions, links, collisions))
    absolute = abs(error) * 1000
    near = abs(truth) <= .01
    inside = truth < -.0005
    away = abs(truth) > .0001
    def mae(mask):
        return float(absolute[mask].mean()) if mask.any() else None
    report = dict(queries=len(error), mae_mm=float(absolute.mean()), rmse_mm=float(np.sqrt(np.mean(error ** 2)) * 1000),
                  p95_mm=float(np.percentile(absolute, 95)), near_10mm_queries=int(near.sum()), near_mae_mm=mae(near),
                  sign_accuracy_away_0p1mm=float(np.mean((prediction[away] < 0) == (truth[away] < 0))) if away.any() else None,
                  penetration_queries=int(inside.sum()),
                  penetration_false_negative_rate=float(np.mean(prediction[inside] >= 0)) if inside.any() else None,
                  collision_pose_queries=int(collision.sum()), collision_pose_mae_mm=mae(collision),
                  free_pose_mae_mm=mae(~collision),
                  per_link=[dict(link=i, queries=int((link == i).sum()), mae_mm=mae(link == i)) for i in range(21)])
    return report


def train(args):
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision('high')
    if args.epochs <= 0 or args.batch_size <= 0 or args.lr <= 0 or args.eikonal_weight < 0 or args.sign_weight < 0:
        raise ValueError('Invalid training hyperparameters')
    manifest = read_manifest(args.data, verify=True)
    geometry = manifest['spec']['geometry']
    domain = geometry['domain']
    model_config = dict(lower=domain['lower'], upper=domain['upper'], joint_mask=geometry['joint_mask'],
                        width=args.width, depth=args.depth, embedding=16)
    signature = dict(dataset_fingerprint=manifest['fingerprint'],
                     shards=[s['sha256'] for s in manifest['shards']], model_config=model_config,
                     epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed,
                     eikonal_weight=args.eikonal_weight, sign_weight=args.sign_weight,
                     val_queries=args.val_queries, steps_per_epoch=args.steps_per_epoch,
                     code={n: sha256(Path(__file__).with_name(n)) for n in ('train.py', 'model.py', 'data.py')})
    key = fingerprint(signature)
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    model = ConfigurationSDF(**model_config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr * .01)
    epoch = next_batch = step = 0
    best = math.inf
    history = []
    partial_sum = partial_count = 0
    checkpoint = folder / 'last.pt'
    if checkpoint.exists():
        if not args.resume:
            raise ValueError('Training output exists; use --resume or a fresh output')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if saved['fingerprint'] != key:
            raise ValueError('Resume configuration/data/code changed; use a fresh output')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        epoch, next_batch, step = saved['epoch'], saved['next_batch'], saved['step']
        best, history = saved['best'], saved['history']
        partial_sum, partial_count = saved['partial_sum'], saved['partial_count']
        torch.set_rng_state(saved['torch_rng'])
        if saved.get('cuda_rng') is not None and str(args.device).startswith('cuda'):
            torch.cuda.set_rng_state(saved['cuda_rng'], args.device)
        if saved.get('complete'):
            return json.loads((folder / 'report.json').read_text())
    atomic_json(folder / 'configuration.json', dict(fingerprint=key, signature=signature, device=args.device,
                                                  data=str(Path(args.data).resolve()), geometry=geometry))
    stopped = False
    def stop(*_):
        nonlocal stopped
        stopped = True
    handlers = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
    def save(complete=False):
        atomic_save(checkpoint, dict(model_config=model_config, model=model.state_dict(), optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(), epoch=epoch, next_batch=next_batch, step=step, best=best,
                    history=history, partial_sum=partial_sum, partial_count=partial_count,
                    fingerprint=key, torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state(args.device) if str(args.device).startswith('cuda') else None,
                    complete=complete))
    began = time.monotonic()
    last_status = began
    try:
        while epoch < args.epochs:
            model.train()
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, epoch, 19]))
            for index, batch in enumerate(batches(args.data, manifest, 0, args.batch_size, rng=rng)):
                if args.steps_per_epoch and index >= args.steps_per_epoch:
                    break
                if index < next_batch:
                    continue
                b = tensors(batch, args.device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(b['q'], b['points'], b['links'])
                # Predict the continuous untruncated metric distance. Near-surface
                # queries receive more weight without saturating the output.
                weight = torch.where(b['sdf'].abs() < .01, 4., 1.)
                distance_loss = (weight * F.smooth_l1_loss(prediction / .01, b['sdf'] / .01,
                                                          reduction='none', beta=.1)).mean()
                sign_mask = b['sdf'].abs() > .0001
                sign_loss = (F.softplus(-b['sdf'][sign_mask].sign() * prediction[sign_mask] / .001).mean()
                             if bool(sign_mask.any()) else prediction.sum() * 0)
                eikonal = prediction.sum() * 0
                if args.eikonal_weight:
                    # Differentiate in metres; only a small independent subset
                    # needs second derivatives, keeping GPU memory bounded.
                    x = b['points'][:128].detach().requires_grad_(True)
                    values = model(b['q'][:128], x, b['links'][:128])
                    grad = torch.autograd.grad(values.sum(), x, create_graph=True)[0]
                    eikonal = (grad.norm(dim=-1) - 1).square().mean()
                loss = distance_loss + args.sign_weight * sign_loss + args.eikonal_weight * eikonal
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Nonfinite training objective')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                step += 1
                next_batch = index + 1
                partial_sum += float(loss.detach()) * len(prediction)
                partial_count += len(prediction)
                now = time.monotonic()
                if now - last_status > 15 or step == 1:
                    status = dict(state='training', epoch=epoch + 1, batch=next_batch, step=step,
                                  loss=partial_sum / partial_count, seconds=now - began)
                    atomic_json(folder / 'status.json', status)
                    print(json.dumps(status), flush=True)
                    last_status = now
                if stopped or step % args.checkpoint_steps == 0:
                    save()
                if stopped:
                    atomic_json(folder / 'status.json', dict(state='paused', epoch=epoch, next_batch=next_batch, step=step))
                    return dict(state='paused')
            val = evaluate(model, args.data, manifest, 1, args.device, max_queries=args.val_queries)
            score = val['near_mae_mm'] + .1 * val['mae_mm']
            improved = score < best
            best = min(best, score)
            epoch += 1
            next_batch = 0
            record = dict(epoch=epoch, step=step, loss=partial_sum / max(1, partial_count), validation=val,
                          seconds=time.monotonic() - began)
            history.append(record)
            scheduler.step()
            partial_sum = partial_count = 0
            if improved:
                atomic_save(folder / 'best.pt', dict(model_config=model_config, model=model.state_dict(),
                            fingerprint=key, dataset_fingerprint=manifest['fingerprint'], geometry=geometry,
                            epoch=epoch, validation=val))
            save()
            atomic_json(folder / 'history.json', history)
            print(json.dumps(dict(phase='validation', epoch=epoch, **val)), flush=True)
        best_model, _ = load_model(folder / 'best.pt', args.device)
        report = evaluate(best_model, args.data, manifest, 2, args.device, max_queries=args.val_queries)
        report.update(state='completed', dataset_fingerprint=manifest['fingerprint'], training_steps=step,
                      epochs=epoch, best_validation_score=best, held_out_split='test_poses',
                      limited_steps_per_epoch=args.steps_per_epoch or None)
        report['configuration_response'] = configuration_response(best_model, args.data, manifest, args.device)
        atomic_json(folder / 'report.json', report)
        save(complete=True)
        atomic_json(folder / 'status.json', dict(state='completed', epochs=epoch, step=step,
                                               test_mae_mm=report['mae_mm'], test_near_mae_mm=report['near_mae_mm']))
        return report
    finally:
        for s, handler in handlers.items():
            signal.signal(s, handler)


@torch.no_grad()
def configuration_response(model, data, manifest, device, count=128):
    """Same points + same moving links at TWO held-out poses, compared to mesh.

    A q-independent static network predicts zero delta here and fails the audit.
    """
    from .geometry import Geometry
    qrows = []
    for s in manifest['shards']:
        with np.load(Path(data) / s['file'], allow_pickle=False) as a:
            qrows.extend(a['qpos'][a['split'] == 2])
        if len(qrows) >= count:
            break
    q = np.asarray(qrows[:count], np.float32)
    if len(q) < 2:
        return dict(pose_pairs=0, reason='At least two held-out poses required')
    teacher = Geometry(device)
    points, links = teacher.queries(teacher.frames(q), 8, np.random.default_rng(98741))
    # Omit fixed palm and use an independent held-out pose for each same point.
    points, links = points[:, 8:], links[:, 8:]
    q2 = np.roll(q, 1, axis=0).copy()
    true1 = teacher.label(points, links, teacher.frames(q))
    true2 = teacher.label(points, links, teacher.frames(q2))
    x = torch.as_tensor(points, device=device)
    l = torch.as_tensor(links.astype(np.int64), device=device)
    pred1 = model(torch.as_tensor(q, device=device)[:, None], x, l).cpu().numpy()
    pred2 = model(torch.as_tensor(q2, device=device)[:, None], x, l).cpu().numpy()
    actual, predicted = true2 - true1, pred2 - pred1
    informative = abs(actual) > .005
    return dict(pose_pairs=len(q), queries=int(actual.size), mesh_delta_mean_abs_mm=float(abs(actual).mean() * 1000),
                predicted_delta_mean_abs_mm=float(abs(predicted).mean() * 1000),
                delta_mae_mm=float(abs(actual - predicted).mean() * 1000),
                static_zero_delta_mae_mm=float(abs(actual).mean() * 1000),
                informative_queries=int(informative.sum()),
                delta_direction_accuracy=float(np.mean(np.sign(actual[informative]) == np.sign(predicted[informative])))
                    if informative.any() else None)
