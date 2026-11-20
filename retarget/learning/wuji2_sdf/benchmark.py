"""Audit every query in a newly sampled dataset against frozen model weights."""
from pathlib import Path
import time

import numpy as np
import torch

from .common import atomic_json, fingerprint, sha256
from .data import read_manifest
from .model import load_model


def pose_keys(q):
    q = np.asarray(q, dtype='<f4').copy()
    q[q == 0] = 0  # Canonicalize signed zero for overlap detection.
    return {row.tobytes() for row in q}


def distance_metrics(truth, prediction):
    """Metric units in mm; sign statistics exclude the ambiguous boundary band."""
    error = (prediction.astype(np.float64) - truth.astype(np.float64)) * 1000
    absolute = abs(error)
    inside, outside = truth < -.0005, truth > .0005
    away = abs(truth) > .0001
    def region(mask):
        return dict(queries=int(mask.sum()), mae_mm=float(absolute[mask].mean()) if mask.any() else None)
    return dict(queries=len(error), mae_mm=float(absolute.mean()), rmse_mm=float(np.sqrt(np.mean(error ** 2))),
                bias_mm=float(error.mean()), p50_mm=float(np.percentile(absolute, 50)),
                p95_mm=float(np.percentile(absolute, 95)), p99_mm=float(np.percentile(absolute, 99)),
                max_mm=float(absolute.max()), near_1mm=region(abs(truth) <= .001),
                near_5mm=region(abs(truth) <= .005), near_10mm=region(abs(truth) <= .01),
                inside=region(inside), outside=region(outside),
                sign_accuracy_away_0p1mm=float(((prediction[away] < 0) == (truth[away] < 0)).mean()) if away.any() else None,
                penetration_false_negative_rate=float((prediction[inside] >= 0).mean()) if inside.any() else None,
                outside_false_positive_rate=float((prediction[outside] < 0).mean()) if outside.any() else None)


@torch.no_grad()
def benchmark(args):
    began = time.monotonic()
    manifest = read_manifest(args.data, verify=True)
    reference = read_manifest(args.reference_data, verify=True)
    model, saved = load_model(args.model, args.device)
    model.requires_grad_(False)
    if saved['dataset_fingerprint'] != reference['fingerprint']:
        raise ValueError('Reference dataset does not match the checkpoint training dataset')
    if fingerprint(saved['geometry']) != fingerprint(manifest['spec']['geometry']):
        raise ValueError('Fresh dataset geometry, coordinates or joint domain differ from the model')
    # Compare against ALL original poses, including original val/test, not just train.
    original = set()
    for record in reference['shards']:
        with np.load(Path(args.reference_data) / record['file'], allow_pickle=False) as a:
            original.update(pose_keys(a['qpos']))
    fresh_keys = set()
    for record in manifest['shards']:
        with np.load(Path(args.data) / record['file'], allow_pickle=False) as a:
            fresh_keys.update(pose_keys(a['qpos']))
    overlap = len(original & fresh_keys)
    if overlap:
        raise ValueError(f'Fresh benchmark overlaps {overlap} original poses')
    if len(fresh_keys) != manifest['retained_poses']:
        raise ValueError('Fresh benchmark has duplicate configurations')
    errors, truths, link_ids, colliding = [], [], [], []
    output = Path(args.output)
    predictions_dir = output.parent / 'predictions'
    predictions_dir.mkdir(parents=True, exist_ok=True)
    prediction_records = []
    pose_mae = []
    for record in manifest['shards']:
        with np.load(Path(args.data) / record['file'], allow_pickle=False) as a:
            q, x, links, truth = (a[k] for k in ('qpos', 'points', 'links', 'sdf'))
            collision, pose_id = a['collision'], a['pose_id']
        k = truth.shape[1]
        prediction = np.empty(truth.size, np.float32)
        for start in range(0, truth.size, args.batch_size):
            ids = np.arange(start, min(start + args.batch_size, truth.size))
            pi, qi = np.divmod(ids, k)
            prediction[ids] = model(torch.as_tensor(q[pi], device=args.device),
                torch.as_tensor(x[pi, qi], device=args.device),
                torch.as_tensor(links[pi, qi].astype(np.int64), device=args.device)).cpu().numpy()
        prediction = prediction.reshape(truth.shape)
        if not np.isfinite(prediction).all():
            raise FloatingPointError('Nonfinite model prediction')
        error = prediction - truth
        errors.append(error.ravel())
        truths.append(truth.ravel())
        link_ids.append(links.ravel())
        colliding.append(np.repeat(collision, k))
        pose_mae.extend((abs(error).mean(1) * 1000).tolist())
        pred_file = predictions_dir / record['file']
        with pred_file.open('wb') as stream:
            np.savez_compressed(stream, pose_id=pose_id, predicted_sdf_m=prediction, error_m=error)
        prediction_records.append(dict(file=str(pred_file.resolve()), sha256=sha256(pred_file),
                                       source_shard_sha256=record['sha256']))
        print(f'Evaluated {sum(len(a) for a in errors):,} / {manifest["queries"]:,} fresh queries', flush=True)
    truth, error, links, collision = map(np.concatenate, (truths, errors, link_ids, colliding))
    prediction = truth + error
    report = distance_metrics(truth, prediction)
    report.update(state='completed', model_epoch=saved['epoch'], checkpoint_sha256=sha256(args.model),
                  model_path=str(Path(args.model).resolve()),
                  training_dataset_fingerprint=reference['fingerprint'],
                  fresh_dataset_fingerprint=manifest['fingerprint'], fresh_seed=manifest['spec']['seed'],
                  poses=manifest['retained_poses'], fresh_queries_per_pose=21 * manifest['spec']['per_link'],
                  original_poses_checked=len(original), overlapping_original_poses=overlap,
                  evaluated_splits='all; the entire fresh dataset is used only for testing',
                  collision_poses=manifest['collision_poses'],
                  pose_mae_p95_mm=float(np.percentile(pose_mae, 95)),
                  collision_pose_queries=distance_metrics(truth[collision], prediction[collision]) if collision.any() else None,
                  free_pose_queries=distance_metrics(truth[~collision], prediction[~collision]) if (~collision).any() else None,
                  per_link=[dict(link=i, name=manifest['spec']['geometry']['link_names'][i],
                                 **distance_metrics(truth[links == i], prediction[links == i])) for i in range(21)],
                  predictions=prediction_records, seconds=time.monotonic() - began,
                  limitations=['Same pose domain and query distribution family as training; not an out-of-domain test',
                               'Point-level sign errors are not whole-pose self-collision detection rates',
                               'Ground truth follows native mesh winding-number sign, including nonwatertight links'])
    atomic_json(output, report)
    return report
