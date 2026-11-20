"""Bounded-memory, deterministic pose shards with collision audits and SDF labels."""
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from retarget.baseline.geort.pose_domain import sample_poses, SAMPLING_POLICY
from .common import atomic_json, fingerprint, sha256


def code_hashes():
    root = Path(__file__).resolve().parents[3]
    files = ['retarget/learning/wuji2_sdf/data.py', 'retarget/learning/wuji2_sdf/geometry.py',
             'retarget/baseline/geort/pose_domain.py', 'retarget/env/grasp/pose_gpu.py',
             'retarget/baseline/card/collision.py', 'moxian_web/wuji_surface.py',
             'retarget/env/grasp/hand.py']
    return {name: sha256(root / name) for name in files}


def read_manifest(folder, verify=False):
    folder = Path(folder)
    m = json.loads((folder / 'manifest.json').read_text())
    if fingerprint(m['spec']) != m['fingerprint']:
        raise ValueError('Dataset manifest specification fingerprint is invalid')
    if m['state'] != 'completed':
        raise ValueError('Sampling must finish before training/evaluation')
    for shard in m['shards']:
        path = folder / shard['file']
        if not path.is_file() or (verify and sha256(path) != shard['sha256']):
            raise ValueError(f'Missing or corrupt shard: {path}')
    return m


def sample(args):
    from .geometry import Geometry
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    if args.poses < 30 or args.per_link < 8 or args.shard_poses < 1 or args.pose_batch < 1:
        raise ValueError('Require poses >= 30, per-link >= 8, positive shard/batch sizes')
    if not np.isfinite(args.collision_threshold_mm) or args.collision_threshold_mm < 0:
        raise ValueError('Collision threshold must be finite and nonnegative')
    geometry = Geometry(args.device)
    spec = dict(version=1, geometry=geometry.spec(), poses=args.poses, per_link=args.per_link,
                shard_poses=args.shard_poses, pose_batch=args.pose_batch, seed=args.seed,
                sampling_policy=SAMPLING_POLICY, collision_threshold_mm=args.collision_threshold_mm,
                pose_filter=args.pose_filter, split_policy='independent_pose_rng_80_10_10',
                query_policy='area_near_1mm_6mm_bbox_global_cross_link_v1', code_sha256=code_hashes())
    key = fingerprint(spec)
    path = folder / 'manifest.json'
    if path.exists():
        if not args.resume:
            raise ValueError('Dataset exists; use --resume or a fresh output')
        manifest = json.loads(path.read_text())
        if manifest['fingerprint'] != key:
            raise ValueError('Sampling specification/assets/code changed; use a fresh output')
        for shard in manifest['shards']:
            if sha256(folder / shard['file']) != shard['sha256']:
                raise ValueError(f'Corrupt committed shard: {shard["file"]}')
        if manifest['state'] == 'completed':
            print(json.dumps(dict(state='already_completed', poses=manifest['retained_poses'])), flush=True)
            return manifest
    else:
        manifest = dict(fingerprint=key, spec=spec, state='sampling', shards=[], proposed_poses=0,
                        retained_poses=0, queries=0, collision_poses=0, split_counts=[0, 0, 0])
        atomic_json(path, manifest)
    start_time = time.monotonic()
    start = manifest['proposed_poses']
    for shard_index, offset in enumerate(range(0, args.poses, args.shard_poses)):
        if offset < start:
            continue
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, shard_index, 0]))
        split_rng = np.random.default_rng(np.random.SeedSequence([args.seed, shard_index, 1]))
        count = min(args.shard_poses, args.poses - offset)
        q = sample_poses(geometry.domain, count, rng)
        split = split_rng.choice(3, count, p=[.8, .1, .1]).astype(np.uint8)
        fields = {k: [] for k in ('qpos', 'pose_id', 'split', 'points', 'links', 'sdf',
                                  'max_penetration_mm', 'collision')}
        proposed_collisions = 0
        for i in range(0, count, args.pose_batch):
            values = q[i:i + args.pose_batch]
            frames = geometry.frames(values)
            with torch.inference_mode():
                depth, valid = geometry.collision.query(*frames, dense=True)
                if not bool(valid.all()):
                    raise ValueError('Invalid collision query; shard not committed')
                depth = depth.amax((1, 2)).cpu().numpy()
            colliding = depth > args.collision_threshold_mm
            proposed_collisions += int(colliding.sum())
            keep = ~colliding if args.pose_filter == 'free' else np.ones(len(values), bool)
            if not keep.any():
                continue
            frames = tuple(x[torch.as_tensor(keep, device=x.device)] for x in frames)
            points, links = geometry.queries(frames, args.per_link, rng)
            labels = geometry.label(points, links, frames)
            batch = dict(qpos=values[keep], pose_id=np.arange(offset + i, offset + i + len(values))[keep],
                         split=split[i:i + len(values)][keep], points=points,
                         links=links.astype(np.uint8), sdf=labels,
                         max_penetration_mm=depth[keep], collision=colliding[keep])
            if not all(np.isfinite(x).all() for x in batch.values()):
                raise ValueError('Nonfinite sample; shard not committed')
            for name, value in batch.items():
                fields[name].append(value)
        # A free-pose shard can be empty. Record its proposal budget without a data file.
        arrays = {k: np.concatenate(v) for k, v in fields.items()} if fields['qpos'] else None
        record = dict(proposed=count, proposed_collisions=proposed_collisions, retained=0, queries=0,
                      split_counts=[0, 0, 0])
        if arrays is not None:
            name = f'shard_{shard_index:06d}.npz'
            temp = folder / (name + '.tmp')
            with temp.open('wb') as stream:
                np.savez_compressed(stream, **arrays)
            os.replace(temp, folder / name)
            record.update(file=name, sha256=sha256(folder / name), retained=len(arrays['qpos']),
                          queries=arrays['sdf'].size, split_counts=np.bincount(arrays['split'], minlength=3).tolist())
            manifest['shards'].append(record)
            manifest['retained_poses'] += len(arrays['qpos'])
            manifest['queries'] += arrays['sdf'].size
            manifest['split_counts'] = (np.array(manifest['split_counts']) + record['split_counts']).tolist()
        manifest['proposed_poses'] = offset + count
        manifest['collision_poses'] += proposed_collisions
        manifest['elapsed_seconds_last_session'] = time.monotonic() - start_time
        atomic_json(path, manifest)
        print(json.dumps(dict(phase='sampling', proposed=manifest['proposed_poses'], total=args.poses,
                              retained=manifest['retained_poses'], queries=manifest['queries'],
                              collision_poses=manifest['collision_poses'], seconds=time.monotonic() - start_time)), flush=True)
    if min(manifest['split_counts']) == 0:
        raise ValueError('A pose split is empty; increase sampling budget or change seed/output')
    manifest['state'] = 'completed'
    atomic_json(path, manifest)
    return manifest


def batches(folder, manifest, split, batch_size, rng=None, max_queries=0):
    """One CPU shard at a time; q is stored once per pose, never repeated on disk."""
    records = list(manifest['shards'])
    if rng is not None:
        rng.shuffle(records)
    emitted = 0
    for record in records:
        with np.load(Path(folder) / record['file'], allow_pickle=False) as a:
            poses = np.flatnonzero(a['split'] == split)
            if not len(poses):
                continue
            points, links, sdf = (a[k][poses] for k in ('points', 'links', 'sdf'))
            q = a['qpos'][poses]
            collision = a['collision'][poses]
        k = sdf.shape[1]
        order = np.arange(sdf.size)
        if rng is not None:
            rng.shuffle(order)
        for i in range(0, len(order), batch_size):
            ids = order[i:i + batch_size]
            if max_queries:
                ids = ids[:max_queries - emitted]
            if not len(ids):
                return
            pi, qi = np.divmod(ids, k)
            yield dict(q=q[pi], points=points[pi, qi], links=links[pi, qi].astype(np.int64),
                       sdf=sdf[pi, qi], collision=collision[pi])
            emitted += len(ids)
            if max_queries and emitted >= max_queries:
                return
