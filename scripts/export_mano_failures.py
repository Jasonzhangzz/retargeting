#!/usr/bin/env python
"""Stratify verified production results and export a small CPU-only viewer bundle."""
from __future__ import annotations

import argparse
import collections
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from retarget.viewer.mano_results import read, retry, snapshot, write, sha
from retarget.viewer.mano_failures import CATEGORIES, category, slot_diagnostics


def export(root, out, per_category=4, seed=20261104):
    root, out = Path(root).resolve(), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    shards = [p.parent for p in sorted(root.glob('*/shards/*/verification.json'))
              if read(p).get('success')]
    counts = collections.Counter({key: 0 for key in CATEGORIES})
    selected = {key: [] for key in CATEGORIES}
    source_shards = []
    extra = collections.Counter()
    diagnostics = collections.Counter()
    coverage_histogram = collections.Counter()
    slot_histogram = collections.Counter()
    by_type = {}
    for path in shards:
        raw = retry((path / 'metrics.json.gz').read_bytes)
        metrics = json.loads(gzip.decompress(raw))
        verified = read(path / 'verification.json')
        if len(metrics) != verified['samples']:
            raise ValueError(f'Metrics length differs from verified shard: {path}')
        source_shards.append(dict(path=str(path), samples=len(metrics),
                                  metrics_sha256=hashlib.sha256(raw).hexdigest()))
        typ = path.parents[1].name
        type_counts = by_type.setdefault(typ, collections.Counter())
        for i, m in enumerate(metrics):
            d = m['geometry_diagnostics']
            group = category(m['projected_closure'], d['mapped_coverage'])
            coverage_ok = d['mapped_coverage'] + 1.e-7 >= .8
            distance_ok = d['mapped_distance_coverage'] + 1.e-7 >= .8
            slot_histogram[str(d['mapped_contact_count'])] += 1
            coverage_histogram[str(round(d['mapped_coverage'], 6))] += 1
            diagnostics['distance_coverage_pass'] += int(distance_ok)
            diagnostics['distance_pass_but_joint_coverage_fail'] += int(distance_ok and not coverage_ok)
            diagnostics['closure_pass_distance_coverage_fail'] += int(m['projected_closure'] and not distance_ok)
            diagnostics['closure_pass_distance_pass_joint_coverage_fail'] += int(m['projected_closure'] and distance_ok and not coverage_ok)
            counts[group] += 1
            type_counts[group] += 1
            extra['geometry_accepted'] += int(m['geometry_accepted'])
            extra['self_penetration_failed'] += int(d['max_hand_hand_penetration_m'] > 1.e-5)
            extra['closure_pass_self_penetration_failed'] += int(m['projected_closure'] and d['max_hand_hand_penetration_m'] > 1.e-5)
            sample = dict(path=str(path), row=i, group=group)
            bucket = selected[group]
            if len(bucket) < per_category:
                bucket.append(sample)
            else:
                j = rng.randrange(counts[group])
                if j < per_category:
                    bucket[j] = sample
        print(f"SCANNED {len(source_shards)}/{len(shards)} {sum(counts.values())}", flush=True)
    entries = []
    for group, samples in selected.items():
        for sample in samples:
            s = snapshot(sample['path'])
            i = sample['row']
            metric, p = s.metrics[i], s.poses
            d = metric['geometry_diagnostics']
            v, f = s.mano_mesh(i)
            ov, of = s.object_mesh(i)
            contacts = s.contacts(i)
            info = slot_diagnostics(p['mapped_distances_m'][i], p['mapped_hand_normals'][i], p['mapped_object_normals'][i])
            coverage = float(info['fitted'].mean())
            if abs(coverage - d['mapped_coverage']) > 1.e-6:
                raise ValueError(f"Saved and reconstructed coverage differ: {sample}")
            if category(metric['projected_closure'], coverage) != group:
                raise ValueError('Stratification changed after reconstruction')
            reconstructed = np.einsum('ni,nij->nj', p['mapped_mano_barycentric'], v[f[p['mapped_mano_face_ids']]])
            np.testing.assert_allclose(reconstructed, p['mapped_hand_points'][i], atol=2.e-6, rtol=1.e-5)
            name = f'case_{len(entries):02d}.npz'
            np.savez_compressed(out / name, hand_vertices=v, hand_faces=f,
                object_vertices=ov, object_faces=of, joints=p['mano_joints'][i],
                hand_points=p['mapped_hand_points'][i], object_points=p['mapped_object_points'][i],
                hand_normals=p['mapped_hand_normals'][i], object_normals=p['mapped_object_normals'][i],
                gap_m=p['mapped_distances_m'][i], near_points=contacts['hand_points_world'], **info)
            entries.append(dict(id=len(entries), arrays=name, category=group,
                grasp_type=s.manifest['cases'][i]['grasp_type'], shard=s.path.name, row=i,
                source_path=str(s.path), source_poses_sha256=sha(s.path/'poses.npz'),
                arrays_sha256=sha(out/name), projected_closure=bool(metric['projected_closure']),
                geometry_accepted=bool(metric['geometry_accepted']), coverage=coverage,
                object_penetration_m=d['max_hand_object_penetration_m'],
                self_penetration_m=d['max_hand_hand_penetration_m'],
                closure_margin=metric['closure_margin'], closure_cost=metric['closure_cost']))
            print(f"EXPORTED {name} {group}", flush=True)
    total = sum(counts.values())
    report = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_root=str(root), seed=seed, samples=total, counts=dict(counts),
        by_type={k:dict(v) for k,v in by_type.items()}, checks=dict(extra),
        thresholds=dict(gap_min_m=-.002, gap_max_m=.002, normal_max_deg=30., coverage_min=.8,
                        self_penetration_max_m=1.e-5),
        optimizer_updates=0, source_modified=False, classification='saved projected QP certificate x fixed-slot coverage',
        sources=source_shards, cases=entries)
    write(out/'manifest.json', report)
    write(out/'diagnostic_breakdown.json', dict(samples=total,
        fixed_slot_count_histogram=dict(slot_histogram), joint_coverage_histogram=dict(coverage_histogram),
        counts=dict(diagnostics), created_utc=report['created_utc']))
    print(json.dumps({k:report[k] for k in ('samples','counts','checks','by_type')},ensure_ascii=False),flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--per-category', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20261104)
    args = parser.parse_args()
    export(args.results, args.out, args.per_category, args.seed)
