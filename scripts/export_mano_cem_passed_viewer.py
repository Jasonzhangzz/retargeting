#!/usr/bin/env python3
"""Freeze per-type receipt counts and a stratified sample of passing CEM poses."""
import argparse
import collections
import hashlib
import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retarget.env.grasp.mano_solver import ManoDecoder
from scripts.mano_campaign_inputs import GeometryLoader, read, retry_io, sha, stamp, write


def export(campaign, model, dataset, output, samples=100, seed=20261106):
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    load = retry_io(read)
    read_bytes = retry_io(Path.read_bytes)
    roots, current = [], campaign.resolve()
    while current:
        if current in roots:
            raise ValueError('Cyclic campaign continuation')
        roots.append(current)
        previous = load(current/'configuration.json').get('resume_from')
        current = Path(previous).resolve() if previous else None
    paths = [p for root in roots for p in sorted(root.glob('*/*/*/verification.json'))]
    by_type = collections.defaultdict(list)
    identities = set()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for path, receipt in zip(paths, pool.map(load, paths)):
            if not receipt.get('success'):
                continue
            identity = '/'.join(path.parent.parts[-3:])
            if identity in identities:
                raise ValueError('Duplicate source batch: '+identity)
            identities.add(identity)
            by_type[path.parents[2].name].append((path, receipt))
    order = sorted(by_type, key=lambda name: int(name.split('_')[0]))
    counts = [dict(type=name, samples=sum(v['samples'] for _, v in by_type[name]),
                   stable=sum(v['stable'] for _, v in by_type[name]),
                   joint=sum(v['geometry_accepted'] for _, v in by_type[name])) for name in order]
    summary = dict(created_utc=stamp(), campaign=str(campaign), roots=list(map(str, roots)),
                   completed_batches=len(identities), by_type=counts,
                   totals={key: sum(row[key] for row in counts) for key in ('samples', 'stable', 'joint')},
                   status_at_scan_end=load(campaign/'status.json'),
                   denominator='All successful immutable batch receipts visible at scan start, including inherited batches.',
                   stable_criterion='Current five-step terminal velocity and complete stability checks.',
                   joint_criterion='patch_joint_accepted: stability, patch coverage, required fingers and penetration checks.')
    write(output/'counts.json', summary)
    print('COUNTS '+json.dumps(summary, ensure_ascii=False), flush=True)
    rng = np.random.default_rng(seed)
    quotas = {r['type']: 0 for r in counts}
    capacities = {r['type']: r['joint'] for r in counts}
    if sum(capacities.values()) < samples:
        raise ValueError('Insufficient joint-passing rows')
    while sum(quotas.values()) < samples:
        for name in rng.permutation(order):
            if quotas[name] < capacities[name]:
                quotas[name] += 1
            if sum(quotas.values()) == samples:
                break
    selected = []
    for name in order:
        ranks = np.sort(rng.choice(capacities[name], quotas[name], replace=False))
        cursor = 0
        for path, receipt in by_type[name]:
            n = receipt['geometry_accepted']
            local = ranks[(ranks >= cursor) & (ranks < cursor+n)]-cursor
            if len(local):
                selected.append((name, path, receipt, local))
            cursor += n
    decoder = ManoDecoder(model, device='cpu')
    model_sha = sha(model)
    loader = GeometryLoader(dataset)
    cases, arrays = [], {'hand_faces': np.asarray(decoder.base.faces, np.int32)}
    for name, path, receipt, ranks in selected:
        batch = path.parent
        result = batch/next(k for k in receipt['files'] if k.endswith('/final.npz'))
        model_path, metrics_path = result.parent/'model.json', result.parent/'metrics.json'
        model_info, metrics = load(model_path), load(metrics_path)
        if model_info['model_sha256'] != model_sha:
            raise ValueError('Result/model fingerprint mismatch')
        raw = read_bytes(result)
        digest = hashlib.sha256(raw).hexdigest()
        if digest != receipt['files'][str(result.relative_to(batch))]:
            raise ValueError('Changed result: '+str(result))
        for p in (model_path, metrics_path, batch/'MANIFEST.json'):
            if hashlib.sha256(read_bytes(p)).hexdigest() != receipt['files'][str(p.relative_to(batch))]:
                raise ValueError('Changed metadata: '+str(p))
        manifest = load(batch/'MANIFEST.json')
        passing = np.flatnonzero([m['after']['geometry_accepted'] for m in metrics])
        if len(passing) != receipt['geometry_accepted'] or len(metrics) != receipt['samples']:
            raise ValueError('Receipt/metrics count mismatch')
        rows = passing[ranks]
        with np.load(io.BytesIO(raw), allow_pickle=False) as data:
            with torch.no_grad():
                vertices, joints, _ = decoder(torch.tensor(data['variables'][rows]), full=True)
            vertices, joints = vertices.numpy(), joints.numpy()
            np.testing.assert_allclose(joints, data['mano_joints'][rows], atol=4e-7, rtol=1e-5)
            for k, row in enumerate(rows):
                row = int(row)
                entry = manifest['source']['cases'][row]
                _, geometry = loader.get(entry)
                pose = data['object_pose_world'][row]
                rotation = Rotation.from_quat(pose[3:], scalar_first=True).as_matrix()
                obj = geometry['vertices']*float(data['scene_scale'][row]) @ rotation.T+pose[:3]
                index = len(cases)
                prefix = str(index)+'_'
                for key, value in dict(hand_vertices=vertices[k], joints=joints[k],
                                       object_vertices=obj.astype(np.float32), object_faces=geometry['faces']).items():
                    arrays[prefix+key] = value
                for key in ('patch_hand_points', 'patch_object_points', 'object_template_points',
                            'patch_fitted', 'patch_gap_m', 'patch_anchor_offset_m',
                            'object_pose_world', 'object_pose_after_rollout'):
                    arrays[prefix+key] = data[key][row]
                center = (obj.max(0)+obj.min(0))*.5
                radius = max(float(np.linalg.norm(np.concatenate((obj, vertices[k]))-center, axis=1).max()), .08)
                cases.append(dict(index=index, type=name, row=row, selection='按抓型分层随机抽取的联合通过样本',
                                  result=str(result), result_sha256=digest, scene=entry['scene_path'],
                                  object_sha256=geometry['sha256'], metrics=metrics[row]['after'],
                                  fingers=metrics[row]['after']['fingers'], center=center.tolist(), radius=radius,
                                  source=entry, model=model_info))
        print(f'EXPORTED {len(cases)}/{samples} · {name}', flush=True)
    if len(cases) != samples or len({(r['result'], r['row']) for r in cases}) != samples:
        raise ValueError('Sample count or uniqueness mismatch')
    np.savez_compressed(output/'geometry.npz', **arrays)
    write(output/'manifest.json', dict(created_utc=stamp(), campaign=str(campaign),
          model_sha256=model_sha, appearance=decoder.base.appearance, cases=cases,
          campaign_snapshot=summary, sampling_seed=seed, per_type_samples=quotas,
          sampling=f'仅联合通过样本；按抓型均匀分层随机抽取 {samples} 条，不重复。此展示集不代表总体通过率。',
          geometry_sha256=sha(output/'geometry.npz')))
    print('COMPLETE '+json.dumps(dict(samples=len(cases), types=len(set(r['type'] for r in cases)),
                                     bytes=(output/'geometry.npz').stat().st_size)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('campaign', 'model', 'dataset', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--samples', type=int, default=100)
    p.add_argument('--seed', type=int, default=20261106)
    a = p.parse_args()
    if a.samples < 1:
        p.error('--samples must be positive')
    export(a.campaign, a.model, a.dataset, a.output, a.samples, a.seed)
