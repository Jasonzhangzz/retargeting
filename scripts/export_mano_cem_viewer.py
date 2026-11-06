#!/usr/bin/env python3
"""Export a bounded, reproducible snapshot of completed MANO CEM results."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retarget.env.grasp.mano_solver import ManoDecoder
from scripts.mano_campaign_inputs import GeometryLoader, read, sha, stamp, write


def export(campaign, model, dataset, output, max_types):
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    decoder = ManoDecoder(model, device='cpu')
    loader = GeometryLoader(dataset)
    rng = np.random.default_rng(20261105)
    cases, arrays = [], {'hand_faces': np.asarray(decoder.base.faces, np.int32)}
    batches = sorted(campaign.glob('*/*/*/verification.json'),
                     key=lambda p: (int(p.parents[2].name.split('_')[0]), str(p)))
    seen = set()
    for receipt in batches:
        typ = receipt.parents[2].name
        if typ in seen or len(seen) >= max_types:
            continue
        verification = read(receipt)
        if not verification.get('success'):
            continue
        batch = receipt.parent
        result = next(batch.rglob('final.npz'))
        model_info = read(result.parent/'model.json')
        if model_info['model_sha256'] != sha(model):
            raise ValueError('Result/model fingerprint mismatch')
        for path in (result, result.parent/'metrics.json', result.parent/'model.json'):
            expected = verification['files'][str(path.relative_to(batch))]
            if sha(path) != expected:
                raise ValueError(f'Incomplete or changed result: {path}')
        manifest = read(batch/'MANIFEST.json')
        metrics = read(result.parent/'metrics.json')
        flags = np.array([m['after']['geometry_accepted'] for m in metrics])
        selected = [(int(rng.integers(len(metrics))), '固定种子随机抽样')]
        for passed, label in ((True, '联合通过样本'), (False, '联合未通过样本')):
            pool = np.flatnonzero(flags == passed)
            pool = pool[~np.isin(pool, [i for i, _ in selected])]
            if len(pool):
                selected.append((int(rng.choice(pool)), label))
        with np.load(result, allow_pickle=False) as data:
            ids = [i for i, _ in selected]
            with torch.no_grad():
                vertices, joints, _ = decoder(torch.tensor(data['variables'][ids]), full=True)
            np.testing.assert_allclose(joints.numpy(), data['mano_joints'][ids], atol=4e-7, rtol=1e-5)
            for k, (row, selection) in enumerate(selected):
                index = len(cases)
                entry = manifest['source']['cases'][row]
                _, geometry = loader.get(entry)
                pose = data['object_pose_world'][row]
                rotation = Rotation.from_quat(pose[3:], scalar_first=True).as_matrix()
                obj = geometry['vertices'] * float(data['scene_scale'][row])
                obj = obj @ rotation.T + pose[:3]
                prefix = str(index)+'_'
                arrays[prefix+'hand_vertices'] = vertices[k].numpy()
                arrays[prefix+'joints'] = joints[k].numpy()
                arrays[prefix+'object_vertices'] = obj.astype(np.float32)
                arrays[prefix+'object_faces'] = geometry['faces']
                for key in ('patch_hand_points', 'patch_object_points', 'object_template_points',
                            'patch_fitted', 'patch_gap_m', 'patch_anchor_offset_m',
                            'object_pose_world', 'object_pose_after_rollout'):
                    arrays[prefix+key] = data[key][row]
                center = (obj.max(0)+obj.min(0))*.5
                combined = np.concatenate((obj, vertices[k].numpy()))
                radius = max(float(np.linalg.norm(combined-center, axis=1).max()), .08)
                cases.append(dict(index=index, type=typ, row=row, selection=selection,
                    result=str(result), result_sha256=sha(result), scene=entry['scene_path'],
                    object_sha256=geometry['sha256'], metrics=metrics[row]['after'],
                    fingers=metrics[row]['after']['fingers'], center=center.tolist(), radius=radius,
                    source=entry, model=model_info))
        seen.add(typ)
        print(f'EXPORTED {typ}: {len(selected)}', flush=True)
    if not cases:
        raise ValueError('No verified completed batches')
    np.savez_compressed(output/'geometry.npz', **arrays)
    write(output/'manifest.json', dict(created_utc=stamp(), campaign=str(campaign),
        model_sha256=sha(model), appearance=decoder.base.appearance, cases=cases,
        campaign_snapshot=read(campaign/'status.json'),
        sampling='Each type: one seeded random row, plus passing/failing rows where available; not a population estimate.',
        geometry_sha256=sha(output/'geometry.npz')))
    print(json.dumps(dict(samples=len(cases), types=len(seen), bytes=(output/'geometry.npz').stat().st_size)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('campaign', 'model', 'dataset', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--max-types', type=int, default=16)
    a = p.parse_args()
    export(a.campaign, a.model, a.dataset, a.output, a.max_types)
