"""Resumable multi-GPU Shadow -> MANO patch CEM / tangent-history rollout, balanced by grasp type."""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MUJOCO_GL', 'egl')
import argparse, collections, fcntl, hashlib, json, sys
from pathlib import Path
from types import SimpleNamespace
from dataclasses import asdict
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retarget.solver.cem import DEFAULTS
from scripts.grasp_campaign import run_campaign, parse_devices
from scripts.mano_campaign_inputs import source_groups, GeometryLoader, sha, write, read, stamp, retry_io
read, write, sha = retry_io(read), retry_io(write), retry_io(sha)

def make_jobs(groups, batch_size, completed=()):
    """Interleave source row batches across types, without resampling or repeats."""
    covered = collections.defaultdict(list)
    sources = {(g['type'], int(g['part'])): g for g in groups}
    for job in completed:
        identity = (job['type'], int(job['part']))
        source = sources[identity]
        start, end = int(job['offset']), int(job['offset']) + int(job['samples'])
        if job['path'] != source['path'] or not 0 <= start < end <= int(source['samples']):
            raise ValueError('Completed source interval mismatch')
        covered[identity].append((start, end))
    for intervals in covered.values():
        intervals.sort()
        if any(b[0] < a[1] for a, b in zip(intervals, intervals[1:])):
            raise ValueError('Overlapping completed source intervals')
    pending = collections.defaultdict(list)
    for group in sorted(groups, key=lambda g: (g['type'], int(g['part']))):
        n = int(group['samples'])
        cursor = 0
        for start, end in [*covered[(group['type'], int(group['part']))], (n, n)]:
            for offset in range(cursor, start, batch_size):
                pending[group['type']].append(dict(type=group['type'], part=int(group['part']), path=group['path'], offset=offset, samples=min(batch_size, start - offset)))
            cursor = end
    order = sorted(pending, key=lambda t: (int(t.split('_')[0]), t))
    for t in order:
        if completed:
            # Finish partial inherited shards after full batches, retaining every row.
            pending[t].sort(key=lambda job: job['samples'] != batch_size)
        pending[t] = collections.deque(pending[t])
    while pending:
        for typ in order:
            if typ not in pending:
                continue
            yield pending[typ].popleft()
            if not pending[typ]:
                del pending[typ]

def key(job):
    return f"{job['type']}/{job['part']:07d}/{job['offset']:07d}"


def inherited_results(root, config, seen=None):
    """Reuse immutable receipts only after the previous controller has stopped."""
    config = json.loads(json.dumps(config))
    root = Path(root).resolve()
    seen = set() if seen is None else seen
    if root in seen or root == Path(config['output']).resolve():
        raise ValueError('Cyclic campaign continuation')
    seen.add(root)
    lock = (root/'campaign.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if read(root/'status.json')['state'] not in ('stopped', 'complete', 'finished_with_errors'):
            raise ValueError('Previous campaign is not stopped')
        prior = read(root/'configuration.json')
        for name in ('source', 'model', 'mapping', 'shadow_xml', 'dataset', 'seed', 'rollout', 'model_sha256', 'mapping_sha256'):
            if prior[name] != config[name]:
                raise ValueError('Continuation changes '+name)
        for name in set(prior['algorithm_parameters']) | set(config['algorithm_parameters']):
            if name != 'resources' and prior['algorithm_parameters'].get(name) != config['algorithm_parameters'].get(name):
                raise ValueError('Continuation changes solver parameters: '+name)
        for name in set(prior['code_sha256']) | set(config['code_sha256']):
            if name not in ('scripts/retarget_mano_comfree_campaign.py', 'scripts/grasp_campaign.py') and prior['code_sha256'].get(name) != config['code_sha256'].get(name):
                raise ValueError('Continuation changes scoring code: '+name)
        inherited = inherited_results(prior['resume_from'], config, seen) if prior.get('resume_from') else {}
        for receipt in sorted(root.glob('*/*/*/verification.json')):
            verification = read(receipt)
            if not verification.get('success'):
                continue
            manifest = receipt.parent/'MANIFEST.json'
            if sha(manifest) != verification['files']['MANIFEST.json']:
                raise ValueError('Changed result manifest: '+str(manifest))
            job = read(manifest)['job']
            name = key(job)
            if verification['config_sha256'] != prior['config_sha256'] or job['samples'] != verification['samples']:
                raise ValueError('Result/configuration identity mismatch')
            if name in inherited:
                raise ValueError('Duplicate inherited result: '+name)
            for rel, fingerprint in verification['files'].items():
                path = (receipt.parent/rel).resolve()
                if not path.is_relative_to(receipt.parent.resolve()) or sha(path) != fingerprint:
                    raise ValueError('Changed completed result: '+str(path))
            inherited[name] = dict(event='reused', key=name, job=job, origin_campaign=str(root), **verification)
        return inherited
    finally:
        lock.close()

def prepare_packet(job, loader, cache):
    import numpy as np
    p = Path(job['path'])
    if p not in cache:
        cache[p] = (read(p / 'manifest.json'), dict(np.load(p / 'poses.npz')), sha(p / 'poses.npz'))
        if len(cache) > 8:
            cache.popitem(last=False)
    original, source, fingerprint = cache[p]
    cache.move_to_end(p)
    start = job['offset']
    end = start + job['samples']
    entries = original['cases'][start:end]
    geo = []
    ids = []
    seen = {}
    for e in entries:
        asset, g = loader.get(e)
        if not g['watertight']:
            raise ValueError('Object mesh is not watertight: ' + g['source_mesh'])
        if asset not in seen:
            seen[asset] = len(geo)
            geo.append(g)
        ids.append(seen[asset])
    pose = source['object_pose_world'][start:end]
    if pose.ndim == 3:
        pose = pose[:, 0]
    a = dict(shadow_qpos=source['grasp_qpos'][start:end], object_pose_world=pose, object_scale=source['object_scale'][start:end], scene_scale=source['scene_scale'][start:end], object_geometry_ids=np.array(ids))
    for i, g in enumerate(geo):
        a[f'object_{i}_vertices'] = g['vertices']
        a[f'object_{i}_faces'] = g['faces']
    return (a, list(range(start, end)), dict(source_manifest=str(p / 'manifest.json'), source_poses_sha256=fingerprint, cases=entries, objects=[{k: v for k, v in g.items() if k not in ('vertices', 'faces')} for g in geo]))

class ManoCampaignWorker:
    """MANO model/input adapter for the shared campaign worker loop."""
    def __init__(self, device, config, progress):
        import torch
        from retarget.env.grasp.mano_solver import ManoDecoder, ShadowSkeleton, MeshCache
        from scripts.solve_mano_comfree import run_packets
        self.torch, self.run_packets = torch, run_packets
        self.config, self.output = config, Path(config['output'])
        torch.set_num_threads(1)
        torch.cuda.set_device(device)
        self.mapping = read(config['mapping'])
        self.decoder = ManoDecoder(config['model'], device)
        self.decoder.configure_contact_vertices(self.mapping['mappings'])
        self.skeleton = ShadowSkeleton(config['shadow_xml'], device)
        self.mesh, self.loader = MeshCache(device), GeometryLoader(config['dataset'])
        self.source_cache = collections.OrderedDict()
        self.args = SimpleNamespace(device=device, initialization='shadow-ik', batch_size=config['batch_size'],
            candidate_chunk=config['candidate_chunk'],
            score_only=False, seed=config['seed'])

    def completed(self, job):
        folder = self.output / key(job)
        if not (folder/'verification.json').exists():
            return None
        v = read(folder/'verification.json')
        if not v.get('success') or v['samples'] != job['samples'] or v['config_sha256'] != self.config['config_sha256']:
            raise ValueError('Invalid completed MANO batch')
        if read(folder/'MANIFEST.json')['job'] != job:
            raise ValueError('Completed MANO source rows changed')
        for rel, digest in v['files'].items():
            path = (folder/rel).resolve()
            if not path.is_relative_to(folder.resolve()) or sha(path) != digest:
                raise ValueError('Changed completed MANO output')
        return v

    def solve(self, job, progress):
        folder = self.output / key(job)
        a, rows, provenance = prepare_packet(job, self.loader, self.source_cache)
        folder.mkdir(parents=True, exist_ok=True)
        write(folder/'MANIFEST.json', dict(config_sha256=self.config['config_sha256'], source=provenance, job=job))
        self.args.seed = (self.config['seed'] + int(hashlib.sha256(key(job).encode()).hexdigest()[:8], 16)) % 2 ** 31
        progress('solving')
        report = self.run_packets(self.args, folder, self.decoder, self.mapping, self.skeleton,
                                 self.mesh, [(job['type'], a, rows, provenance)])[0]
        verified = dict(success=True, samples=report['samples'], stable=report['stable_after'],
            geometry_accepted=report['geometry_after'], solve_seconds=report['solve_seconds'],
            config_sha256=self.config['config_sha256'], source_poses_sha256=provenance['source_poses_sha256'],
            created_utc=stamp(), files={str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file()})
        write(folder/'verification.json', verified)
        return verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['source', 'output', 'model', 'mapping', 'shadow-xml', 'dataset']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--devices', default='cuda:0,cuda:1,cuda:2,cuda:3')
    parser.add_argument('--batch-size', type=int, default=DEFAULTS.resources.batch_size)
    parser.add_argument('--candidate-chunk', type=int, default=DEFAULTS.resources.candidate_chunk)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--max-batches', type=int, default=0, help='0 means the complete frozen source')
    parser.add_argument('--resume-from', help='Stopped campaign whose verified source rows must be retained')
    args = parser.parse_args()
    from retarget.env.grasp.comfree_rollout import RolloutSettings
    if min(args.batch_size, args.candidate_chunk) < 1 or args.max_batches < 0:
        parser.error('Invalid batch budget')
    try:
        devices = parse_devices(args.devices)
    except ValueError as error:
        parser.error(str(error))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / 'campaign.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (output / 'STOP').exists():
        raise RuntimeError('STOP present: explicit new launch/output required')
    config = vars(args).copy()
    config['rollout'] = asdict(DEFAULTS.rollout)
    config['algorithm_parameters'] = asdict(DEFAULTS)
    config['algorithm_parameters']['resources'].update(batch_size=args.batch_size, candidate_chunk=args.candidate_chunk)
    config['model_sha256'] = sha(args.model)
    config['mapping_sha256'] = sha(args.mapping)
    mapping = read(args.mapping)
    if mapping['model_sha256'] != config['model_sha256']:
        raise ValueError('Model/mapping identity mismatch')
    config['code_sha256'] = {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), ROOT / 'scripts/solve_mano_comfree.py', *[ROOT / 'retarget/env/grasp' / f for f in ['mano_solver.py', 'mano_comfree.py', 'mano_contact_collision.py', 'mano_collision_geometry.py', 'contact_geometry.py', 'comfree_rollout.py']], ROOT / 'scripts/mano_campaign_inputs.py', ROOT / 'retarget/solver/base.py']}
    config['code_sha256'].update({f: sha(ROOT / f) for f in ['retarget/env/grasp/mano_patch_anchor.py', 'retarget/solver/cem.py', 'retarget/env/grasp/mano_patch_consistent.py', 'retarget/env/grasp/spring_history.py', 'retarget/env/grasp/spring_history_warp.py', 'retarget/env/grasp/patch_query.py', 'retarget/env/grasp/triangle_penetration.py']})
    config['code_sha256']['scripts/grasp_campaign.py'] = sha(ROOT/'scripts/grasp_campaign.py')
    config['code_sha256']['scripts/calibrate_wuji_mano.py'] = sha(ROOT / 'scripts/calibrate_wuji_mano.py')
    config = json.loads(json.dumps(config))
    config['config_sha256'] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    cfg = output / 'configuration.json'
    if cfg.exists() and read(cfg) != config:
        raise ValueError('Existing campaign/configuration mismatch')
    write(cfg, config)
    inherited = inherited_results(args.resume_from, config) if args.resume_from else {}
    groups = source_groups(args.source)
    jobs = list(make_jobs(groups, args.batch_size, [v['job'] for v in inherited.values()]))
    if args.max_batches:
        jobs = jobs[:args.max_batches]
    write(output / 'inherited_results.json', inherited)
    write(output / 'schedule.json', dict(total_batches=len(jobs)+len(inherited), total_samples=sum((j['samples'] for j in jobs))+sum(v['samples'] for v in inherited.values()), remaining_batches=len(jobs), inherited_samples=sum(v['samples'] for v in inherited.values()), first_round=jobs[:len(set((j['type'] for j in jobs)))], source_frozen=True))
    run_campaign(config=config, jobs=jobs, devices=devices,
                 backend='scripts.retarget_mano_comfree_campaign:ManoCampaignWorker',
                 job_key=key, completed=inherited)
if __name__ == '__main__':
    main()
