"""Resumable multi-GPU Shadow -> MANO patch CEM / tangent-history rollout, balanced by grasp type."""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MUJOCO_GL', 'egl')
import argparse, collections, fcntl, hashlib, json, multiprocessing as mp, queue, signal, sys, time, traceback
from pathlib import Path
from types import SimpleNamespace
from dataclasses import asdict
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retarget.solver.cem import DEFAULTS
from scripts.mano_campaign_inputs import source_groups, round_robin_groups, GeometryLoader, sha, write, read, stamp

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
            if name != 'scripts/retarget_mano_comfree_campaign.py' and prior['code_sha256'].get(name) != config['code_sha256'].get(name):
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

def worker(number, device, config, tasks, events):
    import torch
    from retarget.env.grasp.mano_solver import ManoDecoder, ShadowSkeleton, MeshCache
    from retarget.env.grasp.comfree_rollout import RolloutSettings
    from scripts.solve_mano_comfree import run_packets
    output = Path(config['output'])
    torch.set_num_threads(1)
    torch.cuda.set_device(device)
    write(output / f'worker_{number}.json', dict(pid=os.getpid(), device=device, stage='loading', updated_utc=stamp()))
    mapping = read(config['mapping'])
    decoder = ManoDecoder(config['model'], device)
    write(output / f'worker_{number}_model.json', dict(model_sha256=sha(config['model']), appearance=decoder.base.appearance, asset_metadata=decoder.base.asset_metadata, device=device))
    decoder.configure_contact_vertices(mapping['mappings'])
    skeleton = ShadowSkeleton(config['shadow_xml'], device)
    mesh = MeshCache(device)
    loader = GeometryLoader(config['dataset'])
    source_cache = collections.OrderedDict()
    args = SimpleNamespace(device=device, initialization='shadow-ik', batch_size=config['batch_size'], candidate_chunk=config['candidate_chunk'], benchmark_candidates=0, benchmark_repeats=0, score_only=False, seed=config['seed'])
    while not (output / 'STOP').exists():
        try:
            job = tasks.get(timeout=2)
        except queue.Empty:
            continue
        if job is None:
            break
        name = key(job)
        folder = output / name
        if (folder / 'verification.json').exists():
            events.put(dict(event='reused', key=name, **read(folder / 'verification.json')))
            continue
        write(output / f'worker_{number}.json', dict(pid=os.getpid(), device=device, stage='solving', job=name, updated_utc=stamp()))
        try:
            a, rows, provenance = prepare_packet(job, loader, source_cache)
            folder.mkdir(parents=True, exist_ok=True)
            write(folder / 'MANIFEST.json', dict(config_sha256=config['config_sha256'], source=provenance, job=job))
            args.seed = (config['seed'] + int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)) % 2 ** 31
            reports = run_packets(args, folder, decoder, mapping, skeleton, mesh, [(job['type'], a, rows, provenance)])
            report = reports[0]
            verified = dict(success=True, samples=report['samples'], stable=report['stable_after'], geometry_accepted=report['geometry_after'], solve_seconds=report['solve_seconds'], config_sha256=config['config_sha256'], source_poses_sha256=provenance['source_poses_sha256'], created_utc=stamp(), files={str(p.relative_to(folder)): sha(p) for p in folder.rglob('*') if p.is_file()})
            write(folder / 'verification.json', verified)
            events.put(dict(event='complete', key=name, **verified))
        except Exception:
            error = dict(event='failed', key=name, samples=job['samples'], error=traceback.format_exc(), created_utc=stamp())
            folder.mkdir(parents=True, exist_ok=True)
            write(folder / 'failure.json', error)
            events.put(error)
            if 'CUDA error' in error['error'] or 'illegal memory' in error['error']:
                raise
    write(output / f'worker_{number}.json', dict(pid=os.getpid(), device=device, stage='stopped', updated_utc=stamp()))

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
    devices = args.devices.split(',')
    if len(devices) != len(set(devices)) or any((not d.startswith('cuda:') or not d[5:].isdigit() for d in devices)):
        parser.error('Distinct CUDA devices required')
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
    context = mp.get_context('spawn')
    tasks = context.Queue(maxsize=2 * len(devices))
    events = context.Queue()
    processes = [context.Process(target=worker, args=(i, d, config, tasks, events)) for i, d in enumerate(devices)]
    stop = False

    def request_stop(*_):
        nonlocal stop
        stop = True
        (output / 'STOP').touch()
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    for p in processes:
        p.start()
    write(output / 'process.json', dict(pid=os.getpid(), workers=[p.pid for p in processes], created_utc=stamp()))
    pending = iter(jobs)
    next_job = next(pending, None)
    completed = dict(inherited)
    failures = {}
    sentinels = 0
    last_status = 0.0
    try:
        while any((p.is_alive() for p in processes)):
            if stop or (output / 'STOP').exists():
                stop = True
            if not stop and next_job is not None:
                try:
                    tasks.put(next_job, timeout=0.05)
                    next_job = next(pending, None)
                except queue.Full:
                    pass
            if (stop or next_job is None) and sentinels < len(processes):
                try:
                    tasks.put(None, timeout=0.05)
                    sentinels += 1
                except queue.Full:
                    pass
            try:
                event = events.get(timeout=0.2)
                if event['event'] == 'failed':
                    failures[event['key']] = event
                else:
                    completed[event['key']] = event
            except queue.Empty:
                pass
            if any((p.exitcode not in (None, 0) for p in processes)):
                request_stop()
            status = dict(state='stopping' if stop else 'running', updated_utc=stamp(), samples=sum((e['samples'] for e in completed.values())), stable=sum((e.get('stable', 0) for e in completed.values())), geometry_accepted=sum((e.get('geometry_accepted', 0) for e in completed.values())), completed_batches=len(completed), failed_batches=len(failures), by_type={t: sum((e['samples'] for k, e in completed.items() if k.startswith(t + '/'))) for t in sorted(set(g['type'] for g in groups))})
            if time.monotonic() - last_status >= 2:
                write(output / 'status.json', status)
                last_status = time.monotonic()
        for p in processes:
            p.join()
        while True:
            try:
                e = events.get_nowait()
                (failures if e['event'] == 'failed' else completed)[e['key']] = e
            except queue.Empty:
                break
        status.update(state='stopped' if stop else 'complete' if not failures else 'finished_with_errors', updated_utc=stamp(), completed_batches=len(completed), failed_batches=len(failures), samples=sum((e['samples'] for e in completed.values())), stable=sum(e.get('stable',0) for e in completed.values()), geometry_accepted=sum(e.get('geometry_accepted',0) for e in completed.values()), by_type={t:sum(e['samples'] for k,e in completed.items() if k.startswith(t+'/')) for t in sorted(set(g['type'] for g in groups))})
        write(output / 'status.json', status)
        write(output / 'failures.json', failures)
    finally:
        for p in processes:
            p.join()
if __name__ == '__main__':
    main()
