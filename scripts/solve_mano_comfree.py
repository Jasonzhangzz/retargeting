"""Current MANO patch CEM + five-step tangent-history spring runner; never changes production.

Accepts a saved MANO shard, or a frozen diagnostic directory with MANIFEST.json
and per-type NPZ packets. Final states only; no optimization histories saved.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MUJOCO_GL', 'egl')
import argparse, json, time, hashlib, sys
import math
from types import SimpleNamespace
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retarget.env.grasp.comfree_rollout import RolloutSettings
from retarget.env.grasp.mano_solver import ManoDecoder, FixedContacts, SurfaceBatch, MeshCache, ShadowSkeleton, Settings, initialize, lm_ik, pose_problem
from retarget.solver.cem import DEFAULTS
from retarget.initialization.dexonomy.util.torch_util import torch_matrix_to_quaternion

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
    tmp.replace(path)

def sync_time(device):
    torch.cuda.synchronize(device)
    return time.perf_counter()

def packets(args):
    if args.diagnostic_dir:
        root = Path(args.diagnostic_dir)
        manifest = json.loads((root / 'MANIFEST.json').read_text())
        for typ in manifest['types']:
            if args.types and typ not in args.types.split(','):
                continue
            a = dict(np.load(root / (typ + '.npz')))
            yield (typ, a, list(manifest['rows']), dict(input=str(root / (typ + '.npz')), sha256=sha(root / (typ + '.npz'))))
    else:
        root = Path(args.input)
        manifest = json.loads((root / 'manifest.json').read_text())
        original = dict(np.load(root / 'poses.npz'))
        geometry = dict(np.load(root / 'geometry.npz'))
        n = len(original['qpos'])
        rows = list(range(n)) if args.rows is None else [int(x) for x in args.rows.split(',')]
        if not rows or len(rows) != len(set(rows)) or any((i < 0 or i >= n for i in rows)):
            raise ValueError('Invalid or duplicate row indices')
        a = {k: v[rows] if v.ndim and v.shape[0] == n and (k not in ('mapped_mano_face_ids', 'mapped_mano_barycentric')) else v for k, v in original.items()}
        a.update(geometry)
        source = {}
        q = []
        for i in rows:
            case = manifest['cases'][i]
            p = Path(case['source_manifest']).parent / 'poses.npz'
            if p not in source:
                source[p] = np.load(p)['grasp_qpos']
            q.append(source[p][case['source_cached_case']])
        a['shadow_qpos'] = np.array(q)
        typ = manifest['cases'][rows[0]]['grasp_type']
        if any((manifest['cases'][i]['grasp_type'] != typ for i in rows)):
            raise ValueError('One grasp type required per shard')
        yield (typ, a, rows, dict(input=str(root), poses_sha256=sha(root / 'poses.npz'), geometry_sha256=sha(root / 'geometry.npz')))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--input', help='Saved MANO shard directory')
    source.add_argument('--diagnostic-dir', help='Frozen per-type diagnostic packets')
    parser.add_argument('--model', required=True)
    parser.add_argument('--mapping')
    parser.add_argument('--shadow-xml')
    parser.add_argument('--initialization', choices=['shadow-ik', 'saved-mano'], default='shadow-ik')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--rows')
    parser.add_argument('--types')
    parser.add_argument('--batch-size', type=int, default=DEFAULTS.resources.batch_size)
    parser.add_argument('--candidate-chunk', type=int, default=DEFAULTS.resources.candidate_chunk)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--score-only', action='store_true')
    parser.add_argument('--benchmark-repeats', type=int, default=3)
    parser.add_argument('--benchmark-candidates', type=int, default=0, help='Repeat the same fixed inputs for scoring throughput only')
    args = parser.parse_args()
    if min(args.batch_size, args.candidate_chunk, args.benchmark_repeats) < 1 or args.benchmark_candidates < 0:
        parser.error('Positive batch/chunk/repetition counts required')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'MANIFEST.json').exists():
        raise FileExistsError('Use a fresh output directory; existing experiment is immutable')
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; no CPU fallback')
    mapping_path = Path(args.mapping) if args.mapping else Path(args.diagnostic_dir or Path(args.input).parents[2]) / 'fixed_contact_mapping.json'
    mapping = json.loads(mapping_path.read_text())
    if mapping['model_sha256'] != sha(args.model):
        raise ValueError('MANO model/mapping fingerprint mismatch')
    s = DEFAULTS.rollout
    cem = DEFAULTS.sampling
    torch.set_num_threads(1)
    frozen = list(packets(args))
    if not frozen:
        raise ValueError('No matching input types')
    from retarget.env.grasp.mano_patch_anchor import PatchAnchorSettings
    objective = 'mano_patch_consistent_spring_history_h5_v5'
    criterion = 'patch_consistent_terminal_velocity_h5_v5'
    sources = [Path(__file__), *[ROOT / f for f in ['retarget/env/grasp/contact_geometry.py', 'retarget/env/grasp/mano_solver.py', 'retarget/env/grasp/mano_comfree.py', 'retarget/env/grasp/mano_contact_collision.py', 'retarget/env/grasp/mano_collision_geometry.py', 'retarget/env/grasp/mano_patch_anchor.py', 'retarget/env/grasp/mano_patch_consistent.py', 'retarget/env/grasp/comfree_rollout.py', 'retarget/env/grasp/spring_history.py', 'retarget/env/grasp/spring_history_warp.py', 'retarget/env/grasp/patch_query.py', 'retarget/env/grasp/triangle_penetration.py', 'retarget/solver/cem.py']]]
    manifest = dict(protocol=objective, criterion=criterion, arguments=vars(args), rollout=asdict(s), model_sha256=sha(args.model), mapping_sha256=sha(mapping_path), inputs=[dict(type=t, rows=r, **p) for t, a, r, p in frozen], input_count=sum((len(r) for t, a, r, p in frozen)), production_changed=False, patch_settings=asdict(PatchAnchorSettings()), spring=asdict(DEFAULTS.spring), scoring='Consistent patch fit + 5mm penetration allowances + terminal velocity', contact_response_source='Kelvin--Voigt + tangent history + Coulomb return; GPU backward Euler', early_stop='per pose, only joint patch/penetration/velocity acceptance; retain feasible candidates', geometry_execution_backend='optimized', spring_execution_backend='warp', source_sha256={str(p.relative_to(ROOT)): sha(p) for p in sources})
    write(output / 'MANIFEST.json', manifest)
    start = time.perf_counter()
    decoder = ManoDecoder(args.model, args.device)
    decoder.configure_contact_vertices(mapping['mappings'])
    cache = MeshCache(args.device)
    skeleton = None
    xml = args.shadow_xml or (str(Path(args.diagnostic_dir) / 'shadow_assets/right.xml') if args.diagnostic_dir else None)
    if xml is None:
        raise ValueError('--shadow-xml is required for Shadow IK initialization')
    skeleton = ShadowSkeleton(xml, args.device)
    model_loading = sync_time(args.device) - start
    reports = run_packets(args, output, decoder, mapping, skeleton, cache, frozen)
    write(output / 'report.json', dict(complete=True, samples=sum((r['samples'] for r in reports)), device=torch.cuda.get_device_name(args.device), model_loading_seconds=model_loading, objective=objective, criterion=criterion, reports=reports))
    write(output / 'status.json', dict(complete=True, reports=reports))

def run_packets(args, output, decoder, mapping, skeleton, cache, frozen):
    s=DEFAULTS.rollout
    cem=DEFAULTS.sampling
    reports = []
    for typ, a, source_rows, provenance in frozen:
        records = mapping['mappings'][typ]
        fixed = FixedContacts(decoder, records)
        for start in range(0, len(source_rows), args.batch_size):
            end = min(start + args.batch_size, len(source_rows))
            indices = np.arange(start, end)
            n = len(indices)
            t = sync_time(args.device)
            ids = list(map(int, a['object_geometry_ids'][indices]))
            geo = {i: dict(vertices=a[f'object_{i}_vertices'], faces=a[f'object_{i}_faces'], watertight=True) for i in set(ids)}
            surfaces = SurfaceBatch(geo, ids, a['object_pose_world'][indices], a['scene_scale'][indices], args.device, mesh_cache=cache)
            from retarget.env.grasp.mano_patch_consistent import PatchConsistentAdapter
            shadow_points, _ = skeleton.contacts(a['shadow_qpos'][indices], records)
            anchors = surfaces.query(shadow_points, torch.arange(n, device=args.device))[0]
            adapter = PatchConsistentAdapter(decoder, fixed, surfaces, args.candidate_chunk, rollout=s, anchors=anchors, spring_settings=DEFAULTS.spring)
            preparation = sync_time(args.device) - t
            t = sync_time(args.device)
            if args.initialization == 'saved-mano':
                mean = torch.tensor(np.concatenate((a['qpos'][indices, :6], a['mano_pca'][indices]), -1), device=args.device, dtype=torch.float32)
            else:
                target = skeleton(a['shadow_qpos'][indices])
                initial = initialize(decoder, target)
                mean, _, _ = lm_ik(decoder, initial, target, Settings())
            initialization = sync_time(args.device) - t
            rows = torch.arange(n, device=args.device)
            t = sync_time(args.device)
            adapter.evaluate(mean, rows)
            warmup = sync_time(args.device) - t
            initial_cost = adapter.validate(mean, rows)[0].clone()
            before = {k: v.clone() for k, v in adapter.last.items()}
            initial_accepted = before.get('patch_joint_accepted')
            repeat_count = args.benchmark_candidates or n
            select = torch.arange(repeat_count, device=args.device) % n
            candidates = mean[select]
            bench_rows = rows[select]
            timings = []
            for _ in range(args.benchmark_repeats):
                t = sync_time(args.device)
                adapter.evaluate(candidates, bench_rows)
                timings.append(sync_time(args.device) - t)
            print(json.dumps(dict(type=typ, batch_start=start, phase='scoring_ready', preparation_s=preparation, score_seconds=timings)), flush=True)
            t = sync_time(args.device)
            patch_state = None
            result = None
            if not args.score_only:
                from retarget.solver.cem import solve_patch_cem
                patch_state = solve_patch_cem(pose_problem(mean, adapter), adapter, seed=args.seed)
                result = SimpleNamespace(parameters=patch_state['parameters'])
            best = mean if result is None else result.parameters
            solving = sync_time(args.device) - t
            t = sync_time(args.device)
            final_cost = adapter.validate(best, rows)[0]
            with torch.no_grad():
                v, joints, _ = decoder(best, full=True)
            parts = adapter.last
            if not bool(torch.isfinite(final_cost).all()):
                raise RuntimeError('Non-finite final state; retain failure and stop, no fallback')
            monotonic = final_cost <= initial_cost + 1e-05 + 1e-05 * initial_cost.abs()
            if getattr(adapter, 'retain_feasible', False):
                monotonic |= ~initial_accepted & parts['patch_joint_accepted']
                if bool((initial_accepted & ~parts['patch_joint_accepted']).any()):
                    raise RuntimeError('Lost an initially accepted pose')
            if not bool(monotonic.all()):
                raise RuntimeError('Final score failed monotonic/feasibility verification')
            acceptance = sync_time(args.device) - t
            folder = output / typ / f'{start:07d}'
            folder.mkdir(parents=True, exist_ok=True)
            t = time.perf_counter()
            final_object = torch.cat((parts['object_final_origin'], torch_matrix_to_quaternion(parts['rollout_rotation'])), -1)
            arrays = dict(variables=best, qpos=torch.cat((best[:, :6], decoder.angles(best)), -1), mano_joints=joints, object_pose_world=torch.tensor(a['object_pose_world'][indices], device=args.device), object_pose_after_rollout=final_object, object_scale=torch.tensor(a['object_scale'][indices], device=args.device), scene_scale=torch.tensor(a['scene_scale'][indices], device=args.device), initial_cost=initial_cost, final_cost=final_cost, **parts)
            if patch_state:
                arrays.update({'optimizer_' + k: patch_state[k] for k in ['evaluations', 'iterations', 'early_stopped']})
            np.savez_compressed(folder / 'final.npz', **{k: v.detach().cpu().numpy() for k, v in arrays.items()})
            metrics = []
            for i in range(n):

                def finite(value):
                    value = float(value)
                    return value if math.isfinite(value) else None

                def metric(p):
                    result = dict(stable=bool(p['rollout_stable'][i]),velocity_cost=finite(p['rollout_cost'][i]),
                        linear_speed_m_s=finite(p['rollout_linear_speed_m_s'][i]),angular_speed_rad_s=finite(p['rollout_angular_speed_rad_s'][i]),
                        displacement_m=finite(p['rollout_displacement_m'][i]),rotation_deg=finite(math_degrees(p['rollout_rotation_rad'][i])),
                        contacts=int(p['collision_contact_count'][i]),valid=bool(p['valid'][i]),
                        geometry_accepted=bool(p['patch_joint_accepted'][i]),patch_geometry_accepted=bool(p['patch_geometry_accepted'][i]),
                        coverage=finite(p['patch_coverage'][i]),all_required_fingers=bool(p['patch_all_required_fingers'][i]),
                        anchor_cost=finite(p['anchor_cost'][i]),patch_gap_m=p['patch_gap_m'][i].cpu().tolist(),
                        patch_anchor_offset_m=p['patch_anchor_offset_m'][i].cpu().tolist(),
                        penetration_depth_m=p['penetration_depth_m'][i].cpu().tolist(),
                        fingers={name:dict(fitted=bool(p['patch_group_fitted'][i,j]),coverage=finite(p['patch_fitted'][i,ids].float().mean()))
                            for j,(name,ids) in enumerate(zip(adapter.group_names,adapter.required_groups))})
                    return result
                metrics.append(dict(source_row=source_rows[start + i], before=metric(before), after=metric(parts), criterion=adapter.criterion))
            write(folder / 'metrics.json', metrics)
            write(folder / 'model.json', dict(model_sha256=sha(decoder.base.path), appearance=decoder.base.appearance, asset_metadata=decoder.base.asset_metadata))
            report = dict(type=typ, rows=source_rows[start:end], samples=n, initialization_seconds=initialization, solve_seconds=solving, acceptance_seconds=acceptance, preparation_seconds=preparation, warmup_seconds=warmup, save_seconds=time.perf_counter() - t, score_candidate_count=repeat_count, score_seconds=timings, score_only=args.score_only, pose_solver='cem', objective=adapter.objective, group_names=adapter.group_names, cem_updates_per_pose=patch_state['iterations'].cpu().tolist() if patch_state else cem.iterations if result is not None else 0, candidate_evaluations_per_pose=patch_state['evaluations'].cpu().tolist() if patch_state else 0 if result is None else cem.population * cem.iterations + 1, early_stopped_poses=int(patch_state['early_stopped'].sum()) if patch_state else 0, stable_before=sum((r['before']['stable'] for r in metrics)), stable_after=sum((r['after']['stable'] for r in metrics)), geometry_before=sum((r['before']['geometry_accepted'] for r in metrics)), geometry_after=sum((r['after']['geometry_accepted'] for r in metrics)), output=str(folder))
            write(folder / 'report.json', report)
            reports.append(report)
            write(output / 'status.json', dict(complete=False, reports=reports))
            print(json.dumps(report), flush=True)
    return reports

def math_degrees(value):
    return float(value) * 180 / np.pi
if __name__ == '__main__':
    main()
