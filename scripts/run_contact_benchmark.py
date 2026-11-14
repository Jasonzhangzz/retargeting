#!/usr/bin/env python
"""Self-contained contact CEM benchmark: run, replay, or execute the fixed campaign."""
import argparse
from pathlib import Path
import json
import os
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('MUJOCO_GL', 'egl')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['sharpa_pen', 'sharpa_coin', 'h1_jog'], default='sharpa_pen')
    parser.add_argument('--method', choices=['contact_cem', 'cem', 'source_sampling'], default='contact_cem')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, help='Partial diagnostic only; never a full-task pass')
    parser.add_argument('--reference', type=Path, help='Frozen H1 reference.npz')
    parser.add_argument('--output-name', default='contact_benchmark')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--headless', action='store_true', help='Accepted explicitly; runs are headless by default')
    parser.add_argument('--save-video', action='store_true')
    parser.add_argument('--replay', type=Path)
    parser.add_argument('--campaign', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Continue remaining jobs in an existing campaign')
    args = parser.parse_args()
    from retarget.config.run import Config, ROOT
    from retarget.solver.base import Settings
    from retarget.common.results import write_json
    from retarget.env.contact.runner import run
    if (Path(args.output_name).name != args.output_name or args.output_name.startswith('.') or args.seed < 0
            or args.steps is not None and args.steps < 1):
        parser.error('output-name must be one visible directory name; seed and steps must be valid')
    if args.replay:
        from retarget.viewer.runtime import replay
        cfg = Config(Path(__file__), 'replay', {}, Settings(), {}, {'name': args.replay.resolve().name},
                     {'live': True, 'save_video': args.save_video}, 0)
        replay(cfg)
        return
    output = ROOT / 'output' / args.output_name
    if args.resume and not args.campaign:
        parser.error('--resume requires --campaign')
    if output.exists() and not args.resume:
        parser.error(f'Output already exists: {output}; choose a new output-name')
    output.mkdir(parents=True, exist_ok=args.resume)
    if args.campaign:
        jobs = [(task, method, seed, False) for task, seeds in [('h1_jog', [0, 1, 2]),
            ('sharpa_pen', [0, 1, 2]), ('sharpa_coin', [0, 1, 2, 10000])]
            for method in ['source_sampling', 'cem', 'contact_cem'] for seed in seeds]
        jobs += [('h1_jog', method, seed, True) for method in ['source_sampling', 'cem', 'contact_cem'] for seed in [0, 1, 2]]
        results_file = output / 'results.json'
        results = json.loads(results_file.read_text()) if args.resume and results_file.exists() else []
        done = {(r['task'], r['method'], r['seed'], r['tracking']) for r in results}
        for task, method, seed, tracking in jobs:
            if (task, method, seed, tracking) in done:
                continue
            name = f'{args.output_name}_{task}_{method}_{seed}' + ('_tracking' if tracking else '')
            base, retry = name, 0
            while (ROOT / 'output' / name).exists():
                retry += 1
                name = f'{base}_retry{retry}'
            cmd = [sys.executable, str(Path(__file__).resolve()), '--task', task, '--method', method,
                   '--seed', str(seed), '--output-name', name, '--device', args.device, '--headless']
            if args.save_video:
                cmd.append('--save-video')
            if args.steps:
                cmd += ['--steps', str(args.steps)]
            if tracking:
                source_run = next((r['run'] for r in reversed(results) if r['task'] == 'h1_jog' and
                    r['method'] == 'source_sampling' and r['seed'] == seed and not r['tracking']),
                    f'{args.output_name}_h1_jog_source_sampling_{seed}')
                reference = ROOT / 'output' / source_run / 'reference.npz'
                cmd += ['--reference', str(reference)]
            print('RUN', name, flush=True)
            tick = time.perf_counter()
            with (output / f'{name}.log').open('w') as log:
                child = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
            summary = ROOT / 'output' / name / 'summary.json'
            result = json.loads(summary.read_text()) if summary.exists() else dict(task_success=False)
            result.update(run=name, exit_code=child.returncode, wall_seconds=time.perf_counter() - tick,
                          task=task, method=method, seed=seed, tracking=tracking)
            results.append(result)
            write_json(output / 'results.json', results)
            report = ['# Contact-space CEM benchmark', '',
                'CPU MuJoCo 4 speculative execution; GPU discrete-contact proposals. Partial runs cannot pass.', '',
                '| Task | Method | Seed | Tracking | Complete | Pass | GPU rollouts | CPU4 rollouts | Seconds |',
                '|---|---|---:|---|---|---|---:|---:|---:|']
            for r in results:
                report.append(f"| {r['task']} | {r['method']} | {r['seed']} | {r['tracking']} | {r.get('complete', False)} | {r['task_success']} | {r.get('gpu_rollouts', 0)} | {r.get('cpu4_rollouts', 0)} | {r['wall_seconds']:.1f} |")
            (output / 'REPORT.md').write_text('\n'.join(report) + '\n')
            print('DONE', name, result['task_success'], flush=True)
        return
    options = {'baseline': 'source_sampling'} if args.method == 'source_sampling' else {}
    if args.reference:
        options['physics_reference'] = str(args.reference.resolve())
    environment = {'device': args.device}
    if args.steps:
        environment['frames'] = args.steps
    config = Config(Path(__file__).resolve(), 'solve', dict(type='contact', name=args.task,
        data=str(ROOT / 'data/contact'), options=options), Settings('cem' if args.method == 'source_sampling' else args.method),
        environment, {'name': args.output_name}, {'live': False, 'save_video': args.save_video}, args.seed)
    write_json(output / 'config.json', config.resolved())
    try:
        cases = run(config, output)
        write_json(output / 'manifest.json', dict(version=1, task='contact', method=args.method, cases=cases))
        if args.save_video:
            from retarget.viewer.runtime import video
            for case in cases:
                video(output, case)
        print(output, flush=True)
    except Exception as error:
        status_path = output / 'status.json'
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        status.update(state='failed', error=f'{type(error).__name__}: {error}')
        write_json(status_path, status)
        manifest = output / 'manifest.json'
        if args.save_video and manifest.exists():
            # A solver failure still has a useful partial trajectory. Preserve
            # the original exception even if its optional rendering also fails.
            try:
                from retarget.viewer.runtime import video
                for case in json.loads(manifest.read_text())['cases']:
                    video(output, case)
                write_json(output / 'video_status.json', dict(state='partial_trajectory_rendered'))
            except Exception as rendering_error:
                write_json(output / 'video_status.json', dict(state='failed', error=str(rendering_error)))
        raise


if __name__ == '__main__':
    main()
