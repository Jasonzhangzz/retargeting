#!/usr/bin/env python3
"""Finite, single-owner real-EMF dataset build followed by one selected training job."""
import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def stamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def shared_io(operation, attempts=3):
    """Bounded retry of transient shared-filesystem I/O, never a job retry."""
    for attempt in range(attempts):
        try:
            return operation()
        except OSError as error:
            if error.errno not in (5, 116) or attempt+1 == attempts:
                raise
            time.sleep(.1*(attempt+1))


def read_text(path):
    return shared_io(path.read_text)


def read_json(path):
    return json.loads(read_text(path))


def write(path, value):
    payload = json.dumps(value, indent=2, ensure_ascii=False)+'\n'
    def commit():
        tmp = path.with_name(path.name+'.tmp')
        tmp.write_text(payload)
        os.replace(tmp, path)
    shared_io(commit)


def monitoring_snapshot(root, variant='temporal_causal'):
    """Optional telemetry failure must not stop an otherwise healthy worker."""
    fields = {}
    try:
        progress = root/variant/'train.jsonl'
        if progress.exists():
            lines = read_text(progress).splitlines()
            if lines:
                try:
                    fields['last_update'] = json.loads(lines[-1])
                except json.JSONDecodeError:
                    pass  # Writer may still be appending the last record.
        receipt_path = root/'wandb_sync.json'
        if receipt_path.exists():
            fields['receipt'] = read_json(receipt_path).get('runs', {}).get(variant, {})
    except OSError as error:
        if error.errno not in (5, 116):
            raise
        fields['monitoring_error'] = str(error)
    return fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--resume-pretraining', action='store_true',
                        help='Explicitly continue a stopped/failed data stage before any model training')
    parser.add_argument('--resume-training',action='store_true',help='Resume a stopped owned training process from its committed checkpoint')
    args = parser.parse_args()
    root = args.campaign.resolve()
    conf = read_json(root/'MANIFEST.json')
    variant = conf.get('variant', 'temporal_causal')
    if variant not in ('temporal_causal','temporal_bidirectional','frame_regression','frame_flow'):
        raise ValueError('Unknown training variant')
    control = read_json(root/'CONTROL.json')
    if not control['authorized'] or control['stop_requested'] or (root/'STOP').exists():
        raise RuntimeError('Campaign is not authorized to start or is stopped')
    if not read_json(root/'preflight.json')['passed']:
        raise ValueError('CPU SDK/data/authentication preflight has not passed')
    lock = (root/'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    previous = None
    if (root/'STATUS.json').exists():
        previous = read_json(root/'STATUS.json')
        if args.resume_training:
            if previous['status'] not in ('stopped','failed') or previous.get('active') is not None or not (root/variant/'latest.pt').exists():
                raise RuntimeError('Training resume requires a stopped receipt and complete checkpoint')
            if Path('/proc',str(previous['pid'])).exists() or (previous.get('child') and Path('/proc',str(previous['child']['pid'])).exists()):
                raise RuntimeError('Old training processes must exit before resume')
        elif (not args.resume_pretraining or previous['status'] not in ('stopped','failed')
              or (root/variant).exists() or previous.get('active') is not None):
            raise RuntimeError('Prior receipt is not a safely stopped pretraining stage')
        archive = root/'revisions'/str(time.time_ns())
        archive.mkdir(parents=True)
        write(archive/'previous_STATUS.json', previous)
    state = dict(pid=os.getpid(), created=stamp(), control_revision=control['revision'],
                 status='prepared', active=None, jobs={variant: {'status': 'queued'}},
                 committed_optimizer_steps=conf.get('resume_checkpoint_step',0) if args.resume_training else 0, planned_epochs=conf['epochs'], automatic_retries=0)
    if previous is not None:
        state['continuation'] = dict(previous_controller_pid=previous['pid'],
            previous_status=previous['status'], optimizer_updates_before_resume=conf.get('resume_checkpoint_step',0), new_optimizer_updates_before_start=0,
            reason='Explicit implementation repair; original task authorization remains active',
            training_resume=args.resume_training)
    stop = False
    child = None
    tracker = None
    training_lock = None

    def publish(status, **fields):
        state.update(status=status, updated=stamp(), **fields)
        write(root/'STATUS.json', state)

    def requested(*_):
        nonlocal stop
        stop = True

    def check_stop():
        nonlocal stop
        latest = read_json(root/'CONTROL.json')
        stop |= latest['stop_requested'] or (root/'STOP').exists() or latest['revision'] != control['revision']
        if stop:
            raise InterruptedError('User/controller stop requested')

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, requested)
    ram = Path('/dev/shm/wuji_lm_runtime_2207_20261111')
    env = os.environ.copy()
    env.update(PYTHONPATH=conf['runtime']+':'+str(ram)+':'+conf['sdk_path']+':'+conf['reference'],
        CUDA_VISIBLE_DEVICES=','.join(map(str, conf['gpus'])), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='2', HDF5_USE_FILE_LOCKING='FALSE', TORCHINDUCTOR_COMPILE_THREADS='2',
        TORCHINDUCTOR_CACHE_DIR=str(ram/'inductor_cache'), TRITON_CACHE_DIR=str(ram/'triton_cache'))

    if conf.get('temporary_directory'):
        temporary = Path(conf['temporary_directory']); temporary.mkdir(parents=True,exist_ok=True)
        env.update(TMPDIR=str(temporary),TMP=str(temporary),TEMP=str(temporary))

    def run(stage, argv):
        nonlocal child
        check_stop()
        publish(stage, active=None, command=argv)
        with (root/(stage+'.log')).open('a') as log:
            child = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            state['child'] = dict(pid=child.pid, argv=argv, started=stamp(), stage=stage)
            publish(stage)
            while child.poll() is None:
                check_stop()
                time.sleep(1)
            result = child.returncode
        state['child'].update(exit_code=result, finished=stamp())
        if result:
            raise RuntimeError(stage+' exited '+str(result))
        child = None

    try:
        publish('prepared')
        if conf.get('reuse_prepared_campaign'):
            previous_root = Path(conf['reuse_prepared_campaign'])
            previous = read_json(previous_root/'MANIFEST.json')
            if previous['dataset'] != conf['dataset']:
                raise ValueError('Prepared reuse points to a different dataset')
            metadata = read_json(Path(conf['dataset'])/'manifest.json')
            if metadata['signature'] != conf['expected_data_signature'] or not metadata['complete']:
                raise ValueError('Prepared data signature mismatch')
            for name, expected in metadata['array_sha256'].items():
                check_stop()
                digest = hashlib.sha256()
                with (Path(conf['dataset'])/name).open('rb') as f:
                    for block in iter(lambda:f.read(8<<20), b''): digest.update(block)
                if digest.hexdigest() != expected: raise ValueError('Prepared data changed: '+name)
            if read_json(root/'data_validation.json') != read_json(previous_root/'data_validation.json'):
                raise ValueError('Reused quality validation changed')
            publish('prepared_cache_verified', data_signature=metadata['signature'])
        else:
            if conf.get('reuse_screened_campaign'):
                previous_root = Path(conf['reuse_screened_campaign'])
                for name in ('inventory.json', 'legacy_import_audit.json'):
                    if (root/name).read_bytes() != (previous_root/name).read_bytes():
                        raise ValueError('Imported inventory changed while reusing sensor screening')
                if not read_json(root/'legacy_import_audit.json')['complete']:
                    raise ValueError('Prior legacy import is incomplete')
                publish('legacy_import_reused', previous_campaign=str(previous_root))
            else:
                run('legacy_import', [conf['python'], '-u', str(root/'scripts/import_wuji_legacy_emf.py'),
                                     '--campaign', str(root), '--workers', '4'])
            builder = conf.get('dataset_builder', 'build_wuji_real_sdk_dataset.py')
            if builder not in ('build_wuji_real_sdk_dataset.py', 'build_wuji_quality_sdk_dataset.py'):
                raise ValueError('Unregistered dataset builder')
            run('building_data', [conf['python'], '-u', str(root/'scripts'/builder),
                                  '--campaign', str(root), '--reference', conf['reference'], '--workers', '8'])
        validation = read_json(root/'data_validation.json')
        metadata = read_json(Path(conf['dataset'])/'manifest.json')
        if not validation['passed'] or validation['frames'] != conf['frames']:
            raise ValueError('Data generation/validation is incomplete')
        if conf.get('require_strict_quality') and (not validation.get('quality_gates_passed')
                or validation.get('quality_frames_checked') != conf['frames'] or validation.get('quality_failures') != 0):
            raise ValueError('Strict SDK confidence/FK quality validation is incomplete')
        if (sum(metadata['split_rows'].values()) != conf['frames']
                or metadata['split_rows'].get('test', 0) != 0
                or not .045 <= metadata['split_rows']['val']/conf['frames'] <= .055):
            raise ValueError('Unexpected real-data split')
        # Do not race an older four-card job from the same IK queue.
        repo = Path(__file__).resolve().parents[1]
        # Archived launch scripts live under <campaign>/scripts, so the repository
        # is explicitly recorded rather than inferred from their new location.
        repo = Path(conf.get('repository', str(repo)))
        training_lock = (repo/'results/wuji_ik_training_queue.lock').open('a')
        while True:
            check_stop()
            try:
                fcntl.flock(training_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                publish('waiting_for_previous_queue')
                time.sleep(5)
        while True:
            check_stop()
            output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                                               '--format=csv,noheader,nounits'], text=True)
            free = {int(x.split(',')[0]): int(x.split(',')[1]) for x in output.splitlines()}
            if min(free[g] for g in conf['gpus']) >= conf['memory_gib_per_gpu']*1024+2048:
                break
            publish('waiting_for_memory', free_mib=free)
            time.sleep(10)
        configuration = dict(created=stamp(), prepared=conf['dataset'], data_signature=metadata['signature'],
            split_rows=metadata['split_rows'], epochs=conf['epochs'], variants=[variant],
            gpus=conf['gpus'], effective_frames_per_gpu=conf['frames_per_gpu'],
            initialization='resume' if args.resume_training else 'from_scratch', source_snapshot=str(root/'source_snapshot'),
            dataset_summary={k: metadata[k] for k in ('rows', 'segments', 'label_source', 'eligible_collections')},
            wandb_run_names={variant: conf['wandb'].get('run_name', 'ik distilled')})
        write(root/'configuration.json', configuration)
        argv = [conf['python'], '-u', '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=4',
            '--max-restarts=0', '-m', 'retarget.learning.wuji_ik', 'train', '--prepared', conf['dataset'],
            '--output', str(root/variant), '--variant', variant,
            '--epochs', str(conf['epochs']), '--batch-size', '1', '--effective-frames-per-gpu', str(conf['frames_per_gpu']),
            '--accumulation', '1', '--workers', '2', '--threads', '2', '--checkpoint-every', '1000', '--seed', '42',
            '--gpu-memory-limit-gib', str(conf['memory_gib_per_gpu']), '--checkpoint-blocks', '--backend', conf.get('attention_backend','auto'),
            '--muon-lr', '.02', '--adamw-lr', '.0003', '--warmup-fraction', '.02',
            '--smoothness-weight', str(conf['smoothness_weight'])]
        if args.resume_training:
            argv += ['--resume',str(root/variant/'latest.pt')]
        elif (root/variant/'latest.pt').exists():
            raise RuntimeError('Unexpected checkpoint in fresh real-data campaign')
        check_stop()
        log = (root/(variant+'.log')).open('a')
        child = subprocess.Popen(argv, env=env, cwd=conf['runtime'], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        state['jobs'][variant].update(status='running', pid=child.pid, argv=argv, started=stamp())
        state['child'] = dict(pid=child.pid, argv=argv, started=stamp(), stage='training')
        planned_steps = math.ceil(metadata['split_rows']['train']/(conf['frames_per_gpu']*len(conf['gpus'])))*conf['epochs']
        publish('training', active=variant, planned_optimizer_steps=planned_steps)
        # Only the metrics process needs W&B authentication. Do not include the key
        # in manifests, argv, source snapshots or training-rank environments.
        tracking_env = env.copy()
        tracking_env['WANDB_API_KEY'] = (root/'.wandb_api_key').read_text().strip()
        tracking_env['WANDB_BASE_URL'] = conf['wandb']['base_url']
        tracking_argv = [conf['python'], '-u', str(root/'scripts/sync_wuji_ik_wandb.py'), '--campaign', str(root),
            '--project', conf['wandb']['project'], '--entity', conf['wandb']['entity'],
            '--name-prefix', conf['wandb'].get('run_name','ik distilled'), '--follow', '--interval', '15']
        tracking_log = (root/'wandb_sync.log').open('a')
        tracker = subprocess.Popen(tracking_argv, env=tracking_env, stdout=tracking_log, stderr=subprocess.STDOUT, start_new_session=True)
        (root/'wandb_sync.pid').write_text(str(tracker.pid))
        state['tracker'] = dict(pid=tracker.pid, started=stamp())
        while child.poll() is None:
            check_stop()
            snapshot = monitoring_snapshot(root,variant)
            if 'last_update' in snapshot:
                state['jobs'][variant]['last_update'] = snapshot['last_update']
            if 'receipt' in snapshot:
                state['committed_optimizer_steps'] = snapshot['receipt'].get('last_logged_step', 0)
            if 'monitoring_error' in snapshot:
                state['monitoring_error'] = snapshot['monitoring_error']
            if tracker.poll() is not None:
                state['tracker']['exit_code'] = tracker.returncode
                state['wandb_health'] = 'failed; training continues and curves remain recoverable in JSONL'
            publish('training', active=variant)
            time.sleep(5)
        result = read_json(root/variant/'status.json') if (root/variant/'status.json').exists() else {}
        if child.returncode or not result.get('complete'):
            raise RuntimeError('Training failed; no automatic retry or next variant scheduled')
        state['committed_optimizer_steps'] = result['step']
        state['jobs'][variant].update(status='complete', result=result, finished=stamp())
        publish('complete', active=None)
        # Give the existing metrics process one full cycle to publish completion.
        for _ in range(20):
            check_stop()
            receipt_path = root/'wandb_sync.json'
            if receipt_path.exists():
                receipt = read_json(receipt_path).get('runs', {}).get(variant, {})
                if receipt.get('last_logged_step') == result['step'] and receipt.get('training_state') == 'complete':
                    break
            time.sleep(1)
    except InterruptedError as exc:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        publish('stopped', active=None, reason=str(exc), checkpoint_policy='Only last atomic committed checkpoint is resumable; in-flight updates are not claimed saved')
    except BaseException as exc:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        publish('failed', active=None, reason=type(exc).__name__+':'+str(exc))
        raise
    finally:
        if tracker is not None and tracker.poll() is None:
            os.killpg(tracker.pid, signal.SIGTERM)
            try:
                tracker.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(tracker.pid, signal.SIGKILL)
                tracker.wait()


if __name__ == '__main__':
    main()
