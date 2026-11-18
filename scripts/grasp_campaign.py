"""Shared spawn-based campaign scheduling for MANO and native robot hands.

Backends own model loading, verified receipts and one batch's numerical work.
This module owns process lifetime, exclusive task assignment and accounting.
It deliberately imports no GPU libraries in the controller.
"""
from __future__ import annotations

import importlib
import multiprocessing as mp
import os
from pathlib import Path
import queue
import signal
import time
import traceback

from scripts.mano_campaign_inputs import retry_io, stamp, write

write = retry_io(write)


def parse_devices(value):
    devices = value.split(',') if isinstance(value, str) else list(value)
    if not devices or len(set(devices)) != len(devices) or any(
        not d.startswith('cuda:') or not d[5:].isdigit() for d in devices
    ):
        raise ValueError('Distinct CUDA devices required')
    return devices


def _worker(number, device, config, backend_name, tasks, events):
    output = Path(config['output'])
    current = None

    def progress(stage, **details):
        event = dict(event='progress', worker=number, pid=os.getpid(), device=device,
                     stage=stage, updated_utc=stamp(), **details)
        if current is not None:
            event.update(key=current['key'], type=current['type'], samples=current['samples'])
        write(output / f'worker_{number}.json', event)
        events.put(event)

    try:
        progress('loading')
        module, name = backend_name.split(':')
        backend = getattr(importlib.import_module(module), name)(device, config, progress)
        progress('ready')
        while not (output / 'STOP').exists():
            try:
                item = tasks.get(timeout=.2)
            except queue.Empty:
                continue
            if item is None:
                break
            # STOP may have arrived while waiting for a queued batch.
            if (output / 'STOP').exists():
                break
            job, current = item
            progress('preparing')
            try:
                receipt = backend.completed(job)
                if receipt is None:
                    receipt = backend.solve(job, progress)
                if not receipt.get('success') or receipt['samples'] != current['samples']:
                    raise ValueError('Backend returned an invalid completion receipt')
                events.put(dict(receipt, event='complete', key=current['key'],
                                type=current['type'], worker=number, device=device))
                current = None
                progress('ready')
            except Exception:
                error = dict(event='failed', worker=number, device=device,
                             **current, error=traceback.format_exc(), created_utc=stamp())
                write(output / f'worker_{number}_failure.json', error)
                events.put(error)
                # Stop dispatch after any failed batch; never silently drop rows.
                (output / 'STOP').touch()
                break
        progress('stopped')
    except BaseException:
        error = dict(event='failed', key=f'worker:{number}', type='', samples=0,
                     worker=number, device=device, error=traceback.format_exc(), created_utc=stamp())
        write(output / f'worker_{number}_failure.json', error)
        events.put(error)
        raise


def run_campaign(*, config, jobs, devices, backend, job_key, completed=None,
                 metrics=('stable', 'geometry_accepted'), status_extra=None, detailed_types=False):
    """Run each pending job once, retaining inherited results in all totals.

    The caller holds campaign.lock and validates input/configuration identities.
    Signals and STOP drain only batches already started. A failed worker stops
    new dispatch, lets other workers save their current batches, and is fatal.
    """
    devices = parse_devices(devices)
    output = Path(config['output'])
    if (output / 'STOP').exists():
        raise RuntimeError('STOP present: launch a separate continuation')
    completed = dict(completed or {})
    pending = []
    identities = set()
    for job in jobs:
        key = job_key(job)
        if key in identities:
            raise ValueError('Duplicate campaign job: ' + key)
        identities.add(key)
        if key not in completed:
            pending.append((job, dict(key=key, type=job['type'],
                                      samples=job.get('samples', len(job.get('rows', ()))))))
    total = sum(v['samples'] for v in completed.values()) + sum(x[1]['samples'] for x in pending)
    context = mp.get_context('spawn')
    tasks = context.Queue(maxsize=2 * len(devices))
    events = context.Queue()
    processes = [context.Process(target=_worker, args=(i, d, config, backend, tasks, events))
                 for i, d in enumerate(devices)]
    failures, workers = {}, {}
    requested_stop = False
    previous_handlers = {}

    def request_stop(*_):
        nonlocal requested_stop
        requested_stop = True
        (output / 'STOP').touch()

    def consume(event):
        kind = event['event']
        if kind == 'progress':
            workers[event['worker']] = event
        elif kind == 'failed':
            failures[event['key']] = event
        elif kind == 'complete':
            if event['key'] in completed:
                raise ValueError('Duplicate completion: ' + event['key'])
            completed[event['key']] = event

    def status(state):
        types = sorted({v['type'] for _, v in pending} | {v.get('type', k.split('/')[0]) for k, v in completed.items()})
        processed = sum(v['samples'] for v in completed.values())
        result = dict(status_extra or {}, state=state, updated_utc=stamp(), devices=devices,
                      batch_size=config.get('batch_size'), candidate_chunk=config.get('candidate_chunk'),
                      total_inputs=total, samples=processed, processed=processed,
                      completed_batches=len(completed), failed_batches=len(failures),
                      active=[v for v in workers.values() if v['stage'] in ('preparing', 'solving')],
                      workers=list(workers.values()),
                      by_type={typ:dict(samples=sum(v['samples'] for k,v in completed.items() if v.get('type',k.split('/')[0]) == typ),
                                        processed=sum(v['samples'] for k,v in completed.items() if v.get('type',k.split('/')[0]) == typ),
                                        **{m:sum(v.get(m,0) for k,v in completed.items() if v.get('type',k.split('/')[0]) == typ) for m in metrics}) for typ in types},
                      **{m:sum(v.get(m,0) for v in completed.values()) for m in metrics})
        if not detailed_types:
            result['by_type'] = {typ:counts['samples'] for typ,counts in result['by_type'].items()}
        write(output / 'status.json', result)
        return result

    started = []
    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[sig] = signal.signal(sig, request_stop)
        status('starting')
        for process in processes:
            process.start()
            started.append(process)
        write(output / 'process.json', dict(pid=os.getpid(), workers=[p.pid for p in started],
                                           devices=devices, created_utc=stamp()))
        cursor, sentinels, last_status = 0, 0, 0.
        while any(p.is_alive() for p in started):
            stopping = requested_stop or (output / 'STOP').exists() or bool(failures)
            if not stopping and cursor < len(pending):
                try:
                    tasks.put(pending[cursor], timeout=.05)
                    cursor += 1
                except queue.Full:
                    pass
            if (stopping or cursor == len(pending)) and sentinels < len(started):
                try:
                    tasks.put(None, timeout=.05)
                    sentinels += 1
                except queue.Full:
                    pass
            try:
                consume(events.get(timeout=.1))
            except queue.Empty:
                pass
            for number, process in enumerate(started):
                if process.exitcode not in (None, 0) and f'worker:{number}' not in failures:
                    failures[f'worker:{number}'] = dict(event='failed', samples=0,
                        error=f'Worker {number} exited with code {process.exitcode}')
            if failures:
                (output / 'STOP').touch()
            if time.monotonic() - last_status >= 2:
                status('stopping' if stopping else 'running')
                last_status = time.monotonic()
        for process in started:
            process.join()
        while True:
            try:
                consume(events.get(timeout=.1))
            except queue.Empty:
                break
        state = 'failed' if failures else 'stopped' if requested_stop or (output/'STOP').exists() else 'complete'
        if state == 'complete' and sum(v['samples'] for v in completed.values()) != total:
            failures['controller'] = dict(error='Workers exited before every input was saved', samples=0)
            state = 'failed'
        status(state)
        write(output / 'failures.json', failures)
        return completed, failures
    finally:
        if any(p.is_alive() for p in started):
            (output / 'STOP').touch()
        # Continue draining events while workers finish, avoiding queue-feeder deadlocks.
        while any(p.is_alive() for p in started):
            try:
                events.get(timeout=.1)
            except queue.Empty:
                pass
        for process in started:
            process.join()
        tasks.cancel_join_thread()
        tasks.close()
        events.close()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
