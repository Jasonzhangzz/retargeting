#!/usr/bin/env python3
"""Backfill W&B curves from saved checkpoints; --follow also observes future training."""
import argparse
import fcntl
import json
from pathlib import Path
import signal
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retarget.learning.wuji_ik.config import VARIANTS
from retarget.learning.wuji_ik.tracking import DEFAULT_BASE_URL, WandbCurves


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--base-url', default=DEFAULT_BASE_URL)
    parser.add_argument('--project', default='wuji2-ik')
    parser.add_argument('--entity')
    parser.add_argument('--name-prefix', help='Run display name prefix, e.g. "ik policy"')
    parser.add_argument('--follow', action='store_true')
    parser.add_argument('--interval', type=float, default=30.)
    args = parser.parse_args()
    if args.interval < 1:
        raise ValueError('Polling interval must be at least one second')
    root = Path(args.campaign)
    lock = (root/'wandb_sync.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stop = False
    def request_stop(*_):
        nonlocal stop
        stop = True
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, request_stop)
    tracker = WandbCurves(root, args.project, args.entity, args.base_url,
                         name_prefix=args.name_prefix, keepalive=args.follow)
    try:
        while not stop:
            try:
                for variant in VARIANTS:
                    if stop:
                        break
                    receipt = tracker.sync(variant)
                    if receipt:
                        print(json.dumps(dict(variant=variant, **receipt)), flush=True)
            except OSError as error:
                if error.errno not in (5, 116):
                    raise
                print(f'Shared storage returned errno {error.errno}; retrying without restarting training.', flush=True)
            if not args.follow:
                break
            deadline = time.monotonic()+args.interval
            while not stop and time.monotonic() < deadline:
                time.sleep(min(1., max(0., deadline-time.monotonic())))
    finally:
        tracker.close()


if __name__ == '__main__':
    main()
