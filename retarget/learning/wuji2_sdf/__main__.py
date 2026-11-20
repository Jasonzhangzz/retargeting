import argparse
import fcntl
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description='WUJI2 configuration-conditioned SDF sampling and MLP training')
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('sample')
    s.add_argument('--output', required=True)
    s.add_argument('--poses', type=int, default=100000)
    s.add_argument('--per-link', type=int, default=8)
    s.add_argument('--shard-poses', type=int, default=1024)
    s.add_argument('--pose-batch', type=int, default=16)
    s.add_argument('--collision-threshold-mm', type=float, default=.5)
    s.add_argument('--pose-filter', choices=['all', 'free'], default='all')
    t = sub.add_parser('train')
    t.add_argument('--data', required=True)
    t.add_argument('--output', required=True)
    t.add_argument('--epochs', type=int, default=30)
    t.add_argument('--batch-size', type=int, default=4096)
    t.add_argument('--width', type=int, default=256)
    t.add_argument('--depth', type=int, default=4)
    t.add_argument('--lr', type=float, default=.0005)
    t.add_argument('--eikonal-weight', type=float, default=.01)
    t.add_argument('--sign-weight', type=float, default=.02)
    t.add_argument('--steps-per-epoch', type=int, default=0, help='0 = full pass over training queries')
    t.add_argument('--val-queries', type=int, default=200000)
    t.add_argument('--checkpoint-steps', type=int, default=500)
    e = sub.add_parser('evaluate')
    e.add_argument('--data', required=True)
    e.add_argument('--model', required=True)
    e.add_argument('--output', required=True)
    e.add_argument('--split', choices=['val', 'test'], default='test')
    e.add_argument('--max-queries', type=int, default=0, help='0 = all held-out queries')
    v = sub.add_parser('plot')
    v.add_argument('--model', required=True)
    v.add_argument('--output', required=True)
    b = sub.add_parser('benchmark', help='Evaluate ALL queries in a fresh dataset, checking original pose overlap')
    b.add_argument('--data', required=True)
    b.add_argument('--reference-data', required=True)
    b.add_argument('--model', required=True)
    b.add_argument('--output', required=True)
    b.add_argument('--batch-size', type=int, default=8192)
    for parser in (s, t):
        parser.add_argument('--resume', action='store_true')
        parser.add_argument('--seed', type=int, default=714)
    for parser in (s, t, e, v, b):
        parser.add_argument('--device', default='cuda:0')
        parser.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    folder = Path(args.output)
    # evaluate --output is a JSON file; sample/train output is a directory.
    lock_folder = folder.parent if args.command in ('evaluate', 'benchmark') else folder
    lock_folder.mkdir(parents=True, exist_ok=True)
    with (lock_folder / '.wuji2_sdf.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            p.error('Another SDF job is writing this output directory')
        if args.command == 'sample':
            from .data import sample
            sample(args)
        elif args.command == 'train':
            if args.checkpoint_steps <= 0:
                p.error('--checkpoint-steps must be positive')
            from .train import train
            report = train(args)
            print(json.dumps(report), flush=True)
        elif args.command == 'plot':
            from .plot import plot
            print(plot(args.model, args.output, args.device), flush=True)
        elif args.command == 'benchmark':
            if args.batch_size < 1:
                p.error('--batch-size must be positive')
            from .benchmark import benchmark
            result = benchmark(args)
            print(json.dumps({k: result[k] for k in ('poses', 'queries', 'mae_mm', 'rmse_mm', 'p95_mm', 'p99_mm',
                                                    'near_10mm', 'penetration_false_negative_rate')}), flush=True)
        else:
            from .common import atomic_json, sha256
            from .data import read_manifest
            from .model import load_model
            from .train import evaluate, configuration_response
            m = read_manifest(args.data, verify=True)
            model, info = load_model(args.model, args.device)
            if info.get('dataset_fingerprint') != m['fingerprint']:
                p.error('Checkpoint and dataset fingerprints differ')
            result = evaluate(model, args.data, m, 1 if args.split == 'val' else 2,
                              args.device, max_queries=args.max_queries)
            result.update(dataset_fingerprint=m['fingerprint'], checkpoint_sha256=sha256(args.model),
                          split=args.split, model_epoch=info.get('epoch'), max_queries=args.max_queries or None)
            result['configuration_response'] = configuration_response(model, args.data, m, args.device)
            atomic_json(folder, result)
            print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
