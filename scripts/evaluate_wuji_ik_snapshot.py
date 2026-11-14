#!/usr/bin/env python3
"""Evaluate a fixed checkpoint on the complete held-out test set with bounded GPU memory."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from retarget.learning.wuji_ik.data import SegmentDataset, write_json
from retarget.learning.wuji_ik.metrics import evaluate_model
from retarget.learning.wuji_ik.training import load_inference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--memory-limit-gib', type=float, default=8.)
    parser.add_argument('--backend', choices=('sdpa', 'auto', 'flex'), default='sdpa')
    parser.add_argument('--selection', choices=('best_validation_checkpoint', 'latest_checkpoint',
                        'latest_complete_epoch_checkpoint'), default='best_validation_checkpoint')
    args = parser.parse_args()
    if args.memory_limit_gib <= 0:
        raise ValueError('Memory limit must be positive')
    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(
        args.memory_limit_gib*1024**3/torch.cuda.get_device_properties(device).total_memory, device)
    started = time.perf_counter()
    model, normalizer, state = load_inference(args.checkpoint, device, args.backend)
    if args.selection == 'latest_complete_epoch_checkpoint':
        positions = state.get('next_batch_by_rank', [state['next_batch']])
        if any(position != 0 for position in positions):
            raise ValueError('Selected checkpoint is not at a complete epoch boundary')
    print(f"Loaded {state['variant']} at step {state['step']} for a full test evaluation", flush=True)
    dataset = SegmentDataset(args.prepared, 'test')
    if dataset.metadata['signature'] != state['data_signature']:
        raise ValueError('Test data/checkpoint signature mismatch')
    report = evaluate_model(model, dataset, normalizer, state['variant'], device,
                            batch_size=1, workers=0, bf16=True, seed=42)
    expected = dataset.metadata['split_rows']['test']
    if report['frames'] != expected:
        raise RuntimeError(f"Incomplete test evaluation: {report['frames']} != {expected}")
    result = dict(split='test', selection=args.selection, training_complete=state['step'] >= state['total_steps'],
        checkpoint_step=state['step'], checkpoint_epoch=state['epoch'], checkpoint_path=str(Path(args.checkpoint).resolve()),
        data_signature=state['data_signature'], evaluated_at=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.perf_counter()-started, report=report)
    write_json(args.output, result)
    print('Saved full test snapshot report to '+args.output, flush=True)
    print(report['metrics']['overall'], flush=True)


if __name__ == '__main__':
    main()
