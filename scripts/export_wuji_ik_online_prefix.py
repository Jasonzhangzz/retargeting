#!/usr/bin/env python3
"""Measure actual growing-prefix causal inference, without joint feedback."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from retarget.learning.wuji_ik.data import SegmentDataset, collate_segments, write_json
from retarget.learning.wuji_ik.metrics import autocast_context, move_batch
from retarget.learning.wuji_ik.training import load_inference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--indices', default='273,478,74,447')
    args = parser.parse_args()
    root = Path(args.output)
    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(device).total_memory, device)
    model, normalizer, state = load_inference(args.checkpoint, device, backend='sdpa')
    dataset = SegmentDataset(args.prepared, 'test')
    if state['variant'] != 'temporal_causal' or dataset.metadata['signature'] != state['data_signature']:
        raise ValueError('Expected matching causal model and test data')
    indices = [int(x) for x in args.indices.split(',')]
    predictions = []
    offsets = [0]
    started = time.perf_counter()
    with torch.inference_mode():
        for index in indices:
            sample = dataset[index]
            batch = move_batch(collate_segments([sample]), device)
            sequence = []
            for length in range(1, len(sample['features'])+1):
                lengths = torch.tensor([length], device=device)
                with autocast_context(device):
                    result = model(batch['features'][:, :length], batch['side'], lengths)
                q = normalizer.denormalize(result[:, -1:].float(), batch['side'])[0, 0]
                sequence.append(q.cpu().numpy())
                if length % 100 == 0:
                    print(f'test index {index}: prefix {length}/{len(sample["features"])}', flush=True)
            prediction = np.asarray(sequence)
            if not np.isfinite(prediction).all():
                raise ValueError('Nonfinite prefix prediction')
            predictions.append(prediction)
            offsets.append(offsets[-1]+len(prediction))
            print(f'Finished test index {index}, {len(prediction)} actual prefix calls', flush=True)
    np.savez_compressed(root/'online_prefix.npz', q_pred_rad=np.concatenate(predictions),
                        offsets=np.asarray(offsets), indices=np.asarray(indices))
    write_json(root/'online_prefix_report.json', dict(checkpoint_step=state['step'], indices=indices,
        frames=offsets[-1], seconds=time.perf_counter()-started,
        protocol='Each frame uses an actual BF16 forward call on EMF history [0:t]; no future inputs, no q feedback.'))


if __name__ == '__main__':
    main()
