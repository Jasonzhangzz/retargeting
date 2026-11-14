#!/usr/bin/env python3
"""Export full test trajectories and audit causal prefix inference, without q feedback."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from retarget.learning.wuji_ik.data import SegmentDataset, collate_segments, write_json
from retarget.learning.wuji_ik.features import SIDES, joint_names
from retarget.learning.wuji_ik.metrics import MetricSums, autocast_context, move_batch
from retarget.learning.wuji_ik.training import load_inference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--memory-limit-gib', type=float, default=8.)
    args = parser.parse_args()
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2); device = torch.device('cuda:0'); torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(
        args.memory_limit_gib*1024**3/torch.cuda.get_device_properties(device).total_memory, device)
    model, normalizer, state = load_inference(args.checkpoint, device, backend='sdpa')
    if state['variant'] != 'temporal_causal':
        raise ValueError('This diagnostic is for the causal temporal model')
    dataset = SegmentDataset(args.prepared, 'test')
    if dataset.metadata['signature'] != state['data_signature']:
        raise ValueError('Dataset/checkpoint signature mismatch')
    print(f"Loaded causal checkpoint step {state['step']}; evaluating {len(dataset)} complete test segments", flush=True)
    predictions = []; targets = []; frame_ids = []; offsets = [0]; segments = []; metrics = MetricSums()
    started = time.perf_counter()
    with torch.no_grad():
        for index in range(len(dataset)):
            sample = dataset[index]; batch = move_batch(collate_segments([sample]), device)
            with autocast_context(device):
                output = model(batch['features'], batch['side'], batch['lengths'])
            pred = normalizer.denormalize(output.float(), batch['side'])[0].cpu().numpy()
            target = normalizer.denormalize(batch['target'].float(), batch['side'])[0].cpu().numpy()
            if not np.isfinite(pred).all() or not np.all(np.diff(sample['frames'].numpy()) == 1):
                raise ValueError('Nonfinite prediction or a discontinuity inside a segment')
            predictions.append(pred); targets.append(target); frame_ids.append(sample['frames'].numpy())
            offsets.append(offsets[-1]+len(pred))
            source = dict(dataset.segments[index][1])
            source.update(segment_id=sample['segment_id'], array_offset=offsets[-2],
                          side_name=SIDES[sample['side']], joint_names=joint_names(SIDES[sample['side']]))
            segments.append(source)
            metrics.add(output, batch, normalizer, 0.)
            if (index+1)%100 == 0:
                print(f"Exported {index+1}/{len(dataset)} segments, {offsets[-1]} frames", flush=True)
        # Audit that full-segment causal inference agrees with actual prefix calls.
        longest = sorted(range(len(dataset)), key=lambda i: len(predictions[i]), reverse=True)[:2]
        audit = []
        for index in longest:
            sample = dataset[index]; n = len(predictions[index])
            for length in sorted(set([1, max(1,n//4), max(1,n//2), max(1,3*n//4)])):
                prefix = dict(sample)
                for key in ('features', 'target', 'frames'): prefix[key] = sample[key][:length]
                batch = move_batch(collate_segments([prefix]), device)
                with autocast_context(device):
                    output = model(batch['features'], batch['side'], batch['lengths'])
                last = normalizer.denormalize(output.float(), batch['side'])[0,-1].cpu().numpy()
                audit.append(dict(segment_id=sample['segment_id'], prefix_frames=length,
                    max_abs_difference_rad=float(np.max(np.abs(last-predictions[index][length-1])))))
    if offsets[-1] != dataset.metadata['split_rows']['test']:
        raise ValueError('Incomplete test export')
    np.savez_compressed(root/'trajectories.npz', q_pred_rad=np.concatenate(predictions),
        q_sdk_rad=np.concatenate(targets), frame_ids=np.concatenate(frame_ids), offsets=np.asarray(offsets))
    write_json(root/'segments.json', segments)
    report = dict(checkpoint_step=state['step'], checkpoint_epoch=state['epoch'],
        variant=state['variant'], data_signature=state['data_signature'], frames=offsets[-1], segments=len(segments),
        generated_at=datetime.now(timezone.utc).isoformat(), metrics=metrics.report(), prefix_audit=audit,
        backend=model.core.backend_report(), seconds=time.perf_counter()-started,
        protocol='Complete held-out continuous segments, causal EMF-only inference; no q feedback or closed-loop simulation.',
        units='q in radians; time axis is frame index, not seconds because per-source frame rates are not encoded here.')
    write_json(root/'export_report.json', report)
    print('EXPORT_COMPLETE '+str(root), flush=True)


if __name__ == '__main__':
    main()
