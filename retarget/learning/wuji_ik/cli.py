import argparse
from dataclasses import fields
import json
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from .config import ModelConfig, VARIANTS
from .data import SegmentDataset, prepare, write_json
from .features import wrist_emf_features, wrist_tx_transform
from .inference import predict_wrist
from .metrics import evaluate_model
from .training import TrainConfig, distributed_context, load_inference, train
from .splits import stratified_validation


def parser():
    p = argparse.ArgumentParser(description='Wrist EMF-only Wuji2 SDK distillation; parallel full-segment predictions')
    sub = p.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare'); prep.add_argument('--source', required=True); prep.add_argument('--output', required=True)
    prep.add_argument('--right-urdf', required=True); prep.add_argument('--left-urdf', required=True)
    prep.add_argument('--scratch-dir', help='Stage one verified HDF5 shard at a time on fast local storage')
    split=sub.add_parser('resplit');split.add_argument('--prepared',required=True);split.add_argument('--output',required=True)
    split.add_argument('--fraction',type=float,default=.05);split.add_argument('--seed',type=int,default=42)
    tr = sub.add_parser('train'); tr.add_argument('--prepared', required=True); tr.add_argument('--output', required=True)
    tr.add_argument('--variant', choices=VARIANTS, required=True); tr.add_argument('--resume')
    tr.add_argument('--initialize-from', help='Load matching model weights only; start a separate objective/run with fresh optimizers')
    tr.add_argument('--initialize-optimizer', action='store_true', help='Also retain optimizer moments from the same objective, using the new learning-rate schedule')
    tr.add_argument('--extend-epochs', action='store_true', help='Extend a resumed horizon with cosine decay from its saved learning rates')
    tr.add_argument('--backend', choices=('auto','sdpa','flash','flex'), default='auto')
    tr.add_argument('--checkpoint-blocks', action='store_true')
    defaults = TrainConfig()
    for f in fields(TrainConfig):
        if f.name in ('variant', 'bf16'): continue
        value = getattr(defaults, f.name)
        tr.add_argument('--'+f.name.replace('_','-'), type=type(value), default=value)
    tr.add_argument('--fp32', action='store_true', help='Disable CUDA BF16 autocast')
    ev = sub.add_parser('evaluate'); ev.add_argument('--prepared', required=True); ev.add_argument('--checkpoint', required=True)
    ev.add_argument('--output', required=True); ev.add_argument('--split', choices=('val','test'), default='test')
    ev.add_argument('--device', default='auto'); ev.add_argument('--backend', choices=('auto','sdpa','flash','flex'))
    ev.add_argument('--batch-size', type=int, default=1); ev.add_argument('--workers', type=int, default=4)
    ev.add_argument('--flow-steps', type=int, nargs='+', default=[1,4,8,16]); ev.add_argument('--seed', type=int, default=42)
    ev.add_argument('--independent-noise', action='store_true'); ev.add_argument('--fp32', action='store_true')
    pr = sub.add_parser('predict'); pr.add_argument('--checkpoint', required=True); pr.add_argument('--input', required=True)
    pr.add_argument('--output', required=True); pr.add_argument('--side', choices=('right','left'), required=True)
    pr.add_argument('--device', default='auto'); pr.add_argument('--backend', choices=('auto','sdpa','flash','flex'))
    pr.add_argument('--flow-steps', type=int, default=8); pr.add_argument('--seed', type=int, default=42)
    pr.add_argument('--independent-noise', action='store_true'); pr.add_argument('--normalized-only', action='store_true')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == 'prepare':
            result = prepare(args.source, args.output, args.right_urdf, args.left_urdf, scratch=args.scratch_dir)
            print(json.dumps({k: result[k] for k in ('signature','rows','segments','split_rows')}, indent=2))
        elif args.command == 'resplit':
            report=stratified_validation(args.prepared,args.output,args.fraction,args.seed)
            print(json.dumps({k:v for k,v in report.items() if k!='validation_groups'},indent=2))
        elif args.command == 'train':
            names = {f.name for f in fields(TrainConfig)}
            cfg = TrainConfig(**{k:v for k,v in vars(args).items() if k in names}, bf16=not args.fp32)
            train(args.prepared, args.output, cfg, ModelConfig(backend=args.backend, checkpoint_blocks=args.checkpoint_blocks), args.resume,
                  extend_epochs=args.extend_epochs, initialize_from=args.initialize_from,
                  initialize_optimizer=args.initialize_optimizer)
        elif args.command == 'evaluate':
            device, rank, world = distributed_context(args.device)
            model, normalizer, state = load_inference(args.checkpoint, device, args.backend)
            dataset = SegmentDataset(args.prepared, args.split)
            if dataset.metadata['signature'] != state['data_signature']: raise ValueError('Evaluation data signature mismatch')
            reports = {}
            steps = args.flow_steps if state['variant'] == 'frame_flow' else [8]
            for n in steps:
                report = evaluate_model(model, dataset, normalizer, state['variant'], device,
                    batch_size=args.batch_size, workers=args.workers, bf16=not args.fp32, flow_steps=n, seed=args.seed,
                    shared_noise=not args.independent_noise, rank=rank, world_size=world)
                reports[str(n) if state['variant']=='frame_flow' else 'regression'] = report
            if rank == 0:
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                write_json(args.output, dict(split=args.split, data_signature=state['data_signature'], reports=reports))
                print(json.dumps(reports, indent=2))
        else:
            device = torch.device('cuda:0' if args.device == 'auto' and torch.cuda.is_available() else
                                  ('cpu' if args.device == 'auto' else args.device))
            model, normalizer, state = load_inference(args.checkpoint, device, args.backend)
            with np.load(args.input, allow_pickle=False) as a:
                # These keys explicitly denote wrist-frame data, not the source cache's TX-frame fields.
                result = predict_wrist(model, normalizer, state['variant'], a['position_wrist'], a['orientation_xyzw_wrist'],
                    side=args.side, lengths=a['lengths'] if 'lengths' in a else None, flow_steps=args.flow_steps,
                    seed=args.seed, shared_noise=not args.independent_noise, return_radians=not args.normalized_only)
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.output, **result, variant=np.asarray(state['variant']), data_signature=np.asarray(state['data_signature']))
            print('Saved', args.output, 'backends:', model.core.backend_report())
    finally:
        if dist.is_initialized(): dist.destroy_process_group()
