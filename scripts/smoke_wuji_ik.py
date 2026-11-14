#!/usr/bin/env python3
"""Four production 4x1024 models, one optimizer update each; never a full run."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from retarget.learning.wuji_ik.config import ModelConfig, VARIANTS
from retarget.learning.wuji_ik.data import SegmentDataset, collate_segments, write_json
from retarget.learning.wuji_ik.metrics import autocast_context, move_batch
from retarget.learning.wuji_ik.models import build_model, flow_training_pair, sample_flow
from retarget.learning.wuji_ik.optim import make_optimizers


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--max-frames',type=int,default=8)
    p.add_argument('--backend',default='auto',choices=('auto','sdpa','flash','flex'))
    p.add_argument('--memory-limit-gib',type=float,default=4.)
    p.add_argument('--inference-only',action='store_true')
    p.add_argument('--longest-segment',action='store_true',help='Use the complete longest training segment; overrides max-frames')
    p.add_argument('--variants',nargs='+',choices=VARIANTS,default=list(VARIANTS))
    args=p.parse_args()
    torch.set_num_threads(1);device=torch.device(args.device)
    if args.max_frames<1:raise ValueError('max-frames must be positive')
    if device.type=='cuda':
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(args.memory_limit_gib*1024**3/torch.cuda.get_device_properties(device).total_memory,device)
    data=SegmentDataset(args.prepared,'train')
    index=max(range(len(data)),key=lambda i:data.segments[i][1]['length']) if args.longest_segment else 0
    sample=data[index]
    n=len(sample['features']) if args.longest_segment else min(args.max_frames,len(sample['features']))
    sample={k:v[:n] if torch.is_tensor(v) and v.ndim else v for k,v in sample.items()}
    batch=move_batch(collate_segments([sample]),device)
    reports=[]
    for variant in args.variants:
        torch.manual_seed(42)
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        started=time.perf_counter();config=ModelConfig(backend=args.backend)
        model=build_model(variant,config).to(device)
        optimizers=[] if args.inference_only else make_optimizers(model)[0]
        with torch.set_grad_enabled(not args.inference_only),autocast_context(device):
            if variant=='frame_flow':
                noisy,times,target=flow_training_pair(batch['target'])
                prediction=model(batch['features'],batch['side'],batch['lengths'],noisy_q=noisy,flow_time=times)
            else:
                target=batch['target'];prediction=model(batch['features'],batch['side'],batch['lengths'])
            loss=(prediction.float()-target).square().mean()
        finite=None
        if not args.inference_only:
            loss.backward()
            finite=all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            if not finite or not torch.isfinite(loss):raise RuntimeError('Nonfinite production-model gradient/loss')
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            for optimizer in optimizers:optimizer.step()
        model.eval()
        with torch.no_grad(),autocast_context(device):
            result=sample_flow(model,batch['features'],batch['side'],batch['lengths']) if variant=='frame_flow' else model(batch['features'],batch['side'],batch['lengths'])
        if not torch.isfinite(result).all() or result.abs().max()>1.00001:raise RuntimeError('Invalid normalized inference result')
        if device.type=='cuda':torch.cuda.synchronize(device)
        report=dict(variant=variant,parameters=sum(p.numel() for p in model.parameters()),model_config=config.to_dict(),
            optimizer_updates=0 if args.inference_only else 1,input_frames=n,output_shape=list(result.shape),loss=float(loss.detach()),
            finite_gradients=finite,attention_backends=model.core.backend_report(),seconds=time.perf_counter()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0)
        reports.append(report);print(json.dumps(report),flush=True)
        del model,optimizers,prediction,loss,result,target
        gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    write_json(args.output,dict(passed=True,torch_version=torch.__version__,device=str(device),
        data_signature=data.metadata['signature'],scope=('whole longest training segment' if args.longest_segment else 'short prefix')+
            ('; inference only' if args.inference_only else '; one optimizer update per variant')+'; production width, not full training',reports=reports))


if __name__=='__main__':main()
