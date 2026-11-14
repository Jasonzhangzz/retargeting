#!/usr/bin/env python3
"""Frozen best model: full train/validation errors by source collection."""
import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np
import torch
from retarget.learning.wuji_ik.training import load_inference
from retarget.learning.wuji_ik.metrics import autocast_context


def summarize(error,norm_error):
    return dict(frames=len(error),mae_rad=float(abs(error).mean()),rmse_rad=float(np.sqrt((error*error).mean())),
        max_abs_rad=float(abs(error).max()),normalized_mse=float(np.square(norm_error).mean()))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=a.campaign;out=a.output;out.mkdir(parents=True,exist_ok=False);conf=json.loads((root/'MANIFEST.json').read_text());data=Path(conf['dataset'])
    meta=json.loads((data/'manifest.json').read_text());segments=json.loads((data/'segments.json').read_text())
    torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(6*1024**3/torch.cuda.get_device_properties(0).total_memory,0)
    model,normalizer,state=load_inference(root/'temporal_causal/best.pt','cuda:0','auto');assert state['data_signature']==meta['signature']
    checkpoint=dict(step=state['step'],completed_epochs=state['epoch']);del state;gc.collect()
    features=np.load(data/'features.npy',mmap_mode='r');targets=np.load(data/'q_sdk_rad.npy',mmap_mode='r');qn=np.load(data/'q_normalized.npy',mmap_mode='r')
    pred=np.lib.format.open_memmap(out/'q_pred_rad.npy',mode='w+',dtype=np.float32,shape=targets.shape)
    norm_err=np.lib.format.open_memmap(out/'normalized_error.npy',mode='w+',dtype=np.float32,shape=qn.shape)
    details=[];started=time.perf_counter()
    with torch.no_grad():
        for i,s in enumerate(segments):
            lo=s['offset'];hi=lo+s['length'];x=torch.from_numpy(np.array(features[lo:hi][None],copy=True)).to('cuda:0');side=torch.tensor([s['side']],device='cuda:0');length=torch.tensor([s['length']],device='cuda:0')
            with autocast_context(torch.device('cuda:0'),True):output=model(x,side,length)
            pred[lo:hi]=normalizer.denormalize(output.float(),side)[0].cpu().numpy();norm_err[lo:hi]=output.float()[0].cpu().numpy()-qn[lo:hi]
            item={k:s[k] for k in ['record','offset','length','split','collection','dataset','side']};item.update(summarize(pred[lo:hi].astype(float)-targets[lo:hi],norm_err[lo:hi]));details.append(item)
            if (i+1)%25==0:print('MODEL_SEGMENTS',i+1,'/',len(segments),flush=True)
    pred.flush();norm_err.flush();error=pred.astype(np.float64)-targets
    groups={}
    for split in ['train','val']:
        for group in ['overall']+sorted({s['dataset']+'/'+s['collection'] for s in segments if s['split']==split}):
            matching=[s for s in segments if s['split']==split and (group=='overall' or s['dataset']+'/'+s['collection']==group)]
            ix=np.concatenate([np.arange(s['offset'],s['offset']+s['length']) for s in matching]);groups[split+'/'+group]=summarize(error[ix],norm_err[ix])
    result=dict(complete=True,optimizer_updates=0,frames=len(pred),segments=len(segments),checkpoint=checkpoint,data_signature=meta['signature'],groups=groups,per_segment=details,elapsed_seconds=time.perf_counter()-started,attention_backends=model.core.backend_report())
    (out/'report.json').write_text(json.dumps(result,indent=2));print('MODEL_AUDIT_COMPLETE',json.dumps({k:v for k,v in result.items() if k not in ['groups','per_segment']}),flush=True)
    print('TRAIN',groups['train/overall'],'VAL',groups['val/overall'],flush=True)


if __name__=='__main__':main()
