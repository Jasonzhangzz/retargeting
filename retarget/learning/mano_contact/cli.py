"""CLI: python -m retarget.learning.mano_contact.cli --help."""
import argparse
from pathlib import Path
import json
import numpy as np
import torch
from .data import write_json, audit_corpus


def main():
    p=argparse.ArgumentParser(description=__doc__)
    commands=p.add_subparsers(dest='command',required=True)
    s=commands.add_parser('snapshot');s.add_argument('--source-root',required=True);s.add_argument('--output',required=True)
    s=commands.add_parser('prepare')
    for name in ['dataset','source-root','assets','output']:s.add_argument('--'+name,required=True)
    s.add_argument('--sources',default='hoi4d,h2o,dexycb,epic,generated,primitives');s.add_argument('--limit',type=int,default=0);s.add_argument('--threads',type=int,default=4);s.add_argument('--workers',type=int,default=8)
    s=commands.add_parser('assets');s.add_argument('--assets',required=True)
    s=commands.add_parser('fetch-regressors');s.add_argument('--assets',required=True);s.add_argument('--cache',required=True);s.add_argument('--workers',type=int,default=8)
    s=commands.add_parser('audit');s.add_argument('--corpus',required=True);s.add_argument('--output',required=True)
    s=commands.add_parser('finalize');s.add_argument('--corpus',required=True);s.add_argument('--output',required=True)
    s=commands.add_parser('merge');s.add_argument('--inputs',nargs='+',required=True);s.add_argument('--output',required=True)
    for command in ['train','evaluate','smoke']:
        s=commands.add_parser(command)
        s.add_argument('--corpus',required=True);s.add_argument('--assets',required=True);s.add_argument('--output',required=True)
        s.add_argument('--device',default='cpu');s.add_argument('--threads',type=int,default=4);s.add_argument('--batch-size',type=int,default=None if command=='train' else 32)
        if command=='train':
            s.add_argument('--stage',type=int,choices=[1,2],required=True);s.add_argument('--stage1');s.add_argument('--resume')
            s.add_argument('--steps',type=int);s.add_argument('--seed',type=int);s.add_argument('--config')
            s.add_argument('--checkpoint-every',type=int,default=1000);s.add_argument('--bf16',action='store_true')
            s.add_argument('--stop-file',help='Save a complete checkpoint at the next optimizer boundary and stop')
            s.add_argument('--preload',action='store_true',help='Cache only model inputs and targets in host RAM')
        elif command=='evaluate':
            s.add_argument('--checkpoint',required=True);s.add_argument('--split',choices=['train','val','test'],default='val')
            s.add_argument('--limit',type=int,default=0);s.add_argument('--threshold',type=float,default=.52);s.add_argument('--online',action='store_true')
        else:
            s.add_argument('--stage1-steps',type=int,default=600);s.add_argument('--stage2-steps',type=int,default=5000)
            s.add_argument('--stop-after-stage1',action='store_true');s.add_argument('--reuse-stage1',action='store_true')
            s.add_argument('--reuse-stage2',action='store_true')
    s=commands.add_parser('infer');s.add_argument('--checkpoint',required=True);s.add_argument('--assets',required=True)
    s.add_argument('--pose-npy',required=True);s.add_argument('--output',required=True);s.add_argument('--device',default='cpu')
    s=commands.add_parser('export');s.add_argument('--checkpoint',required=True);s.add_argument('--output',required=True)
    s=commands.add_parser('report')
    for name in ['corpus','assets','run','output']:s.add_argument('--'+name,required=True)
    s.add_argument('--threads',type=int,default=4)
    a=p.parse_args()
    if a.command=='snapshot':
        from .prepare import snapshot_generated
        value=snapshot_generated(a.source_root,a.output);print(json.dumps(dict(sources=len(value['sources']),created_utc=value['created_utc'])))
    elif a.command=='prepare':
        from .prepare import run_prepare
        run_prepare(a)
    elif a.command=='assets':
        from .geometry import prepare_mano_assets
        prepare_mano_assets(a.assets)
    elif a.command=='fetch-regressors':
        from .fetch import fetch_regressors
        fetch_regressors(a.assets,a.cache,a.workers)
    elif a.command=='audit':
        value=audit_corpus(a.corpus);write_json(a.output,value);print(json.dumps(value))
    elif a.command=='finalize':
        from .data import finalize_corpus
        print(json.dumps(finalize_corpus(a.corpus,a.output)))
    elif a.command=='merge':
        from .data import merge_corpora
        print(json.dumps(merge_corpora(a.inputs,a.output)))
    elif a.command=='train':
        from .train import run_train
        run_train(a)
    elif a.command=='evaluate':
        from .train import run_evaluate
        run_evaluate(a)
    elif a.command=='smoke':
        from .smoke import run_smoke
        run_smoke(a)
    elif a.command=='infer':
        from .train import load_model
        model,state=load_model(a.checkpoint,a.assets,a.device)
        if state['stage']!=2:raise ValueError('Inference requires a stage 2 checkpoint')
        pose=np.load(a.pose_npy,allow_pickle=False)
        if pose.ndim==2:pose=pose[None]
        with torch.inference_mode():out=model(torch.as_tensor(pose,dtype=torch.float32,device=a.device))
        np.savez_compressed(a.output,**{k:v.cpu().numpy() for k,v in out.items()},
                            contact_probability=out['contact_out'].sigmoid().cpu().numpy(),
                            frame_contact_probability=out['frame_contact_logit'].sigmoid().cpu().numpy())
    elif a.command=='export':
        state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
        if state['stage']!=2 or 'ema' not in state:raise ValueError('Export requires a trained EMA contact checkpoint')
        output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
        weights=state['ema'];assets=output/'assets';assets.mkdir(exist_ok=True)
        for name in ['V_regressor_336','V_regressor_84','J_regressor']:
            np.save(assets/(name+'.npy'),weights['decoder.'+name].numpy())
        torch.save(dict(stage=2,model=weights,encoder_config=state['encoder_config'],step=state['step'],
                        metadata=dict(source=str(a.checkpoint),weights='ema',input='pose51[B,T,51]',max_history=8)),output/'model.pt')
    elif a.command=='report':
        from .report import build_report
        build_report(a.corpus,a.assets,a.run,a.output,a.threads)


if __name__=='__main__':main()
