"""Prepare/train/evaluate a pose-only Wuji2 RVQ, or encode/decode poses."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .data import DEFAULT_DATA, ROOT, load, prepare, write_json


def array(path, key):
    value=np.load(path,allow_pickle=False)
    if isinstance(value,np.lib.npyio.NpzFile):
        try:return value[key].copy()
        finally:value.close()
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('prepare');p.add_argument('--output',type=Path,default=DEFAULT_DATA);p.add_argument('--seed',type=int,default=20261118)
    p=commands.add_parser('train');p.add_argument('--data',type=Path,default=DEFAULT_DATA)
    p.add_argument('--output',type=Path,default=ROOT/'output/wuji2_rvq_joint_v1');p.add_argument('--device',default='cuda:0')
    p.add_argument('--seed',type=int,default=20261118);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--warmup-epochs',type=int,default=20);p.add_argument('--rvq-epochs',type=int,default=300)
    p.add_argument('--task',choices=['joint','classification','reconstruction'],default='joint')
    p.add_argument('--reconstruction-weight',type=float,default=1.)
    p.add_argument('--classification-weight',type=float,default=.05)
    p.add_argument('--optimizer',choices=['muon','adamw'],default='muon')
    p.add_argument('--batch-size',type=int,default=0,help='Actual per-GPU batch; 0 uses 31 * samples-per-type')
    p.add_argument('--samples-per-type',type=int,default=16,help='Samples per class per GPU')
    p.add_argument('--steps-per-epoch',type=int,default=0,help='0: one global dataset pass')
    p.add_argument('--resume',action='store_true');p.add_argument('--no-compile',dest='compile',action='store_false')
    p=commands.add_parser('evaluate');p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--data',type=Path,default=DEFAULT_DATA)
    p.add_argument('--split',choices=['train','val','test'],default='test');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--skip-fk',action='store_true')
    for command in ['encode','decode','predict']:
        p=commands.add_parser(command)
        p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--input',type=Path,required=True)
        p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu')
    p=commands.add_parser('export');p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='prepare':
        result=prepare(args.output,args.seed)
        print(json.dumps({k:result[k] for k in ['selected_per_type','split_counts_per_type','primitive_anchors','poses_sha256']}))
    elif args.command=='train':
        if min(args.threads,args.warmup_epochs,args.rvq_epochs,args.samples_per_type)<=0 or args.steps_per_epoch<0:parser.error('Epoch and thread counts must be positive')
        if args.batch_size<0 or (args.batch_size and args.batch_size<31):parser.error('Per-GPU batch must be 0 or at least 31')
        if not (0<args.reconstruction_weight<float('inf') and 0<args.classification_weight<float('inf')):parser.error('Both joint loss weights must be finite and positive')
        if not (args.data/'manifest.json').exists():prepare(args.data,args.seed)
        from .train import train
        train(args)
    elif args.command=='evaluate':
        from .train import load_model
        from .evaluate import evaluate
        torch.set_num_threads(4)
        model,checkpoint=load_model(args.checkpoint,args.device)
        arrays,metadata=load(args.data)
        if model.config.get('joint'):
            from .joint import evaluate_joint
            weights=checkpoint.get('metadata',{}).get('loss_weights',{})
            if not weights:
                saved=checkpoint.get('signature',{}).get('joint_weights') or [1.,.05]
                weights=dict(reconstruction=saved[0],classification=saved[1])
            report,_,_=evaluate_joint(model,arrays,metadata,args.split,reconstruction_weight=weights.get('reconstruction',1.),classification_weight=weights.get('classification',.05),fk=not args.skip_fk)
            keys=['accuracy','macro_f1','mae_deg','rmse_deg','p95_deg']
        elif 'class_names' in model.config:
            from .classifier import evaluate_classifier
            report,_,_=evaluate_classifier(model,arrays,metadata,args.split)
            keys=['accuracy','macro_f1','cross_entropy']
        else:
            report,_,_=evaluate(model,arrays,metadata,args.split,fk=not args.skip_fk);keys=['mae_deg','rmse_deg','p95_deg']
        write_json(args.output,report);print(json.dumps({k:report[k] for k in keys}))
    elif args.command=='export':
        from .train import load_model,export_model
        model,state=load_model(args.checkpoint)
        export_model(model,args.output,state.get('metadata'))
    elif args.command=='predict':
        from .train import load_model
        from .classifier import predict_classes
        torch.set_num_threads(4)
        model,_=load_model(args.checkpoint,args.device)
        if 'class_names' not in model.config:raise ValueError('predict requires a classification checkpoint')
        result=predict_classes(model,array(args.input,'qpos'))
        names=np.asarray(model.config['class_names']);result['grasp_type']=names[result['class_index']]
        args.output.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(args.output,class_names=names,**result)
    elif args.command=='encode':
        from .train import load_model
        model,_=load_model(args.checkpoint,args.device)
        q=array(args.input,'qpos')
        results=[]
        with torch.inference_mode():
            for start in range(0,len(q),4096):
                tokens=model.encode(torch.as_tensor(q[start:start+4096],device=args.device,dtype=torch.float32))
                results.append(tokens.cpu().numpy().astype(np.uint8))
        if not results:raise ValueError('Empty pose batch')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(args.output,tokens=np.concatenate(results),joint_names=np.asarray(model.config['joint_names']))
    else:
        from .model import PoseDecoder
        from .train import load_model
        state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        if 'decoder' in state:
            if state['config'].get('joint'):
                from .joint import JointTokenDecoder
                model=JointTokenDecoder(state['config'])
            else:model=PoseDecoder(state['config'])
            model.load_state_dict(state['decoder']);model=model.to(args.device).eval()
        else:model,_=load_model(args.checkpoint,args.device)
        tokens=array(args.input,'tokens');results=[];class_results=[]
        with torch.inference_mode():
            for start in range(0,len(tokens),4096):
                batch=torch.as_tensor(tokens[start:start+4096],device=args.device)
                if 'decoder' in state:
                    out=model(batch);q=out['qpos'] if isinstance(out,dict) else out
                elif model.config.get('joint'):
                    out=model.decode_outputs(batch);q=out['qpos']
                else:out=None;q=model.decode(batch)
                if isinstance(out,dict) and 'logits' in out:class_results.append(out['logits'].cpu())
                results.append(q.cpu().numpy())
        if not results:raise ValueError('Empty token batch')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        values=dict(qpos=np.concatenate(results),joint_names=np.asarray(model.config['joint_names']))
        if class_results:
            logits=torch.cat(class_results);names=np.asarray(model.config['class_names'])
            values.update(probabilities=logits.softmax(-1).numpy(),class_index=logits.argmax(-1).numpy(),class_names=names,grasp_type=names[logits.argmax(-1).numpy()])
        np.savez_compressed(args.output,**values)


if __name__=='__main__':main()
