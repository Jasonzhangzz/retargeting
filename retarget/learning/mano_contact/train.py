"""Two-stage training, complete EMA checkpoints, resume and evaluation."""
from pathlib import Path
import json
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
import torch.nn.functional as F
from .model import PoseAutoencoder, PoseEncoder, ContactModel, ModelEMA
from .data import PoseContactDataset, collate_windows, train_statistics, write_json
from .rotations import pose_features
from .losses import reconstruction_l1, supervised_contrastive, HACOLoss
from .metrics import contact_metrics, calibrated_threshold


def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)


def choose_device(requested):
    if requested=='auto':
        # CPU default avoids silently sharing an occupied GPU.
        return torch.device('cpu')
    return torch.device(requested)


def batches_to_device(groups, device):
    return [{k:v.to(device) if torch.is_tensor(v) else v for k,v in g.items()} for g in groups]


def stage1_loss(model, groups, noise=True, config=None):
    config=config or {}
    recon=0.; outputs=[];labels=[];known=[];frame_logits=[];size=sum(len(g['pose51']) for g in groups)
    for g in groups:
        clean=g['pose51'];x=clean.clone()
        if noise:
            x[...,:3]+=torch.randn_like(x[...,:3])*.002
            x[...,3:]+=torch.randn_like(x[...,3:])*.01
        out=model(x)
        error=(out['reconstruction'][...,:3]-pose_features(clean)[...,:3]).abs().mean((1,2))
        error+=(out['reconstruction'][...,3:]-pose_features(clean)[...,3:]).abs().mean((1,2))
        recon+=(error*g['weight']).sum()/size
        outputs.append(out['projection']);labels.append(g['frame']);known.append(g['valid']);frame_logits.append(out['frame_contact_logit'])
    z=torch.cat(outputs);y=torch.cat(labels).float();valid=torch.cat(known);logits=torch.cat(frame_logits).flatten()
    contrast=supervised_contrastive(z,y,valid,config.get('temperature',.1))
    frame=F.binary_cross_entropy_with_logits(logits[valid],y[valid]) if valid.any() else logits.sum()*0
    return dict(total=recon+config.get('supcon_weight',.1)*contrast+config.get('frame_weight',.1)*frame,reconstruction=recon,supcon=contrast,frame=frame)


def stage2_forward(model, groups):
    predictions=[model(g['pose51']) for g in groups]
    output={k:torch.cat([p[k] for p in predictions]) for k in predictions[0]}
    return output,torch.cat([g['contact'] for g in groups]),torch.cat([g['valid'] for g in groups])


def make_sampler(dataset, stage, samples, seed, temporal_fraction=.75):
    # 75% real temporal, 25% static. Stage 2 keeps natural label proportions;
    # stage 1 compensates with real examples to balance known frame states.
    counts={}
    strata=[]
    for row in dataset.rows:
        _,_,_,known,label,temporal=row
        key=(temporal,known,label if known else False)
        strata.append(key);counts[key]=counts.get(key,0)+1
    masses={}
    for key in counts:
        temporal,known,label=key
        domain=temporal_fraction if temporal else 1-temporal_fraction
        siblings=sum(k[0]==temporal and k[1]==known for k in counts)
        visibility=.8 if known else .2
        present={k[1] for k in counts if k[0]==temporal}
        visibility/=sum(.8 if state else .2 for state in present)
        proportion=1/max(siblings,1)
        if known and (stage==2 or not temporal):
            proportion=counts[key]/sum(v for k,v in counts.items() if k[0]==temporal and k[1])
        masses[key]=domain*visibility*proportion
    if stage==1:
        # Balance all labelled states while retaining the real/static domain
        # ratio. Static grasps can be positive-only, so real negatives must
        # compensate instead of inventing static negative labels.
        known_total=sum(v for k,v in masses.items() if k[1])
        flexible=[True] if (True,True,False) in counts and (True,True,True) in counts else []
        fixed_positive=sum(v for k,v in masses.items() if k[1] and k[2] and k[0] not in flexible)
        flexible_mass=sum(v for k,v in masses.items() if k[1] and k[0] in flexible)
        fraction=min(max((known_total/2-fixed_positive)/max(flexible_mass,1e-12),0.),1.)
        for domain in flexible:
            total=sum(v for k,v in masses.items() if k[0]==domain and k[1])
            masses[(domain,True,True)]=total*fraction
            masses[(domain,True,False)]=total*(1-fraction)
    weights=[masses[key]/counts[key] for key in strata]
    return WeightedRandomSampler(torch.tensor(weights,dtype=torch.double),samples,replacement=True,
                                 generator=torch.Generator().manual_seed(seed))


def save_checkpoint(path, model, optimizer, stage, step, ema=None, **metadata):
    state=dict(stage=stage,step=step,model=model.state_dict(),optimizer=optimizer.state_dict(),
               encoder_config=model.encoder.config,metadata=metadata,rng=torch.get_rng_state(),
               numpy_rng=np.random.get_state(),python_rng=random.getstate())
    if ema is not None:state.update(ema=ema.module.state_dict(),ema_updates=ema.updates,ema_decay=ema.decay)
    if torch.cuda.is_available() and next(model.parameters()).is_cuda:state['cuda_rng']=torch.cuda.get_rng_state_all()
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp')
    torch.save(state,temp);temp.replace(path)


def load_model(checkpoint, assets, device='cpu', ema=True):
    state=torch.load(checkpoint,map_location=device,weights_only=False)
    encoder=PoseEncoder(**state['encoder_config'])
    model=PoseAutoencoder(encoder) if state['stage']==1 else ContactModel(assets,encoder)
    model.load_state_dict(state['ema'] if ema and 'ema' in state else state['model'])
    return model.to(device).eval(),state


@torch.no_grad()
def evaluate(model,dataset,device='cpu',batch_size=32,limit=0,threshold=.52):
    model.eval();rows=Subset(dataset,range(min(limit,len(dataset)))) if limit else dataset
    probs=[];labels=[];frame=[];latencies=[];domains=[]
    for grouped in DataLoader(rows,batch_size=batch_size,collate_fn=collate_windows):
        groups=batches_to_device(grouped,device)
        if torch.device(device).type=='cuda':torch.cuda.synchronize()
        start=time.perf_counter();out,y,valid=stage2_forward(model,groups)
        if torch.device(device).type=='cuda':torch.cuda.synchronize()
        latencies.append((time.perf_counter()-start)/len(y)*1000)
        probs.append(out['contact_out'][valid].sigmoid().cpu().numpy());labels.append(y[valid].cpu().numpy());frame.append(out['frame_contact_logit'][valid].sigmoid().cpu().numpy())
        names=[name for g in grouped for name in g['dataset']]
        domains.extend(name for name,keep in zip(names,valid.cpu().tolist()) if keep)
    if not probs or not sum(len(p) for p in probs):raise ValueError('No labelled evaluation frames')
    p=np.concatenate(probs);y=np.concatenate(labels);f=np.concatenate(frame)
    report=contact_metrics(p,y,f,threshold);report['latency_ms_per_sample']=float(np.median(latencies))
    domains=np.asarray(domains,dtype='U32')
    report['by_dataset']={name:contact_metrics(p[domains==name],y[domains==name],f[domains==name],threshold) for name in np.unique(domains)}
    return report,dict(probability=p,target=y,frame_probability=f,dataset=domains)


def run_train(args):
    cfg=json.loads(Path(args.config).read_text()) if args.config else {}
    stage_cfg=cfg.get('stage'+str(args.stage),{})
    args.seed=args.seed if args.seed is not None else cfg.get('seed',42)
    args.batch_size=args.batch_size or cfg.get('batch_size',32)
    args.steps=args.steps or stage_cfg.get('steps',100000 if args.stage==1 else 50000)
    if min(args.steps,args.batch_size,args.checkpoint_every,args.threads)<1:raise ValueError('Training budgets and batch sizes must be positive')
    seed_everything(args.seed);torch.set_num_threads(args.threads);device=choose_device(args.device)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    import h5py
    with h5py.File(args.corpus,'r') as f:
        if f.attrs.get('preparation_state')!='complete':raise ValueError('Finalize the corpus before training; it is incomplete or has changed')
        corpus_fingerprint=f.attrs.get('corpus_fingerprint','')
    dataset=PoseContactDataset(args.corpus,'train',args.stage,preload=getattr(args,'preload',False))
    if not len(dataset):raise ValueError('No train windows')
    states={r[4] for r in dataset.rows if r[3] and r[5]}
    if states!={False,True}:raise ValueError('Training requires labelled temporal contact and noncontact examples')
    if args.stage==1:model=PoseAutoencoder(PoseEncoder(**cfg.get('encoder',{})))
    else:
        if not args.stage1:raise ValueError('Stage 2 requires --stage1 checkpoint')
        ae,_=load_model(args.stage1,args.assets)
        if not isinstance(ae,PoseAutoencoder):raise ValueError('--stage1 is not an autoencoder checkpoint')
        model=ContactModel.from_autoencoder(ae,args.assets)
    model=model.to(device)
    opt=cfg.get('optimizer',{});weight_decay=opt.get('weight_decay',.01);betas=tuple(opt.get('betas',[.9,.95]))
    if args.stage==1:optimizer=torch.optim.AdamW(model.parameters(),lr=stage_cfg.get('lr',1e-4),weight_decay=weight_decay,betas=betas)
    else:
        enc=list(model.encoder.parameters());other=[p for n,p in model.named_parameters() if not n.startswith('encoder.')]
        optimizer=torch.optim.AdamW([dict(params=enc,lr=stage_cfg.get('encoder_lr',1e-5)),dict(params=other,lr=stage_cfg.get('decoder_lr',1e-4))],weight_decay=weight_decay,betas=betas)
    ema=ModelEMA(model,stage_cfg.get('ema',.999)) if args.stage==2 else None
    loss_fn=None
    if args.stage==2:
        counts=train_statistics(args.corpus)
        loss_fn=HACOLoss(model.decoder,counts,stage_cfg.get('vcb_beta',.9999)).configure_faces(np.load(Path(args.assets)/'faces.npy')).to(device)
    begin=0
    if args.resume:
        state=torch.load(args.resume,map_location=device,weights_only=False)
        if state['stage']!=args.stage:raise ValueError('Resume stage mismatch')
        if state['metadata'].get('config',{})!=cfg:raise ValueError('Resume training configuration changed')
        if state['metadata'].get('corpus_fingerprint',corpus_fingerprint)!=corpus_fingerprint:raise ValueError('Resume corpus changed')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);begin=state['step']
        if ema is not None:
            ema.module.load_state_dict(state['ema']);ema.updates=state['ema_updates']
        torch.set_rng_state(state['rng'].cpu());np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
        if 'cuda_rng' in state:torch.cuda.set_rng_state_all(state['cuda_rng'])
    sampler=make_sampler(dataset,args.stage,args.steps*args.batch_size,args.seed,cfg.get('temporal_fraction',.75))
    if begin:
        from itertools import islice
        # Skip sampled indices, not millions of historical H5 reads.
        sampler=islice(iter(sampler),begin*args.batch_size,None)
    loader=iter(DataLoader(dataset,batch_size=args.batch_size,sampler=sampler,collate_fn=collate_windows,
                           generator=torch.Generator().manual_seed(args.seed)))
    log=output/'train.jsonl';started=time.perf_counter()
    for step in range(begin,args.steps):
        if getattr(args,'stop_file',None) and Path(args.stop_file).exists():
            save_checkpoint(output/'last.pt',model,optimizer,args.stage,step,ema,corpus=str(args.corpus),seed=args.seed,config=cfg,corpus_fingerprint=corpus_fingerprint)
            print(json.dumps(dict(event='stopped_at_checkpoint',stage=args.stage,step=step)),flush=True)
            dataset.close()
            return model,ema
        groups=batches_to_device(next(loader),device);model.train();optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=args.bf16 and device.type=='cuda'):
            if args.stage==1:loss=stage1_loss(model,groups,config=stage_cfg)
            else:
                prediction,target,valid=stage2_forward(model,groups)
                loss=loss_fn(prediction,target,valid,step/max(args.steps-1,1))
        if not torch.isfinite(loss['total']):raise FloatingPointError('Nonfinite training loss')
        loss['total'].backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if ema is not None:ema.update(model)
        if step%20==0 or step+1==args.steps:
            row=dict(step=step+1,examples_seen=(step+1)*args.batch_size,elapsed_seconds=time.perf_counter()-started,**{k:float(v.detach()) for k,v in loss.items()})
            with log.open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
        if (step+1)%args.checkpoint_every==0 or step+1==args.steps:
            save_checkpoint(output/'last.pt',model,optimizer,args.stage,step+1,ema,corpus=str(args.corpus),seed=args.seed,config=cfg,corpus_fingerprint=corpus_fingerprint)
    dataset.close()
    return model,ema


def run_evaluate(args):
    torch.set_num_threads(args.threads);device=choose_device(args.device)
    model,state=load_model(args.checkpoint,args.assets,device,ema=not args.online)
    if state['stage']!=2:raise ValueError('Contact evaluation requires stage 2')
    dataset=PoseContactDataset(args.corpus,args.split,2)
    report,arrays=evaluate(model,dataset,device,args.batch_size,args.limit,args.threshold)
    report.update(split=args.split,weights='online' if args.online else 'ema',checkpoint=str(args.checkpoint))
    if args.split=='val':
        report['calibrated_vertex_threshold']=calibrated_threshold(arrays['probability'],arrays['target'])
        report['calibrated_frame_threshold']=calibrated_threshold(arrays['frame_probability'].ravel(),arrays['target'].any(-1))
        report['calibrated_vertex_metrics']=contact_metrics(arrays['probability'],arrays['target'],arrays['frame_probability'],report['calibrated_vertex_threshold'])
    write_json(args.output,report);np.savez_compressed(Path(args.output).with_suffix('.npz'),**arrays)
    dataset.close();print(json.dumps(report),flush=True)
