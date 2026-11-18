"""Reproducible warmup, EMA RVQ training, model selection and exports."""
import hashlib
import json
import math
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
import random
import signal
import time
import numpy as np
import torch
from torch.nn import functional as F
from .data import load, write_json, sha
from .model import PoseRVQ, PoseDecoder, supervised_contrastive
from .evaluate import evaluate
from .optim import make_optimizer

STOP = False

def rank():return dist.get_rank() if dist.is_initialized() else 0
def world():return dist.get_world_size() if dist.is_initialized() else 1
def local_batch(args):return getattr(args,'batch_size',0) or 31*args.samples_per_type
def barrier():
    if dist.is_initialized():dist.barrier()
def primary_json(path,value):
    if rank()==0:write_json(path,value)



class TrainingStopped(Exception):pass


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def rng_state():
    value=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state())
    if torch.cuda.is_available():value['cuda']=torch.cuda.get_rng_state_all()
    return value


def restore_rng(value):
    random.setstate(value['python']);np.random.set_state(value['numpy']);torch.set_rng_state(value['torch'].cpu())
    if 'cuda' in value and torch.cuda.is_available():torch.cuda.set_rng_state_all([v.cpu() for v in value['cuda']])


def atomic_save(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_suffix(path.suffix+'.pending');torch.save(value,pending);pending.replace(path)


def load_model(path, device='cpu'):
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    if checkpoint['config'].get('joint'):
        from .joint import PoseRVQJoint
        model=PoseRVQJoint(**checkpoint['config'])
    elif 'class_names' in checkpoint['config']:
        from .classifier import PoseRVQClassifier
        model=PoseRVQClassifier(**checkpoint['config'])
    else:model=PoseRVQ(**checkpoint['config'])
    model.load_state_dict(checkpoint['model'])
    return model.to(device).eval(),checkpoint


def export_model(model, output, metadata=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    config=model.config
    atomic_save(output/'model.pt',dict(config=config,model=model.state_dict(),metadata=metadata or {}))
    if config.get('joint'):
        from .joint import JointTokenDecoder
        decoder=JointTokenDecoder.from_model(model)
        atomic_save(output/'decoder.pt',dict(config=config,decoder=decoder.state_dict(),metadata=metadata or {},task='joint'))
        atomic_save(output/'joint.pt',dict(config=config,model=model.state_dict(),metadata=metadata or {},task='joint'))
        primary_json(output/'interface.json',dict(task='joint',input='qpos[B,20], radians',tokens='uint8[B,4]',outputs=['qpos[B,20], radians','probabilities[B,31]'],class_names=config['class_names'],joint_names=config['joint_names'],lower=config['lower'],upper=config['upper'],shared_decoder_trunk=True,encoder_required_for_token_decoder=False,class_labels_required=False))
        return
    if 'class_names' in config:
        atomic_save(output/'classifier.pt',dict(config=config,model=model.state_dict(),metadata=metadata or {},task='classification'))
        primary_json(output/'interface.json',dict(task='classification',input='qpos[B,20], radians',output='probabilities[B,31]',tokens='uint8[B,4]',class_names=config['class_names'],joint_names=config['joint_names'],lower=config['lower'],upper=config['upper'],decoder=False))
        return
    decoder=PoseDecoder.from_model(model)
    atomic_save(output/'decoder.pt',dict(config=config,decoder=decoder.state_dict(),metadata=metadata or {}))
    primary_json(output/'interface.json',dict(input='qpos[B,20], radians',tokens=f'int[B,{config["levels"]}]',
        output='qpos[B,20], radians',joint_names=config['joint_names'],lower=config['lower'],upper=config['upper'],
        token_bits=config['levels']*math.log2(config['entries']),class_labels_required=False,
        encoder_required_for_decoder=False,physical_grasp_success_guaranteed=False))


class BalancedSampler:
    def __init__(self, arrays, device, seed, per_type=16, steps=0, batch_size=0):
        ids=np.flatnonzero(arrays['split']==0)
        self.q=torch.as_tensor(np.concatenate([arrays['qpos'][ids],arrays['anchor_qpos']]),device=device)
        labels=np.concatenate([arrays['label'][ids],arrays['anchor_label']])
        self.labels=torch.as_tensor(labels,device=device,dtype=torch.long)
        self.regular=[np.flatnonzero(arrays['label'][ids]==i) for i in range(31)]
        self.anchors=[len(ids)+np.flatnonzero(arrays['anchor_label']==i) for i in range(31)]
        self.rng=np.random.default_rng(seed+rank())
        self.per_type=per_type
        self.batch_size=batch_size or 31*per_type
        if self.batch_size<31:raise ValueError('Batch must contain at least one sample per class')
        self.steps=steps or math.ceil(len(ids)/(self.batch_size*world()))

    def batch(self):
        result=[]
        counts=np.full(31,self.batch_size//31,dtype=np.int64)
        extra=self.batch_size%31
        if extra:counts[self.rng.permutation(31)[:extra]]+=1
        for i,count in enumerate(counts):
            n=int((self.rng.random(count)<.1).sum()) if len(self.anchors[i]) else 0
            result.extend(self.rng.choice(self.regular[i],int(count)-n,replace=True).tolist())
            if n:result.extend(self.rng.choice(self.anchors[i],n,replace=True).tolist())
        ids=torch.as_tensor(self.rng.permutation(result),device=self.q.device)
        return self.q[ids],self.labels[ids]


@torch.inference_mode()
def validation_rmse(model, q):
    model.eval();error=torch.zeros((),device=q.device)
    continuous=not bool(model.quantizer.initialized)
    for batch in q.split(4096):
        prediction=model(batch,continuous=continuous,update=False)['qpos']
        error+=(prediction-batch).square().sum()
    return float((error/q.numel()).sqrt()*180/math.pi)


def fit_phase(model, arrays, metadata, folder, epochs, weight, args, continuous=False):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    joint=bool(model.config.get('joint'))
    classification='class_names' in model.config and not joint
    task='joint' if joint else ('classification' if classification else 'reconstruction')
    signature=dict(dataset=metadata['poses_sha256'],seed=args.seed,epochs=epochs,semantic_weight=weight,
        continuous=continuous,task=task,joint_weights=[args.reconstruction_weight,args.classification_weight] if joint else None,optimizer=args.optimizer,muon_adjust_lr="match_rms_adamw" if args.optimizer=="muon" else None,lr=3e-4,weight_decay=1e-4,batch=local_batch(args)*world(),world_size=world(),steps_per_epoch=args.steps_per_epoch,anchors=.1,config=model.config)
    fingerprint=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    optimizer=make_optimizer(model,args.optimizer)
    if rank()==0 and hasattr(optimizer,"assignment"):primary_json(folder/"optimizer.json",dict(kind=args.optimizer,lr=3e-4,weight_decay=1e-4,assignment=optimizer.assignment))
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs,eta_min=3e-6)
    sampler=BalancedSampler(arrays,args.device,args.seed,args.samples_per_type,args.steps_per_epoch,getattr(args,'batch_size',0))
    validation=torch.as_tensor(arrays['qpos'][arrays['split']==1],device=args.device)
    epoch=step=0;best=float('inf');stale=0;global_step=0;history=[]
    losses=torch.zeros(5,device=args.device);batches=0
    latest=folder/'last.pt'
    if latest.exists():
        if not args.resume:raise ValueError(f'{latest} exists; pass --resume')
        saved=torch.load(latest,map_location='cpu',weights_only=False)
        if saved['fingerprint']!=fingerprint:raise ValueError('Resume dataset/training configuration changed')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        epoch,step,best,stale,global_step=(saved[k] for k in ['epoch','step','best','stale','global_step'])
        local=saved['rank_states'][rank()] if 'rank_states' in saved else saved
        sampler.rng.bit_generator.state=local['sampler_rng'];restore_rng(local['rng']);history=saved['history']
        losses=saved.get('partial_losses',losses).to(args.device);batches=saved.get('partial_batches',0)
        if saved.get('complete'):return load_model(folder/'best.pt',args.device)[0]
    else:seed_all(args.seed)
    # Compile only MLP calls, leaving state_dict names and EMA operations ordinary PyTorch.
    enc,dec=model.encoder,model.classifier if classification else model.decoder
    if args.compile and str(args.device).startswith('cuda'):
        enc=torch.compile(enc,dynamic=False)
        dec=torch.compile(dec,dynamic=False)
    if world()>1:
        enc=DDP(enc,device_ids=[int(os.environ['LOCAL_RANK'])])
        dec=DDP(dec,device_ids=[int(os.environ['LOCAL_RANK'])])
    began=time.monotonic()
    def save(complete=False,path=latest):
        local=dict(sampler_rng=sampler.rng.bit_generator.state,rng=rng_state())
        states=[None]*world()
        if world()>1:dist.all_gather_object(states,local)
        else:states=[local]
        if rank()!=0:return
        atomic_save(path,dict(rank_states=states,config=model.config,model=model.state_dict(),optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),epoch=epoch,step=step,best=best,stale=stale,global_step=global_step,
            fingerprint=fingerprint,signature=signature,sampler_rng=sampler.rng.bit_generator.state,
            rng=rng_state(),history=history,complete=complete,partial_losses=losses.detach().cpu(),partial_batches=batches))
    while epoch<epochs and (continuous or stale<30):
        model.train();tick=time.monotonic()
        if step==0:losses=torch.zeros(5,device=args.device);batches=0
        semantic_weight=weight*min(1.,(epoch+step/sampler.steps)/20)
        for batch_step in range(step,sampler.steps):
            q,y=sampler.batch();x=2*(q-model.lower)/(model.upper-model.lower)-1
            optimizer.zero_grad(set_to_none=True)
            z=enc(x)
            ce=z.sum()*0
            if joint:
                out=None if continuous else model.quantizer(z)
                heads=dec(z if continuous else out['quantized'])
                commit=z.sum()*0 if continuous else out['commitment']
                contrast=z.sum()*0
                primary=F.mse_loss(heads['normalized'],x)
                ce=F.cross_entropy(heads['logits'],y)
                semantic_weight=0.
                loss=args.reconstruction_weight*primary+args.classification_weight*ce+.25*commit
            elif classification:
                out=None if continuous else model.quantizer(z)
                logits=dec(z if continuous else out['quantized'])
                commit=z.sum()*0 if continuous else out['commitment']
                contrast=z.sum()*0
                primary=F.cross_entropy(logits,y);ce=primary
                semantic_weight=0.
                loss=primary+.25*commit
            else:
                if continuous:
                    norm=dec(z).tanh();commit=z.sum()*0;contrast=commit
                else:
                    out=model.quantizer(z)
                    norm=dec(out['quantized']).tanh();commit=out['commitment']
                    if weight and world()>1:
                        from torch.distributed.nn.functional import all_gather
                        prefix=torch.cat(all_gather(out['prefix']))
                        labels=[torch.empty_like(y) for _ in range(world())];dist.all_gather(labels,y)
                        contrast=supervised_contrastive(prefix,torch.cat(labels))
                    else:contrast=supervised_contrastive(out['prefix'],y) if weight else z.sum()*0
                primary=F.mse_loss(norm,x)
                semantic_weight=weight*min(1.,(epoch+(batch_step+1)/sampler.steps)/20)
                loss=primary+.25*commit+semantic_weight*contrast
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training objective')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            with torch.no_grad():
                losses+=torch.stack([loss.detach(),primary.detach(),commit.detach(),contrast.detach(),ce.detach()]);batches+=1
                if not continuous and global_step>0 and global_step%100==0:
                    # Revive truly unused EMA entries from current residuals; RNG is checkpointed.
                    residual=z.detach().float()
                    for level in range(model.quantizer.levels):
                        dead=model.quantizer.counts[level]<.05
                        if dead.any():
                            fresh=residual[torch.randint(len(residual),(int(dead.sum()),),device=q.device)]
                            if world()>1:dist.broadcast(fresh,src=0)
                            model.quantizer.books[level,dead]=fresh
                            model.quantizer.counts[level,dead]=1
                            model.quantizer.sums[level,dead]=fresh
                        ids=model.quantizer.nearest(residual,model.quantizer.books[level])
                        residual=residual-model.quantizer.books[level,ids]
            global_step+=1;step=batch_step+1
            stopped=torch.tensor(int(STOP),device=args.device)
            if world()>1:dist.all_reduce(stopped,op=dist.ReduceOp.MAX)
            if stopped.item():
                save();primary_json(folder/'status.json',dict(state='paused',epoch=epoch,step=step,global_step=global_step))
                raise TrainingStopped()
        if joint:
            from .joint import evaluate_joint
            metrics=evaluate_joint(model,arrays,metadata,'val',continuous,args.reconstruction_weight,args.classification_weight)[0] if rank()==0 else None
            if world()>1:
                shared=[metrics];dist.broadcast_object_list(shared,src=0);metrics=shared[0]
            score=metrics['joint_score']
        elif classification:
            from .classifier import evaluate_classifier
            metrics=evaluate_classifier(model,arrays,metadata,'val',continuous=continuous)[0] if rank()==0 else None
            if world()>1:
                shared=[metrics];dist.broadcast_object_list(shared,src=0);metrics=shared[0]
            score=-metrics['macro_f1']
        else:
            score_tensor=torch.tensor(validation_rmse(model,validation) if rank()==0 else 0.,device=args.device)
            if world()>1:dist.broadcast(score_tensor,src=0)
            score=float(score_tensor)
        if world()>1:dist.all_reduce(losses);losses/=world()
        improved=score<best-1e-6
        best=min(best,score);stale=0 if improved else stale+1
        scheduler.step();epoch+=1;step=0
        means=(losses/max(batches,1)).cpu().tolist()
        record=dict(epoch=epoch,loss=means[0],commitment=means[2],supcon=means[3],semantic_weight=semantic_weight,
            seconds=time.monotonic()-tick,lr=optimizer.param_groups[0]['lr'])
        if joint:
            record.update(task='joint',reconstruction=means[1],cross_entropy=means[4],val_accuracy=metrics['accuracy'],val_macro_f1=metrics['macro_f1'],val_cross_entropy=metrics['cross_entropy'],val_mae_deg=metrics['mae_deg'],val_rmse_deg=metrics['rmse_deg'],val_p95_deg=metrics['p95_deg'],val_normalized_mse=metrics['normalized_mse'],val_joint_score=score,best_val_joint_score=best,reconstruction_weight=args.reconstruction_weight,classification_weight=args.classification_weight)
        elif classification:
            record.update(task='classification',cross_entropy=means[1],val_accuracy=metrics['accuracy'],
                val_macro_f1=metrics['macro_f1'],val_cross_entropy=metrics['cross_entropy'],best_val_macro_f1=-best)
        else:record.update(reconstruction=means[1],val_rmse_deg=score,best_val_rmse_deg=best)
        if str(args.device).startswith('cuda'):record['gpu_peak_allocated_mib']=torch.cuda.max_memory_allocated(args.device)/1024**2
        history.append(record)
        save()
        if improved:save(path=folder/'best.pt')
        primary_json(folder/'history.json',history)
        primary_json(folder/'status.json',dict(state='running',**record,global_step=global_step))
        if rank()==0:print(json.dumps(dict(run=folder.name,world_size=world(),global_batch=local_batch(args)*world(),**record)),flush=True)
    if world()>1:
        flat=torch.cat([t.detach().float().flatten() for t in model.state_dict().values()])
        reference=flat.clone();dist.broadcast(reference,src=0)
        difference=(flat-reference).abs().max();dist.all_reduce(difference,op=dist.ReduceOp.MAX)
        if difference.item()!=0:raise RuntimeError('Distributed model/EMA replicas diverged')
        primary_json(folder/'distributed_verification.json',dict(world_size=world(),max_replica_difference=float(difference),global_batch=local_batch(args)*world()))
    save(complete=True)
    primary_json(folder/'status.json',dict(state='completed',epochs=epoch,selection_metric='joint_score' if joint else ('macro_f1' if classification else 'rmse_deg'),best_value=-best if classification else best,seconds=time.monotonic()-began))
    barrier()
    return load_model(folder/'best.pt',args.device)[0]


def train(args):
    global STOP
    STOP=False
    def stop(*_):
        global STOP
        STOP=True
    previous={sig:signal.signal(sig,stop) for sig in (signal.SIGINT,signal.SIGTERM)}
    distributed=int(os.environ.get('WORLD_SIZE','1'))>1
    if distributed:
        local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local);args.device=f'cuda:{local}'
        dist.init_process_group('nccl')
    try:
        if args.task=='joint':
            from .joint import joint_campaign
            return joint_campaign(args)
        if args.task=='classification':
            from .classifier import classifier_campaign
            return classifier_campaign(args)
        return campaign(args)
    except TrainingStopped:
        primary_json(Path(args.output)/'status.json',dict(state='paused',reason='signal; checkpoint saved at optimizer boundary'))
        return dict(state='paused')
    finally:
        if dist.is_initialized():dist.destroy_process_group()
        for sig,handler in previous.items():signal.signal(sig,handler)


def campaign(args):
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision('high')
    arrays,metadata=load(args.data)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    config=dict(data=str(args.data),data_sha256=metadata['poses_sha256'],seed=args.seed,device=args.device,optimizer=args.optimizer,
        world_size=world(),global_batch=local_batch(args)*world(),per_gpu_batch=local_batch(args),samples_per_type_per_gpu=args.samples_per_type if not getattr(args,'batch_size',0) else None,steps_per_epoch=args.steps_per_epoch,
        warmup_epochs=args.warmup_epochs,rvq_epochs=args.rvq_epochs,semantic_candidates=[0.,.01,.05],
        code_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')})
    primary_json(output/'configuration.json',config)
    primary_json(output/'status.json',dict(state='warmup'))
    initial=output/'initialized.pt'
    if initial.exists():
        if not args.resume:raise ValueError('Output already initialized; pass --resume')
        model,info=load_model(initial,args.device)
        if info['data_sha256']!=metadata['poses_sha256']:raise ValueError('Initialization dataset changed')
        if info.get('optimizer','adamw')!=args.optimizer:raise ValueError('Initialization optimizer changed; start a fresh output directory')
    else:
        seed_all(args.seed)
        model=PoseRVQ(metadata['lower'],metadata['upper'],metadata['joint_names']).to(args.device)
        model=fit_phase(model,arrays,metadata,output/'warmup',args.warmup_epochs,0,args,continuous=True)
        model.eval()
        ids=np.flatnonzero(arrays['split']==0)
        rng=np.random.default_rng(args.seed);ids=rng.choice(ids,min(65536,len(ids)),replace=False)
        latent=[]
        with torch.inference_mode():
            for start in range(0,len(ids),4096):
                q=torch.as_tensor(arrays['qpos'][ids[start:start+4096]],device=args.device)
                latent.append(model.encoder(model.normalize(q)))
        if rank()==0:
            model.quantizer.initialize(torch.cat(latent),seed=args.seed)
            atomic_save(initial,dict(config=model.config,model=model.state_dict(),data_sha256=metadata['poses_sha256'],optimizer=args.optimizer))
        barrier()
        model,_=load_model(initial,args.device)
    reports={}
    def run(weight):
        tag='lambda_'+str(weight).replace('.','p')
        folder=output/tag
        report_path=folder/'validation.json'
        if args.resume and report_path.exists():
            reports[str(weight)]=json.loads(report_path.read_text());return
        current,_=load_model(initial,args.device)
        current=fit_phase(current,arrays,metadata,folder,args.rvq_epochs,weight,args)
        if rank()==0:report,_,_=evaluate(current,arrays,metadata,'val')
        else:report=None
        if world()>1:
            shared=[report];dist.broadcast_object_list(shared,src=0);report=shared[0]
        report.update(weight=weight,checkpoint=str(folder/'best.pt'))
        primary_json(report_path,report);reports[str(weight)]=report
        primary_json(output/'candidate_reports.json',reports)
    for weight in [0.,.01,.05]:
        primary_json(output/'status.json',dict(state='training',semantic_weight=weight))
        run(weight)
    baseline=reports['0.0']
    def eligible(r):
        return (r['weight']>0 and r['rmse_deg']<=baseline['rmse_deg']*1.05 and r['no_single_codebook_collapse']
            and r['semantics']['balanced_5nn_accuracy']>baseline['semantics']['balanced_5nn_accuracy']
            and r['semantics']['separation_gap']>baseline['semantics']['separation_gap'])
    candidates=[r for r in reports.values() if eligible(r)]
    if not candidates:
        run(.005);candidates=[r for r in reports.values() if eligible(r)]
    met=bool(candidates)
    selected=max(candidates,key=lambda r:(r['semantics']['balanced_5nn_accuracy'],-r['rmse_deg'])) if met else min([r for r in reports.values() if r['weight']>0],key=lambda r:r['rmse_deg'])
    if rank()!=0:
        barrier();return
    selected_model,_=load_model(selected['checkpoint'],args.device)
    test,prediction,ids=evaluate(selected_model,arrays,metadata,'test',fk=True)
    test.update(selected_weight=selected['weight'],semantic_tradeoff_met=met,baseline_validation_rmse_deg=baseline['rmse_deg'],
        selected_validation_rmse_deg=selected['rmse_deg'],selected_validation_semantics=selected['semantics'],
        baseline_validation_semantics=baseline['semantics'],data_sha256=metadata['poses_sha256'])
    export_model(selected_model,output/'export',dict(dataset_sha256=metadata['poses_sha256'],selected_weight=selected['weight']))
    atomic_save(output/'best.pt',dict(config=selected_model.config,model=selected_model.state_dict(),metadata=test))
    np.savez_compressed(output/'test_reconstructions.npz',source_indices=ids,qpos=arrays['qpos'][ids],reconstructed_qpos=prediction['qpos'],tokens=prediction['tokens'].astype(np.uint8),label=arrays['label'][ids])
    primary_json(output/'report.json',test)
    state='completed' if met and test['reconstruction_target_met'] and test['no_single_codebook_collapse'] else 'completed_targets_unmet'
    primary_json(output/'status.json',dict(state=state,selected_weight=selected['weight'],test_mae_deg=test['mae_deg'],test_p95_deg=test['p95_deg'],semantic_tradeoff_met=met))
    from .report import build_report
    build_report(output)
    print(json.dumps(dict(state=state,mae_deg=test['mae_deg'],p95_deg=test['p95_deg'],semantic_tradeoff_met=met)),flush=True)
    barrier()
    return test
