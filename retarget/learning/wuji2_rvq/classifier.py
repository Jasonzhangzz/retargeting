"""Pose-only RVQ classification: supervised CE, no pose decoder or regression loss."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .model import PoseRVQ, mlp


class PoseRVQClassifier(PoseRVQ):
    def __init__(self, lower, upper, joint_names, class_names, **kwargs):
        super().__init__(lower, upper, joint_names, **kwargs)
        del self.decoder
        if len(class_names)!=31 or len(set(class_names))!=31:
            raise ValueError('Expected 31 unique grasp types')
        self.config['class_names']=list(class_names)
        self.classifier=mlp([self.config['dimension'],128,len(class_names)])

    def forward(self, q, continuous=False, update=True):
        z=self.encoder(self.normalize(q))
        out=dict(latent=z,commitment=z.sum()*0) if continuous else self.quantizer(z,update=update)
        out['latent']=z
        out['logits']=self.classifier(z if continuous else out['quantized'])
        return out

    def classify_tokens(self,tokens):
        return self.classifier(self.quantizer.lookup(tokens).sum(1))

    def decode(self,tokens):
        raise ValueError('Classification model has no pose decoder; use predict or classify_tokens')


@torch.inference_mode()
def predict_classes(model,q,continuous=False,batch=4096):
    model.eval();device=model.lower.device
    logits=[];tokens=[];reconstructed=[]
    for start in range(0,len(q),batch):
        out=model(torch.as_tensor(q[start:start+batch],device=device,dtype=torch.float32),continuous=continuous,update=False)
        logits.append(out['logits'].cpu())
        if 'qpos' in out:reconstructed.append(out['qpos'].cpu().numpy())
        if not continuous:tokens.append(out['tokens'].cpu().numpy().astype(np.uint8))
    if not logits:raise ValueError('Empty pose batch')
    scores=torch.cat(logits)
    result=dict(logits=scores.numpy(),probabilities=scores.softmax(-1).numpy(),class_index=scores.argmax(-1).numpy())
    if tokens:result['tokens']=np.concatenate(tokens)
    if reconstructed:result['reconstructed_qpos']=np.concatenate(reconstructed)
    return result


def classification_metrics(logits,labels,class_names):
    labels=np.asarray(labels,dtype=np.int64);prediction=np.asarray(logits).argmax(-1)
    n=len(class_names);cm=np.bincount(labels*n+prediction,minlength=n*n).reshape(n,n)
    support=cm.sum(1);predicted=cm.sum(0);tp=cm.diagonal()
    recall=tp/np.maximum(support,1);precision=tp/np.maximum(predicted,1)
    f1=2*precision*recall/np.maximum(precision+recall,1e-12)
    ce=F.cross_entropy(torch.as_tensor(logits),torch.as_tensor(labels)).item()
    return dict(samples=len(labels),accuracy=float((prediction==labels).mean()),
        balanced_accuracy=float(recall[support>0].mean()),macro_f1=float(f1[support>0].mean()),cross_entropy=ce,
        top3_accuracy=float((np.argsort(logits,axis=1)[:,-3:]==labels[:,None]).any(1).mean()),
        confusion_matrix=cm.tolist(),confusion_rows='true class',confusion_columns='predicted class',class_names=list(class_names),
        per_type={name:dict(support=int(support[i]),precision=float(precision[i]),recall=float(recall[i]),f1=float(f1[i])) for i,name in enumerate(class_names)})


def evaluate_classifier(model,arrays,metadata,split='val',continuous=False):
    ids=np.flatnonzero(arrays['split']=={'train':0,'val':1,'test':2}[split])
    result=predict_classes(model,arrays['qpos'][ids],continuous)
    report=classification_metrics(result['logits'],arrays['label'][ids],metadata['types'])
    report.update(task='classification',split=split,continuous=continuous,data_sha256=metadata['poses_sha256'])
    report['quality_groups']={}
    for field in ['contact','stable']:
        for value in [False,True]:
            mask=arrays[field][ids]==value
            if mask.any():
                group=classification_metrics(result['logits'][mask],arrays['label'][ids][mask],metadata['types'])
                report['quality_groups'][field+'_'+str(value)]={k:group[k] for k in ['samples','accuracy','macro_f1','balanced_accuracy']}
    if not continuous:
        report['codebooks']=[]
        for level in range(model.config['levels']):
            count=np.bincount(result['tokens'][:,level],minlength=model.config['entries']);p=count[count>0]/count.sum()
            report['codebooks'].append(dict(level=level,used=int((count>0).sum()),utilization=float((count>0).mean()),perplexity=float(np.exp(-(p*np.log(p)).sum()))))
    return report,result,ids


def classifier_campaign(args):
    from .data import load,sha
    from .train import (seed_all,primary_json,rank,world,local_batch,barrier,fit_phase,load_model,atomic_save,export_model)
    torch.set_num_threads(args.threads);torch.set_float32_matmul_precision('high')
    arrays,metadata=load(args.data);output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    configuration=dict(task='classification',objective='cross_entropy + 0.25 * commitment',decoder=False,reconstruction_weight=0,
        selection='maximum validation macro F1',optimizer=args.optimizer,seed=args.seed,data_sha256=metadata['poses_sha256'],
        world_size=world(),global_batch=local_batch(args)*world(),per_gpu_batch=local_batch(args),samples_per_type_per_gpu=args.samples_per_type if not getattr(args,'batch_size',0) else None,
        steps_per_epoch=args.steps_per_epoch,warmup_epochs=args.warmup_epochs,rvq_epochs=args.rvq_epochs,
        class_names=metadata['types'],code_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')})
    primary_json(output/'configuration.json',configuration)
    initial=output/'initialized.pt'
    if initial.exists():
        if not args.resume:raise ValueError('Output already initialized; pass --resume')
        model,info=load_model(initial,args.device)
        if (not isinstance(model,PoseRVQClassifier) or info['data_sha256']!=metadata['poses_sha256'] or info['optimizer']!=args.optimizer):
            raise ValueError('Classification initialization configuration changed')
    else:
        seed_all(args.seed)
        model=PoseRVQClassifier(metadata['lower'],metadata['upper'],metadata['joint_names'],metadata['types']).to(args.device)
        primary_json(output/'status.json',dict(state='classification_warmup'))
        model=fit_phase(model,arrays,metadata,output/'warmup',args.warmup_epochs,0,args,continuous=True)
        if rank()==0:
            report,_,_=evaluate_classifier(model,arrays,metadata,'val',continuous=True)
            primary_json(output/'continuous_validation.json',report)
            ids=np.flatnonzero(arrays['split']==0);rng=np.random.default_rng(args.seed)
            ids=rng.choice(ids,min(65536,len(ids)),replace=False)
            with torch.inference_mode():
                latent=torch.cat([model.encoder(model.normalize(torch.as_tensor(arrays['qpos'][chunk],device=args.device))) for chunk in np.array_split(ids,16)])
                model.quantizer.initialize(latent,seed=args.seed)
            atomic_save(initial,dict(config=model.config,model=model.state_dict(),data_sha256=metadata['poses_sha256'],optimizer=args.optimizer,task='classification'))
        barrier();model,_=load_model(initial,args.device)
    primary_json(output/'status.json',dict(state='classification_rvq'))
    model=fit_phase(model,arrays,metadata,output/'rvq',args.rvq_epochs,0,args)
    if rank()!=0:barrier();return
    validation,_,_=evaluate_classifier(model,arrays,metadata,'val')
    primary_json(output/'validation.json',validation)
    test,prediction,ids=evaluate_classifier(model,arrays,metadata,'test')
    baseline,_=load_model(output/'warmup/best.pt',args.device)
    baseline_test,_,_=evaluate_classifier(baseline,arrays,metadata,'test',continuous=True)
    primary_json(output/'continuous_test.json',baseline_test)
    test.update(validation_macro_f1=validation['macro_f1'],continuous_baseline_accuracy=baseline_test['accuracy'],
        continuous_baseline_macro_f1=baseline_test['macro_f1'],quantized_f1_minus_continuous=test['macro_f1']-baseline_test['macro_f1'])
    atomic_save(output/'best.pt',dict(config=model.config,model=model.state_dict(),metadata=test,task='classification'))
    export_model(model,output/'export',test)
    np.savez_compressed(output/'test_predictions.npz',source_indices=ids,true_class=arrays['label'][ids],class_names=np.asarray(metadata['types']),**prediction)
    primary_json(output/'report.json',test)
    primary_json(output/'status.json',dict(state='completed',task='classification',test_accuracy=test['accuracy'],test_macro_f1=test['macro_f1']))
    build_classifier_report(output)
    print(json.dumps(dict(state='completed',task='classification',accuracy=test['accuracy'],macro_f1=test['macro_f1'])),flush=True)
    barrier();return test


def build_classifier_report(output):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    output=Path(output);r=json.loads((output/'report.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for phase in ['warmup','rvq']:
        history=json.loads((output/phase/'history.json').read_text())
        axes[0].plot([v['epoch'] for v in history],[v['val_macro_f1'] for v in history],label=phase)
        axes[1].plot([v['epoch'] for v in history],[v['cross_entropy'] for v in history],label=phase)
    axes[0].set_ylabel('Validation macro F1');axes[1].set_ylabel('Training cross entropy')
    for a in axes:a.set_xlabel('Epoch');a.legend();a.grid(alpha=.2)
    fig.tight_layout();fig.savefig(output/'training_curves.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,10));cm=np.asarray(r['confusion_matrix']);ax.imshow(cm/np.maximum(cm.sum(1,keepdims=True),1),vmin=0,vmax=1)
    names=r['class_names'];ax.set_xticks(range(31),names,rotation=90,fontsize=6);ax.set_yticks(range(31),names,fontsize=6)
    ax.set_xlabel('Predicted');ax.set_ylabel('True');fig.tight_layout();fig.savefig(output/'confusion_matrix.png',dpi=160);plt.close(fig)
    rows=['# Wuji2 RVQ classifier','',f"Test accuracy: {r['accuracy']:.4%}; macro F1: {r['macro_f1']:.4f}.",
        f"Continuous warmup baseline accuracy: {r['continuous_baseline_accuracy']:.4%}; macro F1: {r['continuous_baseline_macro_f1']:.4f}.",'',
        'Input: 20 native joint angles in radians. Output: 31 grasp type probabilities and four RVQ tokens.',
        'Objective: classification cross entropy + 0.25 commitment. No decoder or pose regression. Selection: validation macro F1.',
        'The continuous baseline is the 20-epoch classification warmup, not a separately tuned full-budget MLP experiment.','',
        '| Type | Support | Precision | Recall | F1 |','|---|---:|---:|---:|---:|']
    rows += [f"| {name} | {v['support']} | {v['precision']:.4f} | {v['recall']:.4f} | {v['f1']:.4f} |" for name,v in r['per_type'].items()]
    (output/'REPORT.md').write_text('\n'.join(rows)+'\n')
