"""Shared RVQ decoder trunk with pose and grasp-type output heads."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .model import PoseRVQ
from .classifier import evaluate_classifier,build_classifier_report
from .evaluate import angular_metrics,fingertip_errors


class TwoHeadDecoder(nn.Module):
    def __init__(self,dimension=64,classes=31):
        super().__init__()
        self.trunk=nn.Sequential(nn.Linear(dimension,256),nn.LayerNorm(256),nn.GELU(),
                                 nn.Linear(256,256),nn.LayerNorm(256),nn.GELU())
        self.pose_head=nn.Linear(256,20)
        self.class_head=nn.Linear(256,classes)

    def forward(self,z):
        hidden=self.trunk(z)
        return dict(normalized=self.pose_head(hidden).tanh(),logits=self.class_head(hidden))


class PoseRVQJoint(PoseRVQ):
    def __init__(self,lower,upper,joint_names,class_names,joint=True,**kwargs):
        super().__init__(lower,upper,joint_names,**kwargs)
        if not joint or len(class_names)!=31 or len(set(class_names))!=31:
            raise ValueError('Expected joint model with 31 unique grasp types')
        self.config.update(class_names=list(class_names),joint=True)
        self.decoder=TwoHeadDecoder(self.config['dimension'],len(class_names))

    def forward(self,q,continuous=False,update=True):
        z=self.encoder(self.normalize(q))
        out=dict(latent=z,commitment=z.sum()*0) if continuous else self.quantizer(z,update=update)
        out['latent']=z
        out.update(self.decoder(z if continuous else out['quantized']))
        out['qpos']=self.denormalize(out['normalized'])
        return out

    def decode_outputs(self,tokens):
        out=self.decoder(self.quantizer.lookup(tokens).sum(1))
        out['qpos']=self.denormalize(out['normalized'])
        return out

    def decode(self,tokens):return self.decode_outputs(tokens)['qpos']
    def classify_tokens(self,tokens):return self.decode_outputs(tokens)['logits']


class JointTokenDecoder(nn.Module):
    """Independent codebooks + shared two-head decoder, without encoder."""
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.register_buffer('books',torch.zeros(config['levels'],config['entries'],config['dimension']))
        self.register_buffer('lower',torch.tensor(config['lower'],dtype=torch.float32))
        self.register_buffer('upper',torch.tensor(config['upper'],dtype=torch.float32))
        self.decoder=TwoHeadDecoder(config['dimension'],len(config['class_names']))

    @classmethod
    def from_model(cls,model):
        result=cls(model.config).to(model.lower.device)
        result.books.copy_(model.quantizer.books);result.decoder.load_state_dict(model.decoder.state_dict())
        return result.eval()

    def forward(self,tokens):
        if tokens.ndim!=2 or tokens.shape[1]!=self.config['levels'] or tokens.dtype not in (torch.int32,torch.int64,torch.uint8):
            raise ValueError('Expected integer tokens[B,levels]')
        tokens=tokens.long()
        if tokens.numel() and ((tokens<0).any() or (tokens>=self.config['entries']).any()):
            raise ValueError('Token outside codebook')
        z=torch.stack([self.books[i,tokens[:,i]] for i in range(self.config['levels'])]).sum(0)
        out=self.decoder(z)
        out['qpos']=self.lower+(out['normalized']+1)*.5*(self.upper-self.lower)
        out['probabilities']=out['logits'].softmax(-1)
        return out


def evaluate_joint(model,arrays,metadata,split='val',continuous=False,reconstruction_weight=1.,classification_weight=.05,fk=False):
    report,result,ids=evaluate_classifier(model,arrays,metadata,split,continuous)
    target=arrays['qpos'][ids];prediction=result['reconstructed_qpos']
    report.update(angular_metrics(target,prediction),task='joint')
    normalized_error=2*(prediction-target)/(np.asarray(metadata['upper'])-np.asarray(metadata['lower']))
    report['normalized_mse']=float(np.square(normalized_error.astype(np.float64)).mean())
    report['joint_score']=reconstruction_weight*report['normalized_mse']+classification_weight*report['cross_entropy']
    report['loss_weights']=dict(reconstruction=reconstruction_weight,classification=classification_weight,commitment=.25)
    for i,name in enumerate(metadata['types']):
        mask=arrays['label'][ids]==i
        if mask.any():report['per_type'][name].update(angular_metrics(target[mask],prediction[mask]))
    report['per_joint']={name:angular_metrics(target[:,i:i+1],prediction[:,i:i+1]) for i,name in enumerate(metadata['joint_names'])}
    for field in ['contact','stable']:
        for value in [False,True]:
            mask=arrays[field][ids]==value
            if mask.any():report['quality_groups'][field+'_'+str(value)].update(angular_metrics(target[mask],prediction[mask]))
    report['reconstruction_target_met']=report['mae_deg']<=1 and report['p95_deg']<=5
    if fk:
        errors=fingertip_errors(target,prediction,metadata['joint_names'],model.lower.device)
        report['fingertips']=dict(mean_mm=float(errors.mean()),p95_mm=float(np.percentile(errors,95)),per_finger_mean_mm=errors.mean(0).tolist(),frame='wrist local')
        for i,name in enumerate(metadata['types']):
            report['per_type'][name]['fingertip_mean_mm']=float(errors[arrays['label'][ids]==i].mean())
    return report,result,ids


def joint_campaign(args):
    from .data import load,sha
    from .train import seed_all,primary_json,rank,world,local_batch,barrier,fit_phase,load_model,atomic_save,export_model
    torch.set_num_threads(args.threads);torch.set_float32_matmul_precision('high')
    arrays,metadata=load(args.data);output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    configuration=dict(task='joint',objective=f'{args.reconstruction_weight} * normalized_MSE + {args.classification_weight} * cross_entropy + 0.25 * commitment',
        decoder='shared 64->256->256 trunk; 256->20 pose head and 256->31 class head',
        reconstruction_weight=args.reconstruction_weight,classification_weight=args.classification_weight,
        selection='minimum validation weighted reconstruction MSE + classification cross entropy',
        optimizer=args.optimizer,seed=args.seed,data_sha256=metadata['poses_sha256'],world_size=world(),
        global_batch=local_batch(args)*world(),per_gpu_batch=local_batch(args),samples_per_type_per_gpu=args.samples_per_type if not getattr(args,'batch_size',0) else None,
        steps_per_epoch=args.steps_per_epoch,warmup_epochs=args.warmup_epochs,rvq_epochs=args.rvq_epochs,class_names=metadata['types'],
        code_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')})
    primary_json(output/'configuration.json',configuration)
    resume_key={k:v for k,v in configuration.items() if k!='code_sha256'}
    initial=output/'initialized.pt'
    if initial.exists():
        if not args.resume:raise ValueError('Output already initialized; pass --resume')
        model,info=load_model(initial,args.device)
        if not isinstance(model,PoseRVQJoint) or info.get('training_configuration')!=resume_key:
            raise ValueError('Joint initialization configuration changed')
    else:
        seed_all(args.seed)
        model=PoseRVQJoint(metadata['lower'],metadata['upper'],metadata['joint_names'],metadata['types']).to(args.device)
        primary_json(output/'status.json',dict(state='joint_warmup'))
        model=fit_phase(model,arrays,metadata,output/'warmup',args.warmup_epochs,0,args,continuous=True)
        if rank()==0:
            report,_,_=evaluate_joint(model,arrays,metadata,'val',True,args.reconstruction_weight,args.classification_weight)
            primary_json(output/'continuous_validation.json',report)
            ids=np.flatnonzero(arrays['split']==0);rng=np.random.default_rng(args.seed)
            ids=rng.choice(ids,min(65536,len(ids)),replace=False)
            with torch.inference_mode():
                latent=torch.cat([model.encoder(model.normalize(torch.as_tensor(arrays['qpos'][chunk],device=args.device))) for chunk in np.array_split(ids,16)])
                model.quantizer.initialize(latent,seed=args.seed)
            atomic_save(initial,dict(config=model.config,model=model.state_dict(),data_sha256=metadata['poses_sha256'],optimizer=args.optimizer,task='joint',training_configuration=resume_key))
        barrier();model,_=load_model(initial,args.device)
    primary_json(output/'status.json',dict(state='joint_rvq'))
    model=fit_phase(model,arrays,metadata,output/'rvq',args.rvq_epochs,0,args)
    if rank()!=0:barrier();return
    validation,_,_=evaluate_joint(model,arrays,metadata,'val',False,args.reconstruction_weight,args.classification_weight)
    primary_json(output/'validation.json',validation)
    test,prediction,ids=evaluate_joint(model,arrays,metadata,'test',False,args.reconstruction_weight,args.classification_weight,fk=True)
    baseline,_=load_model(output/'warmup/best.pt',args.device)
    baseline_test,_,_=evaluate_joint(baseline,arrays,metadata,'test',True,args.reconstruction_weight,args.classification_weight)
    primary_json(output/'continuous_test.json',baseline_test)
    test.update(validation_joint_score=validation['joint_score'],continuous_baseline_accuracy=baseline_test['accuracy'],continuous_baseline_macro_f1=baseline_test['macro_f1'],continuous_baseline_mae_deg=baseline_test['mae_deg'])
    atomic_save(output/'best.pt',dict(config=model.config,model=model.state_dict(),metadata=test,task='joint'))
    export_model(model,output/'export',test)
    np.savez_compressed(output/'test_predictions.npz',source_indices=ids,true_class=arrays['label'][ids],qpos=arrays['qpos'][ids],class_names=np.asarray(metadata['types']),**prediction)
    primary_json(output/'report.json',test)
    state='completed' if test['reconstruction_target_met'] else 'completed_targets_unmet'
    primary_json(output/'status.json',dict(state=state,task='joint',test_accuracy=test['accuracy'],test_macro_f1=test['macro_f1'],test_mae_deg=test['mae_deg'],test_p95_deg=test['p95_deg']))
    build_joint_report(output)
    print(json.dumps(dict(state=state,task='joint',accuracy=test['accuracy'],macro_f1=test['macro_f1'],mae_deg=test['mae_deg'],p95_deg=test['p95_deg'])),flush=True)
    barrier();return test


def build_joint_report(output):
    build_classifier_report(output)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    output=Path(output);r=json.loads((output/'report.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for phase in ['warmup','rvq']:
        h=json.loads((output/phase/'history.json').read_text());x=[v['epoch'] for v in h]
        for ax,key in zip(axes,['val_mae_deg','val_macro_f1','val_joint_score']):ax.plot(x,[v[key] for v in h],label=phase);ax.set_ylabel(key)
    for ax in axes:ax.set_xlabel('Epoch');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(output/'training_curves.png',dpi=160);plt.close(fig)
    rows=['# Wuji2 joint RVQ: reconstruction + classification','',
        'One shared decoder trunk feeds a 20-joint pose head and a 31-class classification head.',
        f"Test accuracy {r['accuracy']:.4%}; macro F1 {r['macro_f1']:.4f}; pose MAE {r['mae_deg']:.4f} deg; RMSE {r['rmse_deg']:.4f} deg; P95 {r['p95_deg']:.4f} deg.",
        f"Loss weights: {r['loss_weights']}. Selection: minimum validation weighted normalized MSE + cross entropy.",
        f"Reconstruction targets met: {r['reconstruction_target_met']}. Fingertip mean: {r['fingertips']['mean_mm']:.4f} mm.",'',
        '| Type | Support | Precision | Recall | F1 | MAE deg | RMSE deg | P95 deg |','|---|---:|---:|---:|---:|---:|---:|---:|']
    rows += [f"| {name} | {v['support']} | {v['precision']:.4f} | {v['recall']:.4f} | {v['f1']:.4f} | {v['mae_deg']:.4f} | {v['rmse_deg']:.4f} | {v['p95_deg']:.4f} |" for name,v in r['per_type'].items()]
    (output/'REPORT.md').write_text('\n'.join(rows)+'\n')
