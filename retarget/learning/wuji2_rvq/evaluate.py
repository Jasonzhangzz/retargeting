"""Held-out reconstruction, token use, semantic retrieval and exact native FK."""
import numpy as np
import torch
from torch.nn import functional as F
from .data import ROOT


@torch.inference_mode()
def predict(model, q, batch=4096):
    device=next(model.parameters()).device
    outputs={k:[] for k in ['qpos','tokens','prefix','latent']}
    model.eval()
    for start in range(0,len(q),batch):
        out=model(torch.as_tensor(q[start:start+batch],device=device),update=False)
        for k in outputs:outputs[k].append(out[k].cpu().numpy())
    return {k:np.concatenate(v) for k,v in outputs.items()}


def angular_metrics(target, actual):
    error=np.abs((actual.astype(np.float64)-target)*180/np.pi)
    return dict(mae_deg=float(error.mean()),rmse_deg=float(np.sqrt(np.mean(error**2))),
                p95_deg=float(np.percentile(error,95)),samples=len(target))


@torch.inference_mode()
def semantic_metrics(prefix, labels, bank, bank_labels, device, seed=20261118):
    x=F.normalize(torch.as_tensor(prefix,device=device),dim=-1)
    b=F.normalize(torch.as_tensor(bank,device=device),dim=-1)
    by=torch.as_tensor(bank_labels,device=device,dtype=torch.long)
    predictions=[]
    for chunk in x.split(512):
        ids=(chunk@b.T).topk(5,dim=1).indices
        votes=torch.zeros(len(chunk),31,device=device).scatter_add_(1,by[ids],torch.ones_like(ids,dtype=torch.float32))
        predictions.append(votes.argmax(1).cpu().numpy())
    prediction=np.concatenate(predictions)
    accuracies=[float((prediction[labels==i]==i).mean()) for i in range(31) if (labels==i).any()]
    rng=np.random.default_rng(seed)
    groups=[np.flatnonzero(labels==i) for i in range(31)]
    same=[];different=[]
    values=F.normalize(torch.as_tensor(prefix),dim=-1).numpy()
    for _ in range(20000):
        a=int(rng.integers(31));bclass=(a+int(rng.integers(1,31)))%31
        if len(groups[a])<2 or len(groups[bclass])==0:continue
        ia,ib=rng.choice(groups[a],2,replace=False);ic=int(rng.choice(groups[bclass]))
        same.append(1-float(values[ia]@values[ib]));different.append(1-float(values[ia]@values[ic]))
    return dict(balanced_5nn_accuracy=float(np.mean(accuracies)),per_class_5nn_accuracy=accuracies,
                same_type_cosine_distance=float(np.mean(same)),different_type_cosine_distance=float(np.mean(different)),
                separation_gap=float(np.mean(different)-np.mean(same)),bank_samples=len(bank),pair_samples=len(same))


@torch.inference_mode()
def fingertip_errors(target, actual, joint_names, device):
    import mujoco
    from retarget.env.grasp.pose_gpu import ModelFK
    model=mujoco.MjModel.from_xml_path(str(ROOT/'data/grasp/assets/wuji2/right.xml'))
    native=[model.joint(i).name for i in range(model.njnt)][-20:]
    if native!=joint_names:raise ValueError('FK model joint order differs from dataset')
    fk=ModelFK(model,device)
    site_ids=[model.site('right_'+name+'_tip').id for name in ['thumb','index','middle','ring','pinky']]
    bodies=model.site_bodyid[site_ids]
    local=torch.as_tensor(model.site_pos[site_ids],device=device,dtype=torch.float32)
    result=[]
    def points(q):
        full=torch.cat([q.new_zeros(len(q),6),q],1)
        p,r=fk.frames(full)
        return p[:,bodies]+(r[:,bodies]@local[...,None]).squeeze(-1)
    for start in range(0,len(target),1024):
        q=torch.as_tensor(target[start:start+1024],device=device)
        r=torch.as_tensor(actual[start:start+1024],device=device)
        result.append(torch.linalg.vector_norm(points(q)-points(r),dim=-1).cpu().numpy()*1000)
    return np.concatenate(result)


def evaluate(model, arrays, metadata, split='val', fk=False):
    split_id={'train':0,'val':1,'test':2}[split]
    ids=np.flatnonzero(arrays['split']==split_id)
    q=arrays['qpos'][ids];labels=arrays['label'][ids]
    result=predict(model,q)
    report=dict(split=split,**angular_metrics(q,result['qpos']))
    report['per_type']={name:angular_metrics(q[labels==i],result['qpos'][labels==i]) for i,name in enumerate(metadata['types'])}
    report['per_joint']={name:angular_metrics(q[:,i:i+1],result['qpos'][:,i:i+1]) for i,name in enumerate(metadata['joint_names'])}
    report['quality_groups']={}
    for field in ['contact','stable']:
        for value in [False,True]:
            mask=arrays[field][ids]==value
            if mask.any():report['quality_groups'][field+'_'+str(value)]=angular_metrics(q[mask],result['qpos'][mask])
    report['codebooks']=[]
    for layer in range(model.config['levels']):
        count=np.bincount(result['tokens'][:,layer],minlength=model.config['entries'])
        prob=count[count>0]/count.sum()
        report['codebooks'].append(dict(level=layer,used=int((count>0).sum()),entries=len(count),
            utilization=float((count>0).mean()),perplexity=float(np.exp(-(prob*np.log(prob)).sum()))))
    rng=np.random.default_rng(20261118)
    bank=[]
    for label in range(31):
        candidates=np.flatnonzero((arrays['split']==0)&(arrays['label']==label))
        bank.extend(rng.choice(candidates,min(200,len(candidates)),replace=False).tolist())
    bank=np.asarray(bank)
    bank_pred=predict(model,arrays['qpos'][bank])
    report['semantics']=semantic_metrics(result['prefix'],labels,bank_pred['prefix'],arrays['label'][bank],next(model.parameters()).device)
    if fk:
        errors=fingertip_errors(q,result['qpos'],metadata['joint_names'],next(model.parameters()).device)
        report['fingertips']=dict(mean_mm=float(errors.mean()),p95_mm=float(np.percentile(errors,95)),
            per_finger_mean_mm=errors.mean(0).tolist(),frame='fixed-wrist native model; Euclidean distances are rigid-frame invariant')
        for i,name in enumerate(metadata['types']):report['per_type'][name]['fingertip_mean_mm']=float(errors[labels==i].mean())
    report['reconstruction_target_met']=report['mae_deg']<=1 and report['p95_deg']<=5
    report['no_single_codebook_collapse']=all(x['used']>=2 for x in report['codebooks'])
    return report,result,ids
