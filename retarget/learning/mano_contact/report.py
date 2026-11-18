"""Standalone figures and natural-split evaluation for the small verification."""
from pathlib import Path
import json
import time
import numpy as np
import torch
from .data import PoseContactDataset, write_json
from .train import load_model, evaluate
from .metrics import calibrated_threshold, contact_metrics
from .geometry import ManoGeometry


def build_report(corpus,assets,run,output,threads=4):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    torch.set_num_threads(threads);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    run=Path(run);model,_=load_model(run/'stage2.pt',assets)
    results={};datasets={}
    for split in ['val','test']:
        d=PoseContactDataset(corpus,split,2);datasets[split]=d
        if len(d):
            result,arrays=evaluate(model,d,batch_size=32)
            if split=='val':
                threshold=calibrated_threshold(arrays['probability'],arrays['target'])
                result['calibrated_threshold']=threshold
                result['calibrated_metrics']=contact_metrics(arrays['probability'],arrays['target'],arrays['frame_probability'],threshold)
            results[split]=result
            np.savez_compressed(output/(split+'_predictions.npz'),**arrays)
    train=PoseContactDataset(corpus,'train',2)
    example_ids=json.loads((run/'examples.json').read_text());lookup={f'{r[0]}:{r[2]}':i for i,r in enumerate(train.rows)}
    ae,_=load_model(run/'stage1.pt',assets)
    q=torch.stack([train[lookup[r['row_id']]]['pose51'] for r in example_ids])
    y=np.array([r['contact'] for r in example_ids])
    with torch.inference_mode():
        encoded=ae(q);features={'encoder_latent':encoded['latent'][:,-1].numpy(),'contrastive_projection':encoded['projection'].numpy()}
        coordinates={}
        for name,value in features.items():
            unit=value/np.linalg.norm(value,axis=-1,keepdims=True)
            centered=value-value.mean(0);u,s,_=np.linalg.svd(centered,full_matrices=False);coordinates[name]=u[:,:2]*s[:2]
            similarity=unit@unit.T;same=y[:,None]==y[None];np.fill_diagonal(same,False)
            results[name+'_separation']=dict(within_class_cosine=float(similarity[same].mean()),between_class_cosine=float(similarity[y[:,None]!=y[None]].mean()))
        for _ in range(5):model(q[:1])
        elapsed=[]
        for _ in range(30):
            start=time.perf_counter();model(q[:1]);elapsed.append((time.perf_counter()-start)*1000)
        results['single_window_cpu_latency_ms']=dict(median=float(np.median(elapsed)),p95=float(np.quantile(elapsed,.95)),threads=threads)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,(name,xy) in zip(axes,coordinates.items()):
        for state,color,label in [(False,'#2878b5','Noncontact'),(True,'#d55e00','Contact')]:
            ax.scatter(xy[y==state,0],xy[y==state,1],c=color,label=label,s=35)
        ax.set(xlabel='PC1',ylabel='PC2',title=name.replace('_',' '));ax.legend()
    fig.suptitle('Stage 1: 32 training windows');fig.tight_layout();fig.savefig(output/'latent.png',dpi=180);plt.close(fig)
    d=datasets['val'] if len(datasets['val']) else train
    predictions=[]
    with torch.inference_mode():
        for i in range(len(d)):
            r=d[i];predictions.append(model(r['pose51'][None])['contact_out'].sigmoid().numpy()[0])
    probability=np.stack(predictions);states=np.array([d.rows[i][4] for i in range(len(d))])
    negative=np.flatnonzero(~states);positive=np.flatnonzero(states)
    picks=[]
    def consecutive(left,right):
        a,b=d.rows[left],d.rows[right]
        if a[0]!=b[0]:return False
        frames=d.handle['records'][a[0]]['frame_id']
        return int(frames[b[2]])-int(frames[a[2]])==1
    approach=next((i for i in range(len(d)-1) if not states[i] and states[i+1] and consecutive(i,i+1)),None)
    if approach is not None:picks.append(('Approach: before contact',approach))
    elif len(negative):picks.append(('Noncontact',int(negative[0])))
    if len(positive):picks.append(('Contact',int(positive[len(positive)//2])))
    # An actual annotated contact->noncontact transition in a single sequence.
    release=next((i for i in range(1,len(d)) if states[i-1] and not states[i] and consecutive(i-1,i)),None)
    if release is not None:picks.append(('Release',release))
    if len(negative):
        worst=int(negative[np.argmax(probability[negative].max(-1))])
        picks.append(('False positive' if probability[worst].max()>=.52 else 'Largest noncontact probability',worst))
    geometry=ManoGeometry(Path(assets)/'MANO_RIGHT.pkl')
    fig=plt.figure(figsize=(4*max(len(picks),1),7))
    notes=[]
    for column,(name,index) in enumerate(picks):
        row=d[index];key,_,end,*_=d.rows[index];g=d.handle['records'][key]
        physical=row['pose51'][-1].numpy().copy();beta=g['beta'][end] if 'beta' in g else np.zeros(10)
        rest=geometry.v_template.numpy()+np.einsum('vck,k->vc',geometry.shapedirs.numpy(),beta)
        physical[:3]-=(geometry.J_regressor.numpy()@rest)[0]
        vertices,_=geometry(physical[None],beta[None]);v=vertices[0]
        # Translate only for display; retain the actual pose and vertex order.
        v=v-v.mean(0);truth=row['contact'].numpy().astype(bool);predicted=probability[index]>=.52
        for line,(mask,label) in enumerate([(truth,'GT'),(predicted,'Prediction')]):
            ax=fig.add_subplot(2,len(picks),line*len(picks)+column+1,projection='3d')
            ax.scatter(*v.T,c='#b8bdc5',s=3,alpha=.4)
            if mask.any():ax.scatter(*v[mask].T,c='#d55e00',s=12)
            ax.set_title(f'{name}: {label}');ax.set_box_aspect(np.ptp(v,axis=0).clip(.03));ax.view_init(25,-70);ax.set_axis_off()
        notes.append(dict(case=name,row_id=row['row_id'],max_probability=float(probability[index].max()),gt_contact_vertices=int(truth.sum()),predicted_contact_vertices=int(predicted.sum())))
    fig.tight_layout();fig.savefig(output/'contacts.png',dpi=180);plt.close(fig)
    results['examples']=notes
    write_json(output/'evaluation.json',results)
    for d in datasets.values():d.close()
    train.close()
    return results
