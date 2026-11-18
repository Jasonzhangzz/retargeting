"""Real-label 32-window overfit, full two-stage losses and EMA verification."""
from pathlib import Path
import copy
import json
import time
import numpy as np
import torch
from .data import PoseContactDataset, collate_windows, train_statistics, write_json
from .train import seed_everything, stage1_loss, stage2_forward, batches_to_device, save_checkpoint, load_model
from .model import PoseAutoencoder, ContactModel, ModelEMA
from .losses import HACOLoss
from .metrics import contact_metrics


def select_examples(dataset):
    selected=[]
    # Spread examples across source sequences and across contact states.
    for positive in [False,True]:
        candidates={}
        for i,(key,start,end,known,label,temporal) in enumerate(dataset.rows):
            if known and label==positive and temporal:
                candidates.setdefault(key,[]).append(i)
        if not candidates:raise ValueError('Need genuine labelled temporal positives and negatives')
        sources=sorted(candidates)
        for key,values in candidates.items():
            spaced=values[::17];taken=set(spaced)
            candidates[key]=spaced+[i for i in values if i not in taken]
        count=0
        while count<16:
            available=False
            for key in sources:
                if candidates[key]:
                    available=True;selected.append(candidates[key].pop(0));count+=1
                    if count==16:break
            if not available:raise ValueError('Not enough distinct real examples')
    if len(set(selected))!=32:raise ValueError('Not enough distinct real examples')
    return selected


def run_smoke(args):
    seed_everything(42);torch.set_num_threads(args.threads);device=torch.device(args.device)
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    dataset=PoseContactDataset(args.corpus,'train',2)
    selected=select_examples(dataset);rows=[dataset[i] for i in selected]
    write_json(root/'examples.json',[dict(row_id=r['row_id'],dataset=r['dataset'],contact=bool(r['frame'])) for r in rows])
    groups=batches_to_device(collate_windows(rows),device)
    ae=PoseAutoencoder().to(device);optimizer=torch.optim.AdamW(ae.parameters(),lr=1e-4,weight_decay=.01,betas=(.9,.95))
    ae.eval()
    with torch.no_grad():initial=float(stage1_loss(ae,groups,False)['reconstruction'])
    trace=[];start=time.perf_counter()
    reuse=args.reuse_stage1 and (root/'stage1.pt').exists()
    if reuse:
        ae,saved=load_model(root/'stage1.pt',args.assets,device)
        if saved['metadata']['examples']!=[r['row_id'] for r in rows]:raise ValueError('Stage 1 examples changed')
        initial=saved['metadata']['reconstruction_initial']
    for step in range(0 if reuse else args.stage1_steps):
        ae.train();optimizer.zero_grad(set_to_none=True);loss=stage1_loss(ae,groups)
        loss['total'].backward();torch.nn.utils.clip_grad_norm_(ae.parameters(),1.);optimizer.step()
        if step%50==0:print('stage1',step,{k:float(v.detach()) for k,v in loss.items()},flush=True)
    ae.eval()
    with torch.no_grad():final=float(stage1_loss(ae,groups,False)['reconstruction'])
    if not reuse:
        save_checkpoint(root/'stage1.pt',ae,optimizer,1,args.stage1_steps,None,corpus=args.corpus,
                        examples=[r['row_id'] for r in rows],reconstruction_initial=initial,reconstruction_final=final)
    restored,_=load_model(root/'stage1.pt',args.assets,device)
    with torch.no_grad():
        for g in groups:torch.testing.assert_close(ae(g['pose51'])['reconstruction'],restored(g['pose51'])['reconstruction'])
    if args.stop_after_stage1:
        write_json(root/'stage1_report.json',dict(initial=initial,final=final,reduction=1-final/initial,checkpoint_roundtrip=True))
        dataset.close();return
    model=ContactModel.from_autoencoder(ae,args.assets).to(device)
    encoder=list(model.encoder.parameters());other=[p for n,p in model.named_parameters() if not n.startswith('encoder.')]
    optimizer=torch.optim.AdamW([dict(params=encoder,lr=1e-5),dict(params=other,lr=1e-4)],weight_decay=.01,betas=(.9,.95))
    ema=ModelEMA(model);loss_fn=HACOLoss(model.decoder,train_statistics(args.corpus)).configure_faces(np.load(Path(args.assets)/'faces.npy')).to(device)
    begin=0;schedule_steps=args.stage2_steps
    if args.reuse_stage2 and (root/'stage2.pt').exists():
        state=torch.load(root/'stage2.pt',map_location=device,weights_only=False)
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        ema.module.load_state_dict(state['ema']);ema.updates=state['ema_updates'];begin=state['step']
        schedule_steps=state['metadata'].get('schedule_steps',begin)
        if (root/'report.json').exists():trace=json.loads((root/'report.json').read_text())['trace']
    initial_encoder=next(model.encoder.parameters()).detach().clone();gradient_norm=None
    for step in range(begin,args.stage2_steps):
        model.train();optimizer.zero_grad(set_to_none=True);pred,y,valid=stage2_forward(model,groups)
        loss=loss_fn(pred,y,valid,step/max(schedule_steps-1,1))
        if not torch.isfinite(loss['total']):raise FloatingPointError('Nonfinite stage 2 objective')
        loss['total'].backward()
        if step==begin:
            gradient_norm=float(sum(p.grad.square().sum() for p in model.encoder.parameters() if p.grad is not None).sqrt())
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if step==begin:
            before=next(ema.module.parameters()).clone();online=next(model.parameters()).detach().clone()
        ema.update(model)
        if step==begin:torch.testing.assert_close(next(ema.module.parameters()),.999*before+.001*online,rtol=1e-5,atol=1e-7)
        if step%50==0 or step+1==args.stage2_steps:
            with torch.no_grad():
                model.eval();out,y,v=stage2_forward(model,groups)
                metrics=contact_metrics(out['contact_out'][v].sigmoid().cpu().numpy(),y[v].cpu().numpy(),out['frame_contact_logit'][v].sigmoid().cpu().numpy())
            record=dict(step=step+1,loss=float(loss['total'].detach()),**metrics);trace.append(record)
            print('stage2',json.dumps(record),flush=True)
    results={}
    for name,module in [('online',model.eval()),('ema',ema.module)]:
        with torch.no_grad():
            out,y,v=stage2_forward(module,groups)
            results[name]=contact_metrics(out['contact_out'][v].sigmoid().cpu().numpy(),y[v].cpu().numpy(),out['frame_contact_logit'][v].sigmoid().cpu().numpy())
            np.savez_compressed(root/(name+'_predictions.npz'),probability=out['contact_out'].sigmoid().cpu().numpy(),target=y.cpu().numpy(),frame_probability=out['frame_contact_logit'].sigmoid().cpu().numpy())
    save_checkpoint(root/'stage2.pt',model,optimizer,2,args.stage2_steps,ema,corpus=args.corpus,schedule_steps=schedule_steps)
    reloaded,_=load_model(root/'stage2.pt',args.assets,device)
    with torch.no_grad():
        torch.testing.assert_close(reloaded(groups[0]['pose51'])['contact_out'],ema.module(groups[0]['pose51'])['contact_out'])
    report=dict(samples=32,selection='16 labelled contact-free and 16 contact temporal windows from train only',
                reconstruction_initial=initial,reconstruction_final=final,reconstruction_reduction=1-final/initial,
                encoder_gradient_norm=gradient_norm,encoder_changed=not torch.equal(initial_encoder,next(model.encoder.parameters())),
                ema_updates=ema.updates,ema_formula_checked=True,checkpoint_roundtrip=True,
                device=str(device),wall_seconds=time.perf_counter()-start,stage1_steps=args.stage1_steps,stage2_steps=args.stage2_steps,
                metrics=results,trace=trace)
    report['passed']=report['reconstruction_reduction']>=.7 and all(r['f1']>=.9 and r['frame_f1']>=.95 for r in results.values())
    write_json(root/'report.json',report);dataset.close()
    if not report['passed']:raise RuntimeError('Small-data acceptance gate failed; inspect report.json before scaling')
