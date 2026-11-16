"""CART sequential training, independent holdout evaluation and checkpoints."""
import json,os,signal,time,fcntl
from pathlib import Path
import numpy as np
import torch
from .cart_data import load_sessions,sequence_indices,preprocess,manifest,preprocessing_current
from .cart_convex import geometry_identity
from .cart_network import CartNetwork,hull_descriptor
from .cart_protocol import PROTOCOL,STAGES,PHASES,ACTION_COUNT,NEUTRAL_INDEX,CATEGORIES
from .cart_losses import NativeHand,contact_losses,WEIGHTS
from retarget.baseline.geort.training import atomic_json

def train(root,train_ids,validation_ids,run_id,epochs=200,seed=714,batch_size=64,length=16,smoke=False):
    root=Path(root);folder=root/'models'/run_id;folder.mkdir(parents=True,exist_ok=True)
    lockpath=root.parent/'geort/training.lock';lockpath.parent.mkdir(parents=True,exist_ok=True)
    lock=lockpath.open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:lock.close();raise ValueError('已有 GeoRT／CART 训练任务')
    cancelled=False
    def stop(*_):
        nonlocal cancelled
        cancelled=True
    old={s:signal.signal(s,stop) for s in (signal.SIGTERM,signal.SIGINT)}
    def check():
        if cancelled:raise InterruptedError('CART 任务已取消，已启用模型不变')
    started=time.time()
    def progress(phase,**values):
        state=dict(status='running',run_id=run_id,pid=os.getpid(),phase=phase,elapsed_s=time.time()-started,**values)
        atomic_json(folder/'status.json',state)
        with (folder/'trace.jsonl').open('a') as stream:stream.write(json.dumps(state,allow_nan=False)+'\n')
        print(json.dumps(state,ensure_ascii=False),flush=True)
    try:
        if set(train_ids)&set(validation_ids):raise ValueError('训练和验证不能包含同一会话')
        if not 1<=epochs<=2000:raise ValueError('轮数必须为1–2000')
        if not smoke:
            for ids,split in ((train_ids,'train'),(validation_ids,'validation')):
                metadata=[manifest(root,key)[1] for key in ids]
                if {m['stage'] for m in metadata}!={s['id'] for s in STAGES} or any(m['split']!=split for m in metadata):
                    raise ValueError('训练和独立验证各需全部8个子动作，请先完成五类动作采集')
        np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(1)
        for key in [*train_ids,*validation_ids]:
            check()
            recording,metadata=manifest(root,key)
            if not preprocessing_current(recording,metadata,full=not smoke):
                preprocess(root,key,full=not smoke,progress=progress,check=check)
        sessions,source=load_sessions(root,train_ids,'train',require_complete=not smoke)
        validation,other=load_sessions(root,validation_ids,'validation',require_complete=not smoke)
        if source!=other:raise ValueError('验证数据与训练源不兼容')
        if {m.get('session_id') for m,_,_ in sessions}&{m.get('session_id') for m,_,_ in validation}:
            raise ValueError('同一录制会话不能拆分到训练与验证')
        for data in (sessions,validation):
            if not smoke and {m['stage'] for m,_,_ in data}!={s['id'] for s in STAGES}:
                raise ValueError('训练和验证各需包含全部8个子动作及每段两次完整周期')
            if not smoke and any(info['profile']!='full' for _,_,info in data):raise ValueError('测试预处理数据不能用于正式训练')
        device='cuda' if torch.cuda.is_available() else 'cpu'
        hand=NativeHand(device);model=CartNetwork().to(device)
        config=dict(protocol=PROTOCOL,geometry=geometry_identity(),architecture='cart_shared256_gru128_v3',action_ids=[s['id'] for s in STAGES],categories=list(CATEGORIES),source=source,robot=hand.robot.spec,
            train_ids=train_ids,validation_ids=validation_ids,epochs=epochs,seed=seed,batch_size=batch_size,sequence_length=length,
            optimizer='AdamW',learning_rate=1e-4,weights=WEIGHTS,profile='smoke_test' if smoke else 'full',
            datasets=[dict(id=m['id'],fingerprint=info['fingerprint']) for m,_,info in sessions+validation],
            thresholds=dict(contact_m=.002,release_m=.004,penetration_m=.0005,normal_deg=30))
        atomic_json(folder/'config.json',config)
        windows=[]
        for m,rows,_ in sessions:
            for ids in sequence_indices(rows,length):windows.append((m['stage'],[rows[i] for i in ids]))
        if not windows:raise ValueError('没有连续16帧动作片段')
        # Balanced stage sampling preserves rare actions and synchronous frames.
        groups={s:[w for name,w in windows if name==s] for s in {name for name,_ in windows}}
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
        for epoch in range(epochs):
            sums={};count=0;model.train()
            chosen=[groups[s][np.random.randint(len(groups[s]))] for s in np.random.choice(list(groups),size=max(batch_size,len(windows)))]
            for start in range(0,len(chosen),batch_size):
                check();batch=chosen[start:start+batch_size]
                x=torch.tensor(np.array([[r['features'] for r in w] for w in batch]),device=device,dtype=torch.float32)
                for b,w in enumerate(batch):
                    x[b,0,-1]=0
                    x[b,1:,-1]=torch.tensor(np.diff([r['timestamp_ms'] for r in w])/1000,device=device)
                hull=torch.tensor(np.array([[hull_descriptor(r['target']) for r in w] for w in batch]),device=device,dtype=torch.float32)
                action=torch.tensor([[r['stage_index'] for r in w] for w in batch],device=device)
                phase=torch.tensor([[PHASES.index(r['phase']) for r in w] for w in batch],device=device)
                neutral=action.clone();neutral[phase==0]=NEUTRAL_INDEX
                output=model(x,hull,torch.nn.functional.one_hot(neutral,ACTION_COUNT).float(),torch.nn.functional.one_hot(phase,4).float())
                q=hand.low+(output['q'][:,-1]+1)*.5*(hand.high-hand.low)
                parts,_=contact_losses(hand,q,output['translation'][:,-1],output['rotation'][:,-1],[w[-1] for w in batch])
                # Same model handles neutral transition frames, with no
                # attraction objective in the neutral decoding branch.
                parts['classification']=torch.nn.functional.cross_entropy(output['action'].reshape(-1,ACTION_COUNT),neutral.flatten())
                parts['phase']=torch.nn.functional.cross_entropy(output['phase'].reshape(-1,4),phase.flatten())
                # Soft trend consistency, not equality of source/robot speed.
                dq=output['q'][:,1:]-output['q'][:,:-1]
                parts['motion']=parts['motion']+(dq.diff(dim=1).square().mean() if length>2 else dq.square().mean())*.1
                loss=sum(WEIGHTS.get(k,1.)*v for k,v in parts.items())
                if not torch.isfinite(loss):raise ValueError('CART loss 出现 NaN／Inf')
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);optimizer.step()
                count+=1
                for k,v in dict(total=loss,**parts).items():sums[k]=sums.get(k,0.)+float(v.detach())
            checkpoint=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch+1)
            temp=folder/'checkpoint.tmp';torch.save(checkpoint,temp);temp.replace(folder/'last.pth')
            torch.save(checkpoint,folder/f'epoch_{epoch+1:03d}.pth')
            progress('retargeting',epoch=epoch+1,epochs=epochs,losses={k:v/count for k,v in sums.items()})
        atomic_json(folder/'status.json',dict(status='smoke_test' if smoke else 'completed',run_id=run_id,epochs=epochs,
                    elapsed_s=time.time()-started,losses={k:v/count for k,v in sums.items()}))
        return folder
    except BaseException as exc:
        atomic_json(folder/'status.json',dict(status='cancelled' if isinstance(exc,InterruptedError) else 'failed',run_id=run_id,error=str(exc)))
        raise
    finally:
        for s,handler in old.items():signal.signal(s,handler)
        lock.close()

class Predictor:
    def __init__(self,folder,device='cpu'):
        self.folder=Path(folder);self.config=json.loads((self.folder/'config.json').read_text())
        status=json.loads((self.folder/'status.json').read_text())
        if status['status']!='completed' or self.config['profile']!='full':raise ValueError('只能启用完整训练成功的 CART 模型')
        validate_model_geometry(self.config)
        self.hand=NativeHand(device)
        if self.hand.robot.spec!=self.config['robot']:raise ValueError('CART Wuji2 模型或限位不匹配')
        self.device=device;self.model=CartNetwork().to(device)
        checkpoint=torch.load(self.folder/'last.pth',map_location=device,weights_only=True)
        self.model.load_state_dict(checkpoint['model']);self.model.eval();self.hidden=None;self.last_stamp=None
    def reset(self):self.hidden=None;self.last_stamp=None
    def encode(self,row):
        stamp=row['timestamp_ms']
        if self.last_stamp is not None and not 0<stamp-self.last_stamp<=250:self.reset()
        self.last_stamp=stamp
        x=torch.tensor(row['features'],device=self.device)[None,None]
        with torch.inference_mode():
            context,action,phase,self.hidden=self.model.encode(x,self.hidden)
        return context,action,phase
    def decode(self,encoded,target,stage_index=None):
        context,action,phase=encoded
        conditioning=action.softmax(-1)
        if stage_index is not None:conditioning=torch.nn.functional.one_hot(torch.tensor([[stage_index]],device=self.device),ACTION_COUNT).float()
        h=torch.tensor(hull_descriptor(target),device=self.device,dtype=torch.float32)[None,None]
        with torch.inference_mode():
            q,t,r=self.model.decode(context,conditioning,phase.softmax(-1),h)
        angle=self.hand.low+(q[0,0]+1)*.5*(self.hand.high-self.hand.low)
        return angle,t[0,0],r[0,0]

def validate_model_geometry(config):
    if config.get('protocol')!=PROTOCOL or config.get('geometry')!=geometry_identity():
        raise ValueError('CART 模型的分类协议／凸包估计器／模板版本不匹配，请按五类动作重新训练；旧模型不能启用')

def evaluate(root,model_id,ids,check=lambda:None):
    root=Path(root);predictor=Predictor(root/'models'/model_id)
    sessions,source=load_sessions(root,ids,'validation')
    if source!=predictor.config['source'] or set(ids)&set(predictor.config['train_ids']):raise ValueError('请使用同源独立验证会话')
    confusion=np.zeros((ACTION_COUNT,ACTION_COUNT),int);reports={};durations=[]
    for m,rows,_ in sessions:
        from .cart_data import SourceGeometry
        from .cart_protocol import MotionTracker
        folder=root/'recordings'/m['id']
        geometry=SourceGeometry(json.loads((folder/'topology.json').read_text()),json.loads((folder/'profile.json').read_text()))
        motion=np.load(folder/'motion.npz');tracker=MotionTracker();last_stage=STAGES[0]['id']
        predictor.reset();items=[]
        for i,row in enumerate(rows):
            check();started=time.perf_counter();encoded=predictor.encode(row)
            predicted=int(encoded[1][0,0].argmax());truth=NEUTRAL_INDEX if row['phase']=='open' else row['stage_index']
            confusion[truth,predicted]+=1
            chosen=last_stage if predicted==NEUTRAL_INDEX else STAGES[predicted]['id']
            if chosen!=last_stage:tracker.reset()
            last_stage=chosen
            inferred,_,_=geometry.frame(motion['qpos'][i],chosen,row['timestamp_ms'],tracker)
            q,t,r=predictor.decode(encoded,inferred['target'])
            with torch.no_grad():_,quality=contact_losses(predictor.hand,q[None],t[None],r[None],[row])
            dense=predictor.hand.audit(q)
            item=quality[0];item['dense_penetration_mm']=max(dense['max_penetration_mm'],item.get('object_penetration_mm',0))
            item['joint_limit_violations']=int(((q<predictor.hand.low)|(q>predictor.hand.high)).sum())
            items.append(item);durations.append((time.perf_counter()-started)*1000)
        reports.setdefault(str(m['category']),[]).extend(items)
    summary={}
    for category,rows in reports.items():
        active=[r for r in rows if r['phase'] in ('closing','hold')]
        gaps=[g for r in active for g in r['gaps_mm']];normals=[g for r in active for g in r['normal_errors_deg']]
        summary[category]=dict(frames=len(rows),contact_frames=len(active),contact_coverage=float(np.mean(np.asarray(gaps)<=2)) if gaps else None,
            contact_mean_mm=float(np.mean(gaps)) if gaps else None,contact_max_mm=max(gaps) if gaps else None,
            normal_mean_deg=float(np.mean(normals)) if normals else None,
            max_penetration_mm=max(r['dense_penetration_mm'] for r in rows),
            shape_mean_deg=float(np.mean([r['shape_error_deg'] for r in rows])),
            joint_limit_violations=sum(r['joint_limit_violations'] for r in rows),
            all_required_success_rate=float(np.mean([bool(r['required']) and max(r['gaps_mm'])<=2 and max(r['normal_errors_deg'])<=30
                and r['dense_penetration_mm']<=.5 and not r['object_intersection_over_05mm'] for r in active])) if active else None)
    result=dict(model=model_id,protocol=PROTOCOL,validation_ids=ids,paired_robot_ground_truth=False,
                action_labels=[s['id'] for s in STAGES]+['transition'],category_names={str(i+1):c for i,c in enumerate(CATEGORIES)},
                confusion_matrix=confusion.tolist(),categories=summary,
                end_to_end_ms=dict(mean=float(np.mean(durations)),p95=float(np.percentile(durations,95))))
    atomic_json(root/'models'/model_id/'evaluation.json',result);return result
