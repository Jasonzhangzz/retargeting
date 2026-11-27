"""Shared current CART rollout, checkpoint I/O, and deployed-policy inference."""
import json
from pathlib import Path
import numpy as np
import torch
from .cart_data import load_sessions,manifest
from .cart_network import CartNetwork
from .cart_wrist import FloatingCartHand,state_velocity
from .cart_objective import CartHand
from .cart_emf_observations import batch_future,pack_future
from .cart_fast import PreparedBatch,prepare_windows
from .cart_initialization import attach_ik
from retarget.baseline.geort.training import atomic_json

def neutral(hand):
    return hand.tensor(hand.robot.model.qpos0[6:]).clamp(hand.low,hand.high)


class RolloutTrajectory:
    """Sixteen-step recurrent states and clean commands before process noise."""
    def __init__(self,pending,batch,normalizers=None,controls=None,commands=None):
        self.pending=pending;self.batch=batch;self.normalizers=normalizers
        self.controls=controls;self.commands=commands
        self.predictions=[x[0] for x in pending]


def perturb_rollout_state(model,q,noise_rad):
    """Independent additive finger-joint process noise; retain the BPTT graph.

    q26 wrist translation/rotation keep their physical units and are untouched.
    Only physical joint limits can truncate a sampled perturbation. The random
    draw has no gradient; the state does (apart from saturated joint limits).
    """
    if isinstance(noise_rad,bool) or not np.isfinite(noise_rad) or not 0<=noise_rad<=.1:
        raise ValueError('Per-step finger noise must be in [0, 0.1] rad')
    if noise_rad==0:return q
    noise=torch.empty_like(q[...,:20]).uniform_(-noise_rad,noise_rad)
    finger=(q[...,:20]+noise).clamp(model.lower,model.upper)
    return torch.cat((finger,q[...,20:]),dim=-1)


def predict_trajectory(model,objective,batch,normalizers=None,*,batch_observations=True,step_noise_rad=0.):
    """Unroll only the recurrent policy; no geometry graph or horizon detach."""
    length=min(w['length'] for w in batch);device=objective.hand.device
    encoder_batch=getattr(model,'observation_chunk_size',512 if str(device).startswith('cuda') else 64)
    if encoder_batch<1:raise ValueError('Observation chunk size must be positive')
    offsets=[len(w['rows'])-w['length']-10 for w in batch]
    prepared=isinstance(batch,PreparedBatch)
    def inputs_at(indices):
        values=batch.future_inputs(indices) if prepared else batch_future(batch,indices,device)
        if prepared:values['validate']=False
        return values
    initialized=['initial_q' in w for w in batch]
    if any(initialized) and not all(initialized):raise ValueError('训练窗口不能混合 IK 与中性姿态初始化')
    if all(initialized):
        q=torch.stack([w['initial_q'].to(device) if isinstance(w['initial_q'],torch.Tensor) else objective.hand.tensor(w['initial_q']) for w in batch]).detach().clone()
        q=model.initial_state(q)
    elif prepared:
        q=model.initial_state(neutral(objective.hand)[None].expand(len(batch),-1).clone())
        with torch.no_grad():
            for j in range(max(offsets)):
                candidate=model.apply_delta(q,model(q,**inputs_at([min(j,start) for start in offsets])))
                active=torch.as_tensor([j<start for start in offsets],device=device)[:,None]
                q=torch.where(active,candidate,q)
    else:
        initial=[]
        for w,start in zip(batch,offsets):
            one=model.initial_state(neutral(objective.hand)[None])
            with torch.no_grad():
                for j in range(start):one=model.apply_delta(one,model(one,**batch_future([w],[j],device)))
            initial.append(one[0])
        q=torch.stack(initial)
    older_q=None;old_dt=None;pending=[];commands=[];cached_inputs=None;encoded=None
    controls=[] if getattr(objective,'command_smoothing',False) else None;old_control_velocity=None
    if batch_observations and prepared and getattr(model,'deduplicate_observations',False):
        # Share identical observation encodings inside this graph. Never cache
        # trainable activations across optimizer steps or detach their gradients.
        unique=[];lookup={};inverse=[]
        for t in range(length):
            for w,offset in zip(batch,offsets):
                ids=w['indices'][offset+t:offset+t+11];key=(id(w['store']),tuple(ids))
                if key not in lookup:
                    lookup[key]=len(unique);unique.append(dict(store=w['store'],indices=ids))
                inverse.append(lookup[key])
        prepared_unique=PreparedBatch(unique,device);values=prepared_unique.future_inputs([0]*len(unique))
        pieces=[model.encode_prepared_observations(**{key:value[start:start+encoder_batch] for key,value in values.items()})
                for start in range(0,len(unique),encoder_batch)]
        gather=torch.tensor(inverse,device=device)
        encoded=tuple(torch.cat([part[k] for part in pieces])[gather] for k in range(len(pieces[0])))
        times=values['future_dt'][gather]
        cached_inputs=[dict(future_dt=times[t*len(batch):(t+1)*len(batch)]) for t in range(length)]
    elif batch_observations and prepared and getattr(model,'architecture',None)=='attention':
        # The expensive EMF/taxel cross attention is independent of robot state.
        # Batch it over TIME as well as windows; only the small q pool/decoder
        # remains in the causal recurrence. Every slice still sees its own
        # measured t+1...t+10 inputs, with the same masks and parameter gradients.
        cached_inputs=[inputs_at([i+t for i in offsets]) for t in range(length)]
        values={key:torch.cat([x[key] for x in cached_inputs]) for key in cached_inputs[0] if key!='validate'}
        pieces=[model.encode_prepared_observations(**{key:value[start:start+encoder_batch] for key,value in values.items()})
            for start in range(0,len(batch)*length,encoder_batch)]
        encoded=tuple(torch.cat([part[k] for part in pieces]) for k in range(len(pieces[0])))
    for t in range(length):
        indices=[i+t for i in offsets]
        rows=[w['rows'][i+1] for w,i in zip(batch,indices)]
        previous=[w['rows'][i] for w,i in zip(batch,indices)]
        inputs=cached_inputs[t] if cached_inputs is not None else inputs_at(indices)
        if encoded is not None:
            start=t*len(batch);stop=start+len(batch)
            delta=model.decode_observation_context(q,tuple(x[start:stop] for x in encoded))
        else:delta=model(q,**inputs)
        command_q=model.apply_delta(q,delta)
        commands.append(command_q)
        next_q=perturb_rollout_state(model,command_q,step_noise_rad)
        dt=inputs['future_dt'][:,0]
        if controls is not None:
            control_velocity=state_velocity(command_q,q,dt)
            valid=dt.new_full(dt.shape,old_control_velocity is not None,dtype=torch.bool)
            control_acceleration=torch.zeros_like(control_velocity) if old_control_velocity is None else (control_velocity-old_control_velocity)/(.5*(dt+old_dt))[:,None]
            controls.append((control_velocity,control_acceleration,valid));old_control_velocity=control_velocity
        if next_q.shape[-1]==26:
            velocity=state_velocity(next_q,q,dt)
            acceleration=(velocity-state_velocity(q,older_q,old_dt))/(.5*(dt+old_dt))[:,None] if older_q is not None else None
        else:
            velocity=(next_q-q)/dt[:,None]
            acceleration=(velocity-(q-older_q)/old_dt[:,None])/(.5*(dt+old_dt))[:,None] if older_q is not None else None
        pending.append((next_q,rows,q,acceleration,previous))
        older_q,q,old_dt=q,next_q,dt
    weights=batch.normalizers if prepared and batch.normalizers is not None else normalizers
    return RolloutTrajectory(pending,batch,weights,controls,commands)


def windows(sessions,length=16,stride=0):
    if isinstance(stride,bool) or not isinstance(stride,int) or stride<0:
        raise ValueError('窗口步长必须为非负整数；0 表示 length//2')
    stride=stride or max(1,length//2)
    result=[]
    for meta,rows,_ in sessions:
        # Sampling density belongs to training, not the immutable observation
        # cache. Keep warmup and the complete ten-frame future tail in-segment.
        segments=[];start=0
        for end in range(1,len(rows)+1):
            if end==len(rows) or not 0<rows[end]['timestamp_ms']-rows[end-1]['timestamp_ms']<=250 or rows[end].get('stream_epoch')!=rows[end-1].get('stream_epoch'):
                segments.extend((i,start) for i in range(start,end-length-9,stride));start=end
        for i,start in segments:
            indices=list(range(max(start,i-10),i+length+10))
            current=rows[indices[len(indices)-length-10]]
            if current.get('ik_valid') is False:continue
            window=dict(rows=[rows[i] for i in indices],indices=indices,length=length,dataset=meta['id'])
            if 'ik_q_rad' in current:window['initial_q']=current['ik_q_rad']
            result.append(window)
    return result


def position_statistics(sessions):
    points=np.concatenate([np.asarray(r['tactile_positions_m'],np.float32)[np.asarray(r['tactile_valid'],bool)]
        for _,rows,_ in sessions for r in rows])
    if not len(points):return np.zeros(3,np.float32),np.ones(3,np.float32)*.1
    return points.mean(0),np.maximum(points.std(0),.001)


def save_checkpoint(path,values):
    temp=Path(str(path)+'.tmp');torch.save(values,temp);temp.replace(path)


def validate_model_geometry(config):
    from .cart_deployment import validate_architecture
    validate_architecture(config)
    raise ValueError('CART weights require an explicit inference package with current implementation hashes')


class Predictor:
    """Stateless future-window API; default output is normalized delta q."""
    def __init__(self,folder,device='cpu',allow_smoke=False):
        self.folder=Path(folder);self.device=device
        self.config=json.loads((self.folder/'config.json').read_text());status=json.loads((self.folder/'status.json').read_text())
        if self.config.get('floating_wrist'):
            raise ValueError('Floating-wrist weights require a wrist-aware arm runtime; use cart_jobs.evaluate for offline validation')
        from .cart_deployment import validate_package
        self.deployment=validate_package(self.folder,allow_smoke=allow_smoke)
        if not allow_smoke and self.config['profile']!='full':raise ValueError('测试模型不能启用')
        if self.deployment is None:validate_model_geometry(self.config)
        self.hand=CartHand(device);self.robot=self.hand.robot
        if self.robot.spec!=self.config['robot'] or self.hand.domain!=self.config['output_domain']:raise ValueError('CART 机器人／关节域不匹配')
        domain=self.config['output_domain']
        self.model=CartNetwork(domain['lower'],domain['upper'],self.config['architecture_name'],
            position_mean=self.config['position_mean'],position_scale=self.config['position_scale']).to(device)
        self.model.set_attention_precision(self.config.get('policy_attention_precision','float32'))
        checkpoint=torch.load(self.folder/'last.pth',map_location=device,weights_only=True)
        self.model.load_state_dict(checkpoint['model']);self.model.eval()

    def predict(self,current_q,future_emf,emf_valid,future_dt,tactile_positions=None,tactile_state=None,tactile_valid=None,*,normalized=True,output='delta'):
        if output not in ('delta','next'):raise ValueError('output must be delta or next')
        q=np.asarray(current_q,np.float32);single=q.shape==(20,)
        values=dict(current_q=q,future_emf=future_emf,emf_valid=emf_valid,future_dt=future_dt,
            tactile_positions=tactile_positions,tactile_state=tactile_state,tactile_valid=tactile_valid)
        args={}
        for k,value in values.items():
            if value is None:continue
            tensor=torch.as_tensor(np.asarray(value),device=self.device,dtype=torch.bool if k.endswith('valid') else torch.float32)
            args[k]=tensor[None] if single else tensor
        with torch.inference_mode():
            result=self.model(**args)
            if output=='next':result=self.model.apply_delta(args['current_q'],result,normalized=normalized)
            elif not normalized:result=self.model.delta_to_radians(result)
            result=result.cpu().numpy()
        return result[0] if single else result

    def predict_next(self,*args,**kwargs):
        return self.predict(*args,**kwargs,output='next')

    def predict_rows(self,q,current_stamp,rows):
        inputs=pack_future(current_stamp,rows,self.device)
        with torch.inference_mode():
            current=self.hand.tensor(q)[None]
            delta=self.model(current,**{k:v[None] for k,v in inputs.items()})
            return self.model.apply_delta(current,delta)[0].cpu().numpy()
