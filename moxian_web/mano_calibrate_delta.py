"""IK residual policy calibration; it never learns a replacement for base IK."""
import copy
import time
import numpy as np
import torch
from torch import nn

from .contact import FINGERS,PATCH_IDS
from .mano_objective import rotation_log

REVISION='mano_ik_delta_v1'
FEATURES=45+15+30+30+2*len(PATCH_IDS)+5+1


class DeltaPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(FEATURES,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,45))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
        self.register_buffer('mean',torch.zeros(FEATURES));self.register_buffer('scale',torch.ones(FEATURES))

    def forward(self,features):
        return .35*torch.tanh(self.net((features-self.mean)/self.scale)).reshape(-1,15,3)


def features(objective,shared):
    """Only same-frame observable values; no gesture, cycle, partition or future."""
    with torch.no_grad():
        seed=objective.initial.clone();seed[:objective.shared_size]=shared[:objective.shared_size]
        beta,mp,mr,align,trans,_=objective.unpack(seed)
        if not hasattr(objective,'feature_angles'):
            from .mano_objective import pose_angles
            raw=getattr(objective,'feature_seed_numpy',None)
            if raw is None:raw=objective.seed_pose.detach().cpu().numpy()
            angle=objective.t(pose_angles(objective.model,raw,objective.neutral_np))
            if objective.thumb_chart is not None:
                from .thumb_chart import SLOTS
                angle[:,12:]=0.
                for k,(joint,coordinate) in enumerate(SLOTS):angle[:,joint,coordinate]=objective.t(objective.thumb_chart.seeds[:,k])
            objective.feature_angles=angle.clamp(objective.low,objective.high)
        from .mano_objective import decode
        geometry=seed[46:objective.shared_size] if objective.shared and objective.personalized else objective.geometry0
        _,_,j,r=decode(objective.model,beta,objective.feature_angles,objective.root,objective.basis,objective.neutral,objective.thumb_chart,geometry)
        from scripts.calibrate_wuji_mano import CHAINS
        distal=r[:,[c[-1] for c in CHAINS]]
        predicted=j[:,[c[-1] for c in CHAINS]]+(distal@mp[None,:,:,None]).squeeze(-1)
        position=objective.rawp@align.T+trans
        rotation=align@objective.rawr
        error=torch.cat(((predicted-position).reshape(len(j),-1),rotation_log((distal@mr).transpose(-1,-2)@rotation).reshape(len(j),-1)),1)
        tactile=[];times=[];validity=[]
        for row in objective.rows:
            validity.append([float(row.get('fingers',{}).get(f,{}).get('valid') is not False and row.get('fingers',{}).get(f,{}).get('confidence',1)>0) for f in FINGERS])
            c=row.get('contact') or {};stamp=row.get('timestamp_ms',0);tt=row.get('tactile_timestamp_ms')
            valid=c.get('ready') and isinstance(tt,(float,int)) and abs(tt-stamp)<=25 and abs(c.get('timestamp_ms',tt)-stamp)<=25
            patches={p['id']:p for p in c.get('patches',[])}
            responses=[patches.get(k,{}).get('response') for k in PATCH_IDS]
            usable=[bool(valid and isinstance(v,(float,int)) and np.isfinite(v)) for v in responses]
            tactile.append([float(np.clip(v,0,1)) if ok else 0. for v,ok in zip(responses,usable)]+[float(ok) for ok in usable])
            # Missing timing is represented as one nominal interval, not a future difference.
            times.append(float(row.get('delta_time_s',1/30)))
        return torch.cat((objective.feature_angles.flatten(1),position.flatten(1),rotation[...,:2].flatten(1),error,
                          objective.t(tactile),objective.t(validity),objective.t(times)[:,None]),1)


def corrected_angles(objective,policy,observations):
    return (objective.seed_angles+policy(observations)*objective.policy_mask).clamp(objective.low,objective.high)


def pose_vector(objective,shared,policy,observations):
    angles=corrected_angles(objective,policy,observations)
    angles=angles.clamp(objective.low,objective.high)
    angles=torch.where(objective.free,angles,objective.seed_angles)
    return torch.cat((shared[:objective.shared_size],angles.flatten().index_select(0,objective.free_indices)))


def train_policy(objective,shared,cancel,progress,max_seconds=15,steps=240,compile_loss=True):
    if any(r.get('partition')!='train' for r in objective.rows):raise ValueError('Delta buffer 只能包含训练分区')
    observations=features(objective,shared)
    policy=DeltaPolicy().to(objective.model.device)
    with torch.no_grad():policy.mean.copy_(observations.mean(0));policy.scale.copy_(observations.std(0,unbiased=False).clamp_min(.01))
    optimizer=torch.optim.Adam(policy.parameters(),lr=.001)
    def loss_fn():
        vector=pose_vector(objective,shared,policy,observations)
        delta=policy(observations)
        return objective.residual(vector).square().sum()+.01*delta.square().mean()
    loss=torch.compile(loss_fn,options={'triton.cudagraphs':False},dynamic=False) if compile_loss else loss_fn
    began=time.monotonic();deadline=began+max_seconds;best=None;best_cost=float('inf');compile_s=0.;stale=0;previous=None;exhausted=False
    for step in range(steps):
        if cancel.is_set():raise InterruptedError('Delta policy 训练已取消')
        if step and time.monotonic()>=deadline:exhausted=True;break
        if step%8==0:
            with torch.no_grad():objective.refresh(pose_vector(objective,shared,policy,observations))
        if step==0 and compile_loss:progress(dict(state='running',phase='compile',progress=.75,message='GPU Delta policy 编译／预热',estimated_remaining_s=None))
        optimizer.zero_grad(set_to_none=True);start=time.monotonic();value=loss();value.backward()
        if cancel.is_set():raise InterruptedError('Delta policy 训练已取消')
        if step==0 and compile_loss:torch.cuda.synchronize();compile_s=time.monotonic()-start;deadline+=compile_s
        if not bool(torch.isfinite(value)) or not all(bool(torch.isfinite(p.grad).all()) for p in policy.parameters() if p.grad is not None):raise ValueError('Delta policy 梯度非有限')
        cost=float(value.detach())
        if cost<best_cost:best_cost=cost;best=copy.deepcopy(policy.state_dict())
        torch.nn.utils.clip_grad_norm_(policy.parameters(),10.);optimizer.step()
        if step%20==0:
            stale=stale+1 if previous is not None and abs(previous-cost)/max(abs(previous),1e-8)<1e-4 else 0;previous=cost
            progress(dict(state='running',phase='delta',progress=.75+.04*(step+1)/steps,message=f'IK Delta policy · {step+1}/{steps} 步',estimated_remaining_s=max(0.,deadline-time.monotonic()),eta_scope='optimization_budget'))
            if step>=80 and stale>=3:break
    if best is not None:policy.load_state_dict(best)
    artifact=dict(revision=REVISION,features=FEATURES,state={k:v.detach().cpu().tolist() for k,v in policy.state_dict().items()},
                  normalization_partition='train',steps=step+1,budget_exhausted=exhausted,compilation_s=compile_s)
    return policy,artifact


def load_policy(artifact,device):
    from .mano_joint_delta import REVISION as JOINT_REVISION,JointDeltaPolicy
    if artifact.get('revision') not in (REVISION,JOINT_REVISION) or artifact.get('features')!=FEATURES:raise ValueError('Delta policy 版本不匹配')
    net=(JointDeltaPolicy() if artifact['revision']==JOINT_REVISION else DeltaPolicy()).to(device)
    if artifact['revision']==JOINT_REVISION:
        chart=artifact.get('pose_chart',{})
        for key,shape in dict(basis=(15,3,3),neutral=(15,3,3),low=(15,3),high=(15,3)).items():
            value=np.asarray(chart.get(key))
            if value.shape!=shape or not np.isfinite(value).all():raise ValueError('Delta 关节坐标参考无效')
        if np.any(np.asarray(chart['low'])>np.asarray(chart['high'])):raise ValueError('Delta 关节限位无效')
    expected=net.state_dict()
    if set(expected)!=set(artifact.get('state',{})):raise ValueError('Delta policy 权重字段不匹配')
    state={k:torch.as_tensor(artifact['state'][k],device=device,dtype=torch.float32) for k in expected}
    if any(v.shape!=expected[k].shape or not bool(torch.isfinite(v).all()) for k,v in state.items()):raise ValueError('Delta policy 权重无效')
    if bool((state['scale']<=0).any()):raise ValueError('Delta policy 归一化无效')
    net.load_state_dict(state);return net.eval()


def fit_capture(model_path,profile,mapping,rows,progress,cancel,**kwargs):
    from .mano_calibration import run_backend
    return run_backend('delta',model_path,profile,mapping,rows,progress,cancel,**kwargs)
