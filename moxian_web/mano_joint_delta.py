"""One mixed optimizer for shared geometry and a deployable IK delta network.

Policy observations contain same-frame measurements only. Action supervision
may mask derivatives, but never changes the numerical forward pass or inputs.
No independent per-frame pose parameters, online optimizer or online EMA.
"""
import numpy as np
import torch
import json
from .mano_calibrate_delta import DeltaPolicy,FEATURES
from .mano_mixed_training import MixedGPUManoObjective,masked_geometry
from .mano_objective import anatomical_rotations,pose_angles,neutral_rotations,rotation_log
from .card_math import exp_so3
from .contact import FINGERS,PATCH_IDS
from .glove_geometry import CHAINS

REVISION='mano_joint_geometry_ik_delta_v2'
JOINT_FINGERS=[next(f for f,c in enumerate(CHAINS) if j+1 in c) for j in range(15)]


class JointDeltaPolicy(DeltaPolicy):
    def forward(self,features):
        # Physical source/anatomical coordinates, rather than a separate 20deg
        # output box, bound the corrected joint configuration.
        return self.net((features-self.mean)/self.scale).reshape(-1,15,3)


def observable_inputs(rows,device):
    tactile=[];validity=[];interval=[]
    for row in rows:
        c=row.get('contact') or {};stamp=row.get('timestamp_ms',0);tt=row.get('tactile_timestamp_ms')
        fresh=bool(c.get('ready') and isinstance(tt,(float,int)) and abs(tt-stamp)<=25 and abs(c.get('timestamp_ms',tt)-stamp)<=25)
        patches={p['id']:p for p in c.get('patches',[])};response=[];usable=[]
        for key in PATCH_IDS:
            v=patches.get(key,{}).get('response');ok=fresh and isinstance(v,(float,int)) and np.isfinite(v)
            response.append(float(np.clip(v,0,1)) if ok else 0.);usable.append(float(ok))
        tactile.append(response+usable)
        validity.append([float(f in row.get('fingers',{}) and row.get('fingers',{}).get(f,{}).get('valid') is not False and row.get('fingers',{}).get(f,{}).get('confidence',1)>0) for f in FINGERS])
        # Both capture and runtime use the nominal interval when none is given.
        dt=row.get('delta_time_s',1/30);interval.append(float(dt) if isinstance(dt,(int,float)) and np.isfinite(dt) and dt>0 else 1/30)
    t=lambda a:torch.as_tensor(np.asarray(a),device=device,dtype=torch.float32)
    return t(tactile),t(validity),t(interval)[:,None]


def observable_features(seed,rawp,rawr,base_j,base_r,mp,mr,align,trans,inputs):
    ids=[c[-1] for c in CHAINS];distal=base_r[:,ids]
    predicted=base_j[:,ids]+(distal@mp[None,:,:,None]).squeeze(-1)
    position=rawp@align.T+trans;rotation=align@rawr
    error=torch.cat(((predicted-position).flatten(1),rotation_log((distal@mr).transpose(-1,-2)@rotation).flatten(1)),1)
    return torch.cat((seed.flatten(1),position.flatten(1),rotation[...,:2].flatten(1),error,*inputs),1)


class PoseChart:
    def __init__(self,model,rows,poses,reference=None,objective=None):
        t=lambda a:torch.as_tensor(np.asarray(a),device=model.device,dtype=torch.float32)
        if objective is not None:
            self.basis=objective.basis;self.neutral=objective.neutral;self.low=objective.low;self.high=objective.high
            self.seed=objective.seed_angles.clamp(self.low,self.high);self.thumb=objective.thumb_chart
        else:
            from .thumb_chart import source_thumb_chart,SLOTS
            self.basis=t(reference['basis']);self.neutral=t(reference['neutral'])
            self.low=t(reference['low']);self.high=t(reference['high'])
            q=pose_angles(model,np.asarray(poses),np.asarray(reference['neutral']))
            self.thumb=source_thumb_chart(model,rows)
            if self.thumb is not None:
                q[:,12:]=0
                for k,(j,c) in enumerate(SLOTS):q[:,j,c]=self.thumb.seeds[:,k]
            self.seed=t(q).clamp(self.low,self.high)
        self.base=self.rotations(self.seed).detach()

    def rotations(self,angles):
        r=anatomical_rotations(angles,self.basis,self.neutral)
        if self.thumb is not None:r=torch.cat((r[:,:12],self.thumb.decode(angles,None)),1)
        return r

    def correct(self,seed,delta,validity):
        delta=delta*validity[:,JOINT_FINGERS,None]
        angles=(self.seed+delta).clamp(self.low,self.high)
        effective=angles-self.seed
        original=exp_so3(seed)
        local=original@(self.base.transpose(-1,-2)@self.rotations(angles))
        zero=original@(self.base.transpose(-1,-2)@self.base)
        # Subtract deterministic FP32 chart roundoff. Zero head is exactly IK
        # while its derivatives remain live from the first optimizer update.
        pose=seed+(rotation_log(local)-rotation_log(zero))
        # CUDA GEMM may choose different FP32 kernels for cached and freshly
        # computed chart rotations. Remove their zero-step roundoff without
        # suppressing the residual head's gradient at initialization.
        at_zero=(effective.detach()==0).all(-1,keepdim=True)
        pose=torch.where(at_zero,seed+(pose-pose.detach()),pose)
        return pose,effective

    def reference(self):
        return {k:getattr(self,k).detach().cpu().tolist() for k in ('basis','neutral','low','high')}

    def reseed(self,model,rows,poses,reference):
        """Reuse fixed axes and source transforms; update measured angles only."""
        from .thumb_chart import SLOTS,SUFFIXES
        q=pose_angles(model,np.asarray(poses),np.asarray(reference['neutral']))
        if self.thumb is not None:
            names=[model.side+'_thumb_'+s for s in SUFFIXES]
            seeds=np.array([[dict(zip(r['skeleton']['joint_names'],r['skeleton']['joint_positions_rad']))[n] for n in names] for r in rows])
            if not np.isfinite(seeds).all() or np.any(seeds<self.thumb.bounds[:,0]-1e-5) or np.any(seeds>self.thumb.bounds[:,1]+1e-5):raise ValueError('Source thumb joint coordinates exceed bounds')
            q[:,12:]=0
            for k,(j,c) in enumerate(SLOTS):q[:,j,c]=seeds[:,k]
        self.seed=torch.as_tensor(q,device=model.device,dtype=torch.float32).clamp(self.low,self.high)
        self.base=self.rotations(self.seed).detach()
        return self


class JointGPUManoObjective(MixedGPUManoObjective):
    joint_training=True

    def __init__(self,*args,**kwargs):
        initial_policy=kwargs.pop('initial_policy',None)
        super().__init__(*args,**kwargs)
        if initial_policy is None:self.joint_policy=JointDeltaPolicy().to(self.model.device)
        else:
            from .mano_calibrate_delta import load_policy
            if initial_policy.get('revision')!=REVISION:raise ValueError('续训需要联合 Delta checkpoint')
            self.joint_policy=load_policy(initial_policy,self.model.device).train()
        self.policy_inputs=observable_inputs(self.rows,self.model.device)
        self.pose_chart=PoseChart(self.model,self.rows,self.feature_seed_numpy,
            objective=self if initial_policy is None else None,
            reference=initial_policy['pose_chart'] if initial_policy else None)
        # Masks change gradients only. All forward values match deployed policy.
        self.delta_gradient_mask=self.t(self.mask_np)[:,JOINT_FINGERS,None]
        origin=self.rows[0]['timestamp_ms'];stamps=self.t([(r['timestamp_ms']-origin)/1000 for r in self.rows]);dt=stamps[1:]-stamps[:-1]
        self.rate_dt=dt.clamp_min(1e-6);self.rate_valid=((dt>0)&(dt<.25))[:,None,None]
        same_window=self.t([a.get('window_id')==b.get('window_id') for a,b in zip(self.rows[:-1],self.rows[1:])]).bool()
        self.rate_valid=self.rate_valid&same_window[:,None,None]
        with torch.no_grad():
            self.surface(self.initial)
            obs=self.latest_policy_observations
            if initial_policy is None:
                self.joint_policy.mean.copy_(obs.mean(0));self.joint_policy.scale.copy_(obs.std(0,unbiased=False).clamp_min(.01))

    def refresh(self,x,contacts=False):
        if not hasattr(self,'joint_policy'):return super().refresh(x,contacts)
        with torch.no_grad():
            _,v,j,_,_=self.surface(x)
            self.gpu_witness=self.gpu_collision.query(v)
            self.witnesses=[];self.collision_reports=[]

    def surface(self,x):
        if self._surface_cache is not None:return self._surface_cache
        if not hasattr(self,'joint_policy'):return super().surface(x)
        beta,mp,mr,align,trans,angles=self.unpack(x);betas,g=masked_geometry(x,self.frame_geometry_mask)
        model=self.model.model
        rest,rest_j=torch.vmap(model.official_rest)(betas,g)
        self._mixed_rest=rest,rest_j
        with torch.no_grad():
            _,base_j,base_r=model.official_batched_rest_forward(exp_so3(self.seed_pose),rest.detach(),rest_j.detach())
            base_j=(base_j-rest_j.detach()[:,:1])@self.root.transpose(-1,-2);base_r=self.root[:,None]@base_r
            obs=observable_features(self.seed_pose,self.rawp,self.rawr,base_j,base_r,mp,mr,align,trans,self.policy_inputs)
        self.latest_policy_observations=obs
        delta=self.joint_policy(obs)
        delta=delta.detach()+self.delta_gradient_mask*(delta-delta.detach())
        self.latest_raw_delta=delta
        pose,self.latest_delta=self.pose_chart.correct(self.seed_pose,delta,self.policy_inputs[1])
        v,j,r=model.official_batched_rest_forward(exp_so3(pose),rest,rest_j)
        v=(v-rest_j[:,:1])@self.root.transpose(-1,-2);j=(j-rest_j[:,:1])@self.root.transpose(-1,-2);r=self.root[:,None]@r
        return pose,v,j,r,(beta,mp,mr,align,trans,angles)

    def residual(self,x,record=False,measurement_only=False):
        base=super().residual(x,record,measurement_only)
        if measurement_only:return base
        d=self.latest_delta
        if record:
            raw_angles=self.pose_chart.seed+self.latest_raw_delta
            active=self.delta_gradient_mask.expand_as(raw_angles).bool()
            saturated=((raw_angles<=self.pose_chart.low)|(raw_angles>=self.pose_chart.high))&active
            self.last_training_metrics['joint_limit_saturation_fraction']=float(saturated.sum()/active.sum().clamp_min(1))
        regular=d/.5*(.35/d.numel()**.5)
        # Train-only temporal neighbors; gaps from held-out cycles reset pairs.
        rate=torch.relu((d[1:]-d[:-1]).abs()-self.settings.correction_rate_deg_s*np.pi/180*self.rate_dt[:,None,None])/.035
        rate=rate*self.rate_valid/max(1,rate.numel())**.5
        if record:
            self.last_terms.update(delta_pose_prior=float(regular.detach().square().sum()),delta_rate_prior=float(rate.detach().square().sum()))
            self.last_training_metrics['pose_delta_rms_deg']=float(d.detach().square().mean().sqrt()*180/np.pi)
        return torch.cat((base,regular.flatten(),rate.flatten()))

    def artifact(self):
        return dict(revision=REVISION,features=FEATURES,state={k:v.detach().cpu().tolist() for k,v in self.joint_policy.state_dict().items()},
                    pose_chart=self.pose_chart.reference(),normalization_partition='train',joint_geometry_training=True,
                    online_updates=False,rate_constraint='causal_neighbor_soft_loss_with_gap_reset',inputs='same_frame_observables_only')


@torch.no_grad()
def infer(model,rows,poses,roots,*,fast=True):
    """Identical learned correction in validation and live fixed forward."""
    artifact=getattr(model,'delta_artifact',None);policy=getattr(model,'delta_policy',None)
    t=lambda a:torch.as_tensor(np.asarray(a),device=model.device,dtype=torch.float32)
    seed=t(poses);root=t(roots)
    if policy is None or artifact is None:return seed
    if artifact.get('revision')!=REVISION:raise ValueError('固定前向要求联合 Delta 权重')
    if artifact.get('surface_version',model.shape_version)!=model.shape_version or artifact.get('weights_hash',model.weights_hash)!=model.weights_hash or artifact.get('binding_version',model.binding_version)!=model.binding_version or artifact.get('mapping_version',(model.mapping or {}).get('version'))!=(model.mapping or {}).get('version') or any(r.get('calibration_fingerprint')!=artifact.get('calibration_fingerprint',r.get('calibration_fingerprint')) for r in rows):
        model._delta_disabled_reason='Delta 策略与当前模型／标定版本不匹配'
        return seed
    model._delta_disabled_reason=None
    if fast:
        # Policy features need bone centres/rotations, not an entire uncorrected
        # skinned mesh. Keep the final mesh as the sole LBS pass.
        local=torch.cat((torch.eye(3,device=model.device)[None,None].expand(len(seed),1,3,3),exp_so3(seed)),1)
        j,transforms=model.model.rigid_transforms(local,model.rest_j[None].expand(len(seed),-1,-1))
        j=(j-model.rest_j[0])@root.transpose(-1,-2);r=root[:,None]@transforms[...,:3,:3]
    else:_,j,r,_=model.decode(seed,root)
    from scipy.spatial.transform import Rotation
    rawp=t([[row.get('fingers',{}).get(f,{}).get('position',[0,0,0]) for f in FINGERS] for row in rows])
    rawr=t([Rotation.from_quat([row.get('fingers',{}).get(f,{}).get('quaternion_xyzw',[0,0,0,1]) for f in FINGERS]).as_matrix() for row in rows])
    inputs=observable_inputs(rows,model.device)
    obs=observable_features(seed,rawp,rawr,j,r,t(model.sensor_offsets()),t(model.mount_rot),t(model.raw_rotation),t(model.raw_translation),inputs)
    if fast and len(rows)==1:
        key=(model.shape_version,model.binding_version,getattr(model.official_fk,'binding_hash',None),id(artifact),json.dumps(rows[0].get('skeleton',{}).get('thumb_kinematics'),sort_keys=True))
        cached=getattr(model,'_delta_runtime_chart',None)
        if cached is None or cached[0]!=key:
            chart=PoseChart(model,rows,np.asarray(poses),reference=artifact['pose_chart']);model._delta_runtime_chart=(key,chart)
        else:chart=cached[1].reseed(model,rows,poses,artifact['pose_chart'])
    else:chart=PoseChart(model,rows,np.asarray(poses),reference=artifact['pose_chart'])
    return chart.correct(seed,policy(obs),inputs[1])[0]
