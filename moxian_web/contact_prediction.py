"""Pose-only contact inference and time-aligned, read-only tactile comparison.

The raw official MANO rotations are captured BEFORE tactile pose correction.
The glove has no world wrist tracker: its wrist is explicitly held at identity.
Taxels sample probabilities at their existing fixed triangle/barycentric binding.
"""
from collections import deque
from pathlib import Path
import hashlib
import time
import numpy as np

MODEL_SHA256='cd1764aeb3e335792f53f46b087cc9fe5a535dd88e4076cdf36b70ccb1f590ba'


def sensor_pose51(hand):
    if hand.get('coordinate_frame')!='mano_wrist_local':raise ValueError('需要 MANO 腕部局部坐标')
    if hand.get('hand_side') not in ('left','right'):raise ValueError('缺少有效手侧')
    # Never fall back to pose_axis_angle: that pose can use measured tactile.
    pose=np.asarray(hand.get('raw_official_pose_axis_angle'),np.float32)
    if pose.shape!=(15,3) or not np.isfinite(pose).all():raise ValueError('缺少触觉修正前的 45D MANO 姿态')
    value=np.r_[np.zeros(6,np.float32),pose.reshape(45)]
    if hand['hand_side']=='left':
        value[6:]=(value[6:].reshape(15,3)*[1,-1,-1]).reshape(45)
    return value.astype(np.float32)


def taxel_probabilities(probability,topology):
    p=np.asarray(probability,float);faces=np.asarray(topology['faces'],int)
    ids=np.asarray(topology['sid_face_ids'],int);bary=np.asarray(topology['sid_barycentric'],float)
    if p.shape!=(778,) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('无效的 778 顶点概率')
    if faces.ndim!=2 or faces.shape[1]!=3 or np.any((faces<0)|(faces>=778)):raise ValueError('无效 MANO 拓扑')
    if ids.shape!=(460,) or np.any((ids<0)|(ids>=len(faces))):raise ValueError('无效 SID 面索引')
    if bary.shape!=(460,3) or not np.isfinite(bary).all() or np.any(bary<0) or not np.allclose(bary.sum(1),1,atol=1e-5):raise ValueError('无效 SID 材料坐标')
    return (p[faces[ids]]*bary).sum(1)


def confusion(probability,target,mask,threshold=.52):
    p=np.asarray(probability)>=threshold;y=np.array([v is True for v in target]);mask=np.asarray(mask,bool)
    if not mask.any():return None
    tp=int((mask&p&y).sum());fp=int((mask&p&~y).sum());fn=int((mask&~p&y).sum());tn=int((mask&~p&~y).sum())
    return metrics(dict(tp=tp,fp=fp,fn=fn,tn=tn))


def metrics(counts):
    tp,fp,fn,tn=[counts[k] for k in ('tp','fp','fn','tn')]
    return dict(counts,precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                accuracy=(tp+tn)/(tp+fp+fn+tn),samples=tp+fp+fn+tn)


class ContactPredictor:
    def __init__(self,predict,threshold=.52):
        self.predict=predict;self.threshold=threshold;self.history=deque(maxlen=8)
        self.session=None;self.last_timestamp=None;self.last_result=None;self.frames=0;self.clear_metrics()

    def clear_metrics(self):
        self.totals={name:dict(tp=0,fp=0,fn=0,tn=0) for name in ('illustrative','verified')};self.scored_frames=0

    def step(self,hand,topology,history_length=8):
        if history_length not in (1,8):raise ValueError('历史长度必须为 1 或 8')
        stamp=float(hand['timestamp_ms'])
        if not np.isfinite(stamp):raise ValueError('无效 MANO 时间戳')
        signature=(hand['hand_side'],hand.get('geometry_version'),hand.get('binding_version'),
                   hand.get('skeleton_runtime_revision'),hand.get('pose_calibration_version'),history_length)
        if self.session==signature and self.last_timestamp==stamp:return dict(self.last_result,duplicate=True)
        if topology.get('hand_side',hand['hand_side'])!=hand['hand_side'] or topology.get('binding_version',hand.get('binding_version'))!=hand.get('binding_version'):
            raise ValueError('预测与 SID 映射的手侧或版本不匹配')
        if (signature!=self.session or self.last_timestamp is not None and
            (stamp<self.last_timestamp or stamp-self.last_timestamp>250)):
            self.history.clear();self.clear_metrics();self.session=signature
        pose=sensor_pose51(hand);self.history.append(pose);self.last_timestamp=stamp
        base=dict(timestamp_ms=stamp,hand_side=hand['hand_side'],binding_version=hand.get('binding_version'),
                  history_frames=min(len(self.history),history_length),history_required=history_length,
                  pose_source='raw_official_mano_sensor_only',uses_tactile_input=False,wrist_motion_observed=False,
                  wrist_policy='fixed_identity_unobserved',vertex_threshold=self.threshold,frame_threshold=.5)
        if len(self.history)<history_length:
            self.last_result=dict(base,ready=False,status='warming_up');return self.last_result
        start=time.perf_counter()
        probability,frame_probability=self.predict(np.stack(list(self.history)[-history_length:])[None])
        latency=(time.perf_counter()-start)*1000
        probability=np.asarray(probability,np.float32).reshape(778)
        sampled=taxel_probabilities(probability,topology)
        reference=hand.get('source_contact') or {};cells=reference.get('cell_contact',[])
        contact_stamp=reference.get('timestamp_ms');delta=abs(float(contact_stamp)-stamp) if contact_stamp is not None else None
        aligned=bool(reference.get('ready') and len(cells)==460 and delta is not None and np.isfinite(delta) and delta<=25)
        if not aligned:cells=[None]*460
        known=np.array([type(c) is bool for c in cells])
        registered=np.zeros(460,bool)
        for sid in (hand.get('contact_registration') or {}).get('verified_sids',[]):
            if type(sid) is int and 0<=sid<460:registered[sid]=True
        illustrative=confusion(sampled,cells,known,self.threshold)
        verified=confusion(sampled,cells,known&registered,self.threshold)
        if illustrative:
            self.scored_frames+=1
            for name,row in (('illustrative',illustrative),('verified',verified)):
                if row:
                    for k in self.totals[name]:self.totals[name][k]+=row[k]
        cumulative={k:metrics(v) if sum(v.values()) else None for k,v in self.totals.items()}
        self.frames+=1
        raw=(hand.get('prediction_reference') or {}).get('raw_adc')
        raw_stamp=(hand.get('prediction_reference') or {}).get('timestamp_ms')
        if (not isinstance(raw,list) or len(raw)!=460 or raw_stamp is None or
            not np.isfinite(float(raw_stamp)) or abs(float(raw_stamp)-stamp)>25 or
            not np.isfinite(raw).all() or np.any((np.asarray(raw)<0)|(np.asarray(raw)>255))):raw=None
        normalized=reference.get('normalized')
        if not isinstance(normalized,list) or len(normalized)!=460:normalized=None
        self.last_result=dict(base,ready=True,status='ready',vertex_probability=probability.tolist(),taxel_probability=sampled.tolist(),
                    frame_probability=float(frame_probability),processing_ms=latency,frames=self.frames,
                    comparison=dict(ready=aligned,timestamp_delta_ms=delta,actual_contact=cells,actual_normalized=normalized,
                        actual_raw_adc=raw,verified_sids=np.flatnonzero(registered).tolist(),verified_count=int(registered.sum()),
                        illustrative=illustrative,verified=verified,cumulative=cumulative,scored_frames=self.scored_frames,
                        mapping_scope='measured' if registered.all() else 'illustrative_unverified_positions'))
        return self.last_result


def load_predictor(checkpoint,threads=2):
    import torch
    from retarget.learning.mano_contact.model import PoseEncoder,ContactModel
    path=Path(checkpoint)
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    if h.hexdigest()!=MODEL_SHA256:raise ValueError('接触预测 checkpoint 校验失败')
    torch.set_num_threads(threads)
    state=torch.load(path,map_location='cpu',weights_only=False)
    model=ContactModel(path.parent/'assets',PoseEncoder(**state['encoder_config']))
    model.load_state_dict(state.get('ema',state['model']));model.eval().requires_grad_(False)
    def predict(pose51):
        with torch.inference_mode():values=model(torch.from_numpy(pose51))
        return values['contact_out'].sigmoid().numpy()[0],values['frame_contact_logit'].sigmoid().item()
    return ContactPredictor(predict)
