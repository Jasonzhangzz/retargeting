"""Pose-only RVQ annotations after Moxian -> Wuji2 retargeting."""
from pathlib import Path
import base64
from functools import lru_cache
import hashlib
import json
import os
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch

from retarget.env.grasp.hand import hand_tree,HAND_BODIES
from retarget.learning.wuji2_rvq.joint import PoseRVQJoint

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT=ROOT/'data/models/wuji2_rvq_joint_v1/joint.pt'


@lru_cache(maxsize=1)
def primitive_illustrations():
    """Small static catalog, transported once with model topology, never per frame."""
    folder=Path(__file__).with_name('rvq_primitives')
    catalog=json.loads((folder/'catalog.json').read_text())['primitives']
    return {name:{'image':'data:image/png;base64,'+base64.b64encode((folder/item['file']).read_bytes()).decode('ascii'),
                  'source_row':item['row'],'size_scale':item['size_scale']}
            for name,item in catalog.items()}


class WujiRVQ:
    def __init__(self,checkpoint=None):
        self.path=Path(checkpoint or os.environ.get('MOXIAN_RVQ_MODEL',str(DEFAULT_CHECKPOINT))).expanduser()
        if self.path.is_dir():self.path=self.path/'joint.pt'
        self.enabled=os.environ.get('MOXIAN_RVQ_ENABLED','0')!='0'
        self.model=None;self.error=None;self.signature=None;self.fk=None;self.hash=None
        self.reference_kinematics=None
        self.device=os.environ.get('MOXIAN_RVQ_DEVICE','cpu')
        self.refresh()

    def refresh(self):
        if not self.enabled:return
        try:
            stat=self.path.stat();signature=(stat.st_mtime_ns,stat.st_size)
        except OSError:
            self.model=None;self.error='本地 RVQ 权重不存在：'+str(self.path);return
        if signature==self.signature:return
        self.signature=signature;self.model=None
        try:
            saved=torch.load(self.path,map_location='cpu',weights_only=True)
            if not saved['config'].get('joint'):raise ValueError('RVQ 权重必须包含共享 decoder 和两个输出头')
            model=PoseRVQJoint(**saved['config']);model.load_state_dict(saved['model'])
            fk=mujoco.MjModel.from_xml_string(ET.tostring(hand_tree('wuji2'),encoding='unicode'))
            names=[fk.joint(i).name for i in range(6,fk.njnt)]
            if names!=model.config['joint_names'] or fk.nq!=26:
                raise ValueError('RVQ 与本地 Wuji2 原生关节顺序不一致')
            if not np.allclose(fk.jnt_range[6:],np.stack([model.config['lower'],model.config['upper']],axis=1),atol=1e-6,rtol=0):
                raise ValueError('RVQ 与本地 Wuji2 原生关节范围不一致')
            self.fk=fk;self.data=mujoco.MjData(fk);self.body_ids=[fk.body(name).id for name in HAND_BODIES]
            mujoco.mj_kinematics(fk,self.data)
            self.root_position=self.data.xpos[self.body_ids[0]].copy()
            self.root_rotation=self.data.xmat[self.body_ids[0]].reshape(3,3).copy()
            self.hash=hashlib.sha256(self.path.read_bytes()).hexdigest()
            self.model=model.to(self.device).eval();self.error=None
            self.evaluation={k:saved.get('metadata',{}).get(k) for k in ['accuracy','macro_f1','mae_deg','p95_deg']}
        except Exception as exc:self.error=str(exc)

    def metadata(self):
        self.refresh()
        result=dict(enabled=self.enabled,ready=self.model is not None,checkpoint=str(self.path),sha256=self.hash,
            device=self.device,task='joint_reconstruction_classification',input='retargeted Wuji2 native 20 joints, radians',
            error=self.error)
        if self.model is not None:
            result.update(joint_names=self.model.config['joint_names'],class_names=self.model.config['class_names'],
                          evaluation=self.evaluation)
            try:
                images=primitive_illustrations()
                result['primitives']={name:images[name] for name in self.model.config['class_names']}
            except (OSError,ValueError,KeyError) as exc:
                result['primitive_error']='Primitive 静态示意图不可用：'+str(exc)
        return result

    @torch.inference_mode()
    def infer(self,result,topology):
        """Return an annotation plus FK overlay; original solver output is untouched."""
        self.refresh()
        if not self.enabled:return dict(ready=False,status='disabled')
        if self.model is None:return dict(ready=False,status='unavailable',error=self.error)
        began=time.perf_counter()
        try:
            names=result.get('joint_names',[]);values=np.asarray(result.get('joint_positions_rad',[]),dtype=np.float64)
            if len(names)!=20 or len(set(names))!=20 or values.shape!=(20,) or not np.isfinite(values).all():
                raise ValueError('RVQ 需要20个有名称的有限弧度关节角')
            if set(names)!=set(self.model.config['joint_names']):raise ValueError('输入不是模型支持的 Wuji2 关节')
            ordered=values[[names.index(name) for name in self.model.config['joint_names']]]
            lower=np.asarray(self.model.config['lower']);upper=np.asarray(self.model.config['upper'])
            if np.any(ordered<lower-1e-5) or np.any(ordered>upper+1e-5):raise ValueError('输入关节角超出 Wuji2 原生范围')
            q=torch.as_tensor(ordered,dtype=torch.float32,device=self.device)[None]
            tokens=self.model.encode(q);out=self.model.decode_outputs(tokens)
            reconstructed=out['qpos'][0].cpu().numpy();probabilities=out['logits'][0].softmax(-1).cpu().numpy()
            top=np.argsort(probabilities)[-3:][::-1];classes=self.model.config['class_names'];chosen=int(top[0])
            error=np.abs(reconstructed.astype(np.float64)-ordered)*180/np.pi
            annotation=dict(ready=True,status='ready',model_sha256=self.hash,timestamp_ms=result.get('timestamp_ms'),
                source_timestamp_ms=result.get('source_timestamp_ms'),source_solver=result.get('solver_backend') or topology.get('solver_backend'),
                source_hand_side=result.get('source_hand_side'),class_index=chosen,grasp_type=classes[chosen],
                probability=float(probabilities[chosen]),probabilities=probabilities.tolist(),tokens=tokens[0].cpu().tolist(),
                top3=[dict(class_index=int(i),grasp_type=classes[int(i)],probability=float(probabilities[int(i)])) for i in top],
                joint_names=self.model.config['joint_names'],reconstructed_joint_positions_rad=reconstructed.tolist(),
                reconstruction_mae_deg=float(error.mean()),reconstruction_max_deg=float(error.max()))
            annotation.update(self.reconstruction_geometry(reconstructed,result,topology))
            annotation['inference_ms']=(time.perf_counter()-began)*1000
            return annotation
        except Exception as exc:
            return dict(ready=False,status='invalid_input',error=str(exc),timestamp_ms=result.get('timestamp_ms'))

    def reconstruction_geometry(self,q,result,topology):
        names=topology.get('link_names')
        reference=topology.get('solver_backend')=='wuji-retargeting'
        if reference:
            from retarget.baseline.wuji_retargeting.runtime import HandKinematics,link_poses,render_links
            side=topology.get('native_hand_side','right')
            if side!='right':raise ValueError('原版左手 URDF 角度尚未映射到 RVQ 的右手原生关节约定')
            if self.reference_kinematics is None:self.reference_kinematics=HandKinematics('WujiHand2','right')
            if names!=render_links(self.reference_kinematics):raise ValueError('RVQ 与原版机器人 link 顺序不一致')
        elif names!=HAND_BODIES:raise ValueError('RVQ 重建显示的 Wuji2 link 顺序不一致')
        positions=np.asarray(result.get('link_positions_m'),dtype=np.float64)
        rotations=np.asarray(result.get('link_rotations'),dtype=np.float64)
        if positions.shape!=(21,3) or rotations.shape!=(21,3,3) or not np.isfinite(positions).all() or not np.isfinite(rotations).all():
            raise ValueError('RVQ 重建缺少同帧手腕变换')
        # Derive alignment from the actual root-link frame, including the source
        # left-hand mirror convention. Wrist qpos Euler conventions are irrelevant.
        if reference:
            from scipy.spatial.transform import Rotation
            native_p,native_r=link_poses(self.reference_kinematics,q)
            native_p=np.asarray([native_p[name] for name in names])
            native_r=Rotation.from_quat([native_r[name] for name in names]).as_matrix()
            root_p,root_r=native_p[0],native_r[0]
        else:
            self.data.qpos[:]=np.r_[np.zeros(6),q]
            mujoco.mj_kinematics(self.fk,self.data)
            native_p=self.data.xpos[self.body_ids]
            native_r=self.data.xmat[self.body_ids].reshape(-1,3,3)
            root_p,root_r=self.root_position,self.root_rotation
        rotation=rotations[0]@root_r.T
        translation=positions[0]-rotation@root_p
        p=native_p@rotation.T+translation
        r=rotation@native_r
        return dict(link_positions_m=p.tolist(),link_rotations=r.tolist())
