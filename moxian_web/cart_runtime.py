"""True ten-frame lookahead; source geometry and robot output share a timestamp."""
import json
import time
import uuid
from pathlib import Path
import numpy as np
import torch
from retarget.baseline.geort.runtime import GeoRTRetarget
from retarget.baseline.geort.adapter import SourceAdapter,digest
from .cart_training import Predictor,neutral
from .cart_data import SourceGeometry
from .cart_protocol import PROTOCOL
from .cart_network import FutureBuffer
from .cart_contacts import overlay,transform_contacts
from .cart_objective import CartObjective,policy_settings
from .cart_initialization import RetargetSession


class CartRetarget(GeoRTRetarget):
    def __init__(self,topology,cache=None,device='cpu',model_path=None,**kwargs):
        if not model_path:raise ValueError('请先在 CART 面板选择已训练或已导入的模型')
        self.cart_predictor=Predictor(model_path,device);self.source=SourceAdapter(topology)
        if topology['binding_version']!=self.cart_predictor.config['source_binding_version']:
            raise ValueError('CART 模型与当前触觉材料绑定不匹配')
        self.expected=self.cart_predictor.config['source'];self.source_topology=topology
        self.robot=self.cart_predictor.hand.robot;self.model=self.robot.model
        self.surface=self.cart_predictor.hand.surface;self.predictor=self
        self.model_id=Path(model_path).name;self.session_id=uuid.uuid4().hex
        self.algorithm=PROTOCOL;self.training_mode='cart_'+self.cart_predictor.config['architecture_name'];self.run_id=self.cart_predictor.config['run_id']
        self.topology_version=digest([self.expected,self.robot.spec,self.model_id,self.run_id,PROTOCOL])[:24]
        self.mapping=dict(ready=True,failed_sids=[],records=[None]*460,version=self.topology_version,
            geometry_version=topology['geometry_version'],binding_version=topology['binding_version'],
            mirror_matrix=self.source.mirror.tolist(),policy='cart_observed_contacts_v5')
        profile=json.loads((Path(model_path)/'source_profile.json').read_text())
        self.geometry=SourceGeometry(topology,profile,device)
        self.objective=CartObjective(self.cart_predictor.hand,policy_settings(),self.cart_predictor.collision)
        self.buffer=FutureBuffer();self.external_sdf=None;self.reset()

    def topology(self):
        return dict(self.surface.topology(self.mapping),algorithm=PROTOCOL,training_mode=self.training_mode,
                    topology_version=self.topology_version,model_id=self.model_id,model_run_id=self.run_id,
                    experimental=bool((self.cart_predictor.deployment or {}).get('experimental')))

    def predict(self,points):return self.last_q

    def reset(self):
        self.buffer.reset();self.last_q=neutral(self.cart_predictor.hand).cpu().numpy()
        self.ik_session=RetargetSession.for_hand('WujiHand2','right');self.state_initialized=False
        self.session_id=uuid.uuid4().hex

    def step(self,frame):
        began=time.perf_counter()
        if self.source.identity(frame.get('serial'),frame)!=self.expected:raise ValueError('CART 模型与当前标定／设备不匹配')
        if frame.get('binding_version')!=self.source_topology['binding_version']:raise ValueError('触觉材料绑定已变化，请重新加载兼容模型')
        if not frame.get('calibrated') or frame.get('preview'):raise ValueError('等待已应用源手标定后的输入')
        stamp=frame.get('recorded_timestamp_ms',frame['timestamp_ms']);epoch=frame.get('input_stream_epoch')
        if self.buffer.last_stamp is not None and (not 0<stamp-self.buffer.last_stamp<=250 or epoch!=self.buffer.epoch):
            self.reset()
        def observed(item):
            v,j=np.asarray(item['vertices_m']),np.asarray(item['joints_m'])
            row,_,_=self.geometry.frame(None,item['timestamp_ms'],item,raw=(v,j))
            row['timestamp_ms']=item.get('recorded_timestamp_ms',item['timestamp_ms']);row['stream_epoch']=epoch
            return (row['timestamp_ms'],(dict(item),row))
        if 'cart_future_frames' in frame:
            # Offline replay reads saved true future data immediately, including
            # paused/seek views. It does not wait for ten playback callbacks.
            future_frames=frame['cart_future_frames']
            self.buffer.last_stamp=stamp;self.buffer.epoch=epoch
            window=[observed(frame),*[observed(item) for item in future_frames]] if len(future_frames)==10 else None
        else:
            row=observed(frame)[1][1]
            window=self.buffer.append(stamp,(dict(frame),row),epoch)
        if window is None:
            return dict(ready=False,display_available=False,status='buffering',algorithm=PROTOCOL,
                training_mode=self.training_mode,solver_session_id=self.session_id,
                cart=dict(buffering=True,buffered_frames=len(self.buffer.entries),required_frames=11),
                quality=dict(processing_ms=(time.perf_counter()-began)*1000))
        previous=window[0][1][1];future=[entry[1][1] for entry in window[1:]]
        target_frame=window[1][1][0];row=future[0]
        if not self.state_initialized:
            # Seed the current state once; never solve IK for future frames.
            source_frame=window[0][1][0]
            try:
                if source_frame.get('cart_ik_valid'):
                    initial=np.asarray(source_frame['cart_ik_q_rad'],np.float32)
                else:initial=self.ik_session.step(np.asarray(previous['joints_m']))
                hand=self.cart_predictor.hand
                self.last_q=np.clip(initial,hand.low.cpu().numpy(),hand.high.cpu().numpy())
            except (ValueError,RuntimeError):
                self.ik_session.reset()
                return dict(ready=False,display_available=False,status='waiting_ik',algorithm=PROTOCOL)
            self.state_initialized=True
        old_q=self.last_q.copy()
        self.last_q=self.cart_predictor.predict_rows(old_q,window[0][0],future)
        network_ms=(time.perf_counter()-began)*1000
        result=super().step(target_frame)
        with torch.no_grad():
            collision=self.cart_predictor.collision;collision.auditing=True
            try:
                _,_,quality=self.objective.batch(self.cart_predictor.hand.tensor(self.last_q)[None],[row],
                    self.cart_predictor.hand.tensor(old_q)[None],previous_rows=[previous],audit=True)
            finally:collision.auditing=False
        quality=quality[0]
        joints=np.asarray(target_frame['joints_m']);root=joints[self.source.indices[0]]@self.source.mirror.T
        contacts=transform_contacts(row['contacts'],self.source.basis,root,inverse=True)
        source_overlay=overlay(contacts,timestamp_ms=target_frame['timestamp_ms'],geometry_version=target_frame['geometry_version'],processing_ms=row['processing_ms'])
        robot_points=[]
        for pa,pb in quality['points']:
            robot_points.extend((p[0].cpu().numpy()@self.source.basis.T+root).tolist() for p in (pa,pb))
        metrics=dict(required=quality['contact_count'],max_gap_mm=float(quality['max_gap_m'][0])*1000,
            max_normal_deg=float(quality['max_normal_deg'][0]),bone_direction_deg=float(quality['bone_direction_deg'][0]),
            max_penetration_mm=float(quality['max_penetration_m'][0])*1000)
        result.update(cart_hull=source_overlay,
            cart_robot_hull=dict(ready=True,contact_points_m=robot_points,patch_points_m=[],timestamp_ms=target_frame['timestamp_ms'],geometry_version=target_frame['geometry_version']),
            cart=dict(metrics=metrics,buffering=False,observation_timestamp_ms=row['timestamp_ms'],
                experimental=bool((self.cart_predictor.deployment or {}).get('experimental')),
                acceptance_status=(self.cart_predictor.deployment or {}).get('acceptance_status','completed'),
                buffer_delay_ms=0. if 'cart_future_frames' in frame else stamp-row['timestamp_ms'],self_contacts=len(contacts['self_pairs']),object_contacts=len(contacts['object_contacts']),
                tactile_status=contacts['tactile_status'],online_search=False,hard_contact_guarantee=False),
            recorded_timestamp_ms=target_frame.get('recorded_timestamp_ms'),contact_ready=False,
            contact_anchors=contacts['object_contacts'],self_contact_pairs=contacts['self_pairs'],
            contact_constraints_satisfied=bool(quality['feasible'][0]) if quality['contact_count'] else None,
            solution_feasible=bool(quality['feasible'][0]),status='ready')
        result['quality'].update(processing_ms=(time.perf_counter()-began)*1000,network_ms=network_ms,
                                 inference_only=True,audit_only=True)
        return result
