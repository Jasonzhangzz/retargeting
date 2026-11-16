"""Direct CART network inference with explicit source/robot contact audit."""
import time,uuid
from pathlib import Path
import numpy as np
import torch
from retarget.baseline.geort.runtime import GeoRTRetarget
from retarget.baseline.geort.adapter import SourceAdapter,digest
from .cart_training import Predictor
from .cart_data import SourceGeometry
from .cart_protocol import PROTOCOL,STAGES,BY_ID,NEUTRAL_INDEX,MotionTracker,features
from .cart_geometry import transform_hull
from .cart_losses import contact_losses

class CartRetarget(GeoRTRetarget):
    def __init__(self,topology,cache=None,device='cpu',model_path=None,**kwargs):
        if not model_path:raise ValueError('请先采齐五类动作并选择训练成功的 CART 模型')
        self.cart_predictor=Predictor(model_path,device);self.source=SourceAdapter(topology)
        self.expected=self.cart_predictor.config['source'];self.source_topology=topology
        self.robot=self.cart_predictor.hand.robot;self.model=self.robot.model
        self.surface=self.cart_predictor.hand.surface;self.predictor=self
        self.model_id=Path(model_path).name;self.session_id=uuid.uuid4().hex
        self.algorithm=PROTOCOL;self.training_mode='cart_primitive_delta'
        self.run_id=self.cart_predictor.config.get('run_id',self.model_id)
        self.topology_version=digest([self.expected,self.robot.spec,self.model_id,PROTOCOL])[:24]
        self.mapping=dict(ready=True,failed_sids=[],records=[None]*460,version=self.topology_version,
            geometry_version=topology['geometry_version'],binding_version=topology['binding_version'],
            mirror_matrix=self.source.mirror.tolist(),policy='cart_native_semantic_contact_v3')
        self.geometry=SourceGeometry.__new__(SourceGeometry);self.geometry.adapter=self.source
        self.geometry.topology=topology;self.geometry.regions=topology['regions']
        self.geometry.faces=np.asarray(topology['faces'],int)
        if np.linalg.det(self.source.transform)<0:self.geometry.faces=self.geometry.faces[:,[0,2,1]]
        self.tracker=MotionTracker();self.manual='auto';self.last_action=STAGES[0]['id'];self.last_q=None
        self.external_sdf=None
    def topology(self):
        return dict(self.surface.topology(self.mapping),algorithm=PROTOCOL,
                    topology_version=self.topology_version,model_id=self.model_id)
    def predict(self,points):return self.last_q
    def reset(self):
        self.cart_predictor.reset();self.tracker.reset();self.session_id=uuid.uuid4().hex
    def set_action(self,value):
        if value!='auto' and value not in BY_ID:raise ValueError('未知 CART 子动作')
        self.manual=value;self.reset()
    def step(self,frame):
        began=time.perf_counter()
        if self.source.identity(frame.get('serial'),frame)!=self.expected:raise ValueError('CART 模型与当前标定／设备不匹配')
        if not frame.get('calibrated') or frame.get('preview'):raise ValueError('等待已应用六动作标定后的输入')
        vertices=np.asarray(frame['vertices_m']);joints=np.asarray(frame['joints_m'])
        _,canonical=self.geometry.canonical(vertices,joints)
        stamp=frame.get('recorded_timestamp_ms',frame['timestamp_ms'])
        previous=self.cart_predictor.last_stamp
        x=features(canonical,0 if previous is None else (stamp-previous)/1000)
        encoded=self.cart_predictor.encode(dict(timestamp_ms=stamp,features=x))
        probs=encoded[1][0,0].softmax(-1);index=int(probs.argmax());neutral=index==NEUTRAL_INDEX
        selected=self.manual if self.manual!='auto' else (self.last_action if neutral else STAGES[index]['id'])
        if selected!=self.last_action:self.tracker.reset()
        self.last_action=selected
        row,_,_=self.geometry.frame(None,selected,stamp,self.tracker,raw=(vertices,joints))
        if neutral and self.manual=='auto':row['phase']='open'
        q,t,r=self.cart_predictor.decode(encoded,row['target'],BY_ID[selected]['index'] if self.manual!='auto' else None)
        self.last_q=q.detach().cpu().numpy()
        result=super().step(frame)
        with torch.no_grad():_,metrics=contact_losses(self.cart_predictor.hand,q[None],t[None],r[None],[row])
        quality=metrics[0];dense=self.cart_predictor.hand.audit(q)
        depth=max(dense['max_penetration_mm'],quality.get('object_penetration_mm',0))
        active=row['phase'] in ('closing','hold')
        satisfied=bool(quality['required'] and max(quality['gaps_mm'])<=2 and max(quality['normal_errors_deg'])<=30
                       and depth<=.5 and not quality['object_intersection_over_05mm'])
        root=joints[self.source.indices[0]]@self.source.mirror.T;world=self.source.basis
        source_hull=transform_hull(row['target'],world,root)
        object_rotation=r.detach().cpu().numpy();object_translation=t.detach().cpu().numpy()
        if row['target'].get('ready'):object_translation=object_translation-object_rotation@np.asarray(row['target']['center_m'])
        robot_hull=(transform_hull(transform_hull(row['target'],object_rotation,object_translation),world,root)
                    if row['target'].get('ready') else dict(source_hull))
        for h in (source_hull,robot_hull):h.update(timestamp_ms=frame['timestamp_ms'],geometry_version=frame['geometry_version'])
        contacts=[]
        for c in quality['contacts']:
            contacts.append(dict(c,point_m=(np.asarray(c['point_m'])@world.T+root).tolist(),
                target_point_m=(np.asarray(c['target_point_m'])@world.T+root).tolist()))
        source_hull['contact_points_m']=[p for pair in row['target'].get('pairs',[]) for p in (np.asarray(pair['points_m'])@world.T+root).tolist()]
        source_hull['contact_points_m'] += [p for c in source_hull.get('contacts',[]) for p in (c['point_m'],c['object_point_m'])]
        robot_hull['contact_points_m']=[p for c in contacts for p in (c['point_m'],c['target_point_m'])]
        if row['target'].get('contact_mode')=='hand':
            robot_hull['pairs']=[dict(regions=[c['patch'],c.get('target_patch')],points_m=[c['point_m'],c['target_point_m']]) for c in contacts]
        result.update(algorithm=PROTOCOL,cart_hull=source_hull,cart_robot_hull=robot_hull,
            cart=dict(stage=selected,category=row['category'],category_name=BY_ID[selected]['category_name'],phase=row['phase'],contact_mode=row['target']['contact_mode'],
                action_confidence=float(probs.max()),automatic=self.manual=='auto',neutral=neutral,
                metrics=quality,contacts=contacts,max_penetration_mm=depth),
            solution_feasible=bool(depth<=.5 and (not active or satisfied)),contact_constraints_satisfied=satisfied,
            collision=dense,status='ready' if not active or satisfied else 'infeasible')
        result['quality'].update(processing_ms=(time.perf_counter()-began)*1000,inference_only=True,
                                 geometry_ms=row['target'].get('processing_ms'))
        return result
