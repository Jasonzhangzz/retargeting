"""Direct GeoRT inference and exact FK for the virtual Wuji2 viewer."""
import json
import time
import uuid
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from retarget.baseline.geort.adapter import SourceAdapter, RobotAdapter, digest
from retarget.baseline.geort.training import Predictor


class GeoRTRetarget:
    def __init__(self, topology, cache=None, device='cpu', model_path=None, **kwargs):
        if not model_path:raise ValueError('请先选择训练成功的 GeoRT 模型')
        self.source=SourceAdapter(topology);self.predictor=Predictor(model_path,device)
        self.robot=self.predictor.robot;self.model=self.robot.model
        self.expected=self.predictor.config['source'];self.source_topology=topology
        from moxian_web.wuji_surface import WujiSurface
        self.surface=WujiSurface()
        self.session_id=uuid.uuid4().hex
        self.run_id=self.predictor.config.get('run_id',Path(model_path).name)
        self.training_mode=self.predictor.config.get('training_mode','official')
        self.algorithm='geort_official_v1' if self.training_mode=='official' else 'geort_contact_v1'
        version=digest([self.expected,self.robot.spec,Path(model_path).name,self.run_id])[:24]
        self.mapping=dict(ready=True,failed_sids=[],records=[None]*460,version=version,
            geometry_version=topology['geometry_version'],binding_version=topology['binding_version'],
            mirror_matrix=self.source.mirror.tolist(),policy='geort_no_tactile_mapping')
        self.model_id=Path(model_path).name;self.topology_version=version
        self.external_sdf=None

    def reset(self):self.session_id=uuid.uuid4().hex

    def topology(self):
        return dict(self.surface.topology(self.mapping),algorithm=self.algorithm,training_mode=self.training_mode,
                    topology_version=self.topology_version,model_id=self.model_id,model_run_id=getattr(self,'run_id',self.model_id),checkpoint='last.pth')

    def step(self, frame):
        began=time.perf_counter()
        if not frame.get('calibrated') or frame.get('preview'):raise ValueError('GeoRT 等待已应用标定后的有效手部模型')
        identity=self.source.identity(frame.get('serial'),frame)
        if identity!=self.expected:raise ValueError('GeoRT 模型与当前设备、标定或坐标系不匹配，请重新训练或选择兼容模型')
        if abs(frame['timestamp_ms']-frame.get('skeleton_timestamp_ms',frame['timestamp_ms']))>25:
            raise ValueError('GeoRT 源骨架与手部模型时间不同步')
        joints=np.asarray(frame['joints_m'],float)
        q=self.predictor.predict(self.source.points(joints))
        if not np.isfinite(q).all() or np.any(q<self.robot.lower-1e-6) or np.any(q>self.robot.upper+1e-6):
            raise ValueError('GeoRT 网络输出超出 Wuji2 关节范围')
        self.robot.fk(q)
        j=joints[self.source.indices]@self.source.mirror.T
        vertices=np.asarray(frame['vertices_m'],float)@self.source.mirror.T
        rotation=self.source.basis@self.robot.basis.T
        palm=self.robot.data.xpos[self.robot.body_ids[0]]
        translation=j[0]-rotation@palm
        positions=self.robot.data.xpos[self.robot.body_ids]@rotation.T+translation
        rotations=rotation@self.robot.data.xmat[self.robot.body_ids].reshape(-1,3,3)
        # GeoRT learns geometry only. Unknown contact values must remain unknown.
        zeros=np.zeros((460,3)).tolist()
        elapsed=(time.perf_counter()-began)*1000
        return dict(ready=True,display_available=True,status='ready',algorithm=self.algorithm,training_mode=self.training_mode,
            solver_session_id=self.session_id,model_id=self.model_id,model_run_id=getattr(self,'run_id',self.model_id),checkpoint='last.pth',mapping_version=self.mapping['version'],
            topology_version=self.topology_version,geometry_version=frame['geometry_version'],source_binding_version=frame['binding_version'],
            source_timestamp_ms=frame['timestamp_ms'],timestamp_ms=frame['timestamp_ms'],source_hand_side=frame['hand_side'],
            joint_names=self.robot.joint_names,joint_positions_rad=q.tolist(),
            qpos=np.r_[translation,Rotation.from_matrix(rotation).as_euler('XYZ'),q].tolist(),
            wrist_position_m=translation.tolist(),wrist_rotation=rotation.tolist(),
            link_positions_m=positions.tolist(),link_rotations=rotations.tolist(),source_vertices_m=vertices.tolist(),source_joints_m=j.tolist(),
            source_taxel_positions_m=zeros,taxel_positions_m=zeros,source_taxel_normals=zeros,taxel_normals=zeros,
            contact_ready=False,contact_mask=[None]*460,contact_weights=[0.]*460,contact_anchors=[],contact_edges=[],
            solution_feasible=None,contact_constraints_satisfied=None,self_contact_pairs=[],
            quality=dict(processing_ms=elapsed,joint_bounds_saturated=int(((q-self.robot.lower<.01)|(self.robot.upper-q<.01)).sum()),
                         inference_only=True),virtual_only=True)


class Replay:
    """Reconstruct recorded Moxian mesh from saved same-frame joint angles."""
    def __init__(self, folder, device='cpu'):
        folder=Path(folder)
        self.meta=json.loads((folder/'manifest.json').read_text())
        if not self.meta.get('complete') or self.meta.get('frames',0)<2:raise ValueError('录制尚未完成')
        self.topology=json.loads((folder/'topology.json').read_text())
        self.motion=np.load(folder/'motion.npz',allow_pickle=False)
        self.stamps=np.load(folder/'timestamps.npy',allow_pickle=False)
        profile=json.loads((folder/'profile.json').read_text())
        from moxian_web.source_ik_initialization import validate_recorded_source
        validate_recorded_source(profile,self.meta['source'])
        from moxian_web.qnbot_surface import QnbotSurface
        self.model=QnbotSurface(self.meta['source']['hand_side'],device,finger_twist=True)
        self.model.load_artifact(profile)
        self.index=0;self.playing=True;self.anchor=time.monotonic();self.offset=0.
        self.last_index=-1
        self.epoch=0
        self.contact_rows=None;self.linked_emf=None
        self.cart_ik=None
        if self.meta.get('protocol','').startswith('cart_') and (folder/'cart_ik.npz').exists():
            from moxian_web.cart_data import file_hash
            info=json.loads((folder/'cart_ik.json').read_text())
            if info['sha256']!=file_hash(folder/'cart_ik.npz'):raise ValueError('回放 IK 缓存校验失败')
            with np.load(folder/'cart_ik.npz',allow_pickle=False) as saved:
                if not np.array_equal(saved['timestamps_ms'],self.stamps):raise ValueError('回放 IK 时间轴不匹配')
                self.cart_ik=(saved['q'].copy(),saved['valid'].copy())
        if (folder/'cart_emf.jsonl').exists():
            self.linked_emf=[json.loads(line) for line in (folder/'cart_emf.jsonl').open()]
            if len(self.linked_emf)!=len(self.stamps) or any(r['timestamp_ms']!=s for r,s in zip(self.linked_emf,self.stamps)):
                raise ValueError('关联 EMF 与回放时间轴不一致')
        if self.meta.get('protocol','').startswith('cart_') and (folder/'frames.jsonl').exists():
            self.contact_rows=[json.loads(line) for line in (folder/'frames.jsonl').open()]
            if len(self.contact_rows)!=len(self.stamps):raise ValueError('CART 触觉与运动帧数不一致')

    def seek(self, index):
        self.epoch+=1
        self.index=max(0,min(len(self.stamps)-1,int(index)))
        self.offset=float(self.stamps[self.index]-self.stamps[0])/1000
        self.anchor=time.monotonic();self.last_index=-1

    def pause(self, paused):
        self.seek(self.index);self.playing=not paused

    def frame(self):
        if self.playing:
            elapsed=self.offset+time.monotonic()-self.anchor
            self.index=min(len(self.stamps)-1,int(np.searchsorted(self.stamps-self.stamps[0],elapsed*1000,side='right')-1))
            if self.index==len(self.stamps)-1:self.playing=False
        if self.index==self.last_index:return None
        self.last_index=self.index
        return self._frame_at(self.index,time.time()*1000)

    def future_frames(self,frame,count=10):
        """Saved future observations are available even while replay is paused."""
        if self.index+count>=len(self.stamps):return []
        return [self._frame_at(i,frame['timestamp_ms']+float(self.stamps[i]-self.stamps[self.index]))
                for i in range(self.index+1,self.index+count+1)]

    def _frame_at(self,index,stamp):
        import torch
        with torch.inference_mode():
            q=torch.as_tensor(self.motion['qpos'][index],device=self.model.device)[None]
            v,j,_=self.model(q)
        source=self.meta['source']
        recorded=self.contact_rows[index] if self.contact_rows else {}
        tt=recorded.get('tactile_timestamp_ms')
        contact=recorded.get('source_contact')
        if contact:
            contact=dict(contact)
            if isinstance(contact.get('timestamp_ms'),(int,float)):
                contact['timestamp_ms']=stamp+contact['timestamp_ms']-float(self.stamps[index])
        emf=recorded.get('emf_observation') or (self.linked_emf[index]['emf_observation'] if self.linked_emf else None)
        if emf:
            emf=dict(emf,timestamp_ms=stamp+emf['timestamp_ms']-float(self.stamps[index]))
        ik={} if self.cart_ik is None else dict(cart_ik_valid=bool(self.cart_ik[1][index]),
            cart_ik_q_rad=self.cart_ik[0][index].tolist() if self.cart_ik[1][index] else None)
        return dict(**ik,emf_observation=emf,vertices_m=v[0].cpu().numpy(),joints_m=j[0].cpu().numpy(),
            geometry_version=self.topology['geometry_version'],binding_version=self.topology['binding_version'],
            model_hash=source['model_hash'],hand_side=source['hand_side'],serial=source['serial'],
            surface_backend=self.topology['surface_backend'],coordinate_frame=self.topology['coordinate_frame'],
            calibration_fingerprint=source['calibration_fingerprint'],skeleton_runtime_revision=source['skeleton_runtime_revision'],
            timestamp_ms=stamp,skeleton_timestamp_ms=stamp,calibrated=True,
            input_received_wall_ms=stamp,recorded_timestamp_ms=float(self.stamps[index]),
            tactile_timestamp_ms=stamp+tt-float(self.stamps[index]) if isinstance(tt,(int,float)) else None,
            source_contact=contact,source_joint_positions_rad=self.motion['qpos'][index].tolist(),
            input_stream_epoch='replay:'+self.meta['id']+':'+str(self.epoch))

    def snapshot(self):
        return dict(active=True,id=self.meta['id'],playing=self.playing,index=self.index,frames=len(self.stamps),
                    seconds=float(self.stamps[self.index]-self.stamps[0])/1000,
                    duration_s=float(self.stamps[-1]-self.stamps[0])/1000)
