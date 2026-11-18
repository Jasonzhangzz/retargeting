"""Reconstruct immutable CART source geometry and synchronized sequences."""
import hashlib,json
from pathlib import Path
import numpy as np
from .cart_protocol import PROTOCOL,stage,features,MotionTracker,FINGERS
from retarget.baseline.geort.adapter import SourceAdapter,digest
from retarget.baseline.geort.training import atomic_json

class SourceGeometry:
    def __init__(self,topology,profile,device='cpu'):
        import torch
        from .qnbot_surface import QnbotSurface
        self.torch=torch;self.topology=topology;self.adapter=SourceAdapter(topology)
        self.model=QnbotSurface(topology['hand_side'],device,finger_twist=True)
        self.model.load_artifact(profile);self.model.fixed=True
        self.faces=np.asarray(topology['faces'],int)
        if np.linalg.det(self.adapter.transform)<0:self.faces=self.faces[:,[0,2,1]]
        self.regions=topology['regions']
    def reconstruct(self,q):
        q=np.asarray(q,np.float32)
        if len(q)==25 and len(self.model.q0)==29:q=np.r_[q,np.zeros(4,np.float32)]
        with self.torch.inference_mode():
            v,j,_=self.model(self.torch.tensor(q,device=self.model.device)[None])
        return v[0].cpu().numpy(),j[0].cpu().numpy()
    def canonical(self,vertices,joints):
        j=np.asarray(joints)[self.adapter.indices];root=j[0]
        return ((np.asarray(vertices)-root)@self.adapter.transform,
                (j-root)@self.adapter.transform)
    def witnesses(self,cv,pairs,full=False):
        from .cart_geometry import region_pair
        if full:
            witnesses=[region_pair(cv,self.faces,self.regions,p) for p in pairs]
        else:
            # Existing native mesh BVH query: all region vertices, centroids,
            # edge/face intersections and edge/edge candidates, refitted for
            # this frame. No slow quadratic CPU triangle-pair array.
            import torch,contextlib,sys
            from .qnbot_link_contact import LinkContactQuery
            if not hasattr(self,'contact_queries'):self.contact_queries={}
            key=tuple(tuple(p) for p in pairs)
            tensor_vertices=torch.as_tensor(cv[None],dtype=torch.float32)
            with contextlib.redirect_stdout(sys.stderr):
                if key not in self.contact_queries:
                    if len(self.contact_queries)>=8:self.contact_queries.pop(next(iter(self.contact_queries)))
                    self.contact_queries[key]=LinkContactQuery(tensor_vertices,torch.as_tensor(self.faces),self.regions,
                        [dict(row=0,regions=list(p)) for p in pairs])
                found=self.contact_queries[key].witnesses(tensor_vertices)
            tri=found['triangles'].cpu().numpy();bary=found['barycentric'].cpu().numpy()
            witnesses=[]
            for pair,indices,bc in zip(pairs,tri,bary):
                triangles=cv[indices];points=(triangles*bc[...,None]).sum(1)
                normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]);normals/=np.maximum(np.linalg.norm(normals,axis=-1,keepdims=True),1e-12)
                witnesses.append(dict(regions=list(pair),gap_m=float(np.linalg.norm(points[1]-points[0])),
                    points_m=points.tolist(),normals=normals.tolist()))
        return witnesses
    def frame(self,q,stage_id,stamp,tracker,full=False,raw=None):
        import time
        started=time.perf_counter()
        from .cart_geometry import estimate
        spec=stage(stage_id)
        v,j=self.reconstruct(q) if raw is None else raw
        cv,cj=self.canonical(v,j)
        pairs=spec['pairs'] or [('thumb_distal',FINGERS[f]+'_distal') for f in spec['fingers'] if f]
        witnesses=self.witnesses(cv,pairs,full)
        gap=max(p['gap_m'] for p in witnesses)
        previous=tracker.last
        motion=tracker.update(stamp,gap)
        if spec['category']==1 or motion['closed']:
            target=dict(ready=False,status='hand_contact',contact_mode='hand',reference_gap_m=gap,
                        pairs=witnesses,required_patches=list(spec['patches']),coordinate_frame='palm_canonical',
                        stage=stage_id,category=spec['category'],category_name=spec['category_name'])
        else:
            history=getattr(self,'convex_history',None)
            if full or previous is None or not 0<stamp-previous<=250 or not history or history['stage']!=stage_id:
                history=None
            target=estimate(cv,self.faces,self.regions,stage_id,full=full,witnesses=witnesses,joints=cj,
                previous=history['target'] if history else None)
        if not full:
            # Carry the fair-search cursor even after a failed fit. Geometry
            # itself is always fitted and validated again on the current frame.
            self.convex_history=dict(stage=stage_id,target=target)
        target.update(timestamp_ms=stamp,geometry_version=self.topology['geometry_version'],
                      phase=motion['phase'],cycles=motion['cycles'],coordinate_frame='palm_canonical')
        target['processing_ms']=(time.perf_counter()-started)*1000
        return dict(timestamp_ms=stamp,stage=stage_id,stage_index=spec['index'],category=spec['category'],category_name=spec['category_name'],
            phase=motion['phase'],cycles=motion['cycles'],target=target,joints_m=cj.tolist(),
            features=features(cj,0 if previous is None else (stamp-previous)/1000).tolist()),v,j

def manifest(root,key):
    from .geort_service import safe_id
    folder=Path(root)/'recordings'/safe_id(key);m=json.loads((folder/'manifest.json').read_text())
    if m.get('protocol')!=PROTOCOL:raise ValueError('旧录制的 CART 分类协议不兼容，请按五类动作重新录制或显式重标注')
    if not m.get('complete') or m.get('frames',0)<2:raise ValueError('录制未保存或有效帧不足')
    from .source_ik_initialization import validate_recorded_source
    validate_recorded_source(json.loads((folder/'profile.json').read_text()),m['source'])
    spec=stage(m['stage'])
    if m.get('category')!=spec['category']:raise ValueError('CART 动作与类别标签不一致，请核对录制或重标注')
    return folder,m

def input_digest(folder,m,full):
    from .cart_convex import geometry_identity
    from retarget.baseline.geort.adapter import REVISION as ADAPTER_REVISION
    return digest(dict(protocol=PROTOCOL,geometry=geometry_identity(),adapter_revision=ADAPTER_REVISION,source=m['source'],stage=m['stage'],
        points=m['points_sha256'],full=full,files={name:hashlib.sha256((folder/name).read_bytes()).hexdigest()
        for name in ('motion.npz','timestamps.npy','profile.json','topology.json')}))

def preprocessing_current(folder,m,full=True):
    """An existing target file is reusable only with the current geometry."""
    try:
        info=json.loads((folder/'cart_preprocess.json').read_text())
        return (info['profile']==('full' if full else 'smoke_test') and
            info['fingerprint']==input_digest(folder,m,full) and
            info['targets_sha256']==hashlib.sha256((folder/'cart_targets.jsonl').read_bytes()).hexdigest())
    except (OSError,ValueError,KeyError):return False

def _preprocess(root,key,full=True,progress=None,check=lambda:None):
    folder,m=manifest(root,key)
    if hashlib.sha256((folder/'points.npy').read_bytes()).hexdigest()!=m['points_sha256']:raise ValueError('原始录制指纹不一致')
    fingerprint=input_digest(folder,m,full)
    geometry=SourceGeometry(json.loads((folder/'topology.json').read_text()),json.loads((folder/'profile.json').read_text()))
    data=np.load(folder/'motion.npz');stamps=np.load(folder/'timestamps.npy');tracker=MotionTracker()
    temp=folder/'cart_targets.tmp';count=0;counts=dict(hull=0,hand=0,unavailable=0);contact_frames=0;hull_frames=0
    try:
        with temp.open('w') as out:
            for i,(q,stamp) in enumerate(zip(data['qpos'],stamps)):
                check();row,_,_=geometry.frame(q,m['stage'],float(stamp),tracker,full=full)
                target=row['target'];mode=target.get('contact_mode','unavailable');counts[mode]=counts.get(mode,0)+1
                if row['phase'] in ('closing','hold'):
                    if target.get('ready') and mode=='hull':contact_frames+=1;hull_frames+=1
                    elif mode=='hand' and all(p['gap_m']<=.002 for p in target['pairs']):contact_frames+=1
                out.write(json.dumps(row,allow_nan=False)+'\n');count+=1
                if progress and (i%10==0 or i==len(stamps)-1):progress('preprocess',dataset=key,completed=i+1,total=len(stamps),cycles=tracker.cycles)
        temp.replace(folder/'cart_targets.jsonl')
        from .cart_convex import geometry_identity
        info=dict(protocol=PROTOCOL,geometry=geometry_identity(),fingerprint=fingerprint,source=m['source'],frames=count,cycles=tracker.cycles,
                  targets_sha256=hashlib.sha256((folder/'cart_targets.jsonl').read_bytes()).hexdigest(),
                  contact_modes=counts,supervised_contact_frames=contact_frames,hull_frames=hull_frames,
                  contact_targets_valid=contact_frames>=2 and (m['category']==1 or hull_frames>=2),
                  complete_cycles=tracker.cycles>=2,profile='full' if full else 'smoke_test')
        atomic_json(folder/'cart_preprocess.json',info)
        return info
    finally:
        if temp.exists():temp.unlink()

def preprocess(root,key,full=True,progress=None,check=lambda:None):
    import fcntl
    from .geort_service import safe_id
    folder=Path(root)/'recordings'/safe_id(key)
    with (folder/'cart_preprocess.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('该会话已有预处理任务，请等待完成或取消')
        return _preprocess(root,key,full,progress,check)


def load_sessions(root,ids,split,require_complete=True):
    if not ids or len(set(ids))!=len(ids):raise ValueError('请选择不重复的 CART 会话')
    result=[];identity=None
    for key in ids:
        folder,m=manifest(root,key)
        if m['split']!=split:raise ValueError('训练与验证会话必须分开选择')
        if identity is not None and identity!=m['source']:raise ValueError('CART 会话设备、标定或坐标不兼容')
        identity=m['source']
        info=json.loads((folder/'cart_preprocess.json').read_text())
        expected=input_digest(folder,m,info['profile']=='full')
        if hashlib.sha256((folder/'points.npy').read_bytes()).hexdigest()!=m['points_sha256']:raise ValueError('原始录制指纹不一致')
        if expected!=info['fingerprint'] or hashlib.sha256((folder/'cart_targets.jsonl').read_bytes()).hexdigest()!=info['targets_sha256']:
            raise ValueError('CART 数据／预处理指纹不一致，请重新预处理')
        if require_complete and not info.get('complete_cycles'):raise ValueError('子动作尚未完成两次完整闭合—张开，请重录')
        if require_complete and not info.get('contact_targets_valid'):raise ValueError('该子动作缺少有效接触／凸包帧，请核对标定、接触动作后重录')
        rows=[json.loads(s) for s in (folder/'cart_targets.jsonl').read_text().splitlines()]
        if len(rows)!=m['frames']:raise ValueError('预处理帧数不一致')
        result.append((m,rows,info))
    return result,identity

def sequence_indices(rows,length=16):
    segments=[];start=0
    for end in range(1,len(rows)+1):
        if end==len(rows) or not 0<rows[end]['timestamp_ms']-rows[end-1]['timestamp_ms']<=250:
            for stride in (1,2,4,8):
                segments.extend(list(range(i,i+length*stride,stride)) for i in range(start,end-(length-1)*stride,max(1,length*stride//2)))
            start=end
    return segments
