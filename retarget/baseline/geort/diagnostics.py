"""Read-only signed mesh/contact audits. Never modifies samples or network qpos."""
import argparse
import contextlib
import json
import sys
import time
from pathlib import Path
import numpy as np
from .adapter import FINGERS, SourceAdapter, ROOT

GAP_MM=2.
PENETRATION_MM=.1
NORMAL_DEG=30.


def contact_state(gap,penetration,normal,valid=True):
    if not valid or not np.isfinite([gap,penetration,normal]).all():return 'unknown'
    if penetration>PENETRATION_MM:return 'penetrating'
    if gap>GAP_MM:return 'separated'
    return 'contact' if normal<=NORMAL_DEG else 'normal_mismatch'


def source_allowed(names):
    """Exclude the seams between immediate skin-bone neighbours only."""
    allowed=np.ones((len(names),len(names)),bool);np.fill_diagonal(allowed,False)
    for f in ('thumb','index','middle','ring','little'):
        parts=('metacarpal','proximal','distal') if f=='thumb' else ('metacarpal','proximal','intermediate','distal')
        chain=[next(i for i,n in enumerate(names) if n.endswith('_wrist'))]
        chain += [next(i for i,n in enumerate(names) if n.endswith('_'+f+'_'+p)) for p in parts]
        for a,b in zip(chain,chain[1:]):allowed[a,b]=allowed[b,a]=False
    return allowed


class MeshDiagnostics:
    def __init__(self,topology,profile,device=None):
        import torch
        from moxian_web.qnbot_surface import QnbotSurface
        from moxian_web.qnbot_collision import QnbotCollision
        from .diagnostic_queries import SurfacePairs
        self.torch=torch;self.device=device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.topology=topology;self.adapter=SourceAdapter(topology)
        if profile.get('model_hash')!=topology.get('model_hash'):raise ValueError('诊断标定模型与源拓扑不匹配')
        self.source=QnbotSurface(topology['hand_side'],self.device,finger_twist=True);self.source.load_artifact(profile)
        if self.source.version()!=topology['geometry_version']:raise ValueError('诊断源几何指纹不匹配')
        self.collision=QnbotCollision(self.source,dense=True)
        self.allowed=torch.as_tensor(source_allowed(self.source.bone_names),device=self.device)
        self.names=self.source.bone_names
        self.edges=[dict(row=0,regions=['thumb_distal',f+'_distal']) for f in FINGERS[1:]]
        rest=torch.as_tensor(topology['rest_vertices_m'],device=self.device,dtype=torch.float32)[None]
        self.contact=SurfacePairs(rest[0],self.source.faces,topology['regions'],self.edges)
        self.robot=None

    def _robot_init(self):
        import torch
        from moxian_web.wuji_surface import WujiSurface
        from .diagnostic_queries import SurfacePairs
        from retarget.env.grasp.pose_gpu import ModelFK
        from retarget.baseline.card.collision import WujiMeshCollision
        self.robot=WujiSurface();self.fk=ModelFK(self.robot.model,self.device)
        self.robot_collision=WujiMeshCollision(self.robot,self.device)
        self.robot_local=torch.as_tensor(self.robot.local,device=self.device,dtype=torch.float32)
        self.robot_owners=torch.as_tensor(self.robot.owners,device=self.device,dtype=torch.long)
        self.robot_faces=torch.as_tensor(self.robot.faces,device=self.device,dtype=torch.long)
        face_owners=self.robot.owners[self.robot.faces[:,0]]
        regions={f+'_distal':np.flatnonzero(face_owners==4+4*i).tolist() for i,f in enumerate(FINGERS)}
        p,r=self.fk.frames(torch.zeros((1,26),device=self.device))
        self.robot_contact=SurfacePairs(self._robot_vertices(p,r),self.robot_faces,regions,self.edges)

    def _robot_vertices(self,p,r):
        p=p[0,self.robot.body_ids];r=r[0,self.robot.body_ids]
        return (r[self.robot_owners]@self.robot_local[...,None]).squeeze(-1)+p[self.robot_owners]

    def analyze(self,vertices,angles=None):
        torch=self.torch;started=time.monotonic()
        v=torch.as_tensor(vertices,device=self.device,dtype=torch.float32)
        if v.shape!=(len(self.source.vertices0),3) or not torch.isfinite(v).all():raise ValueError('诊断源网格无效')
        with torch.inference_mode():
            w=self.collision.query(v,part_allowed=self.allowed)
            depth=w['part_depths_mm']
            source_gap,source_normal=self.contact.evaluate(v)
            source_depth=[]
            for name in FINGERS[1:]:
                actual='little' if name=='pinky' else name
                thumb=torch.as_tensor(['thumb' in n for n in self.names],device=self.device)
                finger=torch.as_tensor([actual in n for n in self.names],device=self.device)
                mask=(thumb[:,None]&finger[None])|(finger[:,None]&thumb[None])
                source_depth.append(float(depth[mask].max()))
            source=dict(max_penetration_mm=float(depth.max()) if len(depth) else 0.,query_valid=True,
                method='original_skin_signed_vertices_and_face_centres',scope='all_skin_parts_except_immediate_bone_seams')
            robot=None;robot_gap=robot_normal=robot_depth=None
            if angles is not None:
                q=torch.as_tensor(angles,device=self.device,dtype=torch.float32)
                if q.shape!=(20,) or not torch.isfinite(q).all():raise ValueError('诊断机器人关节角无效')
                if self.robot is None:self._robot_init()
                ranges=torch.as_tensor(self.robot.model.jnt_range[6:],device=self.device,dtype=torch.float32)
                if ((q<ranges[:,0]-1e-6)|(q>ranges[:,1]+1e-6)).any():raise ValueError('诊断角度超出原生关节范围')
                p,r=self.fk.frames(torch.cat((q.new_zeros(6),q))[None])
                raw,valid=self.robot_collision.query(p,r,dense=True)
                robot=self.robot_collision.report(raw[0],valid[0])
                rv=self._robot_vertices(p,r)
                robot_gap,robot_normal=self.robot_contact.evaluate(rv)
                matrix=raw[0]
                robot_depth=[float(torch.maximum(matrix[1:5,1+4*f:5+4*f].max(),matrix[1+4*f:5+4*f,1:5].max())) for f in range(1,5)]
            pairs=[]
            for i,finger in enumerate(FINGERS[1:]):
                one=dict(gap_mm=source_gap[i],penetration_mm=source_depth[i],normal_error_deg=source_normal[i])
                one['state']=contact_state(one['gap_mm'],one['penetration_mm'],one['normal_error_deg'])
                two=None
                if robot is not None:
                    two=dict(gap_mm=robot_gap[i],penetration_mm=robot_depth[i],normal_error_deg=robot_normal[i])
                    two['state']=contact_state(two['gap_mm'],two['penetration_mm'],two['normal_error_deg'],robot['query_valid'])
                pairs.append(dict(finger=finger,source=one,robot=two))
        return dict(source=source,robot=robot,pairs=pairs,processing_ms=(time.monotonic()-started)*1000,
            device=self.device,thresholds=dict(contact_gap_mm=GAP_MM,penetration_mm=PENETRATION_MM,normal_error_deg=NORMAL_DEG),
            continuous_intersection_proof=False,read_only=True,training_loss_changed=False)


def worker():
    """One request in flight; parent samples frames without buffering capture."""
    import torch
    torch.set_num_threads(1);analyzer=None;key=None
    for line in sys.stdin:
        request={}
        try:
            request=json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                if request.get('context'):
                    c=request['context'];analyzer=MeshDiagnostics(c['topology'],c['profile']);key=request['context_key']
                if analyzer is None or key!=request['context_key']:raise ValueError('诊断上下文已变化')
                frame=request['frame']
                if request.get('robot'):
                    from moxian_web.wuji_transport import unpack_result_geometry
                    frame=unpack_result_geometry(frame)
                    vertices=np.asarray(frame['source_vertices_m'])@analyzer.adapter.mirror
                    angles=frame['joint_positions_rad']
                else:
                    from moxian_web.wuji_transport import decode_source_frame
                    frame=decode_source_frame(frame);vertices=frame['vertices_m'];angles=None
                result=analyzer.analyze(vertices,angles)
            result.update(ready=True,status='ready',request_id=request['request_id'],context_key=key,
                geometry_version=analyzer.topology['geometry_version'],timestamp_ms=request['timestamp_ms'],
                input_kind=request.get('input_kind','live'),model_run_id=frame.get('model_run_id'),
                robot_reason=request.get('robot_reason'))
        except Exception as exc:result=dict(ready=False,status='error',error=str(exc),request_id=request.get('request_id'))
        print(json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)


def recording_audit(root,key,model_id='current',stride=50,max_frames=64):
    import torch
    from .training import Predictor,load_recordings
    from moxian_web.geort_service import safe_id
    folder=Path(root)/'recordings'/safe_id(key);points,identity,_=load_recordings(root,[key])
    model=Predictor(Path(root)/'models'/safe_id(model_id))
    if identity!=model.config['source']:raise ValueError('诊断录制与模型源不兼容')
    with contextlib.redirect_stdout(sys.stderr):
        analyzer=MeshDiagnostics(json.loads((folder/'topology.json').read_text()),json.loads((folder/'profile.json').read_text()))
        with np.load(folder/'motion.npz') as motion:angles=motion['qpos']
        rows=[]
        for index in list(range(0,len(points),stride))[:max_frames]:
            with torch.inference_mode():vertices=analyzer.source(torch.as_tensor(angles[index],device=analyzer.device)[None])[0][0]
            rows.append(dict(frame_index=index,**analyzer.analyze(vertices,model.predict(points[index]))))
    return dict(recording=key,model_run_id=model.config.get('run_id'),sampled=True,stride=stride,frames=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',action='store_true')
    p.add_argument('--root',type=Path,default=ROOT/'output/moxian/geort');p.add_argument('--recording')
    p.add_argument('--model',default='current');p.add_argument('--stride',type=int,default=50);p.add_argument('--max-frames',type=int,default=64)
    p.add_argument('--output',type=Path);args=p.parse_args()
    if args.worker:return worker()
    if not args.recording or args.stride<1 or args.max_frames<1:p.error('需要recording及正数stride/max-frames')
    from moxian_web.saved_assets import asset_lock
    with asset_lock(args.root):result=recording_audit(args.root,args.recording,args.model,args.stride,args.max_frames)
    rendered=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)
    if args.output:args.output.write_text(rendered)
    else:print(rendered)


if __name__=='__main__':main()
