"""Read-only cached IK playback. This worker never invokes an IK solve."""
import argparse
import base64
import contextlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from retarget.baseline.geort.adapter import RobotAdapter,SourceAdapter
from retarget.baseline.geort.runtime import Replay
from .cart_initialization import IK_VERSION,identity,RetargetSession
from .cart_data import file_hash
from .wuji_surface import WujiSurface


class IKReplay:
    def __init__(self,folder):
        self.folder=Path(folder);self.key=self.folder.name
        try:self.info=json.loads((self.folder/'cart_ik.json').read_text())
        except FileNotFoundError:raise ValueError('请先点击“准备观测、触点与 IK”，完成后即可回放')
        self.robot=RobotAdapter()
        if self.info['version']!=IK_VERSION or self.info['sha256']!=file_hash(self.folder/'cart_ik.npz'):
            raise ValueError('IK 缓存损坏或版本不兼容，请重新准备')
        session=RetargetSession.for_hand('WujiHand2','right')
        if self.info['fingerprint']!=identity(self.folder,self.robot,session):raise ValueError('IK 缓存已过期，请重新准备')
        self.source=Replay(self.folder,'cpu');self.adapter=SourceAdapter(self.source.topology)
        with np.load(self.folder/'cart_ik.npz',allow_pickle=False) as data:
            self.q=data['q'].copy();self.valid=data['valid'].copy();self.limited=data['domain_limited'].copy()
            self.fallback=data['fallback'].copy()
            if not np.array_equal(data['timestamps_ms'],self.source.stamps):raise ValueError('IK 与手套时间轴不一致')
        if self.q.shape!=(len(self.source.stamps),20) or not np.isfinite(self.q[self.valid]).all():raise ValueError('无效 IK 轨迹')
        from retarget.baseline.geort.pose_domain import extension_domain
        domain=extension_domain(self.robot.lower,self.robot.upper)
        if np.any(self.q[self.valid]<np.asarray(domain['lower'])-1.e-6) or np.any(self.q[self.valid]>np.asarray(domain['upper'])+1.e-6):
            raise ValueError('IK 缓存超出 CART 关节域')
        self.surface=WujiSurface();self.segments=[];segment=0
        for i,stamp in enumerate(self.source.stamps):
            rows=self.source.contact_rows
            if i and (not 0<stamp-self.source.stamps[i-1]<=250 or rows and rows[i].get('input_stream_epoch')!=rows[i-1].get('input_stream_epoch')):segment+=1
            self.segments.append(segment)

    def topology(self):
        stamps=self.source.stamps
        tips=[self.robot.fk(q).tolist() if valid else None for q,valid in zip(self.q,self.valid)]
        return dict(ready=True,id=self.key,frames=len(stamps),label=self.source.meta.get('label') or self.key,
            timestamps_ms=stamps.tolist(),duration_s=float(stamps[-1]-stamps[0])/1000,
            source=self.source.topology,robot=self.surface.topology({}),available=self.valid.tolist(),
            segments=self.segments,coordinate_frame='shared_canonical_palm',joint_names=self.robot.joint_names,
            source_tip_trajectories_m=np.load(self.folder/'points.npy',allow_pickle=False).tolist(),
            robot_tip_trajectories_m=tips,initialization_only=True,ik=self.info)

    @torch.inference_mode()
    def frame(self,index):
        if not 0<=index<len(self.q):raise ValueError('IK 回放帧号越界')
        raw=torch.as_tensor(self.source.motion['qpos'][index],dtype=torch.float32)[None]
        vertices,joints,_=self.source.model(raw)
        vertices=vertices[0].numpy();joints=joints[0].numpy()[self.adapter.indices];root=joints[0].copy()
        vertices=(vertices-root)@self.adapter.transform;joints=(joints-root)@self.adapter.transform
        selected=bool(self.valid[index]);positions=np.empty((0,21,3));rotations=np.empty((0,21,3,3))
        if selected:
            self.robot.fk(self.q[index]);data=self.robot.data;ids=self.robot.body_ids
            positions=((data.xpos[ids]-data.xpos[ids[0]])@self.robot.basis)[None]
            rotations=(self.robot.basis.T@data.xmat[ids].reshape(-1,3,3))[None]
        arrays=dict(source_vertices_m=vertices,source_joints_m=joints,link_positions_m=positions,link_rotations=rotations)
        fields={};chunks=[];offset=0
        for name,values in arrays.items():
            values=np.ascontiguousarray(values,dtype='<f4')
            if not np.isfinite(values).all():raise ValueError('IK 回放几何包含非有限值')
            fields[name]=dict(offset=offset,shape=list(values.shape));blob=values.tobytes();chunks.append(blob);offset+=len(blob)
        return dict(ready=True,id=self.key,index=index,timestamp_ms=float(self.source.stamps[index]),
            source_timestamp_ms=float(self.source.stamps[index]),robot_timestamp_ms=float(self.source.stamps[index]),
            seconds=float(self.source.stamps[index]-self.source.stamps[0])/1000,segment=self.segments[index],
            selected=selected,joint_positions_rad=self.q[index].tolist() if selected else None,
            domain_limited=bool(self.limited[index]),solver_fallback=bool(self.fallback[index]),
            geometry_f32=dict(format='cart-ik-f32-v1',fields=fields,data=base64.b64encode(b''.join(chunks)).decode()))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--folder',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    with contextlib.redirect_stdout(sys.stderr):replay=IKReplay(args.folder)
    for line in sys.stdin:
        try:
            request=json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):response=replay.topology() if request.get('topology') else replay.frame(int(request['index']))
            print(json.dumps(response,allow_nan=False,separators=(',',':')),flush=True)
        except Exception as exc:print(json.dumps(dict(error=str(exc)),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
