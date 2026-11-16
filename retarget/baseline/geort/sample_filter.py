"""Persist only whole robot poses that pass the native geometry checks."""
import json
import numpy as np
from .adapter import ROOT,digest
from .workspace import accepted_poses,domain_poses,sha256

POLICY='nonhyperextended_native_mesh_penetration_le_5mm_v1'


def geometry_spec(robot_key,domain):
    return dict(policy=POLICY,threshold_mm=5.,robot_key=robot_key,
        output_domain={k:domain[k] for k in ('lower','upper')},
        geometry_code_sha256={p:sha256(ROOT/p) for p in (
            'retarget/baseline/card/collision.py','moxian_web/wuji_surface.py',
            'retarget/env/grasp/pose_gpu.py')})


def cache_key(robot,domain,sampling_policy):
    return digest(dict(robot=robot,sampling_policy=sampling_policy,
        lower=domain['lower'],upper=domain['upper'],filter=geometry_spec(digest(robot),domain)))


def query_penetration(robot,q,device,progress,check):
    import torch
    from .gpu import RobotFK
    from moxian_web.wuji_surface import WujiSurface
    from retarget.baseline.card.collision import WujiMeshCollision
    from moxian_web.warp_runtime import serialized
    class Checker:
        def __init__(self):
            self.device=device;self.fk=RobotFK(robot,device)
            self.collision=WujiMeshCollision(WujiSurface(),device)
        @serialized
        def query(self,values):
            with torch.inference_mode():
                x=torch.as_tensor(values,device=device,dtype=torch.float32)
                p,r=self.fk.fk.frames(torch.cat((x.new_zeros((len(x),6)),x),1))
                depth,valid=self.collision.query(p,r,dense=True)
                return depth.amax((1,2)).cpu().numpy(),valid.cpu().numpy()
    checker=Checker();depth=np.empty(len(q),np.float32);valid=np.empty(len(q),bool)
    for start in range(0,len(q),128):
        check();end=min(start+128,len(q));depth[start:end],valid[start:end]=checker.query(q[start:end])
        progress('robot_pose_screening',completed=end,total=len(q))
    return depth,valid


def retained_arrays(q,points,domain,depth,valid,spec):
    q=np.asarray(q);points=np.asarray(points);depth=np.asarray(depth);valid=np.asarray(valid)
    if q.ndim!=2 or q.shape[1]!=20 or points.shape!=(len(q),5,3) or depth.shape!=(len(q),) or valid.shape!=depth.shape:
        raise ValueError('无效机器人采样数组')
    keep=domain_poses(q,domain)&np.isfinite(points).all((1,2))&accepted_poses(depth,valid)
    if not keep.any():raise ValueError('没有通过姿态及网格检查的机器人样本')
    arrays=dict(qpos=q[keep].copy(),keypoints=points[keep].copy(),
        max_penetration_mm=depth[keep].copy(),query_valid=valid[keep].copy(),
        screening_spec=np.array(json.dumps(spec,sort_keys=True,allow_nan=False)))
    return arrays,dict(proposed_samples=len(q),retained_samples=int(keep.sum()),
        deleted_samples=int((~keep).sum()),policy=POLICY)


def certified_depths(labels,robot_key,domain):
    """Reuse verified queries without retaining any rejected pose records."""
    with np.load(labels,allow_pickle=False) as a:
        if 'screening_spec' not in a:return None
        if json.loads(str(a['screening_spec']))!=geometry_spec(robot_key,domain):
            raise ValueError('机器人样本筛选策略已变化，请重新采样')
        q=a['qpos'];p=a['keypoints'];depth=a['max_penetration_mm'];valid=a['query_valid']
    if (p.shape!=(len(q),5,3) or depth.shape!=(len(q),) or valid.shape!=depth.shape
            or not np.isfinite(p).all() or not domain_poses(q,domain).all()
            or not accepted_poses(depth,valid).all()):
        raise ValueError('筛选后的机器人样本仍含违规姿态')
    return depth,valid
