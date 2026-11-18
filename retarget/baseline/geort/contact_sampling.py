"""Contact-mode robot targets: saved FK poses in the actual IK output domain."""
import json
from pathlib import Path
import numpy as np
from .adapter import ROOT,digest
from .workspace import domain_poses,accepted_poses,sha256

REVISION='ik_domain_native_mesh_targets_v1'


def robot_targets(root,robot,output_domain,labels,device,progress,check):
    """Filter original labels, never clamp angles or use neural FK as ground truth.

    Apply the output-domain filter first, also accepting legacy label files.
    New training labels already lie entirely inside this domain.
    """
    import torch
    from .training import atomic_json
    from .gpu import RobotFK
    from moxian_web.wuji_surface import WujiSurface
    from retarget.baseline.card.collision import WujiMeshCollision
    from moxian_web.warp_runtime import serialized
    labels=Path(labels);threshold_mm=5.
    spec=dict(revision=REVISION,labels_sha256=sha256(labels),robot_key=robot.key,
        output_domain={k:output_domain[k] for k in ('lower','upper')},penetration_threshold_mm=threshold_mm,
        code_sha256={str(p):sha256(ROOT/p) for p in map(Path,(
            'retarget/baseline/card/collision.py','moxian_web/wuji_surface.py',
            'retarget/env/grasp/pose_gpu.py','retarget/baseline/geort/contact_sampling.py'))})
    folder=Path(root)/'robot_contact_targets'/digest(spec);folder.mkdir(parents=True,exist_ok=True)
    with np.load(labels,allow_pickle=False) as arrays:q=arrays['qpos'];points=arrays['keypoints']
    if q.ndim!=2 or q.shape[1]!=20 or points.shape!=(len(q),5,3) or not np.isfinite(points).all():
        raise ValueError('训练 FK 标签无效')
    within=domain_poses(q,output_domain);candidates=np.flatnonzero(within);check()
    if not len(candidates):raise ValueError('训练FK样本中没有符合当前输出域的姿态')
    if (folder/'targets.npz').is_file() and (folder/'manifest.json').is_file():
        manifest=json.loads((folder/'manifest.json').read_text())
        with np.load(folder/'targets.npz',allow_pickle=False) as arrays:
            indices=arrays['indices'];depth=arrays['max_penetration_mm'];valid=arrays['query_valid']
        if manifest['spec']!=spec or depth.shape!=(len(candidates),) or valid.shape!=depth.shape:
            raise ValueError('机器人目标缓存指纹或维度不符')
        expected=candidates[accepted_poses(depth,valid,threshold_mm)]
        if not np.array_equal(indices,expected):raise ValueError('机器人目标缓存筛选结果不符')
    else:
        progress('robot_contact_targets',completed=0,total=len(candidates),native_samples=len(q))
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
        from .sample_filter import certified_depths
        certified=certified_depths(labels,robot.key,output_domain)
        if certified is not None:
            depth,valid=(value[candidates] for value in certified)
        else:
            checker=Checker();depth=np.empty(len(candidates),np.float32);valid=np.empty(len(candidates),bool)
            for start in range(0,len(candidates),128):
                check();end=min(start+128,len(candidates))
                depth[start:end],valid[start:end]=checker.query(q[candidates[start:end]])
                progress('robot_contact_targets',completed=end,total=len(candidates),native_samples=len(q))
        indices=candidates[accepted_poses(depth,valid,threshold_mm)]
        tmp=folder/'targets.tmp.npz'
        np.savez_compressed(tmp,indices=indices,max_penetration_mm=depth,query_valid=valid)
        tmp.replace(folder/'targets.npz')
        manifest=dict(spec=spec,total_samples=len(q),output_domain_samples=len(candidates),accepted_samples=len(indices),
            rejected_output_domain_samples=int((~within).sum()),
            rejected_penetration_samples=int((valid & (depth>threshold_mm)).sum()),
            rejected_invalid_samples=int((~valid | ~np.isfinite(depth) | (depth<0)).sum()),
            note='Finite subset of original training FK poses; <=5mm does not mean collision-free.')
        atomic_json(folder/'manifest.json',manifest)
    if not len(indices):raise ValueError('当前输出域中没有通过5mm穿透筛选的机器人目标')
    return points[indices].transpose(1,0,2),dict(manifest,cache_id=folder.name)
