"""Reachable voxels from the exact GeoRT training FK file, filtered by native meshes.

This is a finite Wuji2 training reference, not measured human reachability.
Only the coverage reference is filtered; training samples and weights are read-only.
"""
from pathlib import Path
import hashlib
import json
import os
import time
import numpy as np
from retarget.baseline.geort.adapter import FINGERS, ROOT, digest

REVISION='geort_training_fk_nonhyperextended_scatter_v5'
CELL_M=.005
PENETRATION_MM=5.
DENOMINATOR='geort_training_fk_output_extension_domain_penetration_filtered'


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def training_reference_spec(root):
    """Pure Python: safe in the SDK/web environment without torch or MuJoCo."""
    root=Path(root).resolve()
    # Published configuration, never an arbitrary historical FK directory.
    config=json.loads((root/'models/current/config.json').read_text())
    robot=config['robot'];robot_key=digest(robot)
    output_domain={key:config.get('output_domain',robot)[key] for key in ('lower','upper')}
    domain_poses(np.zeros((0,20)),output_domain)
    from .pose_domain import extension_domain
    from .adapter import fk_cache_key
    glove_policy=(config.get('output_domain') or {}).get('glove_extension_policy')
    policy_source='model_output_domain' if glove_policy else 'default_extension_bounds'
    if not glove_policy:
        datasets=config.get('datasets') or []
        if datasets:
            key=datasets[0].get('id')
            if not isinstance(key,str) or Path(key).name!=key or key.startswith('.'):
                raise ValueError('模型录制来源无效')
            profile_path=root/'recordings'/key/'profile.json'
            if profile_path.exists():
                profile=json.loads(profile_path.read_text())
                glove_policy=(profile.get('bone_extrinsics') or {}).get('joint_limit_policy',profile.get('joint_limit_policy'))
                policy_source='training_calibration_profile'
    reference_domain=extension_domain(output_domain['lower'],output_domain['upper'],glove_policy)
    labels=root/'fk'/fk_cache_key(config)/f"kinematics_{int(config['robot_samples'])}.npz"
    if not labels.is_file():raise FileNotFoundError('当前 GeoRT 模型的训练 FK 样本不存在')
    policy=dict(revision=REVISION,threshold_mm=PENETRATION_MM,voxel_m=CELL_M,
                dense=True,robot_key=robot_key,
                code_sha256={name:sha256(ROOT/name) for name in (
                    'retarget/baseline/card/collision.py','moxian_web/wuji_surface.py',
                    'retarget/env/grasp/pose_gpu.py','retarget/baseline/geort/workspace.py',
                    'retarget/baseline/geort/pose_domain.py',
                    'retarget/baseline/geort/sample_filter.py')})
    return dict(labels=str(labels),labels_sha256=sha256(labels),samples=int(config.get('retained_robot_samples',config['robot_samples'])),
                robot=robot,output_domain=output_domain,reference_domain=reference_domain,
                extension_policy_source=policy_source,
                training_mode=config.get('training_mode','official'),policy=policy)


def accepted_poses(depth_mm,query_valid,threshold_mm=PENETRATION_MM):
    depth=np.asarray(depth_mm);valid=np.asarray(query_valid,dtype=bool)
    if depth.ndim!=1 or valid.shape!=depth.shape:raise ValueError('无效穿透诊断数组')
    return valid & np.isfinite(depth) & (depth>=0) & (depth<=threshold_mm)


def domain_poses(qpos,output_domain):
    """Filter whole poses using exactly the published IK output limits."""
    q=np.asarray(qpos);lower=np.asarray(output_domain['lower']);upper=np.asarray(output_domain['upper'])
    if (q.ndim!=2 or q.shape[1]!=20 or lower.shape!=(20,) or upper.shape!=(20,)
            or not np.isfinite([lower,upper]).all() or np.any(lower>=upper)):
        raise ValueError('无效 GeoRT 输出姿态域')
    # These are saved float32 angles; use the same small representation tolerance
    # as the native-range verification, not an anatomical angle allowance.
    return np.isfinite(q).all(1) & (q>=lower-1e-6).all(1) & (q<=upper+1e-6).all(1)


def reference_cells(points,accepted):
    points=np.asarray(points);accepted=np.asarray(accepted,dtype=bool)
    if points.shape!=(len(accepted),5,3) or not np.isfinite(points).all():raise ValueError('无效训练 FK 指尖样本')
    grid=np.floor(points[accepted]/CELL_M+1e-7).astype(np.int32)
    return [np.unique(grid[:,i],axis=0) for i in range(5)]


def kinematic_spec(robot):
    """Static native model only; the browser evaluates FK after point selection."""
    m=robot.model
    return dict(parents=m.body_parentid.tolist(),positions=m.body_pos.tolist(),
        quaternions_wxyz=m.body_quat.tolist(),
        body_joints=[list(range(int(m.body_jntadr[b]),int(m.body_jntadr[b]+m.body_jntnum[b])))
                     if m.body_jntnum[b] else [] for b in range(m.nbody)],
        axes=m.jnt_axis.tolist(),pivots=m.jnt_pos.tolist(),kinds=m.jnt_type.tolist(),
        addresses=m.jnt_qposadr.tolist(),reference=m.qpos0.tolist(),
        links=list(robot.body_ids),basis=robot.basis.tolist(),
        tip_bodies=m.site_bodyid[robot.tip_ids].tolist(),tip_positions=m.site_pos[robot.tip_ids].tolist())


def build(folder):
    import torch
    from retarget.baseline.geort.adapter import RobotAdapter
    from retarget.baseline.geort.gpu import RobotFK
    from moxian_web.wuji_surface import WujiSurface
    from retarget.baseline.card.collision import WujiMeshCollision
    from moxian_web.warp_runtime import serialized
    from retarget.baseline.geort.training import atomic_json
    folder=Path(folder);source=json.loads((folder/'source.json').read_text())
    spec=json.loads((folder/'reference_spec.json').read_text());labels=Path(spec['labels'])
    device='cuda:0' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(1)
    started=time.monotonic()
    metadata=dict(samples=spec['samples'],voxel_mm=CELL_M*1000,denominator=DENOMINATOR,
        self_collision_filtered=True,penetration_threshold_mm=PENETRATION_MM,
        output_domain_filtered=True,output_domain=spec['output_domain'],training_mode=spec['training_mode'],
        extension_domain_filtered=True,reference_domain=spec['reference_domain'],
        extension_policy_source=spec['extension_policy_source'],
        reference_kind='saved_fk_samples_within_current_ik_domain',reference_cloud_kind='exact_saved_fk_keypoints',
        labels_sha256=spec['labels_sha256'],robot_key=spec['policy']['robot_key'],
        collision_method='native_mesh_vertices_and_face_centres_bidirectional',
        continuous_intersection_proof=False,
        checked_scope='cross-finger, palm and nonadjacent same-finger; native joint mechanical collars excluded')
    def progress(status,**values):
        atomic_json(folder/'status.json',dict(metadata,status=status,pid=os.getpid(),revision=REVISION,source=source,device=device,
                    elapsed_s=time.monotonic()-started,**values))
    try:
        progress('running',completed=0,total=spec['samples'])
        if sha256(labels)!=spec['labels_sha256']:raise ValueError('训练 FK 文件已变化，请重建参考')
        robot=RobotAdapter()
        if robot.spec!=spec['robot'] or robot.key!=spec['policy']['robot_key']:raise ValueError('原生机器人模型与训练 FK 指纹不符')
        with np.load(labels,allow_pickle=False) as arrays:
            q=arrays['qpos'];points=arrays['keypoints']
        if q.shape!=(spec['samples'],20) or not np.isfinite(q).all():raise ValueError('训练 FK 关节样本无效')
        if np.any(q<robot.lower-1e-6) or np.any(q>robot.upper+1e-6):raise ValueError('训练 FK 样本超出原生关节范围')
        fk=RobotFK(robot,device)
        # Saved keypoints, rather than regenerated coordinates, form the denominator.
        # Check their coordinate convention against true native FK before accepting them.
        probe=np.linspace(0,len(q)-1,min(128,len(q)),dtype=int)
        with torch.inference_mode():error=float(np.max(np.abs(fk(q[probe]).cpu().numpy()-points[probe])))
        if error>1e-6:raise ValueError(f'训练 FK 指尖坐标不一致: {error} m')
        metadata['fk_label_max_error_m']=error
        filter_key=digest([spec['labels_sha256'],spec['policy']])
        cache=labels.parent/f'workspace_filter_{filter_key}.npz'
        depths=None
        from .sample_filter import certified_depths
        certified=certified_depths(labels,robot.key,spec['output_domain'])
        if certified is not None:depths,valid=certified
        if cache.is_file():
            with np.load(cache,allow_pickle=False) as arrays:
                if str(arrays['filter_key'])!=filter_key:raise ValueError('穿透筛选缓存指纹不符')
                depths=arrays['max_penetration_mm'];valid=arrays['query_valid']
                if depths.shape!=(len(q),) or valid.shape!=(len(q),):raise ValueError('穿透筛选缓存维度不符')
        if depths is None:
            class Checker:
                def __init__(self):
                    self.device=device;self.collision=WujiMeshCollision(WujiSurface(),device)
                @serialized
                def query(self,values):
                    with torch.inference_mode():
                        x=torch.as_tensor(values,device=device,dtype=torch.float32)
                        p,r=fk.fk.frames(torch.cat((x.new_zeros((len(x),6)),x),1))
                        d,v=self.collision.query(p,r,dense=True)
                        return d.amax((1,2)).cpu().numpy(),v.cpu().numpy()
            checker=Checker();depths=np.empty(len(q),np.float32);valid=np.empty(len(q),bool)
            for start in range(0,len(q),128):
                end=min(start+128,len(q));depths[start:end],valid[start:end]=checker.query(q[start:end])
                if start%1024==0 or end==len(q):
                    progress('running',completed=end,total=len(q),accepted_samples=int(accepted_poses(depths[:end],valid[:end]).sum()))
        if not cache.is_file():
            temporary=cache.with_suffix('.tmp.npz')
            np.savez_compressed(temporary,filter_key=filter_key,max_penetration_mm=depths,query_valid=valid)
            temporary.replace(cache)
        collision_ok=accepted_poses(depths,valid)
        within_domain=domain_poses(q,spec['output_domain'])
        within_extension=domain_poses(q,spec['reference_domain'])
        accepted=collision_ok & within_domain & within_extension
        metadata.update(accepted_samples=int(accepted.sum()),
            collision_accepted_samples=int(collision_ok.sum()),
            output_domain_samples=int(within_domain.sum()),
            rejected_output_domain_samples=int((collision_ok & ~within_domain).sum()),
            rejected_extension_samples=int((collision_ok & within_domain & ~within_extension).sum()),
            rejected_penetration_samples=int((valid & np.isfinite(depths) & (depths>PENETRATION_MM)).sum()),
            rejected_invalid_samples=int((~valid | ~np.isfinite(depths) | (depths<0)).sum()),
            filter_key=filter_key,filter_cache=cache.name)
        if not accepted.any():raise ValueError('训练 FK 样本中没有同时满足当前输出角度范围和穿透阈值的姿态')
        cells=reference_cells(points,accepted)
        accepted_ids=np.flatnonzero(accepted)
        preview_ids=accepted_ids[::max(1,(len(accepted_ids)+1999)//2000)]
        # Pure data for the web process; it never needs a robot simulator or
        # the selected learned solver to inspect a reference sample.
        surface=WujiSurface()
        atomic_json(folder/'robot_topology.json',dict(surface.topology({}),kinematics=kinematic_spec(robot)))
        metadata['pose_preview_available']=True
        metadata['pose_preview_samples']=len(preview_ids)
        metadata['pose_cache']='joint_angles_only'
        temporary=folder/'reference.tmp.npz'
        np.savez_compressed(temporary,**dict(zip(FINGERS,cells)),keypoints_m=points[accepted],
            sample_indices=accepted_ids,preview_sample_indices=preview_ids,qpos=q[preview_ids])
        temporary.replace(folder/'reference.npz')
        progress('completed',completed=len(q),total=len(q),cells=[len(c) for c in cells],
            limits='原训练样本中满足IK输出域、标定反伸边界和5mm穿透阈值的有限子集；不保证人手可达、人体关节耦合或无碰撞' )
    except BaseException as exc:
        progress('failed',error=str(exc));raise


class WorkspaceReference:
    def __init__(self,source,cells,metadata=None,points=None,sample_indices=None,
                 qpos=None,preview_sample_indices=None):
        self.source=source;self.metadata=metadata or {};self.points=points
        self.cells=[set(map(tuple,np.asarray(c,dtype=int).reshape(-1,3))) for c in cells]
        self.sample_indices=sample_indices;self.qpos=qpos
        self.preview_sample_indices=sample_indices if preview_sample_indices is None else preview_sample_indices

    @classmethod
    def load(cls,folder):
        folder=Path(folder);status=json.loads((folder/'status.json').read_text())
        if status.get('status')!='completed' or status.get('revision')!=REVISION:raise ValueError('可达空间参考尚未完成')
        with np.load(folder/'reference.npz',allow_pickle=False) as arrays:
            status=dict(status,reference_id=folder.name)
            extra={name:arrays[key].copy() if key in arrays else None for name,key in (
                ('points','keypoints_m'),('sample_indices','sample_indices'),('qpos','qpos'),
                ('preview_sample_indices','preview_sample_indices'))}
            return cls(status['source'],[arrays[f] for f in FINGERS],status,**extra)

    @property
    def preview_available(self):
        return all(value is not None for value in (self.points,self.sample_indices,self.qpos))

    def pose(self,sample_id,finger,counts=None):
        if not self.preview_available:raise ValueError('旧参考缺少对应姿态，请重建姿态参考')
        if finger not in FINGERS:raise ValueError('无效手指')
        if isinstance(sample_id,bool) or not isinstance(sample_id,int):raise ValueError('无效姿态样本编号')
        found=np.flatnonzero(self.preview_sample_indices==sample_id)
        if len(found)!=1:raise ValueError('该姿态不属于当前已筛选参考')
        index=int(found[0]);f=FINGERS.index(finger)
        point_index=int(np.flatnonzero(self.sample_indices==sample_id)[0])
        cell=tuple(np.floor(self.points[point_index,f]/CELL_M+1e-7).astype(int))
        observed=int((counts[f] if counts else {}).get(cell,0))
        return dict(ready=True,reference_id=self.metadata.get('reference_id'),sample_id=sample_id,
            finger=finger,point_mm=(self.points[point_index,f]*1000).tolist(),
            keypoints_m=self.points[point_index].tolist(),joint_positions_rad=self.qpos[index].tolist(),
            recorded_cell_frames=observed,cell_observed=observed>0,coverage_cell_mm=CELL_M*1000,
            coverage_scope='selected_finger_cell_only',coordinate_frame='palm_canonical',
            robot_only=True)

    def evaluate(self,counts,finger=None):
        result=[]
        for i,name in enumerate(FINGERS):
            reference=self.cells[i];observed=counts[i] if counts else {}
            histogram={cell:int(count) for cell,count in observed.items() if cell in reference and count>0}
            k=len(histogram);total=len(reference);values=np.array(list(histogram.values()),float)
            probabilities=values/values.sum() if len(values) else values
            effective=float(np.exp(-(probabilities*np.log(probabilities)).sum())) if k else 0.
            item=dict(finger=name,reachable_cells=total,covered_cells=k,missing_cells=total-k,
                      coverage_percent=100*k/total if total else 0.,
                      uniformity_percent=100*effective/k if k>1 else 0.,
                      outside_reference_cells=sum(cell not in reference for cell in observed),
                      in_reference_frames=int(values.sum()))
            if name==finger:
                if self.points is not None:
                    # Plot actual FK sample positions, not voxel centres that may
                    # never have been reached by any sampled pose.
                    stride=max(1,(len(self.points)+1999)//2000)
                    cloud=self.points[::stride,i]
                    item['reference_cloud_mm']=(cloud*1000).tolist()
                    if self.preview_available:
                        item['reference_sample_ids']=self.sample_indices[::stride].tolist()
                else:
                    sorted_cells=sorted(reference)
                    item['reference_cloud_mm']=((np.asarray(sorted_cells[::max(1,(total+1999)//2000)]).reshape(-1,3)+.5)*CELL_M*1000).tolist()
                item['reference_cloud_counts']=[int(observed.get(tuple(c),0)) for c in
                    np.floor(np.asarray(item['reference_cloud_mm']).reshape(-1,3)/(CELL_M*1000)+1e-7).astype(int)]
                # API compatibility for offline consumers; the web view uses
                # only the two independent scatter clouds.
                item['projections']=self._projections(reference,histogram)
            result.append(item)
        return dict(self.metadata,status='completed',voxel_mm=CELL_M*1000,
                    denominator=DENOMINATOR,pose_preview_available=self.preview_available,fingers=result)

    @staticmethod
    def _projections(reference,histogram):
        result=[]
        for label,axes in (('XY',(0,1)),('XZ',(0,2)),('YZ',(1,2))):
            bins={}
            for cell in reference:
                key=(int(cell[axes[0]]),int(cell[axes[1]]));value=bins.setdefault(key,[0,0,0])
                value[0]+=1
                if cell in histogram:value[1]+=1;value[2]+=histogram[cell]
            result.append(dict(plane=label,cells=[[x,y,*v] for (x,y),v in sorted(bins.items())]))
        return result
