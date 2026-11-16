"""GeoRT objectives with nonhyperextended pose sampling and optional contact losses.

Official loss equations follow geort/trainer.py at UPSTREAM; both modes restrict
robot samples and IK output to the nonhyperextended robot flexion domain. MuJoCo only
replaces the robot FK label generator; it is never used as an IK teacher.
The optional contact_v1 mode is an adaptation, not an official reproduction.
"""
import importlib.util
import json
import os
from pathlib import Path
import signal
import time
from functools import lru_cache
import numpy as np
from retarget.baseline.geort.adapter import ROOT, UPSTREAM, REVISION, RobotAdapter, digest


@lru_cache(maxsize=None)
def reference(name):
    path = ROOT/'retarget/baseline/geort/upstream'/f'{name}.py'
    spec = importlib.util.spec_from_file_location('geort_reference_'+name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)); tmp.replace(path)


def load_recordings(root, ids):
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('请选择不重复的录制数据')
    records=[]; points=[]; identity=None
    for key in ids:
        if not isinstance(key, str) or Path(key).name != key or not key or key.startswith('.'):
            raise ValueError('无效录制ID')
        folder=Path(root)/'recordings'/key
        meta=json.loads((folder/'manifest.json').read_text())
        if not meta.get('complete') or meta.get('frames',0)<2:
            raise ValueError('录制尚未保存或有效帧不足')
        source=meta['source'];p=np.load(folder/'points.npy', allow_pickle=False)
        profile_path=folder/'profile.json'
        if profile_path.is_file():
            from moxian_web.source_ik_initialization import validate_recorded_source
            validate_recorded_source(json.loads(profile_path.read_text()), source)
        if p.shape != (meta['frames'],5,3) or not np.isfinite(p).all():
            raise ValueError('无效的指尖数据')
        import hashlib
        if hashlib.sha256((folder/'points.npy').read_bytes()).hexdigest()!=meta['points_sha256']:
            raise ValueError('录制数据指纹不匹配')
        record=dict(id=key,sha256=meta['points_sha256'])
        if source.get('revision')=='moxian_wuji2_geort_v1':
            from retarget.baseline.geort.adapter import SourceAdapter
            adapter=SourceAdapter(json.loads((folder/'topology.json').read_text()))
            with np.load(folder/'motion.npz',allow_pickle=False) as motion:joints=motion['joints']
            if len(joints)!=len(p) or not np.isfinite(joints).all():raise ValueError('旧录制缺少完整同步骨架，无法修正坐标')
            landmarks=joints[:,adapter.indices]
            p=((landmarks[:,[4,8,12,16,20]]-landmarks[:,0,None])@adapter.transform).astype(np.float32)
            source=dict(source,**adapter.spec)
            record.update(coordinate_reprojection=dict(from_revision=meta['source']['revision'],to_revision=REVISION,
                          input='motion.npz:joints',effective_points_sha256=hashlib.sha256(p.tobytes()).hexdigest()))
        if identity is not None and source!=identity:raise ValueError('录制来自不同设备、手侧、标定或坐标系，不能合并')
        identity=source
        records.append(record);points.append(p)
    return np.concatenate(points), identity, records


def training_losses(point, ik_model, fk_model, robot_points, *, efficient=False):
    """Preserve upstream RNG order, all-pairs pinch, perturbations and weights."""
    import torch
    import torch.nn.functional as F
    if efficient:
        from .gpu import nearest_chamfer as chamfer_distance
    else:chamfer_distance=reference('loss').chamfer_distance
    embedded=fk_model(ik_model(point))
    pinch=point.new_zeros(())
    for i in range(point.size(1)):
        for j in range(i+1, point.size(1)):
            mask=(torch.norm(point[:,i]-point[:,j],dim=-1)<.015).float()
            distance=((embedded[:,i]-embedded[:,j])**2).sum(-1)
            pinch=pinch+(mask*distance).mean()/(mask.sum()+1e-7)*point.size(0)
    direction=F.normalize(torch.randn_like(point),dim=-1,p=2)
    delta=direction*.002
    curvature=((fk_model(ik_model(point+delta))+fk_model(ik_model(point-delta))-2*embedded)**2).mean()
    selected=np.random.randint(0,robot_points.shape[1],2048)
    target_counts=None
    if efficient:
        # Repeated draws are common in a filtered FK cloud. Preserve their
        # original multiplicities and first-occurrence tie order, querying each
        # distinct target once rather than inflating the distance matrix.
        unique,first,counts=np.unique(selected,return_index=True,return_counts=True)
        order=np.argsort(first);selected=unique[order]
        target_counts=torch.as_tensor(counts[order],device=point.device)
    if torch.is_tensor(robot_points):
        target=robot_points[:,torch.as_tensor(selected,device=point.device)].permute(1,0,2)
    else:target=torch.from_numpy(robot_points[:,selected,:]).permute(1,0,2).float().to(point.device)
    chamfer=point.new_zeros(())
    for i in range(point.size(1)):
        args=(embedded[:,i].unsqueeze(0),target[:,i].unsqueeze(0))
        chamfer=chamfer+(chamfer_distance(*args,target_counts=target_counts) if efficient else chamfer_distance(*args))
    direction=F.normalize(torch.randn_like(point),dim=-1,p=2)
    # Upstream draws this scalar on the CPU, then transfers it to CUDA. Using
    # the CUDA RNG changes fixed-seed perturbations and optimizer updates.
    scale=.001+torch.rand(point.size(0)).to(point.device).unsqueeze(-1).unsqueeze(-1)*.01
    shifted=point+direction*scale
    d1=(shifted-point).reshape(-1,3)
    d2=(fk_model(ik_model(shifted))-embedded).reshape(-1,3)
    direction_loss=-(F.normalize(d1,dim=-1,p=2,eps=1e-5)*F.normalize(d2,dim=-1,p=2,eps=1e-5)).sum(-1).mean()
    collision=point.new_zeros(())
    parts=dict(direction=direction_loss,chamfer=chamfer,curvature=curvature,pinch=pinch,collision=collision)
    return direction_loss+80*chamfer+.1*curvature+pinch, parts


def train(root, dataset_ids, run_id, epochs=200, fk_epochs=200, seed=714,
          robot_samples=100000, resample_to=50000, batch_size=2048, mode='official'):
    import fcntl
    import torch
    from torch.utils.data import DataLoader
    root=Path(root); folder=root/'training'/run_id; folder.mkdir(parents=True,exist_ok=True)
    root.mkdir(parents=True,exist_ok=True)
    lock=(root/'training.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise ValueError('已有 GeoRT 训练正在运行')
    cancelled=False
    def stop(*_):
        nonlocal cancelled
        cancelled=True
    old={s:signal.signal(s,stop) for s in (signal.SIGTERM,signal.SIGINT)}
    def check():
        if cancelled:raise InterruptedError('训练已取消，上一已启用模型保持不变')
    started=time.time()
    device_info={}
    def progress(phase, **value):
        status=dict(run_id=run_id,status='running',pid=os.getpid(),phase=phase,elapsed_s=time.time()-started,device=device_info,training_mode=mode,
                    constraint_weights=config.get('constraint_weights'),**value)
        atomic_json(folder/'status.json',status)
        with (folder/'trace.jsonl').open('a') as f:f.write(json.dumps(status,allow_nan=False)+'\n')
        print(json.dumps(status,ensure_ascii=False),flush=True)
    try:
        if mode not in ('official','contact_v1'):raise ValueError('未知的 GeoRT 训练模式')
        if not torch.cuda.is_available():raise ValueError('完整 GeoRT 训练需要 CUDA')
        device_info.update(network='cuda:0',gpu_name=torch.cuda.get_device_name(0),robot_fk='cuda:0',
                           preprocessing='cuda:0',voxel_backend='torch_cuda_open3d_legacy_centroids')
        if mode=='contact_v1':device_info.update(geometry='cuda:0',voxel_backend='synchronized_whole_pose_numpy_unique',pose_indexing='cpu')
        torch.set_num_threads(1);torch.manual_seed(seed);np.random.seed(seed)
        points,source,records=load_recordings(root,dataset_ids)
        if source.get('revision')!=REVISION:raise ValueError('录制坐标版本不兼容，请重新采集')
        from retarget.baseline.geort.gpu import RobotFK,human_dataset
        robot=RobotAdapter()
        from .pose_domain import extension_domain,sample_poses,SAMPLING_POLICY
        profile=json.loads((root/'recordings'/dataset_ids[0]/'profile.json').read_text())
        glove_policy=(profile.get('bone_extrinsics') or {}).get('joint_limit_policy',profile.get('joint_limit_policy',{}))
        required=('pip_extension_deg','dip_extension_deg','thumb_extension_deg')
        if any(k not in glove_policy for k in required):raise ValueError('标定缺少 Glove 物理伸直位的反伸限位策略')
        output_domain=extension_domain(robot.lower,robot.upper,glove_policy)
        from .sample_filter import cache_key as filtered_cache_key,query_penetration,retained_arrays,geometry_spec,certified_depths,POLICY
        cache_key=filtered_cache_key(robot.spec,output_domain,SAMPLING_POLICY)
        cache=root/'fk'/cache_key;cache.mkdir(parents=True,exist_ok=True)
        exact_fk=RobotFK(robot)
        config=dict(upstream=UPSTREAM,revision=REVISION,source=source,datasets=records,robot=robot.spec,
                    seed=seed,epochs=epochs,fk_epochs=fk_epochs,robot_samples=robot_samples,
                    resample_to=resample_to,batch_size=batch_size,weights=dict(direction=1,chamfer=80,curvature=.1,pinch=1,collision=0),
                    device=device_info,run_id=run_id,model_policy='replace_on_success',
                    checkpoint_policy='latest_only',
                    direction_scale_rng='torch_cpu_upstream',training_mode=mode,
                    output_domain=output_domain,fk_cache_key=cache_key,
                    robot_sampling_policy=SAMPLING_POLICY,
                    reproduction_profile='official_objective_bounded_pose_sampling' if (robot_samples,resample_to,batch_size)==(100000,50000,2048) else 'bounded_pose_smoke_test')
        if mode=='contact_v1':
            from .constraints import WEIGHTS,LOSS_POLICY
            config.update(constraint_weights=WEIGHTS,constraint_policy=LOSS_POLICY,
                          constraint_batch_size=64,reproduction_profile='contact_v1' if (robot_samples,resample_to,batch_size)==(100000,50000,2048) else 'contact_v1_smoke_test',
                          constraint_normalization=dict(penetration_mm=.5,contact_mm=2.,normal_deg=30.,bone_direction_deg=15.,bending_deg=10.,pinch_position_mm=15.,flat_direction_deg=15.),
                          geometry_batchnorm='inference_statistics_gradients_enabled',
                          contact_intent='source_mesh_hysteresis_or_upstream_15mm_tip_distance',
                          contact_reduction='equal_mean_per_observed_pair',
                          contact_shape_scale=.1,flat_hand_shape_scale=10.,
                          sampling='synchronized_whole_pose_1mm_cells_not_independent_fingers',
                          fk_collision_rejection=True,inference_postprocessing=False)
        config.update(robot_sample_filter_policy=POLICY,fk_collision_rejection=True)
        atomic_json(folder/'config.json',config)
        np.save(folder/'human.npy',points)
        labels=cache/f'kinematics_{robot_samples}.npz'
        if not labels.exists():
            progress('robot_fk_samples',completed=0,total=robot_samples)
            q=sample_poses(output_domain,robot_samples,np.random)
            tips=np.empty((robot_samples,5,3),np.float32)
            with torch.inference_mode():
                for i in range(0,robot_samples,4096):
                    check();tips[i:i+4096]=exact_fk(q[i:i+4096]).cpu().numpy()
                    progress('robot_fk_samples',completed=min(i+4096,robot_samples),total=robot_samples)
            depth,valid=query_penetration(robot,q,'cuda:0',progress,check)
            arrays,summary=retained_arrays(q,tips,output_domain,depth,valid,geometry_spec(robot.key,output_domain))
            tmp=labels.with_suffix('.tmp.npz');np.savez_compressed(tmp,**arrays);tmp.replace(labels)
            atomic_json(cache/f'screening_{robot_samples}.json',summary)
        data=np.load(labels);q=data['qpos'];tips=data['keypoints'];robot_points=tips.transpose(1,0,2)
        if certified_depths(labels,robot.key,output_domain) is None:raise ValueError('FK 缓存缺少写盘前姿态筛选，请重新采样')
        config.update(retained_robot_samples=len(q),fk_training_samples=len(q))
        atomic_json(folder/'config.json',config)
        from .workspace import domain_poses
        if not domain_poses(q,output_domain).all():raise ValueError('FK 缓存包含超出机器人无反伸域的姿态，拒绝训练')
        models=reference('model');fk=models.FKModel(robot.spec['keypoint_joints']).cuda()
        fk_path=cache/f'fk_{robot_samples}_{fk_epochs}.pth'
        if fk_path.exists():fk.load_state_dict(torch.load(fk_path,weights_only=True))
        else:
            ds=torch.utils.data.TensorDataset(torch.from_numpy(robot.normalize(q)),torch.from_numpy(tips))
            loader=DataLoader(ds,batch_size=256,shuffle=True)
            optimizer=torch.optim.Adam(fk.parameters(),lr=5e-4)
            for epoch in range(fk_epochs):
                loss_sum=0.
                for angle,tip in loader:
                    check();pred=fk(angle.cuda().float());loss=torch.nn.functional.mse_loss(pred,tip.cuda().float())
                    optimizer.zero_grad();loss.backward();optimizer.step();loss_sum+=loss.item()
                progress('neural_fk',epoch=epoch+1,epochs=fk_epochs,loss=loss_sum/len(loader))
            tmp=fk_path.with_suffix('.tmp');torch.save(fk.state_dict(),tmp);tmp.replace(fk_path)
        fk.eval()
        for parameter in fk.parameters():parameter.requires_grad_(False)
        torch.save(fk.state_dict(),folder/'fk.pth')
        with torch.no_grad():
            check_q=sample_poses(output_domain,1000,np.random)
            pred=fk(torch.from_numpy(robot.normalize(check_q)).cuda()).cpu().numpy()
            exact=exact_fk(check_q).cpu().numpy()
            fk_error=np.linalg.norm(pred-exact,axis=-1)*1000
        atomic_json(folder/'fk_evaluation.json',dict(samples=1000,mean_mm=float(fk_error.mean()),max_mm=float(fk_error.max())))
        if mode=='contact_v1':
            from .contact_sampling import robot_targets
            robot_points,target_manifest=robot_targets(root,robot,config['output_domain'],labels,'cuda:0',progress,check)
            config.update(robot_contact_targets=target_manifest,
                chamfer_target_policy='saved_fk_within_ik_domain_and_penetration_le_5mm',
                chamfer_implementation='exact_nearest_indices_weighted_duplicate_targets_gather_backward',
                contact_execution='batched_source_queries_sparse_penetration_witnesses_v2')
            robot_points=torch.as_tensor(robot_points,device='cuda:0',dtype=torch.float32)
            atomic_json(folder/'config.json',config)
        # Official behavior: independently resample each fingertip cloud. Reject
        # degenerate clouds explicitly instead of upstream randint's cryptic error.
        progress('human_resampling')
        if mode=='official':dataset=human_dataset(points,resample_to=resample_to)
        else:
            from .constraints import synchronized_targets,ContactGeometry,contact_intent,sample_contact_frames
            paired,joints,contacts,bends,identity,_,target_records=synchronized_targets(root,dataset_ids,progress,check)
            if identity!=source or not np.array_equal(points,paired):raise ValueError('训练源在预处理期间变化')
            config['surface_targets']=target_records;atomic_json(folder/'config.json',config)
            _,unique=np.unique(np.floor(points/.001).reshape(len(points),-1),axis=0,return_index=True)
            indices=unique[np.random.randint(0,len(unique),resample_to)]
            dataset=torch.utils.data.TensorDataset(torch.from_numpy(points[indices]))
            geometry=ContactGeometry(robot)
            target_points=torch.tensor(points,device='cuda:0');target_joints=torch.tensor(joints,device='cuda:0')
            contacts=contact_intent(points,contacts)
            target_contacts=torch.tensor(contacts,device='cuda:0')
            target_bends=torch.tensor(bends,device='cuda:0')
            flat_pool=np.flatnonzero(np.abs(bends[:,1:]).max((1,2))<np.deg2rad(10.))
            if not len(flat_pool):flat_pool=np.argsort(np.abs(bends[:,1:]).mean((1,2)))[:min(100,len(points))]
            all_pool=np.arange(len(points))
            config['geometry_sampling']=dict(contact=32,flat=16,uniform=16,
                contact_pool_frames=int(contacts.any(1).sum()),flat_pool_frames=len(flat_pool),
                contact_frames_per_pair=contacts.sum(0).tolist(),
                contact_policy='each_available_pair_then_thumb_priority_2x')
            atomic_json(folder/'config.json',config)
        loader=DataLoader(dataset,batch_size=batch_size,shuffle=True)
        from .constrained_network import create_ik
        ik=create_ik(robot,config).cuda()
        optimizer=torch.optim.AdamW(ik.parameters(),lr=1e-4)
        for epoch in range(epochs):
            sums={};count=0;epoch_started=time.perf_counter()
            for batch in loader:
                if mode!='official':batch=batch[0]
                check();loss,parts=training_losses(batch.cuda().float(),ik,fk,robot_points,efficient=mode=='contact_v1')
                if mode!='official':
                    ids=np.concatenate([sample_contact_frames(contacts,32),np.random.choice(flat_pool,16),np.random.choice(all_pool,16)])
                    ids=torch.tensor(ids,device='cuda:0')
                    # Biased contact/flat batches must not change population BN
                    # statistics. Optimize the same poses used during inference.
                    ik.eval();geometry_norm=ik(target_points[ids]);ik.train()
                    extra,components=geometry.losses(geometry_norm,target_joints[ids],target_contacts[ids],target_bends[ids])
                    loss=loss+extra;parts.update(components)
                if not torch.isfinite(loss):raise ValueError('训练损失出现 NaN/Inf')
                optimizer.zero_grad();loss.backward();optimizer.step();count+=1
                metrics=dict(total=loss,**parts)
                values=torch.stack([v.detach() for v in metrics.values()]).cpu().tolist()
                for name,value in zip(metrics,values):sums[name]=sums.get(name,0.)+value
            tmp=folder/'checkpoint.tmp';torch.save(ik.state_dict(),tmp);tmp.replace(folder/'last.pth')
            progress('retargeting',epoch=epoch+1,epochs=epochs,epoch_s=time.perf_counter()-epoch_started,losses={k:v/count for k,v in sums.items()})
        atomic_json(folder/'status.json',dict(run_id=run_id,status='completed',phase='completed',epochs=epochs,
                    losses={k:v/count for k,v in sums.items()},elapsed_s=time.time()-started,device=device_info,training_mode=mode,
                    constraint_weights=config.get('constraint_weights')))
        from retarget.baseline.geort.model_store import publish
        published=publish(root,folder,run_id)
        # Keep a small job result, rather than another model/checkpoint tree.
        folder.mkdir(parents=True,exist_ok=True)
        result=json.loads((published/'status.json').read_text());result['published_model']='current'
        atomic_json(folder/'status.json',result)
        return published
    except BaseException as exc:
        # Failed/cancelled runs keep diagnostics, not historical weights.
        for checkpoint in folder.glob('*.pth'):checkpoint.unlink(missing_ok=True)
        atomic_json(folder/'status.json',dict(run_id=run_id,status='cancelled' if isinstance(exc,InterruptedError) else 'failed',error=str(exc),elapsed_s=time.time()-started))
        raise
    finally:
        for s,handler in old.items():signal.signal(s,handler)
        lock.close()


class Predictor:
    def __init__(self, folder, device='cpu'):
        from retarget.baseline.geort.model_store import model_lock
        with model_lock(Path(folder).parent.parent):self._load(folder,device)

    def _load(self, folder, device):
        import torch
        self.torch=torch;self.device=device;self.folder=Path(folder)
        self.config=json.loads((self.folder/'config.json').read_text())
        if self.config.get('revision')!=REVISION:raise ValueError('旧GeoRT掌部坐标有偏差，请用原录制重新训练v2模型')
        status=json.loads((self.folder/'status.json').read_text())
        if status.get('status')!='completed':raise ValueError('只能启用完整训练成功的模型')
        self.robot=RobotAdapter()
        if self.robot.spec!=self.config['robot']:raise ValueError('Wuji2模型或关节范围不匹配')
        from .constrained_network import create_ik
        self.model=create_ik(self.robot,self.config).to(device)
        self.model.load_state_dict(torch.load(self.folder/'last.pth',map_location=device,weights_only=True));self.model.eval()

    def predict(self, points):
        torch=self.torch;p=np.asarray(points,np.float32)
        if p.shape!=(5,3) or not np.isfinite(p).all():raise ValueError('需要五指有限位置')
        with torch.inference_mode():
            q=self.model(torch.from_numpy(p)[None].to(self.device)).cpu().numpy()[0]
        return self.robot.unnormalize(q)


def evaluate(root, model_id, dataset_ids):
    from scipy.spatial import cKDTree
    root=Path(root);model=Predictor(root/'models'/model_id)
    p,source,records=load_recordings(root,dataset_ids)
    if source!=model.config['source']:raise ValueError('评估录制与训练源不兼容')
    if set(dataset_ids)&{d['id'] for d in model.config['datasets']}:raise ValueError('请使用未参与训练的独立录制')
    q=[];tips=[];durations=[];direction=[];pinch=[]
    rng=np.random.default_rng(714)
    for point in p:
        t=time.perf_counter();angle=model.predict(point);durations.append((time.perf_counter()-t)*1000)
        embedded=model.robot.fk(angle);q.append(angle);tips.append(embedded)
        delta=rng.normal(size=point.shape);delta=delta/np.linalg.norm(delta,axis=-1,keepdims=True)*.005
        moved=model.robot.fk(model.predict(point+delta))-embedded
        direction.extend(((moved*delta).sum(-1)/np.maximum(np.linalg.norm(moved,axis=-1)*np.linalg.norm(delta,axis=-1),1e-12)).tolist())
        for i in range(5):
            for j in range(i+1,5):
                if np.linalg.norm(point[i]-point[j])<.015:pinch.append(float(np.linalg.norm(embedded[i]-embedded[j])*1000))
    q=np.array(q);tips=np.array(tips)
    from .adapter import fk_cache_key
    data=np.load(root/'fk'/fk_cache_key(model.config)/f'kinematics_{model.config["robot_samples"]}.npz')['keypoints']
    coverage=[float(cKDTree(tips[:,i]).query(data[:,i])[0].mean()*1000) for i in range(5)]
    result=dict(model=model_id,datasets=records,paired_robot_ground_truth=False,frames=len(p),
                workspace_robot_to_output_mean_mm=coverage,direction_cosine_mean=float(np.mean(direction)),
                pinch_events=len(pinch),pinch_distance_mean_mm=float(np.mean(pinch)) if pinch else None,
                joint_limit_violations=int(((q<model.robot.lower-1e-6)|(q>model.robot.upper+1e-6)).sum()),
                inference_ms=dict(mean=float(np.mean(durations)),p95=float(np.percentile(durations,95))),device='cpu')
    atomic_json(root/'models'/model_id/'evaluation.json',result)
    return result
