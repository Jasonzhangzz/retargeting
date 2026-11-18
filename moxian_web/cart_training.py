"""True-lookahead CART: exact FK + online SDF learning + truncated BPTT."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import time
import numpy as np
import torch
from .cart_data import load_sessions,sequence_indices,preprocess,manifest,preprocessing_current,implementation_identity as geometry_identity,file_hash
from .cart_network import CartNetwork,LEVELS
from .cart_observations import batch_future,pack_future
from .cart_collision import CollisionModel,SDFLearner,frozen
from .cart_objective import CartHand,CartObjective,policy_settings
from .cart_protocol import PROTOCOL,ARCHITECTURES
from .cart_fast import DEFAULT_BATCH_SIZE,PreparedBatch,ObservationStore,prepare_windows,group_normalizers,choose_micro_batch
from .cart_contact_field import CONTACT_FIELD_VERSION
from .cart_initialization import prepare_ik,attach_ik,perturb_initial_states,IK_VERSION,DEFAULT_INITIAL_NOISE_RAD
from retarget.baseline.geort.training import atomic_json


def neutral(hand):
    return hand.tensor(hand.robot.model.qpos0[6:]).clamp(hand.low,hand.high)


def implementation_identity():
    return dict(geometry_identity(),batch_execution=file_hash(Path(__file__).with_name('cart_fast.py')),
        contact_field=file_hash(Path(__file__).with_name('cart_contact_field.py')),
        kinematics=file_hash(Path(__file__).with_name('cart_kinematics.py')),
        initialization=file_hash(Path(__file__).with_name('cart_initialization.py')))


def rollout(model,objective,batch,*,audit=False,collect_only=False,normalizers=None):
    """No teacher labels or detach inside the loss-bearing rollout."""
    length=min(w['length'] for w in batch);device=objective.hand.device
    offsets=[len(w['rows'])-w['length']-10 for w in batch]
    prepared=isinstance(batch,PreparedBatch)
    def inputs_at(indices):
        values=batch.future_inputs(indices) if prepared else batch_future(batch,indices,device)
        if prepared:values['validate']=False
        return values
    initialized=['initial_q' in w for w in batch]
    if any(initialized) and not all(initialized):raise ValueError('训练窗口不能混合 IK 与中性姿态初始化')
    if all(initialized):
        q=torch.stack([w['initial_q'].to(device) if isinstance(w['initial_q'],torch.Tensor) else objective.hand.tensor(w['initial_q']) for w in batch]).detach().clone()
    elif prepared:
        q=neutral(objective.hand)[None].expand(len(batch),-1).clone()
        with torch.no_grad():
            for j in range(max(offsets)):
                candidate=model.unnormalize(model(q,**inputs_at([min(j,start) for start in offsets])))
                active=torch.as_tensor([j<start for start in offsets],device=device)[:,None]
                q=torch.where(active,candidate,q)
    else:
        initial=[]
        for w,start in zip(batch,offsets):
            one=neutral(objective.hand)[None]
            with torch.no_grad():
                for j in range(start):one=model.unnormalize(model(one,**batch_future([w],[j],device)))
            initial.append(one[0])
        q=torch.stack(initial)
    older_q=None;old_dt=None
    totals=[];sums={};qualities=[];predictions=[]
    collision=objective.collision;old_audit=collision.auditing if collision is not None else False
    if collision is not None:collision.auditing=audit
    try:
        for t in range(length):
            indices=[i+t for i in offsets]
            rows=[w['rows'][i+1] for w,i in zip(batch,indices)]
            previous=[w['rows'][i] for w,i in zip(batch,indices)]
            inputs=inputs_at(indices)
            next_q=model.unnormalize(model(q,**inputs))
            if collect_only:
                if collision is None:raise ValueError('Collect rollout requires collision model')
                collision.collect_queries(next_q,rows,objective.hand.frames(next_q))
                predictions.append(next_q);q=next_q;continue
            dt=inputs['future_dt'][:,0]
            velocity=(next_q-q)/dt[:,None]
            acceleration=None
            if older_q is not None:
                old_velocity=(q-older_q)/old_dt[:,None]
                acceleration=(velocity-old_velocity)/(.5*(dt+old_dt))[:,None]
            weights=batch.normalizers if prepared and batch.normalizers is not None else normalizers
            options=dict(normalizers=weights[t]) if weights is not None else {}
            loss,parts,metrics=objective.batch(next_q,rows,q,acceleration=acceleration,previous_rows=previous,audit=audit,**options)
            totals.append(loss);predictions.append(next_q)
            # Metrics never retain the training graph; q feedback below does.
            for quality in metrics:
                qualities.append({k:v.detach() if isinstance(v,torch.Tensor) else v for k,v in quality.items() if k not in ('groups','points')})
            for k,v in parts.items():sums.setdefault(k,[]).append(v)
            older_q,q,old_dt=q,next_q,dt
        return (torch.stack(totals).mean() if totals else q.new_zeros(())),{k:torch.stack(v).mean() for k,v in sums.items()},qualities,predictions
    finally:
        if collision is not None:collision.auditing=old_audit


def windows(sessions,length=16):
    result=[]
    for meta,rows,_ in sessions:
        for indices in sequence_indices(rows,length):
            current=rows[indices[len(indices)-length-10]]
            if current.get('ik_valid') is False:continue
            window=dict(rows=[rows[i] for i in indices],indices=indices,length=length,dataset=meta['id'])
            if 'ik_q_rad' in current:window['initial_q']=current['ik_q_rad']
            result.append(window)
    return result


def policy_update(model,objective,learner,optimizer,batch,micro_batch_size,check=lambda:None,progress=None,initial_noise_rad=0.):
    """One logical batch, two SDF steps, one policy step; no horizon detach."""
    device=objective.hand.device;batch=perturb_initial_states(batch,objective.hand,initial_noise_rad)
    total_size=len(batch);normalizers=group_normalizers(batch)
    def chunks():
        for start in range(0,total_size,micro_batch_size):
            windows=batch[start:start+micro_batch_size]
            yield PreparedBatch(windows,device,normalizers) if all('store' in w for w in windows) else windows
    started=time.perf_counter();times={}
    with torch.no_grad():
        for index,chunk in enumerate(chunks()):
            check();rollout(model,objective,chunk,collect_only=True)
            if progress:progress('collect',index+1,math_ceil(total_size,micro_batch_size))
    times['collect_s']=time.perf_counter()-started;started=time.perf_counter()
    learner.collect()
    for _ in range(2):sdf_losses=learner.update()
    times['sdf_s']=time.perf_counter()-started;started=time.perf_counter()
    optimizer.zero_grad(set_to_none=True);summed={};total=None
    with frozen(objective.collision.model):
        for index,chunk in enumerate(chunks()):
            check();weight=len(chunk)/total_size
            loss,parts,_,_=rollout(model,objective,chunk,normalizers=normalizers)
            if not torch.isfinite(loss):raise ValueError('CART loss 出现 NaN/Inf')
            (weight*loss).backward()
            total=weight*loss.detach() if total is None else total+weight*loss.detach()
            for k,v in parts.items():summed[k]=summed.get(k,0.)+weight*v.detach()
            if progress:progress('backward',index+1,math_ceil(total_size,micro_batch_size))
        torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True);optimizer.step()
    times['policy_s']=time.perf_counter()-started
    return total,summed,sdf_losses,times


def position_statistics(sessions):
    points=np.concatenate([np.asarray(r['emf_pose'],np.float32)[np.asarray(r['emf_valid'],bool),:3]
        for _,rows,_ in sessions for r in rows])
    if not len(points):raise ValueError('训练会话没有有效 EMF 观测')
    return points.mean(0),np.maximum(points.std(0),.001)


def save_checkpoint(path,values):
    temp=Path(str(path)+'.tmp');torch.save(values,temp);temp.replace(path)


def quality_summary(metrics):
    def scalar(value):return float(value.detach()) if torch.is_tensor(value) else float(value)
    def values(key):return np.asarray([scalar(m[key][0]) for m in metrics])
    resolved=[m for m in metrics if bool(m['query_valid'][0])]
    def measured(key):return np.asarray([scalar(m[key][0]) for m in resolved])
    active=[m for m in metrics if m['contact_count']]
    keep=[m for m in metrics if m.get('contact_keep_count',0)]
    return dict(frames=len(metrics),contact_frames=len(active),
        invalid_query_frames=len(metrics)-len(resolved),
        unresolved_contact_frames=sum(int(m.get('unresolved_contact_count',[0])[0])>0 for m in metrics),
        contact_satisfied_rate=float(np.mean([bool(m['feasible'][0]) for m in active])) if active else None,
        feasible_rate=float(np.mean([bool(m['feasible'][0]) for m in metrics])) if metrics else None,
        bone_direction_mean_deg=float(values('bone_direction_deg').mean()) if metrics else None,
        gap_p95_mm=float(np.percentile(measured('max_gap_m')*1000,95)) if resolved else None,
        gap_max_mm=float(measured('max_gap_m').max()*1000) if resolved else None,
        normal_max_deg=float(measured('max_normal_deg').max()) if resolved else None,
        penetration_max_mm=float(values('max_penetration_m').max()*1000) if metrics else None,
        contact_keep_reward=float(np.mean([1-scalar(m['contact_keep_loss'][0]) for m in keep])) if keep else None)


def motion_summary(samples):
    previous=None;old_velocity=None;old_dt=None;velocities=[];accelerations=[]
    for stamp,q,segment in samples:
        if q is None:previous=None;old_velocity=None;old_dt=None;continue
        q=np.asarray(q)
        if previous is not None and segment==previous[2] and 0<stamp-previous[0]<=250:
            dt=(stamp-previous[0])/1000;velocity=(q-previous[1])/dt;velocities.extend(velocity.tolist())
            if old_velocity is not None:accelerations.extend(((velocity-old_velocity)/(.5*(dt+old_dt))).tolist())
            old_velocity=velocity;old_dt=dt
        else:old_velocity=None;old_dt=None
        previous=(stamp,q,segment)
    return dict(joint_velocity_rms_rad_s=float(np.sqrt(np.mean(np.square(velocities)))) if velocities else None,
        joint_acceleration_rms_rad_s2=float(np.sqrt(np.mean(np.square(accelerations)))) if accelerations else None)


def acceptance(quality,sdf):
    reasons=[]
    if not quality['frames']:reasons.append('没有完整的未来帧验证窗口')
    if not quality['contact_frames']:reasons.append('独立验证缺少活动接触')
    if quality['feasible_rate']!=1.:reasons.append('真实 mesh 接触／法向／0.5 mm 穿透阈值未全部通过')
    if sdf['distance_p95_mm']>.5:reasons.append('SDF 距离 p95 超过 0.5 mm')
    if sdf.get('contact_samples',0)<20 or sdf.get('contact_position_p95_mm') is None or sdf['contact_position_p95_mm']>.5:
        reasons.append('连续接触场真实 mesh 验证覆盖不足或位置 p95 超过 0.5 mm')
    if sdf.get('contact_normal_p95_deg') is None or sdf['contact_normal_p95_deg']>30:
        reasons.append('连续接触场法向 p95 超过 30 度')
    if sdf['penetration_cases']<20 or sdf['penetration_false_negative_rate'] is None or sdf['penetration_false_negative_rate']>.01:
        reasons.append('SDF 穿透验证覆盖不足或漏检率超过 1%')
    return reasons


def full_session(model,objective,sessions,check=lambda:None):
    """Independent complete-session closed loop; only a short final tail is omitted."""
    qualities=[];samples=[];predictions=[];times=[]
    collision=objective.collision;old=collision.auditing;collision.auditing=True
    try:
        with torch.no_grad():
            for meta,rows,_ in sessions:
                store=ObservationStore(rows,objective.hand.device)
                q=neutral(objective.hand)[None];segment=0
                for i in range(max(0,len(rows)-10)):
                    check()
                    delta=np.diff(store.cpu_stamps[i:i+11])
                    if np.any(delta<=0) or np.any(delta>250) or any(row.get('stream_epoch')!=rows[i].get('stream_epoch') for row in rows[i:i+11]):
                        q=neutral(objective.hand)[None];segment+=1;continue
                    window=dict(store=store,indices=list(range(i,i+11)))
                    inputs=PreparedBatch([window],objective.hand.device).future_inputs([0]);inputs['validate']=False
                    start=time.perf_counter();next_q=model.unnormalize(model(q,**inputs))
                    times.append((time.perf_counter()-start)*1000)
                    row=rows[i+1]
                    _,_,quality=objective.batch(next_q,[row],q,previous_rows=[rows[i]],audit=True)
                    qualities.extend(quality);q=next_q
                    values=q[0].cpu().numpy();predictions.append(dict(dataset=meta['id'],timestamp_ms=row['timestamp_ms'],q=values.tolist()))
                    samples.append((row['timestamp_ms'],values,(meta['id'],segment)))
    finally:collision.auditing=old
    return dict(quality_summary(qualities),**motion_summary(samples)),predictions,times


def train(root,train_ids,validation_ids,run_id,epochs=200,seed=714,batch_size=DEFAULT_BATCH_SIZE,length=16,
          smoke=False,architecture='attention',resume=False,warmup_steps=None,device=None,micro_batch_size=0,initialize_from=None,
          initial_noise_rad=DEFAULT_INITIAL_NOISE_RAD):
    if architecture not in ARCHITECTURES:raise ValueError('architecture must be attention or fsq')
    from .geort_service import safe_id
    run_id=safe_id(run_id)
    if initialize_from:initialize_from=safe_id(initialize_from)
    if resume and initialize_from:raise ValueError('精确恢复与初始化来源不能同时指定')
    root=Path(root);folder=root/'models'/run_id
    folder.mkdir(parents=True,exist_ok=resume)
    lockpath=root.parent/'geort/training.lock';lockpath.parent.mkdir(parents=True,exist_ok=True)
    cancelled=False
    def stop(*_):
        nonlocal cancelled
        cancelled=True
    def check():
        if cancelled:raise InterruptedError('CART 任务已取消')
    old={s:signal.signal(s,stop) for s in (signal.SIGTERM,signal.SIGINT)}
    started=time.time();lock=lockpath.open('a');active=not resume
    def progress(phase,**values):
        check();state=dict(status='running',run_id=run_id,pid=os.getpid(),phase=phase,elapsed_s=time.time()-started,**values)
        atomic_json(folder/'status.json',state)
        with (folder/'trace.jsonl').open('a') as stream:stream.write(json.dumps(state,allow_nan=False)+'\n')
        print(json.dumps(state,ensure_ascii=False),flush=True)
    try:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('已有 GeoRT／CART 训练任务')
        if not 1<=epochs<=2000 or not 1<=batch_size<=65536 or length<2:raise ValueError('无效训练预算')
        if isinstance(initial_noise_rad,bool) or not np.isfinite(initial_noise_rad) or not 0<=initial_noise_rad<=.5:raise ValueError('IK 初始噪声幅度须为 0–0.5 rad')
        if set(train_ids)&set(validation_ids):raise ValueError('训练和验证不能包含同一录制')
        rng=np.random.default_rng(seed);torch.manual_seed(seed);torch.set_num_threads(1)
        for key in [*train_ids,*validation_ids]:
            check();recording,metadata=manifest(root,key)
            if not smoke and metadata.get('fixture_only'):raise ValueError('Smoke 测试录制不能用于正式模型训练')
            if not preprocessing_current(recording,metadata,full=not smoke):preprocess(root,key,full=not smoke,progress=progress,check=check)
            prepare_ik(root,key,check=check,progress=progress)
        sessions,source=load_sessions(root,train_ids,'train',require_complete=not smoke)
        validation,other=load_sessions(root,validation_ids,'validation',require_complete=not smoke)
        if source!=other:raise ValueError('训练与验证标定／设备不兼容')
        ik_reports=attach_ik(root,sessions+validation)
        bindings={json.loads((root/'recordings'/m['id']/'topology.json').read_text())['binding_version'] for m,_,_ in sessions+validation}
        if len(bindings)!=1:raise ValueError('训练与验证的触觉材料绑定不兼容')
        if {m.get('session_id',m['id']) for m,_,_ in sessions}&{m.get('session_id',m['id']) for m,_,_ in validation}:
            raise ValueError('同一采集会话不能跨训练与验证分区')
        train_windows=windows(sessions,length);validation_windows=windows(validation,length)
        if not train_windows or not validation_windows:raise ValueError('训练与验证各需至少连续 length+10 帧真实观测，不跨断流边界')
        device=device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        hand=CartHand(device);collision=CollisionModel(hand);objective=CartObjective(hand,policy_settings(),collision)
        mean,scale=position_statistics(sessions)
        model=CartNetwork(hand.low,hand.high,architecture,position_mean=mean,position_scale=scale).to(device)
        actual_micro=min(batch_size,choose_micro_batch(architecture,device,micro_batch_size))
        prepare_windows(sessions,train_windows,device);prepare_windows(validation,validation_windows,device)
        objective.cache_rows=True
        optimizer=torch.optim.AdamW(model.parameters(),lr=1.e-4)
        learner=SDFLearner(collision,capacity=4096 if smoke else 200000,batch_size=64 if smoke else 2048)
        warmup_steps=(4 if smoke else 2000) if warmup_steps is None else warmup_steps
        if isinstance(warmup_steps,bool) or not isinstance(warmup_steps,int) or warmup_steps<0:raise ValueError('无效 SDF 热启动步数')
        config=dict(protocol=PROTOCOL,architecture=ARCHITECTURES[architecture],architecture_name=architecture,
            geometry=implementation_identity(),run_id=run_id,source=source,source_binding_version=next(iter(bindings)),
            robot=hand.robot.spec,output_domain=hand.domain,levels=list(LEVELS),position_mean=mean.tolist(),position_scale=scale.tolist(),
            train_ids=train_ids,validation_ids=validation_ids,epochs=epochs,seed=seed,batch_size=batch_size,sequence_length=length,
            lookahead=10,optimizer='AdamW',learning_rate=1.e-4,objective=objective.settings.payload(),
            profile='smoke_test' if smoke else 'full',online_search=False,contact_conditioning=architecture=='attention',
            datasets=[dict(id=m['id'],fingerprint=info['fingerprint']) for m,_,info in sessions+validation],
            training='differentiable_bptt_online_mesh_sdf',sdf_warmup_steps=warmup_steps,sdf_updates_per_batch=2,
            selection='minimum_true_mesh_validation_loss',sdf_capacity=learner.capacity,sdf_batch_size=learner.batch_size)
        config.update(micro_batch_size=micro_batch_size,batch_semantics='windows_per_optimizer_step',
            reference_parallel_batch=4096,execution='resident_observations_batched_geometry_gradient_accumulation',initialize_from=initialize_from,
            contact_field=CONTACT_FIELD_VERSION,contact_selection='learned_directional_field_soft_pair',
            state_initialization=IK_VERSION,initial_noise_rad=initial_noise_rad,ik_datasets=ik_reports,
            full_session_initialization='neutral_closed_loop')
        epoch_start=0;resume_cursor=0;order=None;sums={};batches=0;best=float('inf');best_epoch=None
        def state(epoch,next_batch,order):
            return dict(model=model.state_dict(),collision=collision.model.state_dict(),policy_optimizer=optimizer.state_dict(),
                sdf_learner=learner.state_dict(),epoch=epoch,next_batch=next_batch,order=torch.as_tensor(order) if order is not None else None,
                sums=sums,batches=batches,best=best,best_epoch=best_epoch,holdout=tuple(v.cpu() for v in holdout),
                micro_batch_size=actual_micro,
                numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
        def warmup(start):
            for step in range(start,warmup_steps):
                check()
                if step%4==0:learner.collect()
                losses=learner.update()
                if step%20==0 or step+1==warmup_steps:
                    save_checkpoint(folder/'warmup.pth',dict(state(0,0,None),warmup_next=step+1))
                    progress('sdf_warmup',completed=step+1,total=warmup_steps,sdf_losses=losses)
        if resume:
            saved=json.loads((folder/'config.json').read_text())
            if config!=saved:raise ValueError('恢复训练的配置、数据或实现指纹发生变化')
            checkpoint_path=folder/'resume.pth' if (folder/'resume.pth').exists() else folder/'warmup.pth'
            restored=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
            model.load_state_dict(restored['model']);collision.model.load_state_dict(restored['collision'])
            optimizer.load_state_dict(restored['policy_optimizer']);learner.load_state_dict(restored['sdf_learner'])
            epoch_start=restored['epoch'];resume_cursor=restored['next_batch'];order=restored['order'].numpy() if restored['order'] is not None else None
            actual_micro=restored['micro_batch_size']
            sums=restored['sums'];batches=restored['batches'];best=restored['best'];best_epoch=restored['best_epoch']
            holdout=tuple(v.to(device) for v in restored['holdout'])
            rng.bit_generator.state=restored['numpy_rng'];torch.set_rng_state(restored['torch_rng'])
            if torch.cuda.is_available() and restored['cuda_rng']:torch.cuda.set_rng_state_all(restored['cuda_rng'])
            active=True
            if 'warmup_next' in restored:warmup(restored['warmup_next'])
        else:
            atomic_json(folder/'config.json',config)
            first=root/'recordings'/train_ids[0]
            for name in ('profile.json','topology.json'):shutil.copyfile(first/name,folder/('source_'+name))
            holdout=collision.oracle.random_queries(256 if smoke else 4096)
            first_warmup=0
            if initialize_from:
                source_folder=root/'models'/initialize_from
                origin=json.loads((source_folder/'config.json').read_text())
                required=('protocol','architecture_name','robot','output_domain','levels','position_mean','position_scale',
                    'source','source_binding_version','train_ids','validation_ids','datasets','sdf_capacity','sdf_batch_size')
                if any(config[k]!=origin[k] for k in required):raise ValueError('初始化 checkpoint 的架构、数据、坐标或碰撞模型配置不匹配')
                initial=torch.load(source_folder/('resume.pth' if (source_folder/'resume.pth').exists() else 'warmup.pth'),map_location='cpu',weights_only=True)
                model.load_state_dict(initial['model']);optimizer.load_state_dict(initial['policy_optimizer'])
                if origin.get('contact_field')==CONTACT_FIELD_VERSION:
                    collision.model.load_state_dict(initial['collision']);learner.load_state_dict(initial['sdf_learner'])
                    first_warmup=min(learner.steps,warmup_steps)
                elif 'contact_field' not in origin:
                    # Explicit new-run transfer only. The new contact head and
                    # geometry optimizer/replay need their complete warmup.
                    mismatch=collision.model.load_state_dict(initial['collision'],strict=False)
                    expected={k for k in collision.model.state_dict() if k.startswith('contact_field.')}
                    if set(mismatch.missing_keys)!=expected or mismatch.unexpected_keys:
                        raise ValueError('旧碰撞模型结构不兼容，不能初始化连续接触场')
                else:raise ValueError('初始化来源的连续接触场版本不匹配')
            save_checkpoint(folder/'warmup.pth',dict(state(0,0,None),warmup_next=first_warmup))
            warmup(first_warmup)
        for epoch in range(epoch_start,epochs):
            model.train()
            if order is None:order=rng.permutation(len(train_windows));sums={};batches=0
            for start in range(resume_cursor,len(order),batch_size):
                check();batch=[train_windows[i] for i in order[start:start+batch_size]]
                def chunk_progress(stage,completed,total):
                    progress('batch_compute',epoch=epoch+1,epochs=epochs,batch=batches+1,
                        logical_batch_size=batch_size,actual_batch_size=len(batch),micro_batch_size=actual_micro,
                        compute_stage=stage,chunk_completed=completed,chunk_total=total)
                loss,parts,sdf_losses,timings=policy_update(model,objective,learner,optimizer,batch,actual_micro,check,chunk_progress,initial_noise_rad)
                batches+=1
                for k,v in dict(total=loss,**parts).items():sums[k]=sums.get(k,0.)+float(v.detach())
                save_checkpoint(folder/'resume.pth',state(epoch,start+batch_size,order))
                progress('retargeting',epoch=epoch+1,epochs=epochs,batch=batches,batches=math_ceil(len(order),batch_size),
                    logical_batch_size=batch_size,actual_batch_size=len(batch),micro_batch_size=actual_micro,
                    losses={k:v/batches for k,v in sums.items()},sdf_losses=sdf_losses,sdf_updates=learner.steps,timings=timings)
            model.eval();validation_sum=0.;validation_count=0;quality=[]
            with torch.no_grad():
                for start in range(0,len(validation_windows),batch_size):
                    check();batch=validation_windows[start:start+batch_size]
                    normalizers=group_normalizers(batch)
                    for chunk_start in range(0,len(batch),actual_micro):
                        chunk=PreparedBatch(batch[chunk_start:chunk_start+actual_micro],device,normalizers)
                        value,_,metrics,_=rollout(model,objective,chunk,audit=True)
                        validation_sum+=float(value)*len(chunk);validation_count+=len(chunk);quality.extend(metrics)
            validation_loss=validation_sum/validation_count
            if not np.isfinite(validation_loss):raise ValueError('真实几何验证 loss 非有限数值')
            sdf_report=learner.validate(*holdout)
            if validation_loss<best:
                best=validation_loss;best_epoch=epoch+1
                save_checkpoint(folder/'last.pth',dict(model=model.state_dict(),collision=collision.model.state_dict(),epoch=epoch+1,validation_loss=best,sdf_validation=sdf_report))
            progress('validation',epoch=epoch+1,epochs=epochs,validation_loss=validation_loss,best_validation_loss=best,
                best_epoch=best_epoch,metrics=quality_summary(quality),sdf_validation=sdf_report)
            order=None;resume_cursor=0;sums={};batches=0
            save_checkpoint(folder/'resume.pth',state(epoch+1,0,None))
        checkpoint=torch.load(folder/'last.pth',map_location=device,weights_only=True)
        model.load_state_dict(checkpoint['model']);collision.model.load_state_dict(checkpoint['collision'])
        progress('full_session_validation')
        report,poses,timings=full_session(model,objective,validation,check)
        sdf_report=learner.validate(*holdout);reasons=acceptance(report,sdf_report)
        atomic_json(folder/'evaluation.json',dict(metrics=report,sdf_validation=sdf_report,rejection_reasons=reasons,
            inference_ms=dict(mean=float(np.mean(timings)),p95=float(np.percentile(timings,95)))))
        atomic_json(folder/'validation_motion.json',poses)
        atomic_json(folder/'status.json',dict(status='smoke_test' if smoke else 'rejected' if reasons else 'completed',
            run_id=run_id,epochs=epochs,elapsed_s=time.time()-started,best_epoch=best_epoch,validation_loss=best,
            metrics=report,sdf_validation=sdf_report,rejection_reasons=reasons))
        return folder
    except BaseException as exc:
        atomic_json(folder/('status.json' if active else 'resume_error.json'),dict(status='cancelled' if isinstance(exc,InterruptedError) else 'failed',run_id=run_id,error=str(exc)))
        raise
    finally:
        for s,handler in old.items():signal.signal(s,handler)
        lock.close()


def math_ceil(n,d):return (n+d-1)//d


def validate_model_geometry(config):
    name=config.get('architecture_name')
    if config.get('protocol')!=PROTOCOL or name not in ARCHITECTURES or config.get('architecture')!=ARCHITECTURES[name]:
        raise ValueError('旧 CART 模型与未来帧可微训练架构不兼容，请重新训练')
    if config.get('geometry')!=implementation_identity():raise ValueError('CART 几何／训练实现已变化，请重新训练')


class Predictor:
    """Stateless future-window API. Default output is normalized absolute q."""
    def __init__(self,folder,device='cpu',allow_smoke=False):
        self.folder=Path(folder);self.device=device
        self.config=json.loads((self.folder/'config.json').read_text());status=json.loads((self.folder/'status.json').read_text())
        if status['status'] not in (('completed','smoke_test') if allow_smoke else ('completed',)):
            raise ValueError('只能启用真实几何验收通过的正式 CART 模型')
        if not allow_smoke and self.config['profile']!='full':raise ValueError('测试模型不能启用')
        validate_model_geometry(self.config);self.hand=CartHand(device);self.robot=self.hand.robot
        if self.robot.spec!=self.config['robot'] or self.hand.domain!=self.config['output_domain']:raise ValueError('CART 机器人／关节域不匹配')
        domain=self.config['output_domain']
        self.model=CartNetwork(domain['lower'],domain['upper'],self.config['architecture_name'],self.config['levels'],
            self.config['position_mean'],self.config['position_scale']).to(device)
        self.collision=CollisionModel(self.hand)
        checkpoint=torch.load(self.folder/'last.pth',map_location=device,weights_only=True)
        self.model.load_state_dict(checkpoint['model']);self.model.eval()
        self.collision.model.load_state_dict(checkpoint['collision']);self.collision.model.eval()
        for p in self.collision.model.parameters():p.requires_grad_(False)

    def predict(self,current_q,future_emf,emf_valid,future_dt,tactile_positions=None,tactile_state=None,tactile_valid=None,*,normalized=True):
        q=np.asarray(current_q,np.float32);single=q.shape==(20,)
        values=dict(current_q=q,future_emf=future_emf,emf_valid=emf_valid,future_dt=future_dt,
            tactile_positions=tactile_positions,tactile_state=tactile_state,tactile_valid=tactile_valid)
        args={}
        for k,value in values.items():
            if value is None:continue
            tensor=torch.as_tensor(np.asarray(value),device=self.device,dtype=torch.bool if k.endswith('valid') else torch.float32)
            args[k]=tensor[None] if single else tensor
        with torch.inference_mode():
            result=self.model(**args)
            if not normalized:result=self.model.unnormalize(result)
            result=result.cpu().numpy()
        return result[0] if single else result

    def predict_rows(self,q,current_stamp,rows):
        inputs=pack_future(current_stamp,rows,self.device)
        with torch.inference_mode():
            result=self.model(self.hand.tensor(q)[None],**{k:v[None] for k,v in inputs.items()})
            return self.model.unnormalize(result)[0].cpu().numpy()


def evaluate(root,model_id,dataset_ids,check=lambda:None,allow_smoke=False):
    root=Path(root);predictor=Predictor(root/'models'/model_id,'cpu',allow_smoke)
    for key in dataset_ids:
        folder,meta=manifest(root,key)
        if not preprocessing_current(folder,meta,full=not allow_smoke):preprocess(root,key,full=not allow_smoke,check=check)
    sessions,source=load_sessions(root,dataset_ids,'validation',require_complete=not allow_smoke)
    if source!=predictor.config['source']:raise ValueError('验证数据与模型标定不匹配')
    if set(dataset_ids)&set(predictor.config['train_ids']):raise ValueError('不能使用训练录制作为独立验证')
    train_groups={manifest(root,key)[1].get('session_id',key) for key in predictor.config['train_ids']}
    if train_groups&{m.get('session_id',m['id']) for m,_,_ in sessions}:raise ValueError('验证与训练属于同一采集会话')
    if any(json.loads((root/'recordings'/m['id']/'topology.json').read_text())['binding_version']!=predictor.config['source_binding_version'] for m,_,_ in sessions):
        raise ValueError('验证数据触觉材料绑定与模型不匹配')
    objective=CartObjective(predictor.hand,policy_settings(),predictor.collision)
    metrics,poses,times=full_session(predictor.model,objective,sessions,check)
    learner=SDFLearner(predictor.collision)
    with torch.random.fork_rng():
        torch.manual_seed(714);sdf=learner.validate(*predictor.collision.oracle.random_queries(4096))
    result=dict(protocol=PROTOCOL,model=model_id,datasets=dataset_ids,metrics=metrics,sdf_validation=sdf,
        inference_ms=dict(mean=float(np.mean(times)),p95=float(np.percentile(times,95))) if times else None,
        rejection_reasons=acceptance(metrics,sdf),online_search=False,hard_contact_guarantee=False,
        paired_robot_ground_truth=False,tube_required=False)
    atomic_json(root/'models'/model_id/'evaluation.json',result);atomic_json(root/'models'/model_id/'evaluation_motion.json',poses)
    return result
