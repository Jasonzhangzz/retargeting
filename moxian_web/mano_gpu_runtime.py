"""Batched GPU pose refinement, with the same causal projection at runtime.

No action labels are available to the solver. Frames are numerically independent
until a causal rate projection; final contacts/collisions are rechecked after it.
"""
import time
import numpy as np
import torch
from .mano_gpu_objective import GPUManoObjective
from .mano_objective import SolverSettings
from .contact import FINGERS,PATCH_IDS
from scripts.calibrate_wuji_mano import CHAINS,TIPS

BACKEND='gpu_mano_v1'

def _pose_loss(objective,x):return objective.residual(x).square().sum()*len(objective.rows)

_compiled_pose_loss=None
_warmed_shapes=set()

def pose_loss(objective,x):
    global _compiled_pose_loss
    if _compiled_pose_loss is None:_compiled_pose_loss=torch.compile(_pose_loss,options={'triton.cudagraphs':False},dynamic=False)
    return _compiled_pose_loss(objective,x)


def project_sequence(angles,seeds,low,high,rows,previous=None,rate_deg_s=90.):
    outputs=[];old=previous
    for i,row in enumerate(rows):
        stamp=float(row['timestamp_ms']);q=angles[i].clamp(low,high)
        if old is not None:
            correction,then=old;dt=(stamp-then)/1000
            if 0<dt<.25:
                step=rate_deg_s*np.pi/180*dt
                q=seeds[i]+(q-seeds[i]).clamp(correction-step,correction+step)
                q=q.clamp(low,high)
        outputs.append(q);old=(q-seeds[i],stamp)
    return torch.stack(outputs),old


def solve_batch(model,rows,poses,roots,*,tactile=True,previous=None,iterations=4,cancel=None):
    started=time.monotonic()
    # Remove labels before even constructing an objective; only measured
    # bilateral touch plus the seed's proximity can associate self contacts.
    inputs=[{**r,'stage':'runtime','guided_contact':False,'contact_confirmed':False} for r in rows]
    if not tactile:inputs=[{**r,'contact':{},'tactile_timestamp_ms':None} for r in inputs]
    from .mano_gpu_collision import BatchedSurfaceCollision
    cache=getattr(model,'_gpu_collision_cache',{})
    key=(model.shape_version,len(rows))
    if key not in cache:cache[key]=BatchedSurfaceCollision(model,len(rows));model._gpu_collision_cache=cache
    o=GPUManoObjective(model,inputs,np.asarray(poses),np.asarray(roots),training=False,shared=False,tactile=tactile,gpu_collision=cache[key])
    # Non-shared objectives use a feature-only shared view for delta inference.
    initial=o.initial.detach().clone()
    policy=getattr(model,'delta_policy',None)
    artifact=getattr(model,'delta_artifact',None)
    if artifact is not None and (artifact.get('binding_version')!=model.binding_version or artifact.get('mapping_version')!=(model.mapping or {}).get('version') or any(r.get('calibration_fingerprint')!=artifact.get('calibration_fingerprint') for r in rows)):policy=None
    if policy is not None:
        from .mano_calibrate_delta import features,corrected_angles
        with torch.no_grad():
            obs=features(o,initial);q=corrected_angles(o,policy,obs)
            initial=q.flatten().index_select(0,o.free_indices)
    x=torch.nn.Parameter(initial);opt=torch.optim.Adam([x],lr=.01)
    compile_key=(str(model.device),len(rows),hasattr(model,'glove_parameters'))
    compile_first=compile_key not in _warmed_shapes
    for step in range(iterations):
        if cancel is not None and cancel.is_set():raise InterruptedError('GPU 姿态回放已取消')
        if step and step%4==0:o.refresh(x.detach())
        opt.zero_grad(set_to_none=True);before=time.monotonic();loss=pose_loss(o,x)
        loss.backward()
        if cancel is not None and cancel.is_set():raise InterruptedError('GPU 姿态回放已取消')
        if step==0 and compile_first:
            torch.cuda.synchronize();model._gpu_runtime_compile_s=getattr(model,'_gpu_runtime_compile_s',0.)+time.monotonic()-before
            _warmed_shapes.add(compile_key)
        if not bool(torch.isfinite(loss)) or x.grad is None or not bool(torch.isfinite(x.grad).all()):raise ValueError('GPU 姿态求解非有限')
        opt.step()
        with torch.no_grad():x.clamp_(o.lower,o.upper)
    with torch.no_grad():
        raw=o.unpack(x.detach())[-1]
        angles,state=project_sequence(raw,o.seed_angles,o.low,o.high,rows,previous,rate_deg_s=o.settings.correction_rate_deg_s)
        final=angles[o.free];o.refresh(final)
        q,v,j,r,_=o.surface(final)
        predicted=j[:,[c[-1] for c in CHAINS]]+(r[:,[c[-1] for c in CHAINS]]@o.mp0[None,:,:,None]).squeeze(-1)
        target=o.rawp@o.align0.T+o.trans0
        errors=(predicted-target).norm(dim=-1)*1000
        maximum=o.gpu_witness['maximum']*1000;query_valid=bool(o.gpu_witness['valid'] & ~o.gpu_witness['witness_overflow'].any())
        pairs=[[] for _ in rows]
        if len(o.contact_rows):
            points,normals=o.material_contacts(v)
            gaps=(points[:,0]-points[:,1]).norm(dim=-1)*1000;dots=(normals[:,0]*normals[:,1]).sum(-1)
            for record,point,gap,dot in zip(o.contact_records,points.cpu().tolist(),gaps.cpu().tolist(),dots.cpu().tolist()):
                a,b=record['patches'];pairs[record['frame']].append(dict(patch_a=PATCH_IDS[a],patch_b=PATCH_IDS[b],patch_indices=[a,b],point_a_m=point[0],point_b_m=point[1],gap_mm=gap,normal_dot=dot,satisfied=gap<=2 and dot<-.1,evidence=record['evidence'],applied=True))
        cpu_errors=errors.cpu().tolist();depths=maximum.cpu().tolist();elapsed=(time.monotonic()-started)*1000
        reports=[]
        for i,row in enumerate(rows):
            source={f:dict(position_error_mm=cpu_errors[i][k],satisfied=cpu_errors[i][k]<=15.) for k,f in enumerate(FINGERS)}
            collision=dict(max_penetration_m=depths[i]/1000,query_valid=query_valid,constraints_satisfied=query_valid and depths[i]<=2.)
            contact_ok=all(p['satisfied'] for p in pairs[i]);source_ok=all(p['satisfied'] for p in source.values())
            reports.append(dict(revision=BACKEND,status='ok' if collision['constraints_satisfied'] and contact_ok and source_ok else 'constraint_violation',constraints_satisfied=collision['constraints_satisfied'] and contact_ok,contact_constraints_satisfied=contact_ok,source_tracking_satisfied=source_ok,source_tracking_errors=source,collision=collision,elapsed_ms=elapsed,amortized_ms=elapsed/len(rows),batch_size=len(rows),gesture_labels_used=False,contact_pairs=pairs[i]))
        decoded=(v.detach(),j.detach(),r.detach(),torch.cat((j,v[:,list(TIPS)]),1)[:,model.ordered].detach())
    return q.detach(),torch.as_tensor(roots,device=model.device,dtype=torch.float32),decoded,reports,state


def solve_pose(model,frame,pose,root,*,tactile=True):
    version=(model.shape_version,model.binding_version,frame.get('calibration_fingerprint'),frame.get('skeleton_runtime_revision'))
    previous=getattr(model,'_gpu_previous',None) if getattr(model,'_gpu_previous_version',None)==version else None
    model._gpu_previous_version=version
    q,_,decoded,reports,state=solve_batch(model,[frame],np.asarray(pose)[None],np.asarray(root)[None],tactile=tactile,previous=previous)
    model._gpu_previous=state;model._solver_decoded=tuple(t[0].cpu().numpy() for t in decoded)
    report=reports[0];report['revision']=__import__('moxian_web.mano_objective',fromlist=['SOLVER_REVISION']).SOLVER_REVISION
    return q[0].cpu().numpy(),report,dict(applied=bool(report['contact_pairs']),pairs=report['contact_pairs'],policy=BACKEND)


def replay(model,selected,sequence,mounting_calibrated,progress=None,cancel=None):
    model.mounting_calibrated=mounting_calibrated
    values=[model.official_fk.parameters(model,row,CHAINS) for row in sequence]
    poses=np.asarray([v[0] for v in values]);roots=np.asarray([v[1] for v in values])
    if not model.solver_enabled:
        q=torch.as_tensor(poses,device=model.device);r=torch.as_tensor(roots,device=model.device)
        with torch.no_grad():v,j,rr,ordered=model.decode(q,r)
        reports=None
    else:
        qs=[];rs=[];js=[];rot=[];reports=[];previous=None
        for start in range(0,len(sequence),32):
            chunk=sequence[start:start+32]
            q,r,decoded,out,previous=solve_batch(model,chunk,poses[start:start+32],roots[start:start+32],tactile=model.contact_pose_enabled,previous=previous,cancel=cancel)
            qs.append(q);rs.append(r);js.append(decoded[1]);rot.append(decoded[2]);reports.extend(out)
            if progress:progress(start+len(chunk),len(sequence))
        q=torch.cat(qs);r=torch.cat(rs);j=torch.cat(js);rr=torch.cat(rot)
    t=lambda a:torch.as_tensor(a,device=model.device,dtype=torch.float32)
    with torch.no_grad():
        pred=j[:,[c[-1] for c in CHAINS]]+(rr[:,[c[-1] for c in CHAINS]]@t(model.sensor_offsets())[None,:,:,None]).squeeze(-1)
        observed=t([[row['fingers'][f]['position'] for f in FINGERS] for row in sequence])@t(model.raw_rotation).T+t(model.raw_translation)
        error=(pred-observed).square().sum(-1).mean(-1).sqrt().cpu().tolist()
    from .mano_fit import bilateral_contact
    trajectory=[dict(stage=row['stage'],timestamp_ms=row['timestamp_ms'],rmse_mm=e*1000,bilateral_contact=bilateral_contact(row),guided_contact=row.get('guided_contact',False),is_approach=row.get('motion_phase') in ('apart','approach','release')) for row,e in zip(sequence,error)]
    wanted={row['timestamp_ms'] for row in selected};ids=[i for i,row in enumerate(sequence) if row['timestamp_ms'] in wanted]
    model._gpu_replay_reports=reports
    return q[ids],r[ids],trajectory
