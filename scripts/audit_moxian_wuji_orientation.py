#!/usr/bin/env python3
"""Separate objective bias, SLSQP budget and model reachability without changing runtime.

All orientation comparisons remove calibrated RX mounting and use fixed physical
longitudinal/palmar frames. Oracle searches are diagnostic, not deployed solvers.
"""
from pathlib import Path
import argparse,hashlib,json,sys,time
from dataclasses import replace
import numpy as np
import mujoco
import torch
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from moxian_web.moxian_replay import MoxianReplayReconstructor
from moxian_web.wuji_contact_slsqp import ContactSLSQPRetarget,palm_basis
from moxian_web.wuji_contact_cost import ContactObjective
from moxian_web.wuji_surface import basis
from moxian_web.bone_extrinsics import FINGERS


def angle(a,b):
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(a.T@b).as_rotvec())))


def direction_angle(a,b):
    return float(np.degrees(np.arccos(np.clip(np.dot(a,b),-1.,1.))))


def distal_from_rx(raw_rotation,alignment,mount):
    return alignment@raw_rotation@mount.T


def anatomical_target(distal,frame,mirror,wrist):
    parity=np.diag([np.linalg.det(mirror),1.,1.])
    return wrist.T@mirror@distal@frame@parity


def abduction_bound(engine,f,target):
    """Exact necessary orientation bound for non-thumb Rx Ry Rx chains."""
    if f==0:return None
    fk=engine.fk;fk.forward(np.zeros(20))
    base=fk.data.xmat[int(fk.body_ids[1+4*f])].reshape(3,3).copy()
    relative=base.T@target@engine.surface.bases[4+4*f].T
    required=float(np.arccos(np.clip(relative[0,0],-1.,1.)))
    limit=float(max(abs(fk.lower[1+4*f]),abs(fk.upper[1+4*f])))
    return dict(required_middle_angle_deg=float(np.degrees(required)),available_middle_angle_deg=float(np.degrees(limit)),
                proven_orientation_lower_bound_deg=float(np.degrees(max(0.,required-limit))))


def exact_orientation_position_bound(engine,f,target,tip_target):
    """Global tip-position minimum at exact orientation for native Rx Ry Rx chains.

    Orientation fixes MCP flexion/abduction and PIP+DIP (two Euler branches).
    The remaining PIP variable moves the DIP on a known circle arc. Enumerate
    all in-limit branches and minimize distance on each arc analytically.
    Thumb has an intervening static rotation and is deliberately excluded.
    """
    if f==0:return None
    fk=engine.fk;model=fk.model;ids=fk.body_ids[1+4*f:5+4*f];lower=fk.lower[4*f:4*f+4];upper=fk.upper[4*f:4*f+4]
    fk.forward(np.zeros(20));root_r=fk.data.xmat[int(ids[0])].reshape(3,3).copy();root_p=fk.data.xpos[int(ids[0])].copy()
    desired=target@engine.surface.bases[4+4*f].T;relative=root_r.T@desired
    if abs(abs(relative[0,0])-1)<1e-10:return dict(supported=False,reason='Euler singularity')
    for body in ids[1:]:
        if not np.allclose(model.body_quat[body],[1.,0,0,0],atol=1e-12):raise ValueError('Expected native identity intermediate frames')
    a,b,c=Rotation.from_matrix(relative).as_euler('XYX');branches=[(a,b,c),(a+np.pi,-b,c-np.pi)]
    def in_range(value,lo,hi):return [value+2*np.pi*k for k in range(-2,3) if lo-1e-9<=value+2*np.pi*k<=hi+1e-9]
    best=None;count=0
    for a,b,c in branches:
        for q0 in in_range(a,lower[0],upper[0]):
            for q1 in in_range(b,lower[1],upper[1]):
                for total in in_range(c,lower[2]+lower[3],upper[2]+upper[3]):
                    lo=max(lower[2],total-upper[3]);hi=min(upper[2],total-lower[3])
                    if lo>hi+1e-9:continue
                    r1=root_r@Rotation.from_rotvec([q0,0,0]).as_matrix();r2=r1@Rotation.from_rotvec([0,q1,0]).as_matrix()
                    centre=root_p+r1@model.body_pos[ids[1]]+r2@model.body_pos[ids[2]]+desired@fk.sites[f].pos
                    local=r2.T@(tip_target-centre);offset=model.body_pos[ids[3]]
                    phase=np.arctan2(-local[1]*offset[2]+local[2]*offset[1],local[1]*offset[1]+local[2]*offset[2])
                    candidates=[lo,hi,*in_range(phase,lo,hi)]
                    for q2 in candidates:
                        qf=np.array([q0,q1,q2,total-q2]);position=centre+r2@Rotation.from_rotvec([q2,0,0]).as_matrix()@offset
                        error=float(np.linalg.norm(position-tip_target));count+=1
                        if best is None or error<best[0]:best=(error,qf)
    if best is None:return dict(supported=True,orientation_reachable=False)
    return dict(supported=True,orientation_reachable=True,proven_min_tip_error_mm=best[0]*1000,
                minimizing_q=best[1].tolist(),arc_candidates=count)


def source_frames(replay,engine):
    k=replay.ik.kinematics
    rr,pp=k.forward(np.asarray(k.definition['rest_angles_rad']))
    _,normals=engine.source.taxels(engine.source.rest)
    normals=normals@engine.source.mirror
    frames=[];verification=[]
    for f,distal in enumerate(k.rig['distal_bodies']):
        longitudinal=rr[distal].T@(pp[k.rig['tip_bodies'][f]]-pp[distal]);longitudinal/=np.linalg.norm(longitudinal)
        j=k.rig['body_joints'][distal][-1]
        palmar=np.cross(k.joint_axis0[j],longitudinal);palmar/=np.linalg.norm(palmar)
        patch=0 if f==0 else 2+3*(f-1)
        mean=(normals[engine.source.patches==patch]@rr[distal]).mean(0)
        if mean@palmar<0:palmar=-palmar
        frames.append(basis(longitudinal,palmar))
        verification.append(float(mean@palmar))
    return np.asarray(frames),verification


def targets(replay,engine,observation,source,q_source,frames):
    k=replay.ik.kinematics;rs,ps=k.forward(q_source)
    mirror=engine.source.mirror
    wrist=palm_basis(source['joints'])@engine.fk.palm_basis.T
    observed=[];fitted=[];source_errors=[];source_bends=[]
    for f,name in enumerate(FINGERS):
        rx=k.alignment[:3,:3]@Rotation.from_quat(observation['fingers'][name]['quaternion_xyzw']).as_matrix()
        distal=distal_from_rx(Rotation.from_quat(observation['fingers'][name]['quaternion_xyzw']).as_matrix(),
                              k.alignment[:3,:3],k.mounts[f,:3,:3])
        predicted=rs[k.rig['distal_bodies'][f]]
        observed.append(anatomical_target(distal,frames[f],mirror,wrist))
        fitted.append(anatomical_target(predicted,frames[f],mirror,wrist))
        source_errors.append(angle(predicted@k.mounts[f,:3,:3],rx))
        d=k.rig['distal_bodies'][f];parent=k.rig['parents'][d]
        u=ps[d]-ps[parent];v=ps[k.rig['tip_bodies'][f]]-ps[d]
        axis=rs[d]@k.joint_axis0[k.rig['body_joints'][d][-1]]
        source_bends.append(float(np.degrees(np.arctan2(axis@np.cross(u,v),u@v))))
    return dict(observed=np.array(observed),fitted=np.array(fitted),source_errors=source_errors,
        source_bends=source_bends,wrist=wrist,tip_positions=(source['joints'][[4,8,12,16,20]]-source['joints'][0])@wrist)


def robot_frames(fk,q):
    geometry=fk.forward(q);r=fk.data.xmat.reshape(-1,3,3)
    matrices=np.array([r[fk.body_ids[4+4*f]]@fk.surface.bases[4+4*f] for f in range(5)])
    return geometry,matrices


def metrics(engine,q,t):
    g,r=robot_frames(engine.fk,q)
    return dict(orientation_to_emf_deg=[angle(a,b) for a,b in zip(r,t['observed'])],
        orientation_to_source_fit_deg=[angle(a,b) for a,b in zip(r,t['fitted'])],
        longitudinal_to_emf_deg=[direction_angle(a[:,2],b[:,2]) for a,b in zip(r,t['observed'])],
        palmar_to_emf_deg=[direction_angle(a[:,1],b[:,1]) for a,b in zip(r,t['observed'])],
        fingertip_error_mm=(np.linalg.norm(g[0][[4,8,12,16,20]]-t['tip_positions'],axis=1)*1000).tolist(),
        distal_bend_deg=np.degrees(np.asarray(q)[[3,7,11,15,19]]).tolist())


def optimize_objective(engine,objective,start,lower,upper,*,seeds=None,constraints=None):
    best=None;results=[]
    def value(q):
        v,g,_=objective.evaluate(q,engine.fk.forward(q));return v,g
    conditions=[]
    if constraints is not None:
        conditions=[dict(type='ineq',fun=lambda q:constraints.evaluate(q)[0]*1000.,jac=lambda q:constraints.evaluate(q)[1]*1000.)]
    for seed in (seeds or [start]):
        initial=np.clip(seed,lower,upper);began=time.perf_counter()
        r=minimize(value,initial,method='SLSQP',jac=True,bounds=list(zip(lower,upper)),constraints=conditions,
                   options=dict(maxiter=300,ftol=1e-10))
        v=float(value(r.x)[0]);violation=0. if constraints is None else max(0.,-float(constraints.evaluate(r.x)[0].min())*1000)
        results.append(dict(cost=v,success=bool(r.success),status=int(r.status),iterations=int(r.nit),evaluations=int(r.nfev),
            signed_witness_violation_mm=violation,ms=(time.perf_counter()-began)*1000))
        key=(violation>1e-3,violation if violation>1e-3 else 0.,v)
        if best is None or key<best[0]:best=(key,r.x.copy(),len(results)-1)
    return best[1],dict(selected=best[2],runs=results)


class OrientationObjective:
    def __init__(self,fk,base,target,weight=1.):self.fk=fk;self.base=base;self.target=target;self.weight=weight
    def evaluate(self,q,geometry,stage=2):
        cost,gradient,terms=self.base.evaluate(q,geometry,stage);extra=0.;jac=np.zeros(20)
        _,r=robot_frames(self.fk,q)
        for f in range(5):
            body=int(self.fk.body_ids[4+4*f]);linear=np.zeros((3,self.fk.model.nv));angular=np.zeros_like(linear)
            mujoco.mj_jacBody(self.fk.model,self.fk.data,linear,angular,body)
            error=r[f]-self.target[f]
            extra+=self.weight*.25*float(np.sum(error**2))
            for k in range(20):
                derivative=np.cross(angular[:,6+k],r[f].T).T
                jac[k]+=self.weight*.5*float(np.sum(error*derivative))
        return cost+extra,gradient+jac,{**terms,'distal_orientation':extra}


def oracle(engine,f,target,tip_target,start):
    fk=engine.fk;ids=np.arange(4*f,4*f+4);owner=4+4*f;body=int(fk.body_ids[owner])
    lower=fk.lower[ids];upper=fk.upper[ids];base=np.asarray(start).copy();rng=np.random.default_rng(761+f)
    seeds=[base[ids],np.clip(np.zeros(4),lower,upper),(lower+upper)/2,*rng.uniform(lower,upper,size=(5,4))]
    def evaluate(x):
        q=base.copy();q[ids]=x;geometry,r=robot_frames(fk,q)
        linear=np.zeros((3,fk.model.nv));angular=np.zeros_like(linear)
        mujoco.mj_jacBody(fk.model,fk.data,linear,angular,body)
        difference=r[f]-target;dp=geometry[0][4+4*f]-tip_target
        jr=np.stack([np.cross(angular[:,6+j],r[f].T).T.ravel() for j in ids],axis=1)
        return difference.ravel(),jr,dp,geometry[1][4+4*f][:,ids]
    def orientation(x):
        e,j,_,_=evaluate(x);return float(.25*e@e),.5*j.T@e
    def position(x):
        _,_,p,j=evaluate(x);return float(.5e6*p@p),1e6*j.T@p
    def fit(fun,conditions=()):
        best=None
        for x in seeds:
            r=minimize(fun,x,jac=True,bounds=list(zip(lower,upper)),method='SLSQP',constraints=conditions,
                       options=dict(maxiter=250,ftol=1e-11))
            score=float(fun(r.x)[0]);violation=0. if not conditions else max(0.,-float(conditions[0]['fun'](r.x)))
            key=(violation>1e-4,violation if violation>1e-4 else 0.,score)
            if best is None or key<best[0]:best=(key,r)
        return best[1]
    pos=fit(position);best_pos=float(np.linalg.norm(evaluate(pos.x)[2]));seeds.insert(0,pos.x)
    ori=fit(orientation);budget=best_pos+.005
    condition=dict(type='ineq',fun=lambda x:(budget**2-float(evaluate(x)[2]@evaluate(x)[2]))*1e6,
        jac=lambda x:-2e6*evaluate(x)[2]@evaluate(x)[3])
    pose=fit(orientation,[condition]);eo=evaluate(ori.x)[0].reshape(3,3)+target;ep=evaluate(pose.x)[0].reshape(3,3)+target
    q_orientation=base.copy();q_orientation[ids]=ori.x
    q_pose=base.copy();q_pose[ids]=pose.x
    return dict(best_found_orientation_only_deg=angle(eo,target),analytic_abduction_bound=abduction_bound(engine,f,target),
        exact_orientation_position_bound=exact_orientation_position_bound(engine,f,target,tip_target),best_found_tip_position_mm=best_pos*1000,
        near_best_position_budget_mm=budget*1000,near_best_position_orientation_deg=angle(ep,target),
        near_best_position_actual_mm=float(np.linalg.norm(evaluate(pose.x)[2])*1000),
        near_best_position_success=bool(pose.success),orientation_only_q=ori.x.tolist(),near_best_position_q=pose.x.tolist())


def live_metrics(replay,engine,frames,pairs):
    from moxian_web.skeleton import JOINT_CHAINS
    rows=[];k=replay.ik.kinematics
    for row in pairs:
        s=row['source'];p=s['pose'];lookup=dict(zip(p['names'],p['positions_local']))
        if p['model_hash']!=engine.source.identity[2]:raise ValueError('Live calibration/model differs')
        joints=np.array([lookup['wrist']]+[lookup[n] for c in JOINT_CHAINS for n in c[-4:]])@engine.source.mirror.T
        observation=dict(fingers={})
        for f,name in enumerate(FINGERS):
            measured=Rotation.from_quat(s['targets']['quaternions_local_xyzw'][f]).as_matrix()
            raw=k.alignment[:3,:3].T@measured
            observation['fingers'][name]=dict(quaternion_xyzw=Rotation.from_matrix(raw).as_quat().tolist())
        t=targets(replay,engine,observation,dict(joints=joints),np.array(p['joint_positions_rad']),frames)
        assert np.max(abs(t['wrist']-row['robot']['wrist_rotation']))<1e-5
        assert np.max(abs(np.array([v['rotation_error_deg'] for v in s['ik']['fingers']])-t['source_errors']))<1e-5
        rows.append(dict(timestamp_ms=row['robot']['source_timestamp_ms'],source_ik_error_deg=t['source_errors'],
            metrics=metrics(engine,row['robot']['qpos'][6:],t),execution_mode=row['robot']['solver']['execution_mode']))
    return rows


def source_sensor_oracles(replay,pair):
    """Source-only offline controls; release history/motion bounds explicitly."""
    from moxian_web.emf_qp import rotation_error_jacobian
    k=replay.ik.kinematics;c=replay.ik.config;q=np.asarray(pair['source']['pose']['joint_positions_rad'])
    rows=[];rng=np.random.default_rng(872)
    for f,name in enumerate(FINGERS):
        ids=k.finger_indices[f];rest=np.asarray(k.definition['rest_angles_rad'])[ids]
        target_r=Rotation.from_quat(pair['source']['targets']['quaternions_local_xyzw'][f]).as_matrix()
        target_p=np.asarray(pair['source']['targets']['positions_local'][f])
        posture=np.full(len(ids),c['posture_weight']*(c['thumb_posture_scale'] if f==0 else 1.))
        for i,j in enumerate(ids):
            if k.joint_names[j].endswith('_mcp_twist'):posture[i]*=c.get('root_twist_posture_scale',.01)
            if k.joint_names[j].endswith(('_thumb_cmc_flexion','_thumb_mcp_abduction')):posture[i]=c['thumb_palm_posture_weight']
        def evaluate(x,orientation_only=False):
            full=q.copy();full[ids]=x;r,p,jp,jw=replay.ik.native.sensor(f,full)
            er,jr=rotation_error_jacobian(r,target_r,jw)
            if orientation_only:return float(.5*er@er),jr.T@er
            ep=(p-target_p)/c['position_sigma_m'];er=er/c['rotation_sigma_rad'];dp=x-rest
            return float(.5*(ep@ep+er@er+(posture*dp)@dp)),jp.T@ep/c['position_sigma_m']+jr.T@er/c['rotation_sigma_rad']+posture*dp
        def solve(only):
            seeds=[q[ids],rest,*rng.uniform(k.limits[ids,0],k.limits[ids,1],size=(4,len(ids)))]
            best=None
            for seed in seeds:
                r=minimize(lambda x:evaluate(x,only),seed,jac=True,method='SLSQP',bounds=list(map(tuple,k.limits[ids])),
                           options=dict(maxiter=300,ftol=1e-11))
                if best is None or evaluate(r.x,only)[0]<best[0]:best=(evaluate(r.x,only)[0],r.x.copy())
            full=q.copy();full[ids]=best[1];rr,pp,_,_=replay.ik.native.sensor(f,full)
            return dict(rotation_error_deg=angle(rr,target_r),position_error_mm=float(np.linalg.norm(pp-target_p)*1000),q=best[1].tolist())
        rows.append(dict(finger=name,no_history_full_budget=solve(False),orientation_only=solve(True)))
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--frames',type=int,default=15)
    ap.add_argument('--source-model',type=Path,default=Path('results/moxian_wuji_contact_laplacian_20261112_final/source-model.json'))
    ap.add_argument('--capture',type=Path,default=Path('output/moxian/hand_model/348640923d9e2f3f/qnbot/raw-captures/glove_20261112T151631_867012Z.jsonl'))
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True);result_path=a.output/'analysis.json'
    if result_path.exists():raise FileExistsError(result_path)
    torch.set_num_threads(1);candidate=json.loads(a.source_model.read_text());replay=MoxianReplayReconstructor(candidate)
    raw=[json.loads(x) for x in a.capture.open()];engine=None;records=[];frames=None
    for stage in dict.fromkeys(r.get('stage','unknown') for r in raw):
        indices=[i for i,r in enumerate(raw) if r.get('stage','unknown')==stage]
        active=[i for i in indices if any((raw[i].get('contact') or {}).get('cell_contact',[]))];centre=(active or indices)[len(active or indices)//2]
        replay.ik.reset();replay.previous_stamp=None
        if engine:engine.reset()
        for i in range(max(indices[0],centre-a.frames+1),centre+1):
            frame=replay.frame(raw[i])
            if engine is None:
                engine=ContactSLSQPRetarget(replay.topology,a.output/'mappings','cuda:0' if torch.cuda.is_available() else 'cpu',realtime=True)
                frames,frame_checks=source_frames(replay,engine)
            realtime=engine.step(frame)
        q_source=replay.ik.previous.copy();source=engine.source.read(frame);anchors=engine.material_contacts(source);engine.fk.set_contacts(anchors)
        t=targets(replay,engine,raw[centre],source,q_source,frames);wrist=t['wrist'];origin=source['joints'][0]
        skeleton=(source['joints']-origin)@wrist;contacts=(np.array([x['position_m'] for x in anchors]).reshape(-1,3)-origin)@wrist
        normals=np.array([x['direction'] for x in anchors]).reshape(-1,3)@wrist;links=[x['semantic_link'] for x in anchors]
        initial=np.array(realtime['initial_qpos'][6:]);qr=np.array(realtime['qpos'][6:]);settings=engine.settings
        make=lambda settings:ContactObjective(skeleton,contacts,normals,source['edges'],initial,contact_links=links,settings=settings,fast=True)
        objective=make(settings);witness=engine.separation.refresh(initial,wrist,origin,per_pair=1)
        variants={};variants['realtime']=dict(q=qr.tolist(),initial_q=initial.tolist(),metrics=metrics(engine,qr,t),cost_terms=objective.evaluate(qr,engine.fk.forward(qr))[2],
            cost=objective.evaluate(qr,engine.fk.forward(qr))[0],feasible=realtime['solution_feasible'],solver=realtime['solver'])
        configs=[('same_bounds_more_iterations',objective,engine.solve_lower,engine.solve_upper,[qr],witness),
            ('full_bounds_more_iterations',objective,engine.fk.lower,engine.fk.upper,[qr,np.zeros(20),(engine.fk.lower+engine.fk.upper)/2],None),
            ('contact_priority_local',make(replace(settings, active_contact_skeleton_gain=6.,
                                                    active_contact_material_gain=4.,
                                                    active_contact_edge_gain=4.,
                                                    inactive_skeleton_floor=.35)),
             engine.fk.lower,engine.fk.upper,[qr,np.zeros(20)],None),
            ('contact_priority_mild',make(replace(settings, active_contact_skeleton_gain=2.,
                                                   active_contact_material_gain=1.,
                                                   active_contact_edge_gain=1.,
                                                   inactive_skeleton_floor=.5)),
             engine.fk.lower,engine.fk.upper,[qr,np.zeros(20)],None),
            ('distal_direction_contact',make(replace(settings, lambda_contact_distal_direction=10.)),
             engine.fk.lower,engine.fk.upper,[qr,np.zeros(20)],None),
            ('distal_direction_contact_mild',make(replace(settings, lambda_contact_distal_direction=2.)),
             engine.fk.lower,engine.fk.upper,[qr,np.zeros(20)],None),
            ('no_antireverse',make(replace(settings,lambda_antirev=0.)),engine.fk.lower,engine.fk.upper,[qr,np.zeros(20)],None),
            ('pose_only_no_antireverse',make(replace(settings,lambda_antirev=0.,lambda_contact_edge=0.)),engine.fk.lower,engine.fk.upper,[qr,np.zeros(20)],None)]
        for name,obj,lower,upper,seeds,constraints in configs:
            q,trace=optimize_objective(engine,obj,qr,lower,upper,seeds=seeds,constraints=constraints)
            engine.fk.forward(q);ok,audit,_,_=engine.audit(wrist,origin,frame)
            variants[name]=dict(q=q.tolist(),metrics=metrics(engine,q,t),cost_terms=obj.evaluate(q,engine.fk.forward(q))[2],
                cost=obj.evaluate(q,engine.fk.forward(q))[0],trace=trace,penetration_mm=audit['max_penetration_mm'])
        base=make(replace(settings,lambda_antirev=0.));orientation_objective=OrientationObjective(engine.fk,base,t['observed'],1.)
        qo,trace=optimize_objective(engine,orientation_objective,np.array(variants['no_antireverse']['q']),engine.fk.lower,engine.fk.upper,
            seeds=[np.array(variants['no_antireverse']['q']),qr])
        engine.fk.forward(qo);_,audit,_,_=engine.audit(wrist,origin,frame)
        variants['diagnostic_orientation_objective']=dict(q=qo.tolist(),metrics=metrics(engine,qo,t),trace=trace,penetration_mm=audit['max_penetration_mm'])
        oracles=[oracle(engine,f,t['observed'][f],t['tip_positions'][f],qr) for f in range(5)]
        for f,o in enumerate(oracles):
            o['fitted_source_exact_orientation_position_bound']=exact_orientation_position_bound(engine,f,t['fitted'][f],t['tip_positions'][f])
        engine.realtime=False;engine.previous=initial.copy();engine.previous_timestamp=frame['timestamp_ms']-1000/60
        engine.settings=replace(settings,later_stage1_maxeval=80,later_stage2_maxeval=800,contact_stage2_maxeval=800,ftol_abs=1e-9)
        full=engine.step(frame);q_full=np.array(full['qpos'][6:])
        variants['offline_collision_checked']=dict(q=q_full.tolist(),metrics=metrics(engine,q_full,t),feasible=full['solution_feasible'],
            cost=objective.evaluate(q_full,engine.fk.forward(q_full))[0],solver=full['solver'],penetration_mm=full['collision']['max_penetration_mm'])
        engine.realtime=True;engine.settings=settings
        record=dict(stage=stage,timestamp_ms=frame['timestamp_ms'],source_ik_orientation_error_deg=t['source_errors'],source_distal_physical_bend_deg=t['source_bends'],
            variants=variants,orientation_oracles=oracles,q_source=q_source.tolist(),anatomical_frame_source=frames.tolist(),anatomical_frame_robot=engine.surface.bases[[4,8,12,16,20]].tolist())
        records.append(record);(a.output/'progress.json').write_text(json.dumps(records,indent=2))
        print(stage,'source_ik',np.round(t['source_errors'],1),'RT',np.round(variants['realtime']['metrics']['orientation_to_emf_deg'],1),
            'orientation_floor',np.round([o['best_found_orientation_only_deg'] for o in oracles],1),flush=True)
    # A fully reachable synthetic reverse-bend target removes source calibration
    # and robot morphology as possible explanations of antireverse bias.
    truth=np.zeros(20);truth[16:20]=[.4,.1,.8,-np.pi/6];g=engine.fk.forward(truth);synthetic=[]
    _,synthetic_audit,_,_=engine.audit(np.eye(3),np.zeros(3),{})
    _,truth_rotations=robot_frames(engine.fk,truth)
    for gain in (20.,0.):
        obj=ContactObjective(g[0].copy(),[],[],[],truth,settings=replace(engine.settings,lambda_antirev=gain),fast=True)
        straight=truth.copy();straight[19]=0.
        q,trace=optimize_objective(engine,obj,truth,engine.fk.lower,engine.fk.upper,seeds=[truth,straight])
        _,fitted_rotations=robot_frames(engine.fk,q)
        synthetic.append(dict(antireverse_weight=gain,target_q=truth.tolist(),at_target_cost=obj.evaluate(truth,engine.fk.forward(truth))[0],
            at_straight_distal_cost=obj.evaluate(straight,engine.fk.forward(straight))[0],result_q=q.tolist(),trace=trace,
            fingertip_error_mm=float(np.linalg.norm(engine.fk.forward(q)[0][20]-g[0][20])*1000),
            distal_orientation_error_deg=angle(fitted_rotations[4],truth_rotations[4])))
    data=dict(model_hash=candidate['model_hash'],calibration_fingerprint=candidate['calibration_fingerprint'],source_model=str(a.source_model),capture=str(a.capture),
        orientation_convention='remove RX mounting; fixed longitudinal/palmar anatomical frames; palm-aligned robot wrist; mirrored proper-frame parity',
        source_frame_palmar_checks=frame_checks,robot_joints=[dict(name=n,limits_deg=np.degrees(l).tolist()) for n,l in zip(engine.fk.names,np.c_[engine.fk.lower,engine.fk.upper])],
        synthetic_reverse=synthetic,synthetic_target_penetration_mm=synthetic_audit['max_penetration_mm'],
        records=records,production_settings_changed=False)
    live_path=a.output/'live-pairs.json'
    if live_path.exists():
        pairs=json.loads(live_path.read_text());data['live_orientation']=live_metrics(replay,engine,frames,pairs)
        if pairs:data['live_source_sensor_oracles']=source_sensor_oracles(replay,pairs[0])
    result_path.write_text(json.dumps(data,indent=2));print('wrote',result_path,flush=True)

if __name__=='__main__':main()
