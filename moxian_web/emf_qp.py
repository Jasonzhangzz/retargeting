"""Per-finger bounded Gauss--Newton IK with a shared per-frame motion budget."""
from __future__ import annotations
import time
import numpy as np
from scipy.spatial.transform import Rotation
from .bone_extrinsics import BoneKinematics,FINGERS,SCHEMA,validate_candidate

REVISION='moxian_emf_bone_qp_v2_frame_bounds'


def box_qp(h,g,lower,upper):
    """Small SPD box QP, active set and Cholesky, at most n+2 passes."""
    h=np.asarray(h,float); g=np.asarray(g,float); lower=np.asarray(lower); upper=np.asarray(upper)
    if np.any(lower>upper): raise ValueError('QP上下界冲突')
    x=np.clip(np.zeros(len(g)),lower,upper)
    active=np.where(lower==upper,2,0)
    for _ in range(len(g)+2):
        free=np.flatnonzero(active==0); fixed=np.flatnonzero(active!=0)
        if len(free):
            block=h[np.ix_(free,free)]; chol=np.linalg.cholesky(block)
            if np.min(np.diag(chol))**2<1e-12: raise ValueError('QP非正定')
            rhs=-g[free]-h[np.ix_(free,fixed)]@x[fixed]
            target=x.copy(); target[free]=np.linalg.solve(chol.T,np.linalg.solve(chol,rhs))
            delta=target-x; alpha=1.; hit=None; side=0
            for i in free:
                a=(upper[i]-x[i])/delta[i] if delta[i]>0 else (lower[i]-x[i])/delta[i] if delta[i]<0 else np.inf
                if a<alpha: alpha=max(0.,a); hit=i; side=1 if delta[i]>0 else -1
            x=np.clip(x+alpha*delta,lower,upper)
            if hit is not None: active[hit]=side; continue
        gradient=h@x+g
        violations=np.where(active==-1,-gradient,np.where(active==1,gradient,-np.inf))
        if np.max(violations)<=1e-10: break
        active[int(np.argmax(violations))]=0
    return x


def rotation_error_jacobian(r,target,angular):
    error=Rotation.from_matrix(r.T@target).as_rotvec(); angle=np.linalg.norm(error)
    x,y,z=error; k=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    c=1/12+angle*angle/720 if angle<1e-5 else (1-.5*angle/np.tan(.5*angle))/(angle*angle)
    return error,-(np.eye(3)-.5*k+c*k@k)@r.T@angular


class QPSkeleton:
    def __init__(self,candidate):
        validate_candidate(candidate)
        self.candidate=candidate; self.definition=candidate['bone_extrinsics']
        self.kinematics=BoneKinematics(self.definition); self.config=self.definition['solver_config']
        self.model_hash=candidate['model_hash']; self.personalized=dict(geometry_hash=self.model_hash)
        policy=self.definition.get('joint_limit_policy',{}).get('finger_axial_twist',{})
        self.periodic_joints=set(policy.get('joint_names',[])) if policy.get('periodic') else set()
        self.input_registration=None; self.reset()

    def reset(self):
        self.previous=None; self.saved=None; self.previous_timestamp=None; self.frames=0

    def close(self): pass

    def policy(self,original=None):
        return dict(original or {},solver=REVISION,solver_parameters=self.config,
            personalized_geometry_hash=self.model_hash,runtime_policy_sha256=self.model_hash,solver_dof=len(self.kinematics.joint_names),joint_limit_policy=self.definition.get('joint_limit_policy'),
            native_angles_used=False,physics_enabled=False,schema_revision=SCHEMA)

    def _solve(self,f,start,target_p,target_r,dt,history):
        k=self.kinematics; c=self.config; ids=k.finger_indices[f]
        rest=np.asarray(self.definition['rest_angles_rad'])[ids]
        periodic=np.asarray([k.joint_names[i] in self.periodic_joints for i in ids])
        rest=rest.copy();rest[periodic]+=2*np.pi*np.round((start[ids][periodic]-rest[periodic])/(2*np.pi))
        posture=np.full(len(ids),c['posture_weight']*(c['thumb_posture_scale'] if f==0 else 1.))
        twist=np.asarray([k.joint_names[i].endswith('_mcp_twist') for i in ids])
        posture[twist]*=c.get('root_twist_posture_scale',.01)
        # These two nearly collinear palm axes can trade rotation while
        # leaving the distal RX almost unchanged. Prefer the calibrated flat
        # branch and its previous output, without locking thumb opposition.
        palm=np.asarray([k.joint_names[i].endswith(('_thumb_cmc_flexion','_thumb_mcp_abduction')) for i in ids])
        posture[palm]=c.get('thumb_palm_posture_weight',c['posture_weight']*c['thumb_posture_scale'])
        lower=k.limits[ids,0].copy();upper=k.limits[ids,1].copy()
        lower[periodic]=start[ids][periodic]-np.pi;upper[periodic]=start[ids][periodic]+np.pi
        # A physical speed limit bounds the final pose of the entire frame.
        # Solver iterations/retries may never spend that budget again.
        if history is not None:
            budget=c['v_max']*dt
            lower=np.maximum(lower,start[ids]-budget)
            upper=np.minimum(upper,start[ids]+budget)
        temporal=np.full(len(ids),0. if history is None else c['history_weight'])
        if history is not None:temporal[palm]=c.get('thumb_palm_history_weight',c['history_weight'])
        def evaluate(q):
            r,p,jp,jw=k.sensor(f,q,True); er,jr=rotation_error_jacobian(r,target_r,jw)
            residual=np.r_[(p-target_p)/c['position_sigma_m'],er/c['rotation_sigma_rad']]
            j=np.vstack((jp/c['position_sigma_m'],jr/c['rotation_sigma_rad']))
            dp=q[ids]-rest; dh=q[ids]-(history[ids] if history is not None else start[ids])
            cost=float(residual@residual+(posture*dp)@dp+(temporal*dh)@dh)
            return cost,residual,j,float(np.linalg.norm(p-target_p)),float(np.linalg.norm(er))
        best=start.copy(); best_cost=evaluate(best)[0]; iterations=0
        for retry,multiplier in enumerate((1.,2.,4.)):
            q=start.copy(); limit=c['max_iterations'] if retry==0 else c['retry_iterations']
            for _ in range(limit):
                cost,residual,j,_,_=evaluate(q); iterations+=1
                mu=multiplier*c['lm_damping']*float(residual@residual)+c['global_damping']
                h=j.T@j+np.diag(mu+posture+temporal)
                g=j.T@residual+posture*(q[ids]-rest)
                if history is not None: g+=temporal*(q[ids]-history[ids])
                lo=c['limit_gain']*(lower-q[ids])
                hi=c['limit_gain']*(upper-q[ids])
                delta=box_qp(h,g,lo,hi)
                # Retain the best finite descent; nonlinear rotations need a
                # line search even when the local quadratic is positive definite.
                accepted=False
                for scale in (1.,.5,.25,.125,.0625):
                    proposed=q.copy(); proposed[ids]=np.clip(q[ids]+scale*delta,lower,upper)
                    score=evaluate(proposed)[0]
                    if np.isfinite(score) and score<=cost+1e-12:
                        q=proposed; accepted=True
                        if score<best_cost: best=q.copy(); best_cost=score
                        break
                if not accepted or np.max(abs(delta))<1e-7: break
            _,_,_,ep,er=evaluate(best)
            if ep<=c['retry_position_m'] and er<=c['retry_rotation_rad']: break
        _,_,_,ep,er=evaluate(best)
        return best,dict(finger=FINGERS[f],position_error_mm=ep*1000,
            rotation_error_deg=float(np.rad2deg(er)),tracking_satisfied=bool(ep<=.015 and er<=np.deg2rad(15)),
            iterations=iterations,cost=best_cost,success=True)

    def step(self,frame):
        began=time.perf_counter(); stamp=float(frame['timestamp_ms'])
        if not np.isfinite(stamp): raise ValueError('无效时间戳')
        interval=None if self.previous_timestamp is None else (stamp-self.previous_timestamp)/1000
        if interval is not None and interval<=0: raise ValueError('时间戳必须递增')
        if interval is not None and interval>.25: self.reset(); interval=None
        k=self.kinematics; first=self.previous is None; rest=np.asarray(self.definition['rest_angles_rad'])
        start=rest.copy() if first else self.previous.copy(); q=start.copy()
        dt=1/60 if interval is None else min(interval,1/15)
        # Previous output is the history target. Unbounded extrapolation from
        # two poses can select a different IK branch after a bad observation.
        history=None if first else start.copy()
        p=[]; r=[]
        for f in FINGERS:
            node=frame['fingers'][f]; pp=np.asarray(node['position'],float); qq=np.asarray(node['quaternion_xyzw'],float)
            if pp.shape!=(3,) or qq.shape!=(4,) or not np.isfinite(np.r_[pp,qq]).all() or np.linalg.norm(qq)<1e-8 or node.get('valid') is False or node.get('confidence',1)<=0:
                # A single corrupt packet must not erase the last good branch.
                # A genuine gap is reset above using the last accepted stamp.
                raise ValueError('无效EMF观测')
            p.append(pp); r.append(Rotation.from_quat(qq).as_matrix())
        align=k.alignment; p=np.asarray(p)@align[:3,:3].T+align[:3,3]; r=align[:3,:3]@np.asarray(r)
        reports=[]
        for f in range(5):
            q,report=self._solve(f,q,p[f],r[f],dt,history); reports.append(report)
        rotations,positions=k.forward(q)
        from .skeleton import JOINT_NAMES
        # SDK's public node ABI is retained; auxiliary FK bodies stay private.
        names=list(JOINT_NAMES); body_ids=[k.body_names.index(k.rig['side']+'_'+name) for name in names]
        pose=dict(timestamp_ms=stamp,hand_side=k.rig['side'],coordinate_frame='wrist_local',names=names,
            joint_names=k.joint_names,joint_positions_rad=q.tolist(),positions_local=positions[body_ids].tolist(),
            orientations_local_xyzw=Rotation.from_matrix(rotations[body_ids]).as_quat().tolist(),
            ik_solver=REVISION,personalized_geometry_hash=self.model_hash,model_hash=self.model_hash,
            tracking_errors={v['finger']:dict(position_error_mm=v['position_error_mm'],rotation_error_deg=v['rotation_error_deg'],satisfied=v['tracking_satisfied']) for v in reports})
        maximum=0. if first else float(np.rad2deg(abs(q-start).max()))
        self.saved=start.copy(); self.previous=q.copy(); self.previous_timestamp=stamp; self.frames+=1
        from .native_skeleton import joint_limit_diagnostics
        limits={name:value for name,value in zip(k.joint_names,k.limits.tolist()) if name not in self.periodic_joints}
        return dict(pose=pose,targets=dict(timestamp_ms=stamp,hand_side=k.rig['side'],coordinate_frame='wrist_local',names=list(FINGERS),positions_local=p.tolist(),rotations_local=r.reshape(5,9).tolist(),quaternions_local_xyzw=Rotation.from_matrix(r).as_quat().tolist(),orientation_convention='EMF_RX',registration=REVISION),
            physics=dict(enabled=False,steps=0),joint_limit_diagnostics=joint_limit_diagnostics(k.joint_names,q,limits),
            continuity=dict(engine_frames=self.frames,reused_engine=not first,interval_ms=None if interval is None else interval*1000,max_joint_step_deg=maximum),
            ik_diagnostics=dict(algorithm=REVISION,model_hash=self.model_hash,native_angles_used=False,
                previous_pose_initial_guess=not first,fingers=reports,processing_ms=(time.perf_counter()-began)*1000))
