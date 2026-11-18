"""EMF pose IK on the calibrated Moxian URDF, with analytic Jacobians.

The vendor adapter supplies calibrated target transforms only. Its IK angles
are never seeds or outputs of this solver. Position/rotation, posture and
history terms follow emf_wuji_retarget/human_ik_reference; bounded SLSQP
follows the reference project's retarget session.
"""
from pathlib import Path
import hashlib
import json
import math
import time
import xml.etree.ElementTree as ET

import numpy as np
import yaml
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from .native_skeleton import NativeSkeleton, joint_limit_diagnostics
from .thumb_limits import THUMB_EXTENSION_DEG
from .finger_limits import extension_bounds as finger_extension_bounds,DIP_EXTENSION_DEG,PIP_EXTENSION_DEG

REVISION = 'moxian_emf_pose_slsqp_v6_bounded_extension'
FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')
LINK_FINGERS = ('thumb', 'index', 'middle', 'ring', 'little')
PARAMETERS = dict(position_weight_per_mm=.3, rotation_weight=5.729577951308232,
                  posture_weight=1., thumb_posture_scale=.01, history_weight=20.,
                  first_maxiter=80, later_maxiter=30, ftol=1e-8,
                  extension_allowance_deg=PIP_EXTENSION_DEG,thumb_extension_allowance_deg=THUMB_EXTENSION_DEG,dip_extension_allowance_deg=DIP_EXTENSION_DEG)


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def rotation_residual_jacobian(rotation, target, angular):
    """Log(R.T target) and its derivative for a spatial angular Jacobian."""
    error = Rotation.from_matrix(rotation.T @ target).as_rotvec()
    angle = np.linalg.norm(error); k = skew(error)
    coefficient = 1/12 + angle*angle/720 if angle < 1e-5 else (1-.5*angle/math.tan(.5*angle))/(angle*angle)
    inverse = np.eye(3)-.5*k+coefficient*(k@k)
    return error, -inverse @ rotation.T @ angular


class SkeletonKinematics:
    def __init__(self, path, side, joint_names, node_names, bone_lengths=None):
        self.path=str(Path(path).resolve())
        self.side = side; self.joint_names = list(joint_names); self.node_names = list(node_names)
        self.index = {n:i for i,n in enumerate(joint_names)}; self.joints = []; self.by_child = {}
        self.bounds = np.empty((len(joint_names), 2)); self.straight = np.zeros(len(joint_names))
        self.extension_limits = {}
        tree = ET.parse(path).getroot()
        for element in tree.findall('joint'):
            name = element.get('name'); origin = element.find('origin'); tf = np.eye(4)
            if origin is not None:
                tf[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
                rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')
                tf[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
            else:rpy = np.zeros(3)
            axis = element.find('axis'); axis = np.fromstring(axis.get('xyz', '1 0 0'), sep=' ') if axis is not None else np.array([1.,0.,0.])
            axis /= np.linalg.norm(axis)
            k = self.index.get(name)
            item = dict(name=name, parent=element.find('parent').get('link'), child=element.find('child').get('link'), origin=tf, axis=axis, index=k)
            if bone_lengths:
                for finger,semantic in zip(LINK_FINGERS,('thumb','first','middle','ring','little')):
                    prefix=side+'_'+finger+'_'
                    if not name.startswith(prefix):continue
                    suffix=name[len(prefix):]
                    segment=('metacarpal' if suffix=='mcp_abduction' else
                             'proximal' if suffix in ('pip_flexion','ip_flexion') else
                             'intermediate' if suffix=='dip_flexion' else
                             'distal' if item['child']==side+'_'+finger+'_tip' else None)
                    if segment is not None and segment in bone_lengths.get(semantic,{}):
                        length=float(bone_lengths[semantic][segment]);norm=np.linalg.norm(tf[:3,3])
                        if not .003<=length<=.15 or norm<1e-8:raise ValueError('Invalid calibrated bone length: '+name)
                        tf[:3,3]*=length/norm
            self.by_child[item['child']] = item; self.joints.append(item)
            if k is not None:
                limit = element.find('limit'); lo,hi = float(limit.get('lower')),float(limit.get('upper'))
                # Raw q=0 includes the URDF origin's static IP bend. Permit
                # physical extension without changing any bone or joint axis.
                if any(token in name for token in ('_pip_flexion','_dip_flexion','_thumb_ip_flexion')) and np.allclose(axis,[0,0,1]):
                    self.straight[k] = -rpy[2]
                    lo = self.straight[k]-math.radians(PARAMETERS['extension_allowance_deg'])
                    hi = max(hi, self.straight[k]+math.radians(PARAMETERS['extension_allowance_deg']))
                    self.extension_limits[name] = [lo,hi]
                if name.endswith('_thumb_mcp_flexion'):
                    parent = self.by_child.get(item['parent'])
                    if parent is not None:self.straight[k] = -Rotation.from_matrix(parent['origin'][:3,:3]).as_euler('xyz')[2]
                    lo = self.straight[k]-math.radians(3); hi = max(hi,self.straight[k]+math.radians(3))
                    self.extension_limits[name] = [lo,hi]
                if name.endswith(('_thumb_mcp_flexion','_thumb_ip_flexion')):
                    from .thumb_limits import extension_bounds
                    lo,hi=extension_bounds(name,lo,hi,self.straight[k]);self.extension_limits[name]=[lo,hi]
                if name.endswith(('_dip_flexion','_pip_flexion')):
                    lo,hi=finger_extension_bounds(name,lo,hi,self.straight[k]);self.extension_limits[name]=[lo,hi]
                self.bounds[k] = lo,hi
        self.straight = np.clip(self.straight,self.bounds[:,0],self.bounds[:,1])
        self.chains = []; self.finger_indices = []
        for finger in LINK_FINGERS:
            chain = self.chain(side+'_'+finger+'_tip'); self.chains.append(chain)
            self.finger_indices.append(np.array([j['index'] for j in chain if j['index'] is not None],dtype=int))
        self.node_chains = [self.chain(side+'_'+n) for n in node_names]
        self.asset_sha256 = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        zero=self.nodes(np.zeros(len(joint_names)))
        self.reference_model=dict(timestamp_ms=0.,hand_side=side,coordinate_frame='wrist_local',names=list(node_names),
                                  positions_local=zero[:,:3,3].tolist(),orientations_local_xyzw=Rotation.from_matrix(zero[:,:3,:3]).as_quat().tolist())
        self.reference_hash=hashlib.sha256(json.dumps(self.reference_model,sort_keys=True).encode()).hexdigest()
        self.thumb_model = dict(revision='bounded_urdf_thumb_v1', side=side,
            joints=[dict(name=j['name'], parent=j['parent'], child=j['child'],
                         rotation=j['origin'][:3,:3].tolist(),axis=j['axis'].tolist(),
                         bounds=self.bounds[j['index']].tolist())
                    for j in self.chains[0] if j['index'] is not None])

    def chain(self, link):
        chain = []
        while link != self.side+'_wrist':
            if link not in self.by_child:raise ValueError('Skeleton link disconnected from wrist: '+link)
            j = self.by_child[link]; chain.append(j); link = j['parent']
        return list(reversed(chain))

    def forward_chain(self, chain, q, jacobian=False):
        tf = np.eye(4); axes = []; centres = []
        for joint in chain:
            tf = tf @ joint['origin']; k = joint['index']
            if k is not None:
                if jacobian:axes.append(tf[:3,:3] @ joint['axis']); centres.append(tf[:3,3].copy())
                tf[:3,:3] = tf[:3,:3] @ Rotation.from_rotvec(joint['axis']*q[k]).as_matrix()
        if not jacobian:return tf
        angular = np.array(axes).T
        position = np.cross(np.array(axes),tf[:3,3]-np.array(centres)).T
        return tf,position,angular

    def finger(self, f, q):return self.forward_chain(self.chains[f],q,True)

    def nodes(self, q):return np.array([self.forward_chain(chain,q) for chain in self.node_chains])


class SLSQPSkeleton:
    def __init__(self, package_root, config_path, side='left',personalized=None):
        self.mapper = NativeSkeleton(package_root,config_path,side=side)
        self.side = side; self.previous = None; self.previous_timestamp = None; self.frames = 0
        self.input_registration = self.mapper.input_registration
        source = None
        for line in Path(config_path).read_text().splitlines():
            if line.startswith('__resource_root__:'):
                source = line.split(':',1)[1].strip(); source = json.loads(source) if source.startswith('"') else source
        source = Path(source) if source else Path(package_root)/'share/qnbot_hand'
        config=yaml.safe_load(Path(config_path).read_text());hand=config.get('hand',{})
        if hand.get('apply_calibrated_palm_layout'):raise ValueError('SLSQP does not support calibrated palm layout; do not substitute default roots')
        for finger in hand.get('fingers',{}).values():
            if finger.get('fingertip_direction_axis','-x')!='-x' or finger.get('fingertip_pad_axis','-y')!='-y':
                raise ValueError('Unsupported fingertip direction/pad frame')
        anthropometry=hand.get('anthropometry',{})
        lengths=anthropometry.get('fingers') if anthropometry.get('enabled') else None
        path = source/f'hands/dynamic_openxr_hand/resources/pinocchio/{side}.urdf'
        # Vendor names select the side only after a first frame. Read the URDF
        # names directly rather than taking the engine's default right names.
        names = [j.get('name') for j in ET.parse(path).getroot().findall('joint') if j.get('type')!='fixed']
        self.personalized=personalized
        if personalized:
            from .glove_geometry import skeleton_urdf
            path=skeleton_urdf(personalized,path,Path(config_path).parent/('glove-'+personalized['geometry_hash']+'.urdf'))
        self.kinematics = SkeletonKinematics(path,side,names,self.mapper.names,None if personalized else lengths)
        if personalized:
            # MANO has fixed metacarpal roots. Extra vendor palm joints would
            # translate ring/little MCPs without a corresponding skin DOF.
            for name,index in self.kinematics.index.items():
                if '_cmc_' in name and '_thumb_' not in name:
                    self.kinematics.bounds[index]=[0.,0.];self.kinematics.straight[index]=0.
            for joint in self.kinematics.thumb_model['joints']:
                joint['bounds']=self.kinematics.bounds[self.kinematics.index[joint['name']]].tolist()
        if personalized:self.kinematics.thumb_model['geometry_applied']=personalized['geometry_hash']
        if personalized:self.personalized={**personalized,'urdf_path':str(path.resolve())}
        self.joint_names = names; self.joint_limits = dict(zip(names,self.kinematics.bounds.tolist()))
        self.mounts = np.tile(np.eye(4),(5,1,1)); self.mount_report = {'applied':False}
        if not personalized:self.fit_flat_mounts(Path(config_path).parent.parent)

    def effector(self,f,q):
        data=getattr(self,'personalized',None)
        if not data:return self.kinematics.finger(f,q)
        kin=self.kinematics;finger=LINK_FINGERS[f]
        tf,jp,jw=kin.forward_chain(kin.chain(self.side+'_'+finger+'_distal'),q,True)
        tf=tf.copy();correction=np.asarray(data['distal_frame_corrections'][f])
        offset=tf[:3,:3]@correction@np.asarray(data['sensor_position_m'][f])
        jp=jp+np.cross(jw.T,offset).T
        tf[:3,3]+=offset;tf[:3,:3]=tf[:3,:3]@correction@np.asarray(data['sensor_rotation'][f])
        return tf,jp,jw

    def pad(self,f,q):
        kin=self.kinematics;data=self.personalized
        tf,jp,jw=kin.forward_chain(kin.chain(self.side+'_'+LINK_FINGERS[f]+'_distal'),q,True)
        offset=tf[:3,:3]@np.asarray(data['pad_local_m'][f])
        normal=tf[:3,:3]@np.asarray(data['pad_normal_local'][f])
        return tf[:3,3]+offset,normal,jp+np.cross(jw.T,offset).T,np.cross(jw.T,normal).T

    def fit_flat_mounts(self, home):
        """Fixed SE(3) receiver registration from training flat observations.

        No gesture classifier is used at runtime. Only the first half of each
        flat capture is training data; the second half remains held out.
        Thumb posture is unspecified by the flat gesture and is excluded.
        """
        source=max((home/'glove/captures').glob('capture-*.json'),key=lambda p:p.stat().st_mtime,default=None)
        if source is None:return
        payload=json.loads(source.read_text());data=payload['payload']
        if data.get('side')!=self.side:raise ValueError('Flat registration hand side mismatch')
        samples=[]
        for stage in data['stage_samples']:
            if stage['stage_id'] not in ('open','close'):continue
            frames=stage['frames'];part=frames[:len(frames)//2]
            if len(part)<3:raise ValueError('Flat mounting needs independent training and holdout samples')
            for i in sorted({0,len(part)//2,len(part)-1}):
                raw=part[i]['payload'];samples.append({'hand_side':self.side,'timestamp_ms':raw['timestamp_ms'],'fingers':raw['fingertip_local']})
        if len(samples)!=6:return
        corrections=[[] for _ in range(5)];kin=self.kinematics
        for row in samples:
            mapped=self.mapper.step(row)
            for f in range(1,5):
                observed=np.eye(4);observed[:3,3]=mapped['targets']['positions_local'][f]
                observed[:3,:3]=np.asarray(mapped['targets']['rotations_local'][f]).reshape(3,3)@np.diag([-1.,-1.,1.])
                base=kin.straight.copy();abd=self.joint_names.index(self.side+'_'+LINK_FINGERS[f]+'_mcp_abduction')
                root=kin.nodes(base)[kin.node_names.index(LINK_FINGERS[f]+'_proximal'),:3,3]
                target=observed[:3,3]-root;target/=max(np.linalg.norm(target),1e-12)
                def direction_cost(x):
                    q=base.copy();q[abd]=x[0];tip,_,_=kin.finger(f,q)
                    direction=tip[:3,3]-root;direction/=max(np.linalg.norm(direction),1e-12)
                    return float(((direction-target)**2).sum())
                result=minimize(direction_cost,[0.],method='SLSQP',bounds=[kin.bounds[abd]],options={'ftol':1e-10})
                if not result.success:raise ValueError('Flat abduction registration failed: '+FINGERS[f])
                base[abd]=result.x[0];desired,_,_=kin.finger(f,base)
                corrections[f].append(np.linalg.inv(observed)@desired)
        for f in range(1,5):
            values=np.array(corrections[f]);self.mounts[f,:3,3]=values[:,:3,3].mean(0)
            self.mounts[f,:3,:3]=Rotation.from_matrix(values[:,:3,:3]).mean().as_matrix()
            if np.linalg.norm(self.mounts[f,:3,3])>.075:raise ValueError('Flat EMF registration exceeds 75 mm: '+FINGERS[f])
        self.mount_report={'applied':True,'training_samples':6,'training_partition':'first_half_of_open_and_close',
                           'source_payload_sha256':payload['payload_sha256'],'thumb_included':False,
                           'correction_tip_local_se3':self.mounts.tolist(),'type':'fixed_SE3_not_runtime_gesture_correction'}

    def policy(self, original):
        key = hashlib.sha256((REVISION+original['runtime_policy_sha256']+self.kinematics.asset_sha256+json.dumps(PARAMETERS,sort_keys=True)+json.dumps(self.mount_report,sort_keys=True)).encode()).hexdigest()
        if self.personalized:key=hashlib.sha256((key+self.personalized['geometry_hash']).encode()).hexdigest()
        return {**original,'mode':'emf_slsqp','revision':REVISION,'runtime_policy_sha256':key,
                'personalized_geometry_hash':self.personalized['geometry_hash'] if self.personalized else None,
                'local_ik_modifications':True,'solver':'scipy_SLSQP','solver_parameters':PARAMETERS,
                'joint_smoother':False,'open_error_reseed':False,
                'observation_failure_reseed':True,
                'joint_limits_source':'calibrated_URDF_with_physical_extension',
                'extension_limits':self.kinematics.extension_limits,
                'native_role':'calibrated_EMF_target_mapping_only','native_angles_used':False,'flat_mounting':self.mount_report}

    def step(self, frame):
        began = time.perf_counter(); mapped = self.mapper.step(frame)
        kin = self.kinematics; first = self.previous is None
        reference = kin.straight.copy() if first else self.previous.copy(); q = reference.copy(); reports = []
        targets = np.asarray(mapped['targets']['positions_local'])
        # The target frame's +x is the desired finger direction and +y its
        # pad normal. The calibrated Moxian tip link declares -x and -y.
        rotations = np.asarray(mapped['targets']['rotations_local']).reshape(5,3,3) @ np.diag([-1.,-1.,1.])
        if getattr(self,'personalized',None):
            data=self.personalized;a=np.asarray(data['raw_to_wrist_rotation']);b=np.asarray(data['raw_to_wrist_translation_m'])
            targets=np.asarray([frame['fingers'][f]['position'] for f in FINGERS])@a.T+b
            rotations=a@Rotation.from_quat([frame['fingers'][f]['quaternion_xyzw'] for f in FINGERS]).as_matrix()
        mapped['targets']['vendor_positions_local']=targets.tolist()
        targets=targets+np.einsum('fij,fj->fi',rotations,self.mounts[:,:3,3]);rotations=rotations@self.mounts[:,:3,:3]
        mapped['targets'].update(positions_local=targets.tolist(),rotations_local=rotations.reshape(5,9).tolist(),
                                 quaternions_local_xyzw=Rotation.from_matrix(rotations).as_quat().tolist(),
                                 orientation_convention='URDF_tip_link',registration=REVISION)
        for f,indices in enumerate(kin.finger_indices):
            posture = PARAMETERS['posture_weight']*(PARAMETERS['thumb_posture_scale'] if f==0 else 1.)
            history = 0. if first else PARAMETERS['history_weight']
            def objective(x, history_weight=None):
                current = q.copy(); current[indices] = x
                tf,jp,jw = self.effector(f,current)
                er,jr = rotation_residual_jacobian(tf[:3,:3],rotations[f],jw)
                ep = tf[:3,3]-targets[f]; dp = x-kin.straight[indices]; dh = x-reference[indices]
                wp = PARAMETERS['position_weight_per_mm']*1000; wr = PARAMETERS['rotation_weight']
                wh=history if history_weight is None else history_weight
                terms = dict(position=float(wp**2*(ep@ep)),rotation=float(wr**2*(er@er)),
                             posture=float(posture*(dp@dp)),history=float(wh*(dh@dh)))
                gradient = wp**2*(jp.T@ep)+wr**2*(jr.T@er)+posture*dp+wh*dh
                return .5*sum(terms.values()),gradient,terms
            start = reference[indices]; start_cost = objective(start)[0]
            result = minimize(lambda x:objective(x)[:2],start,method='SLSQP',jac=True,bounds=kin.bounds[indices],
                              options={'maxiter':PARAMETERS['first_maxiter'] if first else PARAMETERS['later_maxiter'],'ftol':PARAMETERS['ftol']})
            valid = np.isfinite(result.x).all() and objective(result.x)[0]<=start_cost+1e-10
            x = np.clip(result.x,kin.bounds[indices,0],kin.bounds[indices,1]) if valid else start
            # A folded chain can be a stationary constrained local minimum
            # even when a nearly straight observation is reachable. Compare a
            # second physical seed using measurement/posture cost only; stale
            # history must not keep a visibly failed solution at its limits.
            reseeded=False
            current=q.copy();current[indices]=x;tip,_,_=self.effector(f,current)
            error=np.linalg.norm(tip[:3,3]-targets[f])
            straight=kin.straight[indices]
            measurement_cost=objective(x,0.)[0]
            if error>.015 and objective(straight,0.)[0]<measurement_cost*.75:
                retry=minimize(lambda v:objective(v,0.)[:2],straight,method='SLSQP',jac=True,bounds=kin.bounds[indices],
                               options={'maxiter':PARAMETERS['first_maxiter'],'ftol':PARAMETERS['ftol']})
                if np.isfinite(retry.x).all() and objective(retry.x,0.)[0]<measurement_cost*.75:
                    x=np.clip(retry.x,kin.bounds[indices,0],kin.bounds[indices,1]);result=retry;valid=True;reseeded=True
            q[indices] = x; tf,_,_ = self.effector(f,q)
            reports.append(dict(finger=FINGERS[f],status=int(result.status),success=bool(result.success),accepted=bool(valid),
                                iterations=int(result.nit),cost=objective(x,0. if reseeded else None)[0],cost_terms=objective(x,0. if reseeded else None)[2],
                                observation_reseeded=reseeded,
                                tracking_satisfied=bool(np.linalg.norm(tf[:3,3]-targets[f])<=.015),
                                position_error_mm=float(np.linalg.norm(tf[:3,3]-targets[f])*1000),
                                rotation_error_deg=float(np.rad2deg(np.linalg.norm(Rotation.from_matrix(tf[:3,:3].T@rotations[f]).as_rotvec())))))
        pair_report=self.refine_contact_pair(frame,q,targets,rotations,reference) if getattr(self,'personalized',None) else dict(applied=False)
        if pair_report.get('applied'):
            for f,record in enumerate(reports):
                tf,_,_=self.effector(f,q);error=float(np.linalg.norm(tf[:3,3]-targets[f])*1000)
                record.update(position_error_mm=error,tracking_satisfied=error<=15.,rotation_error_deg=float(np.rad2deg(Rotation.from_matrix(tf[:3,:3].T@rotations[f]).magnitude())))
        nodes = kin.nodes(q)
        interval = None if first else frame['timestamp_ms']-self.previous_timestamp
        max_step = 0. if first else float(np.rad2deg(abs(q-self.previous).max()))
        self.previous = q.copy(); self.previous_timestamp = frame['timestamp_ms']; self.frames += 1
        mapped['pose'].update(joint_names=self.joint_names,joint_positions_rad=q.tolist(),
                              positions_local=nodes[:,:3,3].tolist(),orientations_local_xyzw=Rotation.from_matrix(nodes[:,:3,:3]).as_quat().tolist(),
                              ik_solver=REVISION)
        mapped['pose']['thumb_kinematics']=kin.thumb_model
        mapped['pose']['source_urdf_path']=kin.path
        if not getattr(self,'personalized',None):
            mapped['pose']['source_reference']=kin.reference_model
            mapped['pose']['source_reference_hash']=kin.reference_hash
        mapped['pose']['personalized_geometry_hash']=self.personalized['geometry_hash'] if getattr(self,'personalized',None) else None
        if getattr(self,'personalized',None):
            mapped['pose']['personalized_model']={k:self.personalized[k] for k in ('urdf_path','mano_to_wrist_rotation','binding_corrections','geometry_hash')}
        mapped['pose']['tracking_errors']={r['finger']:dict(position_error_mm=r['position_error_mm'],rotation_error_deg=r['rotation_error_deg'],
                                                           satisfied=r['tracking_satisfied']) for r in reports}
        mapped['joint_limit_diagnostics'] = joint_limit_diagnostics(self.joint_names,q,self.joint_limits)
        mapped['continuity'] = dict(engine_frames=self.frames,reused_engine=not first,interval_ms=interval,max_joint_step_deg=max_step)
        mapped['ik_diagnostics'] = dict(algorithm=REVISION,parameters=PARAMETERS,fingers=reports,processing_ms=(time.perf_counter()-began)*1000,
                                        previous_pose_initial_guess=not first,native_angles_used=False,contact_pair=pair_report)
        return mapped

    def refine_contact_pair(self,frame,q,targets,rotations,reference):
        from .motion_capture import bilateral
        candidates=[]
        for f in range(1,5):
            if not bilateral(frame,FINGERS[f]):continue
            a,na,_,_=self.pad(0,q);b,nb,_,_=self.pad(f,q)
            gap=float(np.linalg.norm(a-b));dot=float(na@nb)
            if gap<=.012 and dot<-.1:candidates.append((gap,f))
        if len(candidates)!=1:return dict(applied=False,reason='requires_unambiguous_bilateral_nearby_opposing_pads')
        _,partner=candidates[0];kin=self.kinematics;fingers=(0,partner)
        ids=np.concatenate([kin.finger_indices[f] for f in fingers]);initial=q[ids].copy()
        n0=len(kin.finger_indices[0]);slices=(slice(0,n0),slice(n0,len(ids)))
        def terms(x):
            current=q.copy();current[ids]=x;residual=[];jac=[]
            for f,sl in zip(fingers,slices):
                tf,jp,jw=self.effector(f,current);er,jr=rotation_residual_jacobian(tf[:3,:3],rotations[f],jw)
                for value,gradient in ((300*(tf[:3,3]-targets[f]),300*jp),(PARAMETERS['rotation_weight']*er,PARAMETERS['rotation_weight']*jr)):
                    block=np.zeros((3,len(ids)));block[:,sl]=gradient;residual.append(value);jac.append(block)
            a,na,ja,jna=self.pad(0,current);b,nb,jb,jnb=self.pad(partner,current)
            residual.extend(((a-b)*math.sqrt(40)/.002,(na+nb)*math.sqrt(10)))
            jac.extend((np.concatenate((ja,-jb),axis=1)*math.sqrt(40)/.002,np.concatenate((jna,jnb),axis=1)*math.sqrt(10)))
            residual.append(math.sqrt(20)*(x-reference[ids]));jac.append(np.eye(len(ids))*math.sqrt(20))
            r=np.concatenate(residual);j=np.concatenate(jac)
            return .5*float(r@r),j.T@r
        def separation(x):
            current=q.copy();current[ids]=x
            a,na,_,_=self.pad(0,current);b,nb,_,_=self.pad(partner,current)
            return np.array([(b-a)@na+.002,(a-b)@nb+.002])
        result=minimize(terms,initial,jac=True,method='SLSQP',bounds=kin.bounds[ids],
                        constraints=[dict(type='ineq',fun=separation)],options=dict(maxiter=12,ftol=1e-7))
        accepted=bool(np.isfinite(result.x).all() and terms(result.x)[0]<terms(initial)[0] and separation(result.x).min()>=-1e-6)
        trial=q.copy();trial[ids]=result.x
        for f in fingers:
            before=np.linalg.norm(self.effector(f,q)[0][:3,3]-targets[f]);after=np.linalg.norm(self.effector(f,trial)[0][:3,3]-targets[f])
            accepted=accepted and after<=before*1.05+.0005
        if accepted:q[ids]=result.x
        a,na,_,_=self.pad(0,q);b,nb,_,_=self.pad(partner,q)
        return dict(applied=bool(accepted),partner=FINGERS[partner],gap_mm=float(np.linalg.norm(a-b)*1000),normal_dot=float(na@nb),
                    evidence='bilateral_tactile_and_nearby_opposing_geometry',iterations=int(result.nit))

    def close(self):self.mapper.close()


def skeleton_engine(package_root, config_path, side='left'):
    enabled = any(line.startswith('__moxian_slsqp__:') for line in Path(config_path).read_text().splitlines())
    return (SLSQPSkeleton if enabled else NativeSkeleton)(package_root,config_path,side=side)
