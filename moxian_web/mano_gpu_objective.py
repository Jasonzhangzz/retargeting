"""CUDA calibration buffer and vectorized, fixed-material contact objective."""
import math,time
import numpy as np
import torch

from .mano_objective import ManoObjective
from .mano_gpu_collision import BatchedSurfaceCollision
from .motion_capture import confirmed_touch,bilateral
from .contact import PATCH_IDS,FINGERS
from .card_geometry import contact_weights
from .calibration_targets import pinch_patches
from scripts.calibrate_wuji_mano import TIPS,CHAINS
from .mano_contact_loss import vertex_normals,normal_residual,normal_error_deg


def _refresh_surface(objective,x):
    result=ManoObjective.surface(objective,x)
    return result[1],result[2]

_compiled_refresh=None
_warmed_refresh=set()


def tip_emf_directions(vertices,joints,rest,rest_joints,distal_ids,tips,raw_rotation,alignment,mount_rotation):
    """Actual mesh tip tangent vs EMF, after sensor-to-distal frame conversion."""
    local=torch.nn.functional.normalize(rest[tips]-rest_joints[distal_ids],dim=-1) if rest.ndim==2 else torch.nn.functional.normalize(rest[:,tips]-rest_joints[:,distal_ids],dim=-1)
    actual=torch.nn.functional.normalize(vertices[:,tips]-joints[:,distal_ids],dim=-1)
    emf_distal=(alignment@raw_rotation)@mount_rotation.transpose(-1,-2)
    desired=torch.nn.functional.normalize((emf_distal@(local[None,:,:,None] if local.ndim==2 else local[...,None])).squeeze(-1),dim=-1)
    return actual,desired


class GPUManoObjective(ManoObjective):
    def __init__(self,model,rows,poses,roots,gpu_collision=None,**kwargs):
        if model.device.type!='cuda':raise ValueError('GPU 标定需要 CUDA')
        self.gpu_collision=gpu_collision or BatchedSurfaceCollision(model,len(rows))
        self._contact_buffer=None;self._surface_cache=None;self.feature_seed_numpy=np.asarray(poses)
        self.distal_ids=torch.tensor([c[-1] for c in CHAINS],device=model.device,dtype=torch.long)
        kwargs.pop('collision',None)
        super().__init__(model,rows,poses,roots,collision=self.gpu_collision.reference,**kwargs)
        self.collision=self.gpu_collision.reference
        finger_for_joint=[next(f for f,c in enumerate(CHAINS) if joint+1 in c) for joint in range(15)]
        self.policy_mask=self.t(self.mask_np)[:,finger_for_joint,None]*self.free
        if not self.training:self.weights=self.t(self.mask_np)/self.t(self.mask_np).sum(1,keepdim=True).clamp_min(1)/len(self.rows)

    def unpack(self,x):
        from .card_math import exp_so3
        if self.shared:
            beta=x[:10];mp=self.mp0+.01*x[10:25].reshape(5,3)
            mr=self.mr0@exp_so3(x[25:40].reshape(5,3))
            align=exp_so3(x[40:43])@self.align0;trans=self.trans0+.01*x[43:46]
        else:beta,mp,mr,align,trans=self.beta0,self.mp0,self.mr0,self.align0,self.trans0
        if not hasattr(self,'free_indices'):self.free_indices=self.free.flatten().nonzero().flatten()
        angles=self.seed_angles.flatten().index_copy(0,self.free_indices,x[self.shared_size:]).reshape_as(self.seed_angles)
        return beta,mp,mr,align,trans,angles

    def surface(self,x):
        return self._surface_cache if self._surface_cache is not None else super().surface(x)

    def refresh(self,x,contacts=False):
        global _compiled_refresh
        with torch.no_grad():
            if not hasattr(self,'free_indices'):self.free_indices=self.free.flatten().nonzero().flatten()
            if _compiled_refresh is None:_compiled_refresh=torch.compile(_refresh_surface,options={'triton.cudagraphs':False},dynamic=False)
            key=(str(x.device),len(self.rows),self.shared,self.personalized,len(self.free_indices))
            began=time.monotonic();v,j=_compiled_refresh(self,x)
            if key not in _warmed_refresh:
                torch.cuda.synchronize();self.model._gpu_surface_compile_s=getattr(self.model,'_gpu_surface_compile_s',0.)+time.monotonic()-began
                _warmed_refresh.add(key)
            self.gpu_witness=self.gpu_collision.query(v)
            self.witnesses=[];self.collision_reports=[];self.contacts=[];self.contact_anchors=[]
            if self._contact_buffer is None:
                self.together=[];self.side_missing=[]
                self._build_contacts(v,j)

    def _build_contacts(self,v,j):
        if self.training and self.rows[0].get('capture_protocol')=='contact_motion_v3':
            self._build_region_contacts();return
        topology=self.model.topology()
        sid_faces=np.asarray(topology['sid_face_ids']);sid_bc=np.asarray(topology['sid_barycentric'])
        patches=np.asarray(topology['sid_patch_indices'])
        ids=[];weights=[];self.contact_records=[]
        for i,row in enumerate(self.rows):
            stage=row.get('stage','');partner=stage[6:] if stage.startswith('pinch_') else None
            if self.training:
                if not partner or not confirmed_touch(row) or not bilateral(row,partner):continue
                desired=pinch_patches(stage)
                if not desired:continue
                data=contact_weights(row.get('contact') or {},patches,row['timestamp_ms'],row.get('tactile_timestamp_ms',-1e20))
                if not all(data['active'][p] for p in desired):continue
                ids.append(i);weights.append(data['center'][list(desired)])
                self.contact_records.append(dict(frame=i,stage=stage,patches=list(desired),evidence='confirmed_calibration_gesture'))
            else:
                if not (row.get('contact') or {}).get('ready'):continue
                # At runtime labels must not create or identify contact partners.
                from .tactile_pose import frame_constraints
                ordered=torch.cat((j[i],v[i,list(TIPS)]),0)[self.model.ordered]
                _,pairs=frame_constraints(self.model,row,v[i].cpu().numpy(),ordered.cpu().numpy(),guided=False)
                data=contact_weights(row.get('contact') or {},patches,row['timestamp_ms'],row.get('tactile_timestamp_ms',-1e20))
                for pair in pairs:
                    desired=pair['patches']
                    if all(data['active'][p] for p in desired):
                        ids.append(i);weights.append(data['center'][desired]);self.contact_records.append(dict(frame=i,patches=desired,evidence='bilateral_tactile_geometry_association'))
        count=len(ids)
        if not self.training:
            capacity=len(self.rows)*10
            if count>capacity:raise ValueError('GPU 自接触配对缓冲区溢出')
            ids+= [0]*(capacity-count);weights += [np.zeros((2,len(sid_faces)))]*(capacity-count)
        self.contact_valid=torch.arange(len(ids),device=self.model.device)<count
        self.contact_rows=self.t(ids).long()
        self.contact_fingers=self.t([[0,FINGERS.index(self.rows[i]['stage'][6:])] if self.training else [0,0] for i in ids]).long().reshape(-1,2)
        self.tip_emf_valid=self.t(self.mask_np)[self.contact_rows[:,None],self.contact_fingers]
        counts={stage:sum(self.rows[i]['stage']==stage for i in ids[:count]) for stage in dict.fromkeys(self.rows[i]['stage'] for i in ids[:count])}
        self.contact_action_weight=self.t([1/counts[self.rows[i]['stage']] for i in ids[:count]]+[0.]*(len(ids)-count))
        self.contact_counts=self.t(np.bincount(ids[:count],minlength=len(self.rows)))[self.contact_rows].clamp_min(1)
        self.taxel_faces=self.t(np.asarray(topology['faces'])[sid_faces]).long()
        self.surface_faces=self.t(topology['faces']).long()
        self.taxel_bc=self.t(sid_bc)
        self._contact_buffer=self.t(np.asarray(weights).reshape(-1,2,len(sid_faces)))
        # Side contacts are fixed material coordinates selected once, never
        # rebound as shape changes. They retain the existing original-skin check.
        if self.training:
            from .side_contact import side_material_pairs
            for i,row in enumerate(self.rows):
                if row.get('stage')!='together':continue
                for f in range(0 if row.get('thumb_index_side_contact') is True else 1,4):
                    try:self.together.extend((i,p) for p in side_material_pairs(self.model,v[i].cpu().numpy(),j[i].cpu().numpy(),finger_pairs=[f]))
                    except ValueError as exc:self.side_missing.append(dict(frame=i,reason=str(exc)))
        self.together_rows=self.t([i for i,p in self.together]).long()
        self.together_ids=self.t(np.asarray([self.model.faces[[p['face_a'],p['face_b']]] for i,p in self.together],dtype=np.int64).reshape(-1,2,3)).long()
        self.together_bc=self.t(np.asarray([[p['barycentric_a'],p['barycentric_b']] for i,p in self.together]).reshape(-1,2,3))
        self.together=[]

    def _build_region_contacts(self):
        from .contact_protocol import region_finger
        records=[]
        for i,row in enumerate(self.rows):
            if not confirmed_touch(row):continue
            if not row.get('contact_edges'):raise ValueError('接触模板未构建；禁止跳过新动作损失')
            for edge in row['contact_edges']:records.append({**edge,'frame':i,'stage':row['stage'],'event':row['contact_event_id']})
        self.contact_records=records
        self.region_contact=True
        self.contact_rows=self.t([r['frame'] for r in records]).long()
        self.contact_valid=torch.ones(len(records),device=self.model.device,dtype=torch.bool)
        self.region_triangles=self.t(np.asarray([[self.model.faces[a['face']] for a in r['anchors']] for r in records],dtype=np.int64).reshape(-1,2,3)).long()
        self.region_bc=self.t(np.asarray([[a['barycentric'] for a in r['anchors']] for r in records]).reshape(-1,2,3))
        self.surface_faces=self.t(self.model.faces).long()
        fingers=[[region_finger(region) for region in r['regions']] for r in records]
        self.contact_fingers=self.t([[f if f is not None else 0 for f in pair] for pair in fingers]).long().reshape(-1,2)
        self.tip_emf_valid=self.t([[float(f is not None and region.endswith('_distal'))*self.mask_np[r['frame'],f if f is not None else 0] for f,region in zip(pair,r['regions'])] for pair,r in zip(fingers,records)]).reshape(-1,2)
        events={stage:{r['event'] for r in records if r['stage']==stage} for stage in {r['stage'] for r in records}}
        totals={event:sum(r.get('weight',1.) for r in records if r['event']==event) for group in events.values() for event in group}
        self.contact_action_weight=self.t([r.get('weight',1.)/totals[r['event']]/len(events[r['stage']]) for r in records])
        self.contact_counts=torch.ones(len(records),device=self.model.device)
        self._contact_buffer=torch.empty(0,device=self.model.device)
        self.together=[];self.side_missing=[]
        self.together_rows=self.t([]).long()

    def material_contacts(self,vertices):
        if getattr(self,'region_contact',False):
            tri=vertices[self.contact_rows[:,None,None],self.region_triangles]
            p=(tri*self.region_bc[...,None]).sum(-2)
            smooth=vertex_normals(vertices,self.surface_faces)
            n=torch.nn.functional.normalize((smooth[self.contact_rows[:,None,None],self.region_triangles]*self.region_bc[...,None]).sum(-2),dim=-1)
            return p,n
        tri=vertices[self.contact_rows[:,None,None],self.taxel_faces[None]]
        points=(tri*self.taxel_bc[None,:,:,None]).sum(-2)
        if self.training:
            smooth=vertex_normals(vertices,self.surface_faces)
            normals=torch.nn.functional.normalize((smooth[self.contact_rows[:,None,None],self.taxel_faces[None]]*self.taxel_bc[None,:,:,None]).sum(-2),dim=-1)
        else:normals=torch.nn.functional.normalize(torch.linalg.cross(tri[:,:,1]-tri[:,:,0],tri[:,:,2]-tri[:,:,0]),dim=-1)
        p=torch.einsum('nks,nsd->nkd',self._contact_buffer,points)
        n=torch.nn.functional.normalize(torch.einsum('nks,nsd->nkd',self._contact_buffer,normals),dim=-1)
        return p,n

    def residual(self,x,record=False,measurement_only=False):
        if record:self.last_training_metrics={}
        surface=self.surface(x);self._surface_cache=surface
        try:base=super().residual(x,record=record,measurement_only=measurement_only)
        finally:self._surface_cache=None
        v=surface[1];parts=[base];terms={}
        active=self.t([not hasattr(self,'phase_stages') or row['stage'] in self.phase_stages for row in self.rows])
        if hasattr(self,'phase_active'):active=active*self.phase_active
        if not measurement_only:
            residual=self.gpu_collision.residual(v,self.gpu_witness)
            count=self.gpu_witness['mask'].sum(1).clamp_min(1)
            collision=residual/(count[:,None]*active.sum().clamp_min(1)).sqrt()*active[:,None]*math.sqrt(self.settings.collision_weight)
            maximum=torch.linalg.vector_norm(residual.flatten(1),ord=8,dim=1)*active*math.sqrt(4*self.settings.collision_weight)/active.sum().clamp_min(1).sqrt()
            parts.extend((collision.flatten(),maximum));terms['collision']=collision;terms['collision_max']=maximum
        if len(self.contact_rows):
            p,n=self.material_contacts(v);mask=active[self.contact_rows]*self.contact_valid
            if self.training:
                weights=mask*self.contact_action_weight
                scale=(weights/(3*weights.sum().clamp_min(1e-12))).sqrt()[:,None]
            else:scale=mask[:,None]/(self.contact_counts*3*len(self.rows)).sqrt()[:,None]
            position=(p[:,0]-p[:,1])/(self.settings.calibration_contact_distance_scale_m if self.training else .002)*scale*math.sqrt(self.settings.calibration_contact_position_weight if self.training else 20)
            direction=normal_residual(n[:,0],n[:,1],self.settings.calibration_contact_normal_tolerance_deg,self.settings.calibration_contact_normal_scale_deg) if self.training else n[:,0]+n[:,1]
            normal=direction*scale*math.sqrt(self.settings.calibration_contact_normal_weight if self.training else 10)
            parts.extend((position.flatten(),normal.flatten()));terms.update(contact_position=position,contact_normal=normal)
            if self.training:
                beta,mp,mr,align,trans,_=surface[4]
                geometry=x[46:self.shared_size] if self.shared and self.personalized else self.geometry0
                rest,rest_j=self.contact_rest(beta,geometry)
                actual,desired=tip_emf_directions(v,surface[2],rest,rest_j,self.distal_ids,list(TIPS),self.rawr,align,mr)
                a=actual[self.contact_rows[:,None],self.contact_fingers]
                b=desired[self.contact_rows[:,None],self.contact_fingers]
                # Two fingertip tangents per contact, same action normalization
                # as contact position, and no direction equality on approach.
                tip=(a-b)/self.settings.orientation_scale_rad*scale[:,None,:]*self.tip_emf_valid[...,None]/math.sqrt(2)*math.sqrt(self.settings.calibration_tip_emf_orientation_weight)
                parts.append(tip.flatten());terms['contact_tip_emf_orientation']=tip
                if record:
                    angles=torch.rad2deg(torch.acos((a.detach()*b.detach()).sum(-1).clamp(-1,1)))
                    measured_gap=(p.detach()[:,0]-p.detach()[:,1]).norm(dim=-1)
                    denominator=mask.sum().clamp_min(1)
                    normal_angles=normal_error_deg(n.detach()[:,0],n.detach()[:,1])
                    self.last_training_metrics=dict(
                        contact_gap_mm=float((measured_gap*mask).sum()/denominator*1000),
                        contact_max_gap_mm=float(torch.where(mask.bool(),measured_gap,0).amax()*1000),
                        tip_emf_angle_deg=float((angles*mask[:,None]*self.tip_emf_valid).sum()/(mask[:,None]*self.tip_emf_valid).sum().clamp_min(1)),
                        contact_evidence_frames=int(mask.sum()),
                        contact_normal_error_deg=float((normal_angles*mask).sum()/denominator),
                        contact_normal_max_error_deg=float(torch.where(mask.bool(),normal_angles,0).amax()),
                        contact_normal_within_tolerance_fraction=float(((normal_angles<=self.settings.calibration_contact_normal_tolerance_deg)*mask).sum()/denominator))
                    if not bool(mask.any()):self.last_training_metrics=dict(contact_gap_mm=None,contact_max_gap_mm=None,tip_emf_angle_deg=None,contact_evidence_frames=0,contact_normal_error_deg=None,contact_normal_max_error_deg=None,contact_normal_within_tolerance_fraction=None)
                    if not bool((mask[:,None]*self.tip_emf_valid).any()):self.last_training_metrics['tip_emf_angle_deg']=None
        if len(self.together_rows) and not measurement_only:
            tri=v[self.together_rows[:,None,None],self.together_ids]
            point=(tri*self.together_bc[...,None]).sum(-2)
            normal=torch.nn.functional.normalize(torch.linalg.cross(tri[:,:,1]-tri[:,:,0],tri[:,:,2]-tri[:,:,0]),dim=-1)
            mask=active[self.together_rows,None]
            gap=(point[:,0]-point[:,1])/.002*mask*8/math.sqrt(3*len(self.together_rows))
            direction=(normal[:,0]+normal[:,1])*mask/math.sqrt(3*len(self.together_rows))
            parts.extend((gap.flatten(),direction.flatten()));terms.update(together_gap=gap,together_normal=direction)
        if hasattr(self,'local_reference') and not measurement_only:
            rest,_=self.model.model.official_rest(x[:10],x[46:89])
            anchor=(rest-self.local_reference)/.0005*self.local_vertex_weight.sqrt()[:,None]
            anchor=anchor*3/math.sqrt(anchor.numel())
            parts.append(anchor.flatten());terms['inactive_surface_anchor']=anchor
        if record:
            if torch.is_grad_enabled():self.last_contact_residuals={k:t for k,t in terms.items() if k in ('contact_position','contact_normal','contact_tip_emf_orientation')}
            self.last_terms.update({k:float(t.detach().square().sum()) for k,t in terms.items()})
            if not len(self.contact_rows):self.last_training_metrics={}
            beta,mp,mr,align,trans,_=surface[4]
            predicted=surface[2][:,self.distal_ids]+(surface[3][:,self.distal_ids]@mp[None,:,:,None]).squeeze(-1)
            error=(predicted-(self.rawp@align.T+trans)).square().sum(-1)
            self.last_training_metrics['sensor_rmse_mm']=float((error.detach()*self.weights).sum().div(self.weights.sum().clamp_min(1e-12)).sqrt()*1000) if self.observations_enabled and bool(self.weights.sum()>0) else None
        return torch.cat(parts)

    def contact_rest(self,beta,geometry):
        return self.model.model.official_rest(beta,geometry)
