"""Instantaneous gesture constraints, before stability/collection timers.

The four pinches retain Wuji's 50mm RX-distance gate. Additional Moxian
gestures use frozen eight-action anatomical proxies; these are deliberately
coarse collection checks, separate from the final 3/5mm surface acceptance.
No trajectory, excursion, repetition or fitted candidate enters this module.
"""
from __future__ import annotations
import copy
import numpy as np
from scipy.spatial.transform import Rotation
from .bone_extrinsics import FINGERS,definition_from_surface
from .calibration_targets import target,DEFINITION

REVISION='moxian_static_pose_constraints_v1'
LIMITS=dict(DEFINITION['protocols']['contact_static_v5']['pose_constraints'])


class StaticPoseValidator:
    def __init__(self,spec):
        if spec.get('revision')!=REVISION:raise ValueError('静态姿态检验版本不匹配')
        self.spec=copy.deepcopy(spec);self.limits=spec['limits']
        self.tx=np.asarray(spec['T_wrist_TX']);self.rx=np.asarray(spec['T_distal_RX'])
        self.patches=spec['patches'];self.mcp=np.asarray(spec['mcp_positions_m'])
        self.flat_tips=np.asarray(spec['flat_tip_positions_m']);self.tip_offsets=np.asarray(spec['tip_offsets_m'])
        self.normal=np.asarray(spec['palm_normal']);self.forward=np.asarray(spec['palm_forward'])
        self.flat_directions=np.asarray(spec['flat_directions'])

    @classmethod
    def from_reference(cls,reference,initial):
        import torch
        m=reference.model;opened=[r for r in initial['rows'] if r['stage']=='open']
        if not opened:raise ValueError('静态姿态检验缺少八动作展平初始化')
        with torch.no_grad():
            q=m.angles(opened).to(m.q0.dtype).mean(0,keepdim=True);v,j,r=m(q)
            names=list(m.patch_anchors);p,n=m.anchors(v,[m.patch_anchors[name] for name in names])
        points=p[0].cpu().numpy();normals=n[0].cpu().numpy();j=j[0].cpu().numpy();r=r[0].cpu().numpy()
        distal=r[m.distal_bodies];origin=j[m.distal_bodies];tips=j[m.tip_bodies]
        mcps=[m.root_bodies[0]]+[m.body_names.index(m.side+'_'+f+'_proximal') for f in ('index','middle','ring','little')]
        mcp=j[mcps];direction=tips-origin;direction/=np.linalg.norm(direction,axis=-1,keepdims=True)
        forward=direction[1:].mean(0);forward/=np.linalg.norm(forward)
        normal=np.cross(forward,mcp[-1]-mcp[1]);normal/=np.linalg.norm(normal)
        patches={}
        for name,pp,nn in zip(names,points,normals):
            if name.startswith('palm_'):patches[name]=dict(position_m=pp.tolist(),normal=nn.tolist())
            else:
                f=FINGERS.index(name.split('_')[0])
                # Folded gestures explicitly target the *extended* proximal
                # pad. Palm and proximal targets are fixed in this reference.
                if name.endswith('_distal'):
                    patches[name]=dict(finger=f,offset_m=(distal[f].T@(pp-origin[f])).tolist(),normal_local=(distal[f].T@nn).tolist())
                else:patches[name]=dict(position_m=pp.tolist(),normal=nn.tolist())
        definition=definition_from_surface(m)
        return cls(dict(revision=REVISION,limits=LIMITS,reference_id=reference.fingerprint,
            source_calibration_fingerprint=initial.get('calibration_fingerprint'),
            T_wrist_TX=definition['T_wrist_TX'],T_distal_RX=definition['T_distal_RX'],patches=patches,
            mcp_positions_m=mcp.tolist(),flat_tip_positions_m=tips.tolist(),
            tip_offsets_m=np.einsum('fji,fj->fi',distal,tips-origin).tolist(),
            direction_local=np.einsum('fji,fj->fi',distal,direction).tolist(),
            flat_directions=direction.tolist(),palm_normal=normal.tolist(),palm_forward=forward.tolist()))

    def __call__(self,row):
        spec=target(row['stage']);limits=self.limits;failures=[];checks=[]
        raw=np.array([row['fingers'][f]['position'] for f in FINGERS])
        rotation=self.tx[:3,:3]@Rotation.from_quat([row['fingers'][f]['quaternion_xyzw'] for f in FINGERS]).as_matrix()
        dr=rotation@self.rx[:,:3,:3].transpose(0,2,1)
        origin=raw@self.tx[:3,:3].T+self.tx[:3,3]-np.einsum('fij,fj->fi',dr,self.rx[:,:3,3])
        def check(label,value,threshold,maximum=True,required=True):
            good=bool(np.isfinite(value) and (value<=threshold if maximum else value>=threshold))
            checks.append(dict(label=label,value=float(value) if np.isfinite(value) else None,threshold=float(threshold),passed=good,required=required))
            if not good and required:failures.append(f'{label}：{value:.1f}，要求'+('≤' if maximum else '≥')+f'{threshold:.1f}')
        def patch(name):
            a=self.patches[name]
            if 'finger' not in a:return np.asarray(a['position_m']),np.asarray(a['normal'])
            f=a['finger'];return origin[f]+dr[f]@a['offset_m'],dr[f]@a['normal_local']
        if spec['kind']=='noncontact_motion':
            tips=origin+np.einsum('fij,fj->fi',dr,self.tip_offsets)
            directions=np.einsum('fij,fj->fi',dr,np.asarray(self.spec['direction_local']))
            for f in range(5):
                flat=self.flat_tips[f]-self.mcp[f];length=np.linalg.norm(flat)
                ratio=float((tips[f]-self.mcp[f])@(flat/max(length,1e-8))/max(length,1e-8))
                check(FINGERS[f]+'伸直距离比例',ratio,limits['straight_reach_ratio'],False)
                if f:
                    plane=float(np.rad2deg(np.arcsin(np.clip(abs(directions[f]@self.normal),0,1))))
                    check(FINGERS[f]+'离掌面角度°',plane,limits['straight_plane_deg'])
                    check(FINGERS[f]+'朝向掌前',float(directions[f]@self.forward),.5,False)
            projected=directions[1:]-(directions[1:]@self.normal)[:,None]*self.normal
            projected/=np.maximum(np.linalg.norm(projected,axis=-1,keepdims=True),1e-8)
            angle=float(np.rad2deg(np.arccos(np.clip((projected[:-1]*projected[1:]).sum(-1),-1,1))).mean())
            if row.get('static_endpoint')=='spread':check('相邻四指张开角度°',angle,limits['spread_direction_deg'],False)
            elif row.get('static_endpoint')=='together':check('相邻四指并拢角度°',angle,limits['together_direction_deg'])
            else:failures.append('请选择张开或并拢静态端点')
        elif row['stage']=='contact_five':
            # Five fingertips form a connected cluster; the little finger need
            # not reach the thumb directly. Use raw RX distances to avoid
            # rejecting a real pose due to uncalibrated mesh mounting offsets.
            distances=np.linalg.norm(raw[:,None]-raw[None,:],axis=-1)
            reached={0};bottleneck=0.
            while len(reached)<5:
                distance,i=min((distances[j,i],i) for j in reached for i in range(5) if i not in reached)
                bottleneck=max(bottleneck,float(distance));reached.add(i)
            check('五指接近距离mm',bottleneck*1000,limits['five_cluster_distance_m']*1000)
            tips=origin+np.einsum('fij,fj->fi',dr,self.tip_offsets)
            flat=self.flat_tips[1:]-self.mcp[1:]
            ratio=float(np.mean(np.sum((tips[1:]-self.mcp[1:])*flat,axis=-1)/np.maximum(np.sum(flat*flat,axis=-1),1e-8)))
            check('四指收拢比例',ratio,limits['five_curl_ratio'])
        else:
            edges=spec.get('edges',[])
            cluster=spec['kind']=='distal_cluster'
            if cluster:edges=[dict(regions=[spec['patches'][0],name]) for name in spec['patches'][1:]]
            for edge in edges:
                if not edge.get('required',True):continue
                a,b=edge['regions'];pa,na=patch(a);pb,nb=patch(b)
                label=a+' / '+b
                palm=a.startswith('palm_') or b.startswith('palm_')
                pinch=row['stage'].startswith('contact_pair_')
                check(label+'静态区域距离mm',float(np.linalg.norm(pa-pb)*1000),1000*(limits['palm_distance_m'] if palm else limits['pad_distance_m']),required=not pinch)
                angle=float(np.rad2deg(np.arccos(np.clip(na@nb,-1,1))))
                check(label+'指腹相向角度°',angle,limits['cluster_normal_deg'] if cluster else limits['opposing_normal_deg'],False,required=not pinch and not palm)
                if a.endswith('_distal') and b.endswith('_distal'):
                    i=FINGERS.index(a.split('_')[0]);j=FINGERS.index(b.split('_')[0])
                    check(label+'RX距离mm',float(np.linalg.norm(raw[i]-raw[j])*1000),1000*limits['rx_distance_m'])
            if not edges:failures.append('缺少静态动作检验定义')
        return dict(revision=REVISION,accepted=not failures,constraints_ok=not failures,
            failures=failures,checks=checks,evidence_source='frozen_eight_action_anatomical_proxy')
