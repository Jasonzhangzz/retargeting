"""v9 source-motion continuity with current source-mesh contact evidence."""
from dataclasses import replace
import math
import numpy as np
import torch
from .retarget import TactileRetarget
from .math import CardSettings

ALGORITHM='card_wuji2_v9_single_frame_pose_aligned_mesh_contact'


class SingleFrameCardRetarget(TactileRetarget):
    requires_future=False

    def __init__(self,*args,settings=None,**kwargs):
        self.mesh_audit_async=True
        settings=settings or replace(CardSettings(),wmotion_joint=300.,motion_sample_fraction=.125,wb=240.,wn=30.)
        super().__init__(*args,settings=settings,**kwargs)
        from .diagnostics import MeshAudit
        self.mesh_audit=MeshAudit(self.mesh_collision,self.device)

    def style_targets(self,vertices,joints):
        target=super().style_targets(vertices,joints)
        # A curved fingertip's averaged normal contains an axial component.
        # A robot pad normal cannot match it without changing bone flexion.
        _,normals=self.extractor.material.taxels(vertices@self.mirror) if hasattr(self.extractor,'material') else (None,None)
        axes=target['directions'][:,-1]
        desired=target['normals'].clone()
        if normals is not None:
            for f in range(5):
                ids=self.extractor.regions==3+3*f
                desired[f]=self.tensor(normals[ids].mean(0))
        desired=desired-(desired*axes).sum(-1,keepdim=True)*axes
        usable=desired.norm(dim=-1,keepdim=True)>1.e-6
        target['normals']=torch.where(usable,torch.nn.functional.normalize(desired,dim=-1),target['normals'])
        return target

    def context(self,vertices,joints,anchors,initial=None):
        context=super().context(vertices,joints,anchors,initial)
        if initial is not None:
            from .robokit import pack
            context['wrist_reference']=pack(initial)
            context['wrist_reference_weights']=(250.**2,5.**2)
            # Relax the accumulated CEM correction towards the current IK.
            # Keep the v9 motion reference, but do not preserve an error forever.
            if context['motion_joint'] is not None:
                dt=max(0.,min(.25,(self.last_seen_timestamp-self.previous_timestamp)/1000.))
                context['motion_joint']=initial[2]+math.exp(-dt/.5)*(context['motion_joint']-initial[2])
        return context

    def evaluate(self,t,r,theta,c,details=False,search=False):
        result=super().evaluate(t,r,theta,c,details,search)
        if c.get('wrist_reference') is None:return result
        from scipy.spatial.transform import Rotation
        ref=c['wrist_reference'];rotation=self.tensor(Rotation.from_rotvec(ref[3:6]).as_matrix())
        weights=c['wrist_reference_weights']
        price=weights[0]*(3-(r*rotation).sum((-1,-2))).clamp_min(0)
        price=price+weights[1]*((t-self.tensor(ref[:3]))*1000).square().sum(-1)
        if not details:return result+price
        cost,detail=result;detail['parts']['style_wrist']=price;detail['groups']['style']+=price
        return cost+price,detail

    def topology(self):
        return dict(super().topology(),algorithm=ALGORITHM,
                    trajectory=dict(horizon=1,future_frames=0,output='current_pose',
                        continuity='v9_source_motion_correction',baseline_commit='98df8ec5',
                        mesh_contacts='CART_directional_witness_zero_gap_soft_closure',post_filter=False,
                        pose_alignment=dict(bone_weight=self.settings.wb,normal_weight=self.settings.wn,
                            normal_policy='physical_taxel_normal_orthogonal_to_distal_axis',
                            wrist_rotation_weight=250.,wrist_translation_weight=5.,history_relaxation_s=.5)))

    def step(self,frame,contact_enabled=True):
        # No future data is read or buffered on this path.
        frame={k:v for k,v in frame.items() if k!='card_future_frames'}
        result=super().step(frame,contact_enabled)
        audit=self.mesh_audit.snapshot(self.session_id)
        self.mesh_audit.submit(result['link_positions_m'],result['link_rotations'],frame['timestamp_ms'],self.session_id)
        result['collision'].update(mesh_audit=audit,mesh_audit_error=self.mesh_audit.error,mesh_audit_async=True)
        result['solver'].update(horizon=1,future_frames=0,source_motion_weight=self.settings.wmotion_joint,
            source_mesh_contacts=self.mesh_contact_evidence(*self.extractor.source(frame)[:2]),
            mesh_contact_policy='CART_directional_witness_zero_gap_soft_closure')
        result.update(algorithm=ALGORITHM,buffer_delay_frames=0,buffer_delay_ms=0.,
                      live_source_timestamp_ms=frame['timestamp_ms'],postprocessing='v9_source_motion_no_output_filter')
        return result

    def close(self):
        self.mesh_audit.close()
