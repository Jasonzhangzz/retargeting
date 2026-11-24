"""CARD tracking/style and wrist smoothness, with direct link-SDF contacts.

This is deliberately a new objective, not claimed equal to CARD's projected
material-point/normal cost. Contacts use fixed source anchor/link identities;
all scalar loss gradients use distance first derivatives, never projections.
"""
from dataclasses import replace
import torch
from torch.nn import functional as F
from .cart_card_cost import CardSurfaceGeometry
from .cart_contacts import link_owner
from .cart_link_sdf import OBJECTIVE
from .cart_wrist import state_velocity,REFERENCE_DT,WRIST_LIMITS


class LinkSDFObjective(CardSurfaceGeometry):
    def __init__(self,hand,collision):
        super().__init__(hand,collision)
        self.settings=replace(self.settings,loss_kind=OBJECTIVE)
        self.collision.card_nearest_surface=False

    def distance(self,local,owner):
        links=torch.full(local.shape[:-1],owner,device=local.device,dtype=torch.long)
        if self.collision.auditing:return self.collision.oracle.label(local.reshape(-1,3),links.reshape(-1))[0].reshape(local.shape[:-1])
        return self.collision.model.distances(local,links)

    def contact_labels(self,rows):
        # Pair labels preserve exact anatomical link identities. Never reduce
        # these to a per-finger flag that exempts unrelated collisions.
        groups={}
        for i,row in enumerate(rows):
            for pair in row['contacts']['self_pairs']:
                owners=tuple(map(link_owner,pair['regions']))
                groups.setdefault(owners,[]).append((i,pair))
        return groups

    def batch(self,q,rows,references=None,targets=None,acceleration=None,previous_rows=None,
              audit=False,normalizers=None,*,collect_metrics=True):
        h=self.hand;s=self.card;size=len(q);frames=h.frames(q);p,r=frames[:2];zero=q.new_zeros(size)
        c={k:torch.stack([row['_card_targets'][k] for row in rows]) for k in rows[0]['_card_targets']}
        positions=p[:,self.region_owners[1:]].reshape(size,5,3,3)
        distal=[3,6,9,12,15];owners=[self.region_owners[i] for i in distal]
        apex=p[:,owners]+(r[:,owners]@self.apices[distal,:,None]).squeeze(-1)
        chains=torch.cat((positions,apex[:,:,None,:]),2);dirs=F.normalize(chains[:,:,1:]-chains[:,:,:-1],dim=-1)
        pad_normal=(r[:,owners]@h.tensor(h.surface.normals[owners])[...,None]).squeeze(-1)
        dot=(dirs*c['directions']).sum(-1).clamp(-1,1)
        parts=dict(style_bone=s.wb**2*(1-dot).sum((-1,-2)),
            style_tip=s.wtip**2*((apex-c['tips'])*1000).square().sum((-1,-2)),
            style_normal=s.wn**2*(1-(pad_normal*c['normals']).sum(-1).clamp(-1,1)).sum(-1),
            smooth_joint=s.wtheta**2*(q[:,:20]-references[:,:20]).square().sum(-1) if references is not None else zero,
            contact_distance=zero.clone(),contact_keep=zero.clone(),contact_self_distance=zero.clone())
        gap=zero.clone();palm=zero.clone();finger=zero.clone();pair_gap=zero.clone();angle=zero.clone();keep_loss=zero.clone()
        count=torch.zeros(size,device=q.device,dtype=torch.long);palm_count=count.clone();finger_count=count.clone();pair_count=count.clone();keep_count=count.clone()
        previous_frames=h.frames(references)[:2] if references is not None and previous_rows and all(v is not None for v in previous_rows) else None
        old_c={k:torch.stack([row['_card_targets'][k] for row in previous_rows]) for k in c} if previous_frames is not None else None
        for region,owner in enumerate(self.region_owners):
            ids=c['mask'][:,region].nonzero().flatten()
            if not len(ids):continue
            target=c['anchor_position'][ids,region]
            local=(r[ids,owner].transpose(-1,-2)@(target-p[ids,owner])[...,None]).squeeze(-1)
            d=self.distance(local,owner)
            rho=1+.5*c['force'][ids,region]/c['force'][ids].amax(-1).clamp_min(1.e-20)
            parts['contact_distance']=parts['contact_distance'].index_add(0,ids,s.wc**2*rho.square()*(d*1000).square())
            if collect_metrics:
                if self.collision.auditing:
                    native_hit,normal,_=h.nearest(local,owner,palmar=True)
                else:
                    with torch.enable_grad():
                        probe=local.detach().requires_grad_();value=self.collision.model(probe,torch.full_like(ids,owner))
                        normal=torch.autograd.grad(value.sum(),probe)[0].detach()
                world_normal=(r[ids,owner]@F.normalize(normal,dim=-1)[...,None]).squeeze(-1)
                degrees=torch.rad2deg(torch.acos((world_normal*c['anchor_normal'][ids,region]).sum(-1).clamp(-1,1)))
                angle=angle.index_copy(0,ids,torch.maximum(angle[ids],degrees.detach()))
            magnitude=(local.detach()-native_hit).norm(dim=-1) if collect_metrics and self.collision.auditing else d.detach().abs()
            gap=gap.index_copy(0,ids,torch.maximum(gap[ids],magnitude))
            count=count.index_add(0,ids,torch.ones_like(ids))
            if region==0:
                palm=palm.index_copy(0,ids,magnitude);palm_count=palm_count.index_add(0,ids,torch.ones_like(ids))
            else:
                finger=finger.index_copy(0,ids,torch.maximum(finger[ids],magnitude));finger_count=finger_count.index_add(0,ids,torch.ones_like(ids))
            if previous_frames is not None:
                op,orr=previous_frames
                old_target=old_c['anchor_position'][ids,region]
                old_local=(orr[ids,owner].transpose(-1,-2)@(old_target-op[ids,owner])[...,None]).squeeze(-1)
                old_d=self.distance(old_local,owner)
                continuous=q.new_tensor([0<row['timestamp_ms']-prev['timestamp_ms']<=250 and row.get('stream_epoch')==prev.get('stream_epoch') for row,prev in zip(rows,previous_rows)],dtype=torch.bool)
                retained=old_c['mask'][ids,region]&continuous[ids]
                keep=(s.wfix**2*((d-old_d)*1000).square()+self.settings.link_continuity_weight**2*(d*1000).square())*rho.square()
                parts['contact_keep']=parts['contact_keep'].index_add(0,ids,torch.where(retained,keep,0.))
                keep_count=keep_count.index_add(0,ids,retained.long())
                keep_loss=keep_loss.index_add(0,ids,torch.where(retained,1-torch.exp(-((d-old_d)/.01).square().detach()),0.))
        for owners,items in self.contact_labels(rows).items():
            ids=torch.tensor([i for i,_ in items],device=q.device)
            anchor=h.tensor([pair['points_m'] for _,pair in items]).mean(1)
            ds=[];native_points=[];native_normals=[]
            for owner in owners:
                local=(r[ids,owner].transpose(-1,-2)@(anchor-p[ids,owner])[...,None]).squeeze(-1)
                ds.append(self.distance(local,owner))
                if collect_metrics and self.collision.auditing:
                    hit,n,_=h.nearest(local,owner,palmar=True)
                    native_points.append((r[ids,owner]@hit[...,None]).squeeze(-1)+p[ids,owner])
                    native_normals.append((r[ids,owner]@n[...,None]).squeeze(-1))
            energy=s.wpinch**2*1.e6*(ds[0].square()+ds[1].square())
            parts['contact_self_distance']=parts['contact_self_distance'].index_add(0,ids,energy)
            # A diagnostic distance bound to the common observed anchor, not a
            # claim of exact closest-surface pair gap.
            if native_points:
                magnitude=(native_points[0]-native_points[1]).norm(dim=-1)
                degrees=torch.rad2deg(torch.acos(-(native_normals[0]*native_normals[1]).sum(-1).clamp(-1,1)))
                angle=angle.scatter_reduce(0,ids,degrees.detach(),reduce='amax',include_self=True)
            else:magnitude=ds[0].detach().abs()+ds[1].detach().abs()
            pair_gap=pair_gap.scatter_reduce(0,ids,magnitude,reduce='amax',include_self=True)
            pair_count=pair_count.index_add(0,ids,torch.ones_like(ids))
            gap=gap.scatter_reduce(0,ids,magnitude,reduce='amax',include_self=True)
            count=count.index_add(0,ids,torch.ones_like(ids))
        native=self.collision.oracle if self.collision.auditing else None
        self_values=self.collision.model.self_terms(q[:,:20],oracle=native)
        parts['collision_self']=s.wself**2*self_values[:,0]
        parts['penetration_cap']=self.settings.penetration_cap_weight**2*self_values[:,1]
        depth=h.penetration(q[:,:20],frames[3],frames[4],dense=True)[0] if self.collision.auditing else self_values[:,2]
        local,links,counts=self.collision.oracle.flat_tactile_queries((p,r),rows)
        if len(local):
            if native is None:d=self.collision.model.distances(local,links)
            else:d=native.label(local.reshape(-1,3),links.reshape(-1))[0].reshape(local.shape[:-1])
            penetration=F.relu(-d*1000)
            per_point=F.relu(penetration-s.allowance_mm).square().sum(-1)
            cap_point=F.relu(penetration-5).square().sum(-1)
            row_ids=torch.arange(size,device=q.device).repeat_interleave(torch.tensor(counts,device=q.device))
            parts['collision_object']=zero.clone().index_add(0,row_ids,s.wobj_pseudo**2*per_point)
            parts['penetration_cap']=parts['penetration_cap']+zero.clone().index_add(0,row_ids,self.settings.penetration_cap_weight**2*cap_point)
            # Max is diagnostics only; the optimized cap is a sum of squared
            # hinges over every point/link, with a continuous first derivative.
            external=zero.clone().scatter_reduce(0,row_ids,penetration.detach().amax(-1)/1000,reduce='amax',include_self=True)
            depth=torch.maximum(depth.detach(),external)
        else:parts['collision_object']=zero.clone()
        velocity=torch.zeros_like(q)
        if references is not None:
            if previous_rows is None or any(v is None for v in previous_rows):raise ValueError('Wrist smoothing requires previous timestamps')
            dt=q.new_tensor([(a['timestamp_ms']-b['timestamp_ms'])/1000 for a,b in zip(rows,previous_rows)])
            if (dt<=0).any() or (dt>.250001).any():raise ValueError('Invalid wrist smoothing interval')
            velocity=state_velocity(q,references,dt)
        acceleration=torch.zeros_like(q) if acceleration is None else acceleration
        parts.update(smooth_translation=s.wt**2*(velocity[:,20:23]*(1000*REFERENCE_DT)).square().sum(-1),
            smooth_rotation=s.womega**2*(velocity[:,23:26]*REFERENCE_DT).square().sum(-1),
            wrist_translation_acceleration=s.wt**2*(acceleration[:,20:23]*(1000*REFERENCE_DT**2)).square().sum(-1),
            wrist_rotation_acceleration=s.womega**2*(acceleration[:,23:26]*REFERENCE_DT**2).square().sum(-1))
        total=sum(v.mean() for v in parts.values());means={k:v.mean() for k,v in parts.items()}
        means.update(direction=means['style_bone'],contact=means['contact_distance']+means['contact_self_distance'],
            smooth=means['smooth_joint']+means['smooth_translation']+means['smooth_rotation'],
            acceleration=acceleration[:,:20].square().mean(),
            wrist_translation_offset_mm=q[:,20:23].norm(dim=-1).mean()*1000,
            wrist_rotation_offset_deg=torch.rad2deg(q[:,23:26].norm(dim=-1)).mean())
        if not collect_metrics:return total,means,[]
        # No contact-normal loss: only distance first derivatives are required.
        # Contact normals are measured separately but do not enter this loss.
        bone=torch.rad2deg(torch.acos(dot.detach())).mean((-1,-2));metrics=[]
        for i in range(size):
            feasible=(gap[i:i+1]<=self.settings.contact_limit_m)&(depth[i:i+1]<=self.settings.penetration_limit_m)&(angle[i:i+1]<=self.settings.normal_deg)
            feasible &= (q[i:i+1,20:].abs()<=q.new_tensor(WRIST_LIMITS)+1.e-6).all(-1)
            metrics.append(dict(query_valid=torch.ones(1,device=q.device,dtype=torch.bool),
                max_gap_m=gap[i:i+1],max_normal_deg=angle[i:i+1],contact_normal_evaluated=True,
                max_penetration_m=depth[i:i+1].detach(),bone_direction_deg=bone[i:i+1],
                contact_count=int(count[i]),feasible=feasible,palm_gap_m=palm[i:i+1],palm_contact_count=int(palm_count[i]),
                finger_gap_m=finger[i:i+1],finger_contact_count=int(finger_count[i]),
                actuated_gap_m=gap[i:i+1],actuated_contact_count=int(count[i]),actuated_contact_satisfied=feasible,
                self_contact_count=int(pair_count[i]),self_contact_gap_m=pair_gap[i:i+1],
                self_contact_satisfied=(pair_gap[i:i+1]<=self.settings.contact_limit_m)&(depth[i:i+1]<=self.settings.penetration_limit_m),
                contact_keep_count=int(keep_count[i]),contact_keep_loss=keep_loss[i:i+1]/keep_count[i].clamp_min(1),groups=[],points=[]))
        return total,means,metrics
