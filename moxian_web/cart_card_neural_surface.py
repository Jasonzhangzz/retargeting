"""CARD expressions with learned closest pad points/normals in the policy graph.

The zero-direction branch of the existing contact MLP predicts the nearest
point on a specified semantic pad. Exact triangle queries are restricted to
supervision generation and independent audit. This approximates the geometry
operator; it does not claim exact equality with the native CARD scalar.
"""
from dataclasses import replace
import torch
from torch.nn import functional as F
from .cart_card_cost import CardCostObjective,selected_surface

VERSION='card_raw_neural_surface_v2'

class NeuralCardCostObjective(CardCostObjective):
    def __init__(self,hand,collision):
        super().__init__(hand,collision)
        self.settings=replace(self.settings,loss_kind=VERSION)
        collision.card_nearest_surface=True
        self.priors=[]
        for region,owner in enumerate(self.region_owners):
            pool=hand.tensor(hand.surface.candidates(owner)['points'])
            n=hand.tensor(hand.surface.normals[owner])
            self.priors.append((n,pool.mean(0)))

    def project_surface(self,query,region):
        if self.collision.auditing:
            local,index=selected_surface(query,self.pad_triangles[region])
            return local,self.pad_normals[region][index]
        return self.collision.contact(query,torch.zeros_like(query),self.region_owners[region],palmar=True)

    def collision_cost(self,q,rows,frames):
        if hasattr(self.collision.model,'q_collision'):
            from .cart_q_collision import collision_cost
            return collision_cost(self,q,rows,frames)
        return super().collision_cost(q,rows,frames)

    def projection_queries(self,frames,targets):
        p,r=frames[:2]
        for region,owner in enumerate(self.region_owners):
            ids=targets['mask'][:,region].nonzero().flatten()
            if not len(ids):
                yield region,ids,None,None
                continue
            target=targets['anchor_position'][ids,region]
            query=(r[ids,owner].transpose(-1,-2)@(target-p[ids,owner])[...,None]).squeeze(-1)
            if region==0:psi=self.palm_psi[None].expand(len(ids),-1)
            else:
                psi=self.hand.tensor(self.hand.surface.axes[owner]*self.hand.surface.lengths[owner])[None]*targets['u'][ids,region,None]
                n,mean=self.priors[region]
                psi=psi+n[None]*((mean[None]-psi)*n).sum(-1)[:,None]
            alpha=.15 if region and (region-1)%3 in (1,2) else 0.
            yield region,ids,query,(query+alpha*psi)/(1+alpha)

    def contact_points(self,frames,targets):
        p,r=frames[:2];positions=[];normals=[];locals_=[]
        for region,ids,query,biased in self.projection_queries(frames,targets):
            if query is None:
                positions.append((ids,None));normals.append(None);locals_.append(None)
                continue
            owner=self.region_owners[region]
            local,normal=self.project_surface(biased,region)
            positions.append((ids,(r[ids,owner]@local[...,None]).squeeze(-1)+p[ids,owner]))
            normals.append((r[ids,owner]@normal[...,None]).squeeze(-1));locals_.append(local)
        return positions,normals,locals_

    @torch.no_grad()
    def collect_geometry_queries(self,q,rows,frames):
        c={k:torch.stack([row['_card_targets'][k] for row in rows]) for k in rows[0]['_card_targets']}
        for region,ids,query,biased in self.projection_queries(frames,c):
            if query is None:continue
            x=torch.cat((query,biased));links=torch.full((len(x),),self.region_owners[region],device=q.device,dtype=torch.long)
            self.collision.collect_contact(x,torch.zeros_like(x),links,torch.ones_like(links,dtype=torch.bool))

    def batch(self,q,rows,references=None,targets=None,acceleration=None,previous_rows=None,audit=False,normalizers=None,*,collect_metrics=True):
        if any('_card_targets' not in r for r in rows):raise ValueError('Missing verified CARD source targets')
        s=self.card;h=self.hand;size=len(q);frames=h.frames(q);p,r=frames[:2]
        c={k:torch.stack([row['_card_targets'][k] for row in rows]) for k in rows[0]['_card_targets']}
        positions=p[:,self.region_owners[1:]].reshape(size,5,3,3)
        distal=[3,6,9,12,15];owners=[self.region_owners[i] for i in distal]
        apex=p[:,owners]+(r[:,owners]@self.apices[distal,:,None]).squeeze(-1)
        chains=torch.cat((positions,apex[:,:,None,:]),2)
        dirs=F.normalize(chains[:,:,1:]-chains[:,:,:-1],dim=-1)
        pad_normals=(r[:,owners]@h.tensor(h.surface.normals[owners])[...,None]).squeeze(-1)
        free=torch.ones((size,5),device=q.device)  # CARD default contact_style_weight=1
        bone_dot=(dirs*c['directions']).sum(-1).clamp(-1,1)
        parts=dict(style_bone=s.wb**2*(1-bone_dot).mul(free[:,:,None]).sum((-1,-2)),
            style_tip=s.wtip**2*((apex-c['tips'])*1000).square().sum(-1).mul(free).sum(-1),
            style_normal=s.wn**2*(1-(pad_normals*c['normals']).sum(-1).clamp(-1,1)).mul(free).sum(-1),
            smooth_joint=s.wtheta**2*(q-references).square().sum(-1) if references is not None else q.new_zeros(size))
        selected,nn,local=self.contact_points(frames,c)
        old=None
        if references is not None and previous_rows and all(row is not None for row in previous_rows):
            old_c={k:torch.stack([row['_card_targets'][k] for row in previous_rows]) for k in c}
            _,_,old=self.contact_points(h.frames(references),old_c)
        for k in ('contact_position','contact_direction','contact_fix'):parts[k]=q.new_zeros(size)
        parts['link_continuity']=q.new_zeros(size)
        max_gap=q.new_zeros(size);max_angle=q.new_zeros(size);keep_cost=q.new_zeros(size);keep_counts=[0]*size
        for region,((ids,pos),normal,material) in enumerate(zip(selected,nn,local)):
            if pos is None:continue
            rho=1+.5*c['force'][ids,region]/c['force'][ids].amax(-1).clamp_min(1.e-20)
            error=(pos-c['anchor_position'][ids,region]).square().sum(-1)
            dot=(normal*c['anchor_normal'][ids,region]).sum(-1).clamp(-1,1)
            parts['contact_position']=parts['contact_position'].index_add(0,ids,rho.square()*s.wc**2*1.e6*error)
            parts['contact_direction']=parts['contact_direction'].index_add(0,ids,rho.square()*s.wc**2*s.direction_price*s.radius_mm**2*(1-dot))
            max_gap=max_gap.index_copy(0,ids,torch.maximum(max_gap[ids],error.detach().sqrt()))
            max_angle=max_angle.index_copy(0,ids,torch.maximum(max_angle[ids],torch.rad2deg(torch.acos(dot.detach()))))
            if old is not None and old[region] is not None:
                # Dense correspondence through row indices; no detached earlier
                # predicted state, so retention also preserves the BPTT path.
                old_ids=old_c['mask'][:,region].nonzero().flatten();dense=q.new_zeros((size,3)).index_copy(0,old_ids,old[region])
                continuous=q.new_tensor([0<row['timestamp_ms']-prev['timestamp_ms']<=250 and row.get('stream_epoch')==prev.get('stream_epoch')
                    for row,prev in zip(rows,previous_rows)],dtype=torch.bool)
                retained=old_c['mask'][ids,region]&continuous[ids]
                delta=(material-dense[ids]).square().sum(-1)
                fix=torch.where(retained,rho.square()*s.wfix**2*1.e6*delta,0.)
                parts['contact_fix']=parts['contact_fix'].index_add(0,ids,fix)
                # Keep the same semantic link in contact, while allowing the
                # material CoP to slide. Use its closest physical pad surface,
                # independently of CARD's material fix and segment-u prior.
                owner=self.region_owners[region]
                query=(r[ids,owner].transpose(-1,-2)@(c['anchor_position'][ids,region]-p[ids,owner])[...,None]).squeeze(-1)
                hit,_=self.project_surface(query,region)
                gap_mm=(query-hit).norm(dim=-1)*1000
                link=self.settings.link_continuity_weight**2*rho.square()*gap_mm.square()
                parts['link_continuity']=parts['link_continuity'].index_add(0,ids,torch.where(retained,link,0.))
                keep_cost=keep_cost.index_add(0,ids,torch.where(retained,1-torch.exp(-delta/.01**2),0.))
                for i in ids[retained].tolist():keep_counts[i]+=1
        collision=self.collision_cost(q,rows,frames)
        parts['collision_self'],parts['collision_object'],depth=collision[:3]
        for name in ('smooth_translation','smooth_rotation','motion_joint','contact_self_closure',
                     'contact_force_vector','contact_normal_force','collision_mesh'):
            parts[name]=q.new_zeros(size)  # fixed wrist and CARD default profile
        parts['penetration_cap']=collision[3] if len(collision)==4 else self.settings.penetration_cap_weight**2*((depth-self.settings.penetration_limit_m)*1000).clamp_min(0).square()
        total=sum(parts.values())
        means={k:v.mean() for k,v in parts.items()}
        # Familiar chart aliases; sum the primary cost parts, not their aliases.
        means.update(direction=means['style_bone'],smooth=means['smooth_joint'],contact=means['contact_position'],
            normal=means['contact_direction'],penetration=means['collision_self']+means['collision_object'],contact_keep=means['contact_fix'])
        if acceleration is not None:means['acceleration']=acceleration.square().mean()  # monitor only
        if not collect_metrics:return total.mean(),means,[]
        valid=torch.ones(size,device=q.device,dtype=torch.bool)
        if audit:depth,valid=self.collision.audit(q,rows,frames)
        feasible=valid&(depth<=self.settings.penetration_limit_m)&(max_gap<=self.settings.contact_limit_m)&(max_angle<=self.settings.normal_deg)
        feasible=feasible&(q>=h.low-1.e-6).all(-1)&(q<=h.high+1.e-6).all(-1)&torch.isfinite(total)
        dir_error=torch.rad2deg(torch.acos(bone_dot.detach())).mean((-1,-2))
        metrics=[dict(feasible=feasible[i:i+1],contact_count=int(c['mask'][i].sum()),max_gap_m=max_gap[i:i+1],
            max_normal_deg=max_angle[i:i+1],max_penetration_m=depth[i:i+1],bone_direction_deg=dir_error[i:i+1],
            query_valid=valid[i:i+1],contact_query_valid=valid[i:i+1],unresolved_contact_count=q.new_zeros(1,dtype=torch.long),
            points=[],groups={},contact_keep_count=keep_counts[i],
            contact_keep_loss=keep_cost[i:i+1]/max(1,keep_counts[i])) for i in range(size)]
        return total.mean(),means,metrics
