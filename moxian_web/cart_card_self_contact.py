"""CARD neural surface objective with explicit source-penetration pair closure."""
import torch
from torch.nn import functional as F
from .cart_card_neural_surface import NeuralCardCostObjective
from .cart_contacts import link_owner
from .cart_q_collision import GROUPS


class SelfContactCardCostObjective(NeuralCardCostObjective):
    def contact_collision_groups(self,rows):
        mask=torch.zeros((len(rows),len(GROUPS)),device=self.hand.device,dtype=torch.bool)
        for i,row in enumerate(rows):
            for pair in row['contacts']['self_pairs']:
                fingers=tuple(sorted((link_owner(name)-2)//4 for name in pair['regions']))
                mask[i,GROUPS.index(fingers)]=True
        return mask

    def pair_queries(self,rows,frames):
        """Discrete observed region labels are fixed; both robot pads move."""
        groups={};p,r=frames[:2]
        for i,row in enumerate(rows):
            for pair in row['contacts']['self_pairs']:
                owners=tuple(map(link_owner,pair['regions']))
                groups.setdefault(owners,[]).append((i,pair))
        for owners,items in groups.items():
            ids=torch.tensor([i for i,_ in items],device=p.device)
            target=self.hand.tensor([pair['points_m'] for _,pair in items]).mean(1)
            queries=[]
            for owner in owners:
                region=self.region_owners.index(owner)
                query=(r[ids,owner].transpose(-1,-2)@(target-p[ids,owner])[...,None]).squeeze(-1)
                queries.append((region,owner,query))
            yield ids,target,queries

    @torch.no_grad()
    def collect_geometry_queries(self,q,rows,frames):
        super().collect_geometry_queries(q,rows,frames)
        for ids,target,queries in self.pair_queries(rows,frames):
            for region,owner,query in queries:
                links=torch.full((len(ids),),owner,device=q.device,dtype=torch.long)
                self.collision.collect_contact(query,torch.zeros_like(query),links,torch.ones_like(links,dtype=torch.bool))

    def batch(self,q,rows,references=None,targets=None,acceleration=None,previous_rows=None,audit=False,normalizers=None,*,collect_metrics=True):
        total,parts,metrics=super().batch(q,rows,references,targets,acceleration,previous_rows,audit,normalizers,collect_metrics=collect_metrics)
        frames=self.hand.frames(q);p,r=frames[:2];size=len(q)
        closure=q.new_zeros(size);normal_cost=q.new_zeros(size)
        gaps=q.new_zeros(size);angles=q.new_zeros(size);counts=torch.zeros(size,device=q.device,dtype=torch.long)
        for ids,target,queries in self.pair_queries(rows,frames):
            positions=[];normals=[]
            for region,owner,query in queries:
                local,normal=self.project_surface(query,region)
                positions.append((r[ids,owner]@local[...,None]).squeeze(-1)+p[ids,owner])
                normals.append(F.normalize((r[ids,owner]@normal[...,None]).squeeze(-1),dim=-1))
            separation=positions[0]-positions[1];dot=(normals[0]*normals[1]).sum(-1).clamp(-1,1)
            closure=closure.index_add(0,ids,self.card.wpinch**2*1.e6*separation.square().sum(-1))
            normal_cost=normal_cost.index_add(0,ids,self.card.wc**2*self.card.direction_price*self.card.radius_mm**2*(1+dot))
            gaps=gaps.scatter_reduce(0,ids,separation.detach().norm(dim=-1),reduce='amax',include_self=True)
            angles=angles.scatter_reduce(0,ids,torch.rad2deg(torch.acos(-dot.detach())),reduce='amax',include_self=True)
            counts=counts.index_add(0,ids,torch.ones_like(ids))
            if collect_metrics:
                for k,i in enumerate(ids.tolist()):metrics[i]['points'].append(tuple(pos[k:k+1] for pos in positions))
        parts['contact_self_closure']=closure.mean()
        parts['contact_self_normal']=normal_cost.mean()
        parts['self_contact_pairs']=counts.float().mean()  # monitor, not another cost
        for i,metric in enumerate(metrics):
            metric['max_gap_m']=torch.maximum(metric['max_gap_m'],gaps[i:i+1])
            metric['max_normal_deg']=torch.maximum(metric['max_normal_deg'],angles[i:i+1])
            metric['contact_count']+=int(counts[i])
            metric['self_contact_count']=int(counts[i])
            metric['self_contact_gap_m']=gaps[i:i+1]
            metric['self_contact_satisfied']=(gaps[i:i+1]<=self.settings.contact_limit_m)&(angles[i:i+1]<=self.settings.normal_deg)&(metric['max_penetration_m']<=self.settings.penetration_limit_m)
            if 'actuated_gap_m' in metric:
                metric['actuated_gap_m']=torch.maximum(metric['actuated_gap_m'],gaps[i:i+1])
                metric['actuated_contact_count']+=int(counts[i])
                metric['actuated_contact_satisfied']=metric['actuated_contact_satisfied']&metric['self_contact_satisfied']
            metric['feasible']=metric['feasible']&(gaps[i:i+1]<=self.settings.contact_limit_m)&(angles[i:i+1]<=self.settings.normal_deg)
        return total+closure.mean()+normal_cost.mean(),parts,metrics


class ActuatedSelfContactCardCostObjective(SelfContactCardCostObjective):
    """Fixed palm is observable but cannot be controlled by the 20 finger DOFs.

    Preserve raw physical audit/acceptance, including palm errors. Only remove
    terms with identically zero policy-input Jacobian from the policy objective.
    Moving fingers touching the palm remain subject to contact/collision losses.
    """
    def __init__(self,hand,collision):
        super().__init__(hand,collision)
        from dataclasses import replace
        self.fixed_palm_diagnostic_only=True
        self.settings=replace(self.settings,loss_kind='card_raw_neural_self_contact_v4_actuated')
