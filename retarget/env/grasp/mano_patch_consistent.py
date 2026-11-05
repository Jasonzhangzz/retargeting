"""Patch witnesses shared by the optimization objective and final acceptance."""
import math
import torch
from retarget.solver.cem import DEFAULTS
from .mano_patch_anchor import PatchGeometry
from .spring_history import SpringHistorySettings
from .comfree_rollout import RolloutSettings


def patch_fit_terms(gap,offset,groups,finger_groups,settings):
    """Zero iff global coverage and every required finger pass the same limits.

    A passing slot has zero loss; its fixed template point is diagnostic only.
    The independent required-finger sum prevents other fingers diluting a miss.
    Palm participates in coverage, exactly as in the existing acceptance rule.
    """
    distance=((gap-settings.surface_tolerance_m-1.e-8).clamp_min(0.)/settings.surface_tolerance_m).square()
    near=((offset-settings.anchor_tolerance_m-1.e-8).clamp_min(0.)/settings.anchor_tolerance_m).square()
    error=distance+near
    count=max(1,math.ceil(settings.coverage_fraction*gap.shape[-1]-1.e-7))
    coverage=error.topk(count,largest=False,sorted=False).values.mean(-1)
    minimum=torch.stack([error[:,ids].amin(-1) for ids in groups],-1)
    required=minimum[:,finger_groups].sum(-1)
    return dict(cost=coverage+required,slot_error=error,coverage=coverage,required=required)


class PatchConsistentAdapter(PatchGeometry):
    retain_feasible=True

    def __init__(self,*args,spring_settings=None,**kwargs):
        kwargs.setdefault('rollout',DEFAULTS.rollout)
        kwargs.setdefault('geometry_backend','optimized')
        super().__init__(*args,spring_settings=spring_settings or DEFAULTS.spring,**kwargs)
        if self.rollout_settings.cost_mode!='velocity':raise ValueError('Version 5 requires terminal velocity scoring')
        self.objective='mano_patch_consistent_spring_history_h5_v5'
        self.criterion='patch_consistent_terminal_velocity_h5_v5'

    @torch.no_grad()
    def _compute(self,value,ids):
        decoded=self.decoder(value,full=True)
        _,parts=super()._compute(value,ids,decoded=decoded)
        vertices=decoded[0]
        parts.update(self.patches.query(vertices,ids))
        fit=patch_fit_terms(parts['patch_gap_m'],parts['patch_anchor_offset_m'],self.required_groups,self.finger_groups,self.patch_settings)
        penalty=self.settings.penetration_weight*parts['penetration']+self.settings.soft_penetration_weight*parts['soft_penetration']
        total=fit['cost']+penalty+parts['rollout_cost']
        cost=total
        groups=torch.stack([parts['patch_fitted'][:,group].any(-1) for group in self.required_groups],-1)
        all_fingers=groups[:,self.finger_groups].all(-1)
        depth=parts['penetration_depth_m'];valid=parts['valid']&torch.isfinite(cost)&torch.isfinite(fit['cost'])
        geometry=valid&(parts['patch_coverage']+1.e-7>=self.patch_settings.coverage_fraction)&all_fingers
        geometry &= (depth[:,0]<=self.settings.hand_object_allowance_m+1.e-8)&(depth[:,1]<=self.settings.penetration_tolerance_m)
        parts.update(anchor_cost=fit['cost'],patch_fit_cost=fit['cost'],
            patch_fit_slot_error=fit['slot_error'],patch_fit_coverage_cost=fit['coverage'],patch_fit_required_cost=fit['required'],
            patch_group_fitted=groups,patch_all_required_fingers=all_fingers,patch_geometry_accepted=geometry,
            patch_joint_accepted=geometry&parts['rollout_stable'],valid=valid,total_cost_v3=total,optimizer_cost_v3=cost)
        return torch.where(valid,cost,torch.inf),parts

    def _candidate_gate(self,parts):
        return parts['patch_joint_accepted']

    @torch.no_grad()
    def validate(self,value,rows):
        cost=self.evaluate(value,rows,collect=True)
        return cost,{k:v.clone() for k,v in self.last.items()}
