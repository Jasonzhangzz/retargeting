"""Pair-local updates with EMA only on the coupled, non-critical rest surface."""
import numpy as np
import torch
from .contact import FINGERS
from .glove_geometry import ROOT_FINGERS, CHAINS

REVISION = 'pair_local_spatial_ema_v2'
BLOCK_STEPS = 32
MAX_RESIDUAL_M = .02


def validate_rest_residual(value=None):
    value=np.zeros((778,3),np.float32) if value is None else np.asarray(value,dtype=np.float32)
    if value.shape!=(778,3) or not np.isfinite(value).all() or np.any(np.linalg.norm(value,axis=-1)>MAX_RESIDUAL_M):
        raise ValueError('无效的固定表面 EMA 补偿')
    return value


@torch.no_grad()
def spatial_ema_step(model,prior,average,proposal,frozen,critical,soft_constraints=False):
    from .mano_shape_prior import EMA_DECAY
    updated=torch.where(frozen,average,proposal)
    previous_raw,previous_j=model.official_rest(average[:10],average[46:89],apply_residual=False)
    previous=previous_raw+model.rest_vertex_residual
    raw,raw_j=model.official_rest(updated[:10],updated[46:89],apply_residual=False)
    # Beta is trainable in mixed batches. Preserve EMA in the fixed wrist
    # coordinate frame rather than leaking the wrist regressor's translation.
    previous=previous+(raw_j[0]-previous_j[0])
    surface=torch.where(critical[:,None],raw,previous.lerp(raw,1-EMA_DECAY))
    residual=surface-raw
    if soft_constraints:
        # Training never rolls back a finite step for anatomical shape. Those
        # constraints are differentiable losses; only final validation gates
        # model application. Important vertices receive no EMA.
        model.rest_vertex_residual.copy_(residual)
        return updated,torch.ones((),device=proposal.device,dtype=torch.bool)
    valid=prior.valid_surface(surface) & (residual.norm(dim=-1)<=MAX_RESIDUAL_M).all()
    model.rest_vertex_residual.copy_(torch.where(valid,residual,model.rest_vertex_residual))
    updated[:10]=torch.where(valid,updated[:10],average[:10])
    updated[46:89]=torch.where(valid,updated[46:89],average[46:89])
    return updated,valid


def finger_coordinates(finger, size=89, device=None):
    """Ownership of mounts and explicit geometry, in the shared vector layout."""
    mask = torch.zeros(size, dtype=torch.bool, device=device)
    mask[10+3*finger:13+3*finger] = True
    mask[25+3*finger:28+3*finger] = True
    if size >= 89:
        mask[46+3*finger:49+3*finger] = True
        mask[61+finger] = True  # width
        mask[66+finger] = True  # depth
        if finger in ROOT_FINGERS:
            start = 71+3*ROOT_FINGERS.index(finger)
            mask[start:start+3] = True
        if finger == 0:
            mask[83:89] = True  # CMC axes (observability still takes precedence)
    return mask


def phase_stages(rows, phase):
    stages = {r['stage'] for r in rows}
    if phase == 'reference':
        return stages & {'open', 'together', 'mcp90', 'hook'}
    if phase == 'geometry':
        return {s for s in stages if not s.startswith('pinch_')}
    return {s for s in stages if s.startswith('pinch_')}


class PairLocalRefinement:
    """Balanced pair blocks; inactive coordinates are restored, including EMA.

    Only training rows select blocks. All phases of a pair remain in its EMF
    loss; the existing tactile evidence gate alone selects contact equalities.
    Palm/web vertices have an additional drift anchor because MANO LBS weights
    blend finger boundaries even when all other finger coordinates are fixed.
    """
    def __init__(self, objective, vector, base_frozen, original_weights):
        self.objective = objective
        self.stages = [f'pinch_{f}' for f in FINGERS[1:]
                       if any(r['stage'] == f'pinch_{f}' for r in objective.rows)]
        if not self.stages or any(r.get('partition', 'train') != 'train' for r in objective.rows):
            raise ValueError('局部对指精修必须使用训练分区的对指记录')
        self.reference = vector.detach().clone()
        self.base_frozen = base_frozen.clone()
        self.original_weights = original_weights
        self.frozen = base_frozen.clone()
        self.fixed = vector.detach().clone()
        self.counts = {s: 0 for s in self.stages}
        self.max_inactive_parameter_drift = 0.
        self.max_inactive_vertex_drift_mm = {s: 0. for s in self.stages}
        self.index = -1
        objective.model.model.surface_residual_enabled=True
        self.accepted_steps = torch.zeros((),device=vector.device,dtype=torch.long)
        self.rejected_steps = torch.zeros_like(self.accepted_steps)
        with torch.no_grad():
            vertices, _ = objective.model.model.official_rest(vector[:10], vector[46:89])
        objective.local_reference = vertices.detach().clone()
        objective.local_vertex_weight = torch.zeros(len(vertices), device=vector.device)
        objective.phase_active = torch.zeros(len(objective.rows), device=vector.device)
        weights = objective.model.model.backend.lbs_weights.detach()
        self.vertex_weights = []
        self.critical_vertices = []
        self.allowed = []
        for stage in self.stages:
            partner = FINGERS.index(stage[6:])
            self.allowed.append(finger_coordinates(0, len(vector), vector.device) |
                                finger_coordinates(partner, len(vector), vector.device))
            joints = list(CHAINS[0]) + list(CHAINS[partner])
            self.vertex_weights.append((1-weights[:, joints].sum(-1)).clamp(0, 1))
            self.critical_vertices.append(weights[:,joints].sum(-1)>=.5)

    @torch.no_grad()
    def select(self, block, vector, average):
        self.index = block % len(self.stages)
        # Commit the accepted parameters; the fixed surface buffer carries the
        # non-critical EMA state across pair boundaries.
        vector.copy_(average)
        self.fixed.copy_(average)
        vertices, _ = self.objective.model.model.official_rest(average[:10], average[46:89])
        self.objective.local_reference.copy_(vertices)
        self.frozen.copy_(self.base_frozen | ~self.allowed[self.index])
        active = self.objective.t([r['stage'] == self.stages[self.index] for r in self.objective.rows])
        self.objective.phase_active.copy_(active)
        self.objective.weights.copy_(self.original_weights*active[:, None])
        self.objective.weights.div_(self.objective.weights.sum().clamp_min(1e-12))
        self.objective.local_vertex_weight.copy_(self.vertex_weights[self.index])

    @torch.no_grad()
    def update_ema(self, average, proposal):
        model=self.objective.model.model
        # Important fingers take the raw proposal without parameter or surface
        # EMA. Only vertices owned by other regions lag towards their coupled
        # proposal. This is EMA of positions, not repeated damping of velocity.
        updated,valid=spatial_ema_step(model,self.objective.shape_prior,average,proposal,
                                      self.frozen,self.critical_vertices[self.index])
        self.accepted_steps.add_(valid.long());self.rejected_steps.add_((~valid).long())
        return updated

    @torch.no_grad()
    def record_block(self, vector, steps):
        stage = self.stages[self.index]
        self.counts[stage] += steps
        drift = (vector-self.fixed)[self.frozen].abs().amax()
        self.max_inactive_parameter_drift = max(self.max_inactive_parameter_drift, float(drift))
        vertices, _ = self.objective.model.model.official_rest(vector[:10], vector[46:89])
        # Report core inactive regions separately from blended web boundaries.
        core = self.vertex_weights[self.index] >= .95
        distance = (vertices-self.objective.local_reference).norm(dim=-1)
        self.max_inactive_vertex_drift_mm[stage] = max(
            self.max_inactive_vertex_drift_mm[stage], float(distance[core].amax())*1000)

    @torch.no_grad()
    def cycle_cost(self, vector):
        """The same train-only, all-pair loss at every balanced-cycle boundary."""
        o=self.objective
        active=o.phase_active.clone();weights=o.weights.clone();anchor=o.local_vertex_weight.clone()
        try:
            o.phase_active.fill_(1)
            gate=o.t([r['stage'] in self.stages for r in o.rows])
            o.weights.copy_(self.original_weights*gate[:,None])
            o.weights.div_(o.weights.sum().clamp_min(1e-12))
            o.local_vertex_weight.zero_()
            return float(o.residual(vector).square().sum())
        finally:
            o.phase_active.copy_(active);o.weights.copy_(weights);o.local_vertex_weight.copy_(anchor)

    def report(self):
        return dict(revision=REVISION, block_steps=BLOCK_STEPS, steps_by_pair=self.counts,
                    frozen_global_parameters=['MANO_betas', 'wrist_alignment', 'wrist_translation'],
                    active_parameters='thumb_and_partner_mounts_lengths_widths_depths_roots_axes',
                    active_coordinates_by_pair={s:int((allowed & ~self.base_frozen).sum())
                        for s,allowed in zip(self.stages,self.allowed)},
                    inactive_parameter_max_drift=self.max_inactive_parameter_drift,
                    inactive_core_vertex_max_drift_mm=self.max_inactive_vertex_drift_mm,
                    inactive_surface_anchor_mm=.5,
                    ema_policy=dict(critical_parameter_decay=0.,critical_surface_decay=0.,
                        noncritical_parameter_update='frozen',noncritical_surface_decay=.95,
                        critical_vertex_rule='thumb_plus_partner_LBS_weight_at_least_0.5',
                        space='rest_vertices_before_pose_correctives_and_LBS',
                        runtime_updates=False,trainable_vertex_parameters=False),
                    accepted_surface_steps=int(self.accepted_steps),rejected_surface_steps=int(self.rejected_steps),
                    fixed_surface_residual_max_mm=float(self.objective.model.model.rest_vertex_residual.norm(dim=-1).amax())*1000,
                    heldout_used=False)

    def message(self):
        names={'index':'食指','middle':'中指','ring':'无名指','pinky':'小指'}
        return '拇指＋'+names[self.stages[self.index][6:]]+'局部精修'

    def close(self):
        for name in ('local_reference', 'local_vertex_weight', 'phase_active'):
            delattr(self.objective, name)
