"""Shared, task-masked MANO residuals and bounded least-squares solvers.

Gesture labels are training supervision only. Runtime and holdout solve from
EMF, the skeleton prior and (optionally) measured bilateral tactile evidence.
All lengths are metres; rotations are radians, on SO(3).
"""
from __future__ import annotations

import math
import time
import copy
import json
from dataclasses import dataclass
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .card_math import exp_so3, log_so3
from .contact import FINGERS
from .hand_model import MANO_SOLVER_REVISION
from .mano_collision import SurfaceCollision, collision_residual, HARD_PENETRATION_M
from scripts.calibrate_wuji_mano import CHAINS, TIPS

SOLVER_REVISION = MANO_SOLVER_REVISION


def participation(rows, training=True):
    mask = np.ones((len(rows), 5), np.float32)
    if training:
        for i, row in enumerate(rows):
            stage = row.get('stage', '')
            from .contact_protocol import observation_mask
            specified=observation_mask(row)
            if specified is not None:mask[i]=specified
            elif stage.startswith('pinch_'):
                mask[i] = 0
                mask[i, [0, FINGERS.index(stage[6:])]] = 1
            elif stage.startswith('range_') or stage=='thumb_opposition':
                mask[i]=0
                mask[i,FINGERS.index(stage[6:] if stage.startswith('range_') else 'thumb')]=1
            elif stage in ('open', 'together', 'mcp90', 'hook', 'fist', 'mcp90_pip90'):
                mask[i, 0] = int(stage=='together' and row.get('thumb_index_side_contact') is True)
    for i, row in enumerate(rows):
        for f, name in enumerate(FINGERS):
            obs = row.get('fingers', {}).get(name, {})
            if (row.get('capture_protocol')=='contact_motion_v3' and name not in row.get('fingers',{})) or obs.get('valid') is False or float(obs.get('confidence', 1.)) <= 0:
                mask[i, f] = 0
    return mask


def stage_weights(rows, mask, training, pinch_multiplier=2.):
    """Each gesture has equal total price, irrespective of frame/finger counts."""
    stages = [r.get('stage', '') if training else 'runtime' for r in rows]
    groups = list(dict.fromkeys(stages))
    weight = np.zeros_like(mask)
    prices={stage:pinch_multiplier if training and (stage.startswith(('pinch_','contact_pair_','contact_tripod_','contact_four_')) or stage=='contact_five') else 1. for stage in groups}
    total=sum(prices.values())
    for stage in groups:
        ids = [i for i, s in enumerate(stages) if s == stage]
        den = float(mask[ids].sum())
        if den: weight[ids] = mask[ids] * prices[stage] / (den * total)
    return weight


def rotation_log(r):
    """SO(3) log with a smooth identity branch and a stable near-pi branch."""
    return log_so3(r)


def neutral_rotations(model):
    """Deterministic straight-bone chart, not a fitted per-joint bias.

    MANO's template bones are not exactly collinear/coplanar at axis-angle=0.
    Flexion bounds must be relative to physical straight bones, as in an
    anatomical axis layer, rather than clipping raw absolute MANO rotations.
    """
    from .mano_runtime import aim_rotation
    neutral=np.tile(np.eye(3),(15,1,1))
    normal=np.asarray(model.anatomy.palmar_normal)
    rest_j=getattr(model,'pose_reference_joints',model.rest_j_np)
    rest_v=getattr(model,'pose_reference_vertices',model.rest_np)
    for f,chain in enumerate(CHAINS):
        desired=rest_j[chain[1]]-rest_j[chain[0]]
        if f:desired=desired-normal*(desired@normal)
        desired=desired/np.linalg.norm(desired)
        parent=np.eye(3)
        for k,joint in enumerate(chain):
            end=rest_j[chain[k+1]] if k<2 else rest_v[TIPS[f]]
            global_rotation=aim_rotation(end-rest_j[joint],desired)
            neutral[joint-1]=parent.T@global_rotation
            parent=global_rotation
    return neutral


def pose_angles(model, poses, neutral=None):
    neutral=neutral_rotations(model) if neutral is None else neutral
    local=Rotation.from_rotvec(np.asarray(poses).reshape(-1,3)).as_matrix().reshape(*np.asarray(poses).shape[:-1],3,3)
    relative=neutral.swapaxes(-1,-2)@local
    q=Rotation.from_matrix(relative.reshape(-1,3,3)).as_rotvec().reshape(np.asarray(poses).shape)
    return model.anatomy.angles(q)


def anatomical_rotations(angles,basis,neutral=None):
    axes = torch.eye(3, device=angles.device, dtype=angles.dtype)
    local = exp_so3(angles[..., 0, None] * axes[0]) @ exp_so3(angles[..., 1, None] * axes[1]) @ exp_so3(angles[..., 2, None] * axes[2])
    rotation=basis @ local @ basis.transpose(-1, -2)
    if neutral is not None:rotation=neutral@rotation
    return rotation


def anatomical_pose(angles,basis,neutral=None):
    return rotation_log(anatomical_rotations(angles,basis,neutral))


def decode(model, beta, angles, root, basis, neutral=None, thumb_chart=None,geometry=None):
    local=anatomical_rotations(angles,basis,neutral)
    if thumb_chart is not None:local=torch.cat((local[...,:12,:,:],thumb_chart.decode(angles,geometry)),-3)
    pose = rotation_log(local)
    rest, joints = model.model.official_rest(beta,geometry)
    v,j,r=model.model.official_matrix_forward(beta,local,geometry)
    v = (v - joints[0]) @ root.transpose(-1, -2)
    j = (j - joints[0]) @ root.transpose(-1, -2)
    r = root[:, None] @ r
    return pose, v, j, r


def _robust(r):
    # Smooth robust least-squares residual, quadratic near zero, linear at range.
    return r / torch.pow(1 + (r / 3).square(), .25)


@dataclass(frozen=True)
class SolverSettings:
    adam_steps: int = 480
    lm_steps: int = 8
    collision_refresh: int = 8
    runtime_iterations: int = 4
    runtime_budget_ms: float = 30.
    cold_iterations: int = 16
    cold_budget_ms: float = 150.
    position_scale_m: float = .003
    orientation_scale_rad: float = .15
    pose_prior_rad: float = .5
    collision_weight: float = 20.
    correction_rate_deg_s: float = 25.
    correction_continuity_rad: float = .035
    calibration_contact_position_weight: float = 320.
    calibration_contact_normal_weight: float = 40.
    calibration_contact_distance_scale_m: float = .002
    calibration_contact_normal_tolerance_deg: float = 15.
    calibration_contact_normal_scale_deg: float = 15.
    pinch_observation_multiplier: float = 2.
    calibration_tip_emf_orientation_weight: float = 40.
    optimizer: str = 'adam'


class ManoObjective:
    """One parameter vector, with explicit shared and per-frame variable blocks."""
    def __init__(self, model, rows, poses, roots, *, training=False, shared=False,
                 tactile=True, settings=None, collision=None, active_fingers=None,previous_correction=None,interval_s=None,correction_rates=None,fixed_forward=False):
        self.model = model; self.rows = rows; self.training = training; self.shared = shared
        self.fixed_forward=fixed_forward
        self.settings = settings or SolverSettings()
        self.t = lambda a: torch.as_tensor(np.asarray(a), device=model.device, dtype=torch.float32)
        self.root = self.t(roots); self.seed_pose = self.t(poses)
        self.basis = self.t(model.anatomy.basis)
        self.neutral_np=neutral_rotations(model);self.neutral=self.t(self.neutral_np)
        self.low = self.t(model.anatomy.lower); self.high = self.t(model.anatomy.upper)
        raw_angles = self.t(pose_angles(model,np.asarray(poses),self.neutral_np))
        from .thumb_chart import source_thumb_chart,SLOTS
        self.thumb_chart=source_thumb_chart(model,rows)
        if self.thumb_chart is not None:
            expected=exp_so3(self.seed_pose[:,12:])
            if not torch.allclose(self.thumb_chart.seed_rotations,expected,atol=2e-5,rtol=0):
                raise ValueError('Source thumb axes disagree with the same-frame skeleton rotations')
            self.low=self.low.clone();self.high=self.high.clone()
            self.low[12:]=0.;self.high[12:]=0.;raw_angles[:,12:]=0.
            for n,(j,k) in enumerate(SLOTS):
                self.low[j,k]=float(self.thumb_chart.bounds[n,0]);self.high[j,k]=float(self.thumb_chart.bounds[n,1])
                raw_angles[:,j,k]=self.t(self.thumb_chart.seeds[:,n])
        self.seed_angles = raw_angles.clamp(self.low, self.high)
        self.mask_np = participation(rows, training)
        self.weights = self.t(stage_weights(rows, self.mask_np, training,self.settings.pinch_observation_multiplier))
        free = np.broadcast_to((self.high-self.low).cpu().numpy()>1e-8, (len(rows), 15, 3)).copy()
        if training:
            for i in range(len(rows)):
                for f, chain in enumerate(CHAINS):
                    if not self.mask_np[i, f]: free[i, np.array(chain)-1] = False
        if active_fingers is not None:
            for f, chain in enumerate(CHAINS):
                if f not in active_fingers: free[:, np.array(chain)-1] = False
        for i in range(len(rows)):
            for f, chain in enumerate(CHAINS):
                if (training and not self.mask_np[i,f]) or (active_fingers is not None and f not in active_fingers):
                    self.seed_angles[i,np.array(chain)-1]=raw_angles[i,np.array(chain)-1]
        if fixed_forward:free[:]=False
        self.free = torch.as_tensor(free, device=model.device)
        self.rawp = self.t([[r.get('fingers',{}).get(f,{}).get('position',[0,0,0]) for f in FINGERS] for r in rows])
        self.rawr = self.t([Rotation.from_quat([r.get('fingers',{}).get(f,{}).get('quaternion_xyzw',[0,0,0,1]) for f in FINGERS]).as_matrix() for r in rows])
        self.beta0 = self.t(model.shape[:10]); self.mp0 = self.t(model.sensor_offsets())
        self.mr0 = self.t(model.mount_rot); self.align0 = self.t(model.raw_rotation); self.trans0 = self.t(model.raw_translation)
        reference=model.flat_reference or {}
        reference_matches=all(not reference.get('calibration_fingerprint') or
                              row.get('calibration_fingerprint')==reference['calibration_fingerprint'] for row in rows)
        mounting_ready=self.training or model.mounting_calibrated or model.mounting_estimated
        self.observations_enabled = bool(reference) and reference_matches and mounting_ready and all('fingers' in r for r in rows)
        self.collision = collision or SurfaceCollision(model)
        self.witnesses = []
        self.collision_reports = []
        self.contacts = []
        self.contact_anchors=[];self.tactile_active_fingers=[]
        self.together = []
        self.last_terms = {}
        from .glove_geometry import COUNT,LOWER,UPPER
        self.personalized=any(r.get('capture_protocol') in ('geort_motion_v2','contact_motion_v3') for r in rows)
        self.geometry0=self.t(getattr(model,'glove_parameters',np.zeros(COUNT))) if self.personalized or hasattr(model,'glove_parameters') else None
        self.shared_size = (46+COUNT if self.personalized else 46) if shared else 0
        self.initial = torch.cat((self.beta0, torch.zeros(36, device=model.device), self.seed_angles[self.free])) if shared else self.seed_angles[self.free].clone()
        self.lower = torch.cat((self.t([-2]*10 + [-3]*15 + [-.8]*15 + [-.3]*3 + [-1.5]*3), self.low.expand_as(self.seed_angles)[self.free])) if shared else self.low.expand_as(self.seed_angles)[self.free].clone()
        self.upper = torch.cat((self.t([2]*10 + [3]*15 + [.8]*15 + [.3]*3 + [1.5]*3), self.high.expand_as(self.seed_angles)[self.free])) if shared else self.high.expand_as(self.seed_angles)[self.free].clone()
        if shared and self.personalized:
            self.initial=torch.cat((self.initial[:46],self.geometry0,self.initial[46:]))
            self.lower=torch.cat((self.lower[:46],self.t(LOWER),self.lower[46:]))
            self.upper=torch.cat((self.upper[:46],self.t(UPPER),self.upper[46:]))
        if shared and fixed_forward:
            from .mano_shape_prior import ShapePrior,GEOMETRY_UPPER,project
            self.shape_prior=ShapePrior(model)
            if self.personalized:
                self.lower[46:self.shared_size]=-self.t(GEOMETRY_UPPER)
                self.upper[46:self.shared_size]=self.t(GEOMETRY_UPPER)
            if getattr(self,'unbounded_shape',False):
                self.lower[:10]=-float('inf');self.upper[:10]=float('inf')
                self.lower[46:self.shared_size]=-float('inf');self.upper[46:self.shared_size]=float('inf')
                self.shape_prior.unbounded_shape=True
            self.initial=project(self.initial,self.lower,self.upper,self.shared_size,unbounded_shape=getattr(self,'unbounded_shape',False))
        self.correction_limit_rad=None if self.observations_enabled else math.radians(8.)
        self.previous_correction=None
        self.interval_s=interval_s
        if not training and not shared:
            if self.correction_limit_rad is not None:
                self.lower=torch.maximum(self.lower,self.initial-self.correction_limit_rad)
                self.upper=torch.minimum(self.upper,self.initial+self.correction_limit_rad)
            if previous_correction is not None and interval_s is not None and 0<interval_s<.25:
                self.previous_correction=self.t(previous_correction)[None]
                centre=(self.initial+self.previous_correction[self.free]).clamp(self.lower,self.upper)
                rates=np.full((15,3),self.settings.correction_rate_deg_s)
                if self.observations_enabled and correction_rates is not None:
                    for f,c in enumerate(CHAINS):rates[np.array(c)-1]=correction_rates[f]
                step=self.t(np.deg2rad(rates))[None][self.free]*interval_s
                self.lower=torch.maximum(self.lower,centre-step);self.upper=torch.minimum(self.upper,centre+step)
                self.initial=centre
        self.refresh(self.initial, contacts=tactile)

    def unpack(self, x):
        if self.shared:
            beta = x[:10]; mp = self.mp0 + .01*x[10:25].reshape(5, 3)
            mr = self.mr0 @ exp_so3(x[25:40].reshape(5, 3))
            align = exp_so3(x[40:43]) @ self.align0; trans = self.trans0 + .01*x[43:46]
        else: beta, mp, mr, align, trans = self.beta0, self.mp0, self.mr0, self.align0, self.trans0
        angles = self.seed_angles.clone().masked_scatter(self.free, x[self.shared_size:])
        return beta, mp, mr, align, trans, angles

    def surface(self, x):
        beta, mp, mr, align, trans, angles = self.unpack(x)
        geometry=x[46:self.shared_size] if self.shared and self.personalized else self.geometry0
        if self.fixed_forward:
            pose=self.seed_pose
            _,rest_j=self.model.model.official_rest(beta,geometry)
            v,j,r=self.model.model.official_matrix_forward(beta,exp_so3(pose),geometry)
            v=(v-rest_j[0])@self.root.transpose(-1,-2)
            j=(j-rest_j[0])@self.root.transpose(-1,-2)
            r=self.root[:,None]@r
        else:pose, v, j, r = decode(self.model, beta, angles, self.root, self.basis,self.neutral,self.thumb_chart,geometry)
        return pose, v, j, r, (beta, mp, mr, align, trans, angles)

    def refresh(self, x, contacts=False):
        from .tactile_pose import frame_constraints
        with torch.no_grad():
            pose, v, j, r, _ = self.surface(x)
            ordered = torch.cat((j, v[:, list(TIPS)]), 1)[:, self.model.ordered]
            self.witnesses = []; self.collision_reports = []
            self.contacts = [] if contacts else self.contacts
            if contacts:self.contact_anchors=[]
            self.together = []
            self.side_missing=[]
            for i, row in enumerate(self.rows):
                w, report = self.collision.query(v[i].cpu().numpy(),tensor_device=self.model.device)
                self.witnesses.append(w); self.collision_reports.append(report)
                if contacts and self.model.mapping:
                    anchors, pairs = frame_constraints(self.model, row, v[i].cpu().numpy(), ordered[i].cpu().numpy(), guided=self.training)
                    self.contact_anchors.extend((i,a) for a in anchors)
                    for pair in pairs:
                        if self.training and pair['evidence'] != 'confirmed_calibration_gesture': continue
                        self.contacts.append((i, pair))
                if self.training and row.get('stage') == 'together':
                    from .side_contact import side_material_pairs
                    for f in range(0 if row.get('thumb_index_side_contact') is True else 1,4):
                        try:
                            pairs=side_material_pairs(self.model,v[i].cpu().numpy(),j[i].cpu().numpy(),finger_pairs=[f])
                            self.together.extend((i,p) for p in pairs)
                        except ValueError as exc:self.side_missing.append(dict(frame=i,pair=f'{f}_{f+1}',reason=str(exc)))

    def residual(self, x, record=False,measurement_only=False):
        pose, v, j, r, values = self.surface(x)
        beta, mp, mr, align, trans, angles = values
        parts = {}; s = self.settings
        def add(name, value, scale=1.):
            value = value.reshape(-1)
            if value.numel(): parts.setdefault(name, []).append(value * scale / math.sqrt(value.numel()))
        ids=getattr(self,'distal_ids',None)
        distal=r.index_select(1,ids) if ids is not None else r[:,[c[-1] for c in CHAINS]]
        centres=j.index_select(1,ids) if ids is not None else j[:,[c[-1] for c in CHAINS]]
        predicted = centres + (distal @ mp[None, :, :, None]).squeeze(-1)
        if self.observations_enabled:
            w = self.weights.sqrt()
            # Normalized by task before flattening; inactive observations have
            # exactly zero gradient, including for shared betas/mounts.
            parts['sensor_position'] = [(_robust((predicted - (self.rawp @ align.T + trans))/s.position_scale_m) * w[..., None] / math.sqrt(3)).reshape(-1)]
            orientation = rotation_log((distal @ mr).transpose(-1, -2) @ (align @ self.rawr))
            parts['sensor_orientation'] = [(_robust(orientation/s.orientation_scale_rad) * w[..., None] / math.sqrt(3)).reshape(-1)]
        prior = rotation_log(exp_so3(self.seed_pose).transpose(-1, -2) @ exp_so3(pose))
        joint_mask = self.free.any(-1)
        active=[i for i,row in enumerate(self.rows) if not hasattr(self,'phase_stages') or row['stage'] in self.phase_stages]
        add('skeleton_prior', (prior[active] / s.pose_prior_rad) * joint_mask[active,...,None], .35 if self.observations_enabled else 1.)
        if self.previous_correction is not None:
            continuity=(angles-self.seed_angles-self.previous_correction)*self.free
            add('correction_continuity',continuity/s.correction_continuity_rad*math.sqrt(.033/self.interval_s))
        for i, witness in enumerate(self.witnesses):
            if i not in active:continue
            penetration=collision_residual(v[i],witness)
            add('collision', penetration, math.sqrt(s.collision_weight/len(active)))
            # A p-norm is a smooth upper bound on the true worst residual.
            # Discrete max switched the active finger on sub-mm input noise.
            if penetration.numel():add('collision_max',torch.linalg.vector_norm(penetration,ord=8)[None],math.sqrt(4*s.collision_weight/len(active)))
        smooth_normals=None
        if self.training and self.contacts:
            from .mano_contact_loss import vertex_normals
            smooth_normals=vertex_normals(v,torch.as_tensor(self.model.faces,device=v.device,dtype=torch.long))
        for i, pair in self.contacts:
            anchors = [pair['a'], pair['b']]
            ids = torch.as_tensor(self.model.faces[[a['source_face'] for a in anchors]], device=v.device)
            bc = self.t([a['barycentric'] for a in anchors])
            tri = v[i, ids]; point = (tri*bc[..., None]).sum(-2)
            normal = torch.nn.functional.normalize(torch.linalg.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), dim=-1)
            if self.training:
                from .mano_contact_loss import normal_residual
                normal=torch.nn.functional.normalize((smooth_normals[i,ids]*bc[...,None]).sum(-2),dim=-1)
                direction=normal_residual(normal[0],normal[1],s.calibration_contact_normal_tolerance_deg,s.calibration_contact_normal_scale_deg)
            else:direction=normal[0]+normal[1]
            add('contact_position', (point[0]-point[1])/(s.calibration_contact_distance_scale_m if self.training else .002), math.sqrt((s.calibration_contact_position_weight if self.training else 20)/max(1, len(self.contacts))))
            add('contact_normal', direction, math.sqrt((s.calibration_contact_normal_weight if self.training else 10)/max(1, len(self.contacts))))
        if self.training:
            for stage in dict.fromkeys(row['stage'] for row in self.rows):
                if hasattr(self,'phase_stages') and stage not in self.phase_stages: continue
                ids = [i for i, row in enumerate(self.rows) if row['stage'] == stage]
                if stage in ('open', 'together'):
                    directions = []
                    for f in range(1, 5):
                        c = CHAINS[f]; p = torch.stack((j[ids, c[0]], j[ids, c[1]], j[ids, c[2]], v[ids, TIPS[f]]), 1)
                        directions.append(torch.nn.functional.normalize(p[:, 1:]-p[:, :-1], dim=-1))
                    d = torch.stack(directions, 1)
                    normal = (self.root[ids] @ self.t(self.model.anatomy.palmar_normal)[:, None]).squeeze(-1)
                    add('flat_straight', (d[:, :, 1:]-d[:, :, :1])/.05, 4.)
                    add('flat_plane', (d*normal[:, None, None]).sum(-1)/.05, 3.)
                    if stage == 'together': add('together_direction', (d[:, :, 0]-d[:, :, 0].mean(1, keepdim=True))/.05, 3.)
                    thumb_ids=[i for i in ids if self.rows[i].get('thumb_index_side_contact') is True]
                    if stage=='together' and thumb_ids:
                        c=CHAINS[0];prox=torch.nn.functional.normalize(j[thumb_ids,c[2]]-j[thumb_ids,c[1]],dim=-1)
                        distal=torch.nn.functional.normalize(v[thumb_ids,TIPS[0]]-j[thumb_ids,c[2]],dim=-1)
                        add('together_thumb_straight',(prox-distal)/.08,2.)
                elif stage in ('mcp90', 'fist', 'mcp90_pip90') and not self.fixed_forward:
                    # Fixed-forward training has no angle variables. These
                    # terms would be constants: they cannot fit geometry and
                    # would obscure relative convergence of the true loss.
                    chains = np.array(CHAINS[1:])-1
                    add('mcp90', (angles[ids][:, chains[:, 0], 0]-math.pi/2)/.15, 3.)
                    target = math.pi/2 if stage in ('fist', 'mcp90_pip90') else 0.
                    add('pip_reference', (angles[ids][:, chains[:, 1], 0]-target)/.15, 3.)
                    if stage == 'mcp90': add('dip_reference', angles[ids][:, chains[:, 2], 0]/.15, 3.)
            for i,pair in self.together:
                if i not in active:continue
                index=torch.as_tensor(self.model.faces[[pair['face_a'],pair['face_b']]],device=v.device)
                bc=self.t([pair['barycentric_a'],pair['barycentric_b']]);tri=v[i,index]
                points=(tri*bc[...,None]).sum(-2)
                normal=torch.nn.functional.normalize(torch.linalg.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),dim=-1)
                add('together_gap',(points[0]-points[1])/.002,8/math.sqrt(max(1,len(self.together))))
                add('together_normal',normal[0]+normal[1],1/math.sqrt(max(1,len(self.together))))
        if self.shared:
            add('shape_prior', beta/.75 if self.fixed_forward else beta, 2. if self.fixed_forward else .2)
            add('mount_prior', x[10:40], .2)
            add('registration_prior', torch.cat((x[40:43]/.15, x[43:46])), .5)
            if self.personalized:
                if self.fixed_forward:
                    from .mano_shape_prior import GEOMETRY_SIGMA
                    add('geometry_prior',x[46:self.shared_size]/self.t(GEOMETRY_SIGMA),2.)
                else:add('geometry_prior',x[46:self.shared_size],.35)
            if self.fixed_forward and not measurement_only:
                from .mano_shape_prior import SOFT_SHAPE_WEIGHT
                for name,value in self.shape_prior.residuals(beta,x[46:self.shared_size] if self.personalized else None).items():
                    add(name,value,math.sqrt(SOFT_SHAPE_WEIGHT) if name=='shape_soft_constraints' else 3.)
        if measurement_only:
            parts={k:val for k,val in parts.items() if k not in ('shape_prior','mount_prior','registration_prior','geometry_prior','skeleton_prior','collision','collision_max','correction_continuity')}
        vector = torch.cat([torch.cat(values) for values in parts.values()])
        if record:
            self.last_terms = {name: float(torch.cat(vals).detach().square().sum()) for name, vals in parts.items()}
            self.last_surface=(pose.detach(),v.detach(),j.detach(),r.detach())
        return vector

    def install(self, x):
        beta, mp, mr, align, trans, _ = self.unpack(x)
        m = self.model
        m.set_shape(beta.detach().cpu().numpy())
        if self.personalized and self.shared:m.set_glove_geometry(x[46:self.shared_size].detach().cpu().numpy())
        m.sensor_joint_position = mp.detach().cpu().numpy(); m.mount_rot = mr.detach().cpu().numpy()
        m.mount_pos = m.sensor_joint_position - (m.tip_anchors-m.rest_j_np[[c[-1] for c in CHAINS]])
        m.raw_rotation = align.detach().cpu().numpy(); m.raw_translation = trans.detach().cpu().numpy()
        m.calibration_pose_offsets = np.zeros((15, 3), np.float32)

    def runtime_batch(self, x):
        """Evaluate perturbations together, with the *same* runtime residuals.

        Tiny MANO CUDA kernels are launch-bound. A central-difference batch
        avoids dozens of autograd forward-dual traversals per live frame.
        Shared offline parameters still use exact autograd Jacobians.
        """
        if self.training or self.shared or len(self.rows)!=1:raise ValueError('Runtime batch requires one fixed-shape frame')
        b=len(x);s=self.settings
        ids=self.free.flatten().nonzero().flatten()
        angles=self.seed_angles.reshape(1,-1).expand(b,-1).index_copy(1,ids,x).reshape(b,15,3)
        pose,v,j,r=decode(self.model,self.beta0,angles,self.root.expand(b,-1,-1),self.basis,self.neutral,self.thumb_chart,self.geometry0)
        values=[]
        distal=r[:,[c[-1] for c in CHAINS]]
        if self.observations_enabled:
            predicted=j[:,[c[-1] for c in CHAINS]]+(distal@self.mp0[None,:,:,None]).squeeze(-1)
            w=self.weights.sqrt()
            values.append((_robust((predicted-(self.rawp@self.align0.T+self.trans0))/s.position_scale_m)*w[...,None]/math.sqrt(3)).reshape(b,-1))
            rotation=rotation_log((distal@self.mr0).transpose(-1,-2)@(self.align0@self.rawr))
            values.append((_robust(rotation/s.orientation_scale_rad)*w[...,None]/math.sqrt(3)).reshape(b,-1))
        prior=rotation_log(exp_so3(self.seed_pose).transpose(-1,-2)@exp_so3(pose))
        values.append(((prior/s.pose_prior_rad)*self.free.any(-1)[...,None]*(.35 if self.observations_enabled else 1.)/math.sqrt(45)).reshape(b,-1))
        if self.previous_correction is not None:
            continuity=(angles-self.seed_angles-self.previous_correction)*self.free
            values.append((continuity/s.correction_continuity_rad*math.sqrt(.033/self.interval_s)/math.sqrt(45)).reshape(b,-1))
        w=self.witnesses[0]
        a=(v[:,w['source_ids']]*w['source_bc'][None,...,None]).sum(-2)
        c=(v[:,w['target_ids']]*w['target_bc'][None,...,None]).sum(-2)
        if a.shape[1]:
            collision=torch.relu(.001-((a-c)*w['normal']).sum(-1))/.001
            values.append(collision*math.sqrt(s.collision_weight/a.shape[1]))
            values.append(torch.linalg.vector_norm(collision,ord=8,dim=1,keepdim=True)*math.sqrt(4*s.collision_weight))
        # Generic residual groups contact position terms before normal terms.
        positions=[];normals=[]
        for i,pair in self.contacts:
            anchors=[pair['a'],pair['b']]
            index=torch.as_tensor(self.model.faces[[a['source_face'] for a in anchors]],device=v.device)
            bc=self.t([a['barycentric'] for a in anchors]);tri=v[:,index]
            points=(tri*bc[None,...,None]).sum(-2)
            normal=torch.nn.functional.normalize(torch.linalg.cross(tri[:,:,1]-tri[:,:,0],tri[:,:,2]-tri[:,:,0]),dim=-1)
            positions.append((points[:,0]-points[:,1])/.002*math.sqrt(20/(max(1,len(self.contacts))*3)))
            normals.append((normal[:,0]+normal[:,1])*math.sqrt(10/(max(1,len(self.contacts))*3)))
        values.extend(positions);values.extend(normals)
        return torch.cat(values,dim=-1)


def _check(cancel, deadline):
    if cancel is not None and cancel.is_set(): raise InterruptedError('MANO optimization cancelled')
    return deadline is not None and time.monotonic() >= deadline


def bounded_lm(objective, initial, *, iterations=6, cancel=None, deadline=None, blocks=None):
    """Projected, damped GN. Every accepted step lowers the same residual cost.

    Shared and independent frame blocks avoid a dense all-frame normal matrix.
    Exact autograd Jacobians are computed for one block at a time.
    """
    x = initial.detach().clone().clamp(objective.lower, objective.upper)
    damping = .05; accepted = 0; timed_out = False
    runtime=not objective.training and not objective.shared and len(objective.rows)==1
    verified=None;verified_cost=float('inf');retained_verified=False
    def remember_verified():
        nonlocal verified,verified_cost
        if runtime:
            with torch.no_grad():cost=float(objective.residual(x).square().sum())
            if np.isfinite(cost) and cost<verified_cost:verified=x.clone();verified_cost=cost
    blocks = blocks or [torch.arange(len(x), device=x.device)]
    for _ in range(iterations):
        if _check(cancel, deadline): timed_out = True; break
        if _ or not torch.equal(x,objective.initial): objective.refresh(x)
        remember_verified()
        for ids in blocks:
            if _check(cancel, deadline): timed_out = True; break
            if not len(ids): continue
            def evaluate(z): return objective.residual(x.scatter(0, ids, z))
            z = x[ids].detach().requires_grad_(True)
            # Forward mode scales with the small variable block, not the much
            # larger surface/contact residual dimension.
            if not objective.training and not objective.shared and len(objective.rows)==1:
                with torch.no_grad():
                    eye=torch.eye(len(z),device=z.device)*.001
                    batch=torch.cat((x[None],x[None].expand(len(z),-1).scatter(1,ids[None].expand(len(z),-1),z.detach()[None]+eye),
                                     x[None].expand(len(z),-1).scatter(1,ids[None].expand(len(z),-1),z.detach()[None]-eye)))
                    batched=objective.runtime_batch(batch)
                    residual=batched[0]
                    jac=((batched[1:1+len(z)]-batched[1+len(z):])/.002).T
            else:
                residual=evaluate(z)
                jac = torch.autograd.functional.jacobian(evaluate, z, vectorize=True, strategy='forward-mode').detach()
            if not torch.isfinite(residual).all(): raise ValueError('Non-finite MANO residual')
            if not torch.isfinite(jac).all(): raise ValueError('Non-finite MANO Jacobian')
            h = jac.T @ jac; gradient = jac.T @ residual.detach()
            movable=~(((z.detach()<=objective.lower[ids]+1e-6)&(gradient>0))|((z.detach()>=objective.upper[ids]-1e-6)&(gradient<0)))
            step=torch.zeros_like(z)
            if not movable.any(): continue
            reduced=h[movable][:,movable]
            step[movable]=torch.linalg.solve(reduced+damping*torch.diag(reduced.diag().clamp_min(1.)),-gradient[movable])
            step *= (.3/step.norm().clamp_min(1e-12)).clamp(max=1)
            cost = float(residual.detach().square().sum())
            rates=x.new_tensor([1.,.5,.25,.125,.0625,.03125,.015625,.0078125])
            trials=x[None].expand(len(rates),-1).clone()
            trials[:,ids]=(z.detach()[None]+step[None]*rates[:,None]).clamp(objective.lower[ids],objective.upper[ids])
            if not objective.training and not objective.shared and len(objective.rows)==1:
                with torch.no_grad():trial_values=objective.runtime_batch(trials).square().sum(-1)
            else:
                with torch.no_grad():trial_values=torch.stack([objective.residual(t).square().sum() for t in trials])
            for k in torch.argsort(trial_values).tolist():
                trial,value=trials[k],trial_values[k]
                if torch.isfinite(value) and float(value) < cost:
                    x = trial; accepted += 1; damping = max(damping*.7, 1e-5); break
            else: damping = min(damping*4, 1e5)
        if timed_out: break
    objective.refresh(x)
    if runtime and verified is not None:
        with torch.no_grad():final_cost=float(objective.residual(x).square().sum())
        if verified_cost<final_cost:
            x=verified;objective.refresh(x);retained_verified=True
    with torch.no_grad():objective.residual(x, record=True)
    return x, dict(accepted_steps=accepted, budget_exhausted=timed_out,retained_verified_incumbent=retained_verified, cost=sum(objective.last_terms.values()), cost_terms=objective.last_terms)


def solve_calibration(objective, cancel, progress, max_seconds=300,resume_phases=None):
    deadline = time.monotonic()+max_seconds
    x = torch.nn.Parameter(objective.initial.clone())
    opt = torch.optim.Adam([x], lr=.01)
    stages = [('reference', {'open', 'together', 'mcp90', 'hook'}),
              ('pinches', {'open', 'together', 'mcp90', 'hook', *('pinch_'+f for f in FINGERS[1:])}),
              ('all_gestures', {r['stage'] for r in objective.rows})]
    reports = []; best = x.detach().clone(); best_cost = float('inf')
    original_weights = objective.weights.clone(); original_contacts = list(objective.contacts)
    steps = objective.settings.adam_steps
    personalized=getattr(objective,'personalized',False) and objective.shared
    frozen_geometry=[];observability={}
    for phase_index,(phase, included) in enumerate(stages):
        if personalized and phase!='reference':included={r['stage'] for r in objective.rows}
        # A single objective and full parameter state survive all phases.
        gate = objective.t([r['stage'] in included for r in objective.rows])
        objective.weights = original_weights*gate[:, None]
        objective.weights=objective.weights/objective.weights.sum().clamp_min(1e-12)
        objective.phase_stages=included
        objective.contacts = [(i, p) for i, p in original_contacts if objective.rows[i]['stage'] in included]
        phase_best=x.detach().clone();phase_cost=float('inf')
        frozen=torch.zeros_like(x,dtype=torch.bool)
        if personalized:
            if phase=='reference':frozen[46:objective.shared_size]=True
            else:
                frozen[:10]=True
                if not observability:
                    from scipy.linalg import qr
                    ids=torch.arange(46,objective.shared_size,device=x.device)
                    def data_residual(z):return objective.residual(x.detach().scatter(0,ids,z),measurement_only=True)
                    jac=torch.autograd.functional.jacobian(data_residual,x[ids].detach(),vectorize=True,strategy='forward-mode').detach().cpu().numpy()
                    if not np.isfinite(jac).all():raise ValueError('Non-finite geometry observability Jacobian')
                    _,r,pivots=qr(jac,mode='economic',pivoting=True)
                    diagonal=np.abs(np.diag(r));threshold=max(float(diagonal.max(initial=0))*1e-4,1e-6)
                    rank=int((diagonal>threshold).sum());frozen_geometry=sorted(int(i) for i in pivots[rank:])
                    observability=dict(method='conditional_geometry_measurement_jacobian_pivoted_QR',rank=rank,parameters=len(ids),threshold=threshold,
                                       frozen_coordinates=frozen_geometry,diagonal=diagonal.tolist(),
                                       note='Conditional on current pose and registration; not a claim of complete anatomical identifiability')
                    with torch.no_grad():
                        for k in frozen_geometry:x[46+k]=objective.initial[46+k]
                if frozen_geometry:frozen[torch.tensor(frozen_geometry,device=x.device)+46]=True
        frozen_values=x.detach().clone()
        # Gesture supervision is retained in every phase; the phases gate data
        # fitting, never disable global collision safety.
        completed=min(steps,int((resume_phases or {}).get(phase,0)))
        performed=0;phase_started=time.monotonic()
        for step in range(completed,steps):
            if _check(cancel, deadline): break
            if step % objective.settings.collision_refresh == 0: objective.refresh(x.detach())
            opt.zero_grad(); residual = objective.residual(x); loss = residual.square().sum()
            if not torch.isfinite(loss): raise ValueError('Non-finite MANO calibration loss')
            if float(loss.detach())<phase_cost:
                phase_cost=float(loss.detach());phase_best=x.detach().clone()
            loss.backward()
            if x.grad is None or not torch.isfinite(x.grad).all(): raise ValueError('Non-finite MANO calibration gradient')
            torch.nn.utils.clip_grad_norm_([x], 100.)
            x.grad[frozen]=0
            opt.step()
            performed+=1
            with torch.no_grad():
                x.clamp_(objective.lower, objective.upper);x[frozen]=frozen_values[frozen]
            if step % 20 == 0: progress(dict(state='running', phase=phase, progress=.08+.65*(phase_index+(step+1)/steps)/3, message=f'参数辨识 {phase_index+1}/3 · {step+1}/{steps} 步',estimated_remaining_s=(time.monotonic()-phase_started)*(steps-step-1)/performed,eta_scope='stage'))
        with torch.no_grad(): x.copy_(phase_best)
        opt.state.clear()
        reports.append(dict(phase=phase, steps=completed+performed,resumed_steps=completed,state='budget_exhausted' if time.monotonic()>=deadline else 'completed',
                            parameter_vector=phase_best.detach().cpu().tolist()))
        if time.monotonic()>=deadline: break
    objective.weights = original_weights; objective.contacts = original_contacts
    del objective.phase_stages
    # Save the whole state against the final objective, with current witnesses.
    objective.refresh(x.detach(), contacts=True)
    with torch.no_grad(): best_cost = float(objective.residual(x).square().sum()); best = x.detach().clone()
    counts = objective.free.reshape(len(objective.rows), -1).sum(1).cpu().tolist()
    shared_ids=torch.arange(objective.shared_size,device=x.device)
    if personalized:
        shared_ids=shared_ids[shared_ids>=10]
        if frozen_geometry:shared_ids=shared_ids[~torch.isin(shared_ids,torch.tensor(frozen_geometry,device=x.device)+46)]
    blocks = [shared_ids] if objective.shared else []
    start = objective.shared_size
    for count in counts:
        blocks.append(torch.arange(start, start+count, device=x.device)); start += count
    progress(dict(state='running',progress=.74,message='有界 LM 联合精修',estimated_remaining_s=None,eta_scope='stage'))
    refined, lm = bounded_lm(objective, best, iterations=objective.settings.lm_steps, cancel=cancel, deadline=deadline, blocks=blocks)
    objective.install(refined)
    return dict(revision=SOLVER_REVISION, phases=reports, lm=lm, budget_exhausted=time.monotonic()>=deadline,
                observability=observability,loss_weights=dict(contact_position=objective.settings.calibration_contact_position_weight,
                  contact_normal=objective.settings.calibration_contact_normal_weight,pinch_observation=objective.settings.pinch_observation_multiplier),
                cost_terms=objective.last_terms, parameter_count=len(refined),
                parameter_snapshot=dict(vector=refined.detach().cpu().tolist(),shared_size=objective.shared_size,
                                        frame_timestamps_ms=[r['timestamp_ms'] for r in objective.rows],
                                        free_coordinates=objective.free.cpu().tolist(),
                                        shared_reference=dict(sensor_joint_position=objective.mp0.cpu().tolist(),mount_rotation=objective.mr0.cpu().tolist(),
                                                              raw_rotation=objective.align0.cpu().tolist(),raw_translation=objective.trans0.cpu().tolist()) if objective.shared else None),
                training_correction_limit_policy='physical_joint_limits_same_as_registered_runtime',
                gesture_masks={r['stage']:objective.mask_np[i].astype(int).tolist() for i,r in enumerate(objective.rows)},
                shared_beta_affects_whole_hand=True)


def solve_pose(model, frame, pose, root, *, tactile=True, iterations=None, budget_ms=None, active_fingers=None):
    """No gesture hints; non-finite candidates never replace the current frame."""
    settings = SolverSettings(); began = time.monotonic()
    signature=(frame['timestamp_ms'],np.asarray(pose).tobytes(),np.asarray(root).tobytes(),model.shape_version,
               model.binding_version if model.mapping else None,tactile,iterations,budget_ms,
               tuple(sorted(active_fingers)) if active_fingers is not None else None,
               np.asarray(model.raw_rotation).tobytes(),np.asarray(model.raw_translation).tobytes(),
               np.asarray(model.sensor_offsets()).tobytes(),np.asarray(model.mount_rot).tobytes(),bool(model.flat_reference),
               json.dumps({k:frame.get(k) for k in ('fingers','contact','tactile_timestamp_ms')},sort_keys=True))
    last=getattr(model,'_mano_solver_cache',None)
    if last is not None and last[0]==signature:
        model._solver_decoded=last[4];model._mano_solver_previous=last[1].copy()
        report=copy.deepcopy(last[2]);report.update(cache_hit=True,elapsed_ms=0.)
        return last[1].copy(),report,copy.deepcopy(last[3])
    cache = getattr(model, '_solver_collision', None)
    if cache is None: cache = model._solver_collision = SurfaceCollision(model)
    view=model
    if model.device.type=='cuda':
        # A 25-DoF single-hand solve is latency/launch bound on CUDA. Evaluate
        # its small batched Jacobian on one CPU thread, while all full-surface
        # winding queries and the offline batched calibration remain on CUDA.
        key=(model.shape_version,model.binding_version if model.mapping else None)
        cached=getattr(model,'_runtime_cpu_view',None)
        if cached is None or cached[0]!=key:
            view=copy.copy(model);view.device=torch.device('cpu');view.model=copy.deepcopy(model.model).to('cpu')
            if model.mapping:view._material_contact_topology=(key,model.topology())
            model._runtime_cpu_view=(key,view)
        else:view=cached[1]
        for field in ('flat_reference','raw_rotation','raw_translation','mount_pos','mount_rot','sensor_joint_position','mounting_calibrated','mounting_estimated'):
            setattr(view,field,getattr(model,field))
    previous = getattr(model, '_mano_solver_previous', None)
    warm_available=previous is not None and model.tracking_time is not None and 0 < frame['timestamp_ms']-model.tracking_time < 250
    previous_angles=getattr(model,'_mano_solver_previous_angles',None)
    previous_seed=getattr(model,'_mano_solver_previous_seed_angles',None)
    correction=previous_angles-previous_seed if warm_available and previous_angles is not None and previous_seed is not None else None
    interval=(frame['timestamp_ms']-model.tracking_time)/1000 if warm_available else None
    rates=np.full(5,settings.correction_rate_deg_s)
    previous_frame=getattr(model,'last_frame',None)
    if warm_available and previous_frame and 'fingers' in previous_frame and 'fingers' in frame:
        a=Rotation.from_quat([previous_frame['fingers'][f]['quaternion_xyzw'] for f in FINGERS]).as_matrix()
        b=Rotation.from_quat([frame['fingers'][f]['quaternion_xyzw'] for f in FINGERS]).as_matrix()
        step=np.rad2deg(Rotation.from_matrix(a.transpose(0,2,1)@b).magnitude())
        rates=np.where(step>3.,np.minimum(360.,rates+2*step/interval),rates)
    objective = ManoObjective(view, [frame], np.asarray(pose)[None], np.asarray(root)[None],
                              tactile=tactile, collision=cache, settings=settings,active_fingers=active_fingers,
                              previous_correction=correction,interval_s=interval,correction_rates=rates)
    initial = objective.initial
    escape_seed=False
    # Initialization and correspondence queries are reported in total latency;
    # they must not consume the entire numerical budget before the first step.
    default_budget=settings.runtime_budget_ms if warm_available else settings.cold_budget_ms
    default_iterations=settings.runtime_iterations if warm_available else settings.cold_iterations
    deadline = time.monotonic()+(default_budget if budget_ms is None else budget_ms)/1000
    x, report = bounded_lm(objective, initial, iterations=default_iterations if iterations is None else iterations, deadline=deadline)
    q,v,j,r=objective.last_surface
    model._mano_solver_previous_angles=objective.unpack(x)[-1][0].detach().cpu().numpy()
    model._mano_solver_previous_seed_angles=objective.seed_angles[0].detach().cpu().numpy()
    q = q[0].cpu().numpy(); model._mano_solver_previous = q.copy()
    ordered=torch.cat((j,v[:,list(TIPS)]),1)[:,model.ordered]
    model._solver_decoded=tuple(a[0].cpu().numpy() for a in (v,j,r,ordered))
    collision = objective.collision_reports[0]
    records = []
    for i, pair in objective.contacts:
        points = []; normals = []
        for a in (pair['a'], pair['b']):
            tri = v[i, model.faces[a['source_face']]]
            points.append((tri*objective.t(a['barycentric'])[:, None]).sum(0))
            normals.append(torch.nn.functional.normalize(torch.linalg.cross(tri[1]-tri[0], tri[2]-tri[0]), dim=0))
        gap = float((points[0]-points[1]).norm())*1000; dot = float((normals[0]*normals[1]).sum())
        records.append(dict(patch_a=pair['a']['patch_id'], patch_b=pair['b']['patch_id'], regions=pair['regions'],
                            patch_indices=pair['patches'], evidence=pair['evidence'], applied=bool(report['accepted_steps']),
                            point_a_m=points[0].cpu().tolist(), point_b_m=points[1].cpu().tolist(),
                            gap_before_mm=float(np.linalg.norm(np.array(pair['a']['position_m'])-pair['b']['position_m']))*1000,
                            gap_mm=gap, normal_dot=dot, satisfied=gap<=2 and dot<-.1,
                            target_overlap_mm=0., permitted_overlap_mm=2., force_units='relative_response_sum'))
    satisfied=collision['constraints_satisfied'] and all(p['satisfied'] for p in records)
    from .side_contact import side_material_pairs
    side_diagnostics={}
    for f in range(4):
        name=FINGERS[f]+'_'+FINGERS[f+1]
        try:
            sections=side_material_pairs(view,v[0].cpu().numpy(),j[0].cpu().numpy(),finger_pairs=[f])
            side_diagnostics[name]=dict(available=True,section_gaps_mm=[p['gap_m']*1000 for p in sections],
                                        normal_dots=[p['normal_dot'] for p in sections],evidence='geometry_only_not_contact_detection')
        except ValueError as exc:side_diagnostics[name]=dict(available=False,reason=str(exc),evidence='geometry_only_not_contact_detection')
    source_tracking=frame.get('skeleton',{}).get('tracking_errors',{})
    source_ok=all(p['satisfied'] for p in source_tracking.values())
    report.update(revision=SOLVER_REVISION, status='source_tracking_violation' if not source_ok else 'constraint_violation' if not satisfied else 'ok',
                  source_tracking_satisfied=source_ok,source_tracking_errors=source_tracking,
                  side_contact_diagnostics=side_diagnostics,
                  constraints_satisfied=satisfied, contact_constraints_satisfied=all(p['satisfied'] for p in records),collision=collision,
                  elapsed_ms=(time.monotonic()-began)*1000, gesture_labels_used=False,
                  anatomical_chart='source_thumb_urdf_axes_v1' if objective.thumb_chart is not None else 'deterministic_straight_bone_reference_v1',jacobian='batched_central_difference',
                  correction_limit_deg=math.degrees(objective.correction_limit_rad) if objective.correction_limit_rad is not None else None,
                  correction_limit_policy='registered_physical_joint_limits' if objective.observations_enabled else 'unregistered_8deg_trust_region',
                  tactile_active_fingers=[FINGERS[f] for f in objective.tactile_active_fingers],
                  correction_rate_limit_deg_s=settings.correction_rate_deg_s,
                  motion_supported_correction_rates_deg_s=rates.tolist() if objective.observations_enabled else [settings.correction_rate_deg_s]*5,
                  continuity_policy='transport_previous_correction_on_current_source_pose',
                  pose_compute_device=str(view.device),collision_compute_device=str(cache.device),
                  collision_escape_seed=escape_seed,
                  output_policy='current_best_finite_pose_with_warning')
    matched={a['patch'] for _,p in objective.contacts for a in (p['a'],p['b'])}
    contact = dict(policy=SOLVER_REVISION, applied=bool(records and report['accepted_steps']), pairs=records,
                   unmatched_patches=[a['patch_id'] for _,a in objective.contact_anchors if a['patch'] not in matched],
                   active_fingers=[FINGERS[f] for f in objective.tactile_active_fingers],target_overlap_mm=0.)
    model._mano_solver_cache=(signature,q.copy(),copy.deepcopy(report),copy.deepcopy(contact),model._solver_decoded)
    return q, report, contact
