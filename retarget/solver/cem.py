"""Current CEM: per-pose stopping and retention of feasible patch grasps."""
import math
import torch


from dataclasses import dataclass, field, asdict

@dataclass(frozen=True)
class SamplingSettings:
    population: int = 128
    iterations: int = 24
    worlds: int = 128
    elite_fraction: float = .05
    temperature: float = .1
    final_noise: float = .2

    def __post_init__(self):
        if any(isinstance(v,bool) or not isinstance(v,int) or v<1 for v in (self.population,self.iterations,self.worlds)) or self.population<2:
            raise ValueError('Positive CEM budget required, with at least two candidates')
        if not 0<self.elite_fraction<=1 or not 0<self.final_noise<=1 or not math.isfinite(self.temperature) or self.temperature<=0:
            raise ValueError('Invalid CEM sampling parameters')

    @property
    def temp(self):return self.temperature

    @property
    def noise_ratio(self):return self.final_noise

@dataclass(frozen=True)
class ContactSettings:
    pos_noise: float = .005
    rot_noise: float = .05
    joint_noise: float = .1
    max_pos_delta: float = .1
    max_rot_delta: float = 1.
    penetration_scale_m: float = .001
    penetration_weight: float = 1.e4
    hand_object_allowance_m: float = .005
    soft_penetration_weight: float = .1
    collision_clearance_m: float = 0.
    collision_margin_m: float = 0.
    penetration_tolerance_m: float = .005
    nconmax: int = 256

    def __post_init__(self):
        if any(not math.isfinite(v) or v<0 for v in vars(self).values()):raise ValueError('Invalid contact geometry parameters')
        if self.penetration_scale_m<=0 or self.nconmax<1:raise ValueError('Positive collision scale/capacity required')
        if self.collision_margin_m<self.collision_clearance_m:raise ValueError('Collision margin must cover clearance')


@dataclass(frozen=True)
class PatchSettings:
    patch_radius_m: float = .010
    anchor_tolerance_m: float = .005
    surface_tolerance_m: float = .002
    coverage_fraction: float = .8

    def __post_init__(self):
        for name, value in vars(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(name+' must be positive and finite')
        if self.coverage_fraction > 1:
            raise ValueError('Invalid coverage fraction')


    @property
    def hand_object_allowance_m(self):return DEFAULTS.contact.hand_object_allowance_m

    @property
    def self_allowance_m(self):return DEFAULTS.contact.penetration_tolerance_m


@dataclass(frozen=True)
class RolloutSettings:
    horizon: int = 5
    timestep: float = 0.01
    friction: float = 0.6
    mass: float = 0.1
    gravity: tuple = (0., 0., -9.81)
    initial_angular_velocity: tuple = (0.5, 0.5, 0.5)
    translation_scale_m: float = 0.002
    rotation_scale_deg: float = 2.
    minimum_improvement: float = 0.05
    contact_capture_m: float = 0.002
    cost_mode: str = 'velocity'
    linear_speed_threshold_m_s: float = .1
    angular_speed_threshold_rad_s: float = 2.

    def __post_init__(self):
        if self.horizon != 5 or isinstance(self.horizon, bool):
            raise ValueError('This scoring protocol requires exactly five steps')
        if self.cost_mode != 'velocity':
            raise ValueError('Unknown ComFree cost mode')
        for name in ('timestep', 'mass', 'translation_scale_m', 'rotation_scale_deg',
                     'linear_speed_threshold_m_s','angular_speed_threshold_rad_s'):
            v = getattr(self, name)
            if not math.isfinite(v) or v <= 0:
                raise ValueError(name + ' must be positive and finite')
        for name in ('friction', 'contact_capture_m'):
            v = getattr(self, name)
            if not math.isfinite(v) or v < 0:
                raise ValueError(name + ' must be nonnegative and finite')
        if self.friction == 0:
            raise ValueError('Contact response requires positive friction')
        if not 0 < self.minimum_improvement < 1:
            raise ValueError('minimum_improvement must lie in (0,1)')
        for name in ('gravity', 'initial_angular_velocity'):
            v = getattr(self, name)
            if len(v) != 3 or not all(math.isfinite(x) for x in v):
                raise ValueError(name + ' must contain three finite components')
        if not any(self.gravity) and not any(self.initial_angular_velocity):
            raise ValueError('A nonzero gravity or initial perturbation is required')


@dataclass(frozen=True)
class SpringSettings:
    normal_stiffness_n_m: float = 500.
    tangent_stiffness_n_m: float = 500.
    normal_damping_ns_m: float = 5.
    tangent_damping_ns_m: float = 5.
    substeps: int = 4
    newton_iterations: int = 8
    linear_residual_m_s: float = 1.e-5
    angular_residual_rad_s: float = 1.e-4

    def __post_init__(self):
        for name,value in vars(self).items():
            if not math.isfinite(value) or value<=0:raise ValueError(name+' must be positive and finite')
        for name in ['substeps','newton_iterations']:
            if isinstance(getattr(self,name),bool) or not isinstance(getattr(self,name),int):raise ValueError(name+' must be an integer')


@dataclass(frozen=True)
class InitializationSettings:
    ik_steps: int = 15
    pca_components: int = 24
    early_stop: bool = True
    stop_absolute_tol: float = 1.0e-5
    lm_damping: float = 0.01
    lm_max_step: float = 5.0
    lm_relative_tol: float = 1.0e-4
    lm_gradient_tol: float = 1.0e-5

    def __post_init__(self):
        for key, value in vars(self).items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(key + " must be finite")
        for key in (
            "pca_components",
        ):
            if getattr(self, key) < 1:
                raise ValueError(key + " must be positive")
        if (
            not 1 <= self.pca_components <= 45
            or self.ik_steps < 0
        ):
            raise ValueError("Invalid solver dimensions")
        for key in (
            "lm_damping",
            "lm_max_step",
        ):
            if getattr(self, key) <= 0:
                raise ValueError(key + " must be positive")
        if self.lm_relative_tol < 0 or self.lm_gradient_tol < 0 or self.stop_absolute_tol < 0:
            raise ValueError("Invalid LM tolerances")



@dataclass(frozen=True)
class Resources:
    batch_size: int = 1024
    candidate_chunk: int = 2048

@dataclass(frozen=True)
class Parameters:
    resources: Resources = field(default_factory=Resources)
    sampling: SamplingSettings = field(default_factory=SamplingSettings)
    contact: ContactSettings = field(default_factory=ContactSettings)
    patch: PatchSettings = field(default_factory=PatchSettings)
    rollout: RolloutSettings = field(default_factory=RolloutSettings)
    spring: SpringSettings = field(default_factory=SpringSettings)
    initialization: InitializationSettings = field(default_factory=InitializationSettings)

DEFAULTS = Parameters()

@torch.no_grad()
def _solve_patch_cem(problem,adapter,settings,seed,*,early_stop=True):
    mean=problem.mean.clone();best=mean.clone();b,d=mean.shape
    rows=torch.arange(b,device=mean.device)
    best_cost,parts=adapter.validate(best,rows)
    accepted=parts['patch_joint_accepted'].clone()
    active=~accepted if early_stop else torch.ones_like(accepted)
    evaluations=torch.ones(b,device=mean.device,dtype=torch.long)
    updates=torch.zeros_like(evaluations);checks=torch.ones_like(evaluations)
    generators=[torch.Generator(device=mean.device).manual_seed(seed+i) for i in range(b)]
    elite_count=max(1,math.ceil(settings.population*settings.elite_fraction))
    retain_feasible=getattr(adapter,'retain_feasible',False)
    for iteration in range(settings.iterations):
        ids=active.nonzero(as_tuple=True)[0]
        if not len(ids):break
        # Host control only: no pose arithmetic leaves CUDA. Independent RNGs
        # keep a pose's candidate sequence invariant when other poses stop.
        noise=torch.stack([torch.randn((settings.population,d),device=mean.device,dtype=mean.dtype,
            generator=generators[i]) for i in ids.tolist()])
        values=mean[:,None].expand(-1,settings.population,-1).clone()
        values[ids]=mean[ids,None]+noise*problem.sigma[ids,None]*settings.noise_ratio**(iteration/settings.iterations)
        values[ids,0]=best[ids]
        if settings.population>1:values[ids,1]=mean[ids]
        values=problem.project(values)[ids]
        flat=values.flatten(0,1);source_rows=ids.repeat_interleave(settings.population)
        if hasattr(adapter,'evaluate_candidates'):
            score,gate=adapter.evaluate_candidates(flat,source_rows)
        else:
            score=adapter.evaluate(flat,source_rows);gate=torch.ones_like(score,dtype=torch.bool)
        cost=score.reshape(len(ids),-1);gate=gate.reshape_as(cost)
        cost=torch.where(torch.isfinite(cost),cost,torch.inf)
        local=torch.arange(len(ids),device=mean.device);choice=cost.argmin(-1)
        if retain_feasible:
            feasible=gate&torch.isfinite(cost)
            feasible_choice=torch.where(feasible,cost,torch.inf).argmin(-1)
            choice=torch.where(feasible.any(-1),feasible_choice,choice)
        minimum=cost[local,choice]
        if retain_feasible:
            passing=gate[local,choice]&torch.isfinite(minimum)
            improved=(passing&~accepted[ids])|((passing==accepted[ids])&(minimum<best_cost[ids]))
        else:improved=minimum<best_cost[ids]
        best[ids]=torch.where(improved[:,None],values[local,choice],best[ids])
        best_cost[ids]=torch.where(improved,minimum,best_cost[ids])
        if retain_feasible:accepted[ids]=torch.where(improved,passing,accepted[ids])
        elite=cost.argsort(dim=-1,stable=True)[:,:elite_count];scores=cost.gather(1,elite)
        finite=torch.isfinite(scores);number=finite.sum(-1).clamp_min(1)
        clean=torch.where(finite,scores,0.)
        average=clean.sum(-1)/number
        variance=torch.where(finite,(clean-average[:,None]).square(),0.).sum(-1)/number
        lowest=torch.where(finite,scores,torch.inf).amin(-1)
        logits=-(scores-lowest[:,None])/(variance.sqrt().clamp_min(1.e-12)[:,None]*settings.temp)
        logits=torch.where(finite,logits,-torch.inf);has=finite.any(-1)
        weights=torch.softmax(torch.where(has[:,None],logits,0.),-1)
        update=(weights[:,:,None]*values.gather(1,elite[:,:,None].expand(-1,-1,d))).sum(1)
        mean[ids]=torch.where(has[:,None],update,mean[ids])
        evaluations[ids]+=settings.population;updates[ids]+=1
        if early_stop:
            # Stability and penetration come from this exact candidate's score.
            # Only an improved, otherwise eligible pose needs an extra skin
            # patch query; an unchanged failing patch need not be rechecked.
            eligible=ids[improved&gate[local,choice]]
            if len(eligible):
                _,parts=adapter.validate(best[eligible],eligible)
                evaluations[eligible]+=1;checks[eligible]+=1
                accepted[eligible]=parts['patch_joint_accepted']
                active[eligible]&=~accepted[eligible]
    final_cost,parts=adapter.validate(best,rows)
    evaluations+=1;checks+=1
    return dict(parameters=best,cost=final_cost,iterations=updates,evaluations=evaluations,
        validation_calls=checks,early_stopped=early_stop&(updates<settings.iterations)&parts['patch_joint_accepted'],
        passed=parts['patch_joint_accepted'],parts=parts)


def _solve_generic(problem,settings,seed=0):
    """Use the same CEM update for non-grasp tasks without a grasp acceptance gate."""
    from dataclasses import replace
    from .base import Result, NoValidCandidate, evaluate
    singleton=problem.mean.ndim==1
    mean=problem.mean[None] if singleton else problem.mean
    sigma=problem.sigma.expand_as(mean)
    def project(values):
        return problem.project(values[0])[None] if singleton else problem.project(values)
    class Adapter:
        def evaluate(self,values,rows):
            shaped=values if singleton else values.reshape(len(mean),-1,values.shape[-1])
            output=evaluate(problem,shaped)
            return torch.where(output.valid,output.search,torch.inf).reshape(-1)
        def validate(self,values,rows):
            return self.evaluate(values,rows),dict(patch_joint_accepted=torch.zeros(len(values),device=values.device,dtype=torch.bool))
    current=replace(problem,mean=mean,sigma=sigma,project=project)
    state=_solve_patch_cem(current,Adapter(),settings,seed,early_stop=False)
    best=state['parameters'][0] if singleton else state['parameters']
    output=evaluate(problem,best[None] if singleton else best[:,None])
    if not bool(output.valid.all()):raise NoValidCandidate([])
    cost=float(output.cost.reshape(-1)[0]) if singleton else output.cost.reshape(-1)
    count=int(state['evaluations'].sum())
    history=[dict(iteration=i+1) for i in range(int(state['iterations'].max()))]
    return Result(best,cost,history,int(state['iterations'].sum())*settings.population,count+len(mean),
        metadata=dict(implementation='current_patch_cem',grasp_gate=False))


def solve_patch_cem(problem,adapter,seed=7):
    """Solve a patch grasp using the canonical algorithm parameters."""
    return _solve_patch_cem(problem,adapter,DEFAULTS.sampling,seed,early_stop=True)


def solve(problem,seed=7):
    """Use the same default CEM for task-supplied candidate scores."""
    return _solve_generic(problem,DEFAULTS.sampling,seed)
