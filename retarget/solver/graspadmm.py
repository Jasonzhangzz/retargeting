"""GraspADMM defaults and the public native/GPU solve entry."""
from dataclasses import dataclass,asdict
import torch

@dataclass(frozen=True)
class Settings:
    step: int = 50
    step_obj: int = 1
    step_hand: int = 1
    substep: int = 10
    rho_admm: float = 1000.0
    lr_obj: float = 0.0002
    square_res: bool = True
    fix_hand_local_cp: bool = False
    opt_miu_coef: tuple = (0.1, 0.0)
    spring_stiffness: float = 500.0
    alignment_stiffness: float = 100.0
    damping: float = 0.01
    qp_tolerance: float = 1.0e-8
    qp_max_steps: int = 10000
    filter_miu_coef: tuple = (.6,0.)
    cdist: float = .001
    device: str = 'cpu'
    init_epoch: int = 1
    init_n_final: int = 1
    init_points: int = 1024
    init_inplane: int = 8
    init_iterations: int = 100


DEFAULTS = Settings()

def settings(values=None):
    if values is not None and not isinstance(values,dict):raise ValueError('GraspADMM parameters must be a mapping')
    result={k:v for k,v in asdict(DEFAULTS).items() if k in ('step','substep','step_obj','step_hand','rho_admm','lr_obj','square_res','fix_hand_local_cp','opt_miu_coef','cdist','device','init_epoch','init_n_final','init_points','init_inplane','init_iterations')}
    result['opt_miu_coef']=list(result['opt_miu_coef'])
    if values and any(k not in result or v!=result[k] for k,v in values.items()):
        raise ValueError('GraspADMM uses its single default parameter set')
    return result

@dataclass
class NativeProblem:
    grasp_data: dict
    scene: object
    directory: object

@dataclass
class GPUProblem:
    physics: object
    qpos: object
    hand_bodies: object
    hand_local: object
    object_cpn: object
    surfaces: object
    qp: object
    external: object
    center: object
    hand_body_count: int

def solve(problem, *, progress=None):
    if isinstance(problem,GPUProblem):
        return _solve_gpu(problem.physics,problem.qpos,problem.hand_bodies,problem.hand_local,
            problem.object_cpn,problem.surfaces,problem.qp,problem.external,problem.center,
            settings=DEFAULTS,hand_body_count=problem.hand_body_count,progress=progress)
    from retarget.env.grasp.graspadmm_optimizer import optimize
    if not isinstance(problem,NativeProblem):raise TypeError('Expected NativeProblem or GPUProblem')
    return optimize(problem.grasp_data,problem.scene,problem.directory,settings())

def validate(config):
    from retarget.env.grasp.grasp_admm_runner import validate as implementation
    return implementation(config)

def run(config,output):
    from retarget.env.grasp.grasp_admm_runner import run as implementation
    return implementation(config,output)


def _solve_gpu(physics,q,hand_bodies,hand_local,object_cpn,surfaces,qp,external,center,
                *,settings,hand_body_count,progress=None):
    """Native object search, virtual-force hand integration, then dual update.

    Post-ADMM native filter/squeeze/pregrasp are deliberately separate stages.
    An ADMM state alone is not advertised as a successful native grasp.
    """
    from retarget.env.grasp.native_admm import object_step,update_dual
    from retarget.env.grasp.native_mjwarp import NativeHandSurfaces
    physics.reset(q);initial_state=physics.snapshot();local=hand_local.clone()
    cp_h=physics.positions(hand_bodies,local).detach();cp_o=object_cpn[...,:3].clone();lam=torch.zeros_like(cp_h)
    hand_surfaces=NativeHandSurfaces(physics.cpu,hand_bodies.detach().cpu().tolist(),physics.device,physics.frozen_mesh_arrays)
    active=torch.ones(len(q),device=q.device,dtype=torch.bool);fallback=torch.zeros_like(active)
    numerical_failure=torch.zeros_like(active);steps=torch.zeros(len(q),device=q.device,dtype=torch.int32)
    for iteration in range(settings.step):
        for _ in range(settings.step_obj):
            updated,audit,gradient,finite=object_step(cp_o,cp_h,lam,surfaces,qp,external,center,settings)
            failed=active&(~audit['valid']|~finite)
            fallback|=failed;active&=~failed
            cp_o=torch.where(active[:,None,None],updated,cp_o)
        target=cp_o-lam
        if not settings.fix_hand_local_cp:
            new_local,valid=hand_surfaces.project(physics,hand_bodies,target)
            numerical_failure|=active&~valid.all(-1);active&=valid.all(-1)
            local[...,:3]=torch.where(active[:,None,None],new_local,local[...,:3])
        for _ in range(settings.step_hand):
            physics.apply_forces(hand_bodies,local,target,mask=active)
            physics.advance(settings.substep,mask=active)
        cp_h=physics.positions(hand_bodies,local).detach()
        next_lam=update_dual(lam,cp_h,cp_o)
        lam=torch.where(active[:,None,None],next_lam,lam)
        finite=torch.isfinite(physics.tensor('qpos')).all(-1)&torch.isfinite(lam).all((1,2))&~physics.overflow()
        numerical_failure|=active&~finite;active&=finite;steps+=active.to(steps.dtype)
        if progress is not None:progress(iteration+1,steps,fallback,numerical_failure)
    fallback_active=fallback&~numerical_failure
    physics.restore(initial_state,fallback_active);physics.forward()
    local=torch.where(fallback_active[:,None,None],hand_local,local)
    previous_diff=torch.zeros_like(cp_h[...,0])
    for i in range(20):
        if not bool(fallback_active.any()):break
        current=physics.apply_forces(hand_bodies,local,object_cpn[...,:3],normal=object_cpn[...,3:],mask=fallback_active)
        physics.advance(10,mask=fallback_active)
        improved=((previous_diff-current).amax(-1)>1e-4) if i else torch.ones_like(fallback_active)
        previous_diff=current
        packet=physics.native_contacts(hand_body_count,.002)
        stable=physics.collision_pass(packet,-.002,-.002)
        fallback_active&=~(~improved&stable)
    finite=torch.isfinite(physics.tensor('qpos')).all(-1)&~physics.overflow()
    numerical_failure|=~finite
    return dict(qpos=physics.tensor('qpos').clone(),hand_cpn_b=local,cp_o=cp_o,cp_h=cp_h,lam=lam,
                active=~numerical_failure,fallback_used=fallback,numerical_failure=numerical_failure,
                steps=steps,initial_state=initial_state,hand_bodies=hand_bodies)
