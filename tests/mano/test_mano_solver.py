"""Lightweight FK/gradients, GPU-compatible LM and shared CEM regression."""
import numpy as np
import torch,pytest
from dataclasses import replace
from tests.mano.helpers import surface
from retarget.env.grasp import mano_solver as solver
from retarget.env.grasp.mano_solver import ManoDecoder,residuals,kinematic_loss,lm_ik

def decoder(f):return ManoDecoder(f['model'].base.path,'cpu',24)

def test_parallel_fk_and_five_tip_path_match_mano_reference(fixture):
    fast=decoder(fixture)
    q=torch.randn(3,30,generator=torch.Generator().manual_seed(71))*.2
    z=lambda *shape:torch.zeros(shape,dtype=q.dtype)
    reference,joints,frames=fast.base(z(10),z(),z(5),z(5),fast.angles(q).reshape(-1,15,3))
    root=solver.rotmat(q[:,3:6])
    reference=(root[:,None]@reference[:,:,:,None]).squeeze(-1)+q[:,None,:3]
    joints=(root[:,None]@joints[:,:,:,None]).squeeze(-1)+q[:,None,:3]
    ordered=torch.cat((joints,reference[:,fast.ik_tip_ids]),1)[:,fast.ordered_joint_ids]
    actual,points,rotations=fast(q,full=True)
    torch.testing.assert_close(actual,reference,atol=2e-7,rtol=2e-5)
    torch.testing.assert_close(points,ordered,atol=2e-7,rtol=2e-5)
    torch.testing.assert_close(rotations,root[:,None]@frames,atol=2e-7,rtol=2e-5)
    torch.testing.assert_close(fast.joints_only(q),ordered,atol=2e-7,rtol=2e-5)

def test_thin_fk_gradient_and_residual_objective_match(fixture):
    f=fixture;fast=decoder(f)
    q=(torch.randn(2,30,generator=torch.Generator().manual_seed(8))*.08).requires_grad_()
    target=torch.tensor(f['target'],dtype=torch.float32)
    original=solver.skeleton_loss(f['model'],q,target)[0]
    thin=kinematic_loss(fast,q,target)
    torch.testing.assert_close(original,thin,atol=2e-6,rtol=2e-5)
    a=torch.autograd.grad(original.sum(),q)[0];b=torch.autograd.grad(thin.sum(),q)[0]
    assert torch.isfinite(b).all()
    torch.testing.assert_close(a,b,atol=2e-4,rtol=2e-4)

def test_lm_autodiff_jacobian_and_monotone_descent(fixture):
    fast=decoder(fixture)
    desired=torch.randn(2,30,generator=torch.Generator().manual_seed(21))*.08;desired[:,:3]*=.02
    target=fast.joints_only(desired).detach();initial=torch.zeros_like(desired)
    settings=solver.Settings(ik_steps=12,early_stop=False)
    before=kinematic_loss(fast,initial,target)
    solution,count,diagnostics=lm_ik(fast,initial,target,settings)
    assert torch.isfinite(solution).all() and (count==12).all()
    assert torch.all(diagnostics['final_cost']<before*.15)
    history=torch.stack(diagnostics['history'])
    assert (history[1:]<=history[:-1]+1e-6).all()


def test_production_defaults_and_removed_legacy_choices():
    settings=solver.Settings()
    from retarget.solver.cem import DEFAULTS
    assert (settings.ik_steps,DEFAULTS.sampling.population,DEFAULTS.sampling.iterations)==(15,128,24)
    assert DEFAULTS.resources.batch_size==1024 and not hasattr(solver,"cem_update")
    assert not hasattr(solver,'adam_thin_ik') and not hasattr(solver,'scheduled_std')
    for kwargs in ({'lm_damping':0},{'lm_max_step':0},{'lm_relative_tol':float('nan')}):
        with pytest.raises(ValueError):solver.Settings(**kwargs)
    for kwargs in ({'ik_method':'adam_full'},{'search_method':'elite'},{'cem_noise_schedule':'geometric'}):
        with pytest.raises(TypeError):solver.Settings(**kwargs)
