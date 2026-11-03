"""Lightweight FK/gradients, GPU-compatible LM and adaptive CEM regression."""
import numpy as np
import torch,pytest
from dataclasses import replace
from test_mano_solver_runtime import fixture,surface
from retarget.env.grasp import mano_solver as solver
from retarget.env.grasp.mano_solver import ManoDecoder,residuals,kinematic_loss,lm_ik,cem_update,optimize

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

def test_cem_fits_covariance_and_keeps_variance_floor():
    elite=torch.tensor([[[1.,2.],[1.,2.]]]);mean=torch.zeros(1,2);std=torch.ones(1,2);sigma=torch.tensor([1.,2.])
    m,s=cem_update(elite,mean,std,sigma,0.,.1)
    torch.testing.assert_close(m,torch.tensor([[1.,2.]]));torch.testing.assert_close(s,torch.tensor([[.1,.2]]))

def test_small_real_cem_is_finite_monotone_and_counts_budget(fixture):
    f=fixture;fast=decoder(f);fast.configure_contact_vertices({'type':f['fixed'].records})
    fixed=solver.FixedContacts(fast,f['fixed'].records)
    settings=solver.Settings(population=16,iterations=6,ik_steps=4,
        candidate_chunk=32,early_stop=False,warm_start=False)
    out=optimize(fast,f['target'],surface(f),settings,7,fixed,f['points'],f['normals'])
    again=optimize(fast,f['target'],surface(f),settings,7,fixed,f['points'],f['normals'])
    np.testing.assert_allclose(out['variables'],again['variables'],atol=1e-7,rtol=1e-6)
    expected=6*17
    np.testing.assert_array_equal(out['candidate_evaluations'],expected)
    assert np.isfinite(out['vertices']).all() and (np.diff(out['history'],axis=0)<=1e-5).all()
    assert (out['final_cost']<=out['initial_cost']+1e-5).all()

def test_cem_quadratic_converges_with_small_population():
    generator=torch.Generator().manual_seed(17);mean=torch.zeros(1,4);std=torch.ones_like(mean);target=torch.tensor([[.3,-.2,.5,-.4]])
    first=((mean-target)**2).sum()
    for _ in range(16):
        values=mean[:,None]+torch.randn(1,32,4,generator=generator)*std[:,None]
        cost=(values-target[:,None]).square().sum(-1);ids=cost.topk(6,largest=False).indices
        elite=values.gather(1,ids[:,:,None].expand(-1,-1,4))
        mean,std=cem_update(elite,mean,std,torch.ones(4),.2,.005)
    assert ((mean-target)**2).sum()<first*.001

def test_production_defaults_and_removed_legacy_choices():
    settings=solver.Settings()
    assert (settings.ik_steps,settings.population,settings.iterations)==(15,128,24)
    assert (settings.cem_mean_step,settings.cem_mean_step_limit)==(2,3)
    assert not hasattr(solver,'adam_thin_ik') and not hasattr(solver,'scheduled_std')
    for kwargs in ({'lm_damping':0},{'lm_max_step':0},{'cem_momentum':1},{'cem_mean_step':0},{'cem_mean_step_limit':0},{'pos_noise_m':float('nan')}):
        with pytest.raises(ValueError):solver.Settings(**kwargs)
    for kwargs in ({'ik_method':'adam_full'},{'search_method':'elite'},{'cem_noise_schedule':'geometric'}):
        with pytest.raises(TypeError):solver.Settings(**kwargs)


def test_cem_large_mean_steps_stay_in_trust_region(fixture):
    f=fixture;fast=decoder(f);fast.configure_contact_vertices({'type':f['fixed'].records})
    fixed=solver.FixedContacts(fast,f['fixed'].records)
    settings=solver.Settings(population=8,iterations=4,ik_steps=2,
        candidate_chunk=16,early_stop=False,warm_start=False,cem_mean_step=3,cem_mean_step_limit=.5)
    out=optimize(fast,f['target'],surface(f),settings,17,fixed,f['points'],f['normals'])
    assert np.isfinite(out['vertices']).all()
    assert (out['final_cost']<=out['initial_cost']+1e-5).all()
    assert (np.abs(out['variables'][:,6:])<=3+1e-6).all()
