"""Independent solver formulas plus real ADMM integration on mjwarp dynamics."""
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA solver')

from retarget.fa.adamw import AdamW
from retarget.humanoid_optimizers import LineSearchAdamW, MPPI


def quadratic(x, gradient):
    value = x.square().sum(dim=1)
    return value, 2*x if gradient else None


def update(solver, x, evaluate=quadratic, **kwargs):
    return solver.update(np.asarray(x,dtype=float),evaluate,iterations=1,horizon=1,
                         advance=0.,evaluate_on_device=True,**kwargs)


def test_armijo_backtracks_independently_and_satisfies_sufficient_decrease():
    solver=LineSearchAdamW(device='cuda:0',learning_rate=4.,weight_decay=0.,max_trials=8)
    result,_,diag=update(solver,[[1.],[.1]])
    step=diag['steps'][0]
    np.testing.assert_allclose(step['alpha'],[.25,.03125])
    assert step['accepted']==[True,True]
    np.testing.assert_allclose(result[:,0],[0.,-.025],atol=1e-7)
    assert np.all(np.asarray(step['value_after']) <= np.asarray(step['value_before'])+
                  solver.armijo*np.asarray(step['alpha'])*step['directional_derivative'])


def test_armijo_matches_shared_adamw_proposal_when_full_step_is_accepted():
    x=np.array([[1.,-.5],[.4,1.2]])
    options=dict(device='cuda:0',learning_rate=.05,weight_decay=.01)
    original=AdamW(**options);line=LineSearchAdamW(**options)
    a,_,_=update(original,x);b,_,diag=update(line,x)
    np.testing.assert_allclose(a,b,atol=1e-15)
    for u,v in zip(original._cuda_window_memory,line._cuda_window_memory):
        np.testing.assert_allclose(u.cpu(),v.cpu(),atol=1e-15)
    assert diag['steps'][0]['alpha']==[1.,1.]


def test_armijo_rejection_keeps_state_and_ascent_momentum_uses_descent_fallback():
    line=LineSearchAdamW(device='cuda:0',learning_rate=100.,weight_decay=0.,max_trials=1)
    result,_,diag=update(line,[[1.]])
    np.testing.assert_array_equal(result,[[1.]])
    assert diag['steps'][0]['accepted']==[False]
    assert int(line._cuda_window_memory[2][0])==1
    line.lr=.1
    line._cuda_window_memory=(torch.tensor([[-100.]],device='cuda:0',dtype=torch.float64),
                              torch.ones((1,1),device='cuda:0',dtype=torch.float64),
                              torch.tensor([10],device='cuda:0'))
    result,_,diag=update(line,[[1.]],continue_window=True)
    assert diag['steps'][0]['descent_fallback']==[True]
    assert diag['steps'][0]['directional_derivative'][0]<0
    assert result[0,0]<1


def test_mppi_matches_independent_gibbs_mean_and_best_sample_safeguard():
    seed=14;n=11;sigma=.5;temperature=2.
    solver=MPPI(samples=n,noise_std=sigma,control_scale=1.,temperature=temperature,seed=seed)
    x=np.tile([1.5,-.5],(4,1))
    calls=[]
    def evaluate(x,gradient):
        calls.append(gradient)
        assert not gradient
        return quadratic(x,gradient)
    result,_,diag=update(solver,x,evaluate)
    generator=torch.Generator(device='cuda:0').manual_seed(seed)
    noise=(sigma*torch.randn((n,2),device='cuda:0',dtype=torch.float64,generator=generator)).cpu().numpy()
    samples=x[0]+noise;cost=np.sum(samples*samples,axis=1)
    weights=np.exp(-(cost-cost.min())/temperature);weights/=weights.sum()
    mean=x[0]+weights@noise
    candidates=np.vstack([x[0],mean,samples[np.argmin(cost)]])
    expected=candidates[np.argmin(np.sum(candidates*candidates,axis=1))]
    np.testing.assert_allclose(result,np.tile(expected,(4,1)),rtol=1e-14,atol=1e-14)
    np.testing.assert_allclose(diag['steps'][0]['effective_sample_size'],1/np.sum(weights**2),rtol=1e-13)
    assert diag['steps'][0]['forward_batches']==3  # padding is excluded
    assert not any(calls) and diag['gradient_evaluations']==0


def test_mppi_weighted_mean_mode_exports_the_mean_not_the_best_sample():
    solver=MPPI(samples=11,noise_std=.5,control_scale=1.,temperature=2.,seed=14,update_rule='weighted-mean')
    result,_,diag=update(solver,np.tile([1.5,-.5],(4,1)))
    generator=torch.Generator(device='cuda:0').manual_seed(14)
    noise=(.5*torch.randn((11,2),device='cuda:0',dtype=torch.float64,generator=generator)).cpu().numpy()
    samples=np.array([1.5,-.5])+noise;cost=np.sum(samples*samples,axis=1)
    weights=np.exp(-(cost-cost.min())/2.);weights/=weights.sum()
    expected=weights@samples
    np.testing.assert_allclose(result,np.tile(expected,(4,1)),atol=1e-14)
    assert not np.allclose(expected,samples[np.argmin(cost)])
    assert diag['export']=='weighted_mean' and not diag['full_objective_safeguard']


def test_mppi_ignores_invalid_samples_and_requires_common_objective():
    solver=MPPI(samples=19,noise_std=1.,control_scale=1.,seed=4)
    def evaluate(x,gradient):
        value=x.square().sum(dim=1)
        value[x[:,0]<0]=torch.nan
        return value,None
    result,_,diag=update(solver,[[.5],[.5]],evaluate)
    assert np.isfinite(result).all()
    assert 0 < diag['steps'][0]['valid_samples'] < 19
    assert diag['steps'][0]['value_after']<=.25
    with pytest.raises(ValueError,match='same nominal'):
        update(solver,[[.5],[.6]],evaluate)
    def inconsistent(x,gradient):
        return x.square().sum(dim=1)+torch.arange(len(x),device=x.device),None
    with pytest.raises(ValueError,match='fixed ADMM objective'):
        update(solver,[[.5],[.5]],inconsistent)


def test_mppi_all_failed_samples_preserves_incumbent():
    solver=MPPI(samples=5,noise_std=1.,control_scale=1.)
    def evaluate(x,gradient):
        assert not gradient
        return torch.where((x==.5).all(dim=1),.25,torch.inf).double(),None
    result,_,diag=update(solver,[[.5],[.5]],evaluate)
    np.testing.assert_array_equal(result,[[.5],[.5]])
    assert diag['steps'][0]['valid_samples']==0
    assert diag['steps'][0]['effective_sample_size']==0


def test_invalid_action_solver_configuration_fails_before_mjwarp_allocation():
    from types import SimpleNamespace
    from retarget.humanoid import Settings,ProjectionADMM
    with pytest.raises(ValueError,match='Unknown humanoid action solver'):
        ProjectionADMM(SimpleNamespace(settings=Settings(action_solver='bad')))
    with pytest.raises(ValueError,match='persistence'):
        ProjectionADMM(SimpleNamespace(settings=Settings(action_solver='mppi',adamw_persist=True)))


@pytest.mark.parametrize('name',['adamw-ls','mppi'])
def test_mjwarp_two_round_warm_mpc_exports_exact_evaluated_trajectory(name):
    from retarget.humanoid import HumanoidProblem,ProjectionADMM,Settings
    root=Path(__file__).resolve().parents[1]/'data/spider/processed/omomo/unitree_g1/humanoid_object/move_largebox'
    settings=Settings(horizon=2,knot_stride=1,substeps=2,multistarts=2,
                      admm_iterations=2,action_iterations=1,action_solver=name,
                      contact_samples=3,contact_qp_iterations=16,mppi_samples=5)
    p=HumanoidProblem(root/'scene.xml',root/'0/trajectory_kinematic.npz',settings=settings)
    solver=ProjectionADMM(p);state=p.initial_state(2);previous=p.aref[2]
    for first in [3,4]:
        old=state.qpos.copy()
        rollout,history=solver.solve(state,[first],previous)
        np.testing.assert_array_equal(state.qpos,old)
        assert all(row['action_cost_after']<=row['action_cost_before']+1e-6 for row in history)
        assert all(row['contact_cost_after']<=row['contact_cost_before']+1e-6 for row in history)
        assert history[-1]['backward_calls']==(0 if name=='mppi' else 2)
        b=solver.backend
        selected=history[-1]['selected']
        np.testing.assert_array_equal(rollout['next_state'].qpos,
            b.steps[settings.substeps-1].recorded.result.qpos.numpy()[selected])
        np.testing.assert_array_equal(rollout['controls_dense'],
            np.stack([a.numpy()[selected] for a in b.controls[:settings.substeps]]))
        if name=='mppi':
            for a in [b.variables,b.targets.z,b.targets.dual,b.targets.forces]:
                array=a.numpy()
                axis=0 if a is b.variables else 1
                np.testing.assert_allclose(np.take(array,0,axis=axis),np.take(array,1,axis=axis),atol=0.)
        state=rollout['next_state'];previous=solver.actions[0]
