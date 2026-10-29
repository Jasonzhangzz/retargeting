"""Independent LM equations, sampling reuse, and nonlinear safeguards."""
import numpy as np
import pytest
import torch

from retarget.humanoid_sample_lm import SampleLM, lm_step


def test_lm_direction_matches_independent_numpy_normal_equations():
    rng=np.random.default_rng(52);J=rng.normal(size=(5,13));r=rng.normal(size=13);damping=.07
    expected=np.linalg.solve(J@J.T+damping*np.diag(np.maximum(np.diag(J@J.T),1.e-8)),-J@r)
    step,H,g=lm_step(torch.tensor(J),torch.tensor(r),damping)
    np.testing.assert_allclose(step,expected,rtol=1.e-12,atol=1.e-12)
    np.testing.assert_allclose(H,J@J.T);np.testing.assert_allclose(g,J@r)


@pytest.mark.parametrize('samples,rank,reuse',[(0,5,False),(0,3,False),(12,5,False),(12,3,True)])
def test_lm_matches_actual_residual_objective_and_decreases(samples,rank,reuse):
    A=torch.tensor(np.random.default_rng(2).normal(size=(5,9)),dtype=torch.float64)
    target=torch.arange(9,dtype=torch.float64)/9
    calls=[]
    def oracle(x,residuals):
        calls.append(len(x));r=x@A-target
        return r.square().sum(dim=1),r if residuals else None
    x=torch.ones((1,5),dtype=torch.float64)
    result,diag=SampleLM(samples=samples,rank=rank,reuse_sample_residuals=reuse,
        control_scale=1.,noise_std=.2,trust_radius=.5,iterations=2).solve(x,oracle)
    assert oracle(result,False)[0][0]<oracle(x,False)[0][0]
    assert diag['value_after']<=diag['value_after_sampling']<=diag['value_before']
    assert all(s['value_after']<=s['value_before'] for s in diag['steps'])
    if reuse:assert diag['steps'][0]['derivative_trajectories']==0
    assert all(s['proposal_max_displacement_rad']<=.5+1.e-12 for s in diag['steps'])


def test_non_decreasing_proposals_keep_incumbent_without_gradient_callback():
    calls=[]
    def oracle(x,residuals):
        calls.append(len(x))
        # At this strict local minimum, all finite probes have a greater cost.
        r=1+x.square()
        return r.square().sum(dim=1),r if residuals else None
    x=torch.zeros((1,3),dtype=torch.float64)
    result,diag=SampleLM(rank=3,iterations=2).solve(x,oracle)
    torch.testing.assert_close(result,x)
    assert all(not s['accepted'] for s in diag['steps'])
    assert diag['analytic_backward_calls']==0


def test_invalid_sampling_residuals_do_not_contaminate_secant_fit():
    def oracle(x,residuals):
        r=x-.1;value=r.square().sum(dim=1)
        bad=x[:,0]<0
        value[bad]=torch.inf;r[bad]=torch.nan
        return value,r if residuals else None
    x=torch.ones((1,4),dtype=torch.float64)*.5
    result,diag=SampleLM(samples=20,rank=3,reuse_sample_residuals=True,
        noise_std=1.,control_scale=1.,trust_radius=.5).solve(x,oracle)
    assert torch.isfinite(result).all()
    assert diag['value_after']<=diag['value_before']
