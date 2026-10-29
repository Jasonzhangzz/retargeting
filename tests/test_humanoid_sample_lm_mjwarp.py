"""The sampled action LM exports the same MJWarp trajectory it accepted."""
from pathlib import Path
import numpy as np
import pytest


@pytest.mark.parametrize('reuse',[True,False])
def test_sample_lm_common_contacts_export_and_next_window(reuse):
    torch=pytest.importorskip('torch')
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from retarget.humanoid import HumanoidProblem,ProjectionADMM,Settings
    from run_humanoid import task_paths
    root=Path(__file__).resolve().parents[1]/'data/spider'
    p=HumanoidProblem(*task_paths(root,'humanoid-object'),settings=Settings(
        horizon=2,knot_stride=1,multistarts=2,admm_iterations=2,action_iterations=1,
        action_solver='sample-lm',sample_lm_samples=6,sample_lm_rank=3,sample_lm_reuse_samples=reuse,
        contact_samples=3,contact_qp_iterations=16))
    solver=ProjectionADMM(p);state=p.initial_state(2);previous=p.aref[2]
    for first in (3,4):
        old=state.qpos.copy();rollout,history=solver.solve(state,[first],previous)
        np.testing.assert_array_equal(state.qpos,old)
        assert all(r['action_cost_after']<=r['action_cost_before']+1.e-5 for r in history)
        assert history[-1]['backward_calls']==0
        b=solver.backend;selected=history[-1]['selected']
        np.testing.assert_array_equal(rollout['next_state'].qpos,b.steps[p.settings.substeps-1].recorded.result.qpos.numpy()[selected])
        np.testing.assert_array_equal(rollout['controls_dense'],np.stack([c.numpy()[selected] for c in b.controls[:p.settings.substeps]]))
        np.testing.assert_array_equal(b.variables.numpy()[0],b.variables.numpy()[1])
        state=rollout['next_state'];previous=solver.actions[0]


def test_exact_dense_base_has_the_same_interpolation_finite_difference_gradient_and_batch_cost():
    torch=pytest.importorskip('torch')
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from retarget.humanoid import HumanoidProblem,Settings
    from retarget.humanoid_mjwarp import MJWarpHumanoid
    from retarget.humanoid_terms import stack_host
    from retarget.humanoid_lm_oracle import MJWarpResidualOracle
    from run_humanoid import task_paths
    root=Path(__file__).resolve().parents[1]/'data/spider'
    p=HumanoidProblem(*task_paths(root,'humanoid-object'),settings=Settings(
        horizon=2,knot_stride=1,multistarts=1,trajectory_cache=False))
    b=MJWarpHumanoid(p);b.prepare(p.initial_state(2),3,p.aref[2]);b.targets.penalty.zero_()
    original=np.zeros(b.variables.shape);b.evaluate(original,False)
    dense=stack_host(b.controls)[:,0].copy()
    dense[1,0]+=.007;dense[2,1]-=.005
    b.use_dense_control_base(dense)
    value,g=b.evaluate(original,True)
    np.testing.assert_array_equal(stack_host(b.controls)[:,0],dense)
    direction=g/np.linalg.norm(g)
    for epsilon in (.001,.003):
        plus=b.evaluate(original+epsilon*direction,False)[0]
        minus=b.evaluate(original-epsilon*direction,False)[0]
        np.testing.assert_allclose(np.sum(g*direction,axis=1),(plus-minus)/(2*epsilon),rtol=.06,atol=3.e-4)
    oracle=MJWarpResidualOracle(b,worlds=3);oracle.sync()
    x=torch.zeros((2,b.variables.shape[1]),device='cuda:0',dtype=torch.float64)
    cost,residual=oracle(x,True)
    np.testing.assert_allclose(cost.cpu(),value.repeat(2),rtol=3.e-6,atol=1.e-5)
    torch.testing.assert_close(residual.double().square().sum(dim=1),cost,rtol=3.e-6,atol=1.e-5)
    correction=np.zeros(p.model.nu);correction[0]=.005
    b.use_dense_control_base(dense,previous_delta=correction);b.evaluate(original,False)
    expected=dense.copy()
    for i,(left,right,fraction) in enumerate(b.maps):
        if left<0:expected[i]+=(1-fraction)*correction
    np.testing.assert_allclose(stack_host(b.controls)[:,0],expected,rtol=0,atol=5.e-8)
    oracle.sync()
    boundary_cost,_=oracle(x,True)
    direct=b.evaluate(original,False)[0]
    np.testing.assert_allclose(boundary_cost.cpu(),direct.repeat(2),rtol=3.e-6,atol=1.e-5)
    b.use_dense_control_base(None);b.evaluate(original,False)
    assert not np.array_equal(stack_host(b.controls)[:,0],dense)
