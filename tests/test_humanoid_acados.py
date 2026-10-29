"""Check actual objective parity, constrained QPs and mjwarp GN globalization."""
from pathlib import Path
import numpy as np
import pytest
from retarget.humanoid import HumanoidProblem,Settings,ProjectionADMM
from run_humanoid import task_paths


def require_acados():
    from retarget.support.acados_env import ensure_acados_env
    try:ensure_acados_env()
    except RuntimeError as error:
        if 'Built acados not found' in str(error):pytest.skip(str(error))
        raise


def problem(task='humanoid-object',**settings):
    scene,ref=task_paths(Path(__file__).resolve().parents[1]/'data/spider',task)
    return HumanoidProblem(scene,ref,settings=Settings(horizon=2,knot_stride=1,substeps=2,
                           multistarts=3,admm_iterations=1,action_iterations=1,**settings))


def test_acados_qp_matches_constrained_kkt():
    require_acados()
    from retarget.humanoid_acados import AcadosQuadratic
    H=np.array([[3.,.6,-.2],[.6,2.,.4],[-.2,.4,1.]])
    g=np.array([-5.,2.,-.1]);lower=np.array([-.2,-.4,-.5]);upper=np.array([.3,.2,.6])
    step,diag=AcadosQuadratic(3).solve(H,g,lower,upper)
    assert diag['status']==diag['qp_status']==0
    assert (step>=lower-1e-7).all() and (step<=upper+1e-7).all()
    gradient=H@step+g
    interior=(step>lower+1e-5)&(step<upper-1e-5)
    assert np.max(np.abs(gradient[interior]))<1e-6
    assert (gradient[step<=lower+1e-5]>=-1e-6).all()
    assert (gradient[step>=upper-1e-5]<=1e-6).all()


def test_shifted_outside_limit_knots_do_not_make_qp_infeasible():
    from retarget.humanoid_acados import knot_step_bounds
    lo,hi=knot_step_bounds(np.array([0.,20.,-20.]),.05,.1)
    assert (lo<=0).all() and (hi>=0).all()
    # Raw endpoints remain free to overshoot; actual dense actuator controls
    # are still clamped by the original mjwarp build_controls kernel.
    np.testing.assert_array_equal(lo,[-2.,-2.,-2.])
    np.testing.assert_array_equal(hi,[2.,2.,2.])


@pytest.mark.parametrize('task',['humanoid','humanoid-object'])
def test_elementwise_residuals_equal_full_scalar_cost(task):
    torch=pytest.importorskip('torch')
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from retarget.humanoid_residuals import Residuals
    p=problem(task);solver=ProjectionADMM(p)
    solver.solve(p.initial_state(2),[3],p.aref[2]);b=solver.backend
    # The complete solved contact block has nonzero dual/auxiliary forces.
    x=np.random.default_rng(4).normal(0,.04,b.variables.shape)
    value,_=b.evaluate(x,False)
    residual=Residuals(b).evaluate()
    np.testing.assert_allclose(residual.double().square().sum(dim=(0,2)).cpu(),value,rtol=3e-6,atol=2e-6)


def test_mjwarp_gn_jacobian_globalization_and_export():
    torch=pytest.importorskip('torch')
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    require_acados()
    p=problem(action_solver='acados-gn',acados_difference='central')
    # Pin this local-derivative fixture independently of tuned defaults.
    # The reference contact geometry is nonsmooth at face/support switches.
    from dataclasses import replace
    p.weights=replace(p.weights,hand_position=200.)
    solver=ProjectionADMM(p);b=solver.backend;optimizer=solver.action_optimizer
    state=p.initial_state(2);previous=p.aref[2]
    for frame in [3,4]:
        rollout,history=solver.solve(state,[frame],previous)
        row=history[-1];steps=row['action_optimizer']['steps']
        assert steps[0]['status']==0
        assert steps[0]['value_after']<=steps[0]['value_before']+1e-6
        assert steps[0]['step_max_rad']<=p.settings.acados_trust_radius+1e-7
        assert row['finite_difference_gradients']
        assert row['backward_calls']==0
        assert np.isfinite(rollout['qpos']).all()
        assert (rollout['controls']>=p.lower-1e-6).all()
        assert (rollout['controls']<=p.upper+1e-6).all()
        state=rollout['next_state'];previous=solver.actions[0]
    x=np.broadcast_to(solver.warm,b.variables.shape).copy()
    H,g,value,diag=optimizer.linearization.linearize(torch.as_tensor(x,device='cuda:0'),.005)
    exact_value,exact_grad=b.evaluate(x,True)
    assert abs(value-exact_value[0])<2e-4
    # A 0.25 mrad perturbation stays within the measured local contact branch;
    # larger differences can cross it after the world-hand task update.
    # Check the actual residual-secant quadratic independently. At contact
    # switches this model need not equal a scalar secant at another epsilon.
    lin=optimizer.linearization
    rr=lin.residuals.evaluate().permute(1,0,2).reshape(lin.backend.worlds,-1).double().cpu().numpy()
    r0=lin.primary_residuals.evaluate().permute(1,0,2).reshape(b.worlds,-1)[0].double().cpu().numpy()
    for coordinate in (0,len(g)-1):
        slope=(rr[1+coordinate]-rr[1+len(g)+coordinate])/.01
        delta=1e-4
        qplus=np.sum((r0+delta*slope)**2)
        qminus=np.sum((r0-delta*slope)**2)
        np.testing.assert_allclose(g[coordinate],(qplus-qminus)/(2*delta),rtol=2e-6,atol=1e-5)
    assert np.linalg.eigvalsh(H).min()>-1e-6
    # Preserve the selected evaluated state, score and solver warm start,
    # including when the candidate is not row 0. A fresh contact replay is a
    # separate numerical experiment and is not a bit-identical export oracle.
    trial=x.copy();trial[:,0]+=[.03,-.04,.07]
    values,_=b.evaluate(trial,False)
    expected=b.stages[2].job.qpos.numpy()[1].copy()
    expected_warm=b.warmstart_history[2].numpy()[1].copy()
    expected_value=values[1]
    chosen=np.broadcast_to(trial[1],trial.shape).copy()
    optimizer.export(torch.as_tensor(chosen,device='cuda:0'),1)
    np.testing.assert_array_equal(b.stages[2].job.qpos.numpy(),np.broadcast_to(expected,b.stages[2].job.qpos.shape))
    assert b._external_rollout
    v1,g1=b.evaluate(chosen,True);b._cached=None
    np.testing.assert_allclose(v1,expected_value,rtol=2e-6,atol=3e-6)
    np.testing.assert_array_equal(b.stages[2].job.qpos.numpy(),np.broadcast_to(expected,b.stages[2].job.qpos.shape))
    np.testing.assert_array_equal(b.warmstart_history[2].numpy(),np.broadcast_to(expected_warm,b.warmstart_history[2].shape))
    v2,g2=b.evaluate(chosen,True)
    assert np.isfinite(v2).all()
    assert np.isfinite(g1).all() and np.isfinite(g2).all()
