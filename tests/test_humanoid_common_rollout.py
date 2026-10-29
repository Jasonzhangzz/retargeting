"""Single-world MPPI evaluation preserves full costs, states and gradients."""
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest

torch=pytest.importorskip('torch')
pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')

from retarget.humanoid import HumanoidProblem,Settings,ProjectionADMM
from retarget.humanoid_common_rollout import CommonRollout
from retarget.humanoid_mjwarp import MJWarpHumanoid


def problem(settings):
    task=Path(__file__).resolve().parents[1]/'data/spider/processed/omomo/unitree_g1/humanoid_object/move_largebox'
    return HumanoidProblem(task/'scene.xml',task/'0/trajectory_kinematic.npz',settings=settings)


@pytest.mark.parametrize('cache',[True,False])
def test_common_value_export_and_gradient_workspace_rebuild(cache):
    settings=Settings(horizon=2,knot_stride=1,substeps=2,multistarts=3,trajectory_cache=cache)
    p=problem(settings);b=MJWarpHumanoid(p)
    b.prepare(p.initial_state(2),3,p.aref[2])
    x=torch.full(b.variables.shape,.012,dtype=torch.float64,device='cuda:0')
    # Nonzero common targets prove the helper copies all dynamic target axes.
    z=np.broadcast_to(b.anchors[:,None],b.targets.z.shape+(3,)).copy()
    b.targets.z.assign(z.astype(np.float32));b.targets.dual.fill_(.003);b.dirty=True
    expected=b.evaluate(x,False)[0].cpu().numpy();snapshot=b.snapshots()
    state=b.steps[1].recorded.result.qpos.numpy().copy()
    common=CommonRollout(b);common.begin()
    value=common.value(x)[0].cpu().numpy()
    np.testing.assert_allclose(value,expected,rtol=2e-6,atol=2e-6)
    common.export(x)
    np.testing.assert_allclose(b.snapshots()['com'],snapshot['com'],rtol=0,atol=1e-7)
    np.testing.assert_allclose(b.snapshots()['realized_wrench'],snapshot['realized_wrench'],rtol=2e-6,atol=2e-6)
    np.testing.assert_allclose(b.steps[1].recorded.result.qpos.numpy(),state,rtol=2e-6,atol=2e-6)
    assert b._external_rollout
    before=b.forward_calls
    value1,g1=b.evaluate(x,True)
    assert b.forward_calls>before  # central differences use physical probes
    b._cached=None
    value2,g2=b.evaluate(x,True)
    np.testing.assert_allclose(value1.cpu(),value2.cpu(),rtol=2e-6,atol=2e-6)
    assert float(torch.linalg.vector_norm(g1-g2)/torch.linalg.vector_norm(g2))<.005


@pytest.mark.parametrize('rule',['safeguarded','weighted-mean'])
def test_same_samples_same_two_round_warm_closed_loop(rule):
    settings=Settings(horizon=2,knot_stride=1,substeps=2,multistarts=3,
                      admm_iterations=2,action_iterations=2,action_solver='mppi',
                      contact_samples=3,contact_qp_iterations=16,mppi_samples=7,mppi_update=rule)
    original=ProjectionADMM(problem(settings))
    optimized=ProjectionADMM(problem(replace(settings,mppi_single_rollout=True)))
    states=[solver.problem.initial_state(2) for solver in (original,optimized)]
    previous=[solver.problem.aref[2] for solver in (original,optimized)]
    for frame in [3,4]:
        outputs=[]
        for i,solver in enumerate((original,optimized)):
            rollout,history=solver.solve(states[i],[frame],previous[i]);outputs.append((rollout,history))
            states[i]=rollout['next_state'];previous[i]=solver.actions[0]
        # Independent MJWarp runs need not make identical contact/sample
        # choices. Each mode must export its own exact accepted trajectory.
        for i,solver in enumerate((original,optimized)):
            b=solver.backend;selected=outputs[i][1][-1]['selected']
            np.testing.assert_array_equal(states[i].qpos,b.steps[settings.substeps-1].recorded.result.qpos.numpy()[selected])
            np.testing.assert_array_equal(previous[i],b.controls[settings.substeps-1].numpy()[selected])
            assert np.isfinite(states[i].qpos).all()
        assert outputs[1][1][-1]['backward_calls']==0
        assert outputs[1][1][-1]['common_forward_calls']>0
