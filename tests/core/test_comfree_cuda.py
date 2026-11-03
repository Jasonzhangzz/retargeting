"""Fused CUDA response versus independent Torch math and finite differences."""
import pytest
import torch

from retarget.env.grasp.comfree_equilibrium import equilibrium_response
from retarget.env.grasp.projected_comfree import ClosureSettings, ProjectedComFreeClosure

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def geometry(n, *, single_side=False, batch=3, torsion=.02):
    generator = torch.Generator().manual_seed(173)
    normal = torch.nn.functional.normalize(torch.randn(batch, n, 3, generator=generator,
                                                      dtype=torch.float64), dim=-1).cuda()
    if n == 2:
        normal[:, 1] = -normal[:, 0]
    if single_side:
        normal[:] = normal.new_tensor([1., 0., 0.])
    closure = ProjectedComFreeClosure(ClosureSettings(equilibrium_backend="cuda_kernel", torsional_friction_m=torsion),
        center=[0., 0., 0.], radius=.05, object_mass=.1, object_inertia=[.0001] * 3)
    rays, _, _, _, drag, loads, _ = closure._geometry(normal * .05, normal)
    alpha = torch.linspace(.8, 2.3, n, device="cuda", dtype=torch.float64)[None].expand(batch, -1).clone()
    return closure, normal, alpha, rays, drag, loads


@pytest.mark.parametrize("n,single_side", [(2, False), (6, False), (16, False), (6, True)])
def test_fused_response_force_balance_sensitivities_and_chunking(n, single_side):
    from retarget.env.grasp.comfree_cuda import equilibrium_response_cuda
    _, _, alpha, rays, drag, loads = geometry(n, single_side=single_side)
    reference = equilibrium_response(alpha, rays, drag, loads, jacobian=True)
    result = equilibrium_response_cuda(alpha, rays, drag, loads, jacobian=True)
    assert result.valid.all()
    torch.testing.assert_close(result.balance,
        torch.einsum("brj,bsr->bsj", rays, result.row_forces) + loads, atol=2.e-8, rtol=2.e-8)
    for name in ("row_forces", "balance", "row_force_jacobian", "balance_jacobian"):
        torch.testing.assert_close(getattr(result, name), getattr(reference, name), atol=2.e-5, rtol=2.e-5)
        chunks = [equilibrium_response_cuda(alpha[i:i+1], rays[i:i+1], drag[i:i+1], loads, jacobian=True)
                  for i in range(len(alpha))]
        torch.testing.assert_close(getattr(result, name), torch.cat([getattr(x, name) for x in chunks]),
                                   atol=1.e-9, rtol=1.e-9)
    if single_side:
        assert not result.converged[:, 1:].all()
        assert result.balance[:, 1:].norm(dim=-1).amax() >= 1.
    if n == 2:
        assert result.converged.all()  # Spin rays supply the previously missing torque.


def test_fused_gradient_matches_force_effort_finite_difference_and_isolation():
    from retarget.env.grasp.comfree_cuda import equilibrium_response_cuda
    _, _, alpha, rays, drag, loads = geometry(2, batch=2)
    result = equilibrium_response_cuda(alpha, rays, drag, loads, jacobian=True)
    derivative = 2. * (result.row_forces[..., None] * result.row_force_jacobian).sum((1, 2))
    epsilon = 1.e-5
    for i in range(2):
        increment = torch.zeros_like(alpha)
        increment[:, i] = epsilon
        plus = equilibrium_response_cuda(alpha + increment, rays, drag, loads).row_forces.square().sum((1, 2))
        minus = equilibrium_response_cuda(alpha - increment, rays, drag, loads).row_forces.square().sum((1, 2))
        torch.testing.assert_close(derivative[:, i], (plus - minus) / (2 * epsilon), atol=2.e-5, rtol=2.e-5)
    rays[1, 0, 0] = torch.nan
    invalid = equilibrium_response_cuda(alpha, rays, drag, loads, jacobian=True)
    assert invalid.valid.tolist() == [True, False]
    torch.testing.assert_close(invalid.row_forces[0], result.row_forces[0])


def test_fused_bounded_gn_graph_replay_is_deterministic_and_accepts_descent():
    closure, normal, _, _, _, _ = geometry(2, batch=5)
    one = closure.solve(normal * .05, normal)
    two = closure.solve(normal * .05, normal)
    torch.testing.assert_close(one.preloads, two.preloads, atol=0, rtol=0)
    assert one.valid.all() and one.certified.all()
    assert ((one.preloads >= 0.) & (one.preloads <= 20.)).all()
    assert (one.objective_history[:, 1:] <= one.objective_history[:, :-1] + 1.e-7).all()


def test_disabling_spin_restores_the_missing_two_point_torque_axis():
    from retarget.env.grasp.comfree_cuda import equilibrium_response_cuda
    _, _, alpha, rays, drag, loads = geometry(2, torsion=0.)
    response = equilibrium_response_cuda(alpha, rays, drag, loads, jacobian=True)
    reference = equilibrium_response(alpha, rays, drag, loads, jacobian=True)
    assert response.valid.all()
    assert (response.balance[:, 1:].norm(dim=-1).amax(1) > .05).all()
    torch.testing.assert_close(response.balance, reference.balance, atol=2.e-5, rtol=2.e-5)
