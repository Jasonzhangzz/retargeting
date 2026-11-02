"""Static ComFree witnesses, finite missing axes and preload sensitivities."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import linprog
import torch

from retarget.env.grasp.comfree_equilibrium import equilibrium_response
from retarget.env.grasp.projected_comfree import ClosureSettings, ProjectedComFreeClosure


FIXTURE = json.loads((Path(__file__).parent / "fixtures/projected_comfree_regression.json").read_text())


def geometry(case, dtype=torch.float64, device="cpu"):
    source = FIXTURE[str(case)]
    closure = ProjectedComFreeClosure(ClosureSettings(torsional_friction_m=0.),
        center=source["center"], radius=source["radius"],
        object_mass=source["object_mass"], object_inertia=source["object_inertia"])
    points = torch.tensor([source["points"]], dtype=dtype, device=device)
    normal = torch.tensor([source["outward_normals"]], dtype=dtype, device=device)
    rays, _, _, _, drag, loads, _ = closure._geometry(points, normal)
    return rays, drag, loads


@pytest.mark.parametrize("case", ["1", "6", "9", "10", "11"])
def test_independent_ray_cone_lp_checks_the_original_projection_classification(case):
    rays, _, loads = geometry(case)
    matrix = rays[0].numpy().T
    feasible = []
    for target in loads[1:].numpy():
        result = linprog(np.ones(matrix.shape[1]), A_eq=matrix, b_eq=-target,
                         bounds=(0, None), method="highs")
        feasible.append(result.success)
        if result.success:
            np.testing.assert_allclose(matrix @ result.x, -target, atol=1.e-7)
    assert all(feasible) is FIXTURE[case]["expected_closure"]
    assert np.linalg.matrix_rank(matrix) == (5 if case == "9" else 6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("preload", [0., 2.])
def test_fixed_budget_static_response_removes_four_false_negatives_and_preserves_missing_axis(dtype, preload):
    for case in FIXTURE:
        rays, drag, loads = geometry(case, dtype)
        alpha = torch.full((1, rays.shape[1] // 4), preload, dtype=dtype)
        result = equilibrium_response(alpha, rays, drag, loads, jacobian=True)
        assert result.valid.all()
        assert torch.isfinite(result.balance).all() and torch.isfinite(result.row_forces).all()
        assert (result.row_forces >= 0).all()
        error = result.balance[:, 1:].norm(dim=-1).amax().item()
        assert (error <= .05) is FIXTURE[case]["expected_closure"]
        reconstructed = torch.einsum("brj,bsr->bsj", rays, result.row_forces) + loads
        torch.testing.assert_close(result.balance, reconstructed, atol=5.e-5, rtol=1.e-5)
        assert result.row_force_jacobian.shape == (1, 13, rays.shape[1], alpha.shape[1])
        if case == "6" and preload == 2.:
            # This weak layout really needs large normal forces; it must incur
            # effort cost instead of being mislabeled geometrically infeasible.
            normal_force = result.row_forces.reshape(1, 13, -1, 4).sum(-1)
            assert normal_force.max() > 40.


def sphere_geometry():
    normal = torch.cat((torch.eye(3), -torch.eye(3))).to(torch.float64)[None]
    closure = ProjectedComFreeClosure(ClosureSettings(torsional_friction_m=0.),
                                    center=[0., 0., 0.], radius=.05,
                                    object_mass=.1, object_inertia=[.0001] * 3)
    rays, _, _, _, drag, loads, _ = closure._geometry(normal * .05, normal)
    return rays, drag, loads


@pytest.mark.parametrize("iterations", [2, 16])
def test_exact_iteration_sensitivity_matches_autograd_and_numerical_force_effort_gradient(iterations):
    rays, drag, loads = sphere_geometry()
    alpha = torch.tensor([[.83, 1.07, 1.41, .96, .67, 1.25]], dtype=torch.float64, requires_grad=True)
    generator = torch.Generator().manual_seed(301)
    force_weights = torch.randn((1, 13, 24), generator=generator, dtype=alpha.dtype)
    balance_weights = torch.randn((1, 13, 6), generator=generator, dtype=alpha.dtype)
    result = equilibrium_response(alpha, rays, drag, loads, iterations=iterations, jacobian=True)
    loss = (result.row_forces * force_weights).sum() + .3 * (result.balance * balance_weights).sum()
    automatic, = torch.autograd.grad(loss, alpha)
    analytic = (result.row_force_jacobian * force_weights[..., None]).sum((1, 2))
    analytic += .3 * (result.balance_jacobian * balance_weights[..., None]).sum((1, 2))
    torch.testing.assert_close(analytic, automatic, atol=2.e-8, rtol=2.e-7)

    def effort(value):
        response = equilibrium_response(value, rays, drag, loads, iterations=iterations)
        normal = response.row_forces.reshape(1, 13, 6, 4).sum(-1)
        return normal.square().mean() + .2 * response.balance.square().mean()

    assert torch.autograd.gradcheck(effort, (alpha,), atol=1.e-5, rtol=1.e-4, fast_mode=True)


def test_batch_nonfinite_inputs_do_not_contaminate_a_finite_sample():
    rays, drag, loads = sphere_geometry()
    rays = rays.expand(2, -1, -1).clone()
    rays[1, 0, 0] = torch.nan
    result = equilibrium_response(torch.ones(2, 6, dtype=rays.dtype), rays,
                                  drag.expand(2, -1), loads, jacobian=True)
    assert result.valid.tolist() == [True, False]
    assert result.converged[0].all() and not result.converged[1].any()
    assert torch.isfinite(result.balance).all() and torch.isfinite(result.row_force_jacobian).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_classifies_the_five_projection_cones_and_matches_cpu():
    for case in FIXTURE:
        rays, drag, loads = geometry(case, torch.float32)
        alpha = torch.full((1, rays.shape[1] // 4), 2., dtype=rays.dtype)
        cpu = equilibrium_response(alpha, rays, drag, loads, jacobian=True)
        gpu = equilibrium_response(alpha.cuda(), rays.cuda(), drag.cuda(), loads.cuda(), jacobian=True)
        assert gpu.valid.all()
        error = gpu.balance[:, 1:].norm(dim=-1).amax().item()
        assert (error <= .05) is FIXTURE[case]["expected_closure"]
        torch.testing.assert_close(gpu.row_forces.cpu(), cpu.row_forces, atol=2.e-5, rtol=2.e-5)
        torch.testing.assert_close(gpu.balance.cpu(), cpu.balance, atol=1.e-4, rtol=1.e-4)


@pytest.mark.parametrize("rays_per_contact", [1, 3, 4, 6, 8])
def test_any_fixed_ray_count_keeps_one_preload_per_contact(rays_per_contact):
    # This algebraic cone spans all wrench axes and checks the n-dimensional
    # source distribution independently of any friction geometry constructor.
    basis = torch.cat((torch.eye(6), -torch.eye(6))).to(torch.float64)
    rays = basis.repeat_interleave(rays_per_contact, dim=0)[None]
    drag = torch.ones(rays.shape[:2], dtype=rays.dtype)
    alpha = torch.full((1, 12), 2., dtype=rays.dtype)
    loads = torch.cat((torch.zeros(1, 6), basis))
    result = equilibrium_response(alpha, rays, drag, loads, jacobian=True)
    explicit = equilibrium_response(alpha, rays, drag, loads, rays_per_contact=rays_per_contact)
    torch.testing.assert_close(result.row_forces, explicit.row_forces)
    torch.testing.assert_close(result.balance, torch.zeros_like(result.balance), atol=1.e-7, rtol=0)
    torch.testing.assert_close(result.row_forces[:, 0].reshape(1, 12, rays_per_contact).sum(-1), alpha)
    assert result.row_force_jacobian.shape == (1, 13, 12 * rays_per_contact, 12)


def soft_finger_geometry():
    normal = torch.tensor([[[1., 0., 0.], [-1., 0., 0.]]], dtype=torch.float64)
    closure = ProjectedComFreeClosure(ClosureSettings(torsional_friction_m=0.),
        center=[0., 0., 0.], radius=.05, object_mass=.1, object_inertia=[.0001] * 3)
    base, _, inverse, _, _, loads, _ = closure._geometry(normal * .05, normal)
    inward = -normal
    spin = torch.zeros(1, 2, 2, 6, dtype=normal.dtype)
    spin[..., :3] = inward[:, :, None]
    spin[..., 3:] = inward[:, :, None] * normal.new_tensor([1., -1.])[None, None, :, None] * (.02 / .05)
    rays = torch.cat((base.reshape(1, 2, 4, 6), spin), dim=2).flatten(1, 2)
    inverse_weight = ((rays @ inverse) * rays).sum(-1)
    drag = (closure.settings.impedance / (1 - closure.settings.impedance)) / inverse_weight
    drag *= closure.settings.damping / closure.settings.timestep + closure.settings.stiffness
    return rays, drag, loads


def test_six_ray_coupled_soft_fingers_close_the_missing_torsion_axis():
    rays, drag, loads = soft_finger_geometry()
    assert np.linalg.matrix_rank(rays[0].numpy()) == 6
    for target in loads[1:].numpy():
        assert linprog(np.ones(12), A_eq=rays[0].numpy().T, b_eq=-target,
                       bounds=(0, None), method="highs").success
    result = equilibrium_response(torch.full((1, 2), 2., dtype=rays.dtype), rays, drag, loads, jacobian=True)
    assert result.valid.all() and result.converged.all()
    assert result.balance[:, 1:].norm(dim=-1).max() < 1.e-6
    rows = result.row_forces.reshape(1, 13, 2, 6)
    normal_force = rows.sum(-1)
    tangential_l1 = .6 * ((rows[..., 0] - rows[..., 1]).abs() + (rows[..., 2] - rows[..., 3]).abs())
    twist_nm = .02 * (rows[..., 4] - rows[..., 5]).abs()
    assert (twist_nm <= .02 * normal_force + 1.e-10).all()
    assert (tangential_l1 / .6 + twist_nm / .02 <= normal_force + 1.e-10).all()
    assert twist_nm.max() > .01  # The formerly missing torque uses actual spin support.


def test_six_ray_force_effort_sensitivity_and_autograd_match():
    rays, drag, loads = soft_finger_geometry()
    alpha = torch.tensor([[1.3, 2.1]], dtype=rays.dtype, requires_grad=True)
    response = equilibrium_response(alpha, rays, drag, loads, jacobian=True)
    normal = response.row_forces.reshape(1, 13, 2, 6).sum(-1)
    loss = normal.square().mean()
    automatic, = torch.autograd.grad(loss, alpha)
    derivative = response.row_force_jacobian.reshape(1, 13, 2, 6, 2).sum(3)
    analytic = (2 * normal[..., None] * derivative).mean((1, 2))
    torch.testing.assert_close(analytic, automatic, atol=1.e-8, rtol=1.e-6)

    def effort(value):
        result = equilibrium_response(value, rays, drag, loads)
        return result.row_forces.reshape(1, 13, 2, 6).sum(-1).square().mean()

    assert torch.autograd.gradcheck(effort, (alpha,), fast_mode=True, atol=1.e-5, rtol=1.e-4)


@pytest.mark.parametrize("count,explicit", [(5, None), (0, None), (8, 0), (8, True), (8, 3)])
def test_malformed_ray_counts_fail_before_solving(count, explicit):
    with pytest.raises(ValueError, match="rays|ray"):
        equilibrium_response(torch.ones(1, 2), torch.ones(1, count, 6), torch.ones(1, count),
                             torch.zeros(13, 6), rays_per_contact=explicit)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_six_ray_soft_finger_matches_cpu():
    rays, drag, loads = soft_finger_geometry()
    alpha = torch.tensor([[1.3, 2.1]], dtype=rays.dtype)
    cpu = equilibrium_response(alpha, rays, drag, loads, jacobian=True)
    gpu = equilibrium_response(alpha.cuda(), rays.cuda(), drag.cuda(), loads.cuda(), jacobian=True)
    assert gpu.valid.all() and gpu.converged.all()
    torch.testing.assert_close(gpu.row_forces.cpu(), cpu.row_forces, atol=1.e-8, rtol=1.e-6)
    torch.testing.assert_close(gpu.row_force_jacobian.cpu(), cpu.row_force_jacobian, atol=1.e-8, rtol=1.e-6)


def same_side_soft_finger_geometry(dtype=torch.float64, device="cpu"):
    # The two contacts face almost the same way. Their dual energy is unbounded
    # for unsupported loads even though their wrench matrix has full rank.
    source = FIXTURE["9"]
    closure = ProjectedComFreeClosure(ClosureSettings(torsional_friction_m=.02),
        center=source["center"], radius=source["radius"], object_mass=source["object_mass"],
        object_inertia=source["object_inertia"])
    points = torch.tensor([[[.0022491291165351868, .017852583900094032, -.030970478430390358],
                            [.005608587991446257, .017064237967133522, -.031102020293474197]]],
                          dtype=dtype, device=device)
    normals = torch.tensor([[[.065403051674366, .9978588819503784, 0.],
                             [.32143962383270264, .9469300508499146, 0.]]], dtype=dtype, device=device)
    rays, _, _, _, drag, loads, _ = closure._geometry(points, normals)
    return rays, drag, loads


def test_infeasible_same_side_cone_retains_best_actual_balance_instead_of_unbounded_dual_iterate():
    rays, drag, loads = same_side_soft_finger_geometry()
    matrix = rays[0].numpy().T
    assert np.linalg.matrix_rank(matrix) == 6
    assert not any(linprog(np.ones(12), A_eq=matrix, b_eq=-target,
                           bounds=(0, None), method="highs").success for target in loads[1:].numpy())
    alpha = torch.tensor([[2.7, 1.3]], dtype=rays.dtype)
    initial = torch.einsum("brj,br->bj", rays, alpha.repeat_interleave(6, 1) / 6.)[:, None] + loads
    previous = initial.square().sum(-1)
    for iterations in (1, 2, 4, 8, 16, 32):
        result = equilibrium_response(alpha, rays, drag, loads, iterations=iterations, jacobian=True)
        score = result.balance.square().sum(-1)
        assert result.valid.all()
        assert bool((score <= previous + 1.e-10).all())
        assert score[:, 1:].amax() > .05 ** 2
        torch.testing.assert_close(result.balance,
            torch.einsum("brj,bsr->bsj", rays, result.row_forces) + loads, atol=1.e-10, rtol=1.e-10)
        previous = score


@pytest.mark.parametrize("iterations", [2, 16])
def test_selected_best_iterate_keeps_matching_force_and_balance_sensitivity(iterations):
    rays, drag, loads = same_side_soft_finger_geometry()
    alpha = torch.tensor([[2.7, 1.3]], dtype=rays.dtype, requires_grad=True)
    generator = torch.Generator().manual_seed(119)
    weights = torch.randn(1, 13, 12, generator=generator, dtype=rays.dtype)
    balance_weights = torch.randn(1, 13, 6, generator=generator, dtype=rays.dtype)
    result = equilibrium_response(alpha, rays, drag, loads, iterations=iterations, jacobian=True)
    objective = (weights * result.row_forces).sum() + .3 * (balance_weights * result.balance).sum()
    automatic, = torch.autograd.grad(objective, alpha)
    analytic = (result.row_force_jacobian * weights[..., None]).sum((1, 2))
    analytic += .3 * (result.balance_jacobian * balance_weights[..., None]).sum((1, 2))
    torch.testing.assert_close(analytic, automatic, atol=1.e-6, rtol=1.e-5)

    def value(preload):
        response = equilibrium_response(preload, rays, drag, loads, iterations=iterations)
        return (weights * response.row_forces).sum() + .3 * (balance_weights * response.balance).sum()

    assert torch.autograd.gradcheck(value, (alpha,), fast_mode=True, atol=2.e-4, rtol=2.e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_best_raw_balance_matches_cpu_on_the_infeasible_same_side_cone():
    rays, drag, loads = same_side_soft_finger_geometry(torch.float32)
    alpha = torch.tensor([[2.7, 1.3]], dtype=rays.dtype)
    cpu = equilibrium_response(alpha, rays, drag, loads, jacobian=True)
    gpu = equilibrium_response(alpha.cuda(), rays.cuda(), drag.cuda(), loads.cuda(), jacobian=True)
    assert gpu.valid.all()
    torch.testing.assert_close(gpu.balance.cpu(), cpu.balance, atol=1.e-5, rtol=1.e-5)
    torch.testing.assert_close(gpu.row_force_jacobian.cpu(), cpu.row_force_jacobian, atol=1.e-5, rtol=1.e-5)
