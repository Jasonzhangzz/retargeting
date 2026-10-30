"""The contact response agrees with an independent NumPy calculation."""

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import warp as wp


def test_original_kernel_bodies_are_preserved():
    folder = Path(__file__).resolve().parents[1] / "retarget/dynamics/comfree"
    source = json.loads((folder / "source.json").read_text())
    functions = {
        node.name: node
        for node in ast.parse((folder / "kernels.py").read_text()).body
        if isinstance(node, ast.FunctionDef)
    }
    for name, expected in source["ast_sha256"].items():
        assert (
            hashlib.sha256(
                ast.dump(functions[name], include_attributes=False).encode()
            ).hexdigest()
            == expected
        )
    assert sorted(p.name for p in folder.rglob("*.py")) == [
        "__init__.py",
        "kernels.py",
        "response.py",
    ]


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_closed_form_contact_force_and_inactive_row_reset(device):
    from retarget.dynamics.comfree import ContactResponse

    wp.init()
    if device != "cpu" and not wp.is_cuda_available():
        pytest.skip("CUDA required")
    rng = np.random.default_rng(72)
    worlds, nv, rows = 3, 5, 8
    dt = np.array([0.01, 0.02, 0.005], np.float32)
    J = rng.normal(size=(worlds, rows, nv)).astype(np.float32)
    qv = rng.normal(size=(worlds, nv)).astype(np.float32)
    acc = rng.normal(size=(worlds, nv)).astype(np.float32)
    distance = rng.uniform(-0.03, 0.03, size=(worlds, rows)).astype(np.float32)
    mass = rng.uniform(0.1, 2.0, size=(worlds, rows)).astype(np.float32)
    count = np.array([8, 3, 0], np.int32)
    smooth = rng.normal(size=(worlds, nv)).astype(np.float32)
    arr = lambda x: wp.array(x, device=device)
    force, constraint = (
        arr(np.full((worlds, rows), 123.0, np.float32)),
        arr(np.zeros_like(qv)),
    )
    response = ContactResponse(worlds, nv, device=device)
    total = response.compute(
        arr(dt),
        arr(qv),
        arr(acc),
        arr(J),
        arr(distance),
        arr(mass),
        arr(count),
        arr(smooth),
        force,
        constraint,
    )
    predicted = qv + dt[:, None] * acc
    velocity = np.einsum("wri,wi->wr", J, predicted)
    expected = np.maximum(
        mass
        * (
            -0.001 / dt[:, None] * velocity
            - 0.2 / dt[:, None] * (velocity * dt[:, None] + distance)
        ),
        0,
    )
    expected[np.arange(rows)[None] >= count[:, None]] = 0
    np.testing.assert_allclose(force.numpy(), expected, atol=1e-6, rtol=2e-6)
    expected_constraint = np.einsum("wri,wr->wi", J, expected)
    np.testing.assert_allclose(
        constraint.numpy(), expected_constraint, atol=2e-6, rtol=2e-6
    )
    np.testing.assert_allclose(
        total.numpy(), smooth + expected_constraint, atol=2e-6, rtol=2e-6
    )
    count[:] = 0
    response.compute(
        arr(dt),
        arr(qv),
        arr(acc),
        arr(J),
        arr(distance),
        arr(mass),
        arr(count),
        arr(smooth),
        force,
        constraint,
    )
    np.testing.assert_array_equal(force.numpy(), 0.0)
    np.testing.assert_array_equal(constraint.numpy(), 0.0)


def test_invalid_response_parameters():
    from retarget.dynamics.comfree import ContactResponse

    for value in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and nonnegative"):
            ContactResponse(1, 1, device="cpu", stiffness=value)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_torch_response_matches_warp_with_padding_and_stale_inactive_rows(device):
    from retarget.dynamics.comfree import ContactResponse, contact_response_torch

    wp.init()
    if device != "cpu" and not wp.is_cuda_available():
        pytest.skip("CUDA required")
    rng = np.random.default_rng(194)
    worlds, nv, rows = 3, 4, 6
    dt = np.array([0.01, 0.02], np.float32)
    qvel = rng.normal(size=(worlds, nv)).astype(np.float32)
    qacc = rng.normal(size=(worlds, nv)).astype(np.float32)
    jacobian = rng.normal(size=(worlds, rows, nv + 2)).astype(np.float32)
    distance = rng.uniform(-0.05, 0.05, size=(worlds, rows)).astype(np.float32)
    mass = rng.uniform(0.1, 2.0, size=(worlds, rows)).astype(np.float32)
    count = np.array([6, 2, 0], np.int32)
    smooth = rng.normal(size=(worlds, nv)).astype(np.float32)
    inactive = np.arange(rows)[None, :] >= count[:, None]
    jacobian[inactive] = np.nan
    jacobian[..., nv:] = np.nan
    distance[inactive] = np.nan
    mass[inactive] = np.nan
    inputs = (dt, qvel, qacc, jacobian, distance, mass, count, smooth)
    response = ContactResponse(worlds, nv, device=device, stiffness=0.3, damping=0.002)
    row_force = wp.full((worlds, rows), 123.0, device=device)
    constraint_force = wp.zeros((worlds, nv), device=device)
    total_force = response.compute(
        *(wp.array(value, device=device) for value in inputs),
        row_force,
        constraint_force,
    )
    actual = contact_response_torch(
        *(torch.as_tensor(value, device=device) for value in inputs),
        stiffness=0.3,
        damping=0.002,
    )
    for actual_value, expected in zip(
        actual, (total_force, row_force, constraint_force)
    ):
        np.testing.assert_allclose(
            actual_value.detach().cpu().numpy(), expected.numpy(), atol=3e-6, rtol=3e-6
        )
        assert bool(torch.isfinite(actual_value).all())
    inactive_force = actual[1][torch.as_tensor(inactive, device=device)]
    torch.testing.assert_close(inactive_force, torch.zeros_like(inactive_force))


def test_torch_response_gradcheck_all_continuous_inputs():
    from retarget.dynamics.comfree import contact_response_torch

    generator = torch.Generator().manual_seed(28)
    worlds, nv, rows = 2, 2, 3
    count = torch.tensor([2, 1], dtype=torch.int32)
    active = torch.arange(rows)[None, :] < count[:, None]
    qvel = torch.randn(worlds, nv, generator=generator, dtype=torch.float64) * 0.01
    qacc = torch.randn(worlds, nv, generator=generator, dtype=torch.float64) * 0.01
    jacobian = torch.randn(
        worlds, rows, nv + 1, generator=generator, dtype=torch.float64
    )
    distance = torch.full((worlds, rows), -0.2, dtype=torch.float64)
    mass = torch.full((worlds, rows), 0.8, dtype=torch.float64)
    jacobian[~active] = float("nan")
    jacobian[..., nv:] = float("nan")
    distance[~active] = float("nan")
    mass[~active] = float("nan")
    smooth = torch.randn(worlds, nv, generator=generator, dtype=torch.float64)
    dt = torch.tensor([0.01, 0.02], dtype=torch.float64)
    stiffness = torch.tensor([0.2], dtype=torch.float64)
    damping = torch.tensor([0.001, 0.002], dtype=torch.float64)
    inputs = tuple(
        value.requires_grad_()
        for value in (
            dt,
            qvel,
            qacc,
            jacobian,
            distance,
            mass,
            smooth,
            stiffness,
            damping,
        )
    )

    def response(dt, qvel, qacc, jacobian, distance, mass, smooth, stiffness, damping):
        return contact_response_torch(
            dt,
            qvel,
            qacc,
            jacobian,
            distance,
            mass,
            count,
            smooth,
            stiffness=stiffness,
            damping=damping,
        )

    assert torch.autograd.gradcheck(response, inputs, atol=1e-5, rtol=1e-4)
    total, row_force, constraint = response(*inputs)
    assert bool((row_force[active] > 0).all())
    gradients = torch.autograd.grad(
        total.square().sum() + constraint.square().sum(), inputs
    )
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    for gradient in gradients[3:6]:
        torch.testing.assert_close(
            gradient[~active], torch.zeros_like(gradient[~active])
        )
    torch.testing.assert_close(
        gradients[3][..., nv:], torch.zeros_like(gradients[3][..., nv:])
    )


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("timestep", 0.0, "finite and positive"),
        ("timestep", float("nan"), "finite and positive"),
        ("timestep", [], "nonempty one-dimensional"),
        ("stiffness", -0.1, "finite and nonnegative"),
        ("damping", float("inf"), "finite and nonnegative"),
        ("qacc_smooth", torch.zeros(1, 3), "shapes do not match"),
        ("jacobian", torch.zeros(1, 2, 1), "dense row shapes"),
        ("count", torch.tensor([3]), "number of rows"),
        ("count", torch.tensor([-1]), "number of rows"),
        ("count", torch.tensor([[1]]), "shape"),
    ],
)
def test_torch_response_validates_shapes_and_parameters(name, value, message):
    from retarget.dynamics.comfree import contact_response_torch

    inputs = {
        "timestep": 0.01,
        "qvel": torch.zeros(1, 2),
        "qacc_smooth": torch.zeros(1, 2),
        "jacobian": torch.zeros(1, 2, 2),
        "distance": torch.zeros(1, 2),
        "mass": torch.ones(1, 2),
        "count": torch.tensor([1]),
        "smooth_force": torch.zeros(1, 2),
    }
    inputs[name] = value
    with pytest.raises(ValueError, match=message):
        contact_response_torch(**inputs)
