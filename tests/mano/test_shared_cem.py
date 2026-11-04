import torch
from retarget.solver.base import Problem, Settings, solve


def test_batched_cem_matches_independent_serial_runs():
    means = torch.tensor([[2.0, -1.0, 0.2], [-3.0, 2.0, 1.0]], dtype=torch.float64)
    sigma = torch.full_like(means, 0.7)
    settings = Settings(population=64, iterations=7)

    def scores(x):
        cost = x.square().sum(-1)
        return cost.masked_fill(x[..., 0] < -2.5, torch.inf)

    batch = solve(Problem(means, sigma, scores, lambda x: x.clamp(-4, 4)), settings, 17)
    for row in range(2):
        serial = solve(
            Problem(means[row], sigma[row], scores, lambda x: x.clamp(-4, 4)),
            settings,
            17 + row,
        )
        torch.testing.assert_close(
            batch.parameters[row], serial.parameters, rtol=1e-10, atol=1e-10
        )
        torch.testing.assert_close(
            batch.cost[row],
            torch.tensor(serial.cost, dtype=torch.float64),
            rtol=1e-10,
            atol=1e-10,
        )


def test_invalid_row_does_not_contaminate_other_rows():
    means = torch.tensor([[1.0, 1.0], [2.0, 2.0]])

    def scores(x):
        cost = x.square().sum(-1)
        cost[1] = torch.inf
        return cost

    result = solve(
        Problem(means, torch.ones_like(means), scores),
        Settings(population=16, iterations=3),
        3,
    )
    assert torch.isfinite(result.cost[0]) and torch.isinf(result.cost[1])
    assert torch.isfinite(result.parameters).all()


def test_preserved_objective_and_qp_formulas():
    import ast
    import hashlib
    import json
    from pathlib import Path

    spec = json.loads(
        Path("tests/fixtures/grasp_objective_fingerprints.json").read_text()
    )
    for check in spec["checks"]:
        body = ast.parse(Path(check["path"]).read_text()).body
        if check["cls"]:
            body = next(
                n
                for n in body
                if isinstance(n, ast.ClassDef) and n.name == check["cls"]
            ).body
        node = next(
            n
            for n in body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))
            and n.name == check["name"]
        )
        assert (
            hashlib.sha256(
                ast.dump(node, include_attributes=False)
                .replace(", type_params=[]", "")
                .encode()
            ).hexdigest()
            == check["ast_sha256"]
        ), check


def test_batched_object_constants_match_independent_force_qps():
    from retarget.env.grasp.closure_types import ClosureSettings
    from retarget.env.grasp.projected_qp import ProjectedQPClosure

    settings = ClosureSettings(cuda_graph=False)
    normals = torch.nn.functional.normalize(
        torch.tensor(
            [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]],
            dtype=torch.float64,
        ),
        dim=-1,
    )
    centers = torch.tensor(
        [[0.1, 0.0, 0.0], [-0.1, 0.2, 0.1], [0.05, 0.1, -0.05]], dtype=torch.float64
    )
    radii = torch.tensor([0.03, 0.06, 0.09], dtype=torch.float64)
    batch = ProjectedQPClosure(
        settings, center=centers[0], radius=float(radii[0]), object_mass=0.1
    )
    batch.set_object_batch(centers, radii)
    points = centers[:, None] + radii[:, None, None] * normals[None]
    actual = batch.solve(points, normals[None].expand_as(points))
    for row in range(len(centers)):
        solver = ProjectedQPClosure(
            settings, center=centers[row], radius=float(radii[row]), object_mass=0.1
        )
        expected = solver.solve(points[row : row + 1], normals[None])
        torch.testing.assert_close(
            actual.cost[row], expected.cost[0], rtol=1e-9, atol=1e-9
        )
        torch.testing.assert_close(
            actual.row_forces[row], expected.row_forces[0], rtol=1e-9, atol=1e-9
        )
        assert actual.certified[row] == expected.certified[0]
