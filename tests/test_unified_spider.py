import ast, hashlib, json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest, torch, warp as wp
from retarget.config.run import ROOT
from retarget.task.humanoid import cost
from retarget.env.humanoid.runtime import make_config


@pytest.mark.parametrize("name", ["move_largebox", "sprint"])
def test_reward_matches_pre_migration_upstream_fixture(name):
    x = np.load(ROOT / f"tests/fixtures/spider_{name}.npz")
    c = make_config(ROOT / f"data/humanoid/{name}", "cpu", 5)
    env = SimpleNamespace(
        data_wp=SimpleNamespace(
            qpos=wp.array(x["qpos"], dtype=float, device="cpu"),
            qvel=wp.array(x["qvel"], dtype=float, device="cpu"),
            site_xpos=wp.array(x["site_xpos"], dtype=wp.vec3, device="cpu"),
        )
    )
    refs = [torch.from_numpy(x[f"ref_{i}"]) for i in range(5)]
    reward, _ = cost.get_reward(c, env, refs)
    terminal, _ = cost.get_terminal_reward(c, env, refs)
    np.testing.assert_array_equal(reward, x["reward"])
    np.testing.assert_array_equal(terminal, x["terminal_reward"])


def test_original_cost_function_syntax_is_unchanged():
    manifest = json.loads((ROOT / "retarget/task/humanoid/source.json").read_text())
    source = ast.parse((ROOT / "retarget/task/humanoid/cost.py").read_text())
    expected = manifest["functions"]
    for n in source.body:
        if isinstance(n, ast.FunctionDef):
            assert (
                hashlib.sha256(
                    ast.dump(n, include_attributes=False).encode()
                ).hexdigest()
                == expected[n.name]
            )


def test_sprint_generated_labels_preserve_raw_arrays_and_include_flight():
    path = ROOT / "data/humanoid/sprint"
    a = np.load(path / "reference_raw.npz")
    b = np.load(path / "reference.npz")
    g = np.load(path / "contact_gaps.npz")["signed_gap_m"]
    for k in ("qpos", "qvel", "ctrl"):
        np.testing.assert_array_equal(a[k], b[k])
    assert b["contact"].shape == (len(a["qpos"]), 2) and b["contact_pos"].shape == (
        len(a["qpos"]),
        2,
        3,
    )
    np.testing.assert_array_equal(b["contact"], g <= 0.035)
    assert b["contact"].any() and (~b["contact"].astype(bool)).all(1).any()
    assert (
        json.loads((path / "contact_provenance.json").read_text())[
            "upstream_annotation"
        ]
        is False
    )
