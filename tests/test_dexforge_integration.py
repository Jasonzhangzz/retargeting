"""Configuration isolation and source-integrity checks for the FA entry point."""

from dataclasses import asdict
import json
from pathlib import Path

import pytest
import yaml

from retarget.config.run import load
from retarget.dynamics import dexforge_upstream as upstream
from retarget.solver.base import Settings


ROOT = Path(__file__).resolve().parents[1]


def test_new_entry_has_official_budget_and_no_admm():
    cfg = load(ROOT / "configs/humanoid_move_largebox_forceaware.yaml")
    defaults = load(ROOT / "task.yaml")
    assert defaults.solver.type == cfg.solver.type == cfg.environment["dynamics"] == "dexforge"
    s = asdict(cfg.solver.dexforge)
    assert (s["multistart_count"], s["n_iter"], s["horizon"]) == (64, 40, 5)
    assert (s["mpc_dt"], s["knot_dt"], s["action_dt"]) == (.01, .06, .03)
    assert cfg.solver.scsp is None
    assert "scsp_contact" not in cfg.environment
    assert cfg.environment["dexforge"]["gradient_check"] is True


@pytest.mark.parametrize("solver,dynamics", [("dexforge", "scsp"), ("mppi", "dexforge")])
def test_incompatible_backends_fail_before_loading_physics(tmp_path, solver, dynamics):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(dict(task=dict(type="humanoid", data="x"),
                                       solver=dict(type=solver), environment=dict(dynamics=dynamics))))
    with pytest.raises(ValueError, match="must be used together"):
        load(path)


def test_settings_never_silently_dispatch_to_old_solver():
    with pytest.raises(ValueError, match="requires solver.type"):
        Settings(type="scsp", dexforge={})
    with pytest.raises(ValueError, match="Invalid solver.dexforge"):
        Settings(type="dexforge", dexforge={"ipopt_max_iter": 2})


def test_pinned_upstream_files_are_unmodified():
    upstream.verify_upstream.cache_clear()
    result = upstream.verify_upstream()
    assert result["commit"] == "fa6d671948ef03d18514062748b94e445b63cd49"
    paths = result["sha256"]
    assert "forceaware/forceaware/optimizer.py" in paths
    assert "forceaware/forceaware/servo_prior.py" in paths
    assert "forceaware/third_party/comfree_warp/comfree_warp/native_adjoint/fast_step_adjoint.py" in paths


def test_modified_source_is_rejected(tmp_path, monkeypatch):
    manifest = dict(repository="test", commit=upstream.COMMIT, sha256={"optimizer.py": "0"*64})
    (tmp_path / "UPSTREAM.json").write_text(json.dumps(manifest))
    (tmp_path / "optimizer.py").write_text("changed")
    upstream.verify_upstream.cache_clear()
    with monkeypatch.context() as patch:
        patch.setattr(upstream, "ROOT", tmp_path)
        with pytest.raises(RuntimeError, match="Modified DexForge upstream files"):
            upstream.verify_upstream()
    upstream.verify_upstream.cache_clear()
