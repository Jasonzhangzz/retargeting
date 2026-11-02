"""SCSP is selectable through the normal config and result browser."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from retarget.config.run import load
from retarget.env.humanoid.scsp import contact_options
from retarget.env.humanoid.runtime import contact_settings
from retarget.solver.base import Settings
from retarget.solver.scsp import SCSPSettings
from retarget.viewer.runtime import scan, filter_rows


def test_scsp_yaml_and_nested_settings():
    config = load(Path(__file__).parents[1] / "configs/humanoid_move_largebox_scsp.yaml")
    assert config.solver.type == config.environment["dynamics"] == "scsp"
    assert isinstance(config.solver.scsp, SCSPSettings)
    assert asdict(config.solver)["scsp"]["admm_iterations"] == 4
    assert contact_options(config.environment["scsp_contact"]).neighbors == 10
    assert contact_settings("scsp") is None


def test_scsp_specific_settings_are_not_silently_used_by_sampling_solvers():
    with pytest.raises(ValueError, match="requires solver.type"):
        Settings(type="cem", scsp={})
    with pytest.raises(ValueError, match="Invalid solver.scsp"):
        Settings(type="scsp", scsp={"typo": 3})
    with pytest.raises(ValueError, match="scsp_contact"):
        contact_settings("scsp", {})
    with pytest.raises(ValueError, match="hand_sites"):
        contact_options({"hand_sites": ["palm", "palm"]})


def test_scsp_rejects_other_task_types(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(dict(task=dict(type="dex", data="data"), solver=dict(type="scsp"))))
    with pytest.raises(ValueError, match="requires task.type: humanoid"):
        load(path)


def test_scsp_results_are_visible_in_browser(tmp_path):
    output = tmp_path / "scsp_run"
    output.mkdir()
    (output / "manifest.json").write_text(json.dumps(dict(cases=[dict(
        task="humanoid", name="move_largebox", method="scsp",
    )])))
    assert len(filter_rows(scan(tmp_path), method="scsp", task="humanoid")) == 1
