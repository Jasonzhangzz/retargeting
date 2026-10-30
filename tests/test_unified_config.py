from pathlib import Path
import pytest, yaml
from retarget.config import run
from retarget.common.results import prepare_output


def write(tmp_path, **changes):
    values = dict(
        task=dict(type="humanoid", name="move_largebox", data="inputs"),
        output=dict(name="job"),
        visualization=dict(save_video=False),
    )
    values.update(changes)
    p = tmp_path / "task.yaml"
    p.write_text(yaml.safe_dump(values))
    return p


def test_relative_input_and_dispatch(tmp_path):
    c = run.load(write(tmp_path))
    assert c.task["data"] == str(tmp_path / "inputs")
    assert c.solver.type == "cem" and c.mode == "solve"
    for task in ("grasp", "dex", "humanoid"):
        module = __import__("retarget.env." + task + ".runner", fromlist=["validate"])
        assert callable(module.validate)


@pytest.mark.parametrize("name", ["../escape", "a/b", "a\\b", "..", ".hidden", ""])
def test_reject_output_traversal(tmp_path, name):
    with pytest.raises(ValueError):
        run.load(write(tmp_path, output=dict(name=name)))


def test_replace_only_same_output_and_symlink_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "ROOT", tmp_path)
    c = run.load(write(tmp_path))
    first = prepare_output(c)
    (first / "old").touch()
    other = tmp_path / "output/other"
    other.mkdir()
    (other / "keep").touch()
    prepare_output(c)
    assert not (first / "old").exists() and (other / "keep").exists()
    import shutil

    shutil.rmtree(first)
    first.symlink_to(other, target_is_directory=True)
    with pytest.raises(ValueError):
        prepare_output(c)
    assert (other / "keep").exists()


def test_missing_dex_reports_preparation_required_before_output(tmp_path):
    from retarget.env.dex.runner import validate

    c = run.load(write(tmp_path, task=dict(type="dex", name="raw", data="missing")))
    with pytest.raises(ValueError, match="Missing prepared dex input"):
        validate(c)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("solver", ["adamw", "admm", "sqp"])
def test_only_sampling_solvers(tmp_path, solver):
    with pytest.raises(ValueError):
        run.load(write(tmp_path, solver=dict(type=solver)))
