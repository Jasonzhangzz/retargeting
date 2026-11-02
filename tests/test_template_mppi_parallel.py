"""Memory sizing and full taxonomy coverage without allocating a CUDA context."""
import pytest
from retarget.env.grasp.template_mppi_parallel import options, worker_count


def test_memory_budget_is_based_on_measured_whole_worker_footprint():
    assert worker_count(8000, 1000, 28, 1.08) == 7
    assert worker_count(8000, 1000, 3, 1.08) == 3
    assert worker_count(1000, 1000, 28, 1.08) == 1
    with pytest.raises(RuntimeError, match="above"):
        worker_count(999, 1000, 28, 1.08)


@pytest.mark.parametrize("values", [{"memory_fraction": 1.}, {"memory_fraction": True},
    {"max_workers": 0}, {"max_workers": 1.5}, {"safety_factor": .5}, {"unknown": 1}])
def test_invalid_parallel_config(values):
    with pytest.raises(ValueError):
        options(values)


def test_full_size_config_schedules_every_supported_template(tmp_path):
    from retarget.config.run import ROOT, load
    from retarget.env.grasp.template_mppi_runner import validate
    from retarget.env.grasp.primitives import jobs
    config = load(ROOT / "configs/grasp_template_mppi_all.yaml")
    args = validate(config)
    tasks, skipped = jobs(tmp_path, args.templates, args)
    assert len(args.templates) == 33
    assert len(tasks) == 183
    assert len({name for _, name, _ in tasks}) == 31
    assert {int(row["template"].split("_")[0]) for row in skipped} == {19, 21}
    assert args.primitive_grid_explicit is False
    assert args.settings.evaluate_native is False
    assert args.settings.objective.closure.equilibrium_backend == "cuda_kernel"
    assert config.solver.worlds == config.solver.population == 256
    assert options(config.environment["template_mppi"]["parallel"])["memory_fraction"] == .8


def test_parallel_worker_preserves_primitive_metadata_for_size_ik_and_side_constraints(tmp_path, monkeypatch):
    import json
    from retarget.config.run import ROOT, load
    from retarget.env.grasp import template_mppi_runner as runner
    from retarget.env.grasp.template_mppi_parallel import _solve
    config = load(ROOT/'configs/grasp_template_mppi_all.yaml')
    primitive = dict(kind='disk', radius_m=.056, height_m=.015, size_scale=1.4,
                     dimensions_m=[.112, .112, .015])
    observed = {}
    def run(child, destination, *, initialization_reference, primitive_override):
        observed.update(primitive=primitive_override, reference=initialization_reference)
        assert 'parallel' not in child.environment['template_mppi']
        assert child.task['object'].endswith('scaled.obj')
        case = destination/'000_10_Power_Disk'
        case.mkdir()
        (destination/'summary.json').write_text(json.dumps(dict(cases=[dict(template='10_Power_Disk')])))
        return [dict(name='000_10_Power_Disk',model='000_10_Power_Disk/scene.xml',
            trajectory='000_10_Power_Disk/trajectory.npz',summary='000_10_Power_Disk/summary.json')]
    monkeypatch.setattr(runner, 'run', run)
    result = _solve(config,tmp_path,0,tmp_path/'scaled.obj','10_Power_Disk',primitive,'reference.json')
    assert observed == dict(primitive=primitive,reference='reference.json')
    assert result['summary']['primitive'] == primitive
