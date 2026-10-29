"""Exercise scene dispatch and persisted summaries without a CUDA device."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('starts', [1, 2])
@pytest.mark.parametrize('population', [8, 1024])
def test_dispatch_counts_cem_and_adamw_without_backend_settings(tmp_path, monkeypatch, starts, population):
    import retarget.grasp as grasp
    import retarget.grasp_device as device

    methods = ['cem', 'adamw', 'cem']
    expected_worlds = [population, 7, population]
    received = []
    batches = []

    def prepared_cases(args):
        method = args._tasks[0][1]
        directory = args.output/args.name
        directory.mkdir()
        # A static runtime deliberately has no dynamics backend settings.
        runtimes = [SimpleNamespace(grasp_optimizer=method,
            optimizer=SimpleNamespace(backend=SimpleNamespace(), last_solver_diag={}))
            for _ in range(args.starts)]
        results = yield runtimes
        assert len(results) == args.starts
        received.append(method)
        row = dict(method=method, starts=len(results), case=str(directory))
        (directory/'summary.json').write_text(json.dumps([row]))
        with (directory/'summary.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['method', 'starts', 'case'])
            writer.writeheader(); writer.writerow(row)
        return directory

    def solved_nodes(runtimes):
        worlds = []
        for runtime in runtimes:
            assert not hasattr(runtime.optimizer.backend, 'settings')
            count = population if runtime.grasp_optimizer == 'cem' else 7
            runtime.optimizer.last_solver_diag = dict(evaluation_worlds=count,
                node_dispatch=dict(observed_overlapping_compute_graphs=1))
            worlds.append(count)
        batches.append(worlds)
        return [('plan', 'contacts', 'pose') for _ in runtimes]

    monkeypatch.setattr(grasp, '_run_cases', prepared_cases)
    monkeypatch.setattr(device, 'solve_nodes', solved_nodes)
    args = SimpleNamespace(output=tmp_path, name='batch', legacy_serial=False,
        gpu_nodes=2, starts=starts, seed=0,
        _tasks=[(Path(f'object_{i}.obj'), method, None) for i, method in enumerate(methods)])
    output = grasp.run(args)

    # Cover the code after solve_nodes: resuming every generator, JSON and CSV
    # aggregation and the exact field that previously raised AttributeError.
    assert received == methods
    dispatch = json.loads((output/'gpu_dispatch.json').read_text())
    assert [row['worlds'] for row in dispatch] == [sum(batch) for batch in batches]
    assert sum(row['worlds'] for row in dispatch) == sum(expected_worlds)*starts
    assert sum(row['nodes'] for row in dispatch) == len(methods)*starts
    assert all(row['host_iteration_syncs'] == 0 for row in dispatch)
    summary = json.loads((output/'summary.json').read_text())
    assert [row['method'] for row in summary] == methods
    assert all(row['starts'] == starts for row in summary)
    with (output/'summary.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert [row['method'] for row in rows] == methods
