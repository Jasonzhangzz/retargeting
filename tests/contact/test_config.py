from pathlib import Path
import pytest
from retarget.config.run import load
from retarget.task.contact.tasks import Task
import numpy as np


def test_contact_configs_and_unsupported_task_rejection(tmp_path):
    root = Path(__file__).resolve().parents[2]
    for task in ('h1_jog', 'sharpa_pen', 'sharpa_coin'):
        config = load(root / f'configs/contact_{task}.yaml')
        assert config.task['type'] == 'contact' and config.solver.type == 'contact_cem'
        assert config.resolved()['algorithm_parameters']['task_profile']['samples'] == (2048 if task == 'h1_jog' else 128)
    f = tmp_path / 'bad.yaml'
    f.write_text('task: {type: humanoid, data: input}\nsolver: {type: contact_cem}\n')
    with pytest.raises(ValueError, match='requires task.type: contact'):
        load(f)


def test_original_h1_interpolation_is_preserved():
    from retarget.task.contact.source import h1
    task = Task('h1_jog')
    basis = task.config.interpolation
    np.testing.assert_allclose(basis.sum(1), 1., atol=1e-12)
    np.testing.assert_allclose(basis[::5], np.eye(6), atol=1e-12)
    assert task.profile.horizon == 25


def test_reference_timestep_mismatch_is_rejected_before_gpu_allocation(tmp_path):
    from retarget.config.run import Config
    from retarget.solver.base import Settings
    from retarget.env.contact.runner import run
    reference = tmp_path / 'wrong_dt.npz'
    np.savez(reference, timestep=.01)
    config = Config(tmp_path / 'config.yaml', 'solve',
        dict(type='contact', name='h1_jog', data=str(tmp_path), options={'physics_reference': str(reference)}),
        Settings('contact_cem'), {'device': 'cuda:0'}, {'name': 'test'}, {}, 0)
    with pytest.raises(ValueError, match='reference timestep'):
        run(config, tmp_path / 'result')
