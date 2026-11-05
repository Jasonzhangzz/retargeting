"""Public method and default-parameter contracts."""
from dataclasses import FrozenInstanceError,asdict
from pathlib import Path
from types import SimpleNamespace
import pytest
from retarget.solver import cem,graspadmm
from retarget.solver.base import Settings


def test_solver_modules_and_parameters_have_single_owners():
    files={p.stem for p in Path(cem.__file__).parent.glob('*.py')}
    assert files=={'__init__','base','cem','graspadmm'}
    from retarget.env.grasp.contact_geometry import PoseObjectiveSettings
    from retarget.env.grasp.spring_history import SpringHistorySettings
    from retarget.env.grasp.comfree_rollout import RolloutSettings
    from retarget.env.grasp.mano_patch_anchor import PatchAnchorSettings
    assert (PoseObjectiveSettings,SpringHistorySettings,RolloutSettings,PatchAnchorSettings)==(cem.ContactSettings,cem.SpringSettings,cem.RolloutSettings,cem.PatchSettings)
    with pytest.raises(FrozenInstanceError):cem.DEFAULTS.sampling.population=4
    assert cem.DEFAULTS.contact.hand_object_allowance_m==cem.DEFAULTS.patch.hand_object_allowance_m
    assert cem.DEFAULTS.contact.penetration_tolerance_m==cem.DEFAULTS.patch.self_allowance_m


def test_native_graspadmm_entry_passes_the_canonical_settings(monkeypatch):
    from retarget.env.grasp import graspadmm_optimizer
    data={};scene=object();directory=object();calls=[]
    def execute(*args):calls.append(args);return 'actual'
    monkeypatch.setattr(graspadmm_optimizer,'optimize',execute)
    assert graspadmm.solve(graspadmm.NativeProblem(data,scene,directory))=='actual'
    assert calls==[(data,scene,directory,graspadmm.settings())]
    with pytest.raises(ValueError):graspadmm.settings({'step':1})


def test_gpu_graspadmm_entry_does_not_replace_algorithm_parameters(monkeypatch):
    objects=[object() for _ in range(9)];calls=[]
    def execute(*args,**kwargs):calls.append((args,kwargs));return 'gpu'
    monkeypatch.setattr(graspadmm,'_solve_gpu',execute)
    problem=graspadmm.GPUProblem(*objects,hand_body_count=16)
    assert graspadmm.solve(problem)=='gpu'
    assert calls[0][0]==tuple(objects)
    assert calls[0][1]['settings'] is graspadmm.DEFAULTS


def test_task_configs_cannot_supply_an_alternate_parameter_set(tmp_path):
    from retarget.config.run import load
    p=tmp_path/'task.yaml'
    p.write_text('mode: solve\ntask: {type: dex, data: input}\nsolver: {type: cem, iterations: 2}\n')
    with pytest.raises(ValueError,match='canonical defaults'):load(p)
    p.write_text('mode: solve\ntask: {type: grasp, data: input}\nsolver: {type: graspadmm}\nenvironment: {dynamics: graspadmm}\n')
    assert load(p).solver.graspadmm==graspadmm.settings()

@pytest.mark.parametrize('entry,arguments',[
    ('solve_mano_comfree.py',['--input','input','--model','model','--output','output']),
    ('retarget_mano_comfree_campaign.py',['--source','input','--output','output','--model','model','--mapping','mapping','--shadow-xml','shadow','--dataset','dataset'])])
def test_production_clis_reject_budget_overrides(entry,arguments):
    import subprocess,sys
    root=Path(__file__).resolve().parents[2]
    result=subprocess.run([sys.executable,str(root/'scripts'/entry),*arguments,'--iterations','1'],capture_output=True,text=True,timeout=20)
    assert result.returncode==2 and 'unrecognized arguments: --iterations 1' in result.stderr
