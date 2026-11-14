"""Keep the user's source-reuse boundary enforceable."""
import hashlib
import json
from pathlib import Path
from robokit.helpers.sim_mpc import SimMPC
from robokit.opt.population_solver import PopulationSolver
from robokit.opt.sampler import GaussianSampler
from retarget.env.contact.robokit_planner import ControlPlanner
from retarget.env.contact.robokit_rollout import GPURollout
from retarget.env.contact.robokit_nominal import SourceNominalPlanner
from retarget.solver.robokit_cem import CEMSampler, CEMPopulationSolver


def test_active_planners_extend_original_robokit():
    assert issubclass(ControlPlanner, SimMPC)
    assert issubclass(GPURollout, SimMPC)
    assert issubclass(CEMSampler, GaussianSampler)
    assert issubclass(CEMPopulationSolver, PopulationSolver)
    assert ControlPlanner.plan is SimMPC.plan
    assert GPURollout._segments is SimMPC._segments
    assert SourceNominalPlanner.plan is SimMPC.plan


def test_vendored_library_matches_recorded_source_files():
    root = Path(__file__).resolve().parents[2] / 'third_party/robokit'
    source = json.loads((root / 'SOURCE.json').read_text())
    mismatches = [name for name, expected in source['files_sha256'].items()
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected]
    assert not mismatches


def test_execution_mirror_preserves_measured_time_and_applied_forces():
    import mujoco
    import numpy as np
    from retarget.env.contact.native import snapshot
    from retarget.task.contact.robokit_tasks import ExecutionMirror
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><joint/><geom size=".1"/></body></worldbody></mujoco>')
    data = mujoco.MjData(model)
    data.time = 1.25
    data.qfrc_applied[:] = .7
    mirror = ExecutionMirror(model, data)
    assert mirror.probe.time == data.time
    np.testing.assert_array_equal(mirror.probe.qfrc_applied, data.qfrc_applied)
