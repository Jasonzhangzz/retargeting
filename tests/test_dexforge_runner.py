"""The controller cannot bypass failed/unstable directional derivative checks."""

from types import SimpleNamespace
import json

import pytest

from retarget.env.humanoid.dexforge_runner import _duration, _environment_options, _fd_assessment


def test_finite_difference_gate_requires_plateau_before_comparing_adjoint():
    trials = [dict(epsilon=.01, derivative=2., same_contact_active=True),
              dict(epsilon=.003, derivative=2.01, same_contact_active=True),
              dict(epsilon=.001, derivative=10., same_contact_active=True)]
    assert _fd_assessment(2.005, trials, rtol=.02, atol=.001)["passed"]
    wrong = _fd_assessment(10., trials, rtol=.02, atol=.001)
    assert not wrong["passed"]
    assert wrong["finite_difference"] == 2.


def test_contact_switches_and_nonfinite_derivatives_do_not_pass():
    trials = [dict(epsilon=.01, derivative=2., same_contact_active=False),
              dict(epsilon=.003, derivative=2., same_contact_active=True)]
    assert not _fd_assessment(2., trials, rtol=.05, atol=.001)["passed"]
    assert not _fd_assessment(float("nan"), trials, rtol=.05, atol=.001)["passed"]


def test_duration_counts_source_intervals_and_clamps_to_full_reference():
    reference = SimpleNamespace(duration_seconds=2., ref_dt=.0333333)
    assert _duration(reference, None) == 2.
    assert _duration(reference, 1) == .0333333
    assert _duration(reference, 1000) == 2.


def test_gradient_options_reject_weak_single_probe_and_invalid_tolerance():
    config = SimpleNamespace(environment={"dexforge": {"gradient_check": {"directions": 1}}})
    with pytest.raises(ValueError, match="directions"):
        _environment_options(config)
    config.environment["dexforge"]["gradient_check"] = {"rtol": -1.}
    with pytest.raises(ValueError, match="rtol"):
        _environment_options(config)


def test_same_contact_slots_with_different_geometry_features_do_not_pass():
    trials = [dict(epsilon=eps, derivative=2., same_contact_active=True,
                   same_contact_features=False) for eps in (.01, .003)]
    assert not _fd_assessment(2., trials, rtol=.05, atol=.001)["passed"]


@pytest.mark.parametrize("frames", [None, 144, 1000])
def test_complete_reference_cannot_disable_gate_by_setting_frames(frames):
    from pathlib import Path
    from retarget.config.run import load
    from retarget.env.humanoid.dexforge_runner import validate

    config = load(Path(__file__).resolve().parents[1] / "configs/humanoid_move_largebox_forceaware.yaml")
    config.environment["dexforge"]["gradient_check"] = False
    if frames is not None:
        config.environment["frames"] = frames
    with pytest.raises(ValueError, match="full DexForge trajectory"):
        validate(config)


def test_cached_summary_cannot_override_a_failed_probe(tmp_path, monkeypatch):
    from retarget.env.humanoid import dexforge_runner as runner

    path = tmp_path / "report.json"
    config = SimpleNamespace(environment={"dexforge": {"gradient_check": {"report": str(path)}}})
    options = runner._environment_options(config)
    identity = {"fingerprint": "exact-inputs"}
    monkeypatch.setattr(runner, "_identity", lambda config, prepared: identity)
    probes = [dict(kind="native_state_chain", horizon=1, passed=True),
              dict(kind="native_state_chain", horizon=30, passed=False),
              dict(kind="forceaware_objective", horizon=30, passed=True),
              dict(kind="hand_object_control", horizon=1, passed=True)]
    from retarget.common.results import clean_json
    path.write_text(json.dumps(clean_json(dict(identity=identity, options=options,
                                              probes=probes, passed=True))))
    with pytest.raises(runner.GradientValidationError, match="validation failed"):
        runner._gradient_gate(config, None, tmp_path)
