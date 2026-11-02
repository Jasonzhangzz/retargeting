from types import SimpleNamespace

import numpy as np
import pytest

from retarget.env.grasp.physics import grasp_physics


def test_standalone_physics_does_not_require_a_dexycb_scene():
    values = grasp_physics(SimpleNamespace(physics_reference=None, friction=0.7))
    assert values["friction"] == [0.7, 0.005, 0.0001]
    assert values["condim"] == 3
    np.testing.assert_equal(values["solref"], [0.02, 1.0])


@pytest.mark.parametrize("friction", [-1.0, float("nan")])
def test_standalone_physics_rejects_invalid_friction(friction):
    with pytest.raises(ValueError, match="friction"):
        grasp_physics(SimpleNamespace(physics_reference=None, friction=friction))


def test_disabled_wrist_prior_and_physics_reference_need_no_files(tmp_path, monkeypatch):
    from retarget.env.grasp import runner, hand

    template = tmp_path / "templates/assets/hand/shadow/raw_anno/test.yaml"
    template.parent.mkdir(parents=True)
    template.touch()
    (template.parent.parent / "right.xml").touch()
    (template.parent.parent / "keypoint.yaml").touch()
    initializer_config = tmp_path / "templates/dexonomy/config/op/init.yaml"
    initializer_config.parent.mkdir(parents=True)
    initializer_config.touch()
    model = tmp_path / "model.yaml"
    model.touch()
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "right.xml").touch()
    (assets / "qp_right").mkdir()
    for name in hand.HAND_BODIES:
        (assets / "qp_right" / f"{name}.npz").touch()
    args = SimpleNamespace(
        hand_model="wuji2", objective="dex_tracking", starts=1,
        init_points=1, init_inplane=1, init_iterations=1,
        radius_scales=[1.0], height_scales=[1.0], templates=["test"],
        dexonomy_root=tmp_path / "templates", solver_config=model,
        physics_reference=None, wrist_pose_prior=False,
        wrist_pose_reference=tmp_path / "absent.json",
    )
    monkeypatch.setattr(runner, "arguments", lambda _: args)
    monkeypatch.setattr(hand, "asset_root", lambda _: assets)
    config = SimpleNamespace(solver=SimpleNamespace(gradient_steps=0), task={})
    assert runner.validate(config) is args
    args.wrist_pose_prior = True
    with pytest.raises(ValueError, match="absent.json"):
        runner.validate(config)
