"""The checkpoint must isolate hand-object coupling from foot/floor contacts."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from retarget.dynamics.dexforge_contact_validation import select_contact_frame, _hand_object_slots, validate_hand_object_control


DATA = Path(__file__).resolve().parents[1] / "data/humanoid/move_largebox"


def test_checkpoint_selects_a_real_bilateral_reference_contact_without_repair():
    model = mujoco.MjModel.from_xml_path(str(DATA / "scene.xml"))
    with np.load(DATA / "reference.npz") as raw:
        qpos = raw["qpos"].copy()
    original = qpos.copy()
    checkpoint = select_contact_frame(model, qpos)
    assert checkpoint is not None and checkpoint["hand_count"] == 2
    assert checkpoint["original_hand_object_contact_count"] >= 2
    assert min(checkpoint["hand_depth_m"]) > 0
    np.testing.assert_array_equal(qpos, original)
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[checkpoint["frame"]]
    mujoco.mj_kinematics(model, data)
    mujoco.mj_collision(model, data)
    contact_names = {frozenset(model.geom(int(g)).name for g in contact.geom)
                     for contact in data.contact if contact.dist < 0}
    assert frozenset(("lh", "largebox_geom")) in contact_names
    assert frozenset(("rh", "largebox_geom")) in contact_names


def test_slot_filter_cannot_accept_box_floor_or_foot_floor_as_hand_object():
    model = mujoco.MjModel.from_xml_path(str(DATA / "scene.xml"))
    names = [("floor", "largebox_geom"), ("lf0", "floor"),
             ("lh", "largebox_geom"), ("largebox_geom", "rh"), ("lh", "floor")]
    pairs = [[model.geom(first).id, model.geom(second).id] for first, second in names]
    sites = [model.site(name).id for name in ("left_palm", "right_palm")]
    mask = _hand_object_slots(model, pairs, sites, model.body("largebox").id)
    np.testing.assert_array_equal(mask, [False, False, True, True, False])


def test_no_physical_hand_contact_produces_no_checkpoint():
    model = mujoco.MjModel.from_xml_path(str(DATA / "scene.xml"))
    with np.load(DATA / "reference.npz") as raw:
        qpos = raw["qpos"][:3].copy()
    # This isolated test fixture deliberately separates the object. It does
    # not touch the source NPZ or controller's initial state.
    qpos[:, -7] += 3.
    assert select_contact_frame(model, qpos) is None


@pytest.mark.parametrize("options", [{"directions": 0}, {"directions": True},
                                     {"epsilons": [.01, .001]}, {"rtol": float("nan")}])
def test_checkpoint_cannot_certify_without_a_real_finite_difference_budget(options):
    with pytest.raises(ValueError):
        validate_hand_object_control(None, **options)
