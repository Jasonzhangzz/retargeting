"""Four native point contacts preserve the source model's inertial dynamics."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from retarget.env.humanoid.point_model import make_point_contact_model


HANDS = ("left_palm", "right_palm")
FEET = ("left_sole", "right_sole")


@pytest.fixture
def source(tmp_path):
    bodies, pairs = [], []
    for side, y in (("left", -.1), ("right", .1)):
        bodies.append(f'''<body name="{side}_hand" pos=".1 {y} .1">
          <joint name="{side}_push" type="slide" axis="1 0 0"/>
          <geom name="{side}_ball" type="sphere" pos=".1 0 0" size=".05" mass="1" contype="0" conaffinity="0"/>
          <site name="{side}_palm" pos=".08 0 0"/>
        </body>''')
        spheres = "".join(f'<geom name="{side}_old_foot_{i}" type="sphere" pos="{x} {yy} 0" size=".05" mass=".1" contype="0" conaffinity="0"/>'
                          for i, (x, yy) in enumerate(((-.02, -.02), (-.02, .02), (.02, -.02), (.02, .02))))
        bodies.append(f'''<body name="{side}_foot" pos="0 {y} -.29">
          <joint name="{side}_ankle" type="slide" axis="0 0 1"/>
          {spheres}<site name="{side}_sole" pos="0 0 -.01"/>
        </body>''')
        pairs.append(f'<pair name="{side}_old_object" geom1="{side}_ball" geom2="box" condim="3" solref=".009 1" friction=".9 .9 .003 .0002 .0002"/>')
        pairs.append(f'<pair name="{side}_old_floor" geom1="{side}_ball" geom2="floor" condim="3" solref=".01 1"/>')
        pairs.extend(f'<pair name="{side}_old_ground_{i}" geom1="{side}_old_foot_{i}" geom2="floor" condim="3" solref=".015 1"/>' for i in range(4))
    path = tmp_path / "source.xml"
    path.write_text(f'''<mujoco><option gravity="0 0 0"/>
      <worldbody><geom name="floor" type="plane" size="2 2 .1"/>
        <body name="robot" pos="0 0 .3"><freejoint/>
          <inertial mass="10" pos="0 0 0" diaginertia="1 1 1"/>
          <geom type="box" size=".01 .01 .01" mass="0" density="0" contype="0" conaffinity="0"/>
          {''.join(bodies)}
        </body>
        <body name="object" pos=".28005 0 .4"><freejoint/>
          <geom name="box" type="box" size=".1 .2 .1" mass="1"/>
        </body>
      </worldbody><actuator><motor joint="left_push"/><motor joint="right_push"/></actuator>
      <contact>{''.join(pairs)}<pair name="object_floor" geom1="box" geom2="floor" condim="3" solref=".02 1" friction=".7 .7 .002 .0001 .0001"/></contact>
    </mujoco>''')
    return path


def test_point_model_keeps_inertia_controls_and_source_file(source):
    contents = source.read_bytes()
    original = mujoco.MjModel.from_xml_path(str(source))
    point = make_point_contact_model(source, HANDS, FEET)
    assert source.read_bytes() == contents
    assert (point.nq, point.nv, point.nu, point.nbody, point.njnt) == (original.nq, original.nv, original.nu, original.nbody, original.njnt)
    assert point.ngeom == original.ngeom + 4
    for name in ("body_mass", "body_inertia", "body_ipos", "body_iquat", "qpos0", "actuator_gear", "actuator_trnid"):
        np.testing.assert_allclose(getattr(point, name), getattr(original, name), atol=1e-12, rtol=0)
    new_names = {f"scsp_{site}" for site in HANDS + FEET}
    point_geoms = {point.geom(name).id for name in new_names}
    robot_bodies = set(range(point.body("robot").id, point.body("object").id))
    for geom in range(point.ngeom):
        if point.geom_bodyid[geom] in robot_bodies and geom not in point_geoms:
            assert point.geom_contype[geom] == point.geom_conaffinity[geom] == 0
    paired = set(map(int, point.pair_geom1)) | set(map(int, point.pair_geom2))
    active_robot = {geom for geom in paired if point.geom_bodyid[geom] in robot_bodies}
    assert active_robot == point_geoms
    assert point.npair == 7  # two palm-object, four point-floor, object-floor


def test_native_contacts_are_at_two_palm_and_two_sole_centres(source):
    model = make_point_contact_model(source, HANDS, FEET)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert data.ncon == 4
    counts = {name: 0 for name in HANDS + FEET}
    for site_name in HANDS + FEET:
        geom, site = model.geom(f"scsp_{site_name}").id, model.site(site_name).id
        np.testing.assert_allclose(data.geom_xpos[geom], data.site_xpos[site], atol=1e-12, rtol=0)
        assert model.geom_size[geom, 0] == pytest.approx(1e-4)
    for contact in data.contact:
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom)) for geom in contact.geom]
        logical = next(name.removeprefix("scsp_") for name in names if name.startswith("scsp_"))
        counts[logical] += 1
        position = data.site_xpos[model.site(logical).id]
        assert np.linalg.norm(contact.pos-position) <= 1.1e-4
        if logical in FEET:
            assert "floor" in names
    assert set(counts.values()) == {1}


def test_pair_impedance_is_copied_and_environment_contacts_preserved(source):
    model = make_point_contact_model(source, HANDS, FEET, friction=.6)
    for side in ("left", "right"):
        palm = model.pair(f"scsp_pair_{side}_palm_box").id
        sole = model.pair(f"scsp_pair_{side}_sole_floor").id
        np.testing.assert_allclose(model.pair_solref[palm], [.009, 1.])
        np.testing.assert_allclose(model.pair_solref[sole], [.015, 1.])
        np.testing.assert_allclose(model.pair_friction[palm], [.6, .6, .003, .0002, .0002])
    original = mujoco.MjModel.from_xml_path(str(source))
    old_id, new_id = original.pair("object_floor").id, model.pair("object_floor").id
    for key in ("pair_solref", "pair_solimp", "pair_friction", "pair_margin", "pair_gap"):
        np.testing.assert_array_equal(getattr(model, key)[new_id], getattr(original, key)[old_id])


def test_actual_largebox_model_has_only_four_robot_contact_geoms():
    path = Path(__file__).resolve().parents[1] / "data/humanoid/move_largebox/scene.xml"
    if not path.exists():
        pytest.skip("largebox assets unavailable")
    point = make_point_contact_model(path, HANDS, ("left_foot", "right_foot"))
    original = mujoco.MjModel.from_xml_path(str(path))
    np.testing.assert_array_equal(point.body_mass, original.body_mass)
    np.testing.assert_array_equal(point.body_inertia, original.body_inertia)
    assert (point.nq, point.nv, point.nu) == (43, 41, 29)
    assert point.npair == 7


@pytest.mark.parametrize("options", [{"radius": 0}, {"radius": float("nan")}, {"friction": -1}])
def test_invalid_point_geometry_settings(source, options):
    with pytest.raises(ValueError):
        make_point_contact_model(source, HANDS, FEET, **options)
