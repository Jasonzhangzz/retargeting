"""Primitive sweep dimensions, explicit overrides and contact adapters."""
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from retarget.config.run import ROOT
from retarget.env.grasp.primitives import CATALOG, jobs, size_specifications


def test_default_primary_inventory_has_183_unique_physical_instances():
    inventory = [p for t, families in CATALOG.items()
                 for p in size_specifications(t, families[0][0])]
    assert len(inventory) == 183
    assert len(size_specifications(10, "disk")) == 11
    assert not size_specifications(19) and not size_specifications(21)
    np.testing.assert_allclose([p["size_scale"] for p in size_specifications(10)], np.linspace(.8, 1.3, 11))
    assert all(p['size_policy_version'] == 4 for p in size_specifications(10))


@pytest.mark.parametrize("number,lower,upper", [(10,.8,1.3),(11,1,1.5),(12,.75,1),
    (26,.75,1.5),(28,1,1.5),(33,1,2)])
def test_requested_radial_ranges(number, lower, upper):
    rows = size_specifications(number, CATALOG[number][0][0])
    assert rows[0]["size_scale"] == lower and rows[-1]["size_scale"] == upper
    assert len({p["radius_m"] for p in rows}) == len(rows)
    if rows[0]["kind"] == "sphere":
        assert all(p["height_m"] == 2*p["radius_m"] for p in rows)


@pytest.mark.parametrize("number,width,depth,base_height", [(16,.075,.0525,.003),(22,.150,.105,.008)])
def test_lateral_and_extension_keep_old_large_width_and_only_change_thickness(number, width, depth, base_height):
    rows = size_specifications(number)
    for p in rows:
        scale=p["size_scale"]
        np.testing.assert_allclose(p["dimensions_m"], [width,depth,base_height*scale])
        assert p["size_axis"] == "thickness" and p["size_scale"] == scale
        assert p["radial_scale"] == 1.5 and p["axial_scale"] == scale


def test_flat_palmar_disk_changes_thickness_at_constant_radius():
    rows = size_specifications(30)
    np.testing.assert_allclose([p["height_m"] for p in rows], [.00025,.0005,.00075,.001,.002,.004,.006,.008,.010,.012])
    assert all(p["radius_m"] == .05 and p["size_axis"] == "thickness" for p in rows)
    assert all(p["size_policy_version"] == 5 for p in rows)


def test_tip_pinch_extends_diameter_below_one_mm_without_changing_reference():
    from retarget.env.grasp.primitives import canonical_specification
    rows = size_specifications(24)
    np.testing.assert_allclose([p['dimensions_m'][0]*1000 for p in rows],
                              [.05, .1, .25, .5, 1, 2, 4, 6, 9, 12, 15, 18])
    assert all(p['height_m'] == .05 and p['size_policy_version'] == 5 for p in rows)
    args = SimpleNamespace(radius=None, height=None, primitive_grid_explicit=False)
    standard = canonical_specification(24, 'cylinder', args)
    assert standard['radius_m'] == pytest.approx(.006)
    assert standard['size_scale'] == 1.
    # Explicit user grids remain authoritative.
    assert len(size_specifications(24, explicit=True, radius_scales=[.5])) == 1


def test_extension_uses_a_broad_disk_and_preserves_other_types_policy():
    assert CATALOG[18] == [("disk", .08, .008)]
    rows = size_specifications(18)
    np.testing.assert_allclose([p["height_m"] for p in rows], np.linspace(.008,.016,9))
    assert all(p["radius_m"] == .08 and p["kind"] == "disk" for p in rows)
    assert all(p["size_axis"] == "thickness" and p["radial_scale"] == 1. for p in rows)
    assert all(p["size_policy_version"] == 6 for p in rows)
    assert size_specifications(12)[0]["size_policy_version"] == 4


def test_changed_cylinders_and_precision_disk_have_correct_dimensions():
    assert CATALOG[9] == [("cylinder", .018, .12)]
    for number in (15,17,29,32):
        assert all(p["height_m"] == .2 for p in size_specifications(number))
    assert all(p["height_m"] == .016 for p in size_specifications(12, "disk"))
    assert all(p["height_m"] == .18 for p in size_specifications(1, "cylinder"))
    # The sphere-only type33 range must not affect its optional plate family.
    assert [p["size_scale"] for p in size_specifications(33, "plate")] == [.5,.75,1,1.25,1.5]


def test_explicit_dimensions_retain_cartesian_radius_height_semantics():
    rows = size_specifications(16, radius=.02, height=.01,
        radius_scales=[.5,2], height_scales=[1,3], explicit=True)
    assert [(p["radius_m"],p["height_m"]) for p in rows] == [(.01,.01),(.01,.03),(.04,.01),(.04,.03)]
    assert len(size_specifications(10, radius=.05, explicit=True)) == 1
    assert size_specifications(10, radius=.05, explicit=True)[0]["radius_m"] == .05


@pytest.mark.parametrize("values", [[0],[float("nan")],[-1],[]])
def test_explicit_invalid_size_grid_rejected(values):
    with pytest.raises(ValueError, match="finite and positive"):
        size_specifications(9, radius_scales=values, explicit=True)


def test_jobs_can_select_policy_without_injected_generic_defaults_overriding_it(tmp_path):
    args = SimpleNamespace(primary_shape_only=True, primitive=None, radius=None,height=None,
        radius_scales=[1.],height_scales=[1.],primitive_grid_explicit=False)
    rows, skipped = jobs(tmp_path, ["10_Power_Disk"], args)
    assert not skipped and len(rows) == 11 and len({row[0] for row in rows}) == 11
    args.primitive_grid_explicit = True
    args.radius = .02
    rows, _ = jobs(tmp_path, ["10_Power_Disk"], args)
    assert len(rows) == 1 and rows[0][2]["radius_m"] == .02


ASSETS = ROOT / "data/grasp/assets/wuji2/right.xml"
TEMPLATES = ROOT / "data/grasp/templates"


@pytest.mark.skipif(not ASSETS.exists(), reason="Wuji2 assets required")
def test_power_sphere_palm_is_local_adapter_and_does_not_mutate_vendor_or_contacts():
    from retarget.task.grasp.dexonomy import load_template
    from retarget.env.grasp.inputs import adapt_primitive_contacts
    path = TEMPLATES / "assets/hand/shadow/raw_anno/11_Power_Sphere.yaml"
    original = path.read_bytes()
    _, _, contacts, _ = load_template(TEMPLATES, "11_Power_Sphere")
    count = len(contacts)
    result, diagnostics = adapt_primitive_contacts(TEMPLATES, "11_Power_Sphere", contacts)
    assert len(contacts) == count and len(result) == count+1
    assert result[-1][0] == "rh_palm" and diagnostics[0]["body"] == "rh_palm"
    assert path.read_bytes() == original
    repeated, changes = adapt_primitive_contacts(TEMPLATES, "11_Power_Sphere", result)
    assert len(repeated) == len(result) and changes == []


@pytest.mark.skipif(not ASSETS.exists(), reason="Wuji2 assets required")
@pytest.mark.parametrize("template", ["9_Palmar_Pinch","16_Lateral","22_Parallel_Extension","30_Palmar"])
def test_orientation_seeds_align_the_requested_axis_without_new_constraints(template):
    from retarget.task.grasp.dexonomy import load_template, map_contacts
    from retarget.env.grasp.inputs import StaticHand
    from retarget.env.grasp.orientation import configure_orientation, primitive_pose_seeds
    from scipy.spatial.transform import Rotation
    model = mujoco.MjModel.from_xml_path(str(ASSETS))
    shadow, _, contacts, _ = load_template(TEMPLATES, template)
    bodies, local, normals, _ = map_contacts(shadow, contacts, model)
    hand = StaticHand(model, None, bodies, local, normals)
    q = np.zeros(26)
    hand.ik_reference_qpos = q
    hand.contact_reference_normals = normals
    primitive = size_specifications(int(template.partition('_')[0]))[2]
    configure_orientation(hand, template, primitive)
    seeds = primitive_pose_seeds(hand, q)
    assert seeds.shape == (8,26) and np.isfinite(seeds).all()
    assert not hasattr(hand, "object_orientation")
    hand.set(seeds[0])
    names = [model.body(int(body)).name for body in bodies]
    thumb = np.asarray([name.startswith('right_finger1_') for name in names])
    normal = np.einsum('nij,nj->ni',hand.data.xmat[bodies].reshape(-1,3,3),normals)[thumb].mean(0)
    normal /= np.linalg.norm(normal)
    if template.startswith(('16_','22_')):
        assert abs(normal[2]) == pytest.approx(1.)
    elif template.startswith('9_'):
        assert abs(normal[2]) < 1.e-8
    else:
        from retarget.env.grasp.hand import palmar_axis
        palm = model.body('right_palm_link').id
        normal = hand.data.xmat[palm].reshape(3,3) @ palmar_axis(model,palm,None)
        assert abs(normal[2]) == pytest.approx(1.)
