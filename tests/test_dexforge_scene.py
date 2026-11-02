"""Native scene conversion keeps the original robot sphere contact surfaces."""

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from retarget.env.humanoid.dexforge_scene import prepare_scene, _upstream_paths


DATA = Path(__file__).resolve().parents[1] / "data/humanoid/move_largebox"


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    return prepare_scene(DATA / "scene.xml", tmp_path_factory.mktemp("native_scene"), timestep=1/60)


def test_physical_spheres_and_all_materials_are_preserved(scene):
    old, preview = scene.source_model, scene.model
    assert len(scene.sphere_assets) == 10  # eight foot spheres and both hand balls
    assert (preview.nq, preview.nv, preview.nu) == (43, 41, 29)
    for field in ("body_mass", "body_inertia", "qpos0", "actuator_gainprm", "actuator_biasprm",
                  "pair_geom1", "pair_geom2", "pair_friction", "pair_solref", "pair_solimp"):
        np.testing.assert_allclose(getattr(preview, field), getattr(old, field), atol=1e-12, rtol=0)
    xml = ET.parse(scene.xml).getroot()
    geoms = {node.get("name"): node for node in xml.findall(".//worldbody//geom") if node.get("name")}
    for name, path in scene.sphere_assets.items():
        geom = old.geom(name).id
        element = geoms[name]
        assert element.get("type") == "gs"
        with np.load(path) as asset:
            np.testing.assert_array_equal(asset["local_pos"], np.zeros((1, 3)))
            np.testing.assert_allclose(asset["radius"], [old.geom_size[geom, 0]], atol=1e-9, rtol=0)
        np.testing.assert_allclose(np.fromstring(element.get("pos"), sep=" "), old.geom_pos[geom], atol=1e-15, rtol=0)
        np.testing.assert_allclose(np.fromstring(element.get("quat"), sep=" "), old.geom_quat[geom], atol=1e-15, rtol=0)
    assert geoms["floor"].get("type") == "plane"
    assert geoms["largebox_geom"].get("type") == "sdfmesh"
    assert len(xml.findall("contact/pair")) == old.npair == 13
    assert tuple(preview.site(i).name for i in scene.observation_site_ids) == ("left_palm", "right_palm", "left_foot", "right_foot")
    assert preview.opt.gravity.tolist() == old.opt.gravity.tolist()


def test_box_is_exact_closed_oriented_mesh_in_unchanged_geom_frame(scene):
    root = ET.parse(scene.xml).getroot()
    box = next(node for node in root.findall(".//worldbody//geom") if node.get("name") == "largebox_geom")
    vertices, faces = [], []
    for line in Path(box.get("file")).read_text().splitlines():
        if line.startswith("v "):
            vertices.append([float(value) for value in line.split()[1:]])
        elif line.startswith("f "):
            faces.append([int(value)-1 for value in line.split()[1:]])
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    assert vertices.shape == (8, 3) and faces.shape == (12, 3)
    half_size = scene.source_model.geom_size[scene.source_model.geom("largebox_geom").id]
    np.testing.assert_array_equal(vertices.min(0), -half_size)
    np.testing.assert_array_equal(vertices.max(0), half_size)
    edges = np.sort(np.concatenate([faces[:, [0,1]], faces[:, [1,2]], faces[:, [2,0]]]), axis=1)
    assert np.unique(edges, axis=0, return_counts=True)[1].tolist() == [2]*18
    triangle = vertices[faces]
    volume = np.einsum("ij,ij->i", triangle[:,0], np.cross(triangle[:,1], triangle[:,2])).sum()/6
    assert volume == pytest.approx(float(8*np.prod(half_size)))


def test_official_native_loader_geometry_and_pair_metadata(scene):
    # The native package requires its original MuJoCo ABI; other repository
    # environments still run the independent asset/reference unit checks above.
    if mujoco.__version__ != "3.6.0":
        pytest.skip("native loader requires isolated .venv-forceaware")
    _upstream_paths()
    from comfree_warp.gaussian_loader import _preprocess_xml, _build_host_graph

    prepared = _preprocess_xml(str(scene.xml))
    native = mujoco.MjModel.from_xml_string(prepared.model_xml, assets=prepared.assets)
    collision = mujoco.MjModel.from_xml_string(prepared.collision_xml, assets=prepared.assets)
    host, meshes = _build_host_graph(native, collision, prepared)
    for field in ("body_mass", "body_inertia", "actuator_gainprm", "actuator_biasprm", "qpos0"):
        np.testing.assert_allclose(getattr(native, field), getattr(scene.source_model, field), atol=1e-12, rtol=0)
    old = scene.source_model
    expected = {frozenset((int(a), int(b))): i for i, (a,b) in enumerate(zip(old.pair_geom1, old.pair_geom2))}
    observed = set()
    for batch in host.pairs:
        for contact, friction, solref, solimp in zip(batch.contact_geom, batch.contact_friction, batch.contact_solref, batch.contact_solimp):
            key = frozenset(map(int, contact))
            observed.add(key)
            index = expected[key]
            np.testing.assert_allclose(friction, old.pair_friction[index], atol=1e-7, rtol=0)
            np.testing.assert_allclose(solref, old.pair_solref[index], atol=1e-8, rtol=0)
            np.testing.assert_allclose(solimp, old.pair_solimp[index], atol=1e-7, rtol=0)
    assert observed == set(expected)
    assert set(meshes) == {old.body("largebox").id}
