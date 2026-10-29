"""Annotation conventions and semantic mapping without an external checkout."""
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
import yaml

from retarget.dexonomy import (ROOT, load_template, map_contacts, shadow_kinematics)


@pytest.fixture
def checkout(tmp_path):
    hand = tmp_path / 'external/assets/hand/shadow'
    (hand / 'raw_anno').mkdir(parents=True)
    tree = ET.Element('mujoco')
    ET.SubElement(tree, 'compiler', angle='radian')
    world = ET.SubElement(tree, 'worldbody')
    palm = ET.SubElement(world, 'body', name='rh_palm', pos='0 0 .034')
    for finger, x in zip(('th', 'ff', 'mf', 'rf', 'lf'), (.04, .033, .011, -.011, -.033)):
        parent = palm
        for part, pos in (('proximal', f'{x} 0 .095'),
                          ('middle', '0 0 .045'), ('distal', '0 0 .025')):
            parent = ET.SubElement(parent, 'body', name=f'rh_{finger}{part}', pos=pos)
            ET.SubElement(parent, 'joint', name=f'{finger}_{part}', axis='1 0 0', range='-2 2')
            ET.SubElement(parent, 'inertial', mass='.01', pos='0 0 .01', diaginertia='.001 .001 .001')
    ET.ElementTree(tree).write(hand / 'right.xml')
    model = shadow_kinematics(hand / 'right.xml')
    data = mujoco.MjData(model)
    data.qpos[:] = .4
    mujoco.mj_kinematics(model, data)
    local = np.array([.001, -.007, .019, 0., -1., 0.])
    body = model.body('rh_ffdistal').id
    rotation = data.xmat[body].reshape(3, 3)
    explicit = np.r_[rotation @ local[:3] + data.xpos[body], rotation @ local[3:]]
    (hand / 'keypoint.yaml').write_text(yaml.safe_dump({'rh_ffdistal': [local.tolist()],
                                                      'rh_mfdistal': 'rh_ffdistal'}))
    annotation = dict(qpos=data.qpos.tolist(), contact={
        'rh_ffdistal': [0, explicit.tolist()], 'rh_mfdistal': 0})
    (hand / 'raw_anno/test.yaml').write_text(yaml.safe_dump(annotation))
    return hand.parents[2], local


def test_annotation_explicit_world_and_local_index_agree(checkout):
    root, expected = checkout
    _, _, contacts, _ = load_template(root, 'test')
    assert [b for b, _ in contacts] == ['rh_ffdistal', 'rh_ffdistal', 'rh_mfdistal']
    for _, local in contacts:
        np.testing.assert_allclose(local, expected, atol=1e-12)
    path = root / 'assets/hand/shadow/raw_anno/test.yaml'
    annotation = yaml.safe_load(path.read_text())
    annotation['contact']['rh_mfdistal'] = -1
    path.write_text(yaml.safe_dump(annotation))
    with pytest.raises(ValueError, match='Invalid keypoint'):
        load_template(root, 'test')


def test_mapping_uses_same_finger_distal_query_surface(checkout, tmp_path):
    root, _ = checkout
    shadow, _, contacts, _ = load_template(root, 'test')
    from retarget.grasp import build_scene
    from retarget.grasp_primitives import primitive_mesh
    wuji, _, _, _ = build_scene(primitive_mesh(tmp_path / 'box.obj', 'cuboid', .035, .09))
    bodies, points, normals, records = map_contacts(shadow, contacts, wuji)
    assert [wuji.body(int(b)).name for b in bodies] == [
        'right_finger2_link4', 'right_finger2_link4', 'right_finger3_link4']
    for point, record in zip(points, records):
        with np.load(ROOT / 'assets/wuji/qp_right' / f'{record["wuji_body"]}.npz') as qp:
            assert np.min(np.linalg.norm(qp['local_pos'] - point, axis=1)) == 0
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1., atol=1e-6)
    np.testing.assert_allclose(points[0], points[1])

