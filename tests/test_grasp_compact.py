"""Independent FK/frame checks for the geometry-free numeric result format."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from retarget.grasp import build_scene
from retarget.grasp_primitives import primitive_mesh
from retarget.grasp_compact import DTYPE, DTYPE_V1, DTYPE_V2, DTYPE_V3, audit_records, read_rows, write_batch, load_records
from scripts.view_grasp_results import available_stages, load_geometry


@pytest.mark.parametrize('hand', ['wuji1', 'wuji2'])
@pytest.mark.parametrize('shape', ['cylinder', 'sphere', 'cuboid', 'mesh'])
def test_compact_pose_and_regenerated_geometry_match_original_fk(tmp_path, monkeypatch, hand, shape):
    import retarget.grasp_compact as compact
    monkeypatch.setattr(compact, 'OBJECT_ASSETS', tmp_path/'shared_objects')
    case = tmp_path/'archive';case.mkdir()
    mesh = primitive_mesh(tmp_path/'object.obj', 'cuboid' if shape == 'mesh' else shape, .025, .05)
    model, surface, xml, _ = build_scene(mesh, hand_model=hand)
    tree = ET.fromstring(xml)
    ET.SubElement(tree.find('.//body[@name="grasp_object"]'), 'freejoint', name='obj_joint')
    xml = ET.tostring(tree, encoding='unicode')
    model = mujoco.MjModel.from_xml_string(xml)
    (case/'scene.xml').write_text(xml)
    initial = np.r_[[.07, -.03, .06, .2, -.3, .4], np.linspace(.05, .35, 20),
        [.04, -.02, .01], Rotation.from_euler('XYZ', [.3, -.2, -.1]).as_quat(scalar_first=True)]
    final = initial.copy();final[:6] += [.01, -.02, .03, -.04, .05, -.06]
    final[6:] += .02
    hold = final[:26].copy(); hold[6:26] += np.linspace(-.1, .1, 20)
    np.savez(case/'grasp.npz', initial_qpos=initial, qpos=final, hand_qpos=final[:26], hold_action=hold)
    (case/'result.json').write_text(json.dumps(dict(object=str(mesh), scale=1.)))
    row = dict(case=str(case), template='2_Test', hand_model=hand,
        primitive=(dict(kind=shape, radius_m=.025, height_m=.05, radial_scale=1., axial_scale=1.)
                   if shape != 'mesh' else None),
        initialization=dict(initial_penetration_mm=.3), force_closure=dict(force_closure=False),
        feasible=False, required_groups_satisfied=False, active_contacts=0,
        max_object_penetration_mm=.5, max_self_penetration_mm=0., optimizer_success=True,
        solver_diagnostics=dict(explicit_prediction=dict(predicted_displacement_mm=3.,
            predicted_rotation_deg=1., predicted_hold_feasible=False)))
    output = tmp_path/'compact'
    write_batch([row], output, method='cem_explicit',
        audited=[dict(case=str(case), saved_data_valid=True, closure_agrees=True)])
    assert [p.name for p in output.iterdir()] == ['grasps.npy']
    with (output/'grasps.npy').open('rb') as stream:
        saved = np.load(stream, allow_pickle=False)
    assert saved.dtype == DTYPE and not saved.dtype.hasobject
    assert all(saved.dtype.fields[k][0].base.kind in 'uifb' for k in DTYPE.names)
    np.testing.assert_allclose(saved['hold_offset'][0], hold[6:26]-final[6:26])
    assert read_rows(output)[0]['termination'] == 'predicted_motion_exceeds_tolerance'
    assert (output/'grasps.npy').stat().st_size < 2048
    compact_case = read_rows(output)[0]['case']
    # Display must remain independent of every per-case geometry/archive file.
    (case/'scene.xml').unlink()
    (case/'grasp.npz').unlink()
    mesh.unlink()
    assert available_stages(compact_case) == ['ik', 'optimized']
    assert audit_records(output)[0]['saved_data_valid']
    for stage, qpos in [('ik', initial), ('optimized', final)]:
        data = mujoco.MjData(model);data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        wrist = model.body('right_palm_link').id
        rw, pw = data.xmat[wrist].reshape(3, 3), data.xpos[wrist]
        meshes, camera, center, _ = load_geometry(compact_case, stage)
        assert np.isfinite(camera).all()
        for vertices, _, _, is_object, position, quaternion, geom in meshes:
            actual = Rotation.from_quat(quaternion, scalar_first=True).apply(vertices)+position+center
            if is_object:
                obj = model.body('grasp_object').id
                world = surface.triangles.reshape(-1, 3)@data.xmat[obj].reshape(3, 3).T+data.xpos[obj]
                expected = (world-pw)@rw
                assert cKDTree(actual).query(expected)[0].max() < 2.e-7
                assert cKDTree(expected).query(actual)[0].max() < 2.e-7
            else:
                mid = model.geom_dataid[geom]
                start, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
                world = model.mesh_vert[start:start+count]@data.geom_xmat[geom].reshape(3, 3).T+data.geom_xpos[geom]
                expected = (world-pw)@rw
                np.testing.assert_allclose(actual, expected, atol=2.e-7)


@pytest.mark.parametrize('dtype,version', [(DTYPE_V1, 1), (DTYPE_V2, 2), (DTYPE_V3, 3)])
def test_older_numeric_archives_remain_readable(tmp_path, dtype, version):
    old = np.zeros(1, dtype=dtype)
    old['version'] = version
    old['type_id'] = 2
    old['hand_id'] = 2
    old['shape_id'] = 2
    old['dimensions'] = [.02, .04, 1., 1.]
    old['stage_valid'] = [1, 1]
    old['object_pose_wrist'][..., 3] = 1.
    old['checks'][..., 8:] = 1.
    np.save(tmp_path/'grasps.npy', old, allow_pickle=False)
    assert load_records(tmp_path/'grasps.npy').dtype == dtype
    row = read_rows(tmp_path)[0]
    assert available_stages(row['case']) == ['ik', 'optimized']
    assert audit_records(tmp_path)[0]['saved_data_valid']
