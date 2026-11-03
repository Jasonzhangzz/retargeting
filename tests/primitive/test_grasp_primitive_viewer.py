"""The size control follows thickness while preserving historical geometries."""
from pathlib import Path

from retarget.viewer.runtime import filter_rows, primary_primitive_rows, size_scale


def test_thickness_size_control_does_not_select_fixed_radial_scale():
    rows=[dict(task='grasp',method='graspadmm',hand='wuji2',template='16_Lateral',
        name=f'size{scale}',batch=Path('/output/new'),radius_scale=1.5,
        primitive=dict(kind='plate',radial_scale=1.5,size_axis='thickness',size_scale=scale))
        for scale in [.5,.75,1,1.25,1.5]]
    assert [size_scale(row) for row in rows]==[.5,.75,1,1.25,1.5]
    selected=filter_rows(rows,size='0.5×')
    assert len(selected)==1 and selected[0]['primitive']['size_axis']=='thickness'


def test_historical_size_without_new_metadata_keeps_legacy_scale():
    assert size_scale(dict(radius_scale=.75))==.75


def test_reevaluated_old_type9_disk_is_visible_with_its_real_geometry():
    base=dict(task='grasp',method='mppi',template='9_Palmar_Pinch',primitive=dict(kind='disk'))
    old=dict(base,historical_reevaluation=True,name='old real disk')
    new=dict(base,primitive=dict(kind='cylinder'),method='graspadmm',name='new cylinder')
    assert primary_primitive_rows([old,new])==[old,new]
    assert primary_primitive_rows([dict(base,name='legacy unselected')])==[]


# Additional checks for the same subsystem.
"""Interpolation preserves solved data and rotates correctly across Euler wraps."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from retarget.viewer.dense_pose import DensePoseGrid,interpolate_pose,load_dense_grid
from scripts.interpolate_grasp_pose_grids import export


def test_root_uses_short_rotation_arc_and_exact_saved_knots():
    poses=np.zeros((2,26));poses[:,5]=np.deg2rad([179,-179]);poses[:,6:]=np.array([.2,.8])[:,None]
    result=interpolate_pose([1.,2.],poses,[1.,1.5,2.])
    np.testing.assert_array_equal(result[[0,2]],poses)
    np.testing.assert_allclose(result[1,6:],.5)
    expected=Rotation.from_euler('Z',180,degrees=True).as_matrix()
    np.testing.assert_allclose(Rotation.from_euler('XYZ',result[1,3:6]).as_matrix(),expected,atol=1.e-12)


def test_export_preserves_every_source_pose_and_only_changes_disk_height(tmp_path):
    batch=tmp_path/'batch';batch.mkdir();rows=[];poses=[]
    for i,scale in enumerate((1.,1.125,2.)):
        case=batch/f'case{i}';case.mkdir();q=np.arange(26,dtype=float)/50+i*.01
        np.savez(case/'grasp.npz',qpos=q);np.savez(case/'trajectory.npz',qpos=q[None]);poses.append(q)
        (case/'projection_force_closure.json').write_text(json.dumps(dict(projected_points=[[.08,0.,.004*scale]])))
        rows.append(dict(name=f'case{i}',task='grasp',template='18_Extensior_Type',trajectory=str(case/'trajectory.npz'),
            primitive=dict(kind='disk',radius_m=.08,height_m=.008*scale,size_scale=scale,dimensions_m=[.16,.16,.008*scale])))
    (batch/'manifest.json').write_text(json.dumps(dict(cases=rows)))
    index=export(batch,tmp_path/'dense',subdivisions=8,activate=True)
    assert index['source_cases']==3 and index['dense_poses']==17
    grid=DensePoseGrid(index['groups'][0]['file'])
    for scale,q in zip((1.,1.125,2.),poses):
        sample=grid.sample(scale)
        assert sample['solved']
        np.testing.assert_array_equal(sample['qpos'],q)
    sample=grid.sample(1.5)
    assert not sample['solved']
    np.testing.assert_allclose([sample['radius_m'],sample['height_m']],[.08,.012])
    np.testing.assert_allclose(sample['dimensions_m'],[.16,.16,.012])
    assert (batch/'dense_pose_index.json').is_file()
    assert load_dense_grid(batch,'18_Extensior_Type','disk') is not None
    data=json.loads((batch/'manifest.json').read_text());data['updated']=True
    (batch/'manifest.json').write_text(json.dumps(data))
    assert load_dense_grid(batch,'18_Extensior_Type','disk') is None


def test_dense_query_at_last_knot_does_not_extrapolate(tmp_path):
    values=dict(scales=np.array([.5,1.]),qpos=np.zeros((2,26)),source_scales=np.array([.5,1.]),
        source_qpos=np.zeros((2,26)),radius_m=np.array([.01,.02]),height_m=np.array([.1,.1]),
        dimensions_m=np.array([[.02,.02,.1],[.04,.04,.1]]))
    path=tmp_path/'poses.npz';np.savez(path,**values);grid=DensePoseGrid(path)
    assert grid.sample(10.)['radius_m']==.02
    assert grid.sample(-10.)['radius_m']==.01
