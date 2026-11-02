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
