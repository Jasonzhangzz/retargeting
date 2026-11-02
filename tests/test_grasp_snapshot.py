"""Regression checks for portable shared-scene storage and size reconstruction."""
from pathlib import Path
import json
import numpy as np
from retarget.config.run import ROOT
from retarget.env.geometry import ObjectMeshSurface
from retarget.viewer.runtime import load_case
from retarget.viewer.snapshot import get_snapshot,read_summary
from retarget.viewer.dense_pose import load_dense_grid

CACHE=ROOT/'results/grasp_current_20261103'


def test_all_cases_share_one_scene_and_have_no_temporary_file_dependencies():
    data=json.loads((CACHE/'manifest.json').read_text())
    assert len(data['cases'])==640
    assert len(list(CACHE.rglob('*.xml')))==1
    assert {e['model'] for e in data['cases']}=={'scene.xml'}
    assert {e['trajectory'] for e in data['cases']}=={'poses.npz'}
    assert {e['cached_case'] for e in data['cases']}==set(range(640))
    assert not list(CACHE.rglob('*.obj'))


def test_disk_height_changes_without_changing_radius_or_pose_precision():
    snapshot=get_snapshot(CACHE)
    rows=[e for e in snapshot.manifest['cases'] if e['template'].startswith('18_')]
    low=min(rows,key=lambda e:e['primitive']['size_scale']);high=max(rows,key=lambda e:e['primitive']['size_scale'])
    for e in (low,high):
        model,q=load_case(CACHE,e,final_only=True)
        assert model.nq==26 and q.dtype==np.float64
        np.testing.assert_array_equal(q[0],snapshot.poses['qpos'][e['cached_case']])
        points=ObjectMeshSurface.from_model(model,model.body('grasp_object').id).triangles.reshape(-1,3)
        extent=np.ptp(points,axis=0)
        np.testing.assert_allclose(extent,[.16,.16,.008*e['primitive']['size_scale']],atol=1.e-7,rtol=0)
    assert read_summary(CACHE,low)['primitive']['height_m']==.008
    assert read_summary(CACHE,high)['primitive']['height_m']==.016


def test_compact_dense_index_resolves_source_rows_and_original_knots():
    snapshot=get_snapshot(CACHE)
    grid=load_dense_grid(CACHE,'26_Sphere_4_Finger','sphere')
    assert grid is not None and len(grid.source_rows)==18
    assert 'projected_points' not in grid.values
    assert 'source_projected_points' in grid.values
    for e in grid.source_rows:
        sample=grid.sample(e['primitive']['size_scale'])
        np.testing.assert_array_equal(sample['qpos'],snapshot.poses['qpos'][e['cached_case']])
        count=int(snapshot.poses['contact_counts'][e['cached_case']])
        np.testing.assert_array_equal(sample['projected_points'],snapshot.poses['projected_points'][e['cached_case'],:count])


def test_cached_grids_can_be_reinterpolated_without_solver_outputs(tmp_path):
    from scripts.interpolate_grasp_pose_grids import export
    index=export(CACHE,tmp_path/'denser',subdivisions=2)
    assert index['source_cases']==640 and len(index['groups'])==31
    assert index['dense_poses']==1249
