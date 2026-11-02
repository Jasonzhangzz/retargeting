from pathlib import Path
import json
import numpy as np
import pytest
import mujoco
from retarget.viewer.runtime import scan, filter_rows


def test_manifest_scanning_and_filters(tmp_path):
    for method in ("cem", "mppi"):
        folder = tmp_path / method
        folder.mkdir()
        cases = [
            dict(task="grasp", name="1_type", template="1_type", hand=h, method=method)
            for h in ("wuji1", "wuji2")
        ]
        (folder / "manifest.json").write_text(json.dumps(dict(cases=cases)))
    rows = scan(tmp_path)
    assert len(rows) == 4
    selected = filter_rows(rows, method="mppi", hand="wuji2", template="1_type")
    assert (
        len(selected) == 1
        and selected[0]["method"] == "mppi"
        and selected[0]["hand"] == "wuji2"
    )
    assert not filter_rows(rows, task="humanoid")


def test_final_results_exclude_process_and_native_replays(tmp_path):
    from retarget.viewer.runtime import final_result_rows
    common = dict(batch=tmp_path, task="grasp", case_id="same", method="mppi")
    rows = [dict(common, name=phase, phase=phase)
            for phase in ("acquisition", "hold", "refinement", "native_test")]
    result = final_result_rows(rows)
    assert len(result) == 1 and result[0]["phase"] == "refinement"
    assert result[0]["initial_index"] == -1
    assert result[0]["stage"] == "最终结果"
    assert rows[2].get("stage") is None


def test_final_view_loads_only_the_optimized_pose(tmp_path):
    from retarget.viewer.runtime import load_case
    (tmp_path / "scene.xml").write_text('<mujoco><worldbody><body><joint type="slide"/>'
        '<geom size=".01"/></body></worldbody></mujoco>')
    np.savez(tmp_path / "trajectory.npz", qpos=[[0.], [.1], [.2]])
    case = dict(model="scene.xml", trajectory="trajectory.npz")
    model, final = load_case(tmp_path, case, final_only=True)
    assert model.nq == 1
    np.testing.assert_array_equal(final, [[.2]])
    assert load_case(tmp_path, case)[1].shape == (3, 1)


def test_final_summary_keeps_projected_and_native_results_distinct():
    from retarget.viewer.runtime import final_summary
    result = final_summary(dict(geometry_diagnostics=dict(coverage=.8, projected_closure=True),
        native_evaluated=True, task_success=False, optimization_seconds=5.))
    assert "80.0%" in result and "投影摩擦锥闭合：通过" in result
    assert "原生物理评测：未通过" in result and "5.00 s" in result


def test_template_only_summary_uses_geometric_criterion():
    from retarget.viewer.runtime import final_summary
    result = final_summary(dict(success_criterion="projected_template_geometry_v1",
        task_success=True, native_evaluated=False))
    assert "几何验收：通过" in result
    assert "原生物理评测" not in result


@pytest.mark.parametrize('version', ['v1', 'v2'])
def test_patch_summary_distinguishes_new_metric_from_original_points(version):
    from retarget.viewer.runtime import final_summary
    result = final_summary(dict(geometry_diagnostics=dict(coverage=1.,
        coverage_definition='anatomical_patch_interior_'+version, covered_patch_count=2,
        required_patch_count=2, template_point_coverage=.5, torsional_friction_m=.02)))
    assert 'patch 内部贴合率：100.0%（2/2）' in result
    assert '原模板点贴合率：50.0%' in result
    assert '扭转摩擦：开启（ComFree 投影平衡，系数 0.02 m）' in result


def test_patch_viewer_reads_metric_sidecar_without_moving_source_artifacts(tmp_path):
    from retarget.viewer.runtime import load_patch_coverage
    original = tmp_path/'source'/'template_contacts.json'
    original.parent.mkdir()
    original.write_text('{}')
    assert load_patch_coverage(dict(batch=tmp_path), original) is None
    sidecar = tmp_path/'metric'/'patch_contact_coverage.json'
    sidecar.parent.mkdir()
    sidecar.write_text(json.dumps(dict(coverage=.5, patches=[])))
    value = load_patch_coverage(dict(batch=tmp_path, patch_coverage=str(sidecar)), original)
    assert value['coverage'] == .5
    assert original.read_text() == '{}'


def test_batch_stage_and_size_filters_keep_ca_previews_distinct(tmp_path):
    folder = tmp_path/"five_sizes"
    folder.mkdir()
    common = dict(task="grasp", template="27_Quadpod", hand="wuji2")
    cases = [dict(common, name="actual", method="mppi", stage="MPPI轨迹", radius_scale=1.5),
             dict(common, name="failed", method="contactaware", stage="CA失败预览", radius_scale=.5)]
    (folder/"manifest.json").write_text(json.dumps(dict(cases=cases)))
    rows = scan(tmp_path)
    assert len(rows) == 2
    assert filter_rows(rows, batch="different") == []
    actual = filter_rows(rows, batch="five_sizes", stage="MPPI轨迹", size="1.5×")
    assert len(actual) == 1 and actual[0]["name"] == "actual"
    preview = filter_rows(rows, method="contactaware", size="0.5×")
    assert len(preview) == 1 and preview[0]["stage"] == "CA失败预览"


def test_size_bar_only_selects_recorded_sizes_and_never_interpolates():
    from retarget.viewer.runtime import nearest_size
    scales = [.5, .75, 1., 1.25, 1.5]
    assert nearest_size(scales, .62) == .5
    assert nearest_size(scales, 1.1) == 1.
    assert nearest_size(scales, 2.) == 1.5
    assert nearest_size([], 1.) is None


def test_rollout_playback_uses_recorded_times_and_stops_at_last_pose():
    from retarget.viewer.runtime import playback_index
    times = np.asarray([10., 10.002, 10.008, 10.02])
    assert playback_index(times, -.1) == 0
    assert playback_index(times, .006) == 1
    assert playback_index(times, .04, .25) == 2
    assert playback_index(times, 5.) == 3
    assert playback_index([0.], 1.) == 0
    with pytest.raises(ValueError, match="increasing"):
        playback_index([0., 0.], 1.)


def test_rollout_path_and_object_only_error_scene_have_correct_geometry():
    from retarget.viewer.runtime import object_rollout_path, camera_for
    moving = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="grasp_object">'
        '<freejoint/><geom type="sphere" size=".01"/></body></worldbody></mujoco>')
    qs = np.repeat(moving.qpos0[None], 3, axis=0)
    qs[:, :3] = [[0., 0., 0.], [.01, .02, .03], [.02, .04, .06]]
    np.testing.assert_allclose(object_rollout_path(moving, qs), qs[:, :3])
    fixed = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="grasp_object">'
        '<geom type="sphere" size=".01"/></body></worldbody></mujoco>')
    data = mujoco.MjData(fixed); mujoco.mj_kinematics(fixed, data)
    assert camera_for(fixed, data).distance > .02
    assert object_rollout_path(fixed, np.empty((1, 0))) is None


def test_primary_primitive_filter_preserves_five_sizes_and_custom_tasks():
    from retarget.viewer.runtime import primary_primitive_rows
    rows = [dict(task="grasp", template=template, primitive=dict(kind=kind), radius_scale=scale)
        for template, kind in [("1_Large_Diameter", "cylinder"), ("1_Large_Diameter", "cuboid"),
                               ("33_Inferior_Pincer", "sphere"), ("33_Inferior_Pincer", "plate")]
        for scale in [.5, .75, 1., 1.25, 1.5]]
    custom = dict(task="humanoid", template="custom", primitive=dict(kind="cuboid"))
    selected = primary_primitive_rows([*rows, custom])
    assert len(selected) == 11 and selected[-1] is custom
    for template in {row["template"] for row in rows}:
        group = [r for r in selected if r["template"] == template]
        assert len(group) == 5 and len({r["primitive"]["kind"] for r in group}) == 1


def test_recomputed_failure_replaces_historical_success_without_losing_pending_sizes(tmp_path):
    from retarget.viewer.runtime import preferred_batch_rows
    old = [dict(batch=tmp_path/"old",name=f"old_{i}",case_id=str(i),success=True) for i in range(5)]
    failed = dict(batch=tmp_path/"new",name="new_failed",case_id="2",success=False)
    rows=preferred_batch_rows([*old,failed],"new","old")
    assert len(rows)==5
    assert next(r for r in rows if r["case_id"]=="2") is failed
    assert sum(r["batch"].name=="old" for r in rows)==4


def test_latest_batch_preserves_acquisition_and_hold_but_replaces_all_historic_phases(tmp_path):
    from retarget.viewer.runtime import preferred_batch_rows
    old=dict(batch=tmp_path/'old',case_id='same',name='historical')
    phases=[dict(batch=tmp_path/'new',case_id='same',name=phase,phase=phase) for phase in ['acquisition','hold']]
    assert preferred_batch_rows([old,*phases],'new','old')==phases


def test_front_camera_follows_rotated_palm_and_frames_hand_without_distant_object():
    from scipy.spatial.transform import Rotation
    from retarget.viewer.runtime import hand_front_camera
    # WUJI2 has +Y as the palmar normal; fingers extend along local -Z.
    model = mujoco.MjModel.from_xml_string('''<mujoco>
      <custom><numeric name="grasp_hand_generation" data="2"/></custom>
      <worldbody><body name="right_palm_link"><freejoint/>
        <geom type="box" size=".04 .015 .05"/>
        <body name="right_finger2_link1" pos=".025 0 -.08"><geom size=".01"/></body>
        <body name="right_finger3_link1" pos="0 0 -.09"><geom size=".01"/></body>
        <body name="right_finger4_link1" pos="-.02 0 -.08"><geom size=".01"/></body>
        <body name="right_finger5_link1" pos="-.04 0 -.06"><geom size=".01"/></body>
      </body><body name="grasp_object" pos="5 0 0"><geom size="2"/></body></worldbody>
    </mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    view = hand_front_camera(model, data)
    np.testing.assert_allclose(view["normal"], [0, 1, 0], atol=1e-10)
    assert view["up"][2] < -.9 and view["distance"] < .3
    for aspect in (.6, 1.5, 2.):
        framed = hand_front_camera(model, data, aspect=aspect)
        offsets = framed["points"]-framed["position"]
        depth = -(offsets @ framed["normal"])
        assert np.all(depth > 0)
        tangent = np.tan(framed["fov"]/2)
        x = offsets @ framed["right"]/(depth*tangent*aspect)
        y = offsets @ framed["up"]/(depth*tangent)
        assert max(np.max(np.abs(x)), np.max(np.abs(y))) <= framed["fill"]+1e-8
        assert max(np.ptp(x), np.ptp(y)) > 1.2
    rotation = Rotation.from_euler("xyz", [.4, -.7, .2])
    translation = np.array([.3, -.2, .4])
    data.qpos[:3] = translation
    data.qpos[3:7] = rotation.as_quat(scalar_first=True)
    mujoco.mj_forward(model, data)
    turned = hand_front_camera(model, data)
    for key in ("position", "look_at"):
        np.testing.assert_allclose(turned[key], rotation.apply(view[key])+translation, atol=1e-8)
    for key in ("normal", "up"):
        np.testing.assert_allclose(turned[key], rotation.apply(view[key]), atol=1e-8)
    assert turned["distance"] == pytest.approx(view["distance"])
