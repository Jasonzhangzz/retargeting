import copy,json
from moxian_web.hand_model import public_fit_report
from moxian_web.web_state import inactive_wuji,needs_wuji


def test_inactive_render_projection_keeps_status_without_mutating_worker():
    state=dict(ready=True,status='ready',source_timestamp_ms=12,quality={'processing_ms':3},source_vertices_m=[[1,2,3]]*778,taxel_positions_m=[[1,2,3]]*460)
    original=copy.deepcopy(state);summary=inactive_wuji(state)
    assert summary=={k:state[k] for k in ('ready','status','source_timestamp_ms','quality')}
    assert state==original


def test_live_fit_summary_does_not_resend_triangle_and_template_solutions():
    report=dict(run_id='test',accepted=False,auxiliary_initialization_checks={'triangles':[1]*50000},
        approach_release_geometry=[{}]*10000,contact_templates=dict(revision='same',source='train_only',validation_used=False,
        regions={'a':[1]*5000},actions=[dict(stage='contact_pair_index',feasible=True,pose_axis_angle=[0]*45,
        edges=[dict(required=True,template_gap_mm=2,template_normal_error_deg=3,template_penetration_mm=0,anchors=[1]*1000)])]))
    before=copy.deepcopy(report);value=public_fit_report(report)
    assert value['run_id']=='test' and value['accepted'] is False
    assert value['contact_templates']['actions'][0]['edges'][0]['template_gap_mm']==2
    assert len(json.dumps(value))<1000 and report==before


def test_unseen_robot_solver_yields_gpu_but_recordings_and_selected_view_keep_it():
    assert needs_wuji({},100)
    assert not needs_wuji({'mano':100},100)
    assert needs_wuji({'mano':100},100,recording=True)
    assert needs_wuji({'mano':100,'wuji2':99},100)
    assert not needs_wuji({'mano':100,'wuji2':97},100)
    assert needs_wuji({'all':99},100)
