import time
from types import SimpleNamespace
from unittest.mock import patch

from moxian_web.wuji_bridge import WujiBridge


def test_latest_frames_do_not_drop_reliable_topology_messages(tmp_path):
    bridge=WujiBridge(tmp_path,tmp_path);bridge.proc=SimpleNamespace(poll=lambda:None)
    source={'ready':True,'timestamp_ms':1000.,'geometry_version':'a','binding_version':'b','source_contact':{'ready':False},
            'contact_registration':{'status':'partial','verified_sids':[3]},'calibrated':True,'sensor_mounting':{'status':'multi_pose_calibrated'}}
    model={'ready':True,'rest_vertices_m':[[0,0,0]],'geometry_version':'a','binding_version':'b'}
    hand=SimpleNamespace(snapshot=lambda **kwargs:source.copy(),model=lambda:model.copy())
    with patch('moxian_web.wuji_bridge.time.monotonic',side_effect=[1.,2.,3.]):
        bridge.dispatch(hand);source['timestamp_ms']=1010.;bridge.dispatch(hand)
        source['timestamp_ms']=1020.;source['geometry_version']='new';model['geometry_version']='new';bridge.dispatch(hand)
    assert bridge.frames.qsize()==1
    sent=bridge.frames.get()['frame'];assert sent['timestamp_ms']==1020.
    assert sent['contact_registration']==source['contact_registration'] and sent['calibrated']
    assert sent['sensor_mounting']==source['sensor_mounting']
    assert bridge.controls.qsize()==2
    assert bridge.controls.get()['model']['geometry_version']=='a'
    assert bridge.controls.get()['model']['geometry_version']=='new'


def test_stale_disconnect_and_worker_failure_clear_virtual_contact_only(tmp_path):
    bridge=WujiBridge(tmp_path,tmp_path);bridge.proc=SimpleNamespace(poll=lambda:None)
    bridge.state={'status':'ready','ready':True,'source_timestamp_ms':time.time()*1000,'contact_ready':True,'contact_mask':[True]*460}
    bridge.pose_at=time.monotonic()
    assert bridge.snapshot()['ready']
    disconnected=bridge.snapshot(False)
    assert not disconnected['ready'] and disconnected['contact_mask']==[None]*460
    assert bridge.state['contact_mask']==[True]*460
    bridge.pose_at-=1
    assert not bridge.snapshot()['ready']
    bridge.proc=SimpleNamespace(poll=lambda:-9)
    failed=bridge.snapshot()
    assert not failed['ready'] and failed['status']=='error'


def test_card_latency_is_reported_separately_from_live_input(tmp_path):
    bridge=WujiBridge(tmp_path,tmp_path);bridge.proc=SimpleNamespace(poll=lambda:None)
    bridge.pose_at=time.monotonic()-.4
    bridge.state=dict(status='solving',solution_feasible=True,ready=True,display_available=True,source_timestamp_ms=time.time()*1000-700,
                      input_received_wall_ms=time.time()*1000,input_timestamp_ms=time.time()*1000,contact_ready=True,contact_mask=[True]*460,quality={'processing_ms':350})
    live=bridge.snapshot();assert live['ready'] and live['result_delayed'] and live['input_live']
    bridge.state['input_received_wall_ms']-=2000
    stale=bridge.snapshot();assert not stale['ready'] and not stale['input_live'] and stale['contact_mask']==[None]*460


def test_error_or_new_topology_cannot_resurrect_old_card_solution(tmp_path):
    bridge=WujiBridge(tmp_path,tmp_path);bridge.proc=SimpleNamespace(poll=lambda:None);bridge.pose_at=time.monotonic()
    bridge.state=dict(status='error',solution_feasible=True,ready=False,display_available=True,input_received_wall_ms=time.time()*1000,input_timestamp_ms=time.time()*1000,contact_ready=True,contact_mask=[True]*460)
    assert not bridge.snapshot()['ready'] and not bridge.snapshot()['display_available']
    bridge.state['status']='mapping';assert not bridge.snapshot()['ready']


def test_slow_solving_does_not_blink_at_source_500ms_boundary(tmp_path):
    bridge=WujiBridge(tmp_path,tmp_path);bridge.proc=SimpleNamespace(poll=lambda:None)
    bridge.pose_at=10.
    bridge.state=dict(status='solving',timestamp_ms=99000.,solution_feasible=True,display_available=True,
                      source_timestamp_ms=99000.,input_received_wall_ms=100000.,input_timestamp_ms=99800.,
                      contact_ready=True,contact_mask=[True]*460,quality={'processing_ms':600})
    # The same accepted result remains on screen across multiple UI polls.
    for elapsed in (0.,.29,.31,.65,1.2):
        with patch('moxian_web.wuji_bridge.time.time',return_value=100.+elapsed),patch('moxian_web.wuji_bridge.time.monotonic',return_value=10.+elapsed):
            state=bridge.snapshot()
        assert state['ready'] and state['display_available'] and state['input_live']
        assert state['contact_ready'] and state['contact_mask']==[True]*460
        assert state['timestamp_ms']==99000. and state['result_delayed']
    with patch('moxian_web.wuji_bridge.time.time',return_value=101.6),patch('moxian_web.wuji_bridge.time.monotonic',return_value=11.6):
        stalled=bridge.snapshot()
    assert not stalled['input_live'] and not stalled['display_available'] and not stalled['contact_ready']


def test_fresh_transport_cannot_keep_stalled_source_or_result_live(tmp_path):
    bridge=WujiBridge(tmp_path,tmp_path);bridge.proc=SimpleNamespace(poll=lambda:None)
    bridge.pose_at=10.
    bridge.state=dict(status='solving',solution_feasible=True,display_available=True,
                      source_timestamp_ms=99900.,input_received_wall_ms=100000.,input_timestamp_ms=94000.,
                      contact_ready=True,contact_mask=[True]*460,quality={'processing_ms':500})
    with patch('moxian_web.wuji_bridge.time.time',return_value=100.),patch('moxian_web.wuji_bridge.time.monotonic',return_value=10.1):
        stalled_source=bridge.snapshot()
        bridge.state['input_timestamp_ms']=99900.
        bridge.pose_at=4.
        stalled_solver=bridge.snapshot()
        disconnected=bridge.snapshot(False)
    assert not stalled_source['input_live'] and not stalled_source['display_available']
    assert stalled_solver['input_live'] and not stalled_solver['display_available']
    assert not disconnected['input_live'] and not disconnected['ready']
    for state in (stalled_source,stalled_solver,disconnected):
        assert not state['contact_ready'] and state['contact_mask']==[None]*460


def test_latency_timeout_is_bounded_even_for_abnormal_processing_reports(tmp_path):
    bridge=WujiBridge(tmp_path,tmp_path);bridge.proc=SimpleNamespace(poll=lambda:None)
    bridge.pose_at=10.
    bridge.state=dict(status='solving',solution_feasible=True,display_available=True,
                      input_received_wall_ms=100000.,input_timestamp_ms=99900.)
    for processing,timeout in ((10000.,15000),(float('nan'),5000),(None,5000)):
        bridge.state['quality']={'processing_ms':processing}
        with patch('moxian_web.wuji_bridge.time.time',return_value=100.),patch('moxian_web.wuji_bridge.time.monotonic',return_value=10.1):
            state=bridge.snapshot()
        assert state['ready'] and state['result_timeout_ms']==timeout


def test_raw_source_epoch_ignores_solver_frame_skips_and_detects_real_stream_gaps(tmp_path):
    bridge=WujiBridge(tmp_path,tmp_path)
    for stamp in range(1000,2000,10):bridge.observe_source_input(stamp)
    bridge.observe_source_input(1990)
    assert bridge.source_epoch==0
    bridge.observe_source_input(2300);assert bridge.source_epoch==1
    bridge.observe_source_input(2290);assert bridge.source_epoch==2


def test_supported_solver_entries_and_removed_edge_is_rejected(tmp_path):
    import pytest
    from moxian_web.wuji_backends import SOLVERS, DEFAULT_SOLVER, create_retarget
    assert SOLVERS == ('cem', 'slsqp_graph', 'wuji-retargeting', 'geort', 'cart')
    assert DEFAULT_SOLVER == 'slsqp_graph'
    bridge = WujiBridge(tmp_path, tmp_path)
    for removed in ('slsqp_edge', 'slsqp_cem', 'slsqp', 'edge_laplacian'):
        with pytest.raises(ValueError): bridge.control('solver_set', {'solver': removed})
        with pytest.raises(ValueError): create_retarget(removed, {})


def test_slsqp_factory_selects_the_requested_objective():
    from unittest.mock import patch
    from moxian_web.wuji_backends import create_retarget
    with patch('moxian_web.wuji_contact_slsqp.ContactSLSQPRetarget') as graph, patch('retarget.baseline.wuji_retargeting.runtime.WujiReferenceRetarget') as reference:
        create_retarget('slsqp_graph', {}, objective_profile='removed')
        assert graph.call_args.kwargs['objective_profile']=='contact_graph'
        create_retarget('wuji-retargeting', {})
        reference.assert_called_once()
