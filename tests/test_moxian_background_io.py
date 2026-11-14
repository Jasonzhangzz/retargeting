import json,threading,time
from pathlib import Path
import pytest
import moxian_web.background_io as persistence
import moxian_web.recording as recording
from moxian_web.recording import SnapshotRecorder
from moxian_web.background_io import CaptureCheckpoints


def test_capture_current_coalesces_without_losing_history_or_blocking_controls(tmp_path,monkeypatch):
    entered=threading.Event();release=threading.Event();original=persistence.write_json_chunks
    def blocked(stream,value,**kwargs):
        if value['v']==0:entered.set();assert release.wait(3)
        original(stream,value,**kwargs)
    monkeypatch.setattr(persistence,'write_json_chunks',blocked)
    saves=CaptureCheckpoints();current=tmp_path/'current.json';alias=tmp_path/'alias.json';history=tmp_path/'history'/'old.json'
    try:
        saves.submit([current,alias],{'v':0});assert entered.wait(1)
        began=time.perf_counter();saves.submit([current,alias],{'v':1});saves.submit([history],{'v':1},history=True);saves.submit([current,alias],{'v':2})
        assert time.perf_counter()-began<.05
        release.set();saves.flush()
        assert json.loads(current.read_text())=={'v':2}
        assert current.read_bytes()==alias.read_bytes() and json.loads(history.read_text())=={'v':1}
    finally:release.set();saves.close()


def test_capture_checkpoint_is_stable_when_contact_labels_and_retry_change():
    from tests.test_moxian_contact_motion import capture,frame
    c=capture();c.confirm(c.request_id)
    for i in range(60):c.feed(frame(i*20,.03-abs(30-i)*-.001))
    old=c.document();before=json.dumps(old,sort_keys=True)
    c.retry()
    assert json.dumps(old,sort_keys=True)==before
    assert not c.rows[-1]['training_eligible'] and old['rows'][-1]['training_eligible']


def payload(t,version='a'):
    return dict(timestamp=t,raw_6dof={'timestamp_ms':t,'fingers':{'thumb':{'position':[t,0,0]}}},
                hand_model={'ready':True,'timestamp_ms':t,'geometry_version':version,'binding_version':'binding','vertices_m':[[t,0,0]]},
                hand_model_topology={'ready':True,'version':version},wuji2={},wuji2_topology=None)


def test_record_start_stop_do_not_wait_for_slow_encoding_and_preserve_timestamps(tmp_path,monkeypatch):
    entered=threading.Event();release=threading.Event();original=recording.write_json_chunks
    def blocked(stream,value,**kwargs):entered.set();assert release.wait(3);original(stream,value,**kwargs)
    monkeypatch.setattr(recording,'write_json_chunks',blocked)
    recorder=SnapshotRecorder(tmp_path)
    try:
        begin=time.perf_counter();state=recorder.start();assert state['active'] and time.perf_counter()-begin<.05
        assert recorder.submit(payload(1));assert entered.wait(1)
        assert recorder.submit(payload(2));assert recorder.submit(payload(3,'b'))
        begin=time.perf_counter();state=recorder.stop();assert state['saving'] and not state['active'] and time.perf_counter()-begin<.05
        release.set();recorder.close()
        rows=[json.loads(s) for s in recorder.path.read_text().splitlines()]
        assert [r['raw_6dof']['timestamp_ms'] for r in rows]==[1,2,3]
        assert [r['hand_model']['vertices_m'][0][0] for r in rows]==[1,2,3]
        assert rows[0]['hand_model_topology'] and rows[1]['hand_model_topology'] is None and rows[2]['hand_model_topology']
        assert recorder.snapshot()['frames']==3 and not recorder.snapshot()['saving']
    finally:release.set();recorder.close()


def test_record_backpressure_is_explicit_and_drains_already_accepted_frames(tmp_path,monkeypatch):
    entered=threading.Event();release=threading.Event();original=recording.write_json_chunks
    def blocked(stream,value,**kwargs):entered.set();assert release.wait(3);original(stream,value,**kwargs)
    monkeypatch.setattr(recording,'write_json_chunks',blocked)
    recorder=SnapshotRecorder(tmp_path,capacity=1)
    try:
        recorder.start();recorder.submit(payload(1));assert entered.wait(1)
        assert recorder.submit(payload(2));assert not recorder.submit(payload(3))
        state=recorder.snapshot();assert not state['active'] and state['saving'] and state['dropped_frames']==1 and state['error']
        release.set();recorder.close()
        assert recorder.snapshot()['frames']==2
        assert json.loads(recorder.path.with_suffix('.status.json').read_text())['dropped_frames']==1
        assert len(recorder.path.read_text().splitlines())==2
    finally:release.set();recorder.close()


def test_record_write_failure_is_reported_without_propagating_into_device_loop(tmp_path,monkeypatch):
    def fail(*a,**k):raise OSError('disk full')
    monkeypatch.setattr(recording,'write_json_chunks',fail)
    recorder=SnapshotRecorder(tmp_path);recorder.start();assert recorder.submit(payload(1));recorder.close()
    assert 'disk full' in recorder.snapshot()['error'] and not recorder.snapshot()['active']


def test_large_json_uses_bounded_rows_and_preserves_strict_json(monkeypatch):
    import io
    rows=[{'timestamp_ms':i,'raw':[i%255]*460,'fingers':{'thumb':[.1,.2,.3]}} for i in range(300)]
    document={'protocol':'测试','rows':rows,'state':{'complete':False}}
    original=json.dumps
    def bounded(value,**kwargs):
        assert value is not rows and value is not document,'whole growing capture encoded in one GIL hold'
        return original(value,**kwargs)
    monkeypatch.setattr(persistence.json,'dumps',bounded)
    stream=io.StringIO();persistence.write_json_chunks(stream,document)
    assert json.loads(stream.getvalue())==document
    with pytest.raises(ValueError):persistence.write_json_chunks(io.StringIO(),{'rows':[float('nan')]*65})


def test_storage_exposes_real_encoding_and_commit_progress(tmp_path,monkeypatch):
    entered=threading.Event();release=threading.Event();original=persistence.write_json_chunks
    def paused(stream,value,progress=None,**kwargs):
        progress(64,len(value['rows']));entered.set();assert release.wait(3)
        original(stream,value,progress=progress,**kwargs)
    monkeypatch.setattr(persistence,'write_json_chunks',paused)
    saves=CaptureCheckpoints()
    try:
        saves.submit([tmp_path/'one.json',tmp_path/'two.json'],{'rows':[{'v':i} for i in range(130)]})
        assert entered.wait(1);current=saves.snapshot()['current']
        assert current['completed']==64 and current['total']==130 and current['phase']=='encoding'
        assert '_started' not in current and current['estimated_remaining_s']>=0
        release.set();saves.flush();state=saves.snapshot()
        assert state['current']['files_completed']==2 and state['current']['phase']=='saved' and state['last_saved']['samples']==130
    finally:release.set();saves.close()
