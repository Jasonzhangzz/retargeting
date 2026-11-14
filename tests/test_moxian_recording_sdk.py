"""Run with the SDK Python; no USB/device connection is created."""
import json,queue,threading,time
from types import SimpleNamespace
import pytest


def runtime(tmp_path):
    pytest.importorskip('qnbot_sdk')
    from moxian_web.server import Runtime
    from moxian_web.recording import SnapshotRecorder
    r=Runtime.__new__(Runtime);r.software_control_lock=threading.Lock();r.lock=threading.Lock();r.commands=queue.Queue()
    r.sdk=object();r.last_frame=time.monotonic();r.recorder=SnapshotRecorder(tmp_path)
    return r


def test_record_controls_do_not_copy_display_snapshots_or_wait_for_device_thread(tmp_path):
    r=runtime(tmp_path)
    r.snapshot=lambda:(_ for _ in ()).throw(AssertionError('record control copied all display meshes'))
    begin=time.perf_counter();result=r.command('record_start');assert result['recording']['active'] and time.perf_counter()-begin<.1
    assert r.commands.empty()
    result=r.command('record_stop');assert result['ok'] and not result['recording']['active'];r.recorder.close()


def test_waiting_for_capture_worker_does_not_enter_sdk_command_queue(tmp_path):
    r=runtime(tmp_path);entered=threading.Event();release=threading.Event();response=[]
    r.ik=SimpleNamespace(snapshot=lambda **k:{'ready':True})
    r.contact=SimpleNamespace(snapshot=lambda:(_ for _ in ()).throw(AssertionError('unknown tactile must be allowed')))
    def control(*a):entered.set();assert release.wait(2);return {'ok':True}
    r.hand=SimpleNamespace(control=control)
    thread=threading.Thread(target=lambda:response.append(r.command('mano_capture_start')));thread.start()
    try:
        assert entered.wait(.5) and r.commands.empty()
        # The serial owner still has no pending work/worker wait to execute.
        release.set();thread.join(1);assert response==[{'ok':True}]
    finally:release.set();thread.join(2)


def test_fit_dispatch_does_not_require_a_fresh_display_pose_or_block_serial(tmp_path):
    r=runtime(tmp_path);r.ik=SimpleNamespace(snapshot=lambda **k:(_ for _ in ()).throw(AssertionError('offline fitting must not inspect live pose')))
    r.hand=SimpleNamespace(control=lambda action,data:{'ok':True,'action':action})
    assert r.command('mano_fit',{'method':'parameters'})=={'ok':True,'action':'fit'}
    assert r.commands.empty()


def test_wuji_solver_switch_waits_outside_serial_owner(tmp_path):
    r=runtime(tmp_path);entered=threading.Event();release=threading.Event();response=[]
    def control(action,data):
        assert action=='solver_set' and data=={'solver':'cem'}
        entered.set();assert release.wait(2);return {'scheduled':True}
    r.wuji=SimpleNamespace(control=control)
    thread=threading.Thread(target=lambda:response.append(r.command('wuji2_solver_set',{'solver':'cem'})))
    thread.start()
    try:
        assert entered.wait(.5) and r.commands.empty()
        release.set();thread.join(1);assert response==[{'scheduled':True}]
    finally:release.set();thread.join(2);r.recorder.close()


def test_capture_download_streams_saved_bytes_without_reading_whole_file(tmp_path,monkeypatch):
    import hashlib,urllib.request
    from pathlib import Path
    from http.server import ThreadingHTTPServer
    pytest.importorskip('qnbot_sdk')
    import moxian_web.server as server
    serial='test-download';home=tmp_path/'hand_model'/hashlib.sha256(serial.encode()).hexdigest()[:16];home.mkdir(parents=True)
    content=json.dumps({'rows':[{'raw':[1]*460}]*100}).encode();(home/'motion-capture.json').write_bytes(content)
    monkeypatch.setattr(server,'OUTPUT',tmp_path)
    monkeypatch.setattr(Path,'read_bytes',lambda self:(_ for _ in ()).throw(AssertionError('download allocated whole capture')))
    runtime=SimpleNamespace(hand=SimpleNamespace(serial=serial,surface_backend='mano'))
    http=ThreadingHTTPServer(('127.0.0.1',0),server.handler(runtime));thread=threading.Thread(target=http.serve_forever,daemon=True);thread.start()
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{http.server_port}/api/hand_model/capture') as response:
            assert response.read()==content and response.headers['Content-Disposition'].endswith('"motion-capture.json"')
    finally:http.shutdown();http.server_close();thread.join(1)
