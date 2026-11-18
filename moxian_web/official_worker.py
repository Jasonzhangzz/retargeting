"""Isolated official SDK baseline; no serial connection or local IK imports."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import signal
import sys
import threading
import time
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from moxian_web.official_sdk import OfficialSDK, calibration_digest


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--event-fd',type=int);ap.add_argument('--home',type=Path,required=True)
    ap.add_argument('--calibration-home',type=Path,required=True);ap.add_argument('--side',choices=('left','right'),required=True)
    ap.add_argument('--serial');ap.add_argument('--replay',type=Path);ap.add_argument('--output',type=Path)
    a=ap.parse_args();a.home.mkdir(parents=True,exist_ok=True)
    if a.replay:
        reference=OfficialSDK(a.calibration_home,a.home,a.side)
        try:
            payload=json.loads(a.replay.read_text());rows=payload['rows'];results=[]
            for row in rows:results.append({'stage':row.get('stage'),**reference.step(row)})
            report={'policy':reference.policy,'input_sha256':hashlib.sha256(a.replay.read_bytes()).hexdigest(),'frames':results}
            a.output.write_text(json.dumps(report,ensure_ascii=False,allow_nan=False))
        finally:reference.close()
        return
    stream=os.fdopen(a.event_fd,'w',buffering=1);stop=threading.Event();frames=queue.Queue(maxsize=1)
    signal.signal(signal.SIGTERM,lambda *_:stop.set());signal.signal(signal.SIGINT,lambda *_:stop.set())
    def emit(data):stream.write(json.dumps(data,ensure_ascii=False,allow_nan=False)+'\n')
    def reader():
        try:
            for line in sys.stdin:
                m=json.loads(line)
                if m.get('type')!='frame':continue
                try:frames.get_nowait()
                except queue.Empty:pass
                frames.put_nowait(m)
        finally:stop.set()
    threading.Thread(target=reader,daemon=True).start();reference=None
    try:
        checked=0.;digest=None
        while not stop.is_set():
            if time.monotonic()-checked>1.:
                checked=time.monotonic();current=calibration_digest(a.calibration_home)
                if reference is None or current!=digest:
                    if reference:reference.close();reference=None
                    digest=current
                    if current:
                        reference=OfficialSDK(a.calibration_home,a.home,a.side)
                        emit({'type':'state','status':'waiting_pose','pose':None,'runtime_policy':reference.policy,
                              'sdk_version':reference.policy['sdk_version'],'pose_source':'official_public_python_sdk','error':None})
                    else:
                        emit({'type':'state','status':'needs_calibration','pose':None,
                              'error':'官方 SDK 对照等待已保存的官方八动作标定；请在 SLSQP 骨架视图采集'})
            if reference is None:
                stop.wait(.05);continue
            try:frame=frames.get(timeout=.05)
            except queue.Empty:continue
            result=reference.step(frame)
            emit({'type':'state','status':'ready','pose':result['pose'],'joint_angles':result['joint_angles'],
                  'processing_ms':result['processing_ms'],'sdk_sequence':result['sdk_sequence'],
                  'pose_source':'official_public_python_sdk','error':None})
    except Exception as exc:
        traceback.print_exc();emit({'type':'state','status':'error','error':str(exc),'pose':None})
    finally:
        if reference:reference.close()
        stream.close()


if __name__=='__main__':main()
