"""Latest contiguous received source window, independent of solver throughput."""
from collections import deque
import threading
FUTURE=3
HORIZON=FUTURE+1

def identity(frame):
    return tuple(frame.get(k) for k in ('input_stream_epoch','geometry_version','binding_version',
        'model_hash','hand_side','serial','calibration_fingerprint','skeleton_runtime_revision','pose_calibration_version'))


class SourceFrameWindow:
    def __init__(self):
        self.frames=deque(maxlen=HORIZON);self.lock=threading.Lock()

    def clear(self):
        with self.lock:self.frames.clear()

    def append(self,frame):
        with self.lock:
            if self.frames:
                last=self.frames[-1]
                delta=frame['timestamp_ms']-last['timestamp_ms']
                received=frame.get('input_received_wall_ms',0)-last.get('input_received_wall_ms',0)
                if identity(last)!=identity(frame) or delta<=0 or delta>250 or received>250:self.frames.clear()
            self.frames.append(frame)

    def snapshot(self):
        with self.lock:return list(self.frames)
