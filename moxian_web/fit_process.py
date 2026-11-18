"""Cancellable replay with frame progress and a stall watchdog."""
import json
import queue
import subprocess
import threading
import time


def run_replay(command, cancel, progress, stall_seconds=120):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
    messages = queue.Queue()
    def read():
        for line in process.stdout:
            try: messages.put(json.loads(line))
            except ValueError: pass
    reader = threading.Thread(target=read, daemon=True); reader.start()
    last = time.monotonic()
    try:
        while process.poll() is None or not messages.empty():
            if cancel.is_set(): raise InterruptedError('标定坐标回放已取消')
            try:
                value = messages.get(timeout=.1)
                last = time.monotonic(); progress(value)
            except queue.Empty:
                if time.monotonic()-last > stall_seconds:
                    raise TimeoutError('坐标回放连续 120 秒无进展，请检查回放日志')
        reader.join(timeout=1)
        while not messages.empty(): progress(messages.get_nowait())
        if process.returncode: raise ValueError(f'坐标回放失败（退出码 {process.returncode}）')
    finally:
        if process.poll() is None:
            process.terminate()
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
        reader.join(timeout=1)
        process.stdout.close()
