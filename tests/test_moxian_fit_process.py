import sys
import threading

import pytest

from moxian_web.fit_process import run_replay


def test_progress_and_completion():
    values=[]
    run_replay([sys.executable,'-c','import json; print(json.dumps(dict(completed=3,total=3,estimated_remaining_s=0)),flush=True)'],threading.Event(),values.append)
    assert values[-1]['completed']==3
    assert values[-1]['estimated_remaining_s']==0


def test_cancel_interrupts_child():
    cancel=threading.Event()
    def progress(_): cancel.set()
    with pytest.raises(InterruptedError):
        run_replay([sys.executable,'-c','import time; print("{}",flush=True); time.sleep(10)'],cancel,progress)


def test_stall_and_failure():
    with pytest.raises(TimeoutError):
        run_replay([sys.executable,'-c','import time; time.sleep(10)'],threading.Event(),lambda _:None,stall_seconds=.15)
    with pytest.raises(ValueError,match='退出码'):
        run_replay([sys.executable,'-c','raise SystemExit(2)'],threading.Event(),lambda _:None)
