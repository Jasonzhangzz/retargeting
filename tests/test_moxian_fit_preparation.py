import threading
import pytest
from moxian_web.fit_preparation import copy_rows


def test_copy_reports_real_counts_and_detaches_mutable_observations():
    source=[{'raw':[i],'nested':{'x':i}} for i in range(130)];events=[]
    copied=copy_rows(source,threading.Event(),events.append)
    assert copied==source and events[0]['completed']==0 and events[-1]['completed']==130
    assert events[-1]['total']==130 and events[-1]['estimated_remaining_s']==0
    copied[0]['raw'][0]=999;assert source[0]['raw']==[0]


def test_copy_can_be_cancelled_before_expensive_work():
    cancel=threading.Event();cancel.set();events=[]
    with pytest.raises(InterruptedError,match='取消'):copy_rows([{'raw':[1]}],cancel,events.append)
    assert events[0]['completed']==0
