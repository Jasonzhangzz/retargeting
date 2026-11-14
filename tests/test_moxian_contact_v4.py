"""Acquisition contract: geometry traces are proxies, never manufactured labels."""
import copy
import numpy as np
import pytest
from moxian_web.contact_capture_v4 import ContactTrajectoryCapture,FlatCycleDetector,validate_document
from moxian_web.contact import FINGERS,default_mapping
from moxian_web.motion_capture import MotionCapture


def frame(t,gap=.04,flat=True):
    return dict(timestamp_ms=t,calibration_fingerprint='fixed',hand_side='right',skeleton=dict(timestamp_ms=t),mapping=dict(assignments=default_mapping('right')),
        fingers={f:dict(position=[0,0,0],quaternion_xyzw=[0,0,0,1]) for f in FINGERS},test_gap=gap,test_flat=flat)


def capture():
    def reference(row):return dict(reference_gap_m=row['test_gap'],flat_valid=row['test_flat'],flat_exit=not row['test_flat'],contact_reference_id='fixed',patch_trajectory=dict(thumb_distal=dict(position_m=[0,0,0],normal=[0,0,1],valid=True)))
    reference.spec={}
    return ContactTrajectoryCapture(dict(rows=[dict(frame(-20),stage='open')],completed_groups=['open'],calibration_fingerprint='fixed'),reference)


def feed(c,origin=0,two=True,partial=False,duration=6):
    c.confirm(c.request_id)
    times=[0,.5,1.,1.6,2.,3.,4.,4.6,5.,duration]
    gaps=[.04,.03,.001,.025,.04,.04,.001,.025,.04,.04] if two else [.04,.03,.001,.025,.04,.04,.04,.04,.04,.04]
    for t in np.arange(0,duration+.001,.02):
        gap=float(np.interp(t,times,gaps));flat=gap>=.039 and (not partial or t<.5)
        c.feed(frame(origin+t*1000,gap,flat))


def test_partial_release_never_counts_and_six_seconds_automatically_extend():
    c=capture();feed(c,partial=True)
    assert c.status=='collecting' and len(c.events)==0 and c.snapshot()['extended']
    assert c.snapshot()['waiting_for']=='展平回位'
    for i in range(5):c.feed(frame(6020+i*20))
    assert len(c.events)==1 and c.status=='collecting'


def test_two_flat_closed_cycles_and_full_trajectory_preserved():
    c=capture();feed(c)
    assert c.index==1 and c.status=='ready'
    rows=c.rows[1:];events=[r for r in rows if r.get('contact_confirmed')]
    assert len(events)==2 and len(rows)>290
    assert all('patch_trajectory' in r for r in rows)
    assert all(r['observation_weights']==[1,1,0,0,0] for r in rows)
    assert c.snapshot()['window_seconds']==6


def test_independent_passes_restore_and_no_validation_relabelling():
    c=capture();c.metadata['motion_schedule']=[['train','contact_pair_index'],['validation','contact_pair_index']]
    feed(c);feed(c,7000);assert c.status=='complete';doc=c.document();validate_document(doc)
    restored=MotionCapture.restore(doc);assert restored.status=='complete'
    assert [r['partition'] for r in doc['rows'] if r.get('contact_confirmed')]==['train','train','validation','validation']
    bad=copy.deepcopy(doc);next(r for r in bad['rows'] if r.get('contact_confirmed'))['capture_pass']='validation'
    with pytest.raises(ValueError,match='分区'):validate_document(bad)


def test_gap_cannot_manufacture_flat_return_and_supplement_keeps_events():
    c=capture();feed(c,two=False);assert len(c.events)==1
    c.pause_cycle('断流');doc=c.document();restored=MotionCapture.restore(doc,True);restored.reference=c.reference
    feed(restored,8000,two=False);assert restored.index==1 and restored.status=='ready'
    detector=FlatCycleDetector()
    for i,g,f in [(0,.04,True),(20,.04,True),(40,.04,True),(60,.001,False),(600,.04,True)]:
        assert detector.feed(dict(timestamp_ms=i,reference_gap_m=g,flat_valid=f,flat_exit=not f),i)==[]


def test_retry_retains_raw_but_invalidates_superseded_labels():
    c=capture();feed(c,two=False);n=len(c.rows);c.pause_cycle('重试');c.retry()
    assert len(c.rows)==n and all(not r['training_eligible'] for r in c.rows[1:])
    assert not any(r.get('contact_confirmed') for r in c.rows)


def test_detector_processes_long_capture_without_history_growth():
    d=FlatCycleDetector()
    for i in range(30000):d.feed(dict(timestamp_ms=i*20,reference_gap_m=.04,flat_valid=True,flat_exit=False),i)
    assert len(d.noise)==0 and d.minimum is None
