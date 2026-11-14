import copy
import numpy as np
import pytest
from moxian_web.static_capture import StaticHandCapture,aggregate,static_schedule
from moxian_web.bone_extrinsics import FINGERS
from moxian_web.motion_capture import MotionCapture
from moxian_web.static_pose import REVISION as POSE_REVISION


class CorrectPose:
    def __call__(self,row):return dict(accepted=True,revision=POSE_REVISION,failures=[],checks=[])


def observation(stamp):
    return dict(timestamp_ms=stamp,calibration_fingerprint='eight',mapping=dict(assignments=[0]*460),
        stage='open',fingers={f:dict(position=[.01*i,.02,.1],quaternion_xyzw=[0,0,0,1]) for i,f in enumerate(FINGERS)})


def capture():
    return StaticHandCapture(dict(rows=[observation(0)],calibration_fingerprint='eight',completed_groups=[]),pose_validator=CorrectPose())


def test_static_aggregation_respects_quaternion_sign():
    rows=[observation(1),observation(2)]
    rows[1]['fingers']['thumb']['quaternion_xyzw']=[0,0,0,-1]
    mean=aggregate(rows)
    np.testing.assert_allclose(mean['fingers']['thumb']['quaternion_xyzw'],[0,0,0,1])
    assert mean['static_sample_count']==2


def test_invalid_and_gap_clear_collection_and_timeout_pauses():
    c=capture();c.confirm(c.request_id)
    for t in range(20,201,20):c.feed(observation(t))
    assert len(c.collected)==10 and not c.aggregates
    bad=observation(220);bad['fingers']['thumb']['quaternion_xyzw']=[0,0,0,0];c.feed(bad)
    assert not c.collected and c.stable_since is None
    for t in range(240,441,20):c.feed(observation(t))
    assert not c.aggregates
    c.feed(observation(800));assert c.stable_since is None and len(c.window)==1 and len(c.collected)==1
    c.feed(observation(121000));assert c.status=='paused' and '超时' in c.error


def test_train_validation_complete_roundtrip_and_retry():
    c=capture();stamp=20
    for index,entry in enumerate(static_schedule()):
        assert c.entry==entry
        if index in (0,18):c.confirm(c.request_id)
        else:assert c.status=='collecting'
        while c.index==index:
            row=observation(stamp)
            for finger in row['fingers'].values():finger['position'][0]+=.01*index
            c.feed(row);stamp+=20
        assert c.aggregates[-1]['static_key']==':'.join(entry)
    assert c.status=='complete' and len(c.aggregates)==36
    state=c.snapshot();assert state['completed_train']==state['completed_validation']==17
    assert [r['static_endpoint'] for r in c.aggregates[16:18]]==['spread','together']
    restored=MotionCapture.restore(c.document());assert restored.status=='complete'
    broken=copy.deepcopy(c.document());broken['aggregates'].pop()
    with pytest.raises(ValueError,match='不完整'):StaticHandCapture.restore(broken)
    c=capture();c.confirm(c.request_id);c.feed(observation(20));c.retry()
    assert c.status=='ready' and not c.rows[-1]['training_eligible']


def test_initialization_change_cancels():
    c=capture();c.confirm(c.request_id);row=observation(20);row['calibration_fingerprint']='other'
    c.feed(row);assert c.status=='cancelled'


def test_wrong_static_pose_never_collects_and_resets_an_active_hold():
    class Gate:
        accepted=True
        def __call__(self,row):return dict(accepted=self.accepted,revision=POSE_REVISION,failures=[] if self.accepted else ['目标RX距离超过50mm'])
    c=capture();gate=Gate();c.pose_validator=gate;c.confirm(c.request_id)
    for t in range(20,201,20):c.feed(observation(t))
    assert c.collected and not c.aggregates
    gate.accepted=False
    for t in range(1820,5001,20):c.feed(observation(t))
    assert c.phase=='waiting_pose' and not c.aggregates and not c.collected
    assert not c.snapshot()['constraints_ok'] and '50mm' in c.snapshot()['constraint_failures'][0]
    gate.accepted=True
    for t in range(5020,8001,20):
        c.feed(observation(t))
        if c.index:break
    assert c.index==1


def test_passes_on_twentieth_stable_frame_without_two_second_hold():
    c=capture();c.confirm(c.request_id)
    for t in range(0,190,10):c.feed(observation(t))
    assert c.index==0
    c.feed(observation(190))
    assert c.index==1
    assert c.aggregates[0]['static_sample_count']==20
    assert c.aggregates[0]['static_end_ms']-c.aggregates[0]['static_start_ms']==190


def test_resume_old_timed_capture_keeps_completed_samples_and_uses_new_timing():
    from moxian_web.static_capture import LEGACY_REVISION,REVISION
    c=capture();c.confirm(c.request_id)
    for t in range(20,401,20):c.feed(observation(t))
    doc=c.document();doc['schedule_revision']=LEGACY_REVISION
    doc.update(hold_seconds=1,collect_seconds=1,window_seconds=2)
    restored=StaticHandCapture.restore(doc,allow_partial=True)
    assert restored.metadata['schedule_revision']==REVISION
    assert restored.metadata['hold_seconds']==restored.metadata['collect_seconds']==0
    assert restored.aggregates==doc['aggregates']


def test_auto_advance_waits_for_a_new_pose_instead_of_reusing_previous_frames():
    c=capture();c.confirm(c.request_id)
    for t in range(20,401,20):c.feed(observation(t))
    assert c.index==1 and c.status=='collecting'
    for t in range(420,1001,20):c.feed(observation(t))
    assert c.index==1 and '下一姿态' in c.snapshot()['constraint_failures'][0]
    for t in range(1020,1401,20):
        row=observation(t);row['fingers']['middle']['position'][0]+=.01
        c.feed(row)
    assert c.index==2 and c.status=='collecting'


def test_first_pass_stops_for_explicit_fitting_instead_of_restarting_capture():
    c=capture();c.confirm(c.request_id);stamp=0
    for index in range(18):
        for _ in range(20):
            stamp+=20;row=observation(stamp)
            for finger in row['fingers'].values():finger['position'][0]+=.01*index
            c.feed(row)
    assert c.index==18 and c.status=='fit_ready'
    assert c.snapshot()['can_fit'] and c.snapshot()['completed_train']==17
    for _ in range(30):stamp+=20;c.feed(observation(stamp))
    assert c.index==18 and c.status=='fit_ready'
    restored=StaticHandCapture.restore(c.document(),allow_partial=True)
    assert restored.status=='fit_ready' and restored.index==18
