import copy
import numpy as np
import pytest

from moxian_web.motion_capture import MotionCapture,STAGES,PROTOCOL,confirmed_touch,validate_document
from moxian_web.mano_fit import choose_rows,holdout_rows,independent_cycles_valid
from moxian_web.hand_model import calibration_report_valid,MANO_FIT_REVISION,MANO_SOLVER_REVISION,tracking_registration_matches


def initial():
    return dict(rows=[dict(stage=s,timestamp_ms=i*100+j,fingers={},mapping={'assignments':[0]},guided_contact=s.startswith('pinch_'))
                      for i,s in enumerate(('open','together','mcp90','fist','pinch_index','pinch_middle','pinch_ring','pinch_pinky')) for j in range(6)],
                completed_groups=['open','together'],calibration_fingerprint='base',source_payload_sha256='source')


def frame(stamp,touch=True):
    return dict(timestamp_ms=stamp,tactile_timestamp_ms=stamp,calibration_fingerprint='base',mapping={'assignments':[0]},
                fingers={f:dict(position=[.01*np.sin(stamp/400),.02,.03],quaternion_xyzw=[0,0,0,1]) for f in ('thumb','index','middle','ring','pinky')},
                contact=dict(ready=touch,patches=[dict(id=f+'_distal',contact=touch) for f in ('thumb','index','middle','ring','pinky')]))


def test_complete_cycles_are_held_out_and_raw_frames_not_mutated():
    capture=MotionCapture(initial());stamp=2000
    for stage in STAGES:
        capture.confirm(capture.request_id)
        for step in range(int(stage['seconds']*50)+1):
            source=frame(stamp+step*20);before=copy.deepcopy(source);capture.feed(source)
            assert source==before
        stamp+=int(stage['seconds']*1000)+100
    assert capture.status=='complete'
    document=validate_document(capture.document())
    assert independent_cycles_valid(document['rows'])
    train,valid=choose_rows(document['rows']);full=holdout_rows(document['rows'])
    assert not {r['timestamp_ms'] for r in train}&{r['timestamp_ms'] for r in full}
    assert all(r['partition']=='train' for r in train)
    assert all(r['partition']=='validation' for r in valid)
    dynamic=[r for r in full if not r.get('initialization')]
    assert all(r.get('cycle')==2 or r['stage'].endswith('_repeat') for r in dynamic)
    assert not any(confirmed_touch(r) for r in document['rows'] if r.get('motion_phase') in ('approach','apart','release'))
    assert MotionCapture.restore(document).snapshot()['state']=='complete'


def test_cancel_retry_stale_confirmation_and_input_gap():
    c=MotionCapture(initial());request=c.request_id;c.confirm(request)
    with pytest.raises(ValueError):c.confirm(request)
    c.feed(frame(1000));c.feed(frame(1300))
    assert c.status=='paused' and len(c.rows)==48 and c.error
    c.confirm(c.request_id);c.feed(frame(2000));assert c.status=='collecting' and len(c.rows)==49
    c.cancel();assert c.status=='cancelled'
    c.retry();assert c.status=='ready' and len(c.rows)==48
    with pytest.raises(ValueError):c.confirm(request)


def test_missing_adc_never_confirms_contact_and_requires_retry():
    c=MotionCapture(initial());c.index=6;c.confirm(c.request_id)
    for i in range(751):c.feed(frame(2000+i*20,False))
    assert c.status=='paused' and c.error and len(c.rows)==48
    assert not confirmed_touch(dict(stage='pinch_index',capture_protocol=PROTOCOL,motion_phase='release',contact_confirmed=True))


def test_a_gap_only_retries_current_cycle_and_resume_never_reuses_old_clock():
    c=MotionCapture(initial());c.confirm(c.request_id)
    for i in range(191):c.feed(frame(2000+i*20))
    c.feed(frame(6400))
    assert c.status=='paused' and c.snapshot()['retained_cycles']==1 and c.snapshot()['cycle']==2
    retained=copy.deepcopy(c.rows)
    assert len(retained)==48+150
    restored=MotionCapture.restore(c.document(),allow_partial=True)
    assert restored.rows==retained and restored.status=='paused'
    restored.confirm(restored.request_id)
    for i in range(301):restored.feed(frame(10000+i*20))
    assert restored.status=='ready' and restored.index==1
    assert restored.rows[:len(retained)]==retained
    assert {r['cycle'] for r in restored.rows[48:]}=={0,1,2}


def test_phase_cues_match_the_motion_and_contact_timer():
    from moxian_web.motion_capture import phase_at
    assert [phase_at(STAGES[0],t) for t in (0.,1.1,2.2)]==['extend','curl','return']
    assert [phase_at(STAGES[6],t) for t in (.2,1.2,2.2,3.7)]==['apart','approach','touch','release']


def test_motion_application_requires_all_new_gates():
    r=dict(fit_revision=MANO_FIT_REVISION,solver_revision=MANO_SOLVER_REVISION,capture_protocol=PROTOCOL,
           validation_pose_source='runtime_mano_sensor_only',max_penetration_mm=1.,runtime_max_penetration_mm=1.)
    keys=('accepted','contact_validation_passed','together_validation_passed','flat_validation_passed','penetration_validation_passed',
          'runtime_validation_passed','per_stage_validation_passed','holdout_validation_passed','independent_cycles_valid','approach_validation_passed','skeleton_validation_passed')
    r.update(dict.fromkeys(keys,True));assert calibration_report_valid(r)
    for key in keys:
        bad={**r,key:False};assert not calibration_report_valid(bad)


def test_pending_bundle_uses_training_source_then_live_bundle_requires_new_geometry():
    candidate=dict(calibration_fingerprint='capture',skeleton_runtime_revision='base',skeleton_model=dict(geometry_hash='candidate'))
    before=dict(calibration_fingerprint='capture',skeleton_runtime_revision='base',skeleton={})
    assert tracking_registration_matches(candidate,before,pending=True)
    assert not tracking_registration_matches(candidate,before)
    after=dict(calibration_fingerprint='capture',skeleton_runtime_revision='new',skeleton=dict(personalized_geometry_hash='candidate'))
    assert not tracking_registration_matches(candidate,after)
    assert tracking_registration_matches({**candidate,'skeleton_runtime_revision':'new'},after)
    assert not tracking_registration_matches(candidate,after,pending=True)
    after['calibration_fingerprint']='other'
    assert not tracking_registration_matches(candidate,after)
