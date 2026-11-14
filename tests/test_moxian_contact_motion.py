"""New protocol contract: synthetic trajectories are tests, never capture data."""
import copy,json
import numpy as np
import pytest
from moxian_web.contact_protocol import PROTOCOL,STAGES,extract_events,cluster_tree
from moxian_web.contact_capture import ContactMotionCapture,validate_document
from moxian_web.motion_capture import MotionCapture,confirmed_touch
from moxian_web.calibration_targets import target
from moxian_web.mano_objective import participation,stage_weights
from moxian_web.training_options import training_rows


def frame(stamp,gap=.03):
    return dict(timestamp_ms=stamp,calibration_fingerprint='fixed',mapping={'assignments':[0]},test_gap=gap,
                fingers={f:dict(position=[0,0,0],quaternion_xyzw=[0,0,0,1]) for f in ('thumb','index','middle','ring','pinky')})


def capture():
    reference=lambda r:dict(reference_gap_m=r['test_gap'],contact_reference_id='frozen-reference')
    reference.spec={}
    from moxian_web.contact_protocol import schedule
    c=ContactMotionCapture(dict(rows=[dict(frame(-20),stage='open')],completed_groups=['open'],calibration_fingerprint='fixed',motion_schedule=schedule(False)),
        reference=reference)
    return c


def window(c,origin,one=False):
    c.confirm(c.request_id)
    times=[0,.45,.9,1.4,2.0,3.6,4.2,5.] if not one else [0,.45,.9,1.4,5.]
    gaps=[.03,.02,.001,.025,.03,.001,.03,.03] if not one else [.03,.02,.001,.03,.03]
    for t in np.arange(0,5.001,.02):c.feed(frame(origin+t*1000,float(np.interp(t,times,gaps))))


def test_16_action_schedule_and_region_contract():
    assert len(STAGES)==16 and sum(s['seconds'] for s in STAGES)==80
    assert [target(s['id'])['kind'] for s in STAGES].count('distal_cluster')==6
    for s in STAGES:
        assert target(s['id'])['repetitions']==2 and s['seconds']==5
    assert target('contact_fist')['active_fingers']==['index','middle','ring','pinky']
    assert all(e['regions'][1].startswith('palm_mcp_') for e in target('contact_fist')['edges'])
    assert all(e['regions'][1].endswith('_proximal') for e in target('contact_fold')['edges'])


def test_default_17_actions_and_unfinished_legacy_extension_preserve_current_stage():
    c=capture();old_schedule=copy.deepcopy(c.schedule);c.index=20
    c.extend_spread_schedule()
    assert c.schedule[:32]==old_schedule and len(c.schedule)==34
    assert c.stage['id']=='contact_tripod_index_middle' and c.partition=='validation'
    assert c.snapshot()['actions_per_pass']==17 and c.snapshot()['total']==34
    assert not c.extend_spread_schedule()
    c.status='complete';assert not c.extend_spread_schedule()
    initial=dict(rows=[dict(frame(-20),stage='open')],completed_groups=['open'],calibration_fingerprint='fixed')
    default=ContactMotionCapture(initial,c.reference)
    assert len(default.schedule)==34 and default.schedule[16]==['train','motion_spread_together']


def test_spread_together_events_supervise_all_fingers_without_contact_equalities():
    c=capture();c.metadata['motion_schedule']=[['train','motion_spread_together'],['validation','motion_spread_together']]
    # Final adduction can finish together, without an artificial third opening.
    c.confirm(c.request_id)
    for t in np.arange(0,5.001,.02):
        gap=np.interp(t,[0,.6,1.,1.4,2.,3.,3.5,4.,5.],[.03,.01,.01,.03,.03,.03,.01,.01,.01])
        c.feed(frame(t*1000,float(gap)))
    assert c.index==1 and c.status=='ready'
    rows=[r for r in c.rows if not r.get('initialization')]
    events=[r for r in rows if r.get('motion_event_confirmed')]
    assert len(events)==2 and not any(confirmed_touch(r) for r in rows)
    assert all(r['observation_weights']==[1.]*5 for r in rows)
    assert all(r['motion_constraint_source']=='trajectory_assumed_adduction' for r in events)


def test_full_extended_passes_validate_motion_events_separately_from_contacts():
    c=capture();c.extend_spread_schedule();origin=0
    while c.status!='complete':window(c,origin);origin+=6000
    doc=c.document();validate_document(doc)
    assert len([r for r in c.rows if r.get('contact_confirmed')])==64
    assert len([r for r in c.rows if r.get('motion_event_confirmed')])==4
    assert c.snapshot()['completed_train']==c.snapshot()['completed_validation']==17
    assert c.snapshot()['progress']==1. and c.snapshot()['estimated_remaining_s']==0.
    damaged=copy.deepcopy(doc)
    next(r for r in damaged['rows'] if r.get('motion_event_confirmed'))['contact_confirmed']=True
    with pytest.raises(ValueError,match='不应作为指尖接触'):validate_document(damaged)


def test_asymmetric_speed_plateaus_noise_and_no_fixed_hold():
    rng=np.random.default_rng(4)
    x=np.arange(0,5,.02)
    y=np.interp(x,[0,.5,.56,.6,1.2,3.8,4.,4.2,4.5,5],[.04,.001,.001,.001,.035,.035,.002,.002,.035,.04])
    rows=[dict(timestamp_ms=float(t*1000),reference_gap_m=float(g)) for t,g in zip(x,y+rng.normal(0,.00005,len(x)))]
    events=extract_events(rows)
    assert len(events)==2 and 450<events[0]['timestamp_ms']<650 and 3950<events[1]['timestamp_ms']<4250
    assert len(extract_events(rows[:70]))==1
    assert extract_events([dict(timestamp_ms=i*20,reference_gap_m=.01) for i in range(250)])==[]
    # A stream discontinuity is not evidence of a release.
    assert extract_events([dict(timestamp_ms=t,reference_gap_m=d) for t,d in [(0,.04),(20,.02),(40,.001),(500,.04)]])==[]


def test_separate_passes_preserved_raw_supplement_restore_cancel():
    c=capture();window(c,0,one=True)
    assert c.status=='paused' and len(c.events)==1
    saved=c.document();before=len(c.rows)
    restored=MotionCapture.restore(saved,allow_partial=True)
    assert isinstance(restored,ContactMotionCapture) and len(restored.events)==1 and len(restored.rows)==before
    restored.reference=c.reference;window(restored,6000,one=True)
    assert restored.status=='ready' and restored.index==1 and len(restored.rows)>before
    restored.cancel();assert restored.status=='cancelled';restored.retry();assert restored.index==1 and restored.status=='ready'
    origin=12000
    while restored.status!='complete':window(restored,origin);origin+=6000
    doc=restored.document();validate_document(doc)
    assert len([r for r in doc['rows'] if confirmed_touch(r) and r['partition']=='train'])==32
    assert len([r for r in doc['rows'] if confirmed_touch(r) and r['partition']=='validation'])==32
    assert all(r['partition']=='initialization' for r in doc['rows'] if r.get('initialization'))
    selected=training_rows(doc['rows'],full_buffer=True)
    assert all(r['partition']=='train' and not r.get('initialization') for r in selected)
    assert len({r['timestamp_ms'] for r in selected})==len(selected)
    corrupted=copy.deepcopy(doc);next(r for r in corrupted['rows'] if r.get('partition')=='validation')['capture_pass']='train'
    with pytest.raises(ValueError,match='分区'):validate_document(corrupted)


def test_missing_target_data_never_manufactures_contact_and_retry_keeps_raw():
    c=capture();c.confirm(c.request_id)
    for i in range(251):
        row=frame(i*20,.001);row['fingers'].pop('thumb');c.feed(row)
    assert c.status=='paused' and not c.events
    before=json.dumps(c.rows,sort_keys=True);n=len(c.rows);c.retry()
    assert len(c.rows)==n and all(not r.get('training_eligible',True) for r in c.rows[1:])
    assert not any(confirmed_touch(r) for r in c.rows)


def two_contacts_before_window_end(c,origin=0):
    c.confirm(c.request_id)
    for t in np.arange(0,3.001,.02):
        gap=np.interp(t,[0,.4,.8,1.1,1.5,1.9,2.3,2.7,3.],[.04,.025,.001,.025,.04,.025,.001,.03,.04])
        c.feed(frame(origin+t*1000,float(gap)))
    assert c.index==0 and len(c.events)==2 and c.status=='collecting'


def test_two_confirmed_contacts_advance_even_if_window_is_interrupted():
    c=capture();two_contacts_before_window_end(c)
    rows_before=copy.deepcopy(c.rows)
    c.pause_cycle('采集断流')
    assert c.status=='ready' and c.index==1 and not c.events and c.error is None
    assert c.rows==rows_before and c.complete_groups.count('train:contact_pair_index')==1


def test_restore_old_paused_two_of_two_advances_without_relabelling_raw():
    c=capture();two_contacts_before_window_end(c)
    document=c.document();document['state']['state']='paused';document['state']['error']='采集断流'
    rows_before=json.dumps(document['rows'],sort_keys=True)
    restored=MotionCapture.restore(document,allow_partial=True)
    assert restored.status=='ready' and restored.index==1 and not restored.events
    assert restored.complete_groups[-1]=='train:contact_pair_index'
    assert json.dumps(restored.rows,sort_keys=True)==rows_before
    again=MotionCapture.restore(restored.document(),allow_partial=True)
    assert again.index==1 and again.complete_groups.count('train:contact_pair_index')==1


def test_resume_old_two_of_two_advances_instead_of_requesting_another_window():
    c=capture();two_contacts_before_window_end(c);c.status='paused'
    n=len(c.rows);c.confirm(c.request_id)
    assert c.index==1 and c.status=='ready' and len(c.rows)==n


def test_stream_gap_preserves_five_second_window_but_never_invents_release():
    c=capture();c.confirm(c.request_id)
    # An approach interrupted at its minimum is not a completed event.
    for stamp,gap in [(0,.04),(20,.025),(40,.001),(600,.04)]:c.feed(frame(stamp,gap))
    assert c.status=='collecting' and not c.events and c.rows[-1]['capture_gap_before_ms']==560
    # One later, fully observed contact still counts despite the earlier gap.
    for t in np.arange(.62,5.001,.02):
        c.feed(frame(t*1000,float(np.interp(t,[.62,1.4,2.,2.6,5.],[.04,.04,.001,.04,.04]))))
    assert c.status=='paused' and len(c.events)==1 and c.index==0
    assert '缺少 1 次' in c.error and '采样缺口' in c.error


def test_two_closed_segments_separated_by_gap_complete_without_extra_attempt():
    c=capture();c.confirm(c.request_id)
    for t in np.arange(0,5.001,.02):
        if 1.5<t<2.1:continue
        gap=np.interp(t,[0,.6,1.1,1.4,2.1,3.,3.7,4.2,5.],[.04,.001,.04,.04,.04,.04,.001,.04,.04])
        c.feed(frame(t*1000,float(gap)))
    assert c.index==1 and c.status=='ready'
    assert sum(bool(r.get('contact_confirmed')) for r in c.rows)==2
    assert any(r.get('capture_gap_before_ms',0)>250 for r in c.rows)


def test_masks_and_weights_do_not_depend_on_repetition_or_edge_counts():
    rows=[dict(stage=s['id'],capture_protocol=PROTOCOL,partition='train',fingers=frame(0)['fingers']) for s in STAGES]
    mask=participation(rows)
    assert mask[0].tolist()==[1,1,0,0,0]
    assert mask[-1].tolist()==[0,1,1,1,1]
    assert mask[10].tolist()==[1,0,0,0,0]
    weights=stage_weights(rows,mask,True)
    repeated=rows+[copy.deepcopy(rows[0])]*7;w=stage_weights(repeated,participation(repeated),True)
    for i,s in enumerate(STAGES):
        assert np.isclose(weights[i].sum(),sum(w[j].sum() for j,r in enumerate(repeated) if r['stage']==s['id']))


def test_cluster_graph_connected_with_preferred_thumb_edges():
    regions=['thumb_distal','index_distal','middle_distal','ring_distal']
    p=np.array([[0,0],[1,0],[2,0],[3,0]])
    edges=cluster_tree(regions,np.linalg.norm(p[:,None]-p[None],axis=-1))
    assert sum(e['required'] for e in edges)==3
    assert {tuple(e['regions']) for e in edges}>={('thumb_distal',r) for r in regions[1:]}
