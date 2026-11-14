import copy,hashlib,json
from pathlib import Path
import pytest
from moxian_web.shared_capture import load_official_capture,OfficialSensorRecorder,STAGE_MAP
from moxian_web.contact import default_mapping


def official_data(tmp_path):
    artifact=tmp_path/'glove/calibrations/test/artifacts/hand_system.yaml';artifact.parent.mkdir(parents=True);artifact.write_text('native calibration')
    fingerprint=hashlib.sha256(artifact.read_bytes()).hexdigest();payload_sha='f'*64
    (artifact.parent.parent/'record.yaml').write_text('capture_payload_sha256: '+payload_sha+'\n')
    groups=[]
    for i,stage in enumerate(STAGE_MAP):
        frames=[]
        for j in range(6):frames.append({'payload':{'timestamp_ms':1000+i*200+j*10,'fingertip_local':{f:{'position':[.01,.02,.03],'quaternion_xyzw':[0.,0.,0.,1.]} for f in ('thumb','index','middle','ring','pinky')}}})
        groups.append({'stage_id':stage,'frames':frames})
    doc={'payload_sha256':payload_sha,'payload':{'side':'left','capture_profile':'qnbot.hand.capture.eight.v1','requirements':[{'stage_id':s,'required_sample_count':6} for s in STAGE_MAP],'stage_samples':groups}}
    path=tmp_path/'glove/captures/test.json';path.parent.mkdir();path.write_text(json.dumps(doc))
    mapping={'version':'test','assignments':default_mapping('left'),'confirmed':[]}
    return fingerprint,mapping,path,doc


def test_reuse_eight_complete_groups_without_fabricating_missing_tactile(tmp_path):
    fingerprint,mapping,path,doc=official_data(tmp_path);before=path.read_bytes();data=load_official_capture(tmp_path,fingerprint,mapping)
    assert data['total']==8 and data['frames']==48 and len(data['completed_groups'])==8
    assert data['tactile_matched_frames']==0 and all(r['raw'] is None and r['tactile_timestamp_ms'] is None and not r['contact']['ready'] for r in data['rows'])
    assert data['rows'][0]['stage']=='together' and data['rows'][-1]['stage']=='pinch_pinky'
    assert all(r['guided_contact'] and r['contact_constraint_source']=='user_confirmed_official_gesture' for r in data['rows'] if r['stage'].startswith('pinch_'))
    assert path.read_bytes()==before
    with pytest.raises(ValueError):load_official_capture(tmp_path,'bad fingerprint',mapping)
    doc['payload']['stage_samples'].pop();path.write_text(json.dumps(doc))
    with pytest.raises(ValueError):load_official_capture(tmp_path,fingerprint,mapping)


def test_record_tactile_during_same_sdk_capture_and_enforce_25_ms_match(tmp_path):
    fingerprint,mapping,_,_=official_data(tmp_path)
    frame={'timestamp_ms':1000,'tactile_timestamp_ms':1025,'raw':[17]*460,'contact':{'ready':True,'patches':[]}}
    recorder=OfficialSensorRecorder(tmp_path/'shared-captures');capture={'session_id':'test','session_state':'running','stage':{'stage_id':'close'}}
    recorder.feed(frame,capture,mapping,{'baseline':[10]*460});recorder.close()
    row=load_official_capture(tmp_path,fingerprint,mapping)['rows'][0]
    assert row['raw']==[17]*460 and row['tactile_matched'] and row['contact']['ready']
    sidecar=tmp_path/'shared-captures/test.jsonl';saved=json.loads(sidecar.read_text());saved['tactile_timestamp_ms']=1026;sidecar.write_text(json.dumps(saved)+'\n')
    row=load_official_capture(tmp_path,fingerprint,mapping)['rows'][0]
    assert row['raw'] is None and not row['contact']['ready']


def test_shared_sampling_has_disjoint_chronological_holdout_without_extra_cycles(tmp_path):
    from moxian_web.mano_fit import choose_rows,holdout_rows,trajectory_summary
    fingerprint,mapping,_,_=official_data(tmp_path);rows=load_official_capture(tmp_path,fingerprint,mapping)['rows'];original=copy.deepcopy(rows)
    train,valid=choose_rows(rows)
    assert len(train)==len(valid)==24 and rows==original
    assert not {r['timestamp_ms'] for r in train}&{r['timestamp_ms'] for r in valid}
    assert {r['timestamp_ms'] for r in valid}<={r['timestamp_ms'] for r in holdout_rows(rows)}
    summary=trajectory_summary([{'stage':'pinch_index','rmse_mm':2,'bilateral_contact':False,'guided_contact':True}])
    assert summary['pinch_index']['approach_frames']==0


def test_thumb_side_annotation_is_bound_to_capture_and_never_fabricates_adc(tmp_path):
    fingerprint,mapping,path,doc=official_data(tmp_path);original=path.read_bytes()
    note=dict(schema_version=1,side='left',source_payload_sha256=doc['payload_sha256'],calibration_fingerprint=fingerprint,
              together=dict(thumb_index_side_contact=True))
    annotation=tmp_path/'gesture-constraints.json';annotation.write_text(json.dumps(note))
    data=load_official_capture(tmp_path,fingerprint,mapping)
    assert all(r.get('thumb_index_side_contact') is True for r in data['rows'] if r['stage']=='together')
    assert not any(r.get('thumb_index_side_contact') for r in data['rows'] if r['stage']!='together')
    assert all(not r['contact']['ready'] and r['raw'] is None for r in data['rows'])
    assert path.read_bytes()==original
    note['source_payload_sha256']='wrong';annotation.write_text(json.dumps(note))
    assert not any(r.get('thumb_index_side_contact') for r in load_official_capture(tmp_path,fingerprint,mapping)['rows'])
