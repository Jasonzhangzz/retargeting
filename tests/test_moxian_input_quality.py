import copy
import json
import pytest
from moxian_web.input_quality import capture_input_quality,PINCH_PARTNERS
from moxian_web.shared_capture import load_official_capture


def document(distance=.03):
    return {'payload':{'stage_samples':[
        {'stage_id':stage,'frames':[{'payload':{'fingertip_local':{
            'thumb':{'position':[0.,0.,0.]},finger:{'position':[distance,0.,0.]}}}}for _ in range(30)]}
        for stage,finger in PINCH_PARTNERS.items()]}}


def test_realistic_sensor_gap_is_not_forced_to_zero_or_rewritten():
    doc=document();before=copy.deepcopy(doc);q=capture_input_quality(doc)
    assert q['accepted'] and q['status']=='passed'
    assert all(p['median_sensor_distance_mm']==30 for p in q['pairs'])
    assert doc==before and not q['pose_correction'] and not q['geometry_modified']


def test_raw_cross_channel_distance_requires_registration_not_hardware_rejection():
    doc=document(.20);q=capture_input_quality(doc)
    assert q['accepted'] and q['status']=='registration_required'
    assert len([p for p in q['pairs']if p['grossly_inconsistent']])==4
    good=document();good['payload']['stage_samples'][0]['frames'][0]['payload']['fingertip_local']['thumb']['position']=[1.,1.,1.]
    assert capture_input_quality(good)['accepted']


def test_missing_or_nonfinite_guided_pose_is_unknown_not_passed():
    doc=document();doc['payload']['stage_samples'][0]['frames']=[]
    assert capture_input_quality(doc)['status']=='unknown'
    doc=document(float('nan'));assert not capture_input_quality(doc)['accepted']


def test_shared_loader_refuses_missing_inputs_before_any_shape_fitting(tmp_path,monkeypatch):
    from moxian_web import shared_capture
    doc=document(float('nan'));p=doc['payload'];p.update(side='left',capture_profile='qnbot.hand.capture.eight.v1',requirements=[])
    for stage in ('close','open','mcp90','mcp90_pip90'):
        p['stage_samples'].append({'stage_id':stage,'frames':[{}]*30})
    monkeypatch.setattr(shared_capture,'official_capture_document',lambda *_:(tmp_path/'source.json',doc))
    with pytest.raises(ValueError,match='无法核验'):load_official_capture(tmp_path,'capture',{'assignments':[]})
