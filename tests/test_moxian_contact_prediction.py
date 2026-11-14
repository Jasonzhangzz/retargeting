import copy
import numpy as np
import pytest
from moxian_web.contact_prediction import ContactPredictor,sensor_pose51,taxel_probabilities


def frame(t=1000,side='right'):
    return dict(timestamp_ms=t,hand_side=side,coordinate_frame='mano_wrist_local',
        raw_official_pose_axis_angle=np.arange(45,dtype=float).reshape(15,3).tolist(),
        pose_axis_angle=np.full((15,3),999.).tolist(),geometry_version='shape',binding_version='binding',
        source_contact=dict(ready=True,timestamp_ms=t,cell_contact=[True,False]+[None]*458,normalized=[0.]*460),
        contact_registration=dict(verified_sids=[0]))


def topology():
    return dict(faces=[[0,1,2]],sid_face_ids=[0]*460,sid_barycentric=[[.2,.3,.5]]*460)


def engine():
    inputs=[]
    def predict(pose):inputs.append(pose.copy());return np.ones(778)*.7,.8
    return ContactPredictor(predict),inputs


def test_predictor_receives_only_raw_rotations_and_left_mirror():
    a=frame();a['pose_axis_angle']=np.zeros((15,3)).tolist()
    p=sensor_pose51(a);assert p.shape==(51,);np.testing.assert_equal(p[:6],0)
    np.testing.assert_equal(p[6:],np.arange(45))
    a['hand_side']='left';q=sensor_pose51(a)
    np.testing.assert_equal(q[6:].reshape(15,3),np.arange(45).reshape(15,3)*[1,-1,-1])
    del a['raw_official_pose_axis_angle']
    with pytest.raises(ValueError):sensor_pose51(a)


def test_tactile_and_tactile_corrected_pose_cannot_change_predictions():
    e,inputs=engine();a=frame();one=e.step(a,topology(),1)
    b=frame(1030);b['source_contact']['cell_contact']=[False]*460
    b['source_contact']['normalized']=[1.]*460;b['pose_axis_angle']=np.full((15,3),-900).tolist()
    b['prediction_reference']=dict(raw_adc=[255]*460)
    two=e.step(b,topology(),1)
    np.testing.assert_equal(inputs[0],inputs[1])
    assert one['vertex_probability']==two['vertex_probability']
    assert one['comparison']['illustrative']['tp']==1 and two['comparison']['illustrative']['tp']==0


def test_history_is_causal_deduplicated_and_resets_on_gap_or_side_change():
    e,inputs=engine()
    for i in range(7):assert not e.step(frame(1000+i*30),topology())['ready']
    assert not inputs
    result=e.step(frame(1210),topology());assert result['ready'] and inputs[-1].shape==(1,8,51)
    count=e.scored_frames
    duplicate=e.step(frame(1210),topology());assert duplicate['duplicate'] and e.scored_frames==count and len(inputs)==1
    assert e.step(frame(1600),topology())['history_frames']==1
    assert e.step(frame(1630,'left'),topology())['history_frames']==1


def test_missing_and_stale_tactile_stay_unknown_but_do_not_stop_inference():
    e,_=engine();a=frame();a['source_contact']['ready']=False
    r=e.step(a,topology(),1);assert r['ready'] and not r['comparison']['ready']
    assert r['comparison']['illustrative'] is None and r['comparison']['actual_contact']==[None]*460
    a=frame(1030);a['source_contact']['timestamp_ms']=1000
    r=e.step(a,topology(),1);assert not r['comparison']['ready'] and r['comparison']['illustrative'] is None


def test_barycentric_sampling_and_only_measured_sids_enter_verified_metrics():
    p=np.zeros(778);p[:3]=[0,1,.5]
    np.testing.assert_allclose(taxel_probabilities(p,topology()),.55)
    t=topology();t['sid_barycentric'][0]=[1,1,1]
    with pytest.raises(ValueError):taxel_probabilities(p,t)
    e,_=engine();r=e.step(frame(),topology(),1)['comparison']
    assert r['illustrative']['samples']==2 and r['verified']['samples']==1
    assert r['illustrative']['fp']==1 and r['verified']['fp']==0
    a=frame(1030);a['contact_registration']['verified_sids']=[]
    assert e.step(a,topology(),1)['comparison']['verified'] is None
    a=frame(1060);t=topology();t['binding_version']='different'
    with pytest.raises(ValueError):e.step(a,t,1)


def test_dispatch_uses_lightweight_pose_packet_without_mesh_or_audit(tmp_path):
    from types import SimpleNamespace
    import threading,time
    from moxian_web.contact_prediction_bridge import ContactPredictionBridge
    h=SimpleNamespace(lock=threading.Lock(),pose_at=time.monotonic(),state=dict(frame(),status='default',vertices_m=[[9,9,9]]*778),model=lambda:dict(topology(),ready=True))
    bridge=ContactPredictionBridge(tmp_path,tmp_path);bridge.proc=SimpleNamespace(poll=lambda:None)
    sent=[];bridge.send=sent.append;bridge.dispatch(h)
    assert len(sent)==1 and 'vertices_m' not in sent[0]['frame'] and 'pose_axis_angle' not in sent[0]['frame']
    assert sent[0]['frame']['raw_official_pose_axis_angle']==h.state['raw_official_pose_axis_angle']
    bridge.dispatch(h);assert len(sent)==1
    h.pose_at=time.monotonic()-1;h.state['timestamp_ms']+=20;bridge.dispatch(h);assert len(sent)==1
