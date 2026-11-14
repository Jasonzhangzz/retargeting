"""Pinned native objective, native pose thresholds and display-only adaptation."""
import copy,json,threading,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from moxian_web.wuji_native_abi import NativeCalibrationABI,SDK_POSES,FINGERS
from moxian_web.wuji_native_capture import WujiNativeCapture,PROTOCOL
from moxian_web.wuji_native_model import NativeURDF,build_candidate,validate_native_candidate
from moxian_web.bone_extrinsics import publish_active,read_active
from moxian_web.contact import default_mapping

ROOT=Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def abi():return NativeCalibrationABI()


def observation(stamp,roll=0.,offset=0.):
    return dict(timestamp_ms=stamp,calibration_fingerprint='native-test',mapping=dict(assignments=[0]*460),
        fingers={f:dict(position=[offset+i*.002,0,.1],quaternion_xyzw=Rotation.from_euler('xyz',[roll,0,0],degrees=True).as_quat().tolist()) for i,f in enumerate(FINGERS)})


def test_original_six_pose_order_and_float32_pinch_boundary(abi):
    assert tuple(p['id'] for p in abi.poses)==SDK_POSES
    row=observation(1);threshold=abi.poses[0]['constraints'][0]['distance_m']
    row['fingers']['thumb']['position']=[0,0,0];row['fingers']['index']['position']=[threshold,0,0]
    assert abi.check_pose('pinch_index',row['fingers'],'right')['accepted']
    row['fingers']['index']['position'][0]=float(np.nextafter(np.float32(threshold),np.float32(1)))
    assert not abi.check_pose('pinch_index',row['fingers'],'right')['accepted']


@pytest.mark.parametrize('side,roll',[('right',-90.),('left',90.)])
def test_original_four_finger_orientation_constraints_and_side_mirroring(abi,side,roll):
    row=observation(1,roll)
    assert abi.check_pose('four_finger_bend_90',row['fingers'],side)['accepted']
    assert not abi.check_pose('flat_open',row['fingers'],side)['accepted']
    row=observation(1,0)
    assert abi.check_pose('flat_open',row['fingers'],side)['accepted']
    assert not abi.check_pose('four_finger_bend_90',row['fingers'],side)['accepted']
    np.testing.assert_allclose(abi.rpy_degrees(Rotation.from_euler('xyz',[-80,20,30],degrees=True).as_quat()),[-80,20,30],atol=1e-5)


def test_native_capture_movement_hold_collection_and_terminal_state(abi):
    c=WujiNativeCapture(observation(0),side='right',abi=abi);c.confirm(c.request_id);stamp=0
    for index,stage in enumerate(SDK_POSES):
        assert c.stage_id==stage and c.status=='collecting'
        roll=-90 if stage=='four_finger_bend_90' else 0
        # Movement detection, then an uninterrupted stable hold + collect.
        for i in range(21):stamp+=20;c.feed(observation(stamp,roll,offset=.03*(i%2)))
        for _ in range(180):
            stamp+=20;c.feed(observation(stamp,roll))
            if c.index!=index:break
        assert c.index==index+1
        assert c.aggregates[-1]['static_end_ms']-c.aggregates[-1]['static_start_ms']>=1000
    assert c.status=='complete' and c.snapshot()['can_fit']
    assert c.snapshot()['completed_train']==6 and len(c.pose_frames)==6
    before=len(c.rows);c.feed(observation(stamp+20));assert len(c.rows)==before
    restored=WujiNativeCapture.restore(c.document());assert restored.status=='complete' and restored.index==6


def test_bad_pose_and_gap_restart_current_native_hold(abi):
    c=WujiNativeCapture(observation(0),abi=abi);c.confirm(c.request_id)
    for i in range(21):c.feed(observation(i*20,offset=.03*(i%2)))
    for i in range(21,66):c.feed(observation(i*20))
    assert c.hold_since is not None and not c.collected
    wrong=observation(1320);wrong['fingers']['index']['position'][0]+=.2;c.feed(wrong)
    assert c.hold_since is None and not c.collected
    c.feed(observation(1800));assert len(c.window)==1
    c.feed(observation(121000));assert c.status=='paused'


def builtin_profiles():
    wrist=(ROOT/'tests/fixtures/wuji_2026_7_1_wrist.urdf').read_text()
    return {p:dict(wrist_urdf=wrist,tx_urdf=wrist) for p in ('wujihand','wujihand2')}


def test_original_sdk_ik_21_dof_fk_and_mesh_consume_same_model(tmp_path):
    import torch
    from moxian_web.wuji_native_ik import WujiNativeSkeleton
    from moxian_web.qnbot_surface import QnbotSurface
    from moxian_web.qnbot_worker import StreamingModel
    torch.set_num_threads(1);mapping=dict(assignments=default_mapping('right'),confirmed=[])
    candidate=build_candidate(builtin_profiles(),dict(serial='NATIVE-TEST',side='right',calibration_fingerprint='native-test'),mapping,'test')
    validate_native_candidate(candidate);publish_active(tmp_path,candidate);assert read_active(tmp_path)['model_hash']==candidate['model_hash']
    native=NativeURDF(candidate['wuji_native']['profiles']['wujihand']['wrist_urdf']);angles=np.clip(np.full(21,.3),native.limits[:,0],native.limits[:,1]);poses=native.forward(angles);tx=np.linalg.inv(poses['emf_tx_base']);nodes={}
    for f in FINGERS:
        t=tx@poses[f+'_rx_coil'];nodes[f]=dict(position=t[:3,3].tolist(),quaternion_xyzw=Rotation.from_matrix(t[:3,:3]).as_quat().tolist())
    engine=WujiNativeSkeleton(candidate)
    try:
        for i in range(30):result=engine.step(dict(timestamp_ms=1000+i*20,fingers=nodes))
        pose=result['pose'];assert pose['native_dof']==21 and len(pose['joint_positions_rad'])==21 and len(pose['positions_local'])==25
        # Public SDK skeleton must agree with our URDF FK, independently of
        # whether IK can exactly attain the requested observations.
        response=engine.request(dict(timestamp_ms=1600,poses=list(nodes.values()),include_skeleton=True))
        q=np.concatenate([response['angles'][0],*[v[:4] for v in response['angles'][1:]]]);fk=native.forward(q)
        np.testing.assert_allclose(response['skeleton_positions'],[fk[name][:3,3] for name in native.points],atol=1e-7)
        model=QnbotSurface('right');model.bind_mapping(mapping);runtime=StreamingModel(model);runtime.load(candidate)
        v,j,_,q,_,_=runtime.infer(dict(timestamp_ms=pose['timestamp_ms'],fingers=nodes,skeleton=pose))
        assert torch.isfinite(v).all() and v.shape==(1,11598,3) and q.shape==(1,21)
        np.testing.assert_allclose(j[0].numpy(),pose['display_body_positions'],atol=1e-7)
        with pytest.raises(ValueError,match='版本'):runtime.infer(dict(timestamp_ms=pose['timestamp_ms'],fingers=nodes,skeleton=dict(pose,model_hash='other')))
        # A discontinuity resets the native solver; out-of-order timestamps fail.
        engine.step(dict(timestamp_ms=3000,fingers=nodes));assert engine.frames==1
        with pytest.raises(ValueError,match='递增'):engine.step(dict(timestamp_ms=2999,fingers=nodes))
    finally:engine.close()


@pytest.mark.parametrize("side",["right","left"])
def test_full_native_objective_child_returns_both_profiles_without_applying(tmp_path,side):
    from moxian_web.wuji_native_calibration import fit_native_capture
    native=NativeURDF(builtin_profiles()['wujihand']['wrist_urdf']);frames={}
    for stage in SDK_POSES:
        q=np.zeros(21) if stage=='flat_open' else np.full(21,.5);fk=native.forward(q);tx=np.linalg.inv(fk['emf_tx_base']);nodes=[]
        for f in FINGERS:
            t=tx@fk[f+'_rx_coil'];nodes.append(dict(position=t[:3,3].tolist(),quaternion_xyzw=Rotation.from_matrix(t[:3,:3]).as_quat().tolist()))
        frames[stage]=[nodes]
    mapping=dict(assignments=default_mapping(side),confirmed=[])
    doc=dict(protocol=PROTOCOL,side=side,calibration_fingerprint='native-test',mapping=mapping,completed_groups=list(SDK_POSES),state=dict(state='complete'),pose_frames=frames)
    profile=dict(serial='SYNTHETIC-NO-DEVICE',side=side,calibration_fingerprint='native-test',capture_session_id='synthetic')
    candidate,report=fit_native_capture(profile,mapping,doc,lambda _:None,threading.Event())
    assert report['accepted'] and report['native_calibration_succeeded']
    assert report['solver']['native_caliber_objective_reused'] and report['solver']['optimized_dimension']==177
    assert set(candidate['wuji_native']['profiles'])=={'wujihand','wujihand2'}
    assert report['validation_scope']=='wuji_native_result_and_urdf_integrity'
    assert not (tmp_path/'active.json').exists()
    validate_native_candidate(candidate)

    assert set(report['solver']['profiles'])=={'wujihand','wujihand2'}
    assert all(v['evaluations']>0 for v in report['solver']['profiles'].values())
    from moxian_web.wuji_native_ik import WujiNativeSkeleton
    engine=WujiNativeSkeleton(candidate)
    try:
        response=engine.request(dict(timestamp_ms=1000,poses=frames['flat_open'][0],include_skeleton=True))
        q=np.concatenate([response['angles'][0],*[v[:4] for v in response['angles'][1:]]])
        kinematics=engine.urdf.forward(q)
        np.testing.assert_allclose(response['skeleton_positions'],[kinematics[name][:3,3] for name in engine.urdf.points],atol=1e-7)
    finally:engine.close()
    cancel=threading.Event()
    def stop_at_native_start(event):cancel.set()
    with pytest.raises(InterruptedError,match='取消'):
        fit_native_capture(profile,mapping,doc,stop_at_native_start,cancel)

@pytest.mark.parametrize('side',['right','left'])
def test_pose_gate_aligns_rx_mount_axes_without_changing_angles_or_native_fit_input(side):
    from moxian_web.wuji_gesture_frame import from_capture,orient_for_gate
    flat={};bent={};shallow={}
    for i,f in enumerate(FINGERS):
        r=Rotation.from_euler('xyz',[15+i*12,60+i*2,-65+i*11],degrees=True).as_matrix()
        axis=np.asarray([.2+i*.2,.7-i*.1,.3]);axis/=np.linalg.norm(axis)
        def node(angle):
            q=Rotation.from_matrix(Rotation.from_rotvec(axis*np.deg2rad(angle)).as_matrix()@r).as_quat().tolist()
            return dict(position=[i*.002,0,.1],quaternion_xyzw=q)
        flat[f]=node(0);bent[f]=node(80);shallow[f]=node(25)
    def frames(nodes):
        opposite=copy.deepcopy(nodes)
        for n in opposite.values():n['quaternion_xyzw']=[-v for v in n['quaternion_xyzw']]
        return [dict(payload=dict(fingertip_local=nodes)),dict(payload=dict(fingertip_local=opposite))]
    doc=dict(payload_sha256='verified-eight',payload=dict(side=side,stage_samples=[dict(stage_id='open',frames=frames(flat)),dict(stage_id='mcp90',frames=frames(bent))]))
    reference=from_capture(doc,'native-test',side);before=copy.deepcopy(bent);canonical=orient_for_gate(bent,reference);abi=NativeCalibrationABI()
    assert bent==before
    assert not abi.check_pose('four_finger_bend_90',bent,side)['accepted']
    assert abi.check_pose('four_finger_bend_90',canonical,side)['accepted']
    assert not abi.check_pose('four_finger_bend_90',orient_for_gate(shallow,reference),side)['accepted']
    assert abi.check_pose('flat_open',orient_for_gate(flat,reference),side)['accepted']
    for f in FINGERS[1:]:assert abs(abs(abi.rpy_degrees(canonical[f]['quaternion_xyzw'])[0])-80)<1e-4
    # Collection checks canonical orientations, but retains the original RX
    # observations for the unchanged full-native calibration objective.
    c=WujiNativeCapture(observation(0),side=side,abi=abi,gesture_frame=reference)
    c.index=4;c.confirm(c.request_id)
    for i in range(21):c.feed(dict(observation(i*20),fingers={f:dict(v,position=[v['position'][0]+.03*(i%2),0,.1]) for f,v in bent.items()}))
    for i in range(21,200):
        c.feed(dict(observation(i*20),fingers=bent))
        if c.index==5:break
    assert c.index==5
    assert c.pose_frames['four_finger_bend_90'][0]==[bent[f] for f in FINGERS]
