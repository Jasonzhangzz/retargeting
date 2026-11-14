"""Static gesture constraints are instantaneous and independent of live IK."""
from types import SimpleNamespace
import copy
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from moxian_web.qnbot_surface import QnbotSurface
from moxian_web.bone_extrinsics import definition_from_surface,BoneKinematics,FINGERS
from moxian_web.static_pose import StaticPoseValidator
from moxian_web.static_capture import static_schedule
from moxian_web.calibration_targets import target


@pytest.fixture(params=['left','right'])
def baseline(request):
    model=QnbotSurface(request.param);definition=definition_from_surface(model);kin=BoneKinematics(definition)
    q=np.asarray(definition['rest_angles_rad']);fingers={}
    for i,f in enumerate(FINGERS):
        r,p=kin.sensor(i,q);fingers[f]=dict(position=p.tolist(),quaternion_xyzw=Rotation.from_matrix(r).as_quat().tolist())
    row=dict(stage='open',timestamp_ms=1000,fingers=fingers,skeleton=dict(joint_names=model.joint_names,joint_positions_rad=q.tolist()))
    gate=StaticPoseValidator.from_reference(SimpleNamespace(model=model,fingerprint='fixed-eight'),dict(rows=[row],calibration_fingerprint='eight'))
    return gate,row


def put_pad(row,gate,f,position,normal):
    a=gate.patches[FINGERS[f]+'_distal'];local=np.asarray(a['normal_local'])
    # This fixture constructs sensor observations, not IK joint targets.
    dr=Rotation.align_vectors([normal],[local])[0].as_matrix()
    origin=np.asarray(position)-dr@a['offset_m']
    raw=gate.tx[:3,:3].T@(origin+dr@gate.rx[f,:3,3]-gate.tx[:3,3])
    rotation=gate.tx[:3,:3].T@dr@gate.rx[f,:3,:3]
    row['fingers'][FINGERS[f]]=dict(position=raw.tolist(),quaternion_xyzw=Rotation.from_matrix(rotation).as_quat().tolist())


def test_all_sixteen_contact_targets_accept_their_static_geometry_and_reject_flat(baseline):
    gate,flat=baseline
    for _,stage,endpoint in static_schedule()[:16]:
        row=copy.deepcopy(flat);row.update(stage=stage,static_endpoint=endpoint)
        assert not gate(row)['accepted'],stage
        spec=target(stage)
        if spec['kind']=='distal_cluster' or stage.startswith('contact_pair_'):
            for f in spec['active_fingers']:
                put_pad(row,gate,FINGERS.index(f),[0.,0.,.1],[0,0,1] if f=='thumb' else [0,0,-1])
        else:
            for edge in spec['edges']:
                distal,fixed=edge['regions'];f=FINGERS.index(distal.split('_')[0]);anchor=gate.patches[fixed]
                put_pad(row,gate,f,anchor['position_m'],-np.asarray(anchor['normal']))
        result=gate(row)
        assert result['accepted'],(stage,result['failures'])
        # A bogus upstream personalized IK pose cannot change admission.
        row['skeleton']['joint_positions_rad']=[1000]*25
        assert gate(row)==result


def test_pinch_uses_wuji_fifty_mm_rx_gate(baseline):
    gate,row=baseline;row=copy.deepcopy(row);row.update(stage='contact_pair_index',static_endpoint='target')
    put_pad(row,gate,0,[0,0,.1],[0,0,1]);put_pad(row,gate,1,[0,0,.1],[0,0,-1])
    assert gate(row)['accepted']
    row['fingers']['thumb']['position'][0]+=.1
    result=gate(row)
    rx=[check for check in result['checks'] if 'RX' in check['label']]
    assert not result['accepted'] and rx[0]['threshold']==50 and not rx[0]['passed']


def test_spread_and_together_are_different_static_targets(baseline):
    gate,flat=baseline
    def pose(spread,spacing=None):
        row=copy.deepcopy(flat);row.update(stage='motion_spread_together',static_endpoint='spread' if spread else 'together')
        for f in range(1,5):
            direction=Rotation.from_rotvec(gate.normal*np.deg2rad((f-2.5)*(28 if spacing is None else spacing) if spread else 0)).apply(gate.forward)
            dr=Rotation.align_vectors([direction],[gate.spec['direction_local'][f]])[0].as_matrix()
            length=np.linalg.norm(gate.flat_tips[f]-gate.mcp[f]);tip=gate.mcp[f]+direction*length
            origin=tip-dr@gate.tip_offsets[f]
            raw=gate.tx[:3,:3].T@(origin+dr@gate.rx[f,:3,3]-gate.tx[:3,3])
            rr=gate.tx[:3,:3].T@dr@gate.rx[f,:3,:3]
            row['fingers'][FINGERS[f]]=dict(position=raw.tolist(),quaternion_xyzw=Rotation.from_matrix(rr).as_quat().tolist())
        return row
    relaxed=pose(True,spacing=14);relaxed['static_endpoint']='together'
    assert gate(relaxed)['accepted'],gate(relaxed)['failures']
    spread=pose(True);together=pose(False)
    assert gate(spread)['accepted'],gate(spread)['failures']
    assert gate(together)['accepted'],gate(together)['failures']
    spread['static_endpoint']='together';together['static_endpoint']='spread'
    assert not gate(spread)['accepted'] and not gate(together)['accepted']


def test_correct_static_pose_collects_without_any_motion_excursion(baseline):
    from moxian_web.static_capture import StaticHandCapture
    gate,flat=baseline;flat=copy.deepcopy(flat);flat['calibration_fingerprint']='eight'
    capture=StaticHandCapture(dict(rows=[flat],calibration_fingerprint='eight'),pose_validator=gate)
    row=copy.deepcopy(flat)
    put_pad(row,gate,0,[0,0,.1],[0,0,1]);put_pad(row,gate,1,[0,0,.1],[0,0,-1])
    capture.confirm(capture.request_id)
    for index in range(1,180):
        row['timestamp_ms']=2000+index*20;capture.feed(row)
        if capture.index:break
    assert capture.index==1 and capture.aggregates[0]['static_pose_validation']['accepted']
    assert capture.aggregates[0]['static_sample_count']==20


def test_pinch_admission_does_not_require_uncalibrated_pad_normals(baseline):
    gate,row=baseline;row=copy.deepcopy(row)
    row.update(stage='contact_pair_index',static_endpoint='target')
    # Co-located RXs with equal orientation meet Wuji's distance gate,
    # regardless of the initial anatomical proxy's normals and offsets.
    row['fingers']['index']=copy.deepcopy(row['fingers']['thumb'])
    result=gate(row)
    assert result['accepted'],result['failures']
    assert all('RX' in c['label'] for c in result['checks'] if c['required'])


def test_five_finger_cluster_allows_neighbors_without_thumb_star(baseline):
    gate,row=baseline;row=copy.deepcopy(row);row['stage']='contact_five'
    for f in range(5):
        put_pad(row,gate,f,[0,0,.1],[0,0,1])
    # Sensor chain: the little RX is 100mm from thumb, but adjacent
    # receivers are only 25mm apart. No opposed-normal condition applies.
    for f in range(5):row['fingers'][FINGERS[f]]['position'][0]=.025*f
    result=gate(row)
    assert result['accepted'],result['failures']
    row['fingers']['pinky']['position'][0]=.5
    result=gate(row)
    assert not result['accepted'] and '五指接近距离' in result['failures'][0]


def test_palm_contact_admission_does_not_require_mesh_normal_angle(baseline):
    gate,row=baseline;row=copy.deepcopy(row);row['stage']='contact_palm_index'
    anchor=gate.patches['palm_mcp_index']
    put_pad(row,gate,0,anchor['position_m'],np.asarray(anchor['normal']))
    result=gate(row)
    assert result['accepted'],result['failures']
    normals=[c for c in result['checks'] if '角度' in c['label']]
    assert normals and not normals[0]['required'] and not normals[0]['passed']
    row['fingers']['thumb']['position'][0]+=.2
    assert not gate(row)['accepted']
