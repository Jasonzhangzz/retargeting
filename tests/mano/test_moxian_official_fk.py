"""Official link rotations are the sole runtime articulation source."""
import copy
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation
from unittest.mock import patch
from moxian_web.hand_model import DEFAULT_MANO
from moxian_web.mano_runtime import ManoSurface,CHAINS
from moxian_web.official_fk import OfficialSkeletonFK,LINKS,TRACKING_SOURCE

@pytest.fixture
def driven():
    if not DEFAULT_MANO.exists() or not torch.cuda.is_available():pytest.skip('Local MANO/CUDA required')
    m=ManoSurface(DEFAULT_MANO);m.set_dimensions(np.zeros(20));m.enable_official_fk()
    s=copy.deepcopy(m.official_fk.reference);s['timestamp_ms']=1000.
    frame={'timestamp_ms':1000.,'hand_side':'left','skeleton':s,'fingers':{f:{'position':[.01,.02,.03],'quaternion_xyzw':[0,0,0,1]} for f in ('thumb','index','middle','ring','pinky')}}
    return m,frame

def test_official_fk_never_runs_second_ik_filter_or_joint_projection(driven):
    m,f=driven
    with patch.object(m,'fit_fingertip_positions',side_effect=AssertionError('second IK')),patch.object(m,'pose_from_bones',side_effect=AssertionError('bone IK')),patch('moxian_web.mano_runtime.least_squares',side_effect=AssertionError('solver')),patch('moxian_web.pose_filter.SensorPoseFilter.update',side_effect=AssertionError('filter')):
        a=m.track(f)[0];b=copy.deepcopy(f);b['contact']={'ready':True,'patches':[{'id':'thumb_distal','contact':True}]};b['fingers']['thumb']['position']=[.5,.6,.7]
        assert np.array_equal(a,m.track(b)[0])
    assert m.alignment_quality['additional_ik'] is False and m.alignment_quality['additional_smoothing'] is False
    assert m.alignment_quality['max_link_rotation_error_deg']<1e-4

def test_all_fifteen_bone_axes_follow_sdk_and_same_frame_has_no_chasing(driven):
    m,f=driven;m.track(f);s=f['skeleton'];names=s['names'];p=np.array(s['positions_local']);r=Rotation.from_quat(s['orientations_local_xyzw']).as_matrix()
    # Change the complete index finger chain by a known rigid MCP motion.
    ids=[i for i,n in enumerate(names) if n.startswith('index_') and n!='index_metacarpal'];pivot=p[names.index('index_proximal')].copy();turn=Rotation.from_euler('x',55,degrees=True).as_matrix()
    p[ids]=(p[ids]-pivot)@turn.T+pivot;r[ids]=turn@r[ids]
    s['positions_local']=p.tolist();s['orientations_local_xyzw']=Rotation.from_matrix(r).as_quat().tolist();f['timestamp_ms']=s['timestamp_ms']=1010.
    a=m.track(f);first=m.last_pose.copy();m.track(f)
    assert np.array_equal(first,m.last_pose)
    for finger,c in enumerate(CHAINS):
        seq=LINKS[finger]+[('thumb' if finger==0 else ('index','middle','ring','little')[finger-1])+'_tip']
        for k,j in enumerate(c):
            end=m.tip_anchors[finger] if k==2 else m.rest_j_np[c[k+1]]
            rendered=m.official_fk.mano_to_sdk@(a[2][j]@(end-m.rest_j_np[j]));sdk=p[names.index(seq[k+1])]-p[names.index(seq[k])]
            assert np.dot(rendered,sdk)/(np.linalg.norm(rendered)*np.linalg.norm(sdk))>1-1e-8

def test_metacarpal_displacement_transfers_and_geometry_lengths_stay_fixed(driven):
    m,f=driven;before=m.track(f);rest=m.rest_np.copy();shape=m.dimension_parameters.copy();s=f['skeleton'];p=np.array(s['positions_local']);delta=np.array([.002,.003,-.001])
    ids=[i for i,n in enumerate(s['names']) if n.startswith('ring_') and n!='ring_metacarpal'];p[ids]+=delta;s['positions_local']=p.tolist();s['timestamp_ms']=f['timestamp_ms']=1020.
    after=m.track(f);transferred=m.official_fk.sdk_to_mano@delta
    assert np.allclose(after[1][list(CHAINS[3])]-before[1][list(CHAINS[3])],transferred,atol=1e-7)
    assert np.array_equal(m.rest_np,rest) and np.array_equal(m.dimension_parameters,shape)
    for c in CHAINS:assert np.allclose(np.linalg.norm(np.diff(before[1][list(c)],axis=0),axis=-1),np.linalg.norm(np.diff(after[1][list(c)],axis=0),axis=-1),atol=1e-7)

def test_official_joint_centres_and_training_decode_share_same_lbs(driven):
    m,f=driven;m.track(f);s=f['skeleton'];p=np.array(s['positions_local'])
    # The whole fixed link TF transfers this displacement, including its
    # calibrated translation; it must not silently restore SDK default spacing.
    p[s['names'].index('index_distal')]+=[.004,-.006,.002]
    s['positions_local']=p.tolist();v,j,r,o,_=m.track(f)
    for joint,source in m.official_fk.indices.items():
        r=Rotation.from_quat(s['orientations_local_xyzw'][source]).as_matrix()
        assert np.allclose(j[joint],m.official_fk.sdk_to_mano@(p[source]+r@m.official_fk.translations[joint]),atol=1e-7)
    q,root,base,centres=m.official_parameters([f])
    with torch.no_grad():decoded=m.decode(q,root,base,centres)
    assert np.allclose(decoded[0][0].cpu().numpy(),v,atol=1e-7)
    # No time-varying compression or rest-geometry writeback is involved.
    saved=m.rest_np.copy();m.track(f);assert np.array_equal(m.rest_np,saved)

def test_sdk_missing_stale_reordered_names_and_antipodal_quaternions(driven):
    m,f=driven;a=m.track(f)[0];reordered=copy.deepcopy(f);s=reordered['skeleton']
    for key in ('names','positions_local','orientations_local_xyzw'):s[key]=list(reversed(s[key]))
    s['orientations_local_xyzw']=(-np.array(s['orientations_local_xyzw'])).tolist()
    assert np.allclose(m.track(reordered)[0],a,atol=1e-7)
    bad=copy.deepcopy(f);bad['skeleton']['timestamp_ms']-=26
    with pytest.raises(ValueError):m.track(bad)
    bad=copy.deepcopy(f);bad['skeleton']['names'][0]='missing'
    with pytest.raises(ValueError):m.track(bad)
    bad=copy.deepcopy(f);bad['skeleton']['orientations_local_xyzw'][2]=[0,0,0,0]
    with pytest.raises(ValueError):m.track(bad)

def test_official_training_keeps_pose_fixed(driven):
    from moxian_web.mano_fit import adam_articulation
    import threading
    m,f=driven;m.track(f);m.flat_reference={'flat_seed':m.last_pose.tolist()};m.mapping={'confirmed':[]};f['stage']='together';f['contact']={'ready':False,'patches':[]}
    q,root,base,centres=m.official_parameters([f]);original=q.clone()
    with patch.object(m,'constrain_pose_',side_effect=AssertionError('projection')):
        result=adam_articulation(m,[f],q,root,2,threading.Event())
    assert torch.equal(result,original)

def test_pinky_root_and_shape_are_not_stretched_to_fill_an_adjacent_gap(driven):
    m,f=driven;before=m.track(f)[0]
    ids=m.model.weights[:,list(CHAINS[4])].sum(1).cpu().numpy()>.99
    m.proximal_lateral_scales[:3]=np.log([1.05,.95,1.02]);m.set_shape(m.shape)
    after=m.track(f)[0]
    assert np.max(np.linalg.norm(after[ids]-before[ids],axis=1))<1e-6
    assert np.array_equal(m.official_fk.translations[CHAINS[4][0]],np.zeros(3))
    assert m.proximal_lateral_scales[3]==0

def test_rigid_wrist_motion_and_dimension_change_rebind_without_pose_solver(driven):
    m,f=driven;a=m.track(f);driver=m.official_fk;old_bind=driver.binding_hash;asset=driver.asset_hash
    turn=Rotation.from_euler('xyz',[13,-9,21],degrees=True).as_matrix();s=f['skeleton']
    s['positions_local']=(np.array(s['positions_local'])@turn.T).tolist();s['orientations_local_xyzw']=Rotation.from_matrix(turn@Rotation.from_quat(s['orientations_local_xyzw']).as_matrix()).as_quat().tolist();s['timestamp_ms']=f['timestamp_ms']=1020.
    b=m.track(f);local=driver.sdk_to_mano@turn@driver.mano_to_sdk
    assert np.allclose(b[0],a[0]@local.T,atol=1e-7)
    x=m.dimension_parameters.copy();x[15]=np.log(1.1);m.set_dimensions(x);m.track(f)
    assert driver.binding_hash!=old_bind and driver.asset_hash==asset
    assert m.alignment_quality['max_link_rotation_error_deg']<1e-4
