"""Right-hand capture uses the real external SDK, without hardware I/O."""
import copy
import time
from pathlib import Path
import pytest
from moxian_web.skeleton import SkeletonBridge,validate_pose,JOINT_NAMES
from moxian_web.hand_side import device_side
from moxian_web.hand_model import aligned_frame,HandInputSynchronizer

ROOT=Path(__file__).resolve().parents[1]


def test_side_resolution_preserves_confirmed_left_usb_exception():
    assert device_side({'serial':'2DF5F3B00DA5','usb_side':'right'})=='left'
    assert device_side({'serial':'2DDDF3AC0DA5','usb_side':'right'})=='right'


def test_right_pose_validation_and_sync_do_not_accept_cross_hand_frames():
    pose={'hand_side':'right','coordinate_frame':'wrist_local','timestamp_ms':1000.,'names':JOINT_NAMES.copy(),
          'positions_local':[[0.,0.,0.]]*25,'orientations_local_xyzw':[[0.,0.,0.,1.]]*25}
    assert validate_pose(pose,'right')==pose
    with pytest.raises(ValueError):validate_pose(pose,'left')
    frame={'hand_side':'right','timestamp_ms':1000.,'tactile_timestamp_ms':1000.,'raw':[0]*460,
           'fingers':{f:{'position':[.03,.02,.01],'quaternion_xyzw':[0.,0.,0.,1.]} for f in ('thumb','index','middle','ring','pinky')}}
    assert aligned_frame(frame)
    sync=HandInputSynchronizer();sync.push(frame)
    assert sync.match({**pose,'hand_side':'left'},1005.) is None
    assert sync.match(pose,1005.)['hand_side']=='right'


@pytest.mark.skipif(not (ROOT/'.venv-moxian-ik/bin/python').exists(),reason='Official SDK required')
def test_right_capture_counts_thirty_frames_and_advances_to_open(tmp_path):
    bridge=SkeletonBridge(ROOT,tmp_path)
    try:
        bridge.start('right-regression-no-hardware',side='right')
        deadline=time.monotonic()+12
        while not bridge.snapshot().get('capture') and time.monotonic()<deadline:time.sleep(.03)
        state=bridge.snapshot();assert state['capture']['side']=='right',state
        assert state['capture']['source_id']=='glove.moxian.right'
        stage=state['capture']['stage'];bridge.control('confirm',{'request_id':stage['request_id']})
        for i in range(90):
            bridge.send({'type':'frame','hand_side':'right','timestamp_ms':1000.+i*30,
                         'fingers':{f:{'position':[.03,.02,.01],'quaternion_xyzw':[0.,0.,0.,1.]} for f in ('thumb','index','middle','ring','pinky')}})
            time.sleep(.025)
            if bridge.snapshot()['capture']['stage']['stage_id']=='open':break
        state=bridge.snapshot()
        assert state['capture']['stage']['stage_id']=='open',state
        assert state['capture']['stage']['state']=='awaiting_confirmation'
        assert state['input_frames']>0
    finally:bridge.stop()


def test_right_native_engine_and_official_mano_use_right_assets(tmp_path):
    import numpy as np
    import torch
    from moxian_web.official_fk import official_urdf
    from moxian_web.kinematic_policy import build_kinematic_config
    from moxian_web.native_skeleton import NativeSkeleton
    from moxian_web.mano_runtime import ManoSurface
    from moxian_web.hand_model import DEFAULT_MANO
    paths=list((ROOT/'.venv-moxian-ik/lib').glob('python*/site-packages/qnbot_sdk/glove/_private/algorithms/skeleton/*/lib/libqnbot_hand.so'))
    if len(paths)!=1 or not DEFAULT_MANO.with_name('MANO_RIGHT.pkl').exists():pytest.skip('Official runtime and right MANO required')
    package=paths[0].parent.parent
    original=package/'share/qnbot_hand/hands/dynamic_openxr_hand/dynamic_openxr_hand_system.yaml'
    config,policy=build_kinematic_config(original.read_text(),package,tmp_path,side='right')
    assert 'hands/dynamic_openxr_hand/resources/pinocchio/right.urdf' in policy['asset_sha256']
    engine=NativeSkeleton(package,config,side='right')
    frame={'hand_side':'right','timestamp_ms':1000.,'fingers':{f:{'position':[.03,.02,.01],'quaternion_xyzw':[0.,0.,0.,1.]} for f in ('thumb','index','middle','ring','pinky')}}
    try:
        output=engine.step(frame)
        validate_pose(output['pose'],'right')
        assert all(n.startswith('right_') for n in output['pose']['joint_names'])
        assert output['targets']['hand_side']=='right' and output['physics']['enabled'] is False
        with pytest.raises(ValueError):engine.step({**frame,'hand_side':'left'})
    finally:engine.close()
    m=ManoSurface(DEFAULT_MANO.with_name('MANO_RIGHT.pkl'),device='cpu',side='right');m.enable_official_fk()
    # This checks asset/FK parity, independently of the bounded pose solver.
    m.solver_enabled=False
    frame['skeleton']=output['pose']
    v,j,r,_,_=m.track(frame)
    assert np.isfinite(v).all() and j.shape==(16,3) and m.model.backend.is_rhand
    assert m.official_fk.reference['hand_side']=='right'
    assert m.alignment_quality['max_link_rotation_error_deg']<1e-3
    q=torch.tensor(m.last_pose[None]);beta=torch.tensor(m.shape[:10],dtype=torch.float32)[None]
    with torch.no_grad():expected=m.model.backend(betas=beta,hand_pose=q.reshape(1,45),global_orient=torch.zeros(1,3))
    expected=(expected.vertices[0].numpy()-m.rest_j_np[0])@m.last_root.T
    np.testing.assert_allclose(v,expected,atol=1e-6)
