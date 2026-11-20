"""Temporal semantics of fixed-lag CARD, independent of camera/SDK hardware."""
import numpy as np
import pytest
from moxian_web.wuji_window import SourceFrameWindow,FUTURE,HORIZON
from retarget.baseline.card.trajectory import valid_window,FixedLagCardRetarget
from retarget.baseline.geort.runtime import Replay


def frame(i,**kw):
    return dict(timestamp_ms=1000+i*20,input_received_wall_ms=2000+i*20,
        input_stream_epoch=1,geometry_version='a',binding_version='b',**kw)


def test_received_window_retains_actual_adjacent_frames_despite_slow_solver():
    ring=SourceFrameWindow()
    for i in range(14):ring.append(frame(i))
    window=ring.snapshot()
    assert [f['timestamp_ms'] for f in window]==[1200,1220,1240,1260]
    assert valid_window(window)
    # The published pose is 3 actual input frames behind the newest glove.
    assert window[-1]['timestamp_ms']-window[0]['timestamp_ms']==60
    ring.append(dict(frame(14),pose_calibration_version=2))
    assert len(ring.snapshot())==1


@pytest.mark.parametrize('change',[{'timestamp_ms':500},{'timestamp_ms':2000},
    {'input_received_wall_ms':4000},{'input_stream_epoch':2},{'geometry_version':'new'}])
def test_discontinuity_discards_history(change):
    ring=SourceFrameWindow()
    for i in range(HORIZON):ring.append(frame(i))
    ring.append(dict(frame(HORIZON),**change))
    assert len(ring.snapshot())==1
    assert not valid_window(ring.snapshot())


def test_window_rejects_missing_futures_and_changed_calibration():
    frames=[frame(i) for i in range(HORIZON)]
    assert not valid_window(frames[:-1])
    frames[1]['binding_version']='different'
    assert not valid_window(frames)


def test_adjacent_smooth_cost_is_main_formula_without_new_objectives():
    import warp as wp
    from retarget.baseline.card.robokit import CardData
    from retarget.baseline.card.trajectory import adjacent_main_cost
    from robokit.utils.warp_utils import wp_vec7
    data=CardData();prior=CardData()
    data.active=wp.zeros(16,dtype=int,device='cpu');prior.active=wp.zeros(16,dtype=int,device='cpu')
    weights=np.ones(15,np.float32);weights[6:9]=[.01,125.,100.]
    data.weights=wp.array(weights,device='cpu')
    x=np.zeros((2,26),np.float32);before=x.copy()
    x[0,0]=.001;x[0,5]=.2;x[0,6:]=.1
    x[1,1]=.002;x[1,6:]=.3
    poses=wp.zeros((2,21),dtype=wp_vec7,device='cpu')
    parts=wp.zeros((2,12),device='cpu');cost=wp.zeros(2,device='cpu')
    wp.launch(adjacent_main_cost,2,[wp.array(x,device='cpu'),wp.array(before,device='cpu'),
        poses,poses,data,prior,parts,cost],device='cpu')
    expected=.01*np.sum((x[:,6:]-before[:,6:])**2,axis=1)
    expected+=125.*np.sum(((x[:,:3]-before[:,:3])*1000)**2,axis=1)
    expected+=100.*np.sum((x[:,3:6]-before[:,3:6])**2,axis=1)
    np.testing.assert_allclose(cost.numpy(),expected,rtol=1e-6,atol=1e-5)


def test_window_material_fix_matches_main_previous_material_formula():
    import warp as wp
    from retarget.baseline.card.robokit import CardData,card_cost
    from retarget.baseline.card.trajectory import adjacent_main_cost
    from robokit.utils.warp_utils import wp_vec7
    device='cpu'
    def array(value,dtype=float):return wp.array(np.asarray(value),dtype=dtype,device=device)
    mesh=wp.Mesh(array([[0,0,0],[.02,0,0],[0,.02,0]],wp.vec3),array([0,1,2],int))
    data=CardData();prior=CardData()
    for d in (data,prior):
        d.active=array([1]+[0]*15,int);d.links=array([0]*16,int)
        d.meshes=array([mesh.id]*16,wp.uint64);d.preference=array(np.zeros((16,3)),wp.vec3)
    data.target=array([[.024,.003,0]]+[[0,0,0]]*15,wp.vec3)
    prior.target=array([[.012,.003,0]]+[[0,0,0]]*15,wp.vec3)
    data.direction=array([[0,0,1]]*16,wp.vec3);data.rho=array([1]*16)
    data.apex=array([[0,1,0]]*16,wp.vec3);data.belly=array([[0,0,1]]*16,wp.vec3)
    data.human_bones=array([[0,1,0]]*15,wp.vec3);data.human_tips=array([[0,0,0]]*5,wp.vec3)
    data.human_normals=array([[0,0,1]]*5,wp.vec3);data.free=array([0]*5)
    data.pairs=wp.zeros(0,dtype=wp.vec2i,device=device);data.plane_count=array([0],int)
    data.object_mesh=array([0],wp.uint64);data.motion_valid=array([0],int)
    weights=np.zeros(15,np.float32);weights[2]=6.25;weights[6:9]=[.01,125.,100.]
    weights[10]=-100.;weights[13]=3.;data.weights=array(weights)
    old=np.zeros((16,3));old[0]=[.002,.003,0];data.old_local=array(old,wp.vec3)
    data.old_valid=array([1]+[0]*15,int);data.previous_valid=array([1],int)
    x=np.zeros((1,26),np.float32);x[0,0]=.02
    before=np.zeros_like(x);before[0,0]=.01;data.previous=array(before)
    poses=array([[[.02,0,0,1,0,0,0]]],wp_vec7)
    previous_poses=array([[[.01,0,0,1,0,0,0]]],wp_vec7)
    x=array(x);before=array(before)
    reference=wp.zeros(1,device=device);parts=wp.zeros((1,12),device=device)
    wp.launch(card_cost,1,[x,poses,data,parts,wp.zeros(1,device=device),
        wp.zeros((1,2),device=device),reference],device=device)
    candidate=wp.zeros(1,device=device);window_parts=wp.zeros((1,12),device=device)
    wp.launch(adjacent_main_cost,1,[x,before,poses,previous_poses,data,prior,window_parts,candidate],device=device)
    np.testing.assert_allclose(candidate.numpy(),reference.numpy(),rtol=1e-6,atol=.02)
    np.testing.assert_allclose(window_parts.numpy()[:,[2,6,7,8]],parts.numpy()[:,[2,6,7,8]],rtol=1e-6,atol=.02)

    # Exceed the old 6-mm capsule and 3-mm object gates independently and
    # together. Original collision costs must remain finite and positive.
    data.cap_a=array(np.zeros((16,3)),wp.vec3);data.cap_b=array(np.zeros((16,3)),wp.vec3)
    data.cap_radius=array([.005]*16)
    data.samples=array([[0,0,0]],wp.vec3);data.offsets=array([0]+[1]*16,int)
    data.planes=array([[1,0,0,-.01],[-1,0,0,-.01],[0,1,0,-.01],
        [0,-1,0,-.01],[0,0,1,-.01],[0,0,-1,-.01]],wp.vec4)
    data.wrist_inverse=array([[-.02,0,0,1,0,0,0]],wp_vec7)
    weights[9:14]=[2.25,6.,12.25,2.,3.];data.weights=array(weights)
    for capsule,obj in ((True,False),(False,True),(True,True)):
        data.pairs=array([(0,1)],wp.vec2i) if capsule else wp.zeros(0,dtype=wp.vec2i,device=device)
        data.plane_count=array([6 if obj else 0],int)
        out=wp.zeros(1,device=device);terms=wp.zeros((1,12),device=device);depths=wp.zeros((1,2),device=device)
        wp.launch(card_cost,1,[x,poses,data,terms,wp.zeros(1,device=device),depths,out],device=device)
        assert np.isfinite(out.numpy()).all()
        if capsule:assert depths.numpy()[0,1]<6. and terms.numpy()[0,9]>0
        if obj:assert depths.numpy()[0,0]>3. and terms.numpy()[0,10]>0
    invalid=np.zeros((1,26),np.float32);invalid[0,0]=np.nan
    out=wp.zeros(1,device=device)
    wp.launch(card_cost,1,[array(invalid),poses,data,wp.zeros((1,12),device=device),
        wp.zeros(1,device=device),wp.zeros((1,2),device=device),out],device=device)
    assert np.isinf(out.numpy()[0])


def test_replay_reads_three_true_future_frames_and_does_not_extend_tail():
    replay=Replay.__new__(Replay);replay.stamps=np.arange(12)*20.;replay.index=3
    replay._frame_at=lambda index,stamp:dict(index=index,timestamp_ms=stamp)
    future=replay.future_frames({'timestamp_ms':2000},FUTURE)
    assert [f['index'] for f in future]==[4,5,6]
    assert [f['timestamp_ms'] for f in future]==[2020,2040,2060]
    replay.index=10
    assert replay.future_frames({'timestamp_ms':2000},FUTURE)==[]


def test_window_adapter_never_pads_missing_observations():
    solver=FixedLagCardRetarget.__new__(FixedLagCardRetarget)
    resets=[];solver.reset=lambda:resets.append(True)
    result=solver.step(frame(0))
    assert result['status']=='buffering' and result['required_frames']==4
    assert not result['display_available'] and not result['solution_feasible']
    assert resets==[True]


def test_window_identity_includes_calibration_fingerprint():
    ring=SourceFrameWindow()
    for i in range(HORIZON):ring.append(frame(i,calibration_fingerprint='a'))
    ring.append(frame(HORIZON,calibration_fingerprint='b'))
    assert len(ring.snapshot())==1


@pytest.mark.parametrize('horizon',[1,4,6])
def test_published_joint_boundary_cost_is_not_diluted_by_future_frames(horizon):
    import warp as wp
    from retarget.baseline.card.trajectory import reduce_cost,BOUNDARY_JOINT_MULTIPLIER
    # Independent per-frame pose costs stay averaged. A boundary displacement
    # pays the same cost regardless of how many future frames are optimized.
    rows=np.full((2,horizon),10.,np.float32);parts=np.zeros((2,12),np.float32)
    parts[0,6]=100.;rows[0,0]+=100.
    out=wp.zeros(2,device='cpu')
    wp.launch(reduce_cost,2,[wp.array(rows,device='cpu'),wp.array(parts,device='cpu'),
        BOUNDARY_JOINT_MULTIPLIER,out],device='cpu')
    np.testing.assert_allclose(out.numpy(),[10.+100.*BOUNDARY_JOINT_MULTIPLIER,10.],rtol=1e-6)


def test_warm_start_uses_actual_overlap_without_freezing_unseen_future():
    from retarget.baseline.card.trajectory import aligned_warm_start
    old=np.array([100.,120.,140.,160.]);values=np.repeat(old[:,None],26,axis=1)
    fallback=np.full((4,26),999.,np.float32)
    aligned,mask=aligned_warm_start(np.array([130.,150.,170.,190.]),old,values,fallback)
    np.testing.assert_array_equal(mask,[True,True,False,False])
    np.testing.assert_allclose(aligned[:,0],[130.,150.,999.,999.])
    # A slow solver may skip past the whole previous lookahead window.
    # Those frames must initialize from new observations, not its old last pose.
    aligned,mask=aligned_warm_start(np.array([200.,220.,240.,260.]),old,values,fallback)
    assert not mask.any()
    np.testing.assert_array_equal(aligned,fallback)
    np.testing.assert_array_equal(fallback,999.)


def test_reset_discards_previous_window_predictions(monkeypatch):
    from types import SimpleNamespace
    from retarget.baseline.card.retarget import TactileRetarget
    monkeypatch.setattr(TactileRetarget,'reset',lambda self:None)
    solver=FixedLagCardRetarget.__new__(FixedLagCardRetarget)
    solver.engine=SimpleNamespace(warm_values=np.ones((4,26)),warm_stamps=np.arange(4))
    solver.window=[frame(0)];solver.live_frame=frame(3)
    solver.reset()
    assert solver.engine.warm_values is None and solver.engine.warm_stamps is None
    assert solver.window==[] and solver.live_frame is None


def test_stronger_live_joint_smoothing_keeps_offline_defaults(monkeypatch):
    from retarget.baseline.card.math import CardSettings
    from retarget.baseline.card.retarget import TactileRetarget
    monkeypatch.setattr(TactileRetarget,'__init__',lambda self,*a,settings,**kw:setattr(self,'settings',settings))
    solver=FixedLagCardRetarget()
    assert solver.settings.wtheta==30.
    assert CardSettings().wtheta==.1
    assert solver.settings.wtip==CardSettings().wtip
    assert solver.settings.wc==CardSettings().wc


def test_rtc_prefix_schedule_preserves_delayed_prefix_then_decays():
    from retarget.baseline.card.trajectory import chunk_prefix_weights
    stamps=np.arange(10)*20.;overlap=np.arange(10)<6
    weights=chunk_prefix_weights(stamps,overlap,40.)
    linear=np.array([1.,1.,4/5,3/5,2/5,1/5,0.,0.,0.,0.])
    expected=linear*np.expm1(linear)/np.expm1(1.)
    np.testing.assert_allclose(weights,expected,rtol=1e-6)
    assert np.all(np.diff(weights)<=0)
    np.testing.assert_array_equal(chunk_prefix_weights(stamps,np.zeros(10,bool),200.),0.)
    # Even latency exceeding the entire old trajectory cannot pin new frames.
    np.testing.assert_array_equal(chunk_prefix_weights(stamps,overlap,1000.),overlap.astype(float))


def test_fused_prefix_cost_only_guides_real_overlap_and_has_no_stale_mask():
    import warp as wp
    from retarget.baseline.card.trajectory import reduce_chunk_cost,reduce_cost
    device='cpu';b,h=3,4
    rows=wp.array(np.full((b,h),10.,np.float32),device=device)
    parts=wp.zeros((b,12),device=device)
    prefix=wp.zeros((h,20),device=device);weights=wp.array([1.,.5,0.,0.],dtype=float,device=device)
    x=np.zeros((b,h*26),np.float32)
    x[0,6]=1.;x[1,26+6]=1.;x[2,3*26+6]=1.
    x=wp.array(x,device=device);out=wp.zeros(b,device=device)
    wp.launch(reduce_chunk_cost,b,[rows,parts,2.,x,prefix,weights,100.,out],device=device)
    np.testing.assert_allclose(out.numpy(),[110.,60.,10.])
    # Reusing the same GPU buffers after reset/no overlap must remove guidance.
    weights.zero_();out.zero_();reference=wp.zeros(b,device=device)
    wp.launch(reduce_chunk_cost,b,[rows,parts,2.,x,prefix,weights,100.,out],device=device)
    wp.launch(reduce_cost,b,[rows,parts,2.,reference],device=device)
    np.testing.assert_array_equal(out.numpy(),reference.numpy())
