"""Real source-IK bad-branch recovery, scoring and per-frame continuity."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from moxian_web.studio_emf_solver import StudioQPSkeleton
from moxian_web.bone_extrinsics import FINGERS, model_hash


@pytest.fixture(scope='module')
def data():
    return json.loads((Path(__file__).parent/'fixtures/moxian_source_ik_branch.json').read_text())


@pytest.fixture
def engine(data):
    e=StudioQPSkeleton(copy.deepcopy(data['candidate']))
    yield e
    e.close()


def observation(data,stage='pinch_index',stamp=1000.):
    return {**copy.deepcopy(data['poses'][stage]), 'timestamp_ms':stamp}


def targets(engine,frame):
    a=engine.kinematics.alignment
    p=np.array([frame['fingers'][f]['position'] for f in FINGERS])@a[:3,:3].T+a[:3,3]
    r=a[:3,:3]@Rotation.from_quat([frame['fingers'][f]['quaternion_xyzw'] for f in FINGERS]).as_matrix()
    return p,r


def independent_cost(engine,q,frame,reference,history):
    """Evaluate the documented residual, posture and temporal equations."""
    p,r=targets(engine,frame);c=engine.config;k=engine.kinematics
    rest=np.array(engine.definition['rest_angles_rad']);cost=[]
    for f,ids in enumerate(k.finger_indices):
        rr,pp=k.sensor(f,q)
        pos=np.linalg.norm(pp-p[f])/c['position_sigma_m']
        rot=Rotation.from_matrix(rr.T@r[f]).magnitude()/c['rotation_sigma_rad']
        value=pos**2+rot**2
        for j in ids:
            name=k.joint_names[j];posture=c['posture_weight']*(c['thumb_posture_scale'] if f==0 else 1.)
            if name.endswith('_mcp_twist'):posture*=c.get('root_twist_posture_scale',.01)
            palm=name.endswith(('_thumb_cmc_flexion','_thumb_mcp_abduction'))
            if palm:posture=c.get('thumb_palm_posture_weight',posture)
            value+=posture*(q[j]-rest[j])**2
            if history:value+=(c.get('thumb_palm_history_weight',c['history_weight']) if palm else c['history_weight'])*(q[j]-reference[j])**2
        cost.append(value)
    return np.array(cost)


@pytest.mark.parametrize('stage',['pinch_index','pinch_middle','pinch_ring','pinch_pinky'])
def test_cold_recovery_improves_same_full_cost_without_weakening_limits(data,engine,stage):
    frame=observation(data,stage);p,r=targets(engine,frame);start=np.array(engine.definition['rest_angles_rad'])
    base,base_report=engine.native.solve(start,p,r,1/60,False)
    result=engine.step(frame);q=engine.previous
    current=independent_cost(engine,q,frame,start,False)
    np.testing.assert_array_less(current,base_report[:,3]+1e-7)
    assert np.all(q>=engine.kinematics.limits[:,0]-1e-12) and np.all(q<=engine.kinematics.limits[:,1]+1e-12)
    if stage=='pinch_index':
        assert base_report[1,0]*1000>9.
        assert result['pose']['tracking_errors']['index']['position_error_mm']<.5
        assert result['pose']['tracking_errors']['index']['rotation_error_deg']<4.
        assert result['ik_diagnostics']['branch_recovery']['selected_fingers']>=1
        assert current[1]<4.
    if stage=='pinch_ring':
        # Already good: no extra solves or altered pose.
        assert result['ik_diagnostics']['branch_recovery']['probe_calls']==0
        np.testing.assert_array_equal(q,base)


def test_cached_bad_warm_branch_recovers_without_spending_speed_budget_twice(data,engine):
    engine.previous=np.array(data['stuck_q']);engine.previous_timestamp=1000.;engine.frames=1
    transitioned=False;first=None
    for n in range(1,31):
        previous=engine.previous.copy();frame=observation(data,stamp=1000+n*1000/60)
        result=engine.step(frame);q=engine.previous
        assert np.max(np.abs(q-previous))<=engine.config['v_max']/60+1e-12
        assert np.all(q>=engine.kinematics.limits[:,0]-1e-12) and np.all(q<=engine.kinematics.limits[:,1]+1e-12)
        cost=independent_cost(engine,q,frame,previous,True)
        np.testing.assert_allclose([f['cost'] for f in result['ik_diagnostics']['fingers']],cost,rtol=1e-7,atol=1e-6)
        p,r=targets(engine,frame)
        score=engine.native.score(q,p,r,previous,True)
        np.testing.assert_allclose([f['cost'] for f in result['ik_diagnostics']['fingers']],score[:,3],atol=1e-10)
        recovery=result['ik_diagnostics']['branch_recovery'];transitioned|=recovery['transitioning']
        if first is None:
            first=result
            assert recovery['transitioning'] and recovery['fingers'][1]['step_fraction']<1.
            assert recovery['fingers'][1]['destination_cost']<recovery['fingers'][1]['local_cost']
        # Published history must be the current endpoint, never the distant probe.
        np.testing.assert_array_equal(result['pose']['joint_positions_rad'],q)
    assert transitioned
    assert result['pose']['tracking_errors']['index']['position_error_mm']<.5
    assert result['pose']['tracking_errors']['index']['rotation_error_deg']<4.
    assert result['ik_diagnostics']['branch_recovery']['fingers'][1]['transitioning'] is False


def test_probes_are_bounded_and_recovery_resets_after_gap_or_explicit_reset(data,engine):
    counts=[]
    for n in range(18):
        result=engine.step(observation(data,'pinch_pinky',1000+n*1000/60))
        counts.append(result['ik_diagnostics']['branch_recovery']['probe_calls'])
    assert all(c in (0,3) for c in counts)
    selected=[i for i,c in enumerate(counts) if c]
    assert all(b-a>=8 for a,b in zip(selected,selected[1:]))
    assert engine.recovery.targets is not None
    engine.reset()
    assert engine.recovery.targets is None and not engine.recovery.valid.any()
    assert engine.recovery.frame_index==0
    first=engine.step(observation(data))
    after_gap=engine.step(observation(data,stamp=2000.))
    np.testing.assert_array_equal(first['pose']['joint_positions_rad'],after_gap['pose']['joint_positions_rad'])
    assert after_gap['ik_diagnostics']['branch_recovery']['frame_index']==1


def test_invalid_packet_cannot_clear_or_advance_recovery_history(data,engine):
    engine.step(observation(data))
    before=(engine.previous.copy(),engine.previous_timestamp,engine.frames,engine.recovery.targets.copy(),engine.recovery.frame_index)
    frame=observation(data,stamp=10000.);frame['fingers']['ring']['position'][0]=float('nan')
    with pytest.raises(ValueError):engine.step(frame)
    np.testing.assert_array_equal(engine.previous,before[0]);assert engine.previous_timestamp==before[1] and engine.frames==before[2]
    np.testing.assert_array_equal(engine.recovery.targets,before[3]);assert engine.recovery.frame_index==before[4]


def test_fast_but_descending_motion_does_not_trigger_a_branch_change(data,engine):
    engine.step(observation(data));previous=engine.previous.copy();target=previous.copy()
    j=engine.kinematics.joint_names.index('right_index_mcp_flexion')
    target[j]+=.3
    a=engine.kinematics.alignment;nodes={}
    for f,name in enumerate(FINGERS):
        r,p=engine.kinematics.sensor(f,target)
        nodes[name]=dict(position=((p-a[:3,3])@a[:3,:3]).tolist(),
                         quaternion_xyzw=Rotation.from_matrix(a[:3,:3].T@r).as_quat().tolist())
    frame=dict(timestamp_ms=1000+1000/60,fingers=nodes)
    p,r=targets(engine,frame)
    base,raw=engine.native.solve(previous,p,r,1/60,True)
    assert raw[1,0]>engine.config['retry_position_m'] or raw[1,1]>engine.config['retry_rotation_rad']
    result=engine.step(frame)
    np.testing.assert_allclose(engine.previous,base,rtol=0,atol=1e-14)
    assert not engine.recovery.transitioning.any()
    assert result['ik_diagnostics']['branch_recovery']['probe_calls']==0


def test_native_probe_scoring_and_motion_bounds_share_actual_previous_reference(data,engine):
    frame=observation(data);p,r=targets(engine,frame)
    previous=np.array(data['stuck_q']);seed=engine.kinematics.limits.mean(1)
    for history in (False,True):
        q,raw=engine.native.solve(seed,p,r,1/60,history,reference=previous,probe=True)
        score=engine.native.score(q,p,r,previous,history)
        np.testing.assert_allclose(score[:,3],raw[:,3],atol=1e-9)
        np.testing.assert_allclose(score[:,3],independent_cost(engine,q,frame,previous,history),rtol=1e-8,atol=1e-7)
    # The probe is allowed to explore a different branch, but normal refinement
    # still uses the box around previous, even with a completely different seed.
    q,_=engine.native.solve(seed,p,r,1/60,True,reference=previous)
    assert np.max(abs(q-previous))<=engine.config['v_max']/60+1e-12


def test_calibration_replay_uses_the_same_recovered_source_ik(data):
    import threading
    from moxian_web.bone_calibration import replay
    result=replay(None,data['candidate'],[observation(data)],threading.Event())[0]
    assert result['pose']['tracking_errors']['index']['position_error_mm']<.5
    assert result['ik_diagnostics']['branch_recovery']['selected_fingers']>=1


@pytest.mark.parametrize('side',['left','right'])
@pytest.mark.parametrize('twist',[False,True])
def test_supported_native_topologies_keep_finite_poses_and_exact_cold_reset(side,twist):
    import torch
    from moxian_web.qnbot_surface import QnbotSurface
    from moxian_web.bone_extrinsics import definition_from_surface
    torch.set_num_threads(1);m=QnbotSurface(side,'cpu',finger_twist=twist)
    c=m.artifact();c.update(side=side,bone_extrinsics=definition_from_surface(m));c['model_hash']=model_hash(c)
    e=StudioQPSkeleton(c)
    try:
        k=e.kinematics;q=np.array(e.definition['rest_angles_rad']);a=k.alignment
        p=[];r=[]
        for f in range(5):
            rr,pp=k.sensor(f,q);p.append((pp-a[:3,3])@a[:3,:3]);r.append(a[:3,:3].T@rr)
        frame=dict(timestamp_ms=1000.,fingers={name:dict(position=pp.tolist(),quaternion_xyzw=Rotation.from_matrix(rr).as_quat().tolist()) for name,pp,rr in zip(FINGERS,p,r)})
        first=e.step(frame)
        assert np.isfinite(first['pose']['joint_positions_rad']).all()
        assert first['ik_diagnostics']['branch_recovery']['probe_calls']==0
        e.reset();second=e.step(frame)
        np.testing.assert_array_equal(first['pose']['joint_positions_rad'],second['pose']['joint_positions_rad'])
    finally:e.close()
