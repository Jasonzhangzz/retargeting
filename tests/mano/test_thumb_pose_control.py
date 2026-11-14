"""Thumb binding/frame offsets must never become clipped physical twist."""
import copy
import xml.etree.ElementTree as ET
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation
from moxian_web.emf_slsqp import SkeletonKinematics
from moxian_web.skeleton import JOINT_NAMES
from moxian_web.hand_model import DEFAULT_MANO
from moxian_web.mano_runtime import ManoSurface
from moxian_web.mano_objective import ManoObjective,solve_pose
from moxian_web.thumb_chart import SLOTS
from scripts.calibrate_wuji_mano import CHAINS


@pytest.fixture(params=['left','right'])
def driven(request):
    side=request.param;path=DEFAULT_MANO.with_name('MANO_'+side.upper()+'.pkl')
    if not path.exists():pytest.skip('Licensed MANO required')
    torch.set_num_threads(1);m=ManoSurface(path,device='cpu',side=side);m.enable_official_fk()
    asset=m.official_fk.path.with_name(side+'.urdf')
    names=[j.get('name') for j in ET.parse(asset).getroot().findall('joint') if j.get('type')!='fixed']
    kin=SkeletonKinematics(asset,side,names,JOINT_NAMES)
    def frame(q,stamp=1000):
        nodes=kin.nodes(q)
        return dict(timestamp_ms=stamp,hand_side=side,
                    fingers={f:dict(position=[0,0,0],quaternion_xyzw=[0,0,0,1]) for f in ('thumb','index','middle','ring','pinky')},
                    skeleton=dict(timestamp_ms=stamp,hand_side=side,coordinate_frame='wrist_local',names=JOINT_NAMES,
                      joint_names=names,joint_positions_rad=q.tolist(),thumb_kinematics=kin.thumb_model,
                      positions_local=nodes[:,:3,3].tolist(),orientations_local_xyzw=Rotation.from_matrix(nodes[:,:3,:3]).as_quat().tolist()))
    return m,kin,frame


def test_source_thumb_seed_is_exact_for_all_six_joint_axes_and_has_pose_gradients(driven):
    m,kin,frame=driven
    for n,index in enumerate(kin.finger_indices[0]):
        q=kin.straight.copy();q[index]=np.clip(q[index]+.3,kin.bounds[index,0],kin.bounds[index,1]);row=frame(q)
        pose,root,*_=m.official_fk.parameters(m,row,CHAINS)
        o=ManoObjective(m,[row],pose[None],root[None],tactile=False)
        x=o.initial.clone().requires_grad_();out,v,_,_,_=o.surface(x)
        np.testing.assert_allclose(Rotation.from_rotvec(out[0,12:].detach().numpy()).as_matrix(),Rotation.from_rotvec(pose[12:]).as_matrix(),atol=1e-6)
        scalar=v[0,744].sum();grad=torch.autograd.grad(scalar,x)[0]
        assert torch.isfinite(grad).all() and grad.abs().max()>1e-4
        fd=[]
        for i in range(len(x)):
            d=torch.zeros_like(x);d[i]=.001
            fd.append((o.surface(x+d)[1][0,744].sum()-o.surface(x-d)[1][0,744].sum()).detach()/.002)
        torch.testing.assert_close(grad,torch.stack(fd),rtol=.03,atol=2e-5)


def test_binding_pad_normal_uses_each_sides_actual_tip_frame(driven):
    m,kin,frame=driven;m.official_fk.bind(m,CHAINS)
    ref=m.official_fk.reference;r=Rotation.from_quat(ref['orientations_local_xyzw']).as_matrix();ids={n:i for i,n in enumerate(ref['names'])}
    pad=r[ids['thumb_distal']].T@r[ids['thumb_tip']]@np.array([0,-1,0.])
    for j in CHAINS[0]:
        local=m.official_fk.corrections[j]@m.anatomy.palmar_normal
        # Compare transverse normals, since basis() removes the bone component.
        i=m.official_fk.indices[j];expected=pad.copy()
        end=ref['positions_local'][ids['thumb_tip']] if j==15 else ref['positions_local'][m.official_fk.indices[j+1]]
        axis=r[i].T@(np.asarray(end)-ref['positions_local'][i]);axis/=np.linalg.norm(axis)
        a=local-axis*(local@axis);b=expected-axis*(expected@axis)
        assert a@b/(np.linalg.norm(a)*np.linalg.norm(b))>1-1e-6


def test_runtime_correction_is_bounded_around_this_frames_command(driven):
    m,kin,frame=driven;q=kin.straight.copy()
    q[kin.finger_indices[0]]=kin.bounds[kin.finger_indices[0],0]+.65*np.diff(kin.bounds[kin.finger_indices[0]],axis=1).ravel()
    row=frame(q);pose,root,*_=m.official_fk.parameters(m,row,CHAINS)
    result,report,_=solve_pose(m,row,pose,root,tactile=False,iterations=2,budget_ms=300)
    actual=m._mano_solver_previous_angles
    expected=np.array([q[i] for i in kin.finger_indices[0]])
    np.testing.assert_array_less(np.abs(np.array([actual[j,k] for j,k in SLOTS])-expected),np.full(6,np.deg2rad(8)+1e-6))
    assert report['anatomical_chart']=='source_thumb_urdf_axes_v1' and not report['collision_escape_seed']
    assert np.isfinite(result).all()


def test_continuity_limits_only_correction_and_does_not_delay_source_motion(driven):
    m,kin,frame=driven;q=kin.straight.copy();row=frame(q)
    pose,root,*_=m.official_fk.parameters(m,row,CHAINS)
    solve_pose(m,row,pose,root,tactile=False,iterations=1,budget_ms=200)
    previous=(m._mano_solver_previous_angles-m._mano_solver_previous_seed_angles).copy();m.tracking_time=1000.
    j,k=SLOTS[0];prior_command=float(m._mano_solver_previous_angles[j,k])
    q[kin.finger_indices[0][0]]+=.2;row=frame(q,1033.)
    pose,root,*_=m.official_fk.parameters(m,row,CHAINS)
    _,report,_=solve_pose(m,row,pose,root,tactile=False,iterations=2,budget_ms=200)
    current=m._mano_solver_previous_angles-m._mano_solver_previous_seed_angles
    assert np.max(np.abs(current-previous))<=np.deg2rad(25)*.033+1e-5
    # The source moved 0.2 rad; no whole-pose low-pass or stale-pose reuse.
    assert abs(float(m._mano_solver_previous_angles[j,k])-prior_command-.2)<=np.deg2rad(25)*.033+1e-5
    assert 'correction_continuity' in report['cost_terms']


def test_fitted_shape_replay_keeps_the_same_pose_chart_and_official_surface(driven):
    m,kin,frame=driven;row=frame(kin.straight.copy());row['stage']='all_fingers'
    pose,root,*_=m.official_fk.parameters(m,row,CHAINS)
    o=ManoObjective(m,[row],pose[None],root[None],training=True,shared=True,tactile=False)
    x=o.initial.clone();x[0]=.7;x[3]=-.4
    before=o.surface(x)
    o.install(x);new_pose,new_root,*_=m.official_fk.parameters(m,row,CHAINS)
    np.testing.assert_allclose(new_pose,pose,atol=1e-6);np.testing.assert_allclose(new_root,root,atol=1e-6)
    replay=ManoObjective(m,[row],new_pose[None],new_root[None],tactile=False)
    after=replay.surface(replay.initial)
    for a,b in zip(before[:4],after[:4]):torch.testing.assert_close(a,b,atol=2e-6,rtol=1e-5)
