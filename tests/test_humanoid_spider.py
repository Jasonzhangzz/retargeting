"""Upstream norm parity, analytic derivatives and shared live contact/AdamW."""
from pathlib import Path
from types import SimpleNamespace
import mujoco
import numpy as np
import pytest
import torch
import warp as wp

from retarget.humanoid import HumanoidProblem, ProjectionADMM, Settings
from retarget.humanoid_mjwarp import MJWarpHumanoid
from retarget.humanoid_terms import Model, Targets, Stage, spider_pose_squared
from retarget.spider_pose import zero_safe_sqrt,quat_log,quat_exp,right_jacobian,inverse_right_jacobian
from scripts.compare_spider_humanoid import source_functions

DATA=Path(__file__).resolve().parents[1]/'data/spider/processed'


def task_paths(object_task):
    folder=DATA/('omomo/unitree_g1/humanoid_object/move_largebox' if object_task else 'amass/unitree_g1/humanoid/sprint')
    return folder/'scene.xml',folder/'0/trajectory_kinematic.npz'


def source_score(actual,reference,object_task):
    f=source_functions()
    c=SimpleNamespace(device='cpu',embodiment_type='humanoid_object' if object_task else 'humanoid',
        nv=41 if object_task else 35,nu=29,base_pos_rew_scale=1.,base_rot_rew_scale=1.,
        joint_rew_scale=1.,pos_rew_scale=1. if object_task else 3.,rot_rew_scale=.3 if object_task else 1.)
    actual,reference=np.broadcast_arrays(actual,reference)
    error=f['_diff_qpos'](c,torch.tensor(actual.copy(),dtype=torch.float64),torch.tensor(reference.copy(),dtype=torch.float64))
    return torch.linalg.vector_norm(error*f['_weight_diff_qpos'](c),dim=-1).numpy()


@wp.kernel
def pose_probe(qpos:wp.array2d(dtype=float),reference:wp.array2d(dtype=float),
               joints:wp.array(dtype=int),object_qpos:int,object_body:int,
               output:wp.array(dtype=float)):
    w=wp.tid()
    m=Model();m.object_body=object_body;m.object_qpos=object_qpos;m.joint_qpos=joints
    t=Targets();t.qpos=reference
    job=Stage();job.model=m;job.target=t;job.qpos=qpos;job.index=0
    output[w]=zero_safe_sqrt(spider_pose_squared(job,w))


@pytest.mark.parametrize('object_task',[False,True])
def test_exact_unsquared_upstream_norm_and_zero_gradient(object_task):
    scene,ref=task_paths(object_task);model=mujoco.MjModel.from_xml_path(str(scene))
    reference=np.load(ref)['qpos'][0].astype(np.float32)
    rng=np.random.default_rng(92)
    actual=np.tile(reference.astype(np.float64),(8,1))
    for i in range(1,8):
        velocity=rng.normal(size=model.nv)*.13
        if i==1:velocity[3:6]=[3.3,.2,-.1]
        mujoco.mj_integratePos(model,actual[i],velocity,1.)
    actual=actual.astype(np.float32)
    actual[6,3:7]*=-1
    if object_task:actual[7,-4:]*=-1
    joints=wp.array(np.arange(7,36),dtype=int,device='cpu')
    ref=wp.array(reference[None],device='cpu')
    qpos=wp.array(actual,device='cpu',requires_grad=True)
    out=wp.zeros(len(actual),device='cpu',requires_grad=True)
    with wp.Tape() as tape:wp.launch(pose_probe,len(actual),inputs=[qpos,ref,joints,36 if object_task else -1,0 if object_task else -1,out],device='cpu')
    tape.backward(grads={out:wp.ones(len(actual),device='cpu')})
    expected=source_score(actual,reference,object_task)
    np.testing.assert_allclose(out.numpy(),expected,atol=5e-7,rtol=3e-6)
    assert np.isfinite(qpos.grad.numpy()).all()
    np.testing.assert_allclose(qpos.grad.numpy()[0],0.,atol=1e-6)
    # Independent differences of the upstream implementation, on quaternion
    # tangent directions rather than changing quaternion normalization.
    direction=rng.normal(size=actual.shape).astype(np.float32)
    direction[0]=0.
    for adr in ([3,39] if object_task else [3]):
        direction[:,adr:adr+4]-=(direction[:,adr:adr+4]*actual[:,adr:adr+4]).sum(1)[:,None]*actual[:,adr:adr+4]
    eps=1.e-4
    fd=(source_score(actual+eps*direction,reference,object_task)-source_score(actual-eps*direction,reference,object_task))/(2*eps)
    np.testing.assert_allclose((qpos.grad.numpy()*direction).sum(1),fd,rtol=.01,atol=.003)


@pytest.fixture(scope='module')
def solved():
    if not torch.cuda.is_available():pytest.skip('MJWarp dynamics requires CUDA')
    scene,ref=task_paths(True)
    p=HumanoidProblem(scene,ref,settings=Settings(cost_mode='spider',horizon=2,knot_stride=1,
        multistarts=2,admm_iterations=2,action_iterations=2,contact_samples=4))
    solver=ProjectionADMM(p)
    state=p.initial_state(2);before=state.qpos.copy()
    rollout,history=solver.solve(state,[3,4],p.aref[2])
    np.testing.assert_array_equal(state.qpos,before)
    return p,solver,rollout,history


def test_spider_task_preserves_shared_contact_and_action_solvers(solved):
    from retarget.fa.adamw import AdamW
    from retarget.contact_lm_warp import DeviceLambdaLM
    from retarget.contact_solvers import SampledComFreeLM
    p,solver,rollout,history=solved
    assert type(solver.action_optimizer) is AdamW
    assert type(solver.spider_contacts.lm) is DeviceLambdaLM
    assert type(solver.spider_contacts.explorer) is SampledComFreeLM
    assert solver.backend.contact_qp is None
    assert p.settings.terminal_weight==1.
    assert solver.backend.stages[0].job.normalizer==0.
    assert all(x['contact']['task_cost'].startswith('SPIDER') for x in history)
    assert all(x['update_order']==['hands_and_feet_z','action','dual'] for x in history)
    assert history[-1]['backward_calls']==4
    assert history[-1]['selection_policy']=='spider_pose_without_ADMM'
    assert np.isfinite(rollout['qpos']).all()
    for row in history:
        assert row['action_cost_after']<=row['action_cost_before']+1.e-6
        assert all(abs(value)<1.e-7 for name,value in row['costs'].items()
                   if name not in ('spider_pose','consensus','anchor'))
        assert row['contact']['max_hand_shift_mm']<=5.+1.e-3
    np.testing.assert_array_equal(solver.contact_forces,0.)


def test_post_step_horizon_cost_matches_upstream_and_mjwarp_finite_differences(solved):
    p,solver,_,_=solved;b=solver.backend
    x=np.broadcast_to(solver.warm,b.variables.shape).copy()
    b.targets.penalty.zero_();b.dirty=True
    value,gradient=b.evaluate(x,True)
    states=np.stack([stage.job.qpos.numpy() for stage in b.stages[1:]])
    reference=b.targets.qpos.numpy()[1:]
    expected=np.array([source_score(q,r,True) for q,r in zip(states,reference)]).mean(axis=0)
    np.testing.assert_allclose(value,expected,rtol=3e-6,atol=5e-7)
    # Action derivatives now use coordinate secants through MJWarp. Their
    # independent coordinate check lives in test_humanoid_mjwarp; a native
    # smooth-branch adjoint/directional assertion does not apply at switches.
    assert np.isfinite(gradient).all()
    # Changing only contact targets must affect the augmented action gradient.
    b.targets.penalty.assign(np.array([p.settings.rho,p.settings.anchor_weight],np.float32));b.dirty=True
    _,coupled=b.evaluate(x,True)
    assert np.linalg.norm(coupled-gradient)>1.e-5


@wp.kernel(enable_backward=False)
def log_jacobian_probe(phi:wp.array(dtype=wp.vec3),goal:wp.array(dtype=wp.vec3),out:wp.array(dtype=wp.mat33)):
    i=wp.tid()
    error=quat_log(wp.quat_inverse(quat_exp(goal[i]))*quat_exp(phi[i]))
    out[i]=inverse_right_jacobian(error)*right_jacobian(phi[i])


def test_contact_lm_quaternion_residual_jacobian_with_noncommuting_rotations():
    from scipy.spatial.transform import Rotation
    phi=np.array([[.35,-.42,.18],[0.,0.,0.]],np.float32)
    goal=np.array([[-.2,.7,-.1],[.1,-.3,.2]],np.float32)
    out=wp.zeros(2,dtype=wp.mat33,device='cpu')
    wp.launch(log_jacobian_probe,2,inputs=[wp.array(phi,dtype=wp.vec3,device='cpu'),
        wp.array(goal,dtype=wp.vec3,device='cpu'),out],device='cpu')
    columns=[];epsilon=1.e-5
    for axis in np.eye(3):
        plus=(Rotation.from_rotvec(goal).inv()*Rotation.from_rotvec(phi+epsilon*axis)).as_rotvec()
        minus=(Rotation.from_rotvec(goal).inv()*Rotation.from_rotvec(phi-epsilon*axis)).as_rotvec()
        columns.append((plus-minus)/(2*epsilon))
    np.testing.assert_allclose(out.numpy(),np.stack(columns,axis=-1),rtol=2e-5,atol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='MJWarp feedback requires CUDA')
def test_actual_box_feedback_updates_contact_goal_and_action_gradient():
    from retarget.humanoid_spider_contacts import gather
    scene,ref=task_paths(True)
    p=HumanoidProblem(scene,ref,settings=Settings(cost_mode='spider',horizon=1,knot_stride=1,multistarts=1))
    solver=ProjectionADMM(p);b=solver.backend
    state=p.initial_state(2);original=state.qpos.copy()
    x=np.zeros(b.variables.shape)
    def sample():
        b.prepare(state,3,p.aref[2])
        b.targets.z.assign(b.anchors[:,None].astype(np.float32));b.dirty=True
        _,gradient=b.evaluate(x,True)
        wp.launch(gather,(1,2),inputs=[b.stages[0].job,solver.spider_contacts.batch,b.dense_steps],device=b.device)
        return solver.spider_contacts.batch.goal.numpy()[0,0].copy(),gradient.copy()
    first,gradient=sample()
    delta=np.array([.01,-.005,.002])
    rotation=state.xmat[b.model.object_body].reshape(3,3).copy()
    state.qpos[b.model.object_qpos:b.model.object_qpos+3]+=delta
    second,changed_gradient=sample()
    np.testing.assert_allclose(second[:3]-first[:3],-rotation.T@delta,atol=2.e-7)
    assert np.linalg.norm(changed_gradient-gradient)>1.e-5
    np.testing.assert_array_equal(state.qpos[:b.model.object_qpos],original[:b.model.object_qpos])
