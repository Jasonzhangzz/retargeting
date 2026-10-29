"""Independent checks of CUDA actuator-feasible load input construction."""
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch
from scipy.optimize import linprog

from retarget.grasp import build_scene
from retarget.grasp_primitives import primitive_mesh
from retarget.grasp_load import load_matrices,solve_load
from retarget.grasp_audit import force_closure_cuda

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.fixture
def model_and_cases(tmp_path):
    model,_,_,_=build_scene(primitive_mesh(tmp_path/'cylinder.obj','cylinder',.02,.08))
    model.opt.gravity[:]=[0.,0.,-9.81]
    cases=json.loads((Path(__file__).parent/'fixtures/grasp_load.json').read_text())['cases']
    return model,cases


def independent_constraints(model,case):
    q=np.array(case['qpos']);p=np.array(case['points']);n=-np.array(case['outward'])
    n/=np.linalg.norm(n,axis=-1,keepdims=True)
    t1=np.cross(n,np.eye(3)[np.abs(n).argmin(-1)]);t1/=np.linalg.norm(t1,axis=-1,keepdims=True)
    t2=np.cross(n,t1);a=np.arange(8)*2*np.pi/8
    rays=n[:,None]+.65*(t1[:,None]*np.cos(a)[None,:,None]+t2[:,None]*np.sin(a)[None,:,None])
    force=rays.reshape(-1,3)
    G=np.r_[force.T,np.cross(np.repeat(p,8,axis=0),force).T/.05]
    data=mujoco.MjData(model);data.qpos[:]=q;mujoco.mj_forward(model,data)
    jac=[]
    for name,local in zip(case['bodies'],case['local']):
        b=model.body(name).id
        point=data.xpos[b]+data.xmat[b].reshape(3,3)@local
        jp=np.zeros((3,model.nv));jr=np.zeros_like(jp)
        mujoco.mj_jac(model,data,jp,jr,point,b);jac.append(jp)
    B=np.einsum('cij,cki->jck',np.array(jac),rays).reshape(26,-1)
    bias=data.qfrc_bias-data.qfrc_passive
    kp=model.actuator_gainprm[:,0]
    lo=kp*(model.actuator_ctrlrange[:,0]-q);hi=kp*(model.actuator_ctrlrange[:,1]-q)
    lo=np.maximum(lo,np.where(model.actuator_forcelimited,model.actuator_forcerange[:,0],-np.inf))
    hi=np.minimum(hi,np.where(model.actuator_forcelimited,model.actuator_forcerange[:,1],np.inf))
    normal=np.kron(np.eye(len(p)),np.ones((1,8)))
    C=np.r_[B,-B,-normal,-np.ones((1,len(force)))];v=np.r_[hi-bias,-lo+bias,np.full(len(p),-.5),-6.]
    goal=np.array([0.,0.,1.962,0.,0.,0.])
    result=linprog(np.ones(len(force)),A_ub=C,b_ub=v,A_eq=G,b_eq=goal,bounds=(0,None),method='highs')
    return result,np.array(jac),bias,G,B,lo,hi


@pytest.mark.parametrize('name',['geometric_only','joint_reserve'])
def test_cuda_load_matches_independent_actuator_feasibility(model_and_cases,name):
    model,cases=model_and_cases;c=cases[name]
    expected,jac,bias,G,B,lo,hi=independent_constraints(model,c)
    m=load_matrices(model,np.array([model.body(b).id for b in c['bodies']]),np.array(c['local']),
        c['qpos'],c['points'],c['outward'],device='cuda:0',mass=.2,friction=.65,radius=.05)
    np.testing.assert_allclose(m['jacobian'].cpu(),jac,atol=1e-12)
    np.testing.assert_allclose(m['bias'].cpu(),bias,atol=1e-12)
    np.testing.assert_allclose(m['grasp'].cpu(),G,atol=1e-12)
    np.testing.assert_allclose(m['actuator'].cpu(),B,atol=1e-12)
    command,actual=solve_load(m,minimum_per_contact=.5,minimum_total=6.)
    assert actual['feasible']==expected.success
    assert actual['feasible']==(name=='joint_reserve')
    # Both inputs have geometric closure. Only the second can realize the
    # load with its bounded servos, which is exactly the missing distinction.
    closure=force_closure_cuda(m['q'].new_tensor(c['points']),m['q'].new_tensor(c['outward']),
                               friction=1.,radius=.05)
    assert closure['force_closure']
    if actual['feasible']:
        tau=(command-np.array(c['qpos']))*model.actuator_gainprm[:,0]
        assert np.all(tau>=lo-1e-5) and np.all(tau<=hi+1e-5)
        assert np.max(np.abs(actual['wrench_residual']))<2e-4
        assert min(actual['normal_force_N'])>=.5-2e-4
    else:
        assert actual['maximum_scaled_constraint_violation']>=2e-4


def test_load_warm_input_preserves_original_pose_contacts_and_state_reference(model_and_cases,tmp_path):
    from types import SimpleNamespace
    from scipy.spatial.transform import Rotation
    from retarget.dexonomy import ROOT
    from retarget.grasp import StaticHand
    from retarget.grasp_initialization import from_dexonomy_result
    from retarget.grasp_runtime import prepare_experiment,GraspRuntime
    model,cases=model_and_cases;c=cases['joint_reserve']
    bodies=np.array([model.body(b).id for b in c['bodies']])
    q=np.array(c['qpos']);points=np.array(c['points']);outward=np.array(c['outward'])
    matrices=load_matrices(model,bodies,c['local'],q,points,outward,device='cuda:0',mass=.2,friction=.65,radius=.05)
    command,info=solve_load(matrices,minimum_per_contact=.5,minimum_total=6.)
    assert info['feasible'];info['command']=command.tolist()
    raw=dict(obj_pose=np.array([0.,0.,0.,1.,0.,0.,0.]),
        grasp_pose=np.r_[q[:3],Rotation.from_euler('XYZ',q[3:6]).as_quat(scalar_first=True)],
        grasp_qpos=q[6:],obj_cp_w=points,obj_cn_w=-outward)
    guess=from_dexonomy_result(raw);guess.metadata['load_conditioning']=info
    # Local normals do not enter the warm-start packing itself.
    from retarget.contact_exploration import ObjectMeshSurface
    _,surface,_,_=build_scene(primitive_mesh(tmp_path/'input.obj','cylinder',.02,.08))
    hand=StaticHand(model,surface,bodies,np.array(c['local']),np.zeros_like(points))
    args=SimpleNamespace(solver_config=ROOT/'data/example/forceaware.yaml',object_mass=.2,
        friction=1.,load_conditioning=True,init_device='cuda:0',iterations=1,multistarts=1,
        seed=0,hand_solver='adamw',contact_radius_mm=5.,contact_motion_tolerance_mm=.1)
    prepared=prepare_experiment(tmp_path/'prepared',hand,guess,args)
    refs=np.load(prepared/'contactaware/wuji_right_qpos.npy')
    np.testing.assert_allclose(refs,np.tile(q,(len(refs),1)),atol=1e-7)
    with np.load(prepared/'contactaware/contact_guidance.npz') as guide:
        np.testing.assert_allclose(guide['contact_pos_obj'],np.tile(points,(len(refs),1,1)),atol=1e-8)
    runtime=GraspRuntime(prepared,args)
    np.testing.assert_allclose(runtime.qpos[:26],q,atol=1e-7)
    np.testing.assert_allclose(runtime.initial_action,command)
    np.testing.assert_allclose(runtime.initial_warm[26:].reshape(-1,26),
                               np.tile(command-q,(runtime.cfg.optimizer.horizon,1)),atol=1e-7)
    assert np.linalg.norm(runtime.initial_action-q)>.01
    from retarget.fa.solver import optimizer_class,settings_from_args
    direct=optimizer_class(settings_from_args(args))(runtime.model,runtime.device_model,runtime.index,
        guidance_count=len(points),cfg=runtime.cfg)
    expected=direct.solve(runtime.qpos,runtime.qvel,runtime.targets,warm=runtime.initial_warm,
        previous_action=command,allow_init=False,seed=0,execution_steps=1,start_time=0.)
    actual,_,_=runtime.solve()
    np.testing.assert_allclose(actual.raw,expected.raw,atol=2e-6,rtol=2e-6)
    np.testing.assert_allclose(actual.control_sequence,expected.control_sequence,atol=2e-6,rtol=2e-6)
    for key,value in raw.items():
        np.testing.assert_array_equal(guess.raw[key],value)
    args.load_conditioning=False
    unconditioned=GraspRuntime(prepared,args)
    assert unconditioned.load_input is None
    np.testing.assert_array_equal(unconditioned.initial_warm,0.)
    np.testing.assert_allclose(unconditioned.initial_action,q,atol=1e-7)


def test_force_cone_witness_excludes_same_side_contacts_without_actuator_assumptions():
    from retarget.grasp_load import force_cone_diagnostic
    # Surface normals spread over only 80 degrees: even continuous mu=.65
    # cones cannot cancel the horizontal resultant under positive preload.
    a=torch.deg2rad(torch.tensor([-40.,0.,40.],device='cuda',dtype=torch.float64))
    outward=torch.stack((a.cos(),a.sin(),torch.zeros_like(a)),dim=-1)
    gravity=outward.new_tensor([0.,0.,1.962])
    witness=force_cone_diagnostic(outward,.65,gravity,6.)
    assert witness['force_closure_excluded'] and witness['static_load_excluded']
    assert witness['minimum_force_projection_per_normal_N']>.3
    # Independent continuous-cone samples obey the certified lower bound.
    inward=-outward;axis=outward.new_tensor(witness['axis'])
    tangent=torch.linalg.cross(inward,outward.new_tensor([0.,0.,1.]).expand_as(inward))
    z=torch.linalg.cross(inward,tangent)
    theta=torch.arange(1000,device='cuda',dtype=torch.float64)*2*torch.pi/1000
    rays=inward[:,None]+.65*(tangent[:,None]*theta.cos()[None,:,None]+z[:,None]*theta.sin()[None,:,None])
    assert float((rays@axis).min())>=witness['minimum_force_projection_per_normal_N']-1e-12
    # Opposed normals have no such mean-direction witness. This is not a
    # positive force-closure certificate (two points are still insufficient).
    opposite=outward.new_tensor([[1.,0.,0.],[-1.,0.,0.]])
    assert not force_cone_diagnostic(opposite,.65,gravity,6.)['force_closure_excluded']


def test_batched_preload_matches_independent_constraints_and_single_candidate(model_and_cases):
    from retarget.grasp_load import load_matrices_batch, solve_load_batch
    model, cases = model_and_cases
    c = cases['joint_reserve']
    q = np.stack([c['qpos'], c['qpos']])
    q[1, 6] += .03
    points, outward = np.stack([c['points']]*2), np.stack([c['outward']]*2)
    bodies = np.array([model.body(name).id for name in c['bodies']])
    batch = load_matrices_batch(model, bodies, c['local'], q, points, outward,
                               device='cuda:0', mass=.2, friction=.65, radius=.05)
    commands, reports = solve_load_batch(batch, minimum_per_contact=.5, minimum_total=6.)
    for i in range(2):
        single = load_matrices(model, bodies, c['local'], q[i], points[i], outward[i],
                              device='cuda:0', mass=.2, friction=.65, radius=.05)
        for field in ('jacobian', 'bias', 'grasp', 'actuator', 'low', 'high'):
            torch.testing.assert_close(batch[field][i], single[field], atol=1.e-10, rtol=1.e-10)
        command, report = solve_load(single, minimum_per_contact=.5, minimum_total=6.)
        reference, *_ = independent_constraints(model, dict(c, qpos=q[i].tolist()))
        assert reports[i]['feasible'] == report['feasible'] == reference.success
        np.testing.assert_allclose(commands[i], command, atol=2.e-7, rtol=2.e-7)
        assert reports[i]['batch_size'] == 2
