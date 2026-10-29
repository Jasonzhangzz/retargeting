"""CUDA integration: common-solver parity, mesh audit and closure certificates."""
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest
import torch

from retarget.grasp import StaticHand, build_scene
from retarget.grasp_primitives import primitive_mesh
from retarget.grasp_initialization import from_dexonomy_result
from retarget.grasp_runtime import prepare_experiment, GraspRuntime
from retarget.grasp_audit import GraspAudit, force_closure_cuda
from retarget.grasp_closure import force_closure
from retarget.dexonomy import ROOT

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_cuda_closure_matches_independent_highs_and_rejects_infeasible_preload():
    p=np.array([[-.05,0,0],[.05,0,0],[-.05,.001,0.]])
    n=np.array([[-1.,0,0],[1.,0,0],[-1.,0,0]])
    variants=[(p,n),(p[:2],n[:2]),(p,np.tile([0.,0.,-1.],(3,1))),
              (np.array([[-.05,0,0],[.05,0,0],[-.05,0,0.]]),n)]
    for points,normals in variants:
        actual=force_closure_cuda(torch.tensor(points,device='cuda'),torch.tensor(normals,device='cuda'),radius=.1)
        expected=force_closure(points,normals,radius=.1)
        assert actual['force_closure']==expected['force_closure']
        assert actual['rank']==expected['rank']
        if actual['force_closure']:
            assert actual['converged'] and actual['equilibrium_residual']<1e-8
            np.testing.assert_allclose(actual['positive_margin'],expected['positive_margin'],atol=2e-8)


def test_cuda_mesh_signed_distance_and_projection_match_independent_geometry(tmp_path):
    path=primitive_mesh(tmp_path/'box.obj','cuboid',.02,.08)
    _,surface,_,_=build_scene(path)
    audit=GraspAudit(surface,'cuda:0')
    p=torch.tensor([[.031,0,0],[0,.027,0],[0,0,.06],[.005,0,0],[-.06,.04,.03999]],device='cuda')
    projected,normals,distance=audit.query(p)
    vertices=surface.triangles.reshape(-1,3)
    lower,upper=vertices.min(0),vertices.max(0)
    np.testing.assert_allclose(projected[:3].cpu(),np.clip(p[:3].cpu().numpy(),lower,upper),atol=2e-7)
    assert bool((distance[:3]>0).all()) and float(distance[3])<0
    # Outside near a top edge while below its face plane: a face-dot sign
    # would falsely report a large penetration for this distant point.
    assert float(distance[4])>0
    np.testing.assert_allclose(normals[:3].cpu(),np.eye(3),atol=1e-6)


def test_sharp_edge_clearance_identifies_cylinder_rim_without_moving_points(tmp_path):
    from retarget.grasp_initialization import sharp_edge_clearance
    _,surface,_,_=build_scene(primitive_mesh(tmp_path/'cylinder.obj','cylinder',.02,.08))
    p=torch.tensor([[.02,0.,0.],[.02,0.,.039],[.019,0.,.04]],device='cuda',dtype=torch.float64)
    original=p.clone()
    distance=sharp_edge_clearance(surface,p,'cuda:0')
    # The OBJ uses a 48-sided cross section: the cap point's closest rim is
    # on a chord, giving the exact regular-polygon cosine correction.
    np.testing.assert_allclose(distance.cpu(),[.04,.001,.001*np.cos(np.pi/48)],atol=1e-7)
    torch.testing.assert_close(p,original,rtol=0,atol=0)


@pytest.mark.parametrize('hand_model', ['wuji1', 'wuji2'])
def test_template_cuda_fk_and_normal_derivative_match_mujoco(tmp_path, hand_model):
    import mujoco
    from retarget.grasp_template import WujiContactKinematics
    model,surface,_,_=build_scene(primitive_mesh(tmp_path/'sphere.obj','sphere',.02,.04),hand_model=hand_model)
    names=['right_palm_link']+[f'right_finger{i}_link{j}' for i in range(1,6) for j in range(1,5)]
    bodies=np.array([model.body(n).id for n in names])
    rng=np.random.default_rng(3)
    local=rng.normal(size=(len(bodies),3))*.01
    normals=rng.normal(size=local.shape);normals/=np.linalg.norm(normals,axis=-1,keepdims=True)
    hand=StaticHand(model,surface,bodies,local,normals)
    fk=WujiContactKinematics(model,bodies,'cuda:0')
    q=np.r_[[.03,-.02,.01,.4,-.3,.6],rng.uniform(.1,.6,20)]
    value=fk.tensor(q);lp,ln=fk.tensor(local),fk.tensor(normals)
    p,n=fk.contacts(value,lp,ln)
    hand.set(q)
    np.testing.assert_allclose(p.cpu(),hand.points(bodies,local),atol=1e-12)
    expected=np.einsum('nij,nj->ni',hand.data.xmat[bodies].reshape(-1,3,3),normals)
    np.testing.assert_allclose(n.cpu(),expected,atol=1e-12)
    jac=torch.func.jacfwd(lambda v:fk.contacts(v,lp,ln)[1])(value).cpu().numpy()
    for col in (0,3,5,6,10,25):
        delta=np.eye(26)[col]*1e-6
        def normal_at(q):
            hand.set(q)
            return np.einsum('nij,nj->ni',hand.data.xmat[bodies].reshape(-1,3,3),normals)
        finite=(normal_at(q+delta)-normal_at(q-delta))/2e-6
        np.testing.assert_allclose(jac[:,:,col],finite,atol=1e-8)


def test_closing_reference_preserves_root_and_moves_contact_inward(tmp_path):
    from retarget.grasp_reference import closing_posture,closing_sequence
    from retarget.grasp_template import WujiContactKinematics
    model,_,_,_=build_scene(primitive_mesh(tmp_path/'sphere.obj','sphere',.02,.04))
    body=np.array([model.body('right_finger2_link4').id])
    with np.load(ROOT/'assets/wuji/qp_right/right_finger2_link4.npz') as asset:
        local=asset['local_pos'][10:11];normals=asset['local_normal'][10:11]
    fk=WujiContactKinematics(model,body,'cuda:0')
    initial=np.r_[[.03,-.01,.02,.2,.3,-.4],np.full(20,.2)]
    p,n=fk.contacts(fk.tensor(initial),fk.tensor(local),fk.tensor(normals))
    points,outward=p.cpu().numpy(),-n.cpu().numpy()
    untouched=points.copy()
    closed,info=closing_posture(model,body,local,normals,initial,points,outward,
                               depth=.002,device='cuda:0')
    actual,_=fk.contacts(fk.tensor(closed),fk.tensor(local),fk.tensor(normals))
    np.testing.assert_array_equal(closed[:6],initial[:6])
    np.testing.assert_array_equal(points,untouched)
    assert np.linalg.norm(actual.cpu().numpy()-(points-.002*outward))<.0005
    references=closing_sequence(initial,closed,duration=.3,ref_dt=1/30)
    np.testing.assert_array_equal(references[0],initial)
    np.testing.assert_allclose(references[-1],closed,atol=1e-12)
    assert info['root_fixed'] and info['depth_mm']==2.


@pytest.mark.parametrize('radius',[.015,.020,.025])
def test_contact_fit_preserves_shape_and_regions_for_scaled_cylinders(tmp_path,radius):
    from retarget.dexonomy import load_template,map_contacts,anatomical_joints,retarget_from_mano
    from retarget.grasp_template import fit_contact_template
    checkout=ROOT/'third_party/Dexonomy'
    if not (checkout/'assets/hand/shadow/raw_anno/6_Prismatic_4_Finger.yaml').is_file():
        pytest.skip('Needs original Shadow annotation')
    model,surface,_,_=build_scene(primitive_mesh(tmp_path/'cylinder.obj','cylinder',radius,.08))
    shadow,posed,contacts,_=load_template(checkout,'6_Prismatic_4_Finger')
    bodies,local,normals,_=map_contacts(shadow,contacts,model)
    hand=StaticHand(model,surface,bodies,local,normals)
    seed,_=retarget_from_mano(model,anatomical_joints(shadow,posed)[None])
    q,info=fit_contact_template(hand,shadow,posed,contacts,seed[0],device='cuda:0',opening=2*radius)
    assert info['bone_direction_max_deviation_deg'] <= 30.+1.e-5
    assert min(info['contact_region_normal_dot']) > .5
    assert not info['contacts_required']
    assert np.all(q[6:]>=model.jnt_range[6:26,0]) and np.all(q[6:]<=model.jnt_range[6:26,1])
    for i,body in enumerate(bodies):
        with np.load(ROOT/'assets/wuji/qp_right'/f'{model.body(int(body)).name}.npz') as asset:
            j=info['surface_query_indices'][i]
            np.testing.assert_array_equal(hand.local[i],asset['local_pos'][j])
            np.testing.assert_array_equal(hand.normals[i],asset['local_normal'][j])


def test_built_static_inputs_produce_same_plan_as_direct_run_py_solver(tmp_path,monkeypatch):
    import retarget.mano_noise as cpu_sdf
    import retarget.dexonomy as dx
    from retarget.fa.solver import optimizer_class,settings_from_args
    from scipy.spatial.transform import Rotation
    path=primitive_mesh(tmp_path/'object.obj','sphere',.025,.05)
    model,surface,_,_=build_scene(path)
    bodies,local,normals=[],[],[]
    for finger in (1,2,3):
        name=f'right_finger{finger}_link4'
        with np.load(ROOT/'assets/wuji/qp_right'/f'{name}.npz') as asset:
            bodies.append(model.body(name).id);local.append(asset['local_pos'][10]);normals.append(asset['local_normal'][10])
    hand=StaticHand(model,surface,np.array(bodies),np.array(local),np.array(normals))
    q=np.r_[np.zeros(6),np.full(20,.3)]
    hand.set(q);q[:3]-=hand.points(hand.bodies,hand.local).mean(0);hand.set(q)
    audit=GraspAudit(surface,'cuda:0')
    anchor,normal,_=audit.query(torch.tensor(hand.points(hand.bodies,hand.local),device='cuda',dtype=torch.float32))
    guess=from_dexonomy_result(dict(obj_pose=np.array([0.,0.,0.,1.,0.,0.,0.]),
        grasp_pose=np.r_[q[:3],Rotation.from_euler('XYZ',q[3:6]).as_quat(scalar_first=True)],
        grasp_qpos=q[6:],obj_cp_w=anchor.cpu().numpy(),obj_cn_w=-normal.cpu().numpy()))
    args=SimpleNamespace(solver_config=ROOT/'data/example/forceaware.yaml',object_mass=.2,
        friction=.6,iterations=1,multistarts=1,seed=0,init_device='cuda:0',hand_solver='adamw',
        contact_radius_mm=5.,contact_motion_tolerance_mm=.1,contact_tolerance_mm=2.,
        penetration_tolerance_mm=2.,cone_edges=16)
    def forbidden(*a,**kw):
        raise AssertionError('CPU pointwise SDF must not be used by the solver')
    monkeypatch.setattr(cpu_sdf,'nearest_sdf',forbidden);monkeypatch.setattr(dx,'nearest_sdf',forbidden)
    experiment=prepare_experiment(tmp_path/'prepared',hand,guess,args)
    runtime=GraspRuntime(experiment,args)
    actual,snapshot,next_state=runtime.solve()
    # Instantiate the public factory used by run.py directly, on exactly the
    # same prepared scene/state/targets. No grasp-specific optimization wrapper.
    direct=optimizer_class(settings_from_args(args))(runtime.model,runtime.device_model,runtime.index,
        guidance_count=len(hand.local),cfg=runtime.cfg)
    expected=direct.solve(runtime.qpos,runtime.qvel,runtime.targets,
        warm=np.zeros(runtime.index.action_dim*(1+runtime.cfg.optimizer.horizon),np.float32),
        previous_action=runtime.qpos[runtime.index.hand_qpos],allow_init=False,seed=0,
        execution_steps=1,start_time=0.)
    selected=int(expected.diag['selected'])
    np.testing.assert_allclose(actual.raw,expected.raw,atol=1e-7,rtol=1e-6)
    np.testing.assert_array_equal(next_state,actual.next_qpos)
    np.testing.assert_allclose(next_state,expected.next_qpos,atol=1e-7)
    np.testing.assert_allclose(runtime.predicted_terminal_qpos,
        direct.backend.steps[-1].recorded.result.qpos.numpy()[selected],atol=1e-7)
    assert runtime.output_workspace is runtime.optimizer.backend.loss.immediate[1]
    _,audited= audit.evaluate(runtime,hand,args,int(actual.diag['selected']))
    np.testing.assert_allclose(audited['hand_points'],snapshot['x'][0],atol=1e-6)
    other=direct.objective.snapshot(selected)
    for name in ('z','x','dual','template'):
        np.testing.assert_allclose(snapshot[name],other[name],atol=1e-7)
    assert direct.last_solver_diag['gradient_source']=='analytic_adjoint'
    assert direct.last_solver_diag['contact_surface_optimizer']=='sampled_comfree_batched_lm'
    assert direct.last_solver_diag['hand_subproblems'][0]['parameter_update_backend']=='torch_cuda'
    assert direct.last_solver_diag['hand_subproblems'][0]['device_objective_callback']
    # Two feedback windows, including a partial final interval, must match
    # run.py's original explicit solve/execute/warm-shift contract. Feed hard
    # executor states into both solvers, never the soft predicted endpoint.
    from forceaware.workflow import RollingController,InitialRun
    from forceaware.time_grid import SolveRequest
    from forceaware.control import resample_warm_corrections
    from forceaware.executor import WarpActionExecutor
    initial=InitialRun(runtime.qpos,runtime.qvel,
        np.zeros_like(actual.raw),runtime.qpos[runtime.index.hand_qpos],0.,
        SolveRequest.create(runtime.cfg.simulator.time_grid,duration_seconds=.04))
    controller=RollingController(runtime.cfg,runtime.index,runtime.sequence,
        runtime.optimizer,initial,allow_init=False)
    executor=WarpActionExecutor(runtime.model,runtime.device_model,index=runtime.index,cfg=runtime.cfg)
    q,v=initial.qpos,initial.qvel
    warm,previous=initial.warm,initial.previous_action
    from forceaware.targets import make_window
    elapsed_steps=0
    for window,steps in enumerate((3,1)):
        targets=make_window(runtime.sequence,executed_steps=elapsed_steps,
            horizon=runtime.cfg.optimizer.horizon,time_grid=runtime.cfg.simulator.time_grid,
            start_frame=0.,end_frame=runtime.cfg.sequence.end,hold_frame=runtime.cfg.sequence.hold_frame)
        expected=direct.solve(q,v,targets,warm=warm,previous_action=previous,allow_init=False,
            seed=window,execution_steps=steps,start_time=executor.elapsed)
        actual,_=controller.solve(q,v,execution_steps=steps,start_time=executor.elapsed)
        np.testing.assert_allclose(actual.raw,expected.raw,atol=2e-6,rtol=2e-6)
        np.testing.assert_allclose(actual.control_sequence,expected.control_sequence,atol=2e-6)
        result=executor.execute_action_sequence(q,v,actions=actual.control_sequence)
        q,v=result.next_qpos,result.next_qvel
        controller.commit(actual,execution_steps=steps)
        warm=resample_warm_corrections(expected.raw,action_dim=runtime.index.action_dim,
            horizon=runtime.cfg.optimizer.horizon,executed_dense_steps=steps,
            physics_steps_per_knot=runtime.cfg.simulator.time_grid.knot_substeps,
            warm_scale=runtime.cfg.optimizer.warm_shift_scale)
        previous=expected.control_sequence[-1]
        elapsed_steps+=steps
        np.testing.assert_allclose(controller.previous,previous,atol=2e-6)
        np.testing.assert_allclose(controller.warm,warm,atol=2e-6)
    assert controller.executed_steps==4 and controller.window==2
    # Auditing actual physics must not alias/replace Warp struct storage in
    # the differentiable planner. A subsequent solve must still match run.py.
    original=runtime.optimizer.loss.immediate[0].job.state
    pointers={k:getattr(original,k).ptr for k in ('qpos','qvel','body_position','body_matrix')}
    gradients={k:getattr(original,k).grad for k in pointers}
    audit.evaluate_state(runtime,hand,args,q,v)
    assert pointers=={k:getattr(original,k).ptr for k in pointers}
    assert all(getattr(original,k).grad is gradients[k] for k in gradients)
    targets=make_window(runtime.sequence,executed_steps=4,horizon=runtime.cfg.optimizer.horizon,
        time_grid=runtime.cfg.simulator.time_grid,start_frame=0.,end_frame=runtime.cfg.sequence.end,
        hold_frame=runtime.cfg.sequence.hold_frame)
    expected=direct.solve(q,v,targets,warm=warm,previous_action=previous,allow_init=False,
        seed=2,execution_steps=1,start_time=executor.elapsed)
    actual,_=controller.solve(q,v,execution_steps=1,start_time=executor.elapsed)
    np.testing.assert_allclose(actual.raw,expected.raw,atol=2e-6,rtol=2e-6)


def test_input_self_collision_matches_shared_warp_capsules_and_derivative(tmp_path):
    import mujoco
    import warp as wp
    from forceaware import mpc_loss as fa
    from retarget.grasp_input_geometry import WujiSelfCollision
    from retarget.grasp_audit import capsule_depth
    model,_,_,_=build_scene(primitive_mesh(tmp_path/'object.obj','cylinder',.02625,.1))
    helper=WujiSelfCollision(model,'cuda:0')
    # Build the common geometry through its real query-source interface.
    clouds=[];bodies=[]
    for finger in range(1,6):
        for link in range(1,5):
            name=f'right_finger{finger}_link{link}'
            with np.load(ROOT/'assets/wuji/qp_right'/f'{name}.npz') as asset:
                p=asset['local_pos']
            clouds.extend(p);bodies.extend([model.body(name).id]*len(p))
    collision=SimpleNamespace(batches=[SimpleNamespace(model=SimpleNamespace(
        source_centers=wp.array(clouds,dtype=wp.vec3,device='cuda:0'),
        source_radii=wp.zeros(len(clouds),device='cuda:0'),
        source_body_ids=wp.array(bodies,dtype=int,device='cuda:0')))])
    ids,a,b,r,pairs=fa._finger_joint_geometry(model,collision,'wuji_right')
    loss_model=fa.LossModel()
    for name,value,dtype in [('finger_joint_body',ids,int),('finger_joint_start',a,wp.vec3),
        ('finger_joint_end',b,wp.vec3),('finger_joint_radius',r,float),
        ('finger_joint_pair_left',pairs[:,0],int),('finger_joint_pair_right',pairs[:,1],int)]:
        setattr(loss_model,name,wp.array(value,dtype=dtype,device='cuda:0'))
    data=mujoco.MjData(model)
    for angle in (.2,.8):
        q=np.r_[[.02,-.01,.03,.4,-.2,.6],np.full(20,angle)]
        data.qpos[:]=q;mujoco.mj_kinematics(model,data)
        state=fa.LossState()
        state.body_position=wp.array(data.xpos[None],dtype=wp.vec3,device='cuda:0')
        state.body_matrix=wp.array(data.xmat.reshape(1,-1,3,3),dtype=wp.mat33,device='cuda:0')
        job=fa.LossJob();job.model=loss_model;job.state=state
        depth=wp.zeros(len(pairs),device='cuda:0')
        wp.launch(capsule_depth,len(pairs),inputs=[job,0],outputs=[depth],device='cuda:0')
        actual=helper.depths(helper.fk.tensor(q))
        np.testing.assert_allclose(actual.cpu(),depth.numpy(),atol=4e-8)
    q=helper.fk.tensor(q)
    jac=torch.func.jacfwd(helper.depths)(q)
    assert bool(torch.isfinite(jac).all())
    for col in (6,11,20):
        delta=torch.zeros_like(q);delta[col]=1e-6
        finite=(helper.depths(q+delta)-helper.depths(q-delta))/(2e-6)
        torch.testing.assert_close(jac[:,col],finite,atol=1e-7,rtol=1e-4)


def test_dense_power_input_preserves_annotations_and_repairs_envelope_and_self_collision(tmp_path):
    from retarget.dexonomy import load_template,map_contacts,anatomical_joints,retarget_from_mano
    from retarget.grasp_template import fit_contact_template
    from retarget.grasp_input_geometry import fit_power_cylinder_input,dense_power_template,WujiSelfCollision
    checkout=ROOT/'third_party/Dexonomy'
    if not (checkout/'assets/hand/shadow/raw_anno/1_Large_Diameter.yaml').is_file():
        pytest.skip('Needs original Shadow annotation')
    model,surface,_,_=build_scene(primitive_mesh(tmp_path/'cylinder.obj','cylinder',.02625,.1))
    shadow,posed,contacts,_=load_template(checkout,'1_Large_Diameter')
    bodies,local,normals,_=map_contacts(shadow,contacts,model)
    hand=StaticHand(model,surface,bodies,local,normals)
    assert dense_power_template(hand,dict(kind='cylinder'))
    assert not dense_power_template(hand,dict(kind='sphere'))
    seed,_=retarget_from_mano(model,anatomical_joints(shadow,posed)[None])
    q,_=fit_contact_template(hand,shadow,posed,contacts,seed[0],device='cuda:0',opening=.0525,joint_prior=1.)
    # Original HandObjMatcher passes these joint angles as float32.
    q[6:]=q[6:].astype(np.float32)
    fitted,info=fit_power_cylinder_input(hand,q,device='cuda:0')
    np.testing.assert_array_equal(hand.bodies,bodies)
    assert len(hand.local)==len(contacts)==16 and info['all_annotation_contacts_retained']
    assert info['bone_direction_max_deviation_deg'] <= 30.+1.e-5
    assert not info['contacts_required']
    helper=WujiSelfCollision(model,'cuda:0')
    assert float(helper.depths(helper.fk.tensor(fitted)).max())<.0006
    hand.set(fitted)
    audit=GraspAudit(surface,'cuda:0')
    _,_,distance=audit.query(torch.tensor(hand.points(hand.all_bodies,hand.all_local),device='cuda:0'))
    assert float((-distance.min()).clamp(min=0.))<.001
    for body,local,normal,index in zip(bodies,hand.local,hand.normals,info['surface_query_indices']):
        with np.load(ROOT/'assets/wuji/qp_right'/f'{model.body(int(body)).name}.npz') as asset:
            np.testing.assert_array_equal(local,asset['local_pos'][index])
            np.testing.assert_array_equal(normal,asset['local_normal'][index])
