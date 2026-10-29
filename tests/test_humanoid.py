"""Whole-body task costs, coupled ADMM and MJWarp execution checks."""
from pathlib import Path

import mujoco
import numpy as np
import pytest
import warp as wp

from retarget.humanoid import HumanoidProblem, ProjectionADMM, Settings, project_box
from retarget.humanoid_terms import TERMS, robust_squared
from retarget.fa.adamw import AdamW

DATA = Path(__file__).resolve().parents[1] / 'data/spider'
OMOMO = DATA / 'processed/omomo/unitree_g1/humanoid_object/move_largebox'


@wp.kernel
def _robust_probe(values:wp.array(dtype=float),output:wp.array(dtype=float)):
    i=wp.tid()
    output[i]=robust_squared(values[i],1.)


def test_robust_wrench_penalty_value_and_bounded_derivative():
    values=np.array([0.,1.e-8,.25,1.,100.],np.float32)
    inputs=wp.array(values,device='cpu',requires_grad=True)
    output=wp.zeros(len(values),device='cpu',requires_grad=True)
    with wp.Tape() as tape:wp.launch(_robust_probe,len(values),inputs=[inputs,output],device='cpu')
    tape.backward(grads={output:wp.ones(len(values),device='cpu')})
    expected=2.*values.astype(float)/(np.sqrt(1.+values.astype(float))+1.)
    np.testing.assert_allclose(output.numpy(),expected,rtol=2.e-6,atol=1.e-10)
    np.testing.assert_allclose(inputs.grad.numpy(),1./np.sqrt(1.+values.astype(float)),rtol=2.e-6)


@pytest.fixture(scope='module')
def problem():
    return HumanoidProblem(OMOMO/'scene.xml', OMOMO/'0/trajectory_kinematic.npz',
        settings=Settings(horizon=2,knot_stride=1,substeps=2,multistarts=2,
                          admm_iterations=2,action_iterations=1,adamw_lr=.3,contact_samples=3,contact_qp_iterations=16))


@pytest.fixture(scope='module')
def solved(problem):
    pytest.importorskip('torch')
    import torch
    if not torch.cuda.is_available():
        pytest.skip('MJWarp optimization requires CUDA')
    solver=ProjectionADMM(problem)
    state=problem.initial_state(2)
    original=state.qpos.copy()
    rollout,history=solver.solve(state,[3,4],problem.aref[2])
    np.testing.assert_array_equal(state.qpos,original)
    return solver,rollout,history


def test_box_projection_is_closest_surface():
    projected=project_box(np.array([[0.,0.,0.],[2.,3.,.2],[-.8,.2,.1]]),np.ones(3))
    np.testing.assert_allclose(projected,[[1,0,0],[1,1,.2],[-1,.2,.1]])


def test_exact_shared_adamw_and_joint_update_order(problem,solved):
    solver,rollout,history=solved
    assert type(solver.backend.adamw) is AdamW
    row=history[-1]
    assert row['update_order']==['hands_and_feet_z','action','dual']
    assert row['adamw']['parameter_update_backend']=='torch_cuda'
    assert row['adamw']['device_objective_callback']
    assert row['finite_difference_gradients']
    updates=problem.settings.action_iterations*problem.settings.admm_iterations
    assert row['backward_calls']==updates
    assert row['forward_calls']>=updates+1
    assert row['action_variables']==problem.settings.horizon*problem.model.nu
    assert all(r['action_cost_after']<=r['action_cost_before']+1e-6 for r in history)
    assert all(r['contact_cost_after']<=r['contact_cost_before']+1e-6 for r in history)
    assert history[1]['adamw']['continued_within_window']
    assert np.isfinite(rollout['qpos']).all()
    assert (np.linalg.norm(solver.dual_dense,axis=-1)<=4*np.array([problem.settings.contact_radius]*2+[problem.settings.foot_contact_radius]*2)+1e-8).all()
    mask=problem.mask[solver.backend.reference_indices]
    np.testing.assert_array_equal(solver.dual_dense[~np.broadcast_to(mask[:,None,:,None],solver.dual_dense.shape)],0.)
    assert problem.model.nu==29


def test_surface_radius_and_force_feasibility(problem,solved):
    from retarget.humanoid_contacts import project
    solver,_,_=solved
    b=solver.backend
    np.testing.assert_allclose(project(problem,solver.z_dense),solver.z_dense,atol=2e-6)
    assert (np.linalg.norm(solver.z_dense-b.anchors[:,None],axis=-1)<=np.array([problem.settings.contact_radius]*2+[problem.settings.foot_contact_radius]*2)+2e-6).all()
    assert (solver.force_dense[:,:,2:,2]>=-1e-6).all()
    tangent=np.linalg.norm(solver.force_dense[:,:,2:,:2],axis=-1)
    mu=np.asarray(problem.contact_friction)[2:]
    assert (tangent<=mu[None,None]*solver.force_dense[:,:,2:,2]+1e-4).all()






def test_object_reference_pose_velocity_are_not_tracking_targets(solved):
    solver,_,_=solved;b=solver.backend
    q,v=b.targets.qpos.numpy(),b.targets.qvel.numpy()
    x=np.zeros(b.variables.shape)
    try:
        before=b.evaluate(x,False)[0]
        changedq,changedv=q.copy(),v.copy()
        changedq[:,-7:-4]+=10;changedv[:,-6:]+=100
        b.targets.qpos.assign(changedq);b.targets.qvel.assign(changedv);b.dirty=True
        after=b.evaluate(x,False)[0]
        np.testing.assert_allclose(after,before,atol=1e-6)
    finally:
        b.targets.qpos.assign(q);b.targets.qvel.assign(v);b.dirty=True


def test_hand_world_tracking_uses_collision_centers_and_reference_contact_mask(solved):
    solver,_,_=solved;b=solver.backend
    weights=b.model.weights.numpy().copy();penalty=b.targets.penalty.numpy().copy()
    masks=b.targets.mask.numpy().copy();positions=b.targets.position.numpy().copy()
    x=np.zeros(b.variables.shape)
    try:
        only=np.zeros_like(weights);only[TERMS.index('hand_position')]=weights[TERMS.index('hand_position')]
        b.model.weights.assign(only);b.targets.penalty.zero_();b.dirty=True
        value,_=b.evaluate(x,False)
        expected=np.zeros(b.worlds);centers=b.model.hand_centers.numpy()
        for stage in b.stages:
            job=stage.job;k=job.index
            actual=job.position.numpy();rotation=job.rotation.numpy()
            reference=job.target.hand_position.numpy()
            for role,body in enumerate(b.role_bodies[:2]):
                hand=actual[:,body]+rotation[:,body]@centers[role]
                desired=reference[k,role]
                expected+=only[13]*job.normalizer*masks[k,role]*np.sum((hand-desired)**2,axis=1)
        np.testing.assert_allclose(value,expected,rtol=3e-6,atol=1e-7)
        # Changing only the reference pelvis cannot move the world hand target.
        changed=positions.copy();changed[:,b.problem.root,0]+=.3
        b.targets.position.assign(changed);b.dirty=True
        np.testing.assert_allclose(b.evaluate(x,False)[0],value,rtol=0,atol=1e-7)
        disabled=masks.copy();disabled[:,:2]=0.
        b.targets.mask.assign(disabled);b.dirty=True
        value,gradient=b.evaluate(x,True)
        np.testing.assert_array_equal(value,0.)
        np.testing.assert_array_equal(gradient,0.)
    finally:
        b.model.weights.assign(weights);b.targets.penalty.assign(penalty)
        b.targets.mask.assign(masks);b.targets.position.assign(positions);b.dirty=True


def test_object_wrench_feedforward_matches_reference_acceleration(tmp_path):
    model=mujoco.MjModel.from_xml_path(str(OMOMO/'scene.xml'))
    with np.load(OMOMO/'0/trajectory_kinematic.npz') as arrays:
        q=np.repeat(arrays['qpos'][:1],7,axis=0)
        ctrl=np.repeat(arrays['ctrl'][:1],7,axis=0)
    body=model.body('largebox').id;joint=model.body_jntadr[body]
    qa,va=model.jnt_qposadr[joint],model.jnt_dofadr[joint]
    dt=.05;acceleration=.4;time=np.arange(len(q))*dt
    q[:,qa]+=.5*acceleration*time**2
    v=np.zeros((len(q),model.nv));v[:,va]=acceleration*time
    path=tmp_path/'accelerating_box.npz';np.savez(path,qpos=q,qvel=v,ctrl=ctrl)
    p=HumanoidProblem(OMOMO/'scene.xml',path,ref_dt=dt)
    expected=model.body_subtreemass[body]*np.array([acceleration,0.,9.81])
    np.testing.assert_allclose(p.reference_object_wrench[:,0],np.broadcast_to(expected,(len(q),3)),atol=1e-8)
    np.testing.assert_allclose(p.reference_object_wrench[:,1],0.,atol=1e-8)


def test_world_hand_targets_touch_the_same_reference_box_anchors(solved):
    b=solved[0].backend;p=b.problem;body=b.model.object_body
    world=b.targets.hand_position.numpy()
    local=np.einsum('nji,nsj->nsi',b.rotations_ref[:,body],world-b.positions_ref[:,body,None])
    geom=p.object_geom;rotation=np.empty(9);mujoco.mju_quat2Mat(rotation,p.model.geom_quat[geom])
    local=(local-p.model.geom_pos[geom])@rotation.reshape(3,3)
    anchor=p.anchors[b.reference_indices,:2]
    normal=p.surface.normal(local,anchor)
    surface=local-p.model.geom_size[p.hand_geoms,:1]*normal
    np.testing.assert_allclose(surface,anchor,atol=1.e-7,rtol=0.)


def test_contact_block_is_task_coupled_at_fixed_action(solved):
    solver,_,_=solved;b=solver.backend
    weights=b.model.weights.numpy();forces=b.targets.forces.numpy();z=b.targets.z.numpy()
    penalties=b.targets.penalty.numpy()
    try:
        only_wrench=np.zeros_like(weights);only_wrench[TERMS.index('contact_wrench')]=1.
        b.model.weights.assign(only_wrench);b.targets.penalty.zero_();b.dirty=True
        x=np.zeros(b.variables.shape)
        before=b.evaluate(x,False)[0]
        changed=forces.copy();changed[:,:,2,2]+=50.
        b.targets.forces.assign(changed);b.dirty=True
        after=b.evaluate(x,False)[0]
        assert np.max(np.abs(before-after))>1e-3
        changed_z=z.copy();changed_z[:,:,2,0]+=.005
        b.targets.z.assign(changed_z);b.dirty=True
        moved=b.evaluate(x,False)[0]
        assert np.max(np.abs(moved-after))>1e-7
    finally:
        b.model.weights.assign(weights);b.targets.forces.assign(forces);b.targets.z.assign(z)
        b.targets.penalty.assign(penalties);b.dirty=True


def test_reference_velocities_and_tilted_foot_support(problem):
    assert not problem.vref.any()
    assert problem.tracking_vref.any()
    np.testing.assert_array_equal(problem.initial_state(2).qvel,0.)
    assert 'manifold_derivative' in problem.reference_velocity_source
    for feature in problem.reference:
        for foot in range(2):
            assert abs(feature['contacts'][foot+2,2]-feature['foot_corners'][foot,:,2].min())<=.001
    assert min(feature['torso_up'][2] for feature in problem.reference)<.1


def test_object_local_contact_translation_invariance(problem):
    state=problem.initial_state(2)
    before=problem.features(state)['contacts'][:2]
    shift=np.array([.7,-.3,.2]);state.qpos[:3]+=shift;state.qpos[-7:-4]+=shift
    mujoco.mj_forward(problem.model,state)
    np.testing.assert_allclose(problem.features(state)['contacts'][:2],before,atol=1e-12)


def test_amass_airborne_contact_masks():
    task=DATA/'processed/amass/unitree_g1/humanoid/sprint'
    p=HumanoidProblem(DATA/'processed/amass/assets/robots/unitree_g1/scene.xml',task/'0/trajectory_kinematic.npz')
    assert not p.mask[:,:2].any()
    assert (~p.mask[:,2:].any(axis=1)).any()
    assert p.model.dof_frictionloss.max()==pytest.approx(.1)
    assert p.mask[0,2:].all()  # deep penetration is contact, not flight


def test_bimanual_scene_and_reference():
    task=DATA/'processed/arcticv2/xhand/bimanual/s01-box_use_01'
    model=mujoco.MjModel.from_xml_path(str(task/'scene.xml'))
    with np.load(task/'0/trajectory_kinematic.npz',allow_pickle=False) as data:
        assert data['qpos'].shape==(117,model.nq)
        assert data['qvel'].shape==(117,model.nv)




def test_cuda_contact_qp_matches_independent_cpu_reference():
    from retarget.humanoid_qp import ContactQP
    from retarget.humanoid_contacts import force_qp
    import torch
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    rng=np.random.default_rng(23)
    matrix=rng.normal(size=(16,12,32));target=rng.normal(size=(16,12))
    expected,cost=force_qp(matrix,target,48)
    actual,gpu_cost=ContactQP(wp.get_device('cuda:0')).solve(matrix,target,48)
    np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=2e-5)
    np.testing.assert_allclose(gpu_cost,cost,atol=3e-6,rtol=2e-5)
    assert actual.min()>=0
    assert actual.reshape(-1,8,4).sum(axis=-1).max()<=1+1e-6


def test_prediction_contact_queries_match_actual_geometry(problem,solved):
    _,rollout,_=solved
    geom=problem.object_geom
    rotation=np.empty(9);mujoco.mju_quat2Mat(rotation,problem.model.geom_quat[geom]);rotation=rotation.reshape(3,3)
    data=mujoco.MjData(problem.model)
    for q,v,query in zip(rollout['qpos'],rollout['qvel'],rollout['contacts']):
        data.qpos[:]=q;data.qvel[:]=v;mujoco.mj_forward(problem.model,data)
        actual=problem.features(data)['contacts']
        actual[:2]=actual[:2]@rotation.T+problem.model.geom_pos[geom]
        np.testing.assert_allclose(query,actual,atol=5e-5)


def test_balance_detects_collapse_and_velocity_with_upright_torso(solved):
    """Leg collapse is visible even when torso tilt is exactly the reference."""
    from retarget.humanoid_terms import whole_body_loss,diagnose_terms,BALANCE_PARTS
    b=solved[0].backend;job=b.stages[1].job
    originals={name:getattr(job,name).numpy().copy() for name in ('com','previous_com','position','rotation')}
    weights=b.model.weights.numpy();penalty=b.targets.penalty.numpy()
    target_com=b.targets.com.numpy();target_position=b.targets.position.numpy();target_rotation=b.targets.rotation.numpy()
    try:
        w=np.zeros_like(weights);w[7]=weights[7];b.model.weights.assign(w);b.targets.penalty.zero_()
        c=np.broadcast_to(target_com[1]+np.array([-.03,.02,-.18]),(b.worlds,3)).copy()
        prior=np.broadcast_to(target_com[0]+np.array([-.03,.02,-.17]),(b.worlds,3)).copy()
        position=originals['position'].copy();position[:,b.model.root,2]=target_position[1,b.model.root,2]-.2
        rotation=originals['rotation'].copy();rotation[:,b.model.torso]=target_rotation[1,b.model.torso]
        job.com.assign(c.astype(np.float32));job.previous_com.assign(prior.astype(np.float32));job.position.assign(position);job.rotation.assign(rotation)
        wp.launch(diagnose_terms,b.worlds,inputs=[job],device=b.device)
        parts=job.balance_parts.numpy()
        assert (parts[:,BALANCE_PARTS.index('collapse')]>0).all()
        assert (parts[:,BALANCE_PARTS.index('vertical_velocity')]>0).all()
        from retarget.humanoid_terms import Stage
        for array in (job.com,job.previous_com,job.position,job.loss):
            array.requires_grad=True
            if array.grad is not None:array.grad.zero_()
        probe=Stage()
        for name in Stage.vars:setattr(probe,name,getattr(job,name))
        with wp.Tape() as tape:
            wp.launch(whole_body_loss,b.worlds,inputs=[probe],device=b.device)
        tape.backward(grads={job.loss:b.ones})
        error=c-target_com[1];dv=(c-prior-target_com[1]+target_com[0])/b.model.dt
        omega=np.sqrt(9.81/max(target_com[1,2]-b.model.ground,.2));dcm=error+dv/omega
        gpos=2*error/np.array([b.model.balance_xy_scale**2,b.model.balance_xy_scale**2,b.model.balance_z_scale**2])
        gdcm=dcm.copy()/b.model.balance_xy_scale**2;gdcm[:,2]=0
        gv=gdcm/omega;gv[:,2]=.5*dv[:,2]/b.model.balance_velocity_scale**2
        factor=weights[7]*job.normalizer
        np.testing.assert_allclose(job.com.grad.numpy(),factor*(gpos+gdcm+gv/b.model.dt),rtol=3e-5,atol=.04)
        np.testing.assert_allclose(job.previous_com.grad.numpy(),-factor*gv/b.model.dt,rtol=3e-5,atol=.04)
        assert (job.position.grad.numpy()[:,b.model.root,2]<0).all()
    finally:
        for name,array in originals.items():getattr(job,name).assign(array)
        b.model.weights.assign(weights);b.targets.penalty.assign(penalty);b.dirty=True


def test_packed_snapshots_equal_independent_full_geometry(solved):
    b=solved[0].backend;b.evaluate(np.zeros(b.variables.shape),False)
    snapshot=b.snapshots();assert b.snapshots() is snapshot
    for i in (0,len(b.stages)-1):
        job=b.stages[i].job;pos=job.position.numpy();rot=job.rotation.numpy()
        com=sum(b.problem.model.body_mass[k]*(pos[:,k]+np.einsum('wij,j->wi',rot[:,k],b.problem.model.body_ipos[k]))/b.model.robot_mass for k in b.model.robot_bodies.numpy())
        np.testing.assert_allclose(snapshot['com'][i],com,atol=3e-7)
        np.testing.assert_array_equal(snapshot['torso_rotation'][i],rot[:,b.model.torso])
        np.testing.assert_array_equal(snapshot['object_position'][i],pos[:,b.model.object_body])
        np.testing.assert_array_equal(snapshot['qvel'][i],job.qvel.numpy()[:,:3])


def test_rollout_cache_preserves_mixed_worlds_and_warmstarts(solved):
    b=solved[0].backend
    if b.rollout_cache is None:pytest.skip('cache disabled')
    b.prepare(b.problem.initial_state(2),3,b.problem.aref[2])
    first=np.zeros(b.variables.shape);second=np.random.default_rng(81).normal(0,.05,first.shape)
    b.evaluate(first,False);b.evaluate(second,False)
    mixed=second.copy();mixed[0]=first[0]
    count=b.forward_calls
    cached_value,_=b.evaluate(mixed,False)
    assert b.forward_calls==count
    assert not b.rollout_cache.workspace_valid
    cached={k:v.copy() for k,v in b.snapshots().items()}
    _,cached_gradient=b.evaluate(mixed,True)
    assert b.forward_calls>count
    b.rollout_cache.clear();b._cached=None
    direct_value,direct_gradient=b.evaluate(mixed,True)
    np.testing.assert_allclose(cached_value,direct_value,rtol=2e-6,atol=3e-6)
    assert np.linalg.norm(cached_gradient-direct_gradient)/max(np.linalg.norm(direct_gradient),1e-12)<.005
    for key,value in b.snapshots().items():np.testing.assert_allclose(cached[key],value,rtol=2e-6,atol=3e-6)


def test_ground_points_can_adapt_without_relaxing_hand_grasp(problem):
    from retarget.humanoid_contacts import bounded_project
    anchors=problem.anchors[2].copy()
    # Convert the object collision-geom local anchors to body coordinates.
    geom=problem.object_geom;rotation=np.empty(9)
    mujoco.mju_quat2Mat(rotation,problem.model.geom_quat[geom])
    anchors[:2]=anchors[:2]@rotation.reshape(3,3).T+problem.model.geom_pos[geom]
    proposal=anchors.copy();proposal[2:,0]+=.10
    result=bounded_project(problem,proposal,anchors,np.array([.005,.005,.15,.15]))
    np.testing.assert_allclose(result[2:,0]-anchors[2:,0],.10,atol=1e-8)
    np.testing.assert_allclose(result[:2],anchors[:2],rtol=0,atol=1e-12)
    np.testing.assert_allclose(result[2:,2],problem.ground_height,atol=1e-8)




def test_realization_qp_completion_of_square_matches_expanded_system(solved):
    from retarget.humanoid_contacts import force_qp
    rng=np.random.default_rng(49)
    matrix=rng.normal(0,.2,(3,12,32));desired=rng.normal(size=(3,12));actual=rng.normal(size=(3,12));lam=1.
    root=np.sqrt(1+lam)
    weighted=matrix*root;target=(desired+lam*actual)/root
    cuda,weighted_cost=solved[0].backend.contact_qp.solve(weighted,target,64)
    expanded=np.concatenate((matrix,np.sqrt(lam)*matrix),axis=1)
    expanded_target=np.concatenate((desired,np.sqrt(lam)*actual),axis=1)
    cpu,expanded_cost=force_qp(expanded,expanded_target,64)
    constant=lam/(1+lam)*np.sum((desired-actual)**2,axis=1)
    np.testing.assert_allclose(cuda,cpu,rtol=1e-4,atol=3e-6)
    np.testing.assert_allclose(weighted_cost+constant,expanded_cost,rtol=2e-6,atol=1e-5)


def test_relative_keypoints_do_not_repeat_absolute_root_tracking(solved):
    from retarget.humanoid_terms import whole_body_loss
    b=solved[0].backend;job=b.stages[1].job
    position=job.position.numpy();weights=b.model.weights.numpy();penalty=b.targets.penalty.numpy()
    try:
        w=np.zeros_like(weights);w[4]=weights[4];b.model.weights.assign(w);b.targets.penalty.zero_()
        wp.launch(whole_body_loss,b.worlds,inputs=[job],device=b.device);before=job.loss.numpy()
        job.position.assign(position+np.array([.3,-.2,.1],np.float32))
        wp.launch(whole_body_loss,b.worlds,inputs=[job],device=b.device)
        np.testing.assert_allclose(job.loss.numpy(),before,rtol=1e-5,atol=2e-6)
        w[4]=0.;w[0]=weights[0];b.model.weights.assign(w)
        wp.launch(whole_body_loss,b.worlds,inputs=[job],device=b.device);translated_root=job.loss.numpy()
        job.position.assign(position)
        wp.launch(whole_body_loss,b.worlds,inputs=[job],device=b.device)
        assert (translated_root>job.loss.numpy()+.01).all()
    finally:
        job.position.assign(position);b.model.weights.assign(weights);b.targets.penalty.assign(penalty);b.dirty=True


def test_default_amass_scene_matches_spider_task_environment():
    from run_humanoid import task_paths
    scene,reference=task_paths(DATA,'humanoid')
    assert scene==DATA/'processed/amass/unitree_g1/humanoid/sprint/scene.xml'
    p=HumanoidProblem(scene,reference)
    np.testing.assert_array_equal(p.model.actuator_gainprm[:,0],500.)
    np.testing.assert_array_equal(p.model.dof_armature[6:],1.)
    np.testing.assert_array_equal(p.model.dof_frictionloss,0.)
    assert [len(ids) for ids in p.foot_geoms]==[4,4]
    assert p.model.opt.iterations==5 and p.model.opt.ls_iterations==10
    np.testing.assert_allclose(p.model.opt.o_solimp,[.9,.95,.001,.5,2.])




