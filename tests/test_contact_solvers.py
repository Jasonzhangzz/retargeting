"""Independent numeric checks for contact-set LM and pressure QP solvers."""
import numpy as np
import pytest
from scipy.optimize import minimize
from retarget.contact_exploration import ObjectMeshSurface
from retarget.contact_solvers import (
    SampledComFreeLM, QPContactSolver, contact_basis, grasp_matrix,
    batched_force_lm, comfree_velocity, motion_residual, wrench_qp,
)


def surface():
    return ObjectMeshSurface([[-.1,-.1,0],[.1,-.1,0],[.1,.1,0],[-.1,.1,0]],[[0,1,2],[0,2,3]])


def test_comfree_force_jacobian_includes_floor_and_quaternion_step():
    points=np.array([[[-.05,0.,0.],[.05,0.,0.]]])
    normals=np.array([[[-1.,0,0],[1.,0,0]]])
    matrix=grasp_matrix(points,contact_basis(normals),np.zeros(3))
    inverse=np.array([np.diag([5.]*3+[100.]*3)])
    velocity=np.array([[.01,.02,-.1,.2,-.1,.3]])
    external=np.array([[0.,0.,-1.962,0,0,0]])
    dt=np.array([.01]);row=np.array([[0.,0,1.,0,0,0]]);gap=np.array([-.001])
    force=np.array([[1.,.1,.2,1.,-.1,-.2]])
    def evaluate(f):
        v,j=comfree_velocity(f,matrix,inverse,velocity,external,dt,row,gap)
        return motion_residual(v,j,np.zeros((1,6)),dt,.1)
    _,analytic=evaluate(force)
    numeric=np.empty_like(analytic)
    for col in range(6):
        plus,minus=force.copy(),force.copy();plus[0,col]+=1e-5;minus[0,col]-=1e-5
        numeric[0,:,col]=(evaluate(plus)[0]-evaluate(minus)[0])[0]/2e-5
    np.testing.assert_allclose(analytic,numeric,atol=1e-9,rtol=1e-6)


def test_batched_lm_improves_force_cost_and_preserves_cones_and_inactive_slots():
    points=np.tile([[[-.05,0.,0.],[.05,0.,0.]]],(3,1,1))
    normals=np.tile([[[-1.,0.,0.],[1.,0.,0.]]],(3,1,1))
    friction=np.full((3,2,2),.6);active=np.array([[1,1],[1,0],[1,1]],bool)
    force,cost,before=batched_force_lm(points,normals,friction,active,
        inverse=np.tile(np.diag([5.]*3+[100.]*3),(3,1,1)),velocity=np.zeros((3,6)),
        external=np.tile([0,0,-1.962,0,0,0],(3,1)),goal=np.zeros((3,6)),
        dt=np.full(3,.03),center=np.zeros(3),radius=.1,iterations=8)
    assert np.all(cost<=before+1e-12)
    assert cost[0]<before[0]-.00001
    np.testing.assert_array_equal(force[1,1],0.)
    assert np.all(force[...,0]>=0)
    assert np.all(np.linalg.norm(force[...,1:],axis=-1)<=.6*force[...,0]+1e-10)
    assert np.all(np.linalg.norm(force,axis=-1)<=10.+1e-10)


def test_qp_matches_scipy_and_cannot_use_zero_pressure():
    points=np.array([[[-.05,0.,0.],[.05,0.,0.]]])
    normals=np.array([[[-1.,0.,0.],[1.,0.,0.]]]);mu=np.full((1,2,2),.6)
    target=np.array([[0.,0.,8.,0.,0.,0.]])
    cost,_,info=wrench_qp(points,normals,mu,np.ones((1,2),bool),target,
        center=np.zeros(3),radius=.1,iterations=1024,force_limit=1.)
    b=contact_basis(normals);n,t1,t2=b[...,0],b[...,1],b[...,2]
    primitive=np.stack((n+.6*t1,n-.6*t1,n+.6*t2,n-.6*t2),axis=-1)
    G=grasp_matrix(points,primitive,np.zeros(3),.1)[0]
    def objective(x):
        return .5*np.sum((G@x-target[0])**2)+.5e-7*np.sum(x*x)
    reference=minimize(objective,np.full(8,.25),method='SLSQP',bounds=[(0.,None)]*8,
        constraints=[{'type':'ineq','fun':lambda x:1-x.reshape(2,4).sum(1)},
                     {'type':'ineq','fun':lambda x:x.sum()-1.}],options={'ftol':1e-12,'maxiter':200})
    assert reference.success
    np.testing.assert_allclose(cost[0],reference.fun,atol=1e-8)
    alpha=info['pressures'][0]
    assert alpha.min()>=-1e-12 and alpha.sum()>=1-1e-8
    assert alpha.sum(1).max()<=1+1e-8
    _,_,zero=wrench_qp(points,normals,mu,np.ones((1,2),bool),np.zeros((1,6)),
                       center=np.zeros(3),radius=.1,iterations=128)
    assert zero['pressures'].sum()>=1-1e-8


@pytest.mark.parametrize('solver_type',[SampledComFreeLM,QPContactSolver])
def test_contact_updates_keep_surface_radius_inactive_and_batch_parity(solver_type):
    template=np.array([[[0.,0.,0.],[.02,0.,0.]],[[.001,0.,0.],[.03,0.,0.]]])
    hand=template+np.array([.01,.002,.001])
    weights=np.array([[1.,0.],[1.,1.]]);friction=np.full((2,2,2),.6)
    solver=solver_type(surface(),samples=8)
    points,diag=solver.solve_batch(template,template,hand,weights,friction,center=np.zeros(3),object_radius=.1)
    assert np.max(np.linalg.norm(points-template,axis=-1))<=.005+1e-8
    np.testing.assert_array_equal(points[0,1],template[0,1].astype(np.float32))
    np.testing.assert_allclose(points[...,2],0.,atol=1e-10)
    for d in diag:
        key='motion_error_mm' if solver_type is SampledComFreeLM else 'wrench_residual'
        initial_key='initial_motion_error_mm' if solver_type is SampledComFreeLM else 'initial_wrench_residual'
        assert d[key]<=d[initial_key]+1e-8
    # Randomness is keyed deterministically by group input, not call ordering.
    actual,_=solver.solve(template[0],template[0],hand[0],weights[0],friction[0],center=np.zeros(3),object_radius=.1)
    np.testing.assert_array_equal(points[0],actual)


def test_sampled_lm_includes_incumbent_and_n_counts_whole_contact_sets():
    template=np.array([[0.,0.,0.],[.02,0.,0.]])
    args=(template,template,template+[.01,0,0],np.ones(2),np.full((2,2),.6))
    baseline,one=SampledComFreeLM(surface(),samples=1).solve(*args,center=np.zeros(3),object_radius=.1)
    selected,many=SampledComFreeLM(surface(),samples=8).solve(*args,center=np.zeros(3),object_radius=.1)
    np.testing.assert_array_equal(baseline,template.astype(np.float32))
    assert one['candidate_count']==1 and one['selected_candidate']==0
    assert many['candidate_count']==8 and many['includes_incumbent']
    assert many['initial_value']==one['initial_value']
    assert many['motion_error_mm']<=one['motion_error_mm']
    assert np.linalg.norm(selected-template)>0


def test_qp_envelope_contact_gradient_matches_reoptimized_force_problem():
    points=np.array([[[-.05,.01,.01],[.05,-.01,.01]]])
    normals=np.array([[[-1.,0.,0.],[1.,0.,0.]]]);mu=np.full((1,2,2),.6)
    active=np.ones((1,2),bool);target=np.array([[.3,.2,2.,.1,-.2,.05]])
    def solve(p):
        return wrench_qp(p,normals,mu,active,target,center=np.zeros(3),radius=.1,
                          force_limit=1.,iterations=256)
    _,gradient,_=solve(points)
    numeric=np.zeros((2,3));eps=1e-5
    for contact in range(2):
        for axis in range(3):
            plus,minus=points.copy(),points.copy()
            plus[0,contact,axis]+=eps;minus[0,contact,axis]-=eps
            numeric[contact,axis]=(solve(plus)[0][0]-solve(minus)[0][0])/(2*eps)
    np.testing.assert_allclose(gradient[0],numeric,atol=1e-5,rtol=1e-5)


def test_warp_patch_projection_matches_numpy_and_reuses_geometry():
    from retarget.contact_projection import WarpPatchProjector
    s=surface()
    # A second disconnected sheet is covered by the separate topology test;
    # here test boundary fallback and inactive/off-surface guidance as well.
    templates=np.array([[[0.,0.,0.],[.02,0.,0.],[0.,0.,.03]]])
    weights=np.array([[1.,0.,1.]])
    proposals=np.array([[[[.001,.001,.003],[.08,0.,0.],[0.,0.,0.]],
                          [[.09,.09,.01],[.04,0.,.01],[0.,.01,0.]]]])
    expected,expected_normals,movable=SampledComFreeLM(s)._project(proposals,templates,weights)
    projector=WarpPatchProjector(s,.005,'cpu')
    actual,normals,actual_movable=projector.project(proposals,templates,weights)
    np.testing.assert_allclose(actual,expected,atol=1e-8)
    np.testing.assert_allclose(normals,expected_normals,atol=1e-7)
    np.testing.assert_array_equal(actual_movable,movable)
    geometry=projector.geometry
    projector.project(proposals,templates,weights)
    assert projector.geometry is geometry


def test_lm_contact_wrench_and_inertia_use_same_origin_with_offset_com():
    from retarget.contact_solvers import skew
    center=np.array([.04,0.,0.]);mass=.2;cross=skew(center)
    spatial=np.block([[mass*np.eye(3),-mass*cross],
                      [mass*cross,np.eye(3)*.0001-mass*cross@cross]])
    gravity=np.array([0.,0.,-mass*9.81])
    external=np.r_[gravity,np.cross(center,gravity)][None]
    _,_,before=batched_force_lm(center[None,None],np.array([[[0.,0.,-1.]]]),
        np.full((1,1,2),.6),np.ones((1,1),bool),inverse=np.linalg.inv(spatial)[None],
        velocity=np.zeros((1,6)),external=external,goal=np.zeros((1,6)),dt=np.array([.01]),
        center=center,radius=.1,iterations=1)
    # Applying the gravity-balancing force at the COM cancels both force and
    # moment at the body origin. Only the explicit force regularizer remains.
    np.testing.assert_allclose(before,[1e-6*(mass*9.81)**2],atol=1e-12)


def test_qp_required_wrench_is_transformed_from_body_origin_to_com(monkeypatch):
    from retarget import contact_solvers as module
    center=np.array([.02,0.,0.]);gravity=np.array([0.,0.,-1.962])
    captured=[];original=module.wrench_qp
    def record(*args,**kwargs):
        captured.append(args[4].copy())
        return original(*args,**kwargs)
    monkeypatch.setattr(module,'wrench_qp',record)
    points=np.array([[0.,0.,0.],[.02,0.,0.]])
    QPContactSolver(surface(),qp_iterations=16).solve(points,points,points,np.ones(2),
        np.full((2,2),.6),center=center,object_radius=.1,
        context={'external':np.r_[gravity,np.cross(center,gravity)]})
    np.testing.assert_allclose(captured[0],[[0.,0.,1.962,0.,0.,0.]],atol=1e-12)


@pytest.mark.parametrize('solver_type',[SampledComFreeLM,QPContactSolver])
def test_feasible_incumbent_skips_search_despite_hand_geometry_cost(solver_type,monkeypatch):
    underside=ObjectMeshSurface([[-.1,-.1,0],[.1,-.1,0],[.1,.1,0],[-.1,.1,0]],[[0,2,1],[0,3,2]])
    # One upward contact through the COM balances gravity within the cone.
    initial=np.array([[.001,0.,0.]])
    solver=solver_type(underside,sweeps=3)
    projections=[];original=solver._project
    def record(proposals,*args):
        projections.append(proposals.shape)
        return original(proposals,*args)
    monkeypatch.setattr(solver,'_project',record)
    points,diag=solver.solve(initial,initial,initial+[.1,0.,0.],np.ones(1),np.full((1,2),.6),
                             center=initial[0],object_radius=.1,
                             context={'external':[0.,0.,-1.962,0.,.001962,0.]})
    np.testing.assert_allclose(points,initial,atol=1e-8)
    assert diag['held_feasible'] and diag['feasible'] and not diag['searched']
    assert len(projections)==1 and projections[0][1]==1


def test_infeasible_lm_search_respects_material_improvement_threshold():
    initial=np.array([[0.,0.,0.],[.02,0.,0.]])
    args=(initial,initial,initial+[.01,0.,0.],np.ones(2),np.full((2,2),.6))
    # Downward-only normals cannot balance gravity. A lower geometry cost
    # without a better physical witness must not move the contacts.
    solver=SampledComFreeLM(surface(),motion_tolerance=0.,min_improvement=.99)
    points,diag=solver.solve(*args,center=np.zeros(3),object_radius=.1)
    np.testing.assert_allclose(points,initial,atol=1e-8)
    assert diag['searched'] and not diag['held_feasible']
