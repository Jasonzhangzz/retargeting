"""The live Warp Lambda force solver checked against independent NumPy math."""
from types import SimpleNamespace
import numpy as np
import pytest
import warp as wp
from retarget.contact_exploration import ObjectMeshSurface
from retarget.contact_solvers import (SampledComFreeLM,batched_force_lm,
    grasp_matrix,contact_basis,comfree_velocity,motion_residual)
from retarget.contact_lm_warp import DeviceLambdaLM


@pytest.mark.parametrize('hold',[False,True])
def test_device_lambda_cost_forces_and_selection_match_reference(hold):
    device=wp.get_device('cpu')
    surface=ObjectMeshSurface([[-.1,-.1,0],[.1,-.1,0],[.1,.1,0],[-.1,.1,0]],[[0,2,1],[0,3,2]])
    explorer=SampledComFreeLM(surface,samples=4,lm_iterations=3,device=device,
                               motion_tolerance=1. if hold else 0.)
    points=np.array([[[[0.,0.,0.],[.02,0.,0.]]]],np.float32)
    hand=points+np.array([.001,0.,.002],np.float32)
    weights=np.ones((1,2),np.float32);friction=np.full((1,1,2,2),.6,np.float32)
    velocity=np.zeros((1,1,6),np.float32);goal=np.array([[[0.,0.,.001,0.,.002,0.]]],np.float32)
    array=lambda value,dtype:wp.array(value,dtype=dtype,device=device)
    batch=SimpleNamespace(c=array(points,wp.vec3),x=array(hand,wp.vec3),dual=array(np.zeros_like(points),wp.vec3),
        weights=array(weights,float),friction=array(friction,wp.vec2),
        rotation=array(np.eye(3)[None,None],wp.mat33),position=array(np.zeros((1,1,3)),wp.vec3),
        velocity=array(velocity,wp.spatial_vector),goal=array(goal,wp.spatial_vector))
    inverse=np.diag([5.]*3+[100.]*3)
    physics=dict(inverse=inverse,com=np.zeros(3),gravity=np.array([0.,0.,-9.81]),
                 floor=None,support_vertices=np.array([[0.,0.,0.]]))
    solver=DeviceLambdaLM(explorer,groups=1,contacts=2,worlds=1,physics=physics,device=device)
    solver.bind(batch,points.reshape(1,2,3),weights,.1)
    solver.job.costs.fill_(12345.)
    selected=solver.solve().numpy()
    if hold:
        np.testing.assert_allclose(selected,points.reshape(2,3),atol=1e-8)
        np.testing.assert_array_equal(solver.job.costs.numpy()[1:],12345.)
        assert solver.job.statistics.numpy()[0,7] == 1
        assert solver.job.statistics.numpy()[0,8] == 0
        return
    candidates=solver.job.points.numpy().reshape(4,2,3)
    normals=solver.job.normals.numpy().reshape(4,2,3)
    force,cost,_=batched_force_lm(candidates,normals,np.repeat(friction.reshape(1,2,2),4,axis=0),
        np.ones((4,2),bool),inverse=np.repeat(inverse[None],4,axis=0),
        velocity=np.repeat(velocity.reshape(1,6),4,axis=0),external=np.tile([0,0,-1.962,0,0,0],(4,1)),
        goal=np.repeat(goal.reshape(1,6),4,axis=0),dt=np.full(4,.01),center=np.zeros(3),radius=.1,
        iterations=3,weight=(.1/.005)**2)
    geometry=explorer._geometry_cost(candidates[None],points.reshape(1,2,3),hand.reshape(1,2,3),weights)[0]
    np.testing.assert_allclose(solver.job.costs.numpy(),cost+geometry,rtol=3e-5,atol=1e-6)
    np.testing.assert_allclose(solver.job.forces.numpy().reshape(4,2,3),force,rtol=3e-4,atol=1e-4)
    matrix=grasp_matrix(candidates,contact_basis(normals),np.zeros(3))
    predicted,jacobian=comfree_velocity(force.reshape(4,-1),matrix,np.repeat(inverse[None],4,axis=0),
        np.repeat(velocity.reshape(1,6),4,axis=0),np.tile([0,0,-1.962,0,0,0],(4,1)),
        np.full(4,.01),np.zeros((4,6)),np.zeros(4))
    residual,_=motion_residual(predicted,jacobian,np.repeat(goal.reshape(1,6),4,axis=0),np.full(4,.01),.1)
    error=np.linalg.norm(residual,axis=1)*.1
    np.testing.assert_allclose(solver.job.motion_errors.numpy(),error,atol=1e-8,rtol=1e-5)
    eligible=error<error[0]-max(1e-9,.01*error[0])
    eligible[0]=True
    winner=int(np.argmin(np.where(eligible,error,np.inf)))
    np.testing.assert_array_equal(selected,candidates[winner])
    np.testing.assert_allclose(candidates[0],points.reshape(2,3),atol=1e-8)
    assert solver.job.motion_errors.numpy()[int(solver.job.statistics.numpy()[0,2])] <= error[0]+1e-8
    # Captured warm-up / repeated calls may not mutate the input incumbent.
    np.testing.assert_array_equal(batch.c.numpy(),points)
    again=solver.solve().numpy()
    np.testing.assert_array_equal(again,selected)

    # A tiny physical improvement does not justify moving an infeasible incumbent.
    solver.job.costs.assign(np.array([1.,.995,2.,2.],np.float32))
    solver.job.motion_errors.assign(np.array([.01,.00995,.02,.02],np.float32))
    wp.launch(solver.check_kernel,1,inputs=[solver.job],device=device)
    wp.launch(solver.select_kernel,1,inputs=[solver.job],device=device)
    assert solver.job.statistics.numpy()[0,2] == 0
    # Reaching feasibility takes precedence over the soft geometry penalty.
    solver.job.motion_errors.assign(np.array([.01,0.,.02,.02],np.float32))
    solver.job.costs.assign(np.array([1.,2.,3.,3.],np.float32))
    wp.launch(solver.select_kernel,1,inputs=[solver.job],device=device)
    assert solver.job.statistics.numpy()[0,2] == 1
