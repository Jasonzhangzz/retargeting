"""Local task-wrench repair, without imposing full force closure."""
from types import SimpleNamespace
import numpy as np
import warp as wp
import pytest
from scipy.optimize import linprog
from retarget.contact_exploration import ObjectMeshSurface
from retarget.contact_solvers import SampledComFreeLM,QPContactSolver,contact_basis,grasp_matrix
from retarget.contact_lm_warp import DeviceLambdaLM


def problem():
    vertices=[[-.05,-.1,-.1],[-.05,.1,-.1],[-.05,.1,.1],[-.05,-.1,.1],
              [.05,-.1,-.1],[.05,.1,-.1],[.05,.1,.1],[.05,-.1,.1]]
    surface=ObjectMeshSurface(vertices,[[0,2,1],[0,3,2],[4,5,6],[4,6,7]])
    points=np.array([[-.05,0,0],[.05,0,0],[-.05,0,0],[.05,0,0]],np.float32)
    hand=points.copy();hand[2:,1]=.004
    normals=np.tile([[-1.,0,0],[1.,0,0]],(2,1))
    return surface,points,hand,normals


def task_witness(points,normals,torque):
    b=contact_basis(normals);normal,t1,t2=b[...,0],b[...,1],b[...,2]
    primitives=np.stack((normal+.6*t1,normal-.6*t1,normal+.6*t2,normal-.6*t2),axis=-1)
    G=grasp_matrix(points[None],primitives[None],np.zeros(3))[0]
    # Nonnegative friction edge forces, <=10 N pressure per contact.
    return linprog(np.zeros(16),A_eq=G,b_eq=[0,0,1.962,torque,0,0],
        A_ub=np.kron(np.eye(4),np.ones((1,4))),b_ub=np.full(4,10./np.sqrt(1+.6**2)),bounds=(0,None),method='highs')


@pytest.mark.parametrize('spider_pose',[False,True])
def test_default_device_lm_repairs_task_infeasible_points_within_radius(spider_pose):
    surface,initial,hand,normals=problem();torque=.003
    assert not task_witness(initial,normals,torque).success
    assert task_witness(hand,normals,torque).success
    device='cpu';array=lambda x,t:wp.array(x,dtype=t,device=device)
    inverse=np.diag([5.]*3+[1e4]*3)
    goal=np.array([0,0,0,2*np.arctan(.5*.01**2*1e4*torque),0,0])
    batch=SimpleNamespace(c=array(initial[None,None],wp.vec3),x=array(hand[None,None],wp.vec3),
        dual=array(np.zeros((1,1,4,3)),wp.vec3),weights=array(np.ones((1,4)),float),
        friction=array(np.full((1,1,4,2),.6),wp.vec2),rotation=array(np.eye(3)[None,None],wp.mat33),
        position=array(np.zeros((1,1,3)),wp.vec3),velocity=array(np.zeros((1,1,6)),wp.spatial_vector),
        goal=array(goal[None,None],wp.spatial_vector))
    explorer=SampledComFreeLM(surface,device=device)
    physics=dict(inverse=inverse,com=np.zeros(3),gravity=np.array([0,0,-9.81]),floor=None,support_vertices=np.zeros((1,3)))
    if spider_pose:physics['spider_pose_weights']=[1.,1.,1.,.3,.3,.3]
    solver=DeviceLambdaLM(explorer,groups=1,contacts=4,worlds=1,device=wp.get_device(device),physics=physics)
    solver.bind(batch,initial[None],np.ones((1,4)),.1)
    result=solver.solve().numpy();stats=solver.job.statistics.numpy()[0]
    assert stats[7]==0 and stats[8]==1 and stats[10]==1
    assert np.linalg.norm(result-initial,axis=1).max()<=.005+1e-8
    assert task_witness(result,normals,torque).success
    # Once this task-feasible combination is found, its next check holds it.
    batch.c.assign(result[None,None])
    again=solver.solve().numpy()
    np.testing.assert_allclose(again,result,atol=1e-8)
    assert solver.job.statistics.numpy()[0,7]==1


def test_qp_reduces_task_error_even_when_geometry_cost_increases():
    surface,initial,hand,normals=problem();torque=.01
    assert not task_witness(initial,normals,torque).success
    assert task_witness(hand,normals,torque).success
    solver=QPContactSolver(surface)
    solver.inverse=np.diag([5.]*3+[1e4]*3)
    goal=np.array([0,0,0,2*np.arctan(.5*.01**2*1e4*torque),0,0])
    result,diag=solver.solve(initial,initial,hand,np.ones(4),np.full((4,2),.6),
        center=np.zeros(3),object_radius=.1,context={'goal':goal})
    assert not diag['held_feasible'] and diag['feasible']
    assert diag['wrench_residual']<diag['initial_wrench_residual']
    assert diag['final_value']>diag['initial_value']  # Soft geometry must not veto repair.
    assert np.linalg.norm(result-initial,axis=1).max()<=.005+1e-8
