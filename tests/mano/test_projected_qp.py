import numpy as np
import torch
from retarget.env.grasp.reference.wrench_qp import wrench_qp as reference,contact_basis,grasp_matrix
from retarget.env.grasp.projected_qp import wrench_qp,ProjectedQPClosure,project_pressures
from retarget.env.grasp.closure_types import ClosureSettings


def test_original_qp_equivalence():
    rng=np.random.default_rng(8)
    p=rng.normal(size=(3,4,3));n=rng.normal(size=p.shape);n/=np.linalg.norm(n,axis=-1,keepdims=True)
    basis=contact_basis(n);a,b,c=[basis[...,i] for i in range(3)]
    primitive=np.stack((a+.6*b,a-.6*b,a+.6*c,a-.6*c),-1)
    matrix=grasp_matrix(p,primitive,np.zeros(3),1.)
    target=rng.normal(size=(3,6))
    cost,_,info=reference(p,n,np.full((3,4,2),.6),np.ones((3,4),bool),target,center=np.zeros(3),radius=1.)
    x,res,actual=wrench_qp(torch.tensor(matrix),torch.tensor(target),4)
    np.testing.assert_allclose(x,info['pressures'],atol=1e-10)
    np.testing.assert_allclose(actual,cost,atol=1e-10)


def test_closure_and_one_sided_rejection():
    solver=ProjectedQPClosure(ClosureSettings(cuda_graph=False),center=[0,0,0],radius=1,object_mass=.1,object_inertia=np.eye(3))
    n=torch.nn.functional.normalize(torch.tensor([[1.,1.,1.],[1.,-1.,-1.],[-1.,1.,-1.],[-1.,-1.,1.]],dtype=torch.float64),dim=-1)
    assert solver.solve(n[None],n[None]).certified.item()
    assert not solver.solve(n[None],n[:1].expand_as(n)[None]).certified.item()


def test_cuda_projection_and_graph_witnesses():
    import pytest
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    value=torch.randn(2,13,5,4)*4-3
    caps=torch.full((2,13,5),10.);minimum=torch.ones(2,13)
    expected=project_pressures(value,caps,minimum)
    actual=project_pressures(value.cuda(),caps.cuda(),minimum.cuda()).cpu()
    torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-5)
    solver=ProjectedQPClosure(ClosureSettings(),center=[0,0,0],radius=1,object_mass=.1,object_inertia=np.eye(3))
    n=torch.nn.functional.normalize(torch.randn(2,8,3,device='cuda'),dim=-1)
    for normals in (n,-n):
        result=solver.solve(n,normals)
        eager=solver._compute(n,normals,torch.zeros(2,3,device='cuda'),torch.ones(2,device='cuda'))
        torch.testing.assert_close(result.cost,eager[0])
        assert torch.equal(result.certified,eager[4])
