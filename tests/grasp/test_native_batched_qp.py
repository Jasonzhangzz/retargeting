import numpy as np
import pytest
import torch
from retarget.env.grasp.native_batched_qp import NativeBatchedQP


@pytest.mark.parametrize('mode,minimum',[('optimization',.01),('filter',0.)])
def test_two_native_constraint_sets_are_retained(mode,minimum):
    qp=NativeBatchedQP(4,'cpu',mode=mode)
    np.testing.assert_allclose(qp.h[:24],-minimum,atol=1e-9)
    np.testing.assert_allclose(qp.h[24:28],1.)
    assert qp.h[-1]==-1
    np.testing.assert_allclose(qp.G[:24],-np.eye(24))
    np.testing.assert_allclose(qp.G[-1],-1.)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('mode',['optimization','filter'])
def test_batched_qp_matches_cpu_objective_constraints_wrench_and_gradient(mode,monkeypatch):
    torch.set_num_threads(1)
    qp=NativeBatchedQP(4,'cuda:0',mode=mode)
    source=qp.source.qp
    gen=torch.Generator().manual_seed(20261104)
    points=(torch.randn((4,4,3),generator=gen)*.04).cuda().requires_grad_(True)
    normal=torch.nn.functional.normalize(torch.randn((4,4,3),generator=gen),dim=-1).cuda()
    normal[2]=torch.tensor([1.,0.,0.],device='cuda')
    external=torch.zeros((4,6),device='cuda');external[0]=torch.tensor([.02,-.03,.01,.001,.002,-.001],device='cuda')
    center=torch.zeros((4,3),device='cuda');solved=qp.solve(points,normal,external,center)
    assert solved['valid'].all(),{k:v.tolist() for k,v in solved.items() if k!='alpha'}
    H,g,M=qp.matrices(points,normal,external,center)
    wrench,error=qp.contact_wrenches(points,normal,external,center,solved['alpha'])
    loss=error.square();gradient=torch.autograd.grad(loss.sum(),points)[0]
    for i in range(4):
        p=points[i].detach().cpu().numpy();n=normal[i].cpu().numpy();ext=external[i].cpu().numpy()
        if mode=='optimization':
            ref=source.ContactQPTorch([.1,0.]);captured=[];original=source.solve_qp_numpy_interface
            def capture(*args):
                result=original(*args);captured.append((args,result));return result
            with monkeypatch.context() as patch:
                patch.setattr(source,'solve_qp_numpy_interface',capture)
                x=torch.tensor(p,dtype=torch.float64,requires_grad=True);e=ref.solve(x,torch.tensor(n,dtype=torch.float64),ext,np.zeros(3))
                expected_gradient=torch.autograd.grad(e.square(),x)[0]
            args,alpha=captured[0];grasp,gravity,G,E,h,_=args
            expected=(grasp@E).transpose(1,0,2).reshape(6,-1)
            expected_wrench=expected@alpha+gravity
            torch.testing.assert_close(gradient[i].cpu().double(),expected_gradient,atol=1e-4,rtol=1e-3)
        else:
            ref=source.ContactQP([.6,0.]);expected_contacts,e=ref.solve(p,n,ext,np.zeros(3));expected_wrench=expected_contacts.sum(0)+ext
            rot=source.np_normal_to_rot(n);grasp=np.zeros((4,6,6));grasp[:,:3,:3]=rot;grasp[:,3:,3:]=rot
            grasp[:,3:,:3]=np.cross((p-np.zeros(3))[:,:,None],rot,axisa=1,axisb=1,axisc=1)
            G,h,E=ref.build_constraint();expected=(grasp@E).transpose(1,0,2).reshape(6,-1)
        np.testing.assert_allclose(M[i].detach().cpu(),expected,atol=1e-7,rtol=1e-5)
        np.testing.assert_allclose(H[i].detach().cpu(),expected.T@expected,atol=1e-6,rtol=1e-5)
        np.testing.assert_allclose(g[i].detach().cpu(),ext@expected,atol=1e-7,rtol=1e-5)
        np.testing.assert_allclose(qp.G.cpu(),G,atol=1e-8);np.testing.assert_allclose(qp.h.cpu(),h,atol=1e-8)
        np.testing.assert_allclose((wrench[i].sum(0)+external[i]).detach().cpu(),expected_wrench,atol=1e-5,rtol=1e-4)
        expected_error=float(e.detach()) if isinstance(e,torch.Tensor) else float(e)
        np.testing.assert_allclose(loss[i].detach().cpu(),expected_error**2,atol=1e-5,rtol=1e-4)
