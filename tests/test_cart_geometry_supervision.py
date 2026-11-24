"""Regression checks for native derivative labels and differentiable geometry."""
import torch
from torch import nn

from moxian_web.cart_contact_field import ContactTeacher,PadPotentialField,projection_jvp
from moxian_web.cart_geometry_audit import directional_check
from moxian_web.cart_q_collision import QCollisionLearner,QCollisionTeacher


def test_projection_jvp_is_not_transposed_for_legacy_vector_fields():
    matrix=nn.Parameter(torch.tensor([[1.,2.,0.],[0.,1.,3.],[4.,0.,1.]],dtype=torch.double))
    points=torch.tensor([[.01,.02,.03]],dtype=torch.double,requires_grad=True)
    probe=torch.tensor([[.2,.3,.4]],dtype=torch.double)
    hit=points@matrix.T
    jvp=projection_jvp(hit,points,probe,create_graph=True)
    torch.testing.assert_close(jvp,probe@matrix.T)
    assert not torch.allclose(jvp,probe@matrix)
    jvp.square().sum().backward()
    assert matrix.grad is not None and torch.isfinite(matrix.grad).all()


def test_scalar_potential_retains_higher_order_input_derivatives():
    torch.manual_seed(7)
    model=PadPotentialField([[0.,0.,0.]]*21,width=8,depth=2).double()
    with torch.no_grad():model.decoders['0'][-1].weight[0].normal_(std=.01)
    links=torch.zeros(1,dtype=torch.long)
    points=torch.tensor([[.001,.002,.003]],dtype=torch.double,requires_grad=True)
    assert torch.autograd.gradcheck(lambda x:model(x,links)[0],(points,),fast_mode=True)
    assert torch.autograd.gradgradcheck(lambda x:model(x,links)[0],(points,),fast_mode=True)
    hit,_=model(points,links)
    probe=torch.tensor([[.2,.3,.4]],dtype=torch.double)
    actual=projection_jvp(hit,points,probe,potential=True)
    _,expected=torch.autograd.functional.jvp(lambda x:model(x,links)[0],points,probe)
    torch.testing.assert_close(actual,expected)


class SwitchingTeacher(ContactTeacher):
    def __init__(self):pass
    def label(self,points,directions,links,palmar):
        hit=torch.zeros_like(points);hit[:,0]=torch.where(points[:,0]<0,-.001,.001)
        return hit,torch.ones_like(points),torch.ones(len(points),dtype=torch.bool)
    def projection_derivative(self,points,links,palmar,probes):
        return torch.zeros_like(points),torch.ones(len(points),dtype=torch.bool)


def test_derivative_filter_detects_projection_switches_without_deleting_values():
    teacher=SwitchingTeacher();points=torch.tensor([[-1.e-6,0.,0.],[.01,0.,0.]])
    links=torch.zeros(2,dtype=torch.long);flags=torch.ones(2,dtype=torch.bool)
    probe=torch.tensor([[1.,0.,0.],[1.,0.,0.]])
    _,valid,error=teacher.verified_projection_derivative(points,links,flags,probe)
    assert valid.tolist()==[False,True] and error[0]>1
    assert teacher.label(points,torch.zeros_like(points),links,flags)[2].all()


def test_independent_audit_does_not_hide_wrong_gradients_as_unstable():
    report,stable=directional_check(torch.tensor([2.]),torch.tensor([1.]),torch.tensor([1.]),atol=.01,rtol=.01)
    assert stable.all() and report['correct_on_stable_fraction']==0


def test_collision_strata_cover_safe_boundary_shallow_and_deep_states():
    depth=torch.tensor([0.,.2,2.,7.]);energy=torch.tensor([0.,2.,3.,4.])
    ids=QCollisionLearner.stratified_indices(depth,energy,64)
    assert torch.bincount(ids,minlength=4).tolist()==[16,16,16,16]
    ids=QCollisionLearner.stratified_indices(torch.zeros(4),torch.zeros(4),32)
    assert len(ids)==32 and ids.min()>=0 and ids.max()<4


def test_depth_teacher_and_student_use_the_same_smooth_aggregation():
    signed=torch.tensor([[-.001,-.001,.003]],dtype=torch.double,requires_grad=True)
    valid=torch.ones_like(signed,dtype=torch.bool);groups=torch.tensor([0,0,1])
    depths=QCollisionTeacher.smooth_group_depths(signed,valid,groups)
    output=.05*torch.logsumexp(depths/.05,-1)
    assert output.item()>=1.
    assert torch.autograd.gradcheck(lambda x:.05*torch.logsumexp(QCollisionTeacher.smooth_group_depths(x,valid,groups)/.05,-1),(signed,))
