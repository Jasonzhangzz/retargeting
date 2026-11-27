import torch
from moxian_web.cart_gradient_check import directional_gradient_check


def test_float32_quadratic_multiscale_and_cancellation():
    q=torch.tensor([[.2,.4],[.7,.1]],requires_grad=True);d=torch.tensor([[.1,-.2],[.2,.3]])
    result=directional_gradient_check(lambda x:(x.double().square()*100).sum(-1),q,d)
    assert result['passed'] and result['relative_error']<1e-3


def test_incorrect_backward_is_rejected_at_every_scale():
    class Wrong(torch.autograd.Function):
        @staticmethod
        def forward(ctx,x):ctx.save_for_backward(x);return x.square()
        @staticmethod
        def backward(ctx,g):return ctx.saved_tensors[0]*g  # Missing factor two.
    q=torch.tensor([[.2,.4],[.7,.1]],requires_grad=True);d=torch.ones_like(q)*.1
    result=directional_gradient_check(lambda x:Wrong.apply(x).double().sum(-1),q,d)
    assert not result['passed'] and result['relative_error']>.3
