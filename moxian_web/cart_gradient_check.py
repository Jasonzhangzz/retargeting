"""Independent finite-difference check with numerical stability across scales."""
import torch


def directional_gradient_check(energy, q, direction, *, tolerance=.08):
    """Select a stable FD pair without using autograd, then compare derivatives.

    energy returns one scalar per pose. Float64 reduction limits cancellation;
    geometry retains its production precision. No row or failing gradient is
    discarded. The finite-difference stability tolerance is separate from the
    derivative acceptance threshold.
    """
    value=energy(q)
    gradient=torch.autograd.grad(value.sum(),q)[0]
    analytic=(gradient.double()*direction.double()).sum(-1)
    steps=(.003,.001,.0003,.0001)
    with torch.no_grad():
        slopes=torch.stack([(energy(q+eps*direction)-energy(q-eps*direction))/(2*eps) for eps in steps])
    disagreement=(slopes[:-1]-slopes[1:]).abs()/(slopes[:-1].abs()+slopes[1:].abs()+1e-3)
    selected=disagreement.argmin(0)  # Depends only on numerical samples.
    ids=torch.arange(len(q),device=q.device)
    stability=disagreement[selected,ids];numerical=slopes[selected+1,ids]
    error=(numerical-analytic).abs()/(numerical.abs()+analytic.abs()+1e-3)
    result=dict(steps=list(steps),finite_difference=slopes.detach().cpu().tolist(),
        analytic=analytic.detach().cpu().tolist(),selected_pair=selected.cpu().tolist(),
        numerical_stability=stability.cpu().tolist(),relative_errors=error.detach().cpu().tolist(),
        relative_error=float(error.max().detach()),passed=bool(torch.isfinite(error).all() and (stability<.02).all() and (error<tolerance).all()),
        stability_tolerance=.02,gradient_tolerance=tolerance)
    return result
