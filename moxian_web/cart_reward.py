"""Dimensionless exponential policy costs; geometry regression stays metric."""
import torch


def exp_cost(squared_error, scale=1., broad_ratio=1.):
    """1-exp(-e²/s²), optionally averaged with a broader recovery kernel.

    expm1 retains accuracy near zero. The broad kernel supplies useful gradients
    outside the narrow contact tolerance instead of saturating both kernels.
    """
    energy=squared_error.clamp_min(0)/(scale*scale)
    narrow=-torch.expm1(-energy)
    if broad_ratio==1.:return narrow
    return .5*(narrow-torch.expm1(-energy/(broad_ratio*broad_ratio)))
