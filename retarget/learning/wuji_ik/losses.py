"""Temporal regularization with endpoint ownership for exact frame budgets."""
import torch


def acceleration_mask(batch):
    """Assign a contiguous triple to its last loss-owned frame, once per epoch.

    Prior frames may be context outside the owned range; masking all three by
    `valid` would incorrectly drop triples at microbatch/rank budget boundaries.
    """
    frames = batch['frames']
    indices = torch.arange(frames.shape[1], device=frames.device)
    present = indices[None, :] < batch['lengths'][:, None]
    return (batch['valid'][:, 2:] & present[:, :-2] & present[:, 1:-1]
            & present[:, 2:] & (frames[:, :-2] >= 0) & (frames[:, 1:-1]-frames[:, :-2] == 1)
            & (frames[:, 2:]-frames[:, 1:-1] == 1))


def acceleration_sum(prediction, mask):
    q = prediction.float()
    acceleration = q[:, 2:]-2*q[:, 1:-1]+q[:, :-2]
    return (acceleration.square()*mask[..., None]).sum()
