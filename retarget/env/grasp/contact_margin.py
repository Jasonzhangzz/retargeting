"""GPU contact eligibility using MuJoCo 3.3.3 geom margin/gap semantics."""
from dataclasses import dataclass
import torch


@dataclass
class ContactMarginSelection:
    margin: torch.Tensor
    gap: torch.Tensor
    detected: torch.Tensor
    active: torch.Tensor
    valid: torch.Tensor


def geom_contact_margins(distance, hand_margin, object_margin, *, hand_gap=0.,
                         object_gap=0., pair_margin=None, pair_gap=None):
    """Select force-bearing slots; distance is signed and measured in metres.

    Implicit pairs use max, not sum. Explicit pair parameters override geom
    parameters. A detected contact in the gap contributes no force. This does
    not move the physical surfaces or impose a penetration acceptance limit.
    """
    def value(x):
        return torch.as_tensor(x, device=distance.device, dtype=distance.dtype)
    margin = (torch.maximum(value(hand_margin), value(object_margin))
              if pair_margin is None else value(pair_margin))
    gap = (torch.maximum(value(hand_gap), value(object_gap))
           if pair_gap is None else value(pair_gap))
    distance, margin, gap = torch.broadcast_tensors(distance, margin, gap)
    valid = torch.isfinite(distance) & torch.isfinite(margin) & torch.isfinite(gap)
    detected = valid & (distance < margin)
    active = detected & (distance < margin - gap)
    return ContactMarginSelection(margin, gap, detected, active, valid)
