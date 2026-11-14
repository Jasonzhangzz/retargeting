"""Public wrist-coordinate API. No past joint angles or serial IK calls."""
import numpy as np
import torch
from .features import SIDES, wrist_emf_features
from .metrics import autocast_context
from .models import sample_flow


@torch.no_grad()
def predict_wrist(model, normalizer, variant, position, orientation_xyzw, side='right', lengths=None,
                  flow_steps=8, seed=42, shared_noise=True, return_radians=True, bf16=True):
    p = np.asarray(position); q = np.asarray(orientation_xyzw)
    unbatched = p.ndim == 3
    if unbatched: p, q = p[None], q[None]
    if p.ndim != 4 or p.shape[-2:] != (5, 3) or q.shape != p.shape[:-1]+(4,):
        raise ValueError('Expected wrist EMF [T,5,3/4] or [B,T,5,3/4]')
    b, t = p.shape[:2]
    if lengths is None: lengths = [t]*b
    lengths = np.asarray(lengths, dtype=np.int64)
    if lengths.shape != (b,) or np.any(lengths < 1) or np.any(lengths > t):
        raise ValueError('Invalid sequence lengths')
    if isinstance(side, str): sides = [SIDES.index(side)]*b
    else:
        sides = [SIDES.index(v) if isinstance(v, str) else int(v) for v in side]
    if len(sides) != b or any(s not in (0, 1) for s in sides):
        raise ValueError('Expected side right/left or side indices 0/1 for each sequence')
    features = np.zeros((b, t, 5, 9), np.float32)
    for i, n in enumerate(lengths): features[i, :n] = wrist_emf_features(p[i, :n], q[i, :n])
    device = next(model.parameters()).device
    x = torch.from_numpy(features).to(device); side_tensor = torch.tensor(sides, device=device)
    length_tensor = torch.from_numpy(lengths).to(device)
    model.eval()
    with autocast_context(device, bf16):
        if variant == 'frame_flow':
            predicted = sample_flow(model, x, side_tensor, length_tensor, steps=flow_steps, seed=seed, shared_noise=shared_noise)
        else:
            predicted = model(x, side_tensor, length_tensor)
    predicted = predicted.float()
    valid = torch.arange(t, device=device)[None] < length_tensor[:, None]
    result = dict(q_normalized=predicted.masked_fill(~valid[..., None], 0.).cpu().numpy(), lengths=lengths)
    if return_radians:
        angles = normalizer.denormalize(predicted, side_tensor)
        result['q_rad'] = angles.masked_fill(~valid[..., None], 0.).cpu().numpy()
    if unbatched:
        for key in ('q_normalized', 'q_rad'):
            if key in result: result[key] = result[key][0]
    return result
