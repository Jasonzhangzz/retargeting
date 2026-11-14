"""Strict SDK label quality predicates, independent of model predictions."""
import numpy as np


def sdk_quality_mask(ik_confidence, skeleton_confidence, position_error, rotation_error):
    arrays = [np.asarray(v) for v in (ik_confidence, skeleton_confidence, position_error, rotation_error)]
    n = len(arrays[0])
    if any(v.shape != (n, width) for v, width in zip(arrays, (5, 21, 5, 5))):
        raise ValueError('Expected SDK IK[5], skeleton[21], FK position[5], rotation[5] per frame')
    ik, sk, pe, re = arrays
    return (np.logical_and.reduce([np.isfinite(v).all(1) for v in arrays])
            & ((ik >= .8) & (ik <= 1.00001)).all(1)
            & ((sk >= .8) & (sk <= 1.00001)).all(1)
            & ((pe >= 0) & (pe <= .01)).all(1)
            & ((re >= 0) & (re <= .7)).all(1))


def passing_runs(mask, minimum=64):
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1 or minimum < 1:
        raise ValueError('Expected one validity flag per frame')
    changes = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return [(int(a), int(b)) for a, b in zip(changes[::2], changes[1::2]) if b-a >= minimum]
