"""Pinned SDK 2026.8.31 default offline human-hand solver parameters.

Recovered from first-call constructor state, before native optimization, for
both CQQ URDFs. No SDK import or frame-output lookup occurs in this module.
See ../human-model-parameters.json and ../human-model-parameter-evidence.txt.
"""
from copy import deepcopy

DEFAULT_PARAMETERS = {
    'w_pos': 0.3, 'w_rot': 5.729577951308232, 'error_gain': 0.5,
    'damping_scale': 0.0001, 'damping_floor': 1e-6,
    'max_step_scale': 0.008333333333333333, 'retry_base': 3.141592653589793,
    'bound_scale': 0.5, 'posture_weight': 1.0, 'thumb_posture_scale': 0.01,
    'history_weight': 20.0, 'error_to_m': 1.0, 'target_scale': 1.0,
}
PARAMETER_OFFSETS = dict(zip(range(0xe8, 0x150, 8), DEFAULT_PARAMETERS))

_THUMB_BOUNDS = {
    'left': [[-0.7853981633974483, 0.7853981633974483],
             [-0.8726646259971648, 0.6108652381980153],
             [-1.1344640137963142, 1.0471975511965976],
             [-1.7453292519943295, 0.2617993877991494],
             [-1.6580627893946132, 0.2617993877991494]],
    'right': [[-0.7853981633974483, 0.7853981633974483],
              [-0.6108652381980153, 0.8726646259971648],
              [-1.1344640137963142, 1.0471975511965976],
              [-0.2617993877991494, 1.7453292519943295],
              [-0.2617993877991494, 1.6580627893946132]],
}
_OTHER_BOUNDS = {
    'left': [[-1.562069680534925, 0.17453292519943295],
             [-0.6108652381980153, 0.7853981633974483],
             [-1.562069680534925, -0.008726646259971648],
             [-1.562069680534925, -0.008726646259971648]],
    'right': [[-0.17453292519943295, 1.562069680534925],
              [-0.6108652381980153, 0.7853981633974483],
              [0.008726646259971648, 1.562069680534925],
              [0.008726646259971648, 1.562069680534925]],
}

def default_parameters(side='right'):
    """Return an independent mutable dict, five q vectors, and solver bounds.

    These native human-solver bounds differ from CQQ URDF limits and must not
    be replaced by those URDF limits. Observed with both CQQ left/right URDFs.
    """
    side = str(side).lower()
    if side not in ('left', 'right'):
        raise ValueError('side must be left or right')
    sign = -1.0 if side == 'left' else 1.0
    q = [[0.0, sign*0.26, 0.0, sign*0.35, sign*0.17]]
    q += [[sign*0.26, 0.0, sign*0.35, sign*0.17] for _ in range(4)]
    result = DEFAULT_PARAMETERS.copy()
    result.update(seed=deepcopy(q), history2=deepcopy(q), reference=deepcopy(q),
                  bounds=deepcopy([_THUMB_BOUNDS[side]] + [_OTHER_BOUNDS[side]]*4))
    return result
