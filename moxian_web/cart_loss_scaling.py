"""Monotone scaling of the complete CARD cost, preserving its term weights."""
import math


def normalized_policy_cost(cost, scale=0.):
    """Map nonnegative mean cost C to C/(C+S); zero scale keeps raw cost.

    Apply once to the complete batch mean, not separately per term/frame.
    This preserves the raw objective's gradient direction and minimizers.
    Unlike an exponential reward, its derivative decays polynomially.
    """
    if isinstance(scale,bool) or not math.isfinite(scale) or scale<0:
        raise ValueError('Policy cost scale must be finite and nonnegative')
    if scale==0:return cost
    return cost/(cost+scale)


def policy_cost_scaling_spec(scale):
    normalized_policy_cost(0.,scale)
    return dict(kind='card_rational_mean_v1' if scale else 'raw',scale=float(scale),
        formula='mean_raw_cost/(mean_raw_cost+scale)' if scale else 'mean_raw_cost',
        range='[0,1)' if scale else 'nonnegative',applied_after='complete_batch_mean',
        raw_term_weights_preserved=True)
