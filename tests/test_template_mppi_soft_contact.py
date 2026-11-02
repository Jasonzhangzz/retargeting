"""Soft compression guidance must not weaken self-penetration or force checks."""
import numpy as np
import pytest
import torch

from retarget.env.grasp.template_mppi import (
    TemplateMPPISettings, acceptance, contact_terms, penetration_terms,
)


def test_soft_layer_guides_separation_and_deep_compression_but_accepts_contact():
    settings = TemplateMPPISettings(outlier_weight=0., contact_fraction=1.)
    gap = torch.tensor([[.003, 0., -.001, -.002, -.003]], requires_grad=True)
    outward = torch.tensor([[[1., 0., 0.]]]).expand(1, 5, 3)
    terms = contact_terms(gap, -outward, outward, settings)
    assert terms['fitted'].tolist() == [[False, True, True, True, False]]
    assert terms['slot_error'][0, 1:4].tolist() == [0., 0., 0.]
    terms['distance'].backward()
    assert gap.grad[0, 0] > 0  # Move separated points inward.
    assert gap.grad[0, -1] < 0  # Move excessively compressed points outward.
    assert gap.grad[0, 1:4].tolist() == [0., 0., 0.]
    reversed_normal = contact_terms(torch.full((1, 5), -.001), outward, outward, settings)
    assert not reversed_normal['fitted'].any()
    assert reversed_normal['normal'].item() > 1.


def test_touch_and_small_hand_object_compression_are_cheap_but_self_overlap_is_not():
    settings = TemplateMPPISettings()
    depths = torch.tensor([[0., 0.], [.0005, 0.], [.002, 0.], [.003, 0.], [0., .0005]])
    strict, soft, excess = penetration_terms(depths, settings)
    cost = settings.penetration_weight * strict + settings.soft_penetration_weight * soft
    assert cost[0].item() == 0.
    assert cost[1].item() == pytest.approx(.00625)
    assert cost[2].item() == pytest.approx(.1)
    assert cost[3].item() > 9999.
    assert cost[4].item() == pytest.approx(2500.)
    assert excess[:3].tolist() == [0., 0., 0.]
    assert excess[3].item() == pytest.approx(.001)


def test_soft_boundary_is_continuous_and_allows_zero_allowance_for_strict_runs():
    settings = TemplateMPPISettings()
    eps = 1.e-9
    depth = torch.tensor([[.002 - eps, 0.], [.002, 0.], [.002 + eps, 0.]], dtype=torch.float64)
    strict, soft, _ = penetration_terms(depth, settings)
    costs = settings.penetration_weight * strict + settings.soft_penetration_weight * soft
    assert float(costs.max() - costs.min()) < 1.e-6
    strict_settings = TemplateMPPISettings(hand_object_allowance_m=0.)
    strict, _, _ = penetration_terms(torch.tensor([[.0005, 0.]]), strict_settings)
    assert (strict * strict_settings.penetration_weight).item() == pytest.approx(2500.)


def test_optional_clearance_is_removed_from_object_depth_but_kept_for_self_guidance():
    settings = TemplateMPPISettings(collision_clearance_m=.00005, collision_margin_m=.0001)
    # Native buffers store max(clearance - signed separation, 0), not raw depth.
    deficits = torch.tensor([[.00055, .00002], [0., 0.]])
    strict, soft, excess = penetration_terms(deficits, settings)
    assert strict[0].item() == pytest.approx((.00002 / .001)**2)
    assert soft[0].item() == pytest.approx((.0005 / .002)**2)
    assert excess.tolist() == [0., 0.]
    assert strict[1].item() == soft[1].item() == 0.


@pytest.mark.parametrize('coverage,certified,valid,depth,expected', [
    (.2, True, True, [.0015, 0.], True),  # 80% coverage is diagnostic only.
    (1., True, True, [.002, 0.], True),
    (1., True, True, [.0020019, 0.], True),  # The observed slight overshoot is OK.
    (1., True, True, [.0021, 0.], True),  # 2mm is a cost threshold, not a gate.
    (1., True, True, [.001, .0001], False),
    (1., False, True, [.001, 0.], False),  # No force certificate cannot pass.
    (1., True, False, [.001, 0.], False),
    (1., True, True, [np.nan, 0.], False),
])
def test_closure_and_independent_depth_limits_define_acceptance(coverage, certified, valid, depth, expected):
    result = acceptance(coverage, certified, valid, np.array(depth), TemplateMPPISettings())
    assert result['soft_contact_success'] is expected
    if coverage < .8:
        assert not result['geometry_success']


@pytest.mark.parametrize('settings', [
    {'hand_object_allowance_m': -.001}, {'soft_penetration_weight': -1.},
    {'hand_object_allowance_m': float('nan')}, {'collision_margin_m': -.0001},
])
def test_invalid_soft_contact_parameters_fail_early(settings):
    with pytest.raises(ValueError):
        TemplateMPPISettings(**settings)


def test_viewer_labels_soft_acceptance_and_depth_limits_without_native_evaluation():
    from retarget.viewer.runtime import final_summary
    text = final_summary(dict(success_criterion='projected_soft_contact_closure_v1',
        task_success=True, native_evaluated=False, geometry_diagnostics=dict(
            coverage=.2, projected_closure=True, max_hand_object_penetration_m=.0015,
            hand_object_allowance_m=.002, max_hand_hand_penetration_m=0.)))
    assert '软接触闭合验收：通过' in text
    assert '1.500 mm / 允许 2.000 mm' in text
    assert '手部自穿透：0.000 mm' in text
    assert '原生物理评测：通过' not in text


def test_new_viewer_calls_the_object_allowance_a_soft_threshold():
    from retarget.viewer.runtime import final_summary
    text = final_summary(dict(success_criterion='projected_soft_contact_closure_v2',
        task_success=True, native_evaluated=False, geometry_diagnostics=dict(
            projected_closure=True, max_hand_object_penetration_m=.0020019,
            hand_object_allowance_m=.002, max_hand_hand_penetration_m=0.)))
    assert '软接触闭合验收：通过' in text
    assert '2.002 mm / 软阈值 2.000 mm' in text
