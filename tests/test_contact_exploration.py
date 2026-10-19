"""Surface feasibility, wrench exploration and ADMM consensus contracts."""
import numpy as np
from retarget.contact_exploration import (
    ObjectMeshSurface, LocalContactExplorer, closest_triangles, update_scaled_dual,
)


def plane(z=0.):
    return ObjectMeshSurface([[-.1, -.1, z], [.1, -.1, z], [.1, .1, z], [-.1, .1, z]],
                             [[0, 1, 2], [0, 2, 3]])


def test_triangle_projection_handles_interior_edges_and_vertices():
    triangle = np.array([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]])
    points = [[.2, .3, 1.], [.7, .7, 1.], [-1., -1., 0.]]
    np.testing.assert_allclose(closest_triangles(points, triangle)[:, 0],
                               [[.2, .3, 0.], [.5, .5, 0.], [0., 0., 0.]])


def test_surface_updates_stay_bounded_over_rounds_and_freeze_inactive_points():
    surface = plane()
    template = np.array([[0., 0., 0.], [.01, 0., 0.]])
    points = template.copy()
    solver = LocalContactExplorer(surface, radius=.005, wrench_weight=0, sweeps=2)
    hand = template + [.04, .03, .02]
    for _ in range(12):
        points, diagnostic = solver.solve(points, template, hand, [1., 0.], np.ones((2, 2)),
                                          center=np.zeros(3), object_radius=.1)
        assert np.linalg.norm(points[0]-template[0]) <= .005+1e-8
        assert abs(points[0, 2]) < 1e-9
        np.testing.assert_array_equal(points[1], template[1].astype(np.float32))
        assert diagnostic['final_value'] <= diagnostic['initial_value']+1e-10
    assert np.linalg.norm(points[0]-template[0]) > .001


def test_patch_does_not_jump_to_nearby_disconnected_sheet():
    surface = ObjectMeshSurface(
        [[-.1, -.1, 0.], [.1, -.1, 0.], [0., .1, 0.],
         [-.1, -.1, .001], [.1, -.1, .001], [0., .1, .001]], [[0, 1, 2], [3, 4, 5]])
    patch = surface.patch(np.zeros(3), .005)
    points, _ = patch.project([[0., 0., .001]])
    np.testing.assert_allclose(points, [[0., 0., 0.]], atol=1e-12)


def test_bad_guidance_has_explicit_fallback_without_radius_violation():
    template = np.array([[0., 0., .03]])
    solver = LocalContactExplorer(plane(), radius=.005)
    result, diagnostic = solver.solve(template, template, np.zeros((1, 3)), [1.],
                                      np.ones((1, 2)), center=np.zeros(3), object_radius=.1)
    np.testing.assert_allclose(result, template)
    assert diagnostic['frozen_off_surface_contacts'] == 1
    assert diagnostic['max_shift_mm'] == 0.


def test_wrench_term_separates_collapsed_contacts_even_when_hand_is_aligned():
    template = np.array([[0., 0., .05]]*3)
    solver = LocalContactExplorer(plane(.05), radius=.01, rho=.001, sweeps=2, wrench_weight=10)
    points, diagnostic = solver.solve(template, template, template, np.ones(3), np.zeros((3, 2)),
                                      center=np.zeros(3), object_radius=.1)
    assert diagnostic['wrench_deficit_after'] < diagnostic['wrench_deficit_before']-.01
    assert np.max(np.linalg.norm(points-template, axis=-1)) <= .01+1e-8
    np.testing.assert_allclose(points[:, 2], .05)
    fixed, _ = LocalContactExplorer(plane(.05), radius=.01, wrench_weight=0).solve(
        template, template, template, np.ones(3), np.zeros((3, 2)), center=np.zeros(3), object_radius=.1)
    np.testing.assert_allclose(fixed, template, atol=1e-8)


def test_scaled_dual_sign_mask_and_bound():
    dual = np.zeros((2, 2, 3))
    hand = np.array([[[.001, 0., 0.], [1., 0., 0.]], [[1., 0., 0.], [1., 0., 0.]]])
    updated, residual = update_scaled_dual(dual, hand, np.zeros_like(hand), [True, False], .02)
    np.testing.assert_allclose(updated[0, 0], [.001, 0., 0.])
    np.testing.assert_allclose(updated[1, 0], [.02, 0., 0.])
    np.testing.assert_array_equal(updated[:, 1], 0.)
    np.testing.assert_array_equal(residual, hand)
