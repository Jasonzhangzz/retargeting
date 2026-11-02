"""Root matcher selection is independent of yet-unadapted virtual forces."""

from dataclasses import replace

import numpy as np
import torch
import trimesh
import warp as wp

wp.config.kernel_cache_dir = "/tmp/retarget-template-mppi-warp"

from retarget.env.grasp import template_mppi
from retarget.env.grasp.template_mppi import TemplateMPPISettings, TemplatePoseObjective
from retarget.env.grasp.template_mppi_inputs import TemplateContacts
from retarget.env.grasp.template_mppi_runner import _choose_root


def test_type9_initialization_uses_shared_geometry_but_no_force_cost(tmp_path, monkeypatch):
    mesh = tmp_path / "cylinder.obj"
    trimesh.creation.cylinder(radius=.018, height=.12, sections=16).export(mesh)
    settings = TemplateMPPISettings(closure={"gn_iterations": 1, "equilibrium_iterations": 2,
                                           "equilibrium_line_search_steps": 1, "cuda_graph": False})
    objective = TemplatePoseObjective(mesh, settings, device="cpu")
    q = objective.cpu.qpos0.copy()
    template = TemplateContacts(("right_finger2_link4", "right_finger1_link4"),
        [[0., .005, -.015], [0., .005, -.015]], [[0., 1., 0.], [0., 1., 0.]],
        q[6:], (("right_finger2_link4",), ("right_finger1_link4",)), {"template": "9_Palmar_Pinch"})
    objective.bind(template, q)
    candidates = np.tile(q, (4, 1))
    candidates[:, 0] = [0., .05, .1, .4]
    controls = torch.tensor(candidates, dtype=torch.float32)[:, None]
    objective.evaluate(controls)
    parts = objective.parts
    expected = (settings.distance_weight * parts["distance"]
                + settings.normal_weight * parts["normal"]
                + settings.required_group_weight * parts["required_group"]
                + settings.penetration_weight * parts["penetration"]
                + settings.soft_penetration_weight * parts["soft_penetration"]
                + settings.pose_weight * parts["pose"])

    def forbidden(*args, **kwargs):
        raise AssertionError("Root-only geometry must not evaluate virtual contact forces")

    monkeypatch.setattr(objective.closure_solver, "solve", forbidden)
    monkeypatch.setattr(template_mppi, "force_contact_terms", forbidden)
    actual = objective.evaluate_initialization(controls)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-5)
    objective.settings = replace(settings, closure_weight=1.e8, force_contact_weight=1.e8)
    torch.testing.assert_close(objective.evaluate_initialization(controls), actual, rtol=0, atol=0)
    chosen, selection = _choose_root(objective, template, candidates, "cpu")
    np.testing.assert_array_equal(chosen, candidates[int(actual.argmin())])
    assert selection["root_selection_cost"] == "geometry_only"
    assert selection["selection_objective"] == "template_root_geometry_v1"
    assert selection["actual_contact_or_force_closure_required"] is False


def test_root_selection_prefers_geometry_over_the_opposite_force_ranking():
    candidates = np.zeros((2, 26))
    candidates[:, 0] = [126., 140.]

    class Objective:
        def bind(self, template, q):
            pass

        def evaluate_initialization(self, controls):
            # Same root ordering as the recorded Type 9 geometry regression.
            return controls.new_tensor([31.276, 113.520])

        def evaluate(self, controls):
            raise AssertionError("The full force score must not select the root")

    chosen, selection = _choose_root(Objective(), None, candidates, "cpu")
    assert chosen[0] == 126.
    assert selection["selected"] == 0
    assert selection["root_selection_cost"] == "geometry_only"
