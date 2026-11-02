import numpy as np
import pytest
import trimesh
import warp as wp

from retarget.task.dex import cost as dex
from retarget.task.grasp.tracking_cost import measure_gaps, measure_guidance_gaps


@wp.kernel(enable_backward=False)
def _contact_parts(job: dex.LossJob, result: wp.array2d(dtype=float)):
    w = wp.tid()
    result[w, 0] = dex._penetration_loss(job, w)
    result[w, 1] = dex._physical_contact_loss(job, w, 0)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_other_skin_contact_cannot_satisfy_semantic_patch_maintenance(device):
    wp.init()
    if device != "cpu" and not wp.is_cuda_available():
        pytest.skip("CUDA required")
    arr = lambda value, dtype=float: wp.array(np.asarray(value), dtype=dtype, device=device)
    cube = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
    mesh = wp.Mesh(points=arr(cube.vertices, wp.vec3), indices=arr(cube.faces.flatten(), int))
    positions = arr([[[0, 0, 0], [0.05, 0, 0]]], wp.vec3)
    rotations = arr(np.tile(np.eye(3), (1, 2, 1, 1)), wp.mat33)
    bodies, starts, ends = arr([1], int), arr([0], int), arr([1], int)
    distance, active = wp.empty((1, 2), device=device), wp.zeros((1, 2), dtype=int, device=device)
    wp.launch(measure_gaps, (1, 1), inputs=[mesh.id, 0, bodies, starts, ends,
        arr([[0, 0, 0]], wp.vec3), positions, rotations, distance, active], device=device)
    wp.launch(measure_guidance_gaps, (1, 1), inputs=[mesh.id, 0, bodies, starts, ends,
        arr([[0.03, 0, 0]], wp.vec3), positions, rotations, 1, distance, active], device=device)
    np.testing.assert_allclose(distance.numpy(), [[0, 0.03]], atol=1e-7)

    model, target, state, weights = dex.LossModel(), dex.LossTarget(), dex.LossState(), dex.LossWeights()
    model.output_count = model.guidance_count = 1
    model.output_contact_rows = arr([0, 1], int)
    model.contact_thresholds = arr([0.01, 0.01])
    target.contact_output = arr([1], int)
    target.contact_mask = target.contact_weight = arr([1.0])
    state.contact_distance, state.contact_active = distance, active
    state.contact_position = wp.zeros((1, 2), dtype=wp.vec3, device=device)
    weights.contact_robustness, weights.target_phi = 40.0, -0.001
    weights.soft_penetration, weights.penetration = 15.0, 75.0
    weights.soft_penetration_limit, weights.penetration_limit = 0.002, 0.003
    job = dex.LossJob()
    job.model, job.target, job.state, job.weights = model, target, state, weights
    result = wp.zeros((1, 2), device=device)
    wp.launch(_contact_parts, 1, inputs=[job, result], device=device)
    np.testing.assert_allclose(result.numpy(), [[0, 40 * 31**2]], rtol=1e-6, atol=1e-5)
