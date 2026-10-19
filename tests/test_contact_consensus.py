"""Check the actual Warp loss derivative of the ADMM hand subproblem."""
import numpy as np
import warp as wp
from forceaware import mpc_loss as fa
from retarget.fa.solver import PointState, variable_FA_loss, freeze_contact_points


def test_freeze_only_contact_exploration_preserves_admm_dual():
    device='cpu';job=fa.LossJob();points=PointState()
    template=np.array([[.01,.02,.03],[-.01,0.,.02]],np.float32)
    job.target.query_position_object=wp.array(template,dtype=wp.vec3,device=device)
    for name in ('c','z','last_c'):
        setattr(points,name,wp.ones((3,2),dtype=wp.vec3,device=device))
    dual=np.full((3,2,3),.004,np.float32)
    points.dual=wp.array(dual,dtype=wp.vec3,device=device)
    wp.launch(freeze_contact_points,(3,2),inputs=[job,points],device=device)
    for name in ('c','z','last_c'):
        np.testing.assert_array_equal(getattr(points,name).numpy(),np.broadcast_to(template,(3,2,3)))
    np.testing.assert_array_equal(points.dual.numpy(),dual)


def test_consensus_gradient_has_correct_scale_and_dual_sign():
    device = 'cpu'
    job = fa.LossJob()
    job.model.object_body = 0
    job.model.object_qpos = 0
    job.model.guidance_count = 3
    job.query_enabled = 1
    job.target.query_body = wp.array([1, 1, 1], dtype=int, device=device)
    job.target.query_local = wp.zeros(3, dtype=wp.vec3, device=device)
    weights = np.array([1., .5, 0.], np.float32)
    job.target.query_mask = wp.array(weights, device=device)
    job.target.query_weight = wp.ones(3, device=device)
    job.target.object_qpos = wp.array([0., 0., 0., 1., 0., 0., 0.], dtype=float, device=device)
    job.state.qpos = wp.array([[0., 0., 0., 1., 0., 0., 0.]], dtype=float, device=device)
    positions = np.array([[[0., 0., 0.], [.001, .002, .003]]], np.float32)
    job.state.body_position = wp.array(positions, dtype=wp.vec3, device=device, requires_grad=True)
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], np.float32)
    job.state.body_matrix = wp.array([[rotation, np.eye(3)]], dtype=wp.mat33, device=device)
    job.auxiliary.marked_query_weight = wp.zeros((1, 3), device=device)
    points = PointState()
    c = np.array([[[.002, 0., 0.], [.001, 0., 0.], [0., 0., 0.]]], np.float32)
    dual = np.array([[[.001, 0., 0.], [-.001, 0., 0.], [0., 0., 0.]]], np.float32)
    points.c = wp.array(c, dtype=wp.vec3, device=device, requires_grad=True)
    points.dual = wp.array(dual, dtype=wp.vec3, device=device)
    radius, rho = .005, 1.
    scale = .5*rho/radius**2/weights.sum()
    points.consensus_scale = wp.array([scale*1e6], dtype=float, device=device)
    # Tape maps every struct field to an adjoint, including unused branches.
    def allocate_unused_arrays(struct):
        for name, field in struct._cls.vars.items():
            value = getattr(struct, name)
            if isinstance(field.type, wp.array) and value is None:
                setattr(struct, name, wp.zeros((1,)*field.type.ndim, dtype=field.type.dtype, device=device))
            elif hasattr(value, '_cls'):
                allocate_unused_arrays(value)
    allocate_unused_arrays(job)
    allocate_unused_arrays(points)
    output = wp.zeros(1, device=device, requires_grad=True)
    with wp.Tape() as tape:
        wp.launch(variable_FA_loss, 1, inputs=[job, points], outputs=[output], device=device)
    tape.backward(grads={output: wp.array([1e-6], dtype=float, device=device)})
    error = positions[:, 1:2]-(c-dual)@rotation.T
    expected = 2*scale*weights[None, :, None]*error
    np.testing.assert_allclose(output.numpy()*1e-6, scale*np.sum(weights[None, :, None]*error**2), rtol=1e-5)
    np.testing.assert_allclose(job.state.body_position.grad.numpy()[0, 1], expected.sum(axis=1)[0], rtol=1e-5)
    np.testing.assert_allclose(job.state.body_position.grad.numpy()[0, 0], -expected.sum(axis=1)[0], rtol=1e-5)
    np.testing.assert_allclose(points.c.grad.numpy(), -expected@rotation, rtol=1e-5, atol=1e-5)
