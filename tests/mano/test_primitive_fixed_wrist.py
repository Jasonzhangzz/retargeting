from pathlib import Path
import numpy as np
import pytest
import torch
from retarget.env.grasp.primitive_problem import AnatomicalPrimitiveDecoder,frozen_wrist_problem
SNAP=Path('output/mano_wuji_primitives_20261105/saved_left')
@pytest.mark.skipif(not SNAP.exists(),reason='Personal model required')
def test_anatomical_coordinates_have_finite_zero_jacobian_and_real_bounds():
 m=AnatomicalPrimitiveDecoder(SNAP,'cpu');q=torch.zeros(1,6+m.components)
 assert m.components==27
 jac=torch.func.jacfwd(lambda v:m.joints_only(v[None]).flatten())(q[0])
 assert torch.isfinite(jac).all() and jac.abs().max()>0
 problem=frozen_wrist_problem(q,m)
 trial=problem.project(torch.randn(1,128,q.shape[-1])*100)
 assert torch.equal(trial[:,:,:6],torch.zeros_like(trial[:,:,:6]))
 assert (trial[:,:,6:]>=m.anatomical_low).all() and (trial[:,:,6:]<=m.anatomical_high).all()
 angles=m.angles(trial[:,0]).reshape(15,3).numpy()
 recovered=m.anatomy.angles(angles)
 active=m.active_anatomical.numpy();np.testing.assert_allclose(recovered.ravel()[active],trial[0,0,6:].numpy(),atol=5e-7)
 next_problem=frozen_wrist_problem(q,m,q,continuity_radius=.06)
 nxt=next_problem.project(torch.randn(1,128,q.shape[-1])*100)
 assert (nxt[:,:,6:].abs()<=.0600001).all()


def test_wrist_frame_change_preserves_object_hand_relative_positions():
 from scipy.spatial.transform import Rotation
 from retarget.env.grasp.primitive_problem import canonicalize_wuji_wrist
 rng=np.random.default_rng(42);q=rng.normal(size=(10,26));q[:,:3]*=.03
 fixed,obj=canonicalize_wuji_wrist(q,4)
 np.testing.assert_array_equal(fixed[:,:6],np.broadcast_to(q[4,:6],(10,6)))
 p=rng.normal(size=(10,7,3))*.02
 r=Rotation.from_euler('XYZ',q[:,3:6]).as_matrix()
 hand=np.einsum('nij,nkj->nki',r,p)+q[:,None,:3]
 ro=Rotation.from_quat(obj[:,[4,5,6,3]]).as_matrix()
 moved=np.einsum('nij,nkj->nki',ro,hand)+obj[:,None,:3]
 expected=np.einsum('ij,nkj->nki',r[4],p)+q[4,:3]
 np.testing.assert_allclose(moved,expected,atol=1e-15)
 assert np.max(np.abs(obj[:,:3]))>.001
 np.testing.assert_allclose(obj[4],[0,0,0,1,0,0,0],atol=1e-15)


@pytest.mark.skipif(not SNAP.exists(),reason='Personal model required')
def test_reference_wrist_can_fit_once_then_size_search_cannot_move_it():
 from retarget.env.grasp.primitive_problem import reference_wrist_problem
 m=AnatomicalPrimitiveDecoder(SNAP,'cpu');q=torch.zeros(1,6+m.components)
 reference=reference_wrist_problem(q,m)
 candidate=reference.project(torch.ones(1,1,q.shape[-1])*100)[:,0]
 np.testing.assert_allclose(candidate[0,:3],[.025]*3)
 np.testing.assert_allclose(candidate[0,3:6],[.35]*3)
 fixed=frozen_wrist_problem(candidate,m,candidate,continuity_radius=.1)
 trials=fixed.project(torch.randn(1,128,q.shape[-1])*100)
 assert torch.equal(trials[:,:,:6],candidate[:,None,:6].expand(1,128,6))
