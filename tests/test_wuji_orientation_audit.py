"""Metrology checks for the offline orientation diagnosis, not runtime policy."""
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from scripts.audit_moxian_wuji_orientation import (angle,distal_from_rx,anatomical_target,
    abduction_bound,exact_orientation_position_bound,OrientationObjective,robot_frames)
from moxian_web.wuji_contact_slsqp import NativeContactKinematics
from moxian_web.wuji_surface import WujiSurface


@pytest.fixture(scope='module')
def fk():return NativeContactKinematics(WujiSurface())


@pytest.mark.parametrize('mirror',[np.eye(3),np.diag([-1.,1.,1.])])
def test_rx_mount_removal_and_mirroring_preserve_proper_anatomical_rotation(mirror):
    rng=np.random.default_rng(914)
    for _ in range(5):
        distal,align,mount,frame,wrist=Rotation.random(5,random_state=rng).as_matrix()
        raw=align.T@distal@mount
        recovered=distal_from_rx(raw,align,mount)
        np.testing.assert_allclose(recovered,distal,atol=1e-14)
        target=anatomical_target(recovered,frame,mirror,wrist)
        np.testing.assert_allclose(target.T@target,np.eye(3),atol=1e-14)
        assert np.linalg.det(target)==pytest.approx(1.)
        # Coordinate reflection must not alter a physical rotation error.
        moved=distal@Rotation.from_rotvec([.1,-.2,.05]).as_matrix()
        other=anatomical_target(moved,frame,mirror,wrist)
        assert angle(target,other)==pytest.approx(angle(distal,moved),abs=1e-10)


def test_anatomical_orientation_objective_has_correct_geometric_gradient(fk):
    class Zero:
        def evaluate(self,q,geometry,stage):return 0.,np.zeros(20),{}
    q=np.linspace(.1,.3,20);_,target=robot_frames(fk,q+.05)
    objective=OrientationObjective(fk,Zero(),target,1.7)
    _,gradient,_=objective.evaluate(q,fk.forward(q))
    for j in range(20):
        d=np.eye(20)[j]*1e-6
        difference=(objective.evaluate(q+d,fk.forward(q+d))[0]-objective.evaluate(q-d,fk.forward(q-d))[0])/2e-6
        assert gradient[j]==pytest.approx(difference,abs=1e-9,rel=1e-6)


def test_non_thumb_abduction_invariant_gives_a_necessary_reachability_bound(fk):
    engine=SimpleNamespace(fk=fk,surface=fk.surface)
    rng=np.random.default_rng(68)
    for f in range(1,5):
        for _ in range(5):
            q=rng.uniform(fk.lower,fk.upper);_,r=robot_frames(fk,q)
            bound=abduction_bound(engine,f,r[f])
            assert bound['required_middle_angle_deg']==pytest.approx(abs(np.degrees(q[4*f+1])),abs=1e-8)
            assert bound['proven_orientation_lower_bound_deg']==pytest.approx(0.,abs=1e-8)
        fk.forward(np.zeros(20));base=fk.data.xmat[int(fk.body_ids[1+4*f])].reshape(3,3).copy()
        forbidden=base@Rotation.from_euler('XYX',[.3,1.1,-.5]).as_matrix()@fk.surface.bases[4+4*f]
        bound=abduction_bound(engine,f,forbidden)
        assert bound['proven_orientation_lower_bound_deg']==pytest.approx(np.degrees(1.1-fk.upper[4*f+1]),abs=1e-8)


def test_exact_orientation_circle_bound_matches_native_fk_and_proves_normal_offset(fk):
    engine=SimpleNamespace(fk=fk,surface=fk.surface);rng=np.random.default_rng(823)
    for f in range(1,5):
        for _ in range(4):
            q=rng.uniform(fk.lower,fk.upper);points,rotations=robot_frames(fk,q)
            tip=points[0][4+4*f].copy();orientation=rotations[f].copy()
            bound=exact_orientation_position_bound(engine,f,orientation,tip)
            assert bound['orientation_reachable'] and bound['proven_min_tip_error_mm']<1e-7
            reconstructed=q.copy();reconstructed[4*f:4*f+4]=bound['minimizing_q']
            actual,actual_r=robot_frames(fk,reconstructed)
            np.testing.assert_allclose(actual[0][4+4*f],tip,atol=1e-10)
            assert angle(actual_r[f],orientation)<1e-7
            # The exact-orientation arc has a fixed hinge-axis coordinate.
            axis=fk.data.xmat[int(fk.body_ids[3+4*f])].reshape(3,3)[:,0].copy()
            displaced=exact_orientation_position_bound(engine,f,orientation,tip+.006*axis)
            assert displaced['proven_min_tip_error_mm']==pytest.approx(6.,abs=1e-7)
