import numpy as np
import pytest
import torch
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from moxian_web.contact_force import estimate_normal_resultant,fit_coulomb_force


def test_pressure_resultant_is_relative_and_rotation_covariant():
    normals=np.array([[0.,0,1],[0,1,0]]);response=np.array([3.,1.])
    a=estimate_normal_resultant(normals,response)
    np.testing.assert_allclose(a['vector_relative'],[0,1,3])
    assert a['scalar_normal_load_relative']==4 and not a['shear_measured'] and not a['force_calibrated']
    r=Rotation.from_rotvec([.2,-.5,.7]).as_matrix();b=estimate_normal_resultant(normals@r.T,response*5)
    np.testing.assert_allclose(b['direction'],np.asarray(a['direction'])@r.T)
    assert b['coherence']==pytest.approx(a['coherence'])
    assert b['magnitude_relative']==pytest.approx(a['magnitude_relative']*5)
    cancelled=estimate_normal_resultant([[0,0,1],[0,0,-1]],[1,1])
    assert not cancelled['direction_valid'] and cancelled['direction'] is None
    assert not estimate_normal_resultant(normals,[0,0])['direction_valid']


@pytest.mark.parametrize('mu',[0.,.5,1.,2.])
@pytest.mark.parametrize('normal_weight',[0.,1.,3.])
def test_analytic_force_fit_matches_independent_constrained_optimizer(mu,normal_weight):
    rng=np.random.default_rng(31)
    for desired in rng.normal(size=(10,3)):
        value=fit_coulomb_force(torch.tensor(desired),torch.tensor([[0.,0.,1.]],dtype=torch.float64),mu,normal_weight)
        a=desired[2];b=np.linalg.norm(desired[:2]);target=np.linalg.norm(desired)
        def objective(x):return (x[0]-a)**2+(x[1]-b)**2+normal_weight*(x[0]-target)**2
        oracle=minimize(objective,[max(a,0)+1,0],bounds=[(0,None),(0,None)],constraints=[{'type':'ineq','fun':lambda x:mu*x[0]-x[1]}],method='SLSQP',options={'ftol':1e-12,'maxiter':100})
        assert oracle.success
        actual=float(value['vector_error'][0]+value['normal_error'][0])
        assert actual==pytest.approx(oracle.fun,abs=2e-7)
        assert value['projected_in_cone'][0]


def test_projected_cone_feasibility_does_not_claim_desired_force_is_supported():
    v=fit_coulomb_force(torch.tensor([0.,0.,-1.]),torch.tensor([[0.,0.,1.]]))
    assert not v['desired_in_cone'][0] and v['projected_in_cone'][0]
    assert v['normal_force'][0]==0 and v['vector_error'][0]==1
    inside=fit_coulomb_force(torch.tensor([.2,0.,1.]),torch.tensor([[0.,0.,1.]]))
    torch.testing.assert_close(inside['force'][0],torch.tensor([.2,0.,1.]))
    assert inside['desired_in_cone'][0] and inside['vector_error'][0]==0


def test_force_loss_is_rotation_covariant_and_scale_normalization_is_valid():
    d=torch.tensor([1.,.3,.1],dtype=torch.float64);n=torch.tensor([[0.,0.,1.]],dtype=torch.float64)
    r=torch.tensor(Rotation.from_rotvec([.1,.3,.9]).as_matrix())
    a=fit_coulomb_force(d,n,.5,1.);b=fit_coulomb_force(d@r.T*4,n@r.T,.5,1.)
    torch.testing.assert_close(b['force'],a['force']@r.T*4)
    torch.testing.assert_close(b['vector_error']/16,a['vector_error'])
    torch.testing.assert_close(b['normal_error']/16,a['normal_error'])


def test_force_model_rejects_invalid_inputs_and_settings():
    from moxian_web.card_math import CardSettings
    for response in ([np.nan],[-1.]):
        with pytest.raises(ValueError):estimate_normal_resultant([[0,0,1]],response)
    for kwargs in ({'friction_coefficient':-1},{'normal_force_weight':np.inf},{'contact_direction_model':'pinch'}):
        with pytest.raises(ValueError):CardSettings(**kwargs)
