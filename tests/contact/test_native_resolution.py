import mujoco
import numpy as np
from retarget.env.contact import native_evaluation
from retarget.env.contact.native import snapshot
from retarget.env.contact.prediction import ContactPrediction


def test_unknown_forecast_keeps_its_status_and_uses_observed_native_contact(monkeypatch):
    model = mujoco.MjModel.from_xml_string('<mujoco><option timestep=".005"/><worldbody><geom name="floor" type="plane" size="1 1 .1"/><body pos="0 0 .049"><freejoint/><geom name="ball" type="sphere" size=".05" mass="1"/></body></worldbody></mujoco>')
    if hasattr(mujoco.mjtEnableBit, 'mjENBL_SPECULATIVE'):
        model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_SPECULATIVE
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    event = dict(geom1=1, geom2=0, link_local_point=[0, 0, -.05], object_local_point=[0, 0, 0],
        link_local_normal=[0, 0, -1], object_local_normal=[0, 0, 1], start_time=0., duration=.02,
        normal_velocity=0., tangential_velocity=[0, 0, 0], mode='stick')
    packet = dict(ref=np.zeros((4, 0)), weight=np.zeros(0), terminal=np.zeros(0),
        ctrl_ref=np.zeros((4, 0)), ctrl_weight=np.zeros(0), heading=-1, heading_weight=0.,
        support_obj=-1, owner=np.zeros(2, int), target_geoms=[0], dt=.005)
    monkeypatch.setattr(native_evaluation, 'forecast_contacts', lambda *args: [ContactPrediction(1, 0, 'uncertain').as_dict()])
    result = native_evaluation.evaluate_candidate(model, snapshot(data), np.zeros((3, 0)), packet, [event], True)
    assert np.isfinite(result['cost'])
    prediction = result['predictions'][0]
    assert prediction['status'] == 'uncertain' and prediction['toi'] is None
    assert prediction['evaluation_fallback'] == 'native_speculative_rollout'
    assert prediction['fallback_result'] == 'contact_constraint'
    assert prediction['native_first_contact_time'] is not None
