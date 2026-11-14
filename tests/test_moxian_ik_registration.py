"""The correction is a calibrated constant before the official solver."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation
from moxian_web.kinematic_policy import (
    REGISTRATION_REVISION, apply_ik_registration, load_ik_registration, validate_input_transform,
)
from moxian_web.native_skeleton import registered_observation, quaternion_matrix, NativeSkeleton


def test_channel_transform_is_left_multiplication_and_preserves_raw_sensor_frame():
    raw={'position':[.01,.02,.03],'quaternion_xyzw':Rotation.from_euler('y',.7).as_quat().tolist()}
    before=copy.deepcopy(raw);a=Rotation.from_euler('z',.9).as_matrix();b=np.array([.03,-.01,.04])
    transform={'rotation':a.tolist(),'translation_m':b.tolist()}
    p,r=registered_observation(raw,validate_input_transform(transform))
    np.testing.assert_allclose(p,a@raw['position']+b,atol=1e-12)
    np.testing.assert_allclose(np.array(r).reshape(3,3),a@np.array(quaternion_matrix(raw['quaternion_xyzw'])).reshape(3,3),atol=1e-12)
    assert raw==before
    p,r=registered_observation(raw)
    assert p==raw['position'] and r==quaternion_matrix(raw['quaternion_xyzw'])


@pytest.mark.parametrize('transform',[
    {'rotation':[[-1,0,0],[0,1,0],[0,0,1]],'translation_m':[0,0,0]},
    {'rotation':[[2,0,0],[0,1,0],[0,0,1]],'translation_m':[0,0,0]},
    {'rotation':np.eye(3).tolist(),'translation_m':[float('nan'),0,0]},
    {'rotation':np.eye(3).tolist(),'translation_m':[1,0,0]},
])
def test_registration_rejects_reflections_scale_and_nonfinite_values(transform):
    with pytest.raises(ValueError):validate_input_transform(transform)


def test_configuration_is_single_yaml_block_and_bound_to_exact_artifact(tmp_path):
    text='hand:\n  side: left\nsolver:\n  old: true\ntip_mapping:\n  left:\n    old: true\nqnbot_sdk:\n  physics_enabled: false\n__resource_root__: /derived\n'
    data={'accepted':True,'calibration_sha256':'a','revision':REGISTRATION_REVISION,
          'configuration':{'solver':{'optimizer':{'gn_max_iters':40}},'tip_mapping':{'left':{'pinch_model':{'enabled':False}}},'hand':{'side':'left'}},
          'input_transforms':{'thumb':{'rotation':np.eye(3).tolist(),'translation_m':[.01,.02,.03]}}}
    output=apply_ik_registration(text,data);parsed=yaml.safe_load(output)
    assert parsed['solver']==data['configuration']['solver'] and 'old:' not in output
    assert output.count('\nsolver:')==1 and parsed['qnbot_sdk']['physics_enabled'] is False
    assert parsed['__resource_root__']=='/derived' and parsed['__moxian_input_registration__']==data['input_transforms']
    (tmp_path/'ik-registration.json').write_text(json.dumps(data))
    assert load_ik_registration(tmp_path,'a')==data
    assert load_ik_registration(tmp_path,'different-artifact') is None
    data['revision']='obsolete';(tmp_path/'ik-registration.json').write_text(json.dumps(data))
    assert load_ik_registration(tmp_path,'a') is None


def test_native_registered_input_equals_explicit_transform_with_same_official_engine(tmp_path):
    root=Path(__file__).resolve().parents[1]
    packages=list((root/'.venv-moxian-ik/lib').glob('python*/site-packages/qnbot_sdk/glove/_private/algorithms/skeleton/*'))
    if len(packages)!=1:pytest.skip('local official native package required')
    from moxian_web.kinematic_policy import build_kinematic_config
    source=packages[0]/'share/qnbot_hand/hands/dynamic_openxr_hand/dynamic_openxr_hand_system.yaml'
    config=yaml.safe_load(source.read_text());config['__resource_root__']='/unused';config['joint_smoother']['enabled']=True;config['solver']['error_reseed']['enabled']=True
    baseline,_=build_kinematic_config(yaml.safe_dump(config,sort_keys=False),packages[0],tmp_path)
    transform={'rotation':Rotation.from_euler('z',.12).as_matrix().tolist(),'translation_m':[.01,-.02,.03]}
    registered=tmp_path/'registered.yaml';registered.write_text(baseline.read_text()+'\n__moxian_input_registration__: '+json.dumps({'thumb':transform})+'\n')
    engines=[NativeSkeleton(packages[0],registered),NativeSkeleton(packages[0],baseline)]
    frame={'timestamp_ms':1234567.,'fingers':{f:{'position':[.04,.02,.10],'quaternion_xyzw':[0.,0.,0.,1.]}for f in ('thumb','index','middle','ring','pinky')}}
    before=copy.deepcopy(frame);explicit=copy.deepcopy(frame);p,r=registered_observation(frame['fingers']['thumb'],transform)
    explicit['fingers']['thumb']={'position':p,'quaternion_xyzw':Rotation.from_matrix(np.array(r).reshape(3,3)).as_quat().tolist()}
    try:
        assert engines[0].runtime_config_path.name!=engines[1].runtime_config_path.name
        assert engines[0].runtime_config_path.name.startswith('native-')
        for _ in range(10):a=engines[0].step(frame);b=engines[1].step(explicit)
        np.testing.assert_allclose(a['pose']['positions_local'],b['pose']['positions_local'],atol=1e-9)
        assert a['pose']['timestamp_ms']==frame['timestamp_ms'] and a['pose']['hand_side']=='left'
        assert len(a['pose']['names'])==25 and not a['physics']['enabled'] and a['physics']['steps']==0
        assert frame==before
        assert a['continuity']['engine_frames']==10 and a['continuity']['reused_engine']
        assert len(a['continuity']['thumb_joint_steps'])==6
    finally:
        for e in engines:e.close()
