"""A new official calibration must not black out an existing MANO rest shape."""
import copy,hashlib,json,time
from pathlib import Path
import numpy as np
import pytest
import torch
from moxian_web.hand_model import DEFAULT_MANO,HandModelBridge,DEFAULT_MATERIAL,MANO_FIT_REVISION,MANO_SOLVER_REVISION
from moxian_web.hand_anatomy import GEOMETRY_REVISION
from moxian_web.mano_runtime import ManoSurface
from moxian_web.contact import default_mapping


def test_changed_official_fingerprint_keeps_live_fk_mesh_and_marks_mount_stale(tmp_path):
    if not torch.cuda.is_available() or not DEFAULT_MANO.exists():pytest.skip('local licensed MANO / CUDA required')
    m=ManoSurface(DEFAULT_MANO,'cpu');m.set_dimensions(np.zeros(20));m.enable_official_fk();serial='isolated-shared-worker-test';home=tmp_path/'hand_model'/hashlib.sha256(serial.encode()).hexdigest()[:16];home.mkdir(parents=True)
    profile={'schema_version':1,'serial':serial,'side':'left','operator':'local','weights_hash':m.weights_hash,'geometry_revision':GEOMETRY_REVISION,'calibrated':True,'revision':'keep-this-shape','material':DEFAULT_MATERIAL,
             'shape':m.shape.tolist(),'dimension_mode':'official_mano_betas_v1','dimension_parameters':[0.]*20,'bone_reference':None,'raw_rotation':np.eye(3).tolist(),'raw_translation':[0.]*3,'mount_pos':np.zeros((5,3)).tolist(),'mount_rot':np.tile(np.eye(3),(5,1,1)).tolist(),
             'sensor_joint_position':None,'mounting_calibrated':False,'flat_reference':None,'calibration_fingerprint':'old-official-record','report':dict(accepted=True,contact_validation_passed=True,together_validation_passed=True,
                 flat_validation_passed=True,penetration_validation_passed=True,runtime_validation_passed=True,
                 per_stage_validation_passed=True,holdout_validation_passed=True,capture_protocol='official_eight_v1',
                 max_penetration_mm=1.,runtime_max_penetration_mm=1.,validation_pose_source='runtime_mano_sensor_only',
                 solver_revision=MANO_SOLVER_REVISION,fit_revision=MANO_FIT_REVISION)}
    (home/'current.json').write_text(json.dumps(profile));original=(home/'current.json').read_bytes();bridge=HandModelBridge(Path(__file__).resolve().parents[2],tmp_path)
    try:
        bridge.start(serial);deadline=time.monotonic()+12;state={}
        while time.monotonic()<deadline:
            stamp=time.time()*1000;pose=copy.deepcopy(m.official_fk.reference);pose['timestamp_ms']=stamp
            frame={'type':'frame','timestamp_ms':stamp,'tactile_timestamp_ms':stamp,'hand_side':'left','calibration_fingerprint':'new-official-record','skeleton':pose,
                   'skeleton_targets':{'positions_local':[[0.,0.,.1]]*5},'fingers':{f:{'position':[0.,0.,.1],'quaternion_xyzw':[0.,0.,0.,1.]} for f in ('thumb','index','middle','ring','pinky')},'raw':[0]*460,
                   'contact':{'ready':False,'patches':[]},'mapping':{'version':'test','assignments':default_mapping('left'),'confirmed':[]}}
            bridge.send(frame);time.sleep(.03);state=bridge.snapshot()
            if state.get('ready'):break
        assert state.get('ready'),state
        assert len(state['vertices_m'])==778 and state['applied_revision']=='keep-this-shape'
        assert state['registration_matches_tracking'] is False and state['input_targets'] is None
        assert not state['calibration_matches_tracking']
        assert (home/'current.json').read_bytes()==original
    finally:bridge.stop()
