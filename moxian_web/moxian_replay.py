"""Rebuild a Moxian source mesh from raw EMF and an explicit matching model."""
from __future__ import annotations
import time
import numpy as np
import torch


class MoxianReplayReconstructor:
    def __init__(self, candidate):
        from .bone_extrinsics import validate_candidate
        from .studio_emf_solver import StudioQPSkeleton
        from .qnbot_surface import QnbotSurface
        if candidate.get('wuji_native') or not candidate.get('bone_extrinsics'):
            raise ValueError('Raw replay requires a Moxian bone/extrinsics model, not a Wuji human model')
        validate_candidate(candidate)
        self.candidate = candidate
        self.model = QnbotSurface(candidate['side'], 'cpu', finger_twist=len(candidate['bone_extrinsics']['rig']['joint_names']) > 25)
        self.model.load_artifact(candidate); self.ik = StudioQPSkeleton(candidate)
        self.previous_stamp = None; self.topology = None

    @torch.no_grad()
    def frame(self, observation):
        if observation.get('calibration_fingerprint') != self.candidate['calibration_fingerprint']:
            raise ValueError('Raw EMF capture and Moxian calibration fingerprints differ')
        mapping = observation.get('mapping') or {}
        if self.model.mapping is None: self.model.bind_mapping(mapping)
        if mapping.get('assignments') != self.model.mapping['assignments']:
            raise ValueError('Raw tactile assignments differ from the calibrated material binding')
        if observation.get('hand_side') != self.candidate['side']:
            raise ValueError('Raw EMF and Moxian model hand sides differ')
        stamp = float(observation['timestamp_ms'])
        if self.previous_stamp is not None and stamp <= self.previous_stamp: self.ik.reset()
        began = time.perf_counter(); pose = self.ik.step(observation)['pose']
        q = self.model.angles([{'skeleton': pose}]); vertices, joints, _ = self.model(q)
        self.previous_stamp = stamp
        self.topology = dict(self.model.topology(), model_hash=self.candidate['model_hash'])
        error = np.sqrt(np.mean([v['position_error_mm']**2 for v in pose['tracking_errors'].values()]))
        return dict(vertices_m=vertices[0].cpu().tolist(), joints_m=joints[0].cpu().tolist(),
            geometry_version=self.topology['geometry_version'], binding_version=self.topology['binding_version'],
            model_hash=self.candidate['model_hash'], surface_backend='qnbot', hand_side=self.candidate['side'],
            coordinate_frame='wrist_local', timestamp_ms=stamp, tactile_timestamp_ms=observation.get('tactile_timestamp_ms'),
            source_contact=observation.get('contact'), object_transform=observation.get('object_transform'),
            calibrated=True, sensor_mounting=dict(status='multi_pose_calibrated'),
            quality=dict(position_rmse_mm=float(error), source_reconstruction_ms=(time.perf_counter()-began)*1000))
