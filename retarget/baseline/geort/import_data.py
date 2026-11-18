"""Import an actual archived EMF session using an explicitly matching profile."""
import hashlib
import json
from pathlib import Path
import numpy as np
from retarget.baseline.geort.adapter import SourceAdapter, UPSTREAM
from retarget.baseline.geort.training import atomic_json


def import_raw(root, recording, profile, dataset_id):
    from moxian_web.geort_service import safe_id
    from moxian_web.moxian_replay import MoxianReplayReconstructor
    from moxian_web.studio_emf_solver import REVISION as solver_revision
    root=Path(root);recording=Path(recording);profile=Path(profile)
    candidate=json.loads(profile.read_text());reconstructor=MoxianReplayReconstructor(candidate)
    folder=root/'recordings'/safe_id(dataset_id);folder.mkdir(parents=True,exist_ok=False)
    points=[];joints=[];angles=[];stamps=[];last=-float('inf');invalid=0;source=None
    with recording.open() as stream:
        for line in stream:
            raw=json.loads(line)
            row=raw.get('hand_model_input') or raw
            if not row.get('fingers'):invalid+=1;continue
            stamp=float(row['timestamp_ms'])
            if stamp<=last:continue
            frame=reconstructor.frame(row);topology=reconstructor.topology
            adapter=SourceAdapter(topology);identity=adapter.identity(candidate['serial'],frame)
            if source is not None and source!=identity:raise ValueError('导入过程中源标定发生变化')
            source=identity;points.append(adapter.points(frame['joints_m']));joints.append(frame['joints_m'])
            angles.append(frame['source_joint_positions_rad']);stamps.append(stamp);last=stamp
    if not points:raise ValueError('归档录制没有兼容的原始EMF帧')
    np.save(folder/'points.npy',np.asarray(points,np.float32));np.save(folder/'timestamps.npy',np.asarray(stamps,np.float64))
    np.savez(folder/'motion.npz',qpos=np.asarray(angles,np.float32),joints=np.asarray(joints,np.float32))
    atomic_json(folder/'profile.json',candidate);atomic_json(folder/'topology.json',reconstructor.topology)
    manifest=dict(id=dataset_id,label='归档EMF · '+recording.name,source=source,upstream=UPSTREAM,complete=True,
                  frames=len(points),invalid_frames=invalid,points_sha256=hashlib.sha256((folder/'points.npy').read_bytes()).hexdigest(),
                  provenance=dict(kind='archived_real_emf_reconstructed',path=str(recording.resolve()),
                     sha256=hashlib.sha256(recording.read_bytes()).hexdigest(),profile_sha256=hashlib.sha256(profile.read_bytes()).hexdigest(),
                     source_ik=solver_revision,synthetic=False))
    atomic_json(folder/'manifest.json',manifest)
    return manifest
