#!/usr/bin/env python3
"""Generate untouched SDK reference commands for fixed native EMF recordings."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import h5py


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args(); root = args.root
    sys.path[:0] = [str(root/'sdk831'), str(root/'reference')]
    import wuji_sdk as sdk
    from human_ik_reference.kinematics import HumanHandKinematics
    if sdk.__version__ != '2026.8.31':
        raise RuntimeError('Expected the original project SDK version 2026.8.31')
    sdk.set_log_level('error')
    selection = json.loads((root/'selection.json').read_text())
    names = ('wrist', 'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip')+tuple(
        finger+'_'+joint for finger in ('index_finger','middle_finger','ring_finger','pinky')
        for joint in ('mcp','pip','dip','tip'))
    manifest = dict(sdk_version=sdk.__version__, sdk_file=sdk.__file__,
        sdk_binary_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'sdk831/wuji_sdk').glob('*.so')},
        protocol='Official offline human IK + RetargetSession.WujiHand2, continuous native frames; fresh state per episode/side/configuration; SDK output including native filtering is preserved.', streams=[])
    for configuration in ('builtin', 'hl_urdf'):
        for trajectory in selection['trajectories']:
            input_path=root/trajectory.get('cached_h5_relative',f'inputs/trajectory_{trajectory["index"]}/wuji_glove_offline.h5')
            if not input_path.exists():input_path=Path(trajectory['cached_h5'])
            if not input_path.exists():input_path=Path(trajectory['source_h5'])
            with h5py.File(input_path, 'r', locking=False) as h:
                for side in ('left','right'):
                    output = root/'gt'/configuration/f'trajectory_{trajectory["index"]}_{side}.npz'
                    output.parent.mkdir(parents=True,exist_ok=True)
                    urdf = root/'assets'/f'{configuration}_{side}.urdf'
                    geometry = HumanHandKinematics(side,urdf,sdk_tx_rounding=True)
                    p = h[side]['emf_poses/position'][:]; rotation = h[side]['emf_poses/orientation_xyzw'][:]
                    confidence = h[side]['emf_poses/confidence'][:]
                    ts = h[side]['source_timestamp_ns'][:]; seq = h[side]['source_seq'][:]
                    ids = [x.decode() if isinstance(x,bytes) else str(x) for x in h[side]['source_frame_id'][:]]
                    if not np.all(np.diff(ts)>0) or any(x != side[0]+'_hand_emf_tx' for x in ids):
                        raise ValueError('Expected native chronological TX-coordinate EMF poses')
                    sn = trajectory['recording_metadata']['devices'][side]['sn']
                    initialize_start=time.perf_counter()
                    pipeline = sdk.WujiGlove.offline_pipeline(sn=sn,hand_side=side,
                        urdf_path=str(urdf) if configuration=='hl_urdf' else None)
                    expected = 'override' if configuration=='hl_urdf' else 'builtin_default'
                    if pipeline.urdf_source != expected:raise ValueError('Unexpected SDK model source')
                    session = sdk.RetargetSession.for_hand(sdk.HandModel.WujiHand2,
                        side=sdk.Handedness.Left if side=='left' else sdk.Handedness.Right)
                    initialize_seconds=time.perf_counter()-initialize_start
                    commands=[]; human_q=[]; skeletons=[]; rotations=[]; rx_position=[]; rx_rotation=[]
                    started=time.perf_counter(); fk_checks=[];sdk_solve_seconds=0.;sdk_pipeline_seconds=0.
                    for i in range(len(p)):
                        pipeline_start=time.perf_counter()
                        frame=sdk.EmfPoseArray(header=sdk.FrameHeader(seq=int(seq[i]),timestamp_us=int(ts[i]//1000),frame_id=ids[i]),
                            poses=[sdk.EmfPose(pose=sdk.Pose(position=p[i,j].tolist(),orientation=sdk.Quaternion(*rotation[i,j].tolist())),confidence=float(confidence[i,j])) for j in range(5)])
                        solve_start=time.perf_counter()
                        skeleton=pipeline.hand_skeleton().compute(frame)
                        by_name={j.name:j for j in skeleton.joints}
                        points=np.asarray([by_name[n].pose.position for n in names],dtype=np.float32)
                        command=session.step(points)
                        solve_end=time.perf_counter();sdk_solve_seconds+=solve_end-solve_start
                        sdk_pipeline_seconds+=solve_end-pipeline_start
                        angles=pipeline.hand_joint_angles().compute(frame)
                        q=np.concatenate([np.asarray(v.angles)[:5 if f==0 else 4] for f,v in enumerate(angles.fingers)])
                        fk=geometry.forward(q)
                        if i in (0,len(p)//2,len(p)-1):
                            difference=float(np.abs(fk['skeleton'][:,:3,3]-points).max());fk_checks.append(difference)
                            if difference>2e-6:raise ValueError(f'SDK and selected URDF disagree: {configuration} {side} {difference} m')
                        commands.append(command);human_q.append(q);skeletons.append(points)
                        rotations.append([[getattr(by_name[n].pose.orientation,a) for a in 'xyzw'] for n in names])
                        rx_position.append(fk['tx_rx'][:,:3,3]);rx_rotation.append(fk['tx_rx'][:,:3,:3])
                        if (i+1)%250==0:print(configuration,trajectory['index'],side,i+1,'/',len(p),flush=True)
                    arrays=dict(q_sdk_rad=np.asarray(commands),human_q_sdk_rad=np.asarray(human_q),
                        skeleton_sdk_m=np.asarray(skeletons),skeleton_sdk_xyzw=np.asarray(rotations),
                        rx_sdk_tx_position_m=np.asarray(rx_position),rx_sdk_tx_rotation=np.asarray(rx_rotation),
                        position_tx_m=p,orientation_xyzw=rotation,confidence=confidence,timestamp_ns=ts,source_seq=seq)
                    if not all(np.isfinite(v).all() for v in arrays.values()):raise ValueError('Nonfinite SDK or source data')
                    if output.exists():
                        previous=np.load(output)
                        if not np.array_equal(previous['q_sdk_rad'],arrays['q_sdk_rad']):
                            raise ValueError('SDK repeat changed reference commands')
                    np.savez_compressed(output,**arrays)
                    detail=dict(configuration=configuration,trajectory=trajectory['index'],side=side,frames=len(p),
                        seconds=time.perf_counter()-started,urdf=str(urdf),urdf_sha256=hashlib.sha256(urdf.read_bytes()).hexdigest(),
                        sdk_solve_seconds=sdk_solve_seconds,sdk_end_to_end_seconds=sdk_pipeline_seconds+initialize_seconds,
                        sdk_initialize_seconds=initialize_seconds,
                        sdk_urdf_source=pipeline.urdf_source,sdk_urdf_source_path=pipeline.urdf_source_path,
                        sdk_q_to_urdf_fk_max_abs_m=max(fk_checks),nonunit_source_seq_increments=int(np.sum(np.diff(seq)!=1)),
                        source_seq_increment_values=np.unique(np.diff(seq)).tolist(),output=str(output))
                    manifest['streams'].append(detail)
                    (root/'gt_manifest.json').write_text(json.dumps(manifest,indent=2))
                    print('STREAM_COMPLETE',json.dumps(detail),flush=True)
    (root/'GT_COMPLETE').write_text('complete\n')
    print('GT_COMPLETE',flush=True)


if __name__=='__main__':main()
