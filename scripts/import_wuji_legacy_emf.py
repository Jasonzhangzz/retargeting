#!/usr/bin/env python3
"""Recover measured TX EMF from older camera-sampled JSON lacking offline H5."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
import re
from pathlib import Path
import sys

import h5py
import numpy as np

from build_wuji_real_sdk_dataset import digest, identity, write, stopped


def contains_recorded_emf(text):
    """Legacy writer uses literal ASCII field names; null fields carry no EMF."""
    return re.search(r'"(?:left|right)_glove_emf_poses"\s*:\s*\{', text) is not None


def import_collection(task):
    from retarget.learning.wuji_ik.real_data import continuous_emf_ranges
    import wuji_sdk as sdk
    root, row = task
    root = Path(root)
    sdk.set_log_level('error')
    folder = Path(row['path'])
    episodes = sorted(p for p in folder.iterdir() if p.is_dir() and (p/(p.name+'.json')).is_file())
    rng = np.random.default_rng(int(identity(str(folder))[:12], 16))
    rng.shuffle(episodes)
    recovered, audit = [], []
    for episode in episodes:
        stopped(root)
        source = episode/(episode.name+'.json')
        output = root/'legacy_inputs'/row['device']/row['collection']/episode.name/'wuji_glove_offline.h5'
        receipt = output.with_suffix('.source.json')
        if output.exists() and receipt.exists():
            recovered.append(str(output))
            if len(recovered) >= 6:
                break
            continue
        try:
            text = source.read_text()
            if not contains_recorded_emf(text):
                audit.append(dict(source=str(source), reason='no_recorded_TX_EMF_object'))
                continue
            obj = json.loads(text)
            data = obj.get('data') or []
            sides = {}
            for side in ('left', 'right'):
                ps, qs, cs, times, rows = [], [], [], [], []
                for i, frame in enumerate(data):
                    emf = frame.get(side+'_glove_emf_poses')
                    if not isinstance(emf, dict) or len(emf.get('poses', [])) != 5:
                        continue
                    poses = emf['poses']
                    if not all('confidence' in pose for pose in poses):
                        continue
                    timestamp = frame.get('timestamp')
                    if not isinstance(timestamp, int) or timestamp < 10**16:
                        raise ValueError('Legacy timestamp is not an exact integer nanosecond timestamp')
                    ps.append([[pose['pose']['position'][a] for a in 'xyz'] for pose in poses])
                    qs.append([[pose['pose']['orientation'][a] for a in 'xyzw'] for pose in poses])
                    cs.append([pose['confidence'] for pose in poses])
                    times.append(timestamp)
                    rows.append(i)
                if not ps:
                    continue
                p, q, c = np.asarray(ps, np.float32), np.asarray(qs, np.float32), np.asarray(cs, np.float32)
                ts, seq = np.asarray(times, np.int64), np.asarray(rows, np.int64)
                ranges, _ = continuous_emf_ranges(p, q, c, ts, seq, expected_seq_step=1)
                if not ranges:
                    continue
                # No original serial is invented. The API needs a string; explicit
                # default URDF makes the marked offline identifier geometry-neutral.
                api_sn = 'OFFLINE_LEGACY_UNRECORDED_'+side.upper()
                urdf = root/'assets'/f'builtin_{side}.urdf'
                pipeline = sdk.WujiGlove.offline_pipeline(sn=api_sn, hand_side=side, urdf_path=str(urdf))
                angles = np.full((len(p), 5, 5), np.nan, np.float32)
                for start, end in ranges:
                    pipeline = sdk.WujiGlove.offline_pipeline(sn=api_sn, hand_side=side, urdf_path=str(urdf))
                    for i in range(start, end):
                        frame = sdk.EmfPoseArray(header=sdk.FrameHeader(seq=int(seq[i]), timestamp_us=int(ts[i]//1000), frame_id=side[0]+'_hand_emf_tx'),
                            poses=[sdk.EmfPose(pose=sdk.Pose(position=p[i, j].tolist(), orientation=sdk.Quaternion(*q[i, j].tolist())), confidence=float(c[i, j])) for j in range(5)])
                        result = pipeline.hand_joint_angles().compute(frame)
                        for f, finger in enumerate(result.fingers):
                            values = np.asarray(finger.angles)[:5 if f == 0 else 4]
                            angles[i, f, :len(values)] = values
                sides[side] = dict(position=p, quaternion=q, confidence=c, timestamp=ts, source_row=seq,
                                   angles=angles, api_sn=api_sn)
            if not sides:
                audit.append(dict(source=str(source), reason='no_64_frame_quality_pass_raw_EMF_run'))
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            temp = output.with_name(output.name+'.tmp')
            devices = {}
            with h5py.File(temp, 'w') as h:
                h.attrs['schema'] = 'legacy_measured_emf_import_v1'
                h.attrs['original_json'] = str(source)
                h.attrs['original_json_sha256'] = digest(source)
                h.attrs['timestamp_source'] = 'original camera-sampled JSON integer timestamp; EMF message header was not retained'
                for side, d in sides.items():
                    g = h.create_group(side)
                    for name, key in [('emf_poses/position', 'position'), ('emf_poses/orientation_xyzw', 'quaternion'),
                                      ('emf_poses/confidence', 'confidence'), ('source_timestamp_ns', 'timestamp'),
                                      ('source_seq', 'source_row'), ('hand_joint_angles/angles', 'angles')]:
                        g.create_dataset(name, data=d[key])
                    g.create_dataset('source_frame_id', data=np.asarray([side[0]+'_hand_emf_tx']*len(d['position']), dtype=h5py.string_dtype()))
                    urdf = root/'assets'/f'builtin_{side}.urdf'
                    devices[side] = dict(sn=d['api_sn'], original_serial=None, api_identifier_is_placeholder=True,
                        urdf_source='override', urdf_source_path=str(urdf), urdf_content=urdf.read_text(), urdf_sha256=digest(urdf))
            os.replace(temp, output)
            metadata = dict(schema='legacy_measured_emf_import_v1', sdk_version=sdk.__version__,
                original_source_json=str(source), original_json_sha256=digest(source),
                timing='Original camera-sampled EMF observations; no interpolation; not native EMF header timestamps',
                calibration='Original serial/calibration unavailable in legacy JSON; explicit SDK default side URDF, API placeholder clearly marked',
                metadata=dict(devices=devices))
            write(output.parent/'wuji_glove_offline_manifest.json', metadata)
            write(receipt, metadata)
            recovered.append(str(output))
            print('LEGACY_IMPORTED', row['device'], row['collection'], len(recovered), flush=True)
            if len(recovered) >= 6:
                break
        except Exception as exc:
            audit.append(dict(source=str(source), reason=type(exc).__name__+':'+str(exc)))
    return dict(device=row['device'], collection=row['collection'], json_episode_count=len(episodes),
                h5_files=recovered, inspected_failures=audit,
                status='recovered_camera_sampled_EMF' if recovered else 'no_recoverable_JSON_EMF')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    root = args.campaign
    original = root/'inventory_native.json'
    if not original.exists():
        original.write_bytes((root/'inventory.json').read_bytes())
    inventory = json.loads(original.read_text())
    rows = [row for row in inventory['collections'] if row['h5_count'] == 0]
    receipts = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('fork')) as pool:
        futures = [pool.submit(import_collection, (root, row)) for row in rows]
        for future in as_completed(futures):
            result = future.result()
            receipts.append(result)
            write(root/'legacy_import_audit.json', dict(complete=False, collections=receipts))
    for result in receipts:
        row = next(r for r in inventory['collections'] if (r['device'], r['collection']) == (result['device'], result['collection']))
        row['h5_files'] = result['h5_files']
        row['h5_count'] = len(result['h5_files'])
        row['legacy_import'] = result['status']
    inventory.update(legacy_import_complete=True, collections_with_h5=sum(r['h5_count'] > 0 for r in inventory['collections']),
                     total_h5=sum(r['h5_count'] for r in inventory['collections']))
    write(root/'inventory.json', inventory)
    write(root/'legacy_import_audit.json', dict(complete=True, collections=receipts))
    print('LEGACY_IMPORT_COMPLETE', inventory['collections_with_h5'], '/', len(inventory['collections']), flush=True)


if __name__ == '__main__':
    main()
