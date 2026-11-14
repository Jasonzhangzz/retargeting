#!/usr/bin/env python3
"""Recorded continuous EMF -> untouched sequential Wuji SDK -> immutable arrays."""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import threading
import time

import h5py
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def identity(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp.'+str(os.getpid()))
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    os.replace(tmp, path)


def stopped(root):
    if (Path(root)/'STOP').exists():
        raise InterruptedError('Campaign STOP requested')


def calibration(metadata, side, assets):
    dev = metadata['devices'][side]
    sn = dev.get('sn')
    if not sn:
        raise ValueError('Recording has no glove serial number: '+side)
    content = dev.get('urdf_content')
    source = dev.get('urdf_source') or 'builtin_default'
    if content:
        text = content.encode()
    elif source in ('builtin_default', 'builtin', 'default'):
        return dict(sn=sn, urdf=str(assets/f'builtin_{side}.urdf'), override=False)
    else:
        path = Path(dev.get('urdf_source_path') or '/__missing_recorded_urdf__')
        if not path.is_file():
            raise ValueError('Recorded override URDF unavailable: '+str(path))
        text = path.read_bytes()
    sha = hashlib.sha256(text).hexdigest()
    if dev.get('urdf_sha256') and sha != dev['urdf_sha256']:
        raise ValueError('Recorded calibration hash mismatch')
    path = assets/('recorded_'+sha+'.urdf')
    if not path.exists():
        temp = path.with_name(path.name+'.'+str(os.getpid())+'.'+str(threading.get_ident()))
        temp.write_bytes(text)
        os.replace(temp, path)
    return dict(sn=sn, urdf=str(path), override=True)


def inspect_episode(task):
    from retarget.learning.wuji_ik.real_data import continuous_emf_ranges
    from human_ik_reference.kinematics import HumanHandKinematics
    from select_wuji_clean_trajectories import receiver_fk_batch, quaternion_matrix
    root, row, source, policy_sha = task
    root, source = Path(root), Path(source)
    key = identity(str(source))
    cache = root/'input_audits'/(key+'.json')
    stat = source.stat()
    stamp = [stat.st_size, stat.st_mtime_ns]
    if cache.exists():
        old = json.loads(cache.read_text())
        if old.get('source_stat') == stamp and old.get('policy_sha') == policy_sha:
            return old
    result = dict(record_id=key, source_h5=str(source), source_stat=stamp, policy_sha=policy_sha,
                  device=row['device'], collection=row['collection'], episode=str(source.parent),
                  collection_id=row['device']+'/'+row['collection'], sides={}, reasons=[])
    try:
        manifest = json.loads((source.parent/'wuji_glove_offline_manifest.json').read_text())
        metadata = manifest.get('metadata') or manifest.get('recording_metadata')
        if not metadata:
            raise ValueError('Missing recording metadata')
        with h5py.File(source, 'r', locking=False) as h:
            for side in ('right', 'left'):
                try:
                    cal = calibration(metadata, side, root/'assets')
                    group = h[side]
                    p, q, c = (group[name][:] for name in ('emf_poses/position', 'emf_poses/orientation_xyzw', 'emf_poses/confidence'))
                    ts, seq = group['source_timestamp_ns'][:], group['source_seq'][:]
                    frame_valid = np.ones(len(p), bool)
                    if 'source_frame_id' in group:
                        ids = group['source_frame_id'].asstr()[:]
                        frame_valid &= ids == side[0]+'_hand_emf_tx'
                    consistency = dict(available=False)
                    if 'hand_joint_angles/angles' in group:
                        angles = group['hand_joint_angles/angles'][:]
                        qh = np.concatenate([angles[:, f, :5 if f == 0 else 4] for f in range(5)], -1)
                        model = HumanHandKinematics(side, cal['urdf'], sdk_tx_rounding=True)
                        pf, rf = receiver_fk_batch(model, np.nan_to_num(qh))
                        pe = np.linalg.norm(pf-p, axis=-1)
                        rr = rf.swapaxes(-1, -2)@quaternion_matrix(q)
                        re = np.arccos(np.clip((np.trace(rr, axis1=-2, axis2=-1)-1)*.5, -1, 1))
                        consistency = dict(available=True, finite_frames=int(np.isfinite(qh).all(1).sum()),
                                           consistent_frames=int((np.isfinite(qh).all(1) & (pe <= .01).all(1) & (re <= .7).all(1)).sum()),
                                           used_as_hard_gate=False,
                                           position_p99_m=np.nan_to_num(np.nanquantile(pe, .99, axis=0)).tolist(),
                                           orientation_p99_rad=np.nan_to_num(np.nanquantile(re, .99, axis=0)).tolist())
                    expected_step = 1 if manifest.get('schema') == 'legacy_measured_emf_import_v1' else manifest.get('offline_sdk_input_rate_divider')
                    ranges, quality = continuous_emf_ranges(p, q, c, ts, seq, frame_valid, expected_seq_step=expected_step)
                    windows = []
                    for start, end in ranges:
                        dt = np.diff(ts[start:end]).astype(float)*1e-9
                        activity = float(np.mean(np.linalg.norm(np.diff(p[start:end], axis=0), axis=-1)/dt[:, None]))
                        spread = float(np.mean(np.linalg.norm(p[start:end, :1]-p[start:end, 1:], axis=-1)))
                        windows.append(dict(start=start, end=end, activity=activity, spread=spread))
                    result['sides'][side] = dict(calibration=cal, quality=quality, kinematic=consistency, windows=windows)
                    if 'source_frame_id' in group:
                        result['sides'][side]['source_frame_ids'] = np.unique(ids).tolist()
                except Exception as exc:
                    result['reasons'].append(side+':'+type(exc).__name__+':'+str(exc))
        after = source.stat()
        if [after.st_size, after.st_mtime_ns] != stamp:
            raise RuntimeError('Source H5 changed while screening')
        result['source_sha256'] = digest(source)
    except Exception as exc:
        result['reasons'].append(type(exc).__name__+':'+str(exc))
    result['usable_frames'] = sum(w['end']-w['start'] for s in result['sides'].values() for w in s['windows'])
    write(cache, result)
    return result


def collect(root, inventory, policy_sha, workers):
    rng = np.random.default_rng(42)
    rows = inventory['collections']
    heldout = set(json.loads((root/'MANIFEST.json').read_text()).get('heldout_episode_paths', []))
    result_path = root/'candidate_pool.json'
    records, coverage = [], []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # One collection per work item avoids a large eager queue of H5 reads.
        def one_collection(row):
            paths = [p for p in row['h5_files'] if str(Path(p).parent) not in heldout]
            rng_local = np.random.default_rng(int(identity(row['path'])[:12], 16))
            rng_local.shuffle(paths)
            accepted, audited, frames = [], 0, 0
            side_frames = {'left': 0, 'right': 0}
            for path in paths:
                stopped(root)
                item = inspect_episode((root, row, path, policy_sha))
                audited += 1
                if item['usable_frames']:
                    accepted.append(item)
                    frames += item['usable_frames']
                    for side, detail in item['sides'].items():
                        side_frames[side] += sum(w['end']-w['start'] for w in detail['windows'])
                if len(accepted) >= 6 and frames >= 4000 and (min(side_frames.values()) >= 2000 or audited >= 96):
                    break
            return accepted, dict(device=row['device'], collection=row['collection'],
                                  h5_count=row['h5_count'], inspected=audited,
                                  usable_episodes=len(accepted), candidate_frames=frames,
                                  side_candidate_frames=side_frames,
                                  status='usable' if frames else 'no_h5' if not paths else 'no_quality_pass')
        futures = {executor.submit(one_collection, row): row for row in rows}
        for future in as_completed(futures):
            accepted, detail = future.result()
            records.extend(accepted)
            coverage.append(detail)
            write(result_path, dict(complete=False, policy_sha=policy_sha, records=records, collections=coverage))
            print('COLLECTION_SCREENED', json.dumps(detail), flush=True)
    records.sort(key=lambda x: x['record_id'])
    coverage.sort(key=lambda x: (x['device'], x['collection']))
    value = dict(complete=True, policy_sha=policy_sha, records=records, collections=coverage)
    write(result_path, value)
    return value


def reuse_screened_pool(root, previous, policy, policy_sha, workers):
    """Reuse sensor screening only after proving its code and source bytes unchanged."""
    previous = Path(previous)
    old_policy = json.loads((previous/'data_policy.json').read_text())
    old_pool = json.loads((previous/'candidate_pool.json').read_text())
    if not old_pool['complete'] or old_pool['policy_sha'] != identity(old_policy):
        raise ValueError('Prior screening pool has no complete matching policy')
    def function_hash(path, names):
        tree = ast.parse(Path(path).read_text())
        found = {node.name: ast.dump(node, include_attributes=False) for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names}
        if set(found) != set(names):
            raise ValueError('Missing source function in screening compatibility proof')
        return identity(found)
    from retarget.learning.wuji_ik import real_data, features
    import select_wuji_clean_trajectories as input_qc
    proofs = {}
    pairs = [
        ('sensor_ranges', previous/'source_snapshot/retarget/learning/wuji_ik/real_data.py',
         Path(real_data.__file__), ['continuous_emf_ranges']),
        ('screening', previous/'scripts/build_wuji_real_sdk_dataset.py', Path(__file__),
         ['digest', 'identity', 'calibration', 'inspect_episode', 'collect']),
    ]
    for name, old, new, functions in pairs:
        a, b = function_hash(old, functions), function_hash(new, functions)
        if a != b:
            raise ValueError('Sensor screening implementation changed: '+name)
        proofs[name] = a
    for name, path in [('features', features.__file__), ('input_qc', input_qc.__file__)]:
        if digest(path) != old_policy['helper_sha256'][name]:
            raise ValueError('Screening dependency changed: '+name)
        proofs[name] = digest(path)
    for key in old_policy.keys()-{'code_sha256', 'helper_sha256'}:
        if old_policy[key] != policy.get(key):
            raise ValueError('Screening policy changed: '+key)
    # Verify the entire candidate pool, rather than assuming path/stat identity.
    def verify(record):
        stopped(root)
        source = Path(record['source_h5'])
        stat = source.stat()
        if [stat.st_size, stat.st_mtime_ns] != record['source_stat'] or digest(source) != record['source_sha256']:
            raise ValueError('Prior source changed: '+str(source))
        for side, detail in record['sides'].items():
            asset = Path(detail['calibration']['urdf'])
            if digest(asset) != digest(root/'assets'/asset.name):
                raise ValueError('Calibration asset changed: '+side)
        return record['record_id']
    with ThreadPoolExecutor(max_workers=workers) as executor:
        checked = list(executor.map(verify, old_pool['records']))
    receipt = dict(previous_campaign=str(previous), previous_policy_sha=old_pool['policy_sha'],
        previous_pool_sha256=digest(previous/'candidate_pool.json'),
        source_files_verified=len(checked), unchanged_screening_function_hashes=proofs,
        reuse_scope='Sensor QC candidates only; new selection and all SDK labels regenerated',
        optimizer_updates=0)
    pool = dict(old_pool, policy_sha=policy_sha, screening_reuse=receipt)
    write(root/'screening_reuse.json', receipt)
    write(root/'candidate_pool.json', pool)
    return pool


def task_class(name):
    for word in ('stack', 'plug', 'pick', 'screw', 'wipe', 'writ', 'assembl', 'battery', 'bag', 'cap', 'drawer', 'key', 'doll', 'test'):
        if word in name.lower():
            return word
    return 'other'


def selection(root, pool, policy_sha, prefix_only=False):
    from retarget.learning.wuji_ik.real_data import allocate_even, diverse_windows
    rng = np.random.default_rng(42)
    records = pool['records']
    devices = sorted({r['device'] for r in records})
    if len(devices) != 2:
        raise ValueError('Both requested source devices must have usable data')
    all_windows = []
    for record in records:
        for side, detail in record['sides'].items():
            for window in detail['windows']:
                all_windows.append(dict(**window, record_id=record['record_id'], collection_id=record['collection_id'],
                    source_h5=record['source_h5'], source_sha256=record['source_sha256'],
                    source_stat=record['source_stat'], device=record['device'], collection=record['collection'],
                    side=side, calibration=detail['calibration'], source_seq_step=detail['quality']['nominal_source_seq_step']))
    # Reserve entire episodes, including both hands, before frame selection.
    reserved = set()
    for device in devices:
        by_collection = {}
        for r in records:
            if r['device'] == device:
                by_collection.setdefault(r['collection_id'], []).append(r)
        candidates = []
        for group in by_collection.values():
            if len(group) < 2:
                continue
            rng.shuffle(group)
            # Every side present in the collection must still be represented in training.
            for record in group:
                present = {s for s, d in record['sides'].items() if d['windows']}
                other = {s for r in group if r is not record for s, d in r['sides'].items() if d['windows']}
                if present == {'left', 'right'} and present <= other:
                    candidates.append(record)
                    break
        if len(candidates) < 4:
            raise ValueError('Too few disjoint validation episodes for '+device)
        # Greedy inverse-frequency coverage of date and task; deterministic random ties.
        rng.shuffle(candidates)
        counts = {}
        for _ in range(min(20, len(candidates))):
            def score(r):
                keys = [('month', r['collection'][:6]), ('task', task_class(r['collection']))]
                return sum(1/(1+counts.get(k, 0)) for k in keys)
            record = max(candidates, key=score)
            candidates.remove(record)
            reserved.add(record['record_id'])
            for k in [('month', record['collection'][:6]), ('task', task_class(record['collection']))]:
                counts[k] = counts.get(k, 0)+1
    selected = []
    for device in devices:
        for side in ('right', 'left'):
            for split, budget in [('val', 2500), ('train', 47500)]:
                candidates = [w for w in all_windows if w['device'] == device and w['side'] == side
                              and ((w['record_id'] in reserved) == (split == 'val'))]
                key_name = 'record_id' if split == 'val' else 'collection_id'
                groups = {}
                for w in candidates:
                    groups.setdefault(w[key_name], []).append(w)
                quotas = allocate_even({k: sum(w['end']-w['start'] for w in ws) for k, ws in groups.items()}, budget)
                for key, count in quotas.items():
                    for window in diverse_windows(groups[key], count, rng, prefix_only=prefix_only):
                        window['split'] = split
                        window['id'] = identity({k: window[k] for k in ('record_id', 'source_sha256', 'side', 'start', 'end', 'split')})[:24]
                        selected.append(window)
    selected.sort(key=lambda x: x['id'])
    train_groups = {w['record_id'] for w in selected if w['split'] == 'train'}
    val_groups = {w['record_id'] for w in selected if w['split'] == 'val'}
    if train_groups & val_groups:
        raise AssertionError('An episode leaks across the split')
    covered = {w['collection_id'] for w in selected if w['split'] == 'train'}
    expected = {r['collection_id'] for r in records}
    if covered != expected:
        raise AssertionError('An eligible collection is missing from training')
    if sum(w['end']-w['start'] for w in selected) != 200000:
        raise AssertionError('Expected exactly 200000 unique single-hand frames')
    for record_id in {w['record_id'] for w in selected}:
        for side in ('left', 'right'):
            intervals = sorted((w['start'], w['end']) for w in selected if w['record_id'] == record_id and w['side'] == side)
            if any(b > c for (_, b), (c, _) in zip(intervals[:-1], intervals[1:])):
                raise AssertionError('Repeated/overlapping source frames')
    value = dict(complete=True, policy_sha=policy_sha, segments=selected, frames=200000,
                 split_rows={'train': 190000, 'val': 10000, 'test': 0},
                 collection_coverage=pool['collections'], eligible_collections=len(expected),
                 selected_collections=len(covered), reserved_validation_episodes=sorted(val_groups),
                 selection='Equal device and hand totals; capacity-limited equal collection frame quotas; motion/spread tertile cycling; seed42. Validation holds out whole episodes, stratified by device, date and task. SDK reset at every selected contiguous segment start; no resampling.',
                 short_exact_budget_tails=sum(w['end']-w['start'] < 64 for w in selected))
    value['signature'] = identity(value)
    write(root/'selection.json', value)
    return value


def sdk_label(task):
    root, window, label_signature, save = task
    root = Path(root)
    out = root/'sdk_segments'/(window['id']+'.npz')
    receipt = out.with_suffix('.json')
    if save and receipt.exists() and out.exists():
        old = json.loads(receipt.read_text())
        if old.get('label_signature') == label_signature and old.get('npz_sha256') == digest(out):
            return old
        raise ValueError('Incompatible cached SDK segment: '+window['id'])
    stopped(root)
    import wuji_sdk as sdk
    from retarget.learning.wuji_ik.features import wrist_tx_transform, wrist_emf_features, urdf_limits, joint_names
    if sdk.__version__ != '2026.8.31':
        raise RuntimeError('Expected Wuji SDK 2026.8.31')
    sdk.set_log_level('error')
    source = Path(window['source_h5'])
    if digest(source) != window['source_sha256']:
        raise ValueError('Selected source H5 changed')
    side = window['side']
    with h5py.File(source, 'r', locking=False) as h:
        g = h[side]
        rows = slice(window['start'], window['end'])
        position = g['emf_poses/position'][rows]
        quaternion = g['emf_poses/orientation_xyzw'][rows]
        confidence = g['emf_poses/confidence'][rows]
        ts, seq = g['source_timestamp_ns'][rows], g['source_seq'][rows]
    cal = window['calibration']
    started = time.perf_counter()
    pipeline = sdk.WujiGlove.offline_pipeline(sn=cal['sn'], hand_side=side,
                    urdf_path=cal['urdf'] if cal['override'] else None)
    expected_source = 'override' if cal['override'] else 'builtin_default'
    if pipeline.urdf_source != expected_source:
        raise ValueError('SDK calibration source differs from selection')
    session = sdk.RetargetSession.for_hand(sdk.HandModel.WujiHand2,
                    side=sdk.Handedness.Left if side == 'left' else sdk.Handedness.Right)
    names = ('wrist', 'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip')+tuple(
        finger+'_'+joint for finger in ('index_finger', 'middle_finger', 'ring_finger', 'pinky')
        for joint in ('mcp', 'pip', 'dip', 'tip'))
    commands, skeletons = [], []
    for i in range(len(position)):
        if i % 128 == 0:
            stopped(root)
        frame = sdk.EmfPoseArray(header=sdk.FrameHeader(seq=int(seq[i]), timestamp_us=int(ts[i]//1000), frame_id=side[0]+'_hand_emf_tx'),
                    poses=[sdk.EmfPose(pose=sdk.Pose(position=position[i, f].tolist(),
                            orientation=sdk.Quaternion(*quaternion[i, f].tolist())), confidence=float(confidence[i, f])) for f in range(5)])
        skeleton = pipeline.hand_skeleton().compute(frame)
        by_name = {j.name: j for j in skeleton.joints}
        points = np.asarray([by_name[n].pose.position for n in names], dtype=np.float32)
        commands.append(session.step(points))
        skeletons.append(points)
    q = np.asarray(commands)
    limits = urdf_limits(root/'assets'/f'robot_{side}.urdf', side)
    if not np.isfinite(q).all() or q.shape != (len(position), 20):
        raise ValueError('Invalid SDK q output')
    if np.any(q < limits[:, 0]-1e-5) or np.any(q > limits[:, 1]+1e-5):
        raise ValueError('Official SDK label exceeds physical joint limits; no clipping applied')
    features = wrist_emf_features(position, quaternion, wrist_tx_transform(cal['urdf']))
    qn = (2*(q-limits[:, 0])/(limits[:, 1]-limits[:, 0])-1).astype(np.float32)
    arrays = dict(position_tx_m=position, orientation_xyzw=quaternion, confidence=confidence,
                  timestamp_ns=ts, source_seq=seq, q_sdk_rad=q, skeleton_sdk_m=np.asarray(skeletons),
                  features=features, q_normalized=qn,
                  frame_id=np.arange(window['start'], window['end'], dtype=np.int64))
    result = dict(id=window['id'], label_signature=label_signature, frames=len(q),
                  sdk_version=sdk.__version__, seconds=time.perf_counter()-started,
                  urdf_sha256=digest(cal['urdf']), joint_names=joint_names(side),
                  source_sha256=window['source_sha256'], source_window=window,
                  protocol='Fresh official offline pipeline and WujiHand2 RetargetSession per continuous segment; native chronological frames, SDK history and output filter retained; no LM/network labels.')
    if not save:
        return result, arrays
    temp = out.with_name(out.name+'.tmp.'+str(os.getpid()))
    with temp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    os.replace(temp, out)
    result['npz_sha256'] = digest(out)
    write(receipt, result)
    return result


def generate(root, plan, label_signature, workers):
    (root/'sdk_segments').mkdir(exist_ok=True)
    receipts = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('fork')) as executor:
        futures = [executor.submit(sdk_label, (root, w, label_signature, True)) for w in plan['segments']]
        for future in as_completed(futures):
            item = future.result()
            receipts.append(item)
            frames = sum(r['frames'] for r in receipts)
            write(root/'DATA_STATUS.json', dict(stage='sdk_generation', complete=False,
                  frames=frames, target_frames=200000, segments=len(receipts), elapsed_seconds=time.perf_counter()-started))
            print('SDK_LABELLED', len(receipts), frames, '/200000', flush=True)
    return receipts


def prepare(root, output, plan, label_signature, extra_schema=None):
    from retarget.learning.wuji_ik.features import urdf_limits, joint_names
    if output.exists():
        manifest = json.loads((output/'manifest.json').read_text())
        if manifest.get('label_signature') != label_signature or not manifest.get('complete'):
            raise ValueError('Existing prepared cache is incompatible')
        return manifest
    pending = output.with_name(output.name+'.building.'+str(os.getpid()))
    pending.mkdir(parents=True)
    schema = {'features.npy': (np.float32, (200000, 5, 9)), 'q_normalized.npy': (np.float32, (200000, 20)),
              'frame_id.npy': (np.int64, (200000,)), 'timestamp_ns.npy': (np.int64, (200000,)),
              'source_seq.npy': (np.int64, (200000,)), 'position_tx_m.npy': (np.float32, (200000, 5, 3)),
              'orientation_xyzw.npy': (np.float32, (200000, 5, 4)), 'confidence.npy': (np.float32, (200000, 5)),
              'q_sdk_rad.npy': (np.float64, (200000, 20)), 'skeleton_sdk_m.npy': (np.float32, (200000, 21, 3))}
    if extra_schema:
        schema.update(extra_schema)
    arrays = {name: np.lib.format.open_memmap(pending/name, mode='w+', dtype=dtype, shape=shape) for name, (dtype, shape) in schema.items()}
    segments, offset = [], 0
    for window in plan['segments']:
        path = root/'sdk_segments'/(window['id']+'.npz')
        receipt = json.loads(path.with_suffix('.json').read_text())
        if receipt['label_signature'] != label_signature or digest(path) != receipt['npz_sha256']:
            raise ValueError('SDK cache identity/hash mismatch')
        with np.load(path, allow_pickle=False) as data:
            n = window['end']-window['start']
            for filename, array in arrays.items():
                array[offset:offset+n] = data[filename[:-4]]
        segments.append(dict(offset=offset, length=n, source_row_start=window['start'],
            record=window['id'], shard=str(path), side=0 if window['side'] == 'right' else 1,
            split=window['split'], dataset=window['device'], group_id=window['record_id'],
            source_id=window['source_h5'], collection=window['collection'],
            calibration_urdf=window['calibration']['urdf'], urdf_sha256=receipt['urdf_sha256'],
            sdk_reset=True, motion_bin=window['motion_bin'], activity=window['activity'], spread=window['spread']))
        offset += n
    if offset != 200000:
        raise AssertionError('Wrong prepared frame count')
    for array in arrays.values():
        array.flush()
    arrays.clear()
    write(pending/'segments.json', segments)
    manifest = dict(schema='wuji_ik_segments_v1', complete=True, rows=200000, segments=len(segments),
        signature=label_signature, label_signature=label_signature, source_signature=plan['signature'],
        split_rows=plan['split_rows'], validation_fraction=.05,
        limits=np.stack([urdf_limits(root/'assets'/f'robot_{s}.urdf', s) for s in ('right', 'left')]).tolist(),
        joint_names={s: joint_names(s) for s in ('right', 'left')}, sides=['right', 'left'],
        feature='record-specific SDK calibration TX->wrist; position_m/0.1 + rotation first two columns',
        wrist_T_tx_policy='Per segment matched to archived calibration URDF, not a single shared TX transform',
        label_source='Official Wuji SDK2026.8.31 offline EMF human IK + stateful WujiHand2 RetargetSession.step; native alpha filter retained',
        selection=plan['selection'], eligible_collections=plan['eligible_collections'],
        collection_coverage=plan['collection_coverage'],
        source_metadata=str(root/'selection.json'), array_sha256={name: digest(pending/name) for name in schema})
    manifest['array_sha256']['segments.json'] = digest(pending/'segments.json')
    write(pending/'manifest.json', manifest)
    for path in pending.iterdir():
        path.chmod(0o444)
    pending.rename(output)
    return manifest


def validate(root, output, plan, label_signature):
    manifest = json.loads((output/'manifest.json').read_text())
    for name, expected in manifest['array_sha256'].items():
        if digest(output/name) != expected:
            raise ValueError('Prepared array content changed: '+name)
    segments = json.loads((output/'segments.json').read_text())
    groups = {split: {s['group_id'] for s in segments if s['split'] == split} for split in ('train', 'val')}
    if groups['train'] & groups['val']:
        raise ValueError('Episode split leakage')
    counts = {split: sum(s['length'] for s in segments if s['split'] == split) for split in ('train', 'val')}
    assert counts == {'train': 190000, 'val': 10000}
    q = np.load(output/'q_sdk_rad.npy', mmap_mode='r')
    qn = np.load(output/'q_normalized.npy', mmap_mode='r')
    features = np.load(output/'features.npy', mmap_mode='r')
    limits = np.asarray(manifest['limits'])
    roundtrip_max = 0.
    for s in segments:
        rows = slice(s['offset'], s['offset']+s['length'])
        lo, hi = limits[s['side'], :, 0], limits[s['side'], :, 1]
        recovered = (qn[rows]+1)*.5*(hi-lo)+lo
        roundtrip_max = max(roundtrip_max, float(abs(recovered-q[rows]).max()))
        if not np.isfinite(features[rows]).all() or not np.isfinite(qn[rows]).all() or abs(qn[rows]).max() > 1.00001:
            raise ValueError('Invalid prepared features/labels')
        data = np.load(root/'sdk_segments'/(s['record']+'.npz'))
        window = next(w for w in plan['segments'] if w['id'] == s['record'])
        if np.any(np.diff(data['timestamp_ns']) <= 0) or np.any(np.diff(data['timestamp_ns']) > 50_000_000):
            raise ValueError('Prepared segment crosses timestamp gaps')
        if np.any(np.diff(data['source_seq']) != window['source_seq_step']):
            raise ValueError('Prepared segment crosses missing source frames')
        data.close()
    if roundtrip_max > 5e-7:
        raise ValueError('Joint normalization roundtrip mismatch')
    repeats = []
    for device in sorted({w['device'] for w in plan['segments']}):
        for side in ('right', 'left'):
            window = next(w for w in plan['segments'] if w['device'] == device and w['side'] == side)
            _, repeated = sdk_label((root, window, label_signature, False))
            with np.load(root/'sdk_segments'/(window['id']+'.npz')) as old:
                for key in repeated:
                    np.testing.assert_array_equal(repeated[key], old[key])
            repeats.append(dict(device=device, side=side, segment=window['id'], frames=len(repeated['q_sdk_rad']), max_q_difference_rad=0.))
    result = dict(complete=True, passed=True, frames=200000, split_rows=counts,
        eligible_collections=plan['eligible_collections'], selected_collections=plan['selected_collections'],
        no_episode_split_overlap=True, no_source_frame_overlap=True,
        normalization_roundtrip_max_rad=roundtrip_max, sdk_full_segment_repeats=repeats,
        optimizer_updates=0, signature=label_signature)
    write(root/'data_validation.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    root = args.campaign
    sys.path[:0] = [str(args.reference)]
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    conf = json.loads((root/'MANIFEST.json').read_text())
    inventory = json.loads((root/'inventory.json').read_text())
    if not inventory['complete']:
        raise ValueError('Source inventory is incomplete')
    for folder in ('assets', 'input_audits'):
        (root/folder).mkdir(exist_ok=True)
    from retarget.learning.wuji_ik import real_data, features
    import select_wuji_clean_trajectories as input_qc
    policy = dict(confidence_min=.8, radius_m=[.01, .25], quaternion_norm_tolerance=.002,
        position_jump_max_m=.025, rotation_jump_max_rad=.5, timestamp_gap_max_s=.05,
        recorded_human_fk_policy='diagnostic only; old SDK/URDF-derived angles do not determine raw sensor validity',
        recorded_human_fk_diagnostic_position_m=.01, recorded_human_fk_diagnostic_rotation_rad=.7,
        segment_frames=[64, 1024], selection_seed=42, frames=200000, val_fraction=.05,
        budget_policy='exact 50000 frames per source-device/hand stratum; train47500 and val2500; small budget tails permitted',
        sdk_reset='Every selected contiguous segment starts fresh, matching network context; no hidden prehistory',
        code_sha256=digest(__file__), helper_sha256={
            'real_data': digest(real_data.__file__), 'features': digest(features.__file__),
            'input_qc': digest(input_qc.__file__)})
    policy_sha = identity(policy)
    write(root/'data_policy.json', policy)
    pool_path = root/'candidate_pool.json'
    pool = json.loads(pool_path.read_text()) if pool_path.exists() else None
    if not pool or not pool.get('complete') or pool.get('policy_sha') != policy_sha:
        if conf.get('reuse_screened_campaign'):
            pool = reuse_screened_pool(root, conf['reuse_screened_campaign'], policy, policy_sha, args.workers)
        else:
            pool = collect(root, inventory, policy_sha, args.workers)
    plan_path = root/'selection.json'
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else None
    if not plan or plan.get('policy_sha') != policy_sha:
        plan = selection(root, pool, policy_sha)
    motion_frames = np.bincount([w['motion_bin'][0] for w in plan['segments']],
        weights=[w['end']-w['start'] for w in plan['segments']], minlength=3).astype(int)
    spread_frames = np.bincount([w['motion_bin'][1] for w in plan['segments']],
        weights=[w['end']-w['start'] for w in plan['segments']], minlength=3).astype(int)
    diversity = dict(motion_tertile_frames=motion_frames.tolist(), spread_tertile_frames=spread_frames.tolist(),
        bin_definition='Candidate-window activity/spread tertiles within each collection/hand/split sampling group',
        complete=True, optimizer_updates=0)
    write(root/'selection_diversity.json', diversity)
    if min(motion_frames) < 20000 or max(motion_frames) > 100000:
        raise ValueError('Motion sampling is excessively imbalanced; SDK generation and training blocked')
    import wuji_sdk as sdk
    binaries = {p.name: digest(p) for p in Path(sdk.__file__).parent.glob('*.so')}
    label_signature = identity(dict(policy=policy_sha, selection=plan['signature'], sdk_version=sdk.__version__,
                                    sdk_binaries=binaries, assets={p.name: digest(p) for p in (root/'assets').glob('*.urdf')}))
    write(root/'sdk_provenance.json', dict(sdk_version=sdk.__version__, sdk_file=sdk.__file__, binaries=binaries, label_signature=label_signature))
    generate(root, plan, label_signature, args.workers)
    output = Path(conf['dataset'])
    prepare(root, output, plan, label_signature)
    validated = validate(root, output, plan, label_signature)
    write(root/'DATA_STATUS.json', dict(stage='complete', **validated))
    print('DATASET_COMPLETE', json.dumps(validated), flush=True)


if __name__ == '__main__':
    main()
