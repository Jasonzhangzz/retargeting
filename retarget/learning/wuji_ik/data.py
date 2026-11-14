"""Verified SDK shards -> immutable mmap arrays, complete contiguous segments."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from .features import SIDES, check_joint_order, joint_names, urdf_limits, wrist_emf_features, wrist_tx_transform

SCHEMA = 'wuji_ik_segments_v1'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path); tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def contiguous_ranges(frames, resets):
    frames, resets = np.asarray(frames), np.asarray(resets, bool)
    if frames.ndim != 1 or resets.shape != frames.shape or not len(frames):
        raise ValueError('Invalid frame/reset arrays')
    if np.any(np.diff(frames) <= 0):
        raise ValueError('Temporal record is not strictly ordered')
    cuts = np.r_[0, np.flatnonzero((np.diff(frames) != 1) | resets[1:]) + 1, len(frames)]
    return list(zip(cuts[:-1].tolist(), cuts[1:].tolist()))


@contextmanager
def staged_shard(path, scratch):
    if scratch is None:
        yield path
    else:
        Path(scratch).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='wuji_ik_read_', dir=scratch) as temp:
            target = Path(temp) / Path(path).name
            shutil.copyfile(path, target)
            yield target


def prepare(source, output, right_urdf, left_urdf, scratch=None):
    source, output = Path(source), Path(output)
    manifest_path = source / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get('complete') or manifest.get('errors'):
        raise ValueError('The SDK corpus must be complete without shard errors')
    limits = np.stack([urdf_limits(p, s) for p, s in zip((right_urdf, left_urdf), SIDES)])
    transforms = [wrist_tx_transform(source / f'assets/glove_{s}.urdf') for s in SIDES]
    contract = dict(schema=SCHEMA, source_signature=manifest['signature'], source_manifest_sha256=sha256(manifest_path),
                    limits=limits.tolist(), wrist_T_tx=[t.tolist() for t in transforms],
                    robot_urdf_sha256=[sha256(p) for p in (right_urdf, left_urdf)],
                    glove_urdf_sha256=[sha256(source / f'assets/glove_{s}.urdf') for s in SIDES],
                    joint_names={s: joint_names(s) for s in SIDES}, feature='position_m/0.1 + rotation_first_two_columns',
                    sides=list(SIDES), selection='temporal only; original split; split at frame gaps or SDK reset')
    signature = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    if output.exists():
        old = json.loads((output / 'manifest.json').read_text())
        if old.get('signature') != signature or not old.get('complete'):
            raise ValueError('Existing prepared cache is incompatible or incomplete')
        for name, expected in old['array_sha256'].items():
            if sha256(output / name) != expected:
                raise ValueError('Changed prepared array: ' + name)
        return old
    output.parent.mkdir(parents=True, exist_ok=True)
    target_rows = sum(v for k, v in manifest['counts'].items() if ':temporal:' in k)
    pending = Path(tempfile.mkdtemp(prefix=output.name + '.building_', dir=output.parent))
    try:
        features = np.lib.format.open_memmap(pending / 'features.npy', mode='w+', dtype=np.float32, shape=(target_rows, 5, 9))
        targets = np.lib.format.open_memmap(pending / 'q_normalized.npy', mode='w+', dtype=np.float32, shape=(target_rows, 20))
        frame_array = np.lib.format.open_memmap(pending / 'frame_id.npy', mode='w+', dtype=np.int64, shape=(target_rows,))
        segments = []; offset = 0; split_rows = {}; domain_rows = {}
        seen_records = set()
        for shard_info in sorted(manifest['shards'], key=lambda x: x['shard']):
            path = source / 'shards' / Path(shard_info['shard']).name
            receipt = json.loads(path.with_suffix('.json').read_text())
            if (receipt['sha256'] != shard_info['sha256'] or receipt['signature'] != manifest['signature']):
                raise ValueError('Shard receipt mismatch: ' + str(path))
            with staged_shard(path, scratch) as local:
                if sha256(local) != receipt['sha256']:
                    raise ValueError('Shard checksum mismatch: ' + str(path))
                with h5py.File(local, 'r') as f:
                    if not f.attrs.get('complete') or f.attrs['signature'] != manifest['signature']:
                        raise ValueError('Incomplete or incompatible shard')
                    for key, g in f['records'].items():
                        if g.attrs['kind'] != 'temporal':
                            continue
                        if key in seen_records:
                            raise ValueError('Duplicate source record')
                        seen_records.add(key)
                        side = str(g.attrs['side']); sid = SIDES.index(side)
                        split = str(g.attrs['split'])
                        if split not in ('train', 'val', 'test'):
                            raise ValueError('Unknown source split')
                        check_joint_order(g, side)
                        q = g['q'][:]; frames = g['frame_id'][:]; n = len(q)
                        if q.shape != (n, 20) or not np.isfinite(q).all():
                            raise ValueError('Invalid SDK q')
                        lo, hi = limits[sid, :, 0], limits[sid, :, 1]
                        if np.any(q < lo - 1e-5) or np.any(q > hi + 1e-5):
                            raise ValueError(f'SDK labels outside pinned physical limits: {key}')
                        if offset + n > target_rows:
                            raise ValueError('More temporal rows than source manifest')
                        features[offset:offset+n] = wrist_emf_features(g['emf_position'][:], g['emf_orientation_xyzw'][:], transforms[sid])
                        targets[offset:offset+n] = 2 * (q - lo) / (hi - lo) - 1
                        frame_array[offset:offset+n] = frames
                        for start, end in contiguous_ranges(frames, g['sdk_state_reset'][:]):
                            segments.append(dict(offset=offset+start, length=end-start, source_row_start=start,
                                record=key, shard=path.name, side=sid, split=split, dataset=str(g.attrs['dataset']),
                                group_id=str(g.attrs['group_id']), source_id=str(g.attrs['source_id'])))
                        offset += n
                        split_rows[split] = split_rows.get(split, 0) + n
                        domain_rows[str(g.attrs['dataset'])] = domain_rows.get(str(g.attrs['dataset']), 0) + n
            print(f'prepare {path.name}: {offset}/{target_rows} temporal rows', flush=True)
        if offset != target_rows:
            raise ValueError('Temporal rows disagree with source manifest')
        for a in (features, targets, frame_array):
            a.flush()
        del features, targets, frame_array
        write_json(pending / 'segments.json', segments)
        result = dict(**contract, signature=signature, complete=True, rows=offset, segments=len(segments),
                      split_rows=split_rows, dataset_rows=domain_rows,
                      array_sha256={n: sha256(pending / n) for n in ('features.npy', 'q_normalized.npy', 'frame_id.npy', 'segments.json')})
        write_json(pending / 'manifest.json', result)
        os.replace(pending, output)
        return result
    except BaseException:
        shutil.rmtree(pending, ignore_errors=True)
        raise


class SegmentDataset(Dataset):
    def __init__(self, root, split):
        self.root = Path(root)
        self.metadata = json.loads((self.root / 'manifest.json').read_text())
        if self.metadata.get('schema') != SCHEMA or not self.metadata.get('complete'):
            raise ValueError('Expected a complete prepared Wuji IK cache')
        self.segments = [(i, s) for i, s in enumerate(json.loads((self.root / 'segments.json').read_text())) if s['split'] == split]
        self.handles = None; self.pid = None

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, index):
        loss_range = None
        if isinstance(index, tuple):
            index, start, end = index; loss_range = (start, end)
        if self.handles is None or self.pid != os.getpid():
            self.handles = {n: np.load(self.root / n, mmap_mode='r') for n in ('features.npy', 'q_normalized.npy', 'frame_id.npy')}
            if self.metadata.get('context_rows', 0):
                self.handles['context_features.npy'] = np.load(self.root/'context_features.npy', mmap_mode='r')
            self.pid = os.getpid()
        identity, s = self.segments[index]; rows = slice(s['offset'], s['offset']+s['length'])
        features = torch.from_numpy(self.handles['features.npy'][rows].copy())
        target = torch.from_numpy(self.handles['q_normalized.npy'][rows].copy())
        frames = torch.from_numpy(self.handles['frame_id.npy'][rows].copy())
        context = s.get('context_length', 0)
        if context:
            begin = s['context_offset']
            prefix = torch.from_numpy(self.handles['context_features.npy'][begin:begin+context].copy())
            features = torch.cat((prefix, features))
            target = torch.cat((torch.zeros(context, 20, dtype=target.dtype), target))
            # Context carries measured EMF only; it owns no q labels or smooth triples.
            frames = torch.cat((torch.full((context,), -1, dtype=frames.dtype), frames))
            start, end = loss_range if loss_range is not None else (0, s['length'])
            loss_range = (start+context, end+context)
        return dict(features=features, target=target, frames=frames,
                    side=s['side'], dataset=s['dataset'], segment_id=identity, loss_range=loss_range)

    def __getstate__(self):
        state = self.__dict__.copy(); state['handles'] = None; state['pid'] = None
        return state


def collate_segments(samples):
    lengths = torch.tensor([len(s['features']) for s in samples], dtype=torch.long)
    b, t = len(samples), int(lengths.max())
    features = torch.zeros(b, t, 5, 9); target = torch.zeros(b, t, 20)
    frames = torch.full((b, t), -1, dtype=torch.long)
    for i, s in enumerate(samples):
        n = len(s['features']); features[i, :n] = s['features']; target[i, :n] = s['target']; frames[i, :n] = s['frames']
    valid = torch.arange(t)[None, :] < lengths[:, None]
    for i,s in enumerate(samples):
        if s.get('loss_range') is not None:
            start,end=s['loss_range']
            if not 0 <= start <= end <= lengths[i]: raise ValueError('Invalid loss range')
            valid[i] &= (torch.arange(t)>=start)&(torch.arange(t)<end)
    return dict(features=features, target=target, frames=frames, lengths=lengths,
                valid=valid, side=torch.tensor([s['side'] for s in samples]),
                dataset=[s['dataset'] for s in samples], segment_id=[s['segment_id'] for s in samples])
