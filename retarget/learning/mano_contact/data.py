"""Transactional HDF5 groups, leakage groups, temporal windows and statistics."""
from pathlib import Path
import hashlib
import json
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

SCHEMA = 'pose_only_mano_contact_v1'


def split_for(group_id, seed=42):
    value = int(hashlib.sha256(f'{seed}:{group_id}'.encode()).hexdigest()[:16], 16) % 100
    return 'train' if value < 80 else 'val' if value < 90 else 'test'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    os.replace(tmp, p)


class CorpusWriter:
    """One writer. Completed groups are immutable; interrupted groups roll back."""
    def __init__(self, path):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        import fcntl
        self.lock = open(str(path)+'.lock', 'a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.file = h5py.File(path, 'a')
        if 'schema' in self.file.attrs and self.file.attrs['schema'] != SCHEMA:
            raise ValueError('Incompatible corpus')
        self.file.attrs.update(schema=SCHEMA, input='physical_wrist_xyz_m, wrist_rotvec_rad, absolute_local_joint_rotvec_rad',
                               max_history=8, contact_threshold_m=.01, seed=42)
        self.file.require_group('records')
        # Any preparation session can append records; cached statistics must
        # therefore be recomputed before the corpus is declared complete.
        if 'statistics' in self.file:del self.file['statistics']
        self.file.attrs['preparation_state']='building'
        if '__pending' in self.file:
            del self.file['__pending']

    @staticmethod
    def key(source_id):
        return hashlib.sha256(source_id.encode()).hexdigest()

    def contains(self, source_id):
        return self.key(source_id) in self.file['records']

    def add(self, source_id, pose, contact=None, valid=None, frames=None, metadata=None, **extra):
        key = self.key(source_id)
        if key in self.file['records']:
            return False
        pose = np.asarray(pose, np.float32)
        if pose.ndim != 2 or pose.shape[1] != 51 or not np.isfinite(pose).all():
            raise ValueError(f'Invalid pose: {source_id}')
        n = len(pose)
        if not n:
            return False
        metadata = dict(metadata or {})
        metadata.update(source_id=source_id)
        metadata.setdefault('group_id', source_id)
        metadata.setdefault('split', split_for(metadata['group_id']))
        metadata.setdefault('kind', 'static')
        metadata.setdefault('reconstruction_weight', 1.)
        if metadata['kind'] not in ('static', 'temporal'):
            raise ValueError('Unknown record kind')
        labels = np.zeros((n,778), np.uint8) if contact is None else np.asarray(contact, np.uint8)
        known = np.zeros(n, bool) if contact is None else np.ones(n, bool)
        if valid is not None:
            known &= np.asarray(valid, bool)
        frames = np.arange(n, dtype=np.int64) if frames is None else np.asarray(frames, np.int64)
        if labels.shape != (n,778) or known.shape != (n,) or frames.shape != (n,):
            raise ValueError('Incompatible label/frame shapes')
        if np.any(labels > 1) or (metadata['kind']=='temporal' and np.any(np.diff(frames) <= 0)):
            raise ValueError('Nonbinary labels or unordered sequence')
        g = self.file.create_group('__pending')
        g.attrs.update(**metadata)
        content=hashlib.sha256()
        for value in (pose,labels,known,frames):content.update(np.ascontiguousarray(value).tobytes())
        g.attrs['content_sha256']=content.hexdigest()
        for name, a in dict(pose51=pose, contact=labels, contact_valid=known, frame_id=frames, **extra).items():
            a = np.asarray(a)
            if a.ndim and len(a) != n:
                raise ValueError(f'Row mismatch {name}')
            g.create_dataset(name, data=a, compression='lzf' if a.ndim else None, shuffle=bool(a.ndim))
        self.file.flush()
        self.file.move('__pending', 'records/'+key)
        self.file.flush()
        return True

    def close(self):
        self.file.flush(); self.file.close(); self.lock.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class PoseContactDataset(Dataset):
    def __init__(self, path, split='train', stage=2, history=8, limit=0, preload=False):
        self.path = str(path); self.history = history; self.rows = []; self.handle = None
        self.cached={} if preload else None
        if not 1 <= history <= 8:
            raise ValueError('History must be in [1,8]')
        with h5py.File(path, 'r') as f:
            if f.attrs['schema'] != SCHEMA:
                raise ValueError('Unknown corpus schema')
            for key in sorted(f['records']):
                g = f['records'][key]
                if split != 'all' and g.attrs['split'] != split:
                    continue
                frames = g['frame_id'][:]; known = g['contact_valid'][:]
                contact=g['contact'][:];labels=contact.any(-1)
                temporal = g.attrs['kind'] == 'temporal'
                if preload:
                    self.cached[key]=dict(pose51=g['pose51'][:],contact=contact,dataset=g.attrs['dataset'],
                        weight=g['reconstruction_weight'][:] if 'reconstruction_weight' in g else float(g.attrs['reconstruction_weight']))
                for end in range(len(frames)):
                    if stage == 2 and not known[end]:
                        continue
                    start = end-history+1 if temporal else end
                    if start < 0 or (temporal and np.any(np.diff(frames[start:end+1]) != 1)):
                        continue
                    self.rows.append((key, start, end, bool(known[end]), bool(labels[end]), temporal))
                    if limit and len(self.rows) >= limit:
                        return

    def __len__(self):
        return len(self.rows)

    def __getstate__(self):
        state = dict(self.__dict__); state['handle'] = None
        return state

    def __getitem__(self, index):
        key, start, end, known, label, temporal = self.rows[index]
        if self.cached is not None:
            g=self.cached[key];weight=g['weight'][end] if isinstance(g['weight'],np.ndarray) else g['weight']
            return dict(pose51=torch.from_numpy(g['pose51'][start:end+1]),contact=torch.from_numpy(g['contact'][end].astype(np.float32)),
                        valid=torch.tensor(known),frame=torch.tensor(label),weight=torch.tensor(float(weight)),
                        temporal=temporal,row_id=f'{key}:{end}',dataset=g['dataset'])
        if self.handle is None:
            self.handle = h5py.File(self.path, 'r')
        g = self.handle['records'][key]
        return dict(pose51=torch.from_numpy(g['pose51'][start:end+1]),
                    contact=torch.from_numpy(g['contact'][end].astype(np.float32)),
                    valid=torch.tensor(known), frame=torch.tensor(label),
                    weight=torch.tensor(float(g['reconstruction_weight'][end] if 'reconstruction_weight' in g else g.attrs['reconstruction_weight'])),
                    temporal=temporal, row_id=f'{key}:{end}', dataset=g.attrs['dataset'])

    def close(self):
        if self.handle is not None:
            self.handle.close(); self.handle = None


def collate_windows(rows):
    """Preserve exact HACO attention: bucket by length, with no padded keys."""
    result = []
    for length in sorted({len(r['pose51']) for r in rows}):
        subset = [r for r in rows if len(r['pose51']) == length]
        entry={k: torch.stack([r[k] for r in subset]) for k in ['pose51','contact','valid','frame','weight']}
        entry['dataset']=[r['dataset'] for r in subset]
        entry['row_id']=[r['row_id'] for r in subset]
        result.append(entry)
    return result


def train_statistics(path):
    counts = np.zeros((778,2), np.int64)
    with h5py.File(path, 'r') as f:
        if 'statistics/train_counts' in f and f.attrs.get('preparation_state')=='complete':
            return f['statistics/train_counts'][:]
        for g in f['records'].values():
            if g.attrs['split'] != 'train':
                continue
            known = g['contact_valid'][:]
            y = g['contact'][:][known]
            counts[:,1] += y.sum(0, dtype=np.int64)
            counts[:,0] += len(y)-y.sum(0, dtype=np.int64)
    return counts


def finalize_corpus(path, report_path=None):
    report,counts,fingerprint=audit_corpus(path,with_statistics=True)
    with h5py.File(path,'a') as f:
        if 'statistics' in f:del f['statistics']
        g=f.create_group('statistics');g.create_dataset('train_counts',data=counts)
        g.create_dataset('train_contact_means',data=counts[:,1]/counts.sum(-1).clip(1))
        f.attrs['preparation_state']='complete'
        f.attrs['inventory_json']=json.dumps(report)
        f.attrs['corpus_fingerprint']=fingerprint
        f.flush()
    if report_path is not None:write_json(report_path,report)
    return report


def merge_corpora(paths, output, finalize=True):
    """Merge completed independent preparations, rejecting conflicting records."""
    with CorpusWriter(output) as writer:
        requested=set(str(writer.file.attrs.get('requested_sources','')).split(','))-{''}
        for path in paths:
            if Path(path).resolve()==Path(output).resolve():raise ValueError('Cannot merge corpus into itself')
            with h5py.File(path,'r') as source:
                if source.attrs.get('schema')!=SCHEMA or source.attrs.get('preparation_state')!='complete':
                    raise ValueError('Merge input must be a completed compatible corpus: '+str(path))
                requested.update(str(source.attrs.get('requested_sources','')).split(','))
                for key,g in source['records'].items():
                    if key in writer.file['records']:
                        existing=writer.file['records'][key]
                        if (not g.attrs.get('content_sha256') or
                            existing.attrs.get('content_sha256')!=g.attrs['content_sha256']):
                            raise ValueError('Conflicting duplicate record: '+key)
                        continue
                    source.copy(g,writer.file,name='__pending')
                    writer.file.flush()
                    writer.file.move('__pending','records/'+key)
                writer.file.flush()
        writer.file.attrs['requested_sources']=','.join(sorted(requested-{''}))
    if finalize:return finalize_corpus(output,Path(output).with_suffix('.inventory.json'))


def audit_corpus(path,with_statistics=False):
    from collections import defaultdict
    report = defaultdict(lambda: dict(rows=0, labelled=0, contact=0, noncontact=0, windows8=0, transitions=0))
    seen = {}; fingerprints = set();timesteps=defaultdict(set);hand_instants=defaultdict(set);models=set();tiers=defaultdict(int);rejected=defaultdict(int)
    counts=np.zeros((778,2),np.int64);corpus_hash=hashlib.sha256()
    with h5py.File(path, 'r') as f:
        # Open one group at a time; sorting Group objects keeps every HDF5
        # handle alive and is expensive for older fine-grained shards.
        for key in sorted(f['records']):
            g=f['records'][key]
            a = g.attrs
            leakage_ids=([v.decode() for v in np.unique(g['leakage_id'][:])] if 'leakage_id' in g else
                         [hashlib.sha256(a['group_id'].encode()).hexdigest()])
            for identity in leakage_ids:
                if identity in seen and seen[identity] != a['split']:
                    raise ValueError('Split leakage: '+identity)
                seen[identity]=a['split']
            if a['source_id'] in fingerprints:
                raise ValueError('Duplicate source')
            fingerprints.add(a['source_id'])
            known = g['contact_valid'][:]; labels=g['contact'][:];y=labels.any(-1);frames = g['frame_id'][:]
            if with_statistics:
                corpus_hash.update((key+':'+str(len(y))+':'+str(a.get('content_sha256','legacy'))).encode())
                if a['split']=='train':
                    positives=labels[known].sum(0,dtype=np.int64)
                    counts[:,1]+=positives;counts[:,0]+=int(known.sum())-positives
            r = report[f"{a['dataset']}:{a['split']}:{a['kind']}"]
            tiers[a.get('label_source','unknown')]+=len(y)
            rejected[a['dataset']]+=int(a.get('rejected_pose_rows',0))
            if a['dataset'].startswith('generated'):models.add(a['model_sha256'])
            r['rows'] += len(y); r['labelled'] += int(known.sum())
            r['contact'] += int((known & y).sum()); r['noncontact'] += int((known & ~y).sum())
            if a['kind']=='temporal':
                dataset=a['dataset'];group_id=a['group_id'];side=a.get('side','unknown')
                timesteps[dataset].update((group_id,int(frame)) for frame in frames)
                hand_instants[dataset].update((group_id,side,int(frame)) for frame in frames)
                continuous = np.diff(frames)==1
                r['transitions'] += int((continuous & known[1:] & known[:-1] & (y[1:]!=y[:-1])).sum())
                if len(y)>=8:
                    r['windows8'] += int(np.lib.stride_tricks.sliding_window_view(continuous,7).all(-1).sum())
    result=dict(schema=SCHEMA, sources=len(fingerprints), groups=len(seen), datasets=dict(report),
                unique_temporal_timesteps={k:len(v) for k,v in timesteps.items()},
                unique_temporal_hand_instants={k:len(v) for k,v in hand_instants.items()},
                generated_model_assets=len(models),label_tier_rows=dict(tiers),rejected_pose_rows=dict(rejected))
    return (result,counts,corpus_hash.hexdigest()) if with_statistics else result
