"""Frozen native joint data, provenance, object/duplicate-disjoint partitions."""
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = ROOT/'data/datasets/wuji2_rvq_v1'


def sha(path):
    with Path(path).open('rb') as stream:
        digest=hashlib.sha256()
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
        return digest.hexdigest()


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_suffix(path.suffix+'.pending')
    pending.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    pending.replace(path)


def pose_keys(q):
    return np.ascontiguousarray(np.rint(q.astype(np.float64)*1e5).astype('<i4')).view('V80').ravel()


def download_pool(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    path=output/'remote_pool.npz'
    if path.exists():return path
    pending=path.with_suffix('.npz.pending')
    with Path(__file__).with_name('remote_extract.py').open('rb') as script, pending.open('wb') as stream:
        subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','1016','8.147.67.185','python3','-'],stdin=script,stdout=stream,check=True)
    with np.load(pending,allow_pickle=False) as z:assert z['qpos'].shape[1]==20
    pending.replace(path)
    return path


def prepare(output=DEFAULT_DATA, seed=20261118):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    manifest=output/'manifest.json'
    if manifest.exists():
        load(output)
        return json.loads(manifest.read_text())
    pool=download_pool(output)
    with np.load(pool,allow_pickle=False) as z:
        metadata=json.loads(str(z['metadata']));raw={k:z[k].copy() for k in z.files if k!='metadata'}
    _,unique=np.unique(raw['source_id'],return_index=True);unique.sort()
    raw={k:v[unique] for k,v in raw.items()}
    q=raw['qpos'];labels=raw['label'];types=metadata['types']
    if len(types)!=31 or q.shape!=(len(labels),20):raise ValueError('Invalid source corpus')
    _,objects=np.unique(raw['object_id'],return_inverse=True)
    parent=np.arange(int(objects.max())+1)
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=int(parent[i])
        return i
    keys=pose_keys(q)
    order=np.argsort(keys)
    pairs=np.flatnonzero(keys[order[1:]]==keys[order[:-1]])
    for p in pairs:
        a,b=find(int(objects[order[p]])),find(int(objects[order[p+1]]))
        if a!=b:parent[max(a,b)]=min(a,b)
    roots=np.array([find(i) for i in range(len(parent))])
    components=roots[objects]
    unique_components,component_index=np.unique(components,return_inverse=True)
    targets=[4000,500,500]
    assigned=None
    for attempt in range(100):
        rng=np.random.default_rng(seed+attempt)
        group_split=rng.choice(3,len(unique_components),p=[.8,.1,.1])
        split=group_split[component_index]
        available=np.array([[np.count_nonzero((labels==label)&(split==s)) for s in range(3)] for label in range(31)])
        if (available>=np.array(targets)).all():assigned=split;break
    if assigned is None:
        counts={name:available[i].tolist() for i,name in enumerate(types)}
        write_json(output/'preparation_failure.json',dict(reason='Insufficient object/duplicate-disjoint rows',available=counts,components=len(unique_components)))
        raise ValueError('Cannot obtain 4000/500/500 disjoint rows for every type; see preparation_failure.json')
    rng=np.random.default_rng(seed)
    selected=[]
    for label in range(31):
        for split,count in enumerate(targets):
            candidates=np.flatnonzero((labels==label)&(assigned==split))
            selected.extend(rng.choice(candidates,count,replace=False).tolist())
    selected=np.asarray(selected)
    arrays={k:v[selected] for k,v in raw.items()}
    arrays['split']=assigned[selected].astype(np.int8)
    arrays['component']=components[selected]
    # Every source object and every quantized duplicate belongs to one partition.
    for field in ['object_id','source_id','component']:
        sets=[set(arrays[field][arrays['split']==s].tolist()) for s in range(3)]
        assert all(not sets[a]&sets[b] for a,b in [(0,1),(0,2),(1,2)]),field
    chosen_keys=pose_keys(arrays['qpos'])
    key_sets=[{key.tobytes() for key in chosen_keys[arrays['split']==s]} for s in range(3)]
    assert all(not key_sets[a]&key_sets[b] for a,b in [(0,1),(0,2),(1,2)])
    heldout=key_sets[1]|key_sets[2]
    source=ROOT/'data/datasets/dex_grasping_primitives_20261115'
    primitive_config=json.loads((source/'wuji2/config.json').read_text())
    if primitive_config['joint_names']!=metadata['joint_names']:raise ValueError('Primitive/native joint order mismatch')
    anchors=[];anchor_labels=[];primitive_files=[];primitive_total=0
    low=np.asarray(metadata['lower']);high=np.asarray(metadata['upper'])
    for label,name in enumerate(types):
        path=source/'wuji2/grids'/f'{name}.npz'
        primitive_files.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path)))
        with np.load(path,allow_pickle=False) as z:
            if list(z['joint_names'])!=metadata['joint_names']:raise ValueError('Primitive joint names changed')
            values=z['target_qpos'].astype(np.float32)
        primitive_total+=len(values)
        native=np.isfinite(values).all(1)&(values>=low-1e-5).all(1)&(values<=high+1e-5).all(1)
        for row,key,valid in zip(values,pose_keys(values),native):
            if valid and key.tobytes() not in heldout:
                anchors.append(row);anchor_labels.append(label)
    if primitive_total!=1247:raise ValueError('Expected 1247 primitive rows')
    arrays.update(anchor_qpos=np.asarray(anchors,np.float32),anchor_label=np.asarray(anchor_labels,np.int16))
    pending=output/'poses.npz.pending'
    with pending.open('wb') as f:np.savez_compressed(f,**arrays)
    pending.replace(output/'poses.npz')
    metadata.update(schema='wuji2_pose_rvq_v1',seed=seed,pool_sha256=sha(pool),poses_sha256=sha(output/'poses.npz'),
        selected_per_type=5000,split_counts_per_type=targets,source_unique_records=len(q),
        object_components=len(unique_components),split_attempt=attempt,primitive_total=primitive_total,
        primitive_anchors=len(anchors),primitive_files=primitive_files,duplicate_pose_resolution_rad=1e-5,
        selection='valid AND finite AND native joint bounds; contact/stability retained as metadata',
        counts={name:{'train':4000,'val':500,'test':500} for name in types})
    write_json(manifest,metadata)
    return metadata


def load(path=DEFAULT_DATA):
    path=Path(path);metadata=json.loads((path/'manifest.json').read_text())
    if sha(path/'poses.npz')!=metadata['poses_sha256']:raise ValueError('Prepared dataset fingerprint mismatch')
    with np.load(path/'poses.npz',allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
    return arrays,metadata
