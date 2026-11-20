"""Read-only SSH extraction helper; writes one compact NPZ to stdout."""
import hashlib
import io
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path('/mnt/cpfs/zbl-cpfs-new/USERS/jasonzhang/retarget')
SOURCE = ROOT/'results/wuji2_multi_20261107_x8iizrs9'


def sha(path):
    with Path(path).open('rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
        return digest.hexdigest()


def main():
    model = ROOT/'data/grasp/assets/wuji2/right.xml'
    xml = ET.parse(model).getroot()
    if xml.find('compiler').get('angle') != 'radian':
        raise ValueError('Source XML must use radians')
    joints = [j for j in xml.findall('.//joint') if j.get('name')]
    if len(joints) != 26 or not all(j.get('name').startswith('grasp_wrist_') for j in joints[:6]):
        raise ValueError('Expected six serial wrist joints followed by 20 hand joints')
    names = [j.get('name') for j in joints[6:]]
    bounds = np.array([list(map(float, j.get('range').split())) for j in joints[6:]])
    types = sorted([p.name for p in SOURCE.iterdir() if p.is_dir() and p.name.split('_')[0].isdigit()], key=lambda x:int(x.split('_')[0]))
    if len(types) != 31:
        raise ValueError('Expected all 31 grasp types')
    columns = {k:[] for k in ['qpos', 'label', 'object_id', 'source_id', 'stable', 'contact', 'shard_id', 'row']}
    metadata = dict(source=str(SOURCE), model_sha256=sha(model), joint_names=names,
                    lower=bounds[:,0].tolist(), upper=bounds[:,1].tolist(), types=types, files=[], units='radians')
    for label, name in enumerate(types):
        count = rejected = 0
        for file in sorted((SOURCE/name).glob('*/final.npz')):
            side = file.parent/'sources.json'
            record = json.loads(side.read_text())
            cases = record['cases']
            shard = len(metadata['files'])
            metadata['files'].append(dict(path=str(file), sha256=sha(file), sources_sha256=sha(side)))
            with np.load(file, allow_pickle=False) as z:
                q = z['qpos']
                if q.ndim != 2 or q.shape[1] != 26 or len(cases) != len(q):
                    raise ValueError(f'Invalid source shape/lineage: {file}')
                q = q[:,6:].astype(np.float32)
                keep = z['valid'].astype(bool) & np.isfinite(q).all(1) & (q >= bounds[:,0]-1e-5).all(1) & (q <= bounds[:,1]+1e-5).all(1)
                ids = np.flatnonzero(keep)
                source_rows = z['source_rows']
                source_keys = [hashlib.sha256((record['input_manifest_sha256']+':'+':'.join(map(str,source_rows[i]))).encode()).hexdigest()[:32] for i in ids]
                objects = []
                for i in ids:
                    scene = cases[i]['scene_path']
                    if '/scene_cfg/' not in scene:
                        raise ValueError('Missing object identity: '+scene)
                    family, tail = scene.split('/scene_cfg/',1)
                    objects.append(hashlib.sha256((family+':'+tail.split('/')[0]).encode()).hexdigest()[:32])
                values = dict(qpos=q[ids], label=np.full(len(ids),label,np.int16),
                    object_id=np.asarray(objects,dtype='S32'), source_id=np.asarray(source_keys,dtype='S32'),
                    stable=z['rollout_stable'][ids], contact=z['patch_joint_accepted'][ids],
                    shard_id=np.full(len(ids),shard,np.int32), row=ids.astype(np.int32))
                for k,v in values.items():columns[k].append(v)
                count += len(ids);rejected += len(q)-len(ids)
        print(f'{name}: valid native poses {count}, rejected {rejected}',file=sys.stderr,flush=True)
    arrays = {k:np.concatenate(v) for k,v in columns.items()}
    arrays['metadata'] = np.array(json.dumps(metadata,sort_keys=True))
    stream = io.BytesIO()
    np.savez_compressed(stream,**arrays)
    sys.stdout.buffer.write(stream.getvalue())


if __name__ == '__main__':main()
