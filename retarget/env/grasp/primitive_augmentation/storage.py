"""Atomic HDF5 shards, immutable geometry and public float32 dataset reader."""
import json
import os
from pathlib import Path
import hashlib
import h5py
import numpy as np
from . import SCHEMA
from .templates import matrix, sha256


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as f:
        json.dump(value,f,ensure_ascii=False,allow_nan=False,sort_keys=True,indent=2)
        f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)


def geometry_id(shape):
    h=hashlib.sha256()
    h.update(np.asarray(shape.mesh.vertices,np.float32).tobytes())
    h.update(np.asarray(shape.mesh.faces,np.int32).tobytes())
    return h.hexdigest()[:24]


def save_geometry(root,shape):
    gid=geometry_id(shape);path=Path(root)/'geometry'/f'{gid}.npz'
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_suffix('.tmp')
        with tmp.open('wb') as f:
            np.savez_compressed(f,vertices=np.asarray(shape.mesh.vertices,np.float32),faces=np.asarray(shape.mesh.faces,np.int32),
                kind=np.array(shape.kind),radius_m=shape.radius,height_m=shape.height,
                requested_mask_fraction=shape.requested_fraction,mask_fraction=shape.removed_fraction,
                cutters=shape.cutters,surface_seed_m=shape.surface_seed)
            f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    return gid


def load_geometry(root,gid):
    import trimesh
    from .geometry import Shape
    with np.load(Path(root)/'geometry'/f'{gid}.npz',allow_pickle=False) as z:
        return Shape(trimesh.Trimesh(z['vertices'],z['faces'],process=False),str(z['kind']),float(z['radius_m']),float(z['height_m']),
                     float(z['requested_mask_fraction']),float(z['mask_fraction']),z['cutters'],z['surface_seed_m'])


def save_template(root,template,surface):
    path=Path(root)/'templates'/f'{template["name"]}.npz'
    if path.exists():return
    path.parent.mkdir(parents=True,exist_ok=True)
    arrays=dict(owners=template['owners'],body_ids=template['bodies'],
                mapping=np.array(json.dumps(template['mapping'])),source_sha256=np.array(template['source_sha256']),
                annotation_sha256=np.array(template['annotation_sha256']))
    for i,p in enumerate(template['patches']):
        for key,value in p.items():arrays[f'patch_{i}_{key}']=value
    # These exact local mesh arrays, not reprocessed STL vertex orders, define
    # all face IDs and barycentric coordinates in training records.
    for owner in set(template['owners']):
        arrays[f'owner_{owner}_vertices']=np.asarray(surface.meshes[owner].vertices)
        arrays[f'owner_{owner}_faces']=np.asarray(surface.meshes[owner].faces)
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
    os.replace(tmp,path)


def write_shard(root,name,sequence,arrays,commit,config_hash):
    folder=Path(root)/'shards'/name;folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'{sequence:08d}.h5';tmp=path.with_suffix('.tmp')
    if path.exists():raise ValueError('Refusing to overwrite a committed shard')
    lengths={len(v) for v in arrays.values()}
    if len(lengths)!=1 or not next(iter(lengths)):raise ValueError('Misaligned/empty shard')
    with h5py.File(tmp,'w') as f:
        f.attrs['schema']=SCHEMA;f.attrs['config_hash']=config_hash
        f.attrs['commit']=json.dumps(commit,sort_keys=True,allow_nan=False)
        for key,value in arrays.items():
            value=np.asarray(value)
            if value.dtype.kind=='U':value=value.astype('S')
            f.create_dataset(key,data=value,compression='gzip',compression_opts=1,shuffle=True,
                             chunks=(min(len(value),64),*value.shape[1:]))
        f.flush()
    with tmp.open('rb') as f:os.fsync(f.fileno())
    os.replace(tmp,path)
    return path


class Dataset:
    """Reader: qpos in radians, cloud in object-local metres, poses XYZ/WXYZ.

    ``sample(..., world=True)`` transforms cloud into the saved world frame.
    Contact world points and normals are stored explicitly as float32 arrays.
    """
    def __init__(self,root):
        self.root=Path(root);self.config=json.loads((self.root/'config.json').read_text())
        if self.config['schema']!=SCHEMA:raise ValueError('Unknown augmentation schema')
        self.shards=sorted((self.root/'shards').glob('*/*.h5'))

    def sample(self,path,index,world=False):
        path=Path(path)
        if not path.is_absolute():path=self.root/path
        with h5py.File(path,'r') as f:
            result={k:f[k][index] for k in f.keys()}
        for k,v in list(result.items()):
            if isinstance(v,bytes):result[k]=v.decode()
        result['volume_points_m']=np.asarray(result['volume_points_m'],np.float32)
        if world:
            pose=result['object_pose_world']
            result['volume_points_m']=result['volume_points_m']@matrix(pose).T+pose[:3]
        return result

    def records(self):
        for path in self.shards:
            with h5py.File(path,'r') as f:n=len(f['qpos'])
            for i in range(n):yield path,i

    def manifest(self):
        result={str(p.relative_to(self.root)):sha256(p) for folder in ('geometry','templates','calibration','shards')
                for p in sorted((self.root/folder).rglob('*')) if p.is_file() and p.suffix in ('.h5','.npz')}
        return result
