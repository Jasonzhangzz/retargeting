"""Frozen Shadow inputs and CPFS-safe IO for the current CEM campaign."""
import os,json,hashlib,datetime,collections,time,math
from pathlib import Path
import numpy as np

def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    temp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n');temp.replace(path)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def retry_io(function):
    import errno,functools
    @functools.wraps(function)
    def wrapped(*args,**kwargs):
        for attempt in range(7):
            try:return function(*args,**kwargs)
            except OSError as error:
                if error.errno not in (errno.ESTALE,errno.EIO,errno.EAGAIN) or attempt==6:raise
                time.sleep(.05*2**attempt)
    return wrapped


def round_robin_groups(groups):
    """One existing batch per available grasp type per round, without resampling."""
    by_type = collections.defaultdict(dict)
    for group in groups:
        by_type[group['type']][int(group['part'])] = group
    def type_order(name):
        prefix = name.partition('_')[0]
        return (int(prefix) if prefix.isdigit() else math.inf, name)
    pending = collections.OrderedDict(
        (typ, collections.deque(parts[index] for index in sorted(parts)))
        for typ, parts in sorted(by_type.items(), key=lambda item: type_order(item[0]))
    )
    ordered = []
    while pending:
        for typ in list(pending):
            ordered.append(pending[typ].popleft())
            if not pending[typ]:
                del pending[typ]
    return ordered


@retry_io
def source_groups(source):
    source=Path(source);journal=source/'shards.jsonl'
    if journal.exists():
        entries={}
        for line in journal.read_text().splitlines():
            try:v=json.loads(line)
            except ValueError:continue
            entries[(v['type'],v['part'])]=v
        return list(entries.values())
    return [dict(type=p.parents[2].name,part=int(p.parent.name),path=str(p.parent),
                 samples=len(read(p)['cases'])) for p in sorted(source.glob('*/shards/*/manifest.json'))]


class GeometryLoader:
    def __init__(self,dataset):self.dataset=dataset;self.cache=collections.OrderedDict()
    @retry_io
    def get(self,entry):
        import trimesh
        from retarget.env.grasp.dexonomy_grasp_static import base_scene
        scene=base_scene(str(self.dataset),entry['scene_path'])
        cfg=scene['scene'][scene['task']['obj_name']]
        scale=np.asarray(cfg['scale'],float)
        key=(cfg['file_path'],tuple(scale))
        if key not in self.cache:
            mesh=trimesh.load_mesh(cfg['file_path'],process=False)
            if isinstance(mesh,trimesh.Scene):mesh=mesh.to_mesh()
            if not isinstance(mesh,trimesh.Trimesh):raise ValueError('Object asset is not a triangle mesh')
            vertices=np.asarray(mesh.vertices,float)*scale
            faces=np.asarray(mesh.faces,np.int32).copy()
            good=np.linalg.norm(np.cross(vertices[faces[:,1]]-vertices[faces[:,0]],vertices[faces[:,2]]-vertices[faces[:,0]]),axis=1)>1e-14
            faces=faces[good]
            checked=trimesh.Trimesh(vertices=vertices,faces=faces,process=False)
            flipped=bool(checked.is_watertight and checked.volume<0)
            if flipped:faces=faces[:,[0,2,1]]
            self.cache[key]=dict(vertices=vertices.astype(np.float32),faces=faces,watertight=bool(checked.is_watertight),
                source_mesh=cfg['file_path'],base_scale=scale.tolist(),winding_flipped=flipped,sha256=sha(cfg['file_path']))
            if len(self.cache)>128:self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return key,self.cache[key]



read=retry_io(read)
write=retry_io(write)
sha=retry_io(sha)
@retry_io
def save_npz(path,**values):
    import torch
    np.savez_compressed(path,**{k:v.detach().cpu().numpy() if isinstance(v,torch.Tensor) else v for k,v in values.items()})


@retry_io
def load_npz(path):
    with np.load(path,allow_pickle=False) as archive:return {key:archive[key].copy() for key in archive.files}


def worker_layout(devices,workers_per_device):
    if workers_per_device<1:raise ValueError('workers-per-device must be positive')
    devices=[d.strip() for d in devices.split(',')]
    if not devices or any(not d.startswith('cuda:') or not d[5:].isdigit() for d in devices):
        raise ValueError('Use explicit CUDA devices, e.g. cuda:0,cuda:1')
    devices=['cuda:'+str(int(d[5:])) for d in devices]
    if len(devices)!=len(set(devices)):raise ValueError('Use distinct CUDA devices')
    return devices,[d for d in devices for _ in range(workers_per_device)]


def shard_owner(key,workers):
    return int(hashlib.sha256(key.encode()).hexdigest()[:8],16)%workers
