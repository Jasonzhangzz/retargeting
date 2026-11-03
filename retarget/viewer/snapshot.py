"""Read compact pose snapshots: one hand scene, shared primitive geometry, arrays."""
from functools import lru_cache
import gzip,json
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from retarget.config.run import ROOT


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False)


class GraspSnapshot:
    def __init__(self,path):
        self.path=Path(path);self.manifest=json.loads((self.path/'manifest.json').read_text())
        with np.load(self.path/'poses.npz',allow_pickle=False) as data:self.poses={k:data[k].copy() for k in data.files}
        with np.load(self.path/'geometry.npz',allow_pickle=False) as data:self.geometry={k:data[k].copy() for k in data.files}
        self.templates=json.loads(gzip.decompress((self.path/'templates.json.gz').read_bytes()))
        self.metrics=json.loads(gzip.decompress((self.path/'metrics.json.gz').read_bytes()))
        self.model=lru_cache(maxsize=8)(self.model)

    def diagnostics(self,index):
        result=dict(self.metrics[index]['geometry_diagnostics'])
        count=int(self.poses['contact_counts'][index])
        result['projected_points']=self.poses['projected_points'][index,:count].tolist()
        return result

    def template(self,index):return self.templates[self.manifest['cases'][index]['template_id']]
    def patch_coverage(self,index):return None
    def summary(self,index):return self.metrics[index]

    def object_bytes(self,index):
        p=self.manifest['cases'][index]['primitive'];kind=p['kind']
        vertices=self.geometry[kind+'_vertices']*np.array([p['radius_m'],p['radius_m'],p['height_m']])
        faces=self.geometry[kind+'_faces']
        text=''.join('v '+' '.join(format(float(x),'.17g') for x in v)+'\n' for v in vertices)
        text+=''.join('f '+' '.join(str(int(x)+1) for x in face)+'\n' for face in faces)
        return text.encode()

    def model(self,index):
        tree=ET.fromstring((self.path/'scene.xml').read_text())
        for element in tree.iter():
            file=element.get('file')
            if file is None:continue
            if file=='__OBJECT__':element.set('file','cached_object.obj')
            else:
                asset=(ROOT/file).resolve()
                if not asset.is_relative_to(ROOT) or not asset.is_file():raise ValueError(f'Missing repository asset: {file}')
                element.set('file',str(asset))
        return mujoco.MjModel.from_xml_string(ET.tostring(tree,encoding='unicode'),assets={'cached_object.obj':self.object_bytes(index)})


class DexonomySnapshot(GraspSnapshot):
    """Same cache reader interface for original Shadow poses and scene assets."""
    def __init__(self,path):
        self.path=Path(path);self.manifest=json.loads((self.path/'manifest.json').read_text())
        with np.load(self.path/'poses.npz',allow_pickle=False) as data:self.poses={k:data[k].copy() for k in data.files}
        self.geometry={}
        self.templates=json.loads(gzip.decompress((self.path/'templates.json.gz').read_bytes()))
        self.metrics=json.loads(gzip.decompress((self.path/'metrics.json.gz').read_bytes()))
        self.model=lru_cache(maxsize=8)(self.model)

    def model(self,index):
        path=(self.path/self.manifest['cases'][index]['model']).resolve()
        if not path.is_relative_to(self.path.resolve()):raise ValueError('Invalid cached model path')
        return mujoco.MjModel.from_xml_path(str(path))

    def object_bytes(self,index):
        # Export scaled collision geometry in the object root's local frame.
        import trimesh
        from retarget.env.geometry import ObjectMeshSurface
        model=self.model(index)
        bodies=[i for i in range(model.nbody) if model.body(i).name.startswith('obj-') and model.body_mass[i]>0]
        if len(bodies)!=1:raise ValueError('Expected one rigid object')
        triangles=ObjectMeshSurface.from_model(model,bodies[0]).triangles
        mesh=trimesh.Trimesh(vertices=triangles.reshape(-1,3),faces=np.arange(triangles.size//3).reshape(-1,3),process=False)
        return mesh.export(file_type='obj').encode()


class DexonomyStaticSnapshot(DexonomySnapshot):
    """Rebuild models on demand; pose shards share original hand/object assets."""
    def __init__(self,path):
        super().__init__(path)
        self._static_models=None

    def model(self,index):
        from retarget.env.grasp.dexonomy_static import StaticModels
        if self._static_models is None:self._static_models=StaticModels(self.manifest['dataset_root'],8,visual=True)
        entry=self.manifest['cases'][index]
        record={'scene_path':entry['scene_path'],'scene_scale':np.array([entry['scene_scale']]),'grasp_qpos':np.zeros((1,29))}
        return self._static_models.get(record,0)[0]._model


@lru_cache(maxsize=8)
def _snapshot(path,modified):
    manifest=json.loads((Path(path)/'manifest.json').read_text())
    if manifest.get('storage')=='dexonomy_shadow_snapshot_v1':return DexonomySnapshot(path)
    if manifest.get('storage') in ('dexonomy_shadow_static_snapshot_v1','dexonomy_shadow_grasp_snapshot_v1'):return DexonomyStaticSnapshot(path)
    return GraspSnapshot(path)


def get_snapshot(batch):
    path=Path(batch).resolve();return _snapshot(str(path),(path/'manifest.json').stat().st_mtime_ns)


def read_summary(batch,row):
    if 'cached_case' in row:return get_snapshot(batch).summary(int(row['cached_case']))
    return json.loads((Path(batch)/row['summary']).read_text())
