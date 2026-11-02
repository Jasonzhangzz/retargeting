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


@lru_cache(maxsize=8)
def _snapshot(path,modified):return GraspSnapshot(path)


def get_snapshot(batch):
    path=Path(batch).resolve();return _snapshot(str(path),(path/'manifest.json').stat().st_mtime_ns)


def read_summary(batch,row):
    if 'cached_case' in row:return get_snapshot(batch).summary(int(row['cached_case']))
    return json.loads((Path(batch)/row['summary']).read_text())
