"""Shared real-asset fixtures and mesh surfaces for MANO regression checks."""
import json
import numpy as np
import torch,pytest
from tests._paths import ROOT
from retarget.env.grasp.mano_solver import ManoDecoder,ShadowSkeleton,FixedContacts,SurfaceBatch
from scripts.retarget_dexonomy_mano import GeometryLoader

@pytest.fixture(scope='module')
def fixture():
    torch.set_num_threads(1)
    raw=json.loads((ROOT/'output/dexonomy_shadow_mano_mapping_20261103/data.json').read_text())
    source=ROOT/'results/dexonomy_shadow_grasp_full/7_Prismatic_3_Finger/shards/0000000'
    entries=json.loads((source/'manifest.json').read_text())['cases'][:2]
    with np.load(source/'poses.npz') as a:q=a['grasp_qpos'][:2];op=a['object_pose_world'][:2];scale=a['scene_scale'][:2]
    model=ManoDecoder(ROOT/'data/grasp/assets/mano/MANO_RIGHT.pkl');model.configure_contact_vertices(raw['mappings'])
    skeleton=ShadowSkeleton(ROOT/'data/dexonomy/shadow_assets/right.xml')
    mapping=raw['mappings']['7_Prismatic_3_Finger'];points,normals=skeleton.contacts(q,mapping)
    loader=GeometryLoader(ROOT/'data/dexonomy');geos=[];ids=[];keys={}
    for e in entries:
        k,g=loader.get(e)
        if k not in keys:keys[k]=len(geos);geos.append(g)
        ids.append(keys[k])
    return dict(model=model,target=skeleton(q),points=points,normals=normals,fixed=FixedContacts(model,mapping),
        geo=geos,ids=ids,op=op,scale=scale)

def surface(f,cache=None):return SurfaceBatch(f['geo'],f['ids'],f['op'],f['scale'],'cpu',mesh_cache=cache)
