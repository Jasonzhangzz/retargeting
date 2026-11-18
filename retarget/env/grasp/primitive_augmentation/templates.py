"""Immutable template patches on actual native hand triangles."""
import hashlib
import heapq
import json
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from retarget.baseline.geort.adapter import RobotAdapter, ROOT
from retarget.baseline.geort.pose_domain import extension_domain
from retarget.env.grasp.inputs import adapt_primitive_contacts
from retarget.task.grasp.dexonomy import load_template, map_contacts
from moxian_web.wuji_surface import WujiSurface

SOURCE = ROOT/'data/datasets/dex_grasping_primitives_20261115'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def matrix(pose):
    return Rotation.from_quat(np.asarray(pose)[[4,5,6,3]]).as_matrix()


def local_pose(base, obj):
    r = matrix(obj)
    return np.r_[(base[:3]-obj[:3])@r,
                 Rotation.from_matrix(r.T@matrix(base)).as_euler('XYZ')]


def world_pose(root, obj):
    r = matrix(obj)
    rot = r@Rotation.from_euler('XYZ', root[3:6]).as_matrix()
    xyzw = Rotation.from_matrix(rot).as_quat()
    return np.r_[r@root[:3]+obj[:3], xyzw[[3,0,1,2]]]


def geodesic_patch(mesh, anchor, normal, radius=.01, count=128, seed=0):
    _, _, face = trimesh.proximity.closest_point(mesh, anchor[None])
    face = int(face[0]); anchor = mesh.triangles[face].mean(0) if not np.isfinite(anchor).all() else anchor
    adj = [[] for _ in mesh.faces]
    centers = mesh.triangles_center
    for a,b in mesh.face_adjacency:
        distance = float(np.linalg.norm(centers[a]-centers[b]))
        adj[a].append((int(b),distance)); adj[b].append((int(a),distance))
    distances={face:0.};queue=[(0.,face)]
    while queue:
        d,i=heapq.heappop(queue)
        if d != distances[i]:continue
        for j,step in adj[i]:
            nd=d+step
            if (nd<=radius and np.dot(mesh.face_normals[j],normal)>.0
                    and nd<distances.get(j,float('inf'))):
                distances[j]=nd;heapq.heappush(queue,(nd,j))
    faces=np.array(sorted(distances),np.int32)
    rng=np.random.default_rng(seed)
    ids=rng.choice(faces,max(count*8,1024),p=mesh.area_faces[faces]/mesh.area_faces[faces].sum())
    uv=rng.random((len(ids),2));flip=uv.sum(1)>1;uv[flip]=1-uv[flip]
    bary=np.c_[1-uv.sum(1),uv]
    points=(mesh.triangles[ids]*bary[...,None]).sum(1)
    valid=np.linalg.norm(points-anchor,axis=1)<=radius
    points,ids,bary=points[valid],ids[valid],bary[valid]
    projected,_,face_id=trimesh.proximity.closest_point(mesh,anchor[None])
    anchor=projected[0];face_id=int(face_id[0])
    anchor_bary=trimesh.triangles.points_to_barycentric(mesh.triangles[face_id:face_id+1],anchor[None])[0]
    points=np.vstack((anchor,points));ids=np.r_[face_id,ids];bary=np.vstack((anchor_bary,bary))
    # Farthest-point witnesses cover the whole immutable patch.
    chosen=[0];dist=np.linalg.norm(points-points[0],axis=1)
    for _ in range(1,min(count,len(points))):
        i=int(dist.argmax());chosen.append(i);dist=np.minimum(dist,np.linalg.norm(points-points[i],axis=1))
    chosen=np.resize(np.array(chosen),count)
    return dict(face_ids=ids[chosen].astype(np.int32),barycentric=bary[chosen],
                points=points[chosen],normals=mesh.face_normals[ids[chosen]],
                region_faces=faces,anchor=anchor,radius_m=radius)


class Templates:
    def __init__(self, source=SOURCE, samples=128):
        self.source=Path(source);self.robot=RobotAdapter();self.surface=WujiSurface()
        self.domain=extension_domain(self.robot.lower,self.robot.upper)
        self.lower=np.array(self.domain['lower']);self.upper=np.array(self.domain['upper'])
        manifest=json.loads((self.source/'source-manifest.json').read_text())
        self.entries={e['type']:e for e in manifest['entries']}
        self.names=sorted(self.entries,key=lambda n:int(n.split('_')[0]))
        self.cache={};self.samples=samples
        if len(self.names)!=31:raise ValueError('Expected 31 primitive grasp types')

    def get(self,name):
        if name in self.cache:return self.cache[name]
        e=self.entries[name];path=self.source/'wuji2'/e['file']
        with np.load(path,allow_pickle=False) as z:
            grid={k:z[k].copy() for k in z.files}
        if not np.isfinite(grid['target_qpos']).all():raise ValueError('Nonfinite source poses')
        # Dataset/Robokit names differ from the native task names. Both are
        # finger-major 4-DOF chains; verify against the frozen solver contract.
        cfg=json.loads((self.source/'wuji2/config.json').read_text())
        if list(grid['joint_names'])!=cfg['joint_names']:raise ValueError('Source joint order changed')
        checkout=ROOT/'data/grasp/templates'
        shadow,_,contacts,annotation=load_template(checkout,name)
        contacts,adapt=adapt_primitive_contacts(checkout,name,contacts)
        bodies,local,normals,mapping=map_contacts(shadow,contacts,self.surface.model)
        owners=np.array([self.surface.body_ids.index(int(b)) for b in bodies],np.int32)
        patches=[geodesic_patch(self.surface.meshes[o],p,n,count=self.samples,seed=2718+i)
                 for i,(o,p,n) in enumerate(zip(owners,local,normals))]
        roots=np.stack([local_pose(b,o) for b,o in zip(grid['target_base_pose'],grid['object_pose_world'])])
        result=dict(name=name,type_id=int(name.split('_')[0]),kind=e['kind'],grid=grid,
                    seeds=np.c_[roots,grid['target_qpos']],owners=owners,bodies=bodies,
                    patches=patches,mapping=mapping,adaptations=adapt,
                    source_sha256=sha256(path),annotation_sha256=sha256(annotation))
        self.cache[name]=result
        return result

    def identity(self):
        return dict(robot=self.robot.spec,domain=self.domain,
                    source_manifest_sha256=sha256(self.source/'source-manifest.json'),
                    grids={n:sha256(self.source/'wuji2'/self.entries[n]['file']) for n in self.names},
                    annotations={n:sha256(ROOT/'data/grasp/templates/assets/hand/shadow/raw_anno'/f'{n}.yaml') for n in self.names},
                    keypoints_sha256=sha256(ROOT/'data/grasp/templates/assets/hand/shadow/keypoint.yaml'),
                    shadow_xml_sha256=sha256(ROOT/'data/grasp/templates/assets/hand/shadow/right.xml'),
                    patch_radius_m=.01,patch_witnesses=self.samples,
                    plate_dimensions='[2*radius, 1.4*radius, height] repository benchmark reconstruction')
