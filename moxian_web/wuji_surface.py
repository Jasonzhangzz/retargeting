"""Immutable anatomical material correspondence to the native WUJI2 surface."""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import trimesh
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from retarget.env.grasp.hand import HAND_BODIES, hand_tree, rigid_mesh_arrays, palmar_axis
from .tactile_surface import CurvedTaxels

POLICY = 'native_semantic_triangles_card_v5'
CHAINS = ((13,14,15),(1,2,3),(4,5,6),(10,11,12),(7,8,9))


def unit(x):
    x=np.asarray(x,float)
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)


def basis(axis,normal):
    z=unit(axis);y=unit(normal-z*np.dot(z,normal))
    return np.stack((np.cross(y,z),y,z),axis=1)


def mirror_matrix(topology):
    if topology.get('hand_side')=='right':return np.eye(3)
    normal=unit(topology['palm_view']['normal']);forward=unit(topology['palm_view']['forward'])
    lateral=unit(np.cross(forward,normal))
    return np.eye(3)-2*np.outer(lateral,lateral)


class WujiSurface:
    def __init__(self):
        self.model=mujoco.MjModel.from_xml_string(ET.tostring(hand_tree('wuji2'),encoding='unicode'))
        self.body_ids=[self.model.body(n).id for n in HAND_BODIES]
        self.meshes=[];self.face_offsets=[];self.vertex_offsets=[]
        vertices=[];faces=[];owners=[];nf=nv=0
        for owner,body in enumerate(self.body_ids):
            v,f=rigid_mesh_arrays(self.model,body)
            mesh=trimesh.Trimesh(v,f,process=True);mesh.fix_normals(multibody=True)
            self.meshes.append(mesh);self.face_offsets.append(nf);self.vertex_offsets.append(nv)
            vertices.extend(mesh.vertices.tolist());faces.extend((mesh.faces+nv).tolist())
            owners.extend([owner]*len(mesh.vertices));nv+=len(mesh.vertices);nf+=len(mesh.faces)
        self.local=np.array(vertices);self.faces=np.array(faces,int);self.owners=np.array(owners,int)
        root=Path(__file__).resolve().parents[1]/'data/grasp/assets/wuji2'
        digest=hashlib.sha256()
        for p in sorted(root.rglob('*')):
            if p.is_file() and (p.suffix.lower() in ('.xml','.stl') or p.name=='source.json'):
                digest.update(str(p.relative_to(root)).encode());digest.update(p.read_bytes())
        self.asset_hash=digest.hexdigest()
        self.neutral=mujoco.MjData(self.model);mujoco.mj_kinematics(self.model,self.neutral)
        self.axes=[];self.normals=[];self.lengths=[]
        for owner,body in enumerate(self.body_ids):
            if owner==0:
                roots=[self.neutral.xpos[self.body_ids[2+4*f]] for f in range(1,5)]
                endpoint=np.mean(roots,axis=0)-self.neutral.xpos[body]
                endpoint=endpoint@self.neutral.xmat[body].reshape(3,3)
            elif owner%4==0:
                f=(owner-1)//4;site=self.model.site('right_'+('thumb','index','middle','ring','pinky')[f]+'_tip')
                endpoint=site.pos.copy()
            else:
                endpoint=self.model.body_pos[self.body_ids[owner+1]].copy()
            self.axes.append(unit(endpoint));self.lengths.append(np.linalg.norm(endpoint))
            self.normals.append(palmar_axis(self.model,body,endpoint))
        self.axes=np.array(self.axes);self.normals=np.array(self.normals)
        self.bases=np.array([basis(a,n) for a,n in zip(self.axes,self.normals)])
        self.pools={}

    def candidates(self,owner):
        if owner in self.pools:return self.pools[owner]
        mesh=self.meshes[owner];rng=np.random.default_rng(714+owner)
        ids=rng.choice(len(mesh.faces),6144,p=mesh.area_faces/mesh.area)
        uv=rng.random((len(ids),2));flip=uv.sum(1)>1;uv[flip]=1-uv[flip]
        bary=np.c_[1-uv.sum(1),uv]
        points=(mesh.triangles[ids]*bary[...,None]).sum(1);normals=mesh.face_normals[ids]
        # Work only on the outer palmar envelope, never the mounting flange.
        valid=(normals@self.normals[owner])>.25
        if owner==0:
            z=points@self.axes[0];valid &= (z>0)&(z<self.lengths[0]*1.12)
        # A finite contact footprint must not straddle a sharp or open edge.
        edges=mesh.face_adjacency_edges[mesh.face_adjacency_angles>np.pi/4]
        boundary=mesh.edges_unique[np.bincount(mesh.edges_unique_inverse,minlength=len(mesh.edges_unique))==1]
        edges=np.concatenate((edges,boundary))
        for start in range(0,len(edges),256):
            a=mesh.vertices[edges[start:start+256,0]];e=mesh.vertices[edges[start:start+256,1]]-a
            t=np.clip(((points[:,None]-a)*e).sum(-1)/np.maximum((e*e).sum(-1),1e-20),0,1)
            distance=np.linalg.norm(points[:,None]-a-t[...,None]*e,axis=-1).min(1)
            valid &= distance>=.0015
        chosen=np.flatnonzero(valid)
        if len(chosen):
            # First visible surface from outside must be this point.
            locations,rays,_=mesh.ray.intersects_location(points[chosen]+normals[chosen]*.004,-normals[chosen],multiple_hits=False)
            visible=np.zeros(len(chosen),bool)
            visible[rays]=np.linalg.norm(locations-points[chosen[rays]],axis=1)<.00015
            valid[chosen]&=visible
        chosen=np.flatnonzero(valid)
        if len(chosen):
            tree=cKDTree(points)
            neighbors=tree.query_ball_point(points[chosen],.0015)
            smooth=np.array([np.min(normals[n]@normals[i])>=np.cos(np.pi/4) for i,n in zip(chosen,neighbors)])
            valid[chosen]&=smooth
        result={'points':points[valid],'normals':normals[valid],'face_ids':ids[valid],'barycentric':bary[valid]}
        self.pools[owner]=result
        return result

    def contact_surface(self,owner):
        """Native facets supported by the outer palmar candidate mask.

        Face masks are geometry-derived and auditable, not measured taxel
        registration. They are shared by mapping display and Eq. (2).
        """
        pool=self.candidates(owner);ids=np.unique(pool['face_ids'])
        mesh=self.meshes[owner]
        return dict(triangles=np.asarray(mesh.triangles)[ids],normals=np.asarray(mesh.face_normals)[ids],
                    face_ids=ids,policy=POLICY,physical_pad_verified=False)

    def patch_samples(self, owner, side='front', count=32):
        """Return deterministic local surface samples for one anatomical face.

        The candidate pool is already restricted to the visible palmar envelope.
        A link is split by local surface normals into front/left/right faces so
        a contact patch can match a region rather than a single mapped point.
        """
        pool = self.candidates(int(owner))
        points = np.asarray(pool['points'], dtype=np.float64)
        normals = np.asarray(pool['normals'], dtype=np.float64)
        if not len(points):
            return dict(points=np.empty((0, 3)), normals=np.empty((0, 3)))
        local_n = normals @ self.bases[int(owner)]
        if side == 'left':
            mask = local_n[:, 0] >= 0
        elif side == 'right':
            mask = local_n[:, 0] < 0
        else:
            mask = local_n[:, 1] >= 0
        ids = np.flatnonzero(mask)
        if len(ids) < max(4, count // 4):
            ids = np.arange(len(points))
        # Farthest-point subsampling is deterministic and keeps the patch
        # footprint instead of clustering all samples near one triangle.
        chosen = [int(ids[0])]
        distances = np.linalg.norm(points[ids] - points[ids[0]], axis=1)
        for _ in range(1, min(int(count), len(ids))):
            pick = int(np.argmax(distances)); chosen.append(int(ids[pick]))
            distances = np.minimum(distances, np.linalg.norm(points[ids] - points[ids[pick]], axis=1))
        chosen = np.asarray(chosen, dtype=np.int64)
        return dict(points=points[chosen], normals=normals[chosen],
                    face_ids=np.asarray(pool['face_ids'])[chosen], side=str(side), owner=int(owner))

    def mapping(self,topology,cache=None,rebuild=False):
        payload=[POLICY,self.asset_hash,topology['geometry_version'],topology['binding_version'],topology.get('surface_policy',CurvedTaxels.policy),topology.get('hand_side')]
        version=hashlib.sha256(json.dumps(payload).encode()).hexdigest()[:24]
        path=Path(cache)/(version+'.json') if cache else None
        if path and path.exists() and not rebuild:
            result=json.loads(path.read_text())
            if result.get('version')==version:return result
        v=np.array(topology['rest_vertices_m']);j=np.array(topology['rest_joints_m']);tip=np.array(topology['tip_anchors_m'])
        source,normals=CurvedTaxels(topology).evaluate(v)
        patch=np.array(topology['sid_patch_indices']);weights=np.array(topology['skin_weights'])
        palm_coordinates=(source-j[0])@basis(topology['palm_view']['forward'],topology['palm_view']['normal'])
        palm_coordinates[:,0]*=(-1 if topology.get('hand_side')!='right' else 1)
        palm_middle=np.median(palm_coordinates[patch>=14,0])
        skin=(weights[np.array(topology['faces'])[topology['sid_face_ids']]]*np.array(topology['sid_barycentric'])[...,None]).sum(1)
        records=[None]*460;failed=[]
        for p in range(20):
            sids=np.flatnonzero(patch==p)
            if p>=14:
                groups=[(0,0,sids)]
            else:
                f=next(f for f in range(5) if p<2+3*f) if p>=2 else 0
                if f==0:
                    owners=np.argmax(skin[sids][:,list(CHAINS[0][:2])],axis=1) if p==1 else np.full(len(sids),2)
                    groups=[(2+k,CHAINS[0][k],sids[owners==k]) for k in np.unique(owners)]
                else:
                    k=(4-p)%3  # patch order distal, intermediate, proximal
                    groups=[(1+4*f+1+k,CHAINS[f][k],sids)]
            for owner,joint,ids in groups:
                if not len(ids):continue
                pool=self.candidates(owner)
                points=pool['points'];nn=pool['normals'];keep=np.arange(len(points))
                if p>=14 and len(points):
                    local=points@self.bases[0];lo,hi=np.percentile(local,[0,100],axis=0)
                    row=(p-14)//2;col=(p-14)%2
                    u=(local[:,2]-lo[2])/max(hi[2]-lo[2],1e-6)
                    side=(local[:,0]-lo[0])/max(hi[0]-lo[0],1e-6)
                    high_side=palm_coordinates[ids,0].mean()>=palm_middle
                    keep=np.flatnonzero((u>=row/3)&(u<=(row+1)/3)&((side>=.5) if high_side else (side<.5)))
                    origin=j[0];axis=topology['palm_view']['forward'];normal=topology['palm_view']['normal']
                    length=np.linalg.norm(j[4]-j[0])
                else:
                    f=next(f for f,c in enumerate(CHAINS) if joint in c);k=CHAINS[f].index(joint)
                    origin=j[joint];end=j[CHAINS[f][k+1]] if k<2 else tip[f]
                    axis=end-origin;length=np.linalg.norm(axis)
                    normal=unit(normals[ids].mean(0))
                if len(keep)<len(ids):failed.extend(ids.tolist());continue
                sb=basis(axis,normal);tb=self.bases[owner]
                sp=(source[ids]-origin)@sb;sn=normals[ids]@sb
                # An anatomical reflection reverses the transverse coordinate
                # while forward and outward/pad axes retain their meanings.
                sp[:,0]*=(-1 if topology.get('hand_side')!='right' else 1);sn[:,0]*=(-1 if topology.get('hand_side')!='right' else 1)
                tp=points[keep]@tb;tn=nn[keep]@tb
                # Longitudinal coordinate and cross-section proportions retain
                # dimensions through distinct source/target anatomical scales.
                ss=np.array([max(np.percentile(abs(sp[:,0]),95),.002),max(np.percentile(abs(sp[:,1]),95),.002),max(length,.01)])
                ts=np.array([max(np.percentile(abs(tp[:,0]),95),.002),max(np.percentile(abs(tp[:,1]),95),.002),max(self.lengths[owner],.01)])
                a=sp/ss;b=tp/ts
                if p>=14:
                    # Match each palm tile's local rectangular coordinates.
                    a=(sp-sp.min(0))/np.maximum(np.ptp(sp,axis=0),.001)
                    b=(tp-tp.min(0))/np.maximum(np.ptp(tp,axis=0),.001)
                dot=sn@tn.T
                ray=unit(sp/ss)@unit(tp/ts).T
                cost=4*((a[:,None]-b[None])**2).sum(-1)+3*(1-dot)+2*(1-ray)
                cost[dot<.5]=1e9
                rows,cols=linear_sum_assignment(cost)
                for row,col in zip(rows,cols):
                    sid=int(ids[row]);c=int(keep[col])
                    if cost[row,col]>=1e8:failed.append(sid);continue
                    records[sid]={'sid':sid,'patch':int(p),'owner':int(owner),'link':HAND_BODIES[owner],
                        'source_joint_index':0 if p>=14 else 1+4*f+k,
                        'face_id':int(self.face_offsets[owner]+pool['face_ids'][c]),'barycentric':pool['barycentric'][c].tolist(),
                        'local_position_m':points[c].tolist(),'local_normal':nn[c].tolist(),
                        'source_origin_m':np.asarray(origin).tolist(),'source_ray':unit(source[sid]-origin).tolist(),
                        'normal_error_deg':float(np.degrees(np.arccos(np.clip(dot[row,col],-1,1)))),
                        'cost':float(cost[row,col]),'footprint_m':.0015,'outer_visible':True}
        result={'version':version,'policy':POLICY,'ready':not failed and all(r is not None for r in records),
                'records':records,'failed_sids':sorted(set(failed)), 'asset_sha256':self.asset_hash,
                'geometry_version':topology['geometry_version'],'binding_version':topology['binding_version'],
                'source_surface':topology.get('surface_policy',CurvedTaxels.policy),'virtual_sensors':True,
                'mirror_matrix':mirror_matrix(topology).tolist(),'source_note':'anatomical initialization, not measured glove or robot taxel placement'}
        if path:
            path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result));tmp.replace(path)
        return result

    def topology(self,mapping):
        return {'ready':True,'hand_side':'right','coordinate_frame':'shared_virtual_world','units':'m',
                'local_vertices_m':self.local.tolist(),'faces':self.faces.tolist(),'vertex_owners':self.owners.tolist(),
                'link_names':HAND_BODIES,'asset_sha256':self.asset_hash,'mapping':mapping}
