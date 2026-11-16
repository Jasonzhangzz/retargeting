"""Primitive mesh transport and full-triangle convex collision utilities."""
import numpy as np
from scipy.spatial import ConvexHull
import trimesh
from .cart_protocol import DIRECTIONS

from .cart_convex import REVISION

def hull_data(points):
    points=np.asarray(points,float);h=ConvexHull(points)
    vertices=points[h.vertices];h=ConvexHull(vertices)
    faces=[]
    for tri,plane in zip(h.simplices,h.equations):
        a,b,c=vertices[tri]
        if np.dot(np.cross(b-a,c-a),plane[:3])<0:tri=tri[[0,2,1]]
        faces.append(tri.tolist())
    edges=sorted({tuple(sorted((int(a),int(b)))) for f in faces for a,b in zip(f,f[1:]+f[:1])})
    center=vertices.mean(0)
    return dict(ready=True,status='volume',algorithm=REVISION,dimension=3,solid=True,
        vertices_m=vertices.tolist(),faces=faces,edges=[list(e) for e in edges],
        planes=h.equations.tolist(),volume_m3=float(h.volume),volume_ml=float(h.volume*1e6),
        surface_area_m2=float(h.area),area_cm2=float(h.area*1e4),center_m=center.tolist(),
        support_m=((vertices-center)@DIRECTIONS.T).max(0).tolist(),
        hull_vertex_count=len(vertices),temporal_accumulation=False)

def triangle_interior_point(tri, planes, eps=1e-7):
    """Clip an entire triangle against the shrunken convex solid, not vertices."""
    poly=list(tri)
    for plane in planes:
        if not poly:return None
        result=[]
        for a,b in zip(poly,poly[1:]+poly[:1]):
            da=float(a@plane[:3]+plane[3]+eps);db=float(b@plane[:3]+plane[3]+eps)
            if da<=0:result.append(a)
            if (da<0)!=(db<0):result.append(a+(b-a)*da/(da-db))
        poly=result
    return np.mean(poly,axis=0) if len(poly)>=3 else None

def triangle_intersects_interior(tri,planes,eps=1e-7):
    return triangle_interior_point(tri,planes,eps) is not None

class FreeSpace:
    def __init__(self,vertices,faces):
        self.mesh=trimesh.Trimesh(np.asarray(vertices),np.asarray(faces),process=False)
        self.triangles=self.mesh.triangles
        self.tree=self.mesh.triangles_tree
        self.seed_validity={}
        from .cart_mesh_query import SignedMesh
        self.signed=SignedMesh(vertices,faces)
    def clear(self,hull,seed):
        self.last_collision_point=None
        v=np.asarray(hull['vertices_m']);planes=np.asarray(hull['planes'])
        bounds=np.r_[v.min(0),v.max(0)]
        ids=list(self.tree.intersection(bounds))
        if ids:
            tri=self.triangles[ids]
            # A triangle wholly outside one plane cannot intersect the solid.
            for plane in planes:
                tri=tri[((tri@plane[:3]+plane[3]).min(1)<=-1e-7)]
                if not len(tri):break
            for t in tri:
                point=triangle_interior_point(t,planes)
                if point is not None:self.last_collision_point=point;return False
        # The supplied seed is inside the object; testing it also covers an
        # object entirely contained in a hand without any boundary crossing.
        key=tuple(np.asarray(seed).tolist())
        if key not in self.seed_validity:self.seed_validity[key]=bool(self.signed.distance(np.asarray(seed)[None])[0]>=0)
        if not self.seed_validity[key]:self.last_collision_point=np.asarray(seed)
        return self.seed_validity[key]

def patch_samples(vertices,faces,regions,name,count=48):
    ids=np.asarray(regions.get(name,[]),int)
    if not len(ids):raise ValueError('缺少语义曲面 '+name)
    ids=ids[np.linspace(0,len(ids)-1,min(count,len(ids))).astype(int)]
    tri=np.asarray(vertices)[np.asarray(faces)[ids]]
    normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-12)
    return tri.mean(1),normal

def region_pair(vertices,faces,regions,pair):
    from .contact_templates import nearest_anchors
    gap,anchors=nearest_anchors(np.asarray(vertices),np.asarray(faces),regions,*pair)
    points=[];normals=[]
    for a in anchors:
        tri=np.asarray(vertices)[np.asarray(faces)[a['face']]]
        points.append((np.asarray(a['barycentric'])@tri).tolist())
        n=np.cross(tri[1]-tri[0],tri[2]-tri[0]);normals.append((n/max(np.linalg.norm(n),1e-12)).tolist())
    return dict(regions=list(pair),gap_m=float(gap),points_m=points,normals=normals,anchors=anchors)

def estimate(vertices,faces,regions,stage_id,full=False,budget_ms=120,witnesses=None,joints=None,previous=None):
    from .cart_convex import estimate_convex
    return estimate_convex(vertices,faces,regions,stage_id,full,budget_ms,witnesses,joints,previous)


def transform_hull(hull,rotation,translation):
    result=dict(hull);r=np.asarray(rotation);t=np.asarray(translation)
    for metadata_key in ('primitive','initialization'):
        if metadata_key not in hull:continue
        primitive=dict(hull[metadata_key])
        primitive['center_m']=(np.asarray(primitive['center_m'])@r.T+t).tolist()
        orientation=r@np.asarray(primitive['rotation'])
        # Cylinder, sphere and box seeds are symmetric under local X reflection.
        # Keep a proper rigid orientation when mirroring to a left hand.
        if np.linalg.det(orientation)<0:orientation[:,0]*=-1
        primitive['rotation']=orientation.tolist()
        result[metadata_key]=primitive
    for key in ('vertices_m','center_m','contact_points_m'):
        if key in hull:result[key]=(np.asarray(hull[key]).reshape(-1,3)@r.T+t).reshape(np.asarray(hull[key]).shape).tolist()
    if 'planes' in hull:
        p=np.asarray(hull['planes']);n=p[:,:3]@r.T
        result['planes']=np.c_[n,p[:,3]-n@t].tolist()
    if 'faces' in hull and np.linalg.det(r)<0:
        result['faces']=np.asarray(hull['faces'])[:,[0,2,1]].tolist()
    if 'support_m' in hull:
        result['support_m']=((np.asarray(result['vertices_m'])-result['center_m'])@DIRECTIONS.T).max(0).tolist()
    if 'pairs' in hull:
        result['pairs']=[]
        for pair in hull['pairs']:
            p=dict(pair)
            if 'points_m' in p:p['points_m']=(np.asarray(p['points_m'])@r.T+t).tolist()
            if 'normals' in p:p['normals']=(np.asarray(p['normals'])@r.T).tolist()
            result['pairs'].append(p)
    if 'contacts' in hull:
        result['contacts']=[]
        for contact in hull['contacts']:
            c=dict(contact)
            for key in ('point_m','object_point_m'):
                if key in c:c[key]=(np.asarray(c[key])@r.T+t).tolist()
            for key in ('normal','object_normal'):
                if key in c:c[key]=(np.asarray(c[key])@r.T).tolist()
            result['contacts'].append(c)
    if 'delta_geometry' in hull:
        delta=dict(hull['delta_geometry'])
        if 'vertices_m' in delta:delta['vertices_m']=(np.asarray(delta['vertices_m']).reshape(-1,3)@r.T+t).tolist()
        if 'offsets_m' in delta:delta['offsets_m']=(np.asarray(delta['offsets_m']).reshape(-1,3)@r.T).tolist()
        result['delta_geometry']=delta
    if hull.get('diagnostic_candidate'):
        result['diagnostic_candidate']=transform_hull(hull['diagnostic_candidate'],rotation,translation)
    result['coordinate_frame']='shared_virtual_world'
    return result
