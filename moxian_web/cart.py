"""Current-fingertip convex hull (CART), with explicit lower-dimensional cases.

Only five measured/modelled fingertip positions are used. No temporal union,
palm/wrist point, artificial thickness, object inference, or tactile assumption.
The small supporting-face enumeration avoids adding SciPy to the WebUI process.
"""
from itertools import combinations
import threading
import time
import numpy as np
from retarget.baseline.geort.adapter import SourceAdapter, FINGERS

ALGORITHM='cart_current_five_fingertips_v1'
DISTANCE_EPS=1e-7
RANK_EPS=1e-6


def _polygon(points, indices, normal):
    """2-D monotone-chain boundary in an oriented supporting plane."""
    center=points[indices].mean(0)
    axis=np.eye(3)[np.argmin(np.abs(normal))]
    u=np.cross(axis,normal);u/=np.linalg.norm(u);v=np.cross(normal,u)
    coords={i:((points[i]-center)@u,(points[i]-center)@v) for i in indices}
    ordered=sorted(indices,key=lambda i:coords[i])
    def cross(a,b,c):
        x,y=np.asarray(coords[b])-coords[a];z,w=np.asarray(coords[c])-coords[a]
        return x*w-y*z
    lower=[];upper=[]
    for i in ordered:
        while len(lower)>=2 and cross(lower[-2],lower[-1],i)<=1e-14:lower.pop()
        lower.append(i)
    for i in reversed(ordered):
        while len(upper)>=2 and cross(upper[-2],upper[-1],i)<=1e-14:upper.pop()
        upper.append(i)
    return lower[:-1]+upper[:-1]


def fingertip_hull(points):
    p=np.asarray(points,dtype=np.float64)
    if p.shape!=(5,3) or not np.isfinite(p).all():
        raise ValueError('CART 需要五个有限的三维指尖位置')
    unique=[]
    for point in p:
        if not unique or all(np.linalg.norm(point-q)>DISTANCE_EPS for q in unique):unique.append(point)
    vertices=np.asarray(unique);center=vertices.mean(0);centered=vertices-center
    _,singular,basis=np.linalg.svd(centered,full_matrices=True)
    dimension=int((singular>RANK_EPS).sum());faces=[];edges=set();polygons=[]
    if dimension==3:
        planes=set()
        for a,b,c in combinations(range(len(vertices)),3):
            normal=np.cross(vertices[b]-vertices[a],vertices[c]-vertices[a]);norm=np.linalg.norm(normal)
            if norm<1e-14:continue
            normal/=norm;distances=(vertices-vertices[a])@normal
            if distances.max()>DISTANCE_EPS and distances.min()<-DISTANCE_EPS:continue
            if distances.max()>DISTANCE_EPS:normal=-normal;distances=-distances
            indices=tuple(np.flatnonzero(np.abs(distances)<=DISTANCE_EPS).tolist())
            if indices in planes:continue
            planes.add(indices);polygons.append(_polygon(vertices,list(indices),normal))
    elif dimension==2:
        polygons=[_polygon(vertices,list(range(len(vertices))),basis[-1])]
    elif dimension==1:
        projection=centered@basis[0];a=int(projection.argmin());b=int(projection.argmax())
        if a!=b:edges.add(tuple(sorted((a,b))))
    for polygon in polygons:
        if len(polygon)<3:continue
        edges.update(tuple(sorted((a,b))) for a,b in zip(polygon,polygon[1:]+polygon[:1]))
        faces.extend([polygon[0],polygon[i],polygon[i+1]] for i in range(1,len(polygon)-1))
    area=volume=0.
    for a,b,c in faces:
        x,y,z=centered[[a,b,c]]
        area+=float(np.linalg.norm(np.cross(y-x,z-x))/2)
        if dimension==3:volume+=float(x@np.cross(y,z)/6)
    hull_indices=sorted({i for f in faces for i in f}|{i for edge in edges for i in edge})
    if not hull_indices:hull_indices=[0]
    return dict(algorithm=ALGORITHM,ready=True,status=('point','line','planar','volume')[dimension],
        dimension=dimension,solid=dimension==3,finger_order=list(FINGERS),finger_points_m=p.tolist(),
        vertices_m=vertices.tolist(),faces=faces,edges=[list(e) for e in sorted(edges)],hull_vertex_indices=hull_indices,
        hull_vertex_count=len(hull_indices),volume_m3=max(0.,volume) if dimension==3 else 0.,
        volume_ml=max(0.,volume)*1e6 if dimension==3 else 0.,surface_area_m2=area,
        area_cm2=area*1e4,rank_tolerance_m=RANK_EPS,
        interpretation='current_fingertips_region',temporal_accumulation=False)


class CartEstimator:
    def __init__(self):
        self.lock=threading.Lock();self.received=0.
        self.state=dict(ready=False,status='waiting_source',algorithm=ALGORITHM,error=None)

    def observe(self, frame, topology, joints, serial):
        try:
            if not frame.get('ready') or not frame.get('calibrated') or frame.get('preview'):
                raise ValueError('等待已应用标定后的有效 Moxian 手部模型')
            adapter=SourceAdapter(topology);identity=adapter.identity(serial,frame)
            points=np.asarray(joints,float)[[chain[-1] for chain in topology['joint_chains']]]
            started=time.perf_counter();hull=fingertip_hull(points)
            hull.update(timestamp_ms=frame['timestamp_ms'],geometry_version=frame['geometry_version'],
                        binding_version=frame.get('binding_version'),hand_side=frame['hand_side'],
                        coordinate_frame='wrist_local',source=identity,processing_ms=(time.perf_counter()-started)*1000)
            with self.lock:self.state=hull;self.received=time.monotonic()
            return hull
        except Exception as exc:
            with self.lock:self.state=dict(ready=False,status='invalid_source',algorithm=ALGORITHM,error=str(exc))
            return None

    def snapshot(self,connected=True):
        with self.lock:result=dict(self.state);received=self.received
        age=(time.monotonic()-received)*1000 if received else None
        result['age_ms']=age
        if not connected or age is None or age>500:
            result.update(ready=False,status='stale' if connected else 'disconnected',error='等待实时 Moxian 指尖')
        return result
