"""Material semantic anchors, exact convex/mesh distances, capsule clearances."""
import hashlib
import json
import numpy as np
import torch
from scipy.spatial import ConvexHull,QhullError
import trimesh
from .tactile_surface import CurvedTaxels
from .wuji_surface import CHAINS,unit
from .contact_force import estimate_normal_resultant
from retarget.env.grasp.pose_gpu import palm_frame


def contact_weights(contact,patches,pose_timestamp_ms,tactile_timestamp_ms,enabled=True):
    detector=contact.get('timestamp_ms',tactile_timestamp_ms)
    valid=bool(enabled and contact.get('ready') and isinstance(detector,(int,float)) and np.isfinite(detector) and
               abs(pose_timestamp_ms-tactile_timestamp_ms)<=25 and abs(pose_timestamp_ms-detector)<=25)
    raw=contact.get('cell_contact') or [];mask=np.array([x is True for x in raw],bool) if len(raw)==460 else np.zeros(460,bool)
    if not valid:mask[:]=False
    response=np.clip(np.nan_to_num(np.asarray(contact.get('normalized') or [0.]*460,float)),0,1)
    if response.shape!=(460,):raise ValueError('460 tactile responses required')
    centers=np.zeros((20,460));cloud=np.zeros(460);active=np.zeros(20,bool);strength=np.zeros(20)
    for p in range(20):
        ids=np.flatnonzero(mask & (patches==p))
        if not len(ids):continue
        v=response[ids];v=v if v.max()>1e-8 else np.ones(len(ids))
        centers[p,ids]=v/v.sum();cloud[ids]=v/(len(ids)*v.max());active[p]=True;strength[p]=response[ids].mean()
    return dict(ready=valid,mask=mask,response=response,center=centers,cloud=cloud,active=active,strength=strength)


def chart_point(triangles,origin,axis,normal,uv,reference=None):
    """Closest point in a 2-D palmar chart, returning a material surface point."""
    lateral=unit(np.cross(normal,axis));chart=np.stack(((triangles-origin)@axis,(triangles-origin)@lateral),-1)
    a,b,c=chart.transpose(1,0,2);ab=b-a;ac=c-a;delta=np.asarray(uv)-a
    cross=lambda x,y:x[...,0]*y[...,1]-x[...,1]*y[...,0]
    det=cross(ab,ac);safe=np.where(abs(det)>1e-12,det,1.)
    v=cross(delta,ac)/safe;w=cross(ab,delta)/safe;bc=np.stack((1-v-w,v,w),-1)
    valid=(bc.min(-1)>=0)&(abs(det)>1e-12)
    options=[bc];scores=[np.where(valid,0,np.inf)]
    for i,j in ((0,1),(1,2),(2,0)):
        start=chart[:,i];edge=chart[:,j]-start;t=np.clip(((uv-start)*edge).sum(-1)/np.maximum((edge*edge).sum(-1),1e-20),0,1)
        bary=np.zeros_like(bc);bary[:,i]=1-t;bary[:,j]=t
        options.append(bary);scores.append(((start+t[:,None]*edge-uv)**2).sum(-1))
    score=np.stack(scores,-1)
    if reference is not None:
        # Several curved faces can overlap in a chart. A flat argmin picks
        # whichever triangle appears first, even on another depth layer.
        points=np.stack([(bc[...,None]*triangles).sum(1) for bc in options],1)
        eligible=score<=score.min()+1e-14
        score=np.where(eligible,((points-reference)**2).sum(-1),np.inf)
    f,k=np.unravel_index(score.argmin(),score.shape)
    bary=options[k][f];return bary@triangles[f],f,bary


class AnchorExtractor:
    def __init__(self,topology,mirror):
        self.topology=topology;self.mirror=mirror;self.faces=np.asarray(topology['faces'],int)
        self.taxels=CurvedTaxels(topology);self.patches=np.asarray(topology['sid_patch_indices'],int)
        weights=np.asarray(topology['skin_weights']);faceweights=weights[self.faces].mean(1)
        facej=faceweights.argmax(1);self.region_faces=[np.flatnonzero(facej==0)]
        for c in CHAINS:
            self.region_faces.extend(np.flatnonzero(facej==j) for j in c)
        bary=np.asarray(topology['sid_barycentric']);sidfaces=np.asarray(topology['sid_face_ids'])
        sw=(weights[self.faces[sidfaces]]*bary[...,None]).sum(1)
        self.regions=np.zeros(460,int)
        for sid,p in enumerate(self.patches):
            if p>=14:continue
            if p<2:f=0;k=2 if p==0 else int(sw[sid,list(CHAINS[0][:2])].argmax())
            else:f=1+(p-2)//3;k=2-(p-2)%3
            self.regions[sid]=1+f*3+k
        rest=np.asarray(topology['rest_vertices_m']);_,taxel_normals=self.taxels.evaluate(rest)
        tri=rest[self.faces];face_normals=unit(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]))
        # Keep only palmar faces; averaging an entire closed link cancels normals.
        for region,faces in enumerate(self.region_faces):
            ids=np.flatnonzero(self.regions==region)
            if not len(ids):raise ValueError('No material taxels for MANO semantic region')
            seed=unit(taxel_normals[ids].mean(0));keep=faces[(face_normals[faces]@seed)>.25]
            if not len(keep):raise ValueError('MANO semantic region has no palmar surface')
            # Registered side contacts can lie beyond the average palmar
            # normal mask; their actual material support must remain eligible.
            self.region_faces[region]=np.union1d(keep,sidfaces[ids])
        if any(not len(f) for f in self.region_faces):raise ValueError('MANO semantic region missing')

    def source(self,frame):
        v=np.asarray(frame['vertices_m'],float);j=np.asarray(frame['joints_m'],float)
        if v.shape!=(778,3) or j.shape!=(21,3) or not np.isfinite(v).all() or not np.isfinite(j).all():raise ValueError('Finite MANO mesh/joints required')
        if np.any(np.linalg.norm(np.diff(j[1:].reshape(5,4,3),axis=1),axis=-1)<1e-6):raise ValueError('Degenerate MANO bone direction')
        if np.linalg.norm(np.cross(j[9]-j[0],j[5]-j[17]))<1e-8:raise ValueError('Degenerate MANO palm frame')
        points,normals=self.taxels.evaluate(v)
        return v@self.mirror.T,j@self.mirror.T,points@self.mirror.T,normals@self.mirror.T

    def chart(self,region,vertices,joints):
        if region==0:
            origin=joints[0];axis=unit(joints[9]-origin)
        else:
            f,k=divmod(region-1,3);origin=joints[1+4*f+k];axis=unit(joints[2+4*f+k]-origin)
        tri=vertices[self.faces[self.region_faces[region]]]
        normal=unit(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]).sum(0))
        # Reflections reverse winding; use the physical outward normal.
        if np.linalg.det(self.mirror)<0:normal=-normal
        return origin,axis,normal,tri

    def extract(self,frame,enabled=True):
        v,j,p,n=self.source(frame);stamp=float(frame['timestamp_ms']);tt=frame.get('tactile_timestamp_ms')
        weights=contact_weights(frame.get('source_contact') or {},self.patches,stamp,float(tt) if isinstance(tt,(int,float)) else -1e20,enabled)
        anchors=[]
        for region in range(16):
            ids=np.flatnonzero(weights['mask']&(self.regions==region))
            if not len(ids):continue
            values=weights['response'][ids];force=float(values.sum());values=values if values.sum()>1e-12 else np.ones(len(ids))
            origin,axis,normal,tri=self.chart(region,v,j);lateral=unit(np.cross(normal,axis))
            uv=np.average(np.stack(((p[ids]-origin)@axis,(p[ids]-origin)@lateral),-1),axis=0,weights=values)
            reference=np.average(p[ids],axis=0,weights=values)
            point,face,bary=chart_point(tri,origin,axis,normal,uv,reference)
            source_face=int(self.region_faces[region][face])
            if len(ids)==1:
                # An isolated taxel is already a registered material point.
                # Reprojection must not move it to another overlapping facet.
                point=p[ids[0]];source_face=int(self.topology['sid_face_ids'][ids[0]])
                bary=np.asarray(self.topology['sid_barycentric'][ids[0]])
            f,k=divmod(region-1,3) if region else (-1,-1)
            u=0. if region==0 else float(np.clip((point-origin)@axis/max(np.linalg.norm(j[2+4*f+k]-origin),1e-12),0,1))
            anchors.append(dict(region=region,finger=f,segment=k,owner=0 if region==0 else 2+4*f+k,
                position_m=point.tolist(),direction=unit(np.average(n[ids],axis=0,weights=values)).tolist(),force=force,
                uv_m=uv.tolist(),u=u,source_face=source_face,barycentric=bary.tolist(),sids=ids.tolist(),
                force_estimate=estimate_normal_resultant(n[ids],weights['response'][ids]),
                source_material_registered=bool(set(ids)<=set((self.topology.get('contact_registration') or {}).get('verified_sids',[])))))
        return (v,j,p,n),weights,anchors

    def relocate(self,anchor,uv,vertices,joints):
        a=dict(anchor);o,x,n,tri=self.chart(a['region'],vertices,joints);point,f,bary=chart_point(tri,o,x,n,uv,np.asarray(a['position_m']))
        a.update(position_m=point.tolist(),uv_m=np.asarray(uv).tolist(),source_face=int(self.region_faces[a['region']][f]),barycentric=bary.tolist())
        if a['region']:
            fi,k=divmod(a['region']-1,3);a['u']=float(np.clip((point-o)@x/max(np.linalg.norm(joints[2+4*fi+k]-o),1e-12),0,1))
        return a


def source_wrist(joints):
    t=torch.as_tensor(joints,dtype=torch.float64)
    _,r=palm_frame(t[0],t[[1,5,9,13,17]])
    return joints[0],r.numpy()


def closest_surface_points(points,triangles,chunk=128):
    """Exact closest features on a fixed triangle set, with material coordinates.

    Eq. (2) reduces to this query at (c_local + alpha*psi)/(1+alpha).
    It does not require discretizing a contact surface into virtual taxels.
    """
    best=points.new_full((len(points),),torch.inf)
    selected=torch.zeros(len(points),dtype=torch.long,device=points.device)
    bary=points.new_zeros((len(points),3))
    for start in range(0,len(triangles),chunk):
        tri=triangles[start:start+chunk];a,b,c=tri.unbind(1);ab=b-a;ac=c-a;d=points[:,None]-a
        aa=(ab*ab).sum(-1);bb=(ab*ac).sum(-1);cc=(ac*ac).sum(-1)
        ad=(d*ab).sum(-1);cd=(d*ac).sum(-1);det=aa*cc-bb*bb
        safe=torch.where(det>1e-24,det,torch.ones_like(det))
        v=(cc*ad-bb*cd)/safe;w=(aa*cd-bb*ad)/safe
        bc=torch.stack((1-v-w,v,w),-1)
        distance=((bc[...,None]*tri).sum(-2)-points[:,None]).square().sum(-1)
        distance=torch.where((bc.amin(-1)>=0)&(det>1e-24),distance,torch.inf)
        for i,j in ((0,1),(1,2),(2,0)):
            p=tri[:,i];edge=tri[:,j]-p
            t=(((points[:,None]-p)*edge).sum(-1)/edge.square().sum(-1).clamp_min(1e-24)).clamp(0,1)
            value=(p+t[...,None]*edge-points[:,None]).square().sum(-1)
            use=value<distance;candidate=torch.zeros_like(bc);candidate[...,i]=1-t;candidate[...,j]=t
            bc=torch.where(use[...,None],candidate,bc);distance=torch.minimum(distance,value)
        value,index=distance.min(-1);use=value<best
        best=torch.minimum(best,value);selected=torch.where(use,index+start,selected)
        bary=torch.where(use[:,None],bc[torch.arange(len(points),device=points.device),index],bary)
    local=(triangles[selected]*bary[...,None]).sum(1)
    return local,selected,bary


def triangle_distance(points,triangles,chunk=64):
    """Exact closest-feature distance, GPU batch; no SDF grid approximation."""
    shape=points.shape[:-1];p=points.reshape(-1,3);answer=torch.full((len(p),),torch.inf,device=p.device,dtype=p.dtype)
    # Point chunks cap memory for mesh/hull facets without reducing samples.
    point_chunk=max(1024,min(65536,2_000_000//max(1,min(chunk,len(triangles)))))
    for start in range(0,len(p),point_chunk):
        x=p[start:start+point_chunk,None];best=answer[start:start+point_chunk]
        for t in triangles.split(chunk):
            a,b,c=t.unbind(1);ab=b-a;ac=c-a;n=torch.linalg.cross(ab,ac);n2=n.square().sum(-1).clamp_min(1e-20)
            delta=x-a;plane=(delta*n).sum(-1);proj=delta-plane[...,None]*n/n2[...,None]
            # proj is already relative to vertex a (delta = x-a).
            v=proj;d00=(ab*ab).sum(-1);d01=(ab*ac).sum(-1);d11=(ac*ac).sum(-1)
            d20=(v*ab).sum(-1);d21=(v*ac).sum(-1);den=(d00*d11-d01*d01).clamp_min(1e-20)
            u=(d11*d20-d01*d21)/den;w=(d00*d21-d01*d20)/den
            dist=torch.where((u>=0)&(w>=0)&(u+w<=1),plane.square()/n2,torch.inf)
            for lo,hi in ((a,b),(b,c),(c,a)):
                edge=hi-lo;tau=((x-lo)*edge).sum(-1)/edge.square().sum(-1).clamp_min(1e-20)
                dist=torch.minimum(dist,(x-lo-tau.clamp(0,1)[...,None]*edge).square().sum(-1))
            best=torch.minimum(best,dist.amin(-1).sqrt())
        answer[start:start+len(best)]=best
    return answer.reshape(shape)


class PseudoObject:
    kind='pseudo_object'
    def __init__(self,device):
        self.device=device;self.points=np.empty((0,3));self.hull=None;self.version=None;self.frozen=False;self.mode='causal';self.range=[None,None]
    def add(self,anchors,joints,stamp):
        if self.frozen:return
        origin,r=source_wrist(joints);p=np.array([a['position_m'] for a in anchors if a['region']!=0])
        if not len(p):return
        self.range[0]=stamp if self.range[0] is None else self.range[0];self.range[1]=stamp
        points=(p-origin)@r
        merged=np.unique(np.round(np.vstack((self.points,points)),7),axis=0)
        if len(merged)==len(self.points) and np.array_equal(merged,self.points):return
        self.points=merged;self.range[0]=stamp if self.range[0] is None else self.range[0];self.range[1]=stamp
        self.hull=None
        if len(merged)<4 or np.linalg.matrix_rank(merged-merged.mean(0),tol=1e-7)<3:return
        try:hull=ConvexHull(merged)
        except QhullError:return
        # Interior Euclidean SDF is distance to the nearest supporting plane.
        self.planes=torch.tensor(hull.equations,device=self.device,dtype=torch.float32)
        self.hull=hull;self.version=hashlib.sha256(merged[hull.vertices].tobytes()).hexdigest()[:16]
        self.points=merged[hull.vertices] # interior points cannot change future hulls
    def penetration(self,points,joints):
        if self.hull is None:return torch.zeros(points.shape[:-1],device=points.device,dtype=points.dtype)
        o,r=source_wrist(joints);x=(points-points.new_tensor(o))@points.new_tensor(r)
        distances=x@self.planes[:,:3].T+self.planes[:,3]
        return (-distances.amax(-1)).clamp_min(0)
    def metadata(self):return dict(type=self.kind,mode=self.mode,ready=self.hull is not None,version=self.version,
        vertices=len(self.points),source_range_ms=self.range,reason=None if self.hull is not None else 'fewer_than_four_non_coplanar_points')


class TrackedMesh:
    kind='tracked_mesh'
    def __init__(self,vertices,faces,device,transform=None):
        mesh=trimesh.Trimesh(vertices,faces,process=True)
        if not mesh.is_watertight:raise ValueError('Tracked object mesh must be watertight for signed distances')
        mesh.fix_normals();self.device=device
        self.vertices=np.asarray(mesh.vertices);self.faces=np.asarray(mesh.faces)
        self.triangles=torch.tensor(np.asarray(mesh.triangles),device=device,dtype=torch.float32)
        self.transform=np.eye(4) if transform is None else np.asarray(transform,float)
        if self.transform.shape!=(4,4) or not np.isfinite(self.transform).all() or not np.allclose(self.transform[3],[0,0,0,1]) or not np.allclose(self.transform[:3,:3].T@self.transform[:3,:3],np.eye(3),atol=1e-5) or np.linalg.det(self.transform[:3,:3])<.99:raise ValueError('Invalid tracked object rigid transform')
        self.version=hashlib.sha256(mesh.vertices.tobytes()+mesh.faces.tobytes()).hexdigest()[:16]
    def penetration(self,points,joints=None):
        tf=points.new_tensor(self.transform);local=(points-tf[:3,3])@tf[:3,:3];shape=local.shape[:-1]
        p=local.reshape(-1,3);inside=[]
        # Generalized winding number sign; deterministic, no CPU ray queries.
        point_chunk=max(1024,min(65536,2_000_000//max(1,min(64,len(self.triangles)))))
        for x in p.split(point_chunk):
            winding=torch.zeros(len(x),device=x.device)
            for tri in self.triangles.split(64):
                a=tri[None,:,0]-x[:,None];b=tri[None,:,1]-x[:,None];c=tri[None,:,2]-x[:,None]
                la=a.norm(dim=-1);lb=b.norm(dim=-1);lc=c.norm(dim=-1)
                num=(a*torch.linalg.cross(b,c)).sum(-1)
                den=la*lb*lc+(a*b).sum(-1)*lc+(b*c).sum(-1)*la+(c*a).sum(-1)*lb
                winding+=2*torch.atan2(num,den).sum(-1)
            inside.append(winding.abs()>2*np.pi)
        mask=torch.cat(inside).reshape(shape)
        return triangle_distance(local,self.triangles)*mask
    def descriptor(self):return dict(type=self.kind,vertices_m=self.vertices.tolist(),faces=self.faces.tolist(),transform=self.transform.tolist(),version=self.version)
    def metadata(self):return dict(type=self.kind,mode='tracked',ready=True,version=self.version,transform=self.transform.tolist())


def segment_gap(a,b,c,d):
    """Exact segment-to-segment distance, including parallel/zero segments."""
    u=b-a;v=d-c;w=a-c
    aa=(u*u).sum(-1);bb=(u*v).sum(-1);cc=(v*v).sum(-1);dd=(u*w).sum(-1);ee=(v*w).sum(-1)
    den=aa*cc-bb*bb;s=(bb*ee-cc*dd)/den.clamp_min(1e-20);t=(aa*ee-bb*dd)/den.clamp_min(1e-20)
    interior=(den>1e-16)&(s>=0)&(s<=1)&(t>=0)&(t<=1)
    dist=torch.where(interior,(w+s[...,None]*u-t[...,None]*v).norm(dim=-1),torch.inf)
    for p,x,y in ((a,c,d),(b,c,d),(c,a,b),(d,a,b)):
        e=y-x;alpha=((p-x)*e).sum(-1)/e.square().sum(-1).clamp_min(1e-20)
        dist=torch.minimum(dist,(p-x-alpha.clamp(0,1)[...,None]*e).norm(dim=-1))
    return dist
