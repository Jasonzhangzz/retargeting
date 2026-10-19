"""Batched bounded local-patch projection; geometry stays cached on Warp."""
import numpy as np
import warp as wp


@wp.func
def closest_triangle(point: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3):
    ab, ac = b-a, c-a
    n = wp.cross(ab, ac)
    area2 = wp.max(wp.dot(n,n),1.e-30)
    ap = point-a
    d00,d01,d11 = wp.dot(ab,ab),wp.dot(ab,ac),wp.dot(ac,ac)
    v = (d11*wp.dot(ap,ab)-d01*wp.dot(ap,ac))/area2
    w = (d00*wp.dot(ap,ac)-d01*wp.dot(ap,ab))/area2
    projected = point-n*(wp.dot(ap,n)/area2)
    nearest = a
    distance = float(1.e30)
    if v >= 0. and w >= 0. and v+w <= 1.:
        nearest = projected
        distance = wp.length_sq(projected-point)
    for edge in range(3):
        start,end = a,b
        if edge == 1:
            start,end = b,c
        elif edge == 2:
            start,end = c,a
        delta = end-start
        t = wp.clamp(wp.dot(point-start,delta)/wp.max(wp.length_sq(delta),1.e-30),0.,1.)
        candidate = start+t*delta
        d = wp.length_sq(candidate-point)
        if d < distance:
            distance,nearest = d,candidate
    return nearest


@wp.kernel(enable_backward=False)
def project_patch(proposals: wp.array(dtype=wp.vec3), templates: wp.array(dtype=wp.vec3),
                  starts: wp.array(dtype=int), counts: wp.array(dtype=int),
                  anchors: wp.array(dtype=wp.vec3), anchor_normals: wp.array(dtype=wp.vec3),
                  triangles: wp.array2d(dtype=wp.vec3), normals: wp.array(dtype=wp.vec3),
                  samples: int, contacts: int, radius: float,
                  output: wp.array(dtype=wp.vec3), output_normals: wp.array(dtype=wp.vec3)):
    row = wp.tid()
    contact = row%contacts
    batch = row//(samples*contacts)
    slot = batch*contacts+contact
    point = proposals[row]
    best,normal = wp.vec3(anchors[slot]),wp.vec3(anchor_normals[slot])
    distance = float(1.e30)
    for offset in range(counts[slot]):
        face = starts[slot]+offset
        candidate = closest_triangle(point,triangles[face,0],triangles[face,1],triangles[face,2])
        if wp.length_sq(candidate-templates[slot]) <= radius*radius+1.e-12:
            d = wp.length_sq(candidate-point)
            if d < distance:
                distance,best,normal = d,candidate,wp.vec3(normals[face])
    output[row] = best
    output_normals[row] = normal


class WarpPatchProjector:
    def __init__(self, surface, radius, device):
        self.surface,self.radius,self.device = surface,radius,device
        self.signature = None
        self.buffers = {}

    def _prepare(self, template, weights):
        signature = (template.tobytes(),(weights>0).tobytes())
        if signature == self.signature:
            return
        batch,contacts = template.shape[:2]
        starts,counts = np.zeros((batch,contacts),np.int32),np.zeros((batch,contacts),np.int32)
        anchors = template.copy();anchor_normals=np.zeros_like(template);anchor_normals[...,2]=1.
        triangles,normals = [],[]
        seen={};total=0
        for b,c in zip(*np.nonzero(weights>0)):
            key=np.asarray(template[b,c],float).tobytes()
            if key not in seen:
                patch=self.surface.patch(template[b,c],self.radius)
                if patch is None:
                    seen[key]=None
                else:
                    seen[key]=(total,len(patch.triangles),patch.anchor,patch.anchor_normal)
                    triangles.append(patch.triangles);normals.append(patch.normals)
                    total+=len(patch.triangles)
            patch=seen[key]
            if patch is not None:
                starts[b,c],counts[b,c],anchors[b,c],anchor_normals[b,c]=patch
        triangles=np.concatenate(triangles) if triangles else np.zeros((1,3,3))
        normals=np.concatenate(normals) if normals else np.zeros((1,3))
        # Stable allocations let the live gather/project/LM/select/scatter
        # CUDA graph survive window changes. Face storage grows geometrically.
        reusable=(hasattr(self,'geometry') and self.geometry[0].shape==(batch*contacts,)
                  and self.geometry[5].shape[0]>=len(triangles))
        capacity=(self.geometry[5].shape[0] if reusable else 1 << (max(len(triangles),1)-1).bit_length())
        padded=np.zeros((capacity,3,3),np.float32);padded[:len(triangles)]=triangles
        padded_normals=np.zeros((capacity,3),np.float32);padded_normals[:len(normals)]=normals
        values=[template.reshape(-1,3),starts.ravel(),counts.ravel(),anchors.reshape(-1,3),
                anchor_normals.reshape(-1,3),padded,padded_normals]
        types=[wp.vec3,int,int,wp.vec3,wp.vec3,wp.vec3,wp.vec3]
        if reusable:
            for target,value in zip(self.geometry,values):target.assign(value)
        else:
            self.geometry=[wp.array(value,dtype=dtype,device=self.device) for value,dtype in zip(values,types)]
        self.movable=counts>0
        self.signature=signature

    def project(self, proposals, template, weights):
        self._prepare(template,weights)
        batch,samples,contacts=proposals.shape[:3]
        size=batch*samples*contacts
        if size not in self.buffers:
            self.buffers[size]=[wp.empty(size,dtype=wp.vec3,device=self.device) for _ in range(3)]
        source,output,normals=self.buffers[size]
        source.assign(np.asarray(proposals,np.float32).reshape(-1,3))
        wp.launch(project_patch,size,inputs=[source,*self.geometry,samples,contacts,self.radius],
                  outputs=[output,normals],device=self.device)
        shape=proposals.shape
        return output.numpy().reshape(shape).astype(float),normals.numpy().reshape(shape).astype(float),self.movable
