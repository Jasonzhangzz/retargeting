"""Conservative anatomical joint projection for the local MANO viewer.

These limits are model priors, not subject-specific range-of-motion measurements.
Non-thumb interphalangeal joints are hinges; MCP and thumb bases can spread.
"""
from __future__ import annotations
import warnings
import numpy as np
from scipy.spatial.transform import Rotation

GEOMETRY_REVISION='fixed-dimensions-v7'

class AnatomicalJoints:
    def __init__(self,joints,vertices,chains,tips,palm_normal,pose_mean=None):
        palm_normal=np.array(palm_normal,copy=True)
        # The legacy helper's cross-product normal is dorsal for this left
        # weight file. Orient flexion using the learned relaxed-hand mean,
        # rather than assuming a global axis or mirroring the right hand.
        if pose_mean is not None:
            votes=[]
            for chain in chains[1:]:
                j=chain[1];d=joints[chain[2]]-joints[j];d/=np.linalg.norm(d)
                axis=np.cross(d,palm_normal);axis/=np.linalg.norm(axis)
                votes.append(np.asarray(pose_mean).reshape(15,3)[j-1]@axis)
            if np.median(votes)<0:palm_normal=-palm_normal
        self.palmar_normal=palm_normal
        self.basis=np.zeros((15,3,3));self.lower=np.zeros((15,3));self.upper=np.zeros((15,3))
        for f,chain in enumerate(chains):
            for k,j in enumerate(chain):
                end=joints[chain[k+1]] if k<2 else vertices[tips[f]]
                twist=end-joints[j];twist/=np.linalg.norm(twist)
                flex=np.cross(twist,palm_normal);flex/=np.linalg.norm(flex)
                if f==0:
                    # Thumb opposition is oblique to the palm. Use the
                    # incoming link and left-thumb reference direction, as
                    # in manotorch AxisAdaptiveLayer's canonical construction:
                    # https://github.com/lixiny/manotorch/blob/master/manotorch/axislayer.py
                    parent=chain[k-1] if k else 0
                    twist=joints[j]-joints[parent];twist/=np.linalg.norm(twist)
                    flex=np.cross(-twist,[-1.,1.,1.]);flex/=np.linalg.norm(flex)
                spread=np.cross(twist,flex)
                self.basis[j-1]=np.stack((flex,spread,twist),axis=1)
                if f==0:
                    bounds=((-65,85),(-65,65),(-45,45)) if k==0 else (((-10,90),(-15,15),(-10,10)) if k==1 else ((-5,95),(0,0),(0,0)))
                else:
                    bounds=((-12,100),(-28,28),(-18,18)) if k==0 else (((0,115),(0,0),(0,0)) if k==1 else ((-5,90),(0,0),(0,0)))
                self.lower[j-1]=np.deg2rad([b[0] for b in bounds]);self.upper[j-1]=np.deg2rad([b[1] for b in bounds])

    def angles(self,pose):
        pose=np.asarray(pose);rot=Rotation.from_rotvec(pose.reshape(-1,3)).as_matrix().reshape(*pose.shape[:-1],3,3)
        local=self.basis.swapaxes(-2,-1)@rot@self.basis
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',UserWarning)
            angles=Rotation.from_matrix(local.reshape(-1,3,3)).as_euler('XYZ')
        return angles.reshape(pose.shape)

    def project(self,pose):
        angles=np.clip(self.angles(pose),self.lower,self.upper)
        local=Rotation.from_euler('XYZ',angles.reshape(-1,3)).as_matrix().reshape(*angles.shape[:-1],3,3)
        rotations=self.basis@local@self.basis.swapaxes(-2,-1)
        return Rotation.from_matrix(rotations.reshape(-1,3,3)).as_rotvec().reshape(angles.shape).astype(np.float32)

    def fit_local_direction(self,index,rest_direction,target_direction):
        """Fit a hinge in its actual parent's frame, avoiding free-axis seeds."""
        if np.all(self.lower[index,1:]==self.upper[index,1:]):
            axis=self.basis[index,:,0]
            a=rest_direction-axis*(rest_direction@axis);b=target_direction-axis*(target_direction@axis)
            angle=np.arctan2(axis@np.cross(a,b),a@b)
            angle=np.clip(angle,self.lower[index,0],self.upper[index,0])
            return Rotation.from_rotvec(axis*angle).as_matrix()
        return None

    def project_local_rotation(self,index,rotation):
        b=self.basis[index]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',UserWarning)
            a=Rotation.from_matrix(b.T@rotation@b).as_euler('XYZ')
        return b@Rotation.from_euler('XYZ',np.clip(a,self.lower[index],self.upper[index])).as_matrix()@b.T


def closed_surface_parts(vertices,faces,face_groups,nparts=6):
    """Cap partition boundary loops for an inside test, preserving original faces.

    Caps participate only in volume classification, never contact-point queries.
    """
    import trimesh
    result=[]
    for part in range(nparts):
        tris=np.asarray(faces)[np.asarray(face_groups)==part].copy()
        directed=np.concatenate((tris[:,[0,1]],tris[:,[1,2]],tris[:,[2,0]]))
        keys=np.sort(directed,axis=1);_,inverse,counts=np.unique(keys,axis=0,return_inverse=True,return_counts=True)
        edges=directed[counts[inverse]==1];following={}
        for a,b in edges:following.setdefault(int(a),[]).append(int(b))
        loops=[];caps=[]
        while following:
            current=next(iter(following));path=[];seen={}
            while current not in seen:
                if current not in following:raise ValueError('解剖分区边界未闭合')
                seen[current]=len(path);path.append(current);current=following[current][0]
            loop=path[seen[current]:]
            for a,b in zip(loop,loop[1:]+loop[:1]):
                following[a].remove(b)
                if not following[a]:del following[a]
                caps.append([b,a,len(vertices)+len(loops)])
            loops.append(np.array(loop,dtype=np.int32))
        closed=np.concatenate((tris,np.array(caps,dtype=np.int32).reshape(-1,3)))
        points=np.concatenate((vertices,np.array([vertices[loop].mean(0) for loop in loops]).reshape(-1,3)))
        mesh=trimesh.Trimesh(points,closed,process=False)
        if not mesh.is_watertight or not mesh.is_winding_consistent:raise ValueError('解剖分区无法建立可靠闭合体')
        if mesh.volume<0:closed=closed[:,::-1].copy()
        result.append((points,closed,loops))
    return result
