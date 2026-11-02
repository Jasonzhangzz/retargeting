"""Adjacent-size contact transport, with bounded motion of the object in the wrist."""
from dataclasses import dataclass
import math
import numpy as np
import torch
from .template_mppi import TemplatePoseObjective
from .template_mppi_reference import pose_matrix

POLICY = 'template_adjacent_solved_grasp_v1'


def densify(primitives, subdivisions=4):
    """Interpolate physical dimensions; retain every original size and endpoint."""
    if isinstance(subdivisions, bool) or not isinstance(subdivisions, int) or subdivisions < 1:
        raise ValueError('subdivisions must be a positive integer')
    rows = sorted(primitives, key=lambda p:p['size_scale'])
    if len({p['size_scale'] for p in rows}) != len(rows):
        raise ValueError('A continuation family requires unique size scales')
    result = []
    for a,b in zip(rows,rows[1:]):
        for j in range(subdivisions):
            u=j/subdivisions
            p=dict(a)
            for key in ('radius_m','height_m','size_scale','radial_scale','axial_scale'):
                p[key]=a[key]*(1-u)+b[key]*u
            p['dimensions_m']=(np.array(a['dimensions_m'])*(1-u)+np.array(b['dimensions_m'])*u).tolist()
            result.append(p)
    if rows:
        result.append(dict(rows[-1]))
    return result


def outward_chains(primitives, canonical):
    """Two sequential dependency chains, starting at the exact standard size."""
    rows=list(primitives)
    if not any(abs(p['size_scale']-1.)<1e-10 for p in rows):
        rows.append(dict(canonical))
    rows.sort(key=lambda p:p['size_scale'])
    pivot=next(i for i,p in enumerate(rows) if abs(p['size_scale']-1.)<1e-10)
    return rows, (list(range(pivot-1,-1,-1)),list(range(pivot+1,len(rows)))),pivot


def pose_change(parent, child):
    a,b=np.linalg.inv(pose_matrix(parent)),np.linalg.inv(pose_matrix(child))
    cosine=np.clip((np.trace(a[:3,:3].T@b[:3,:3])-1)/2,-1.,1.)
    return dict(object_translation_m=float(np.linalg.norm(b[:3,3]-a[:3,3])),
        object_rotation_deg=float(np.degrees(np.arccos(cosine))),
        max_joint_change_deg=float(np.degrees(np.abs(np.asarray(child)[6:]-np.asarray(parent)[6:])).max()))


def rotation_xyz(q):
    """Intrinsic XYZ, matching the three MuJoCo wrist hinge coordinates."""
    x,y,z=q[...,3],q[...,4],q[...,5]
    sx,cx=x.sin(),x.cos();sy,cy=y.sin(),y.cos();sz,cz=z.sin(),z.cos()
    return torch.stack((cy*cz,-cy*sz,sy,sx*sy*cz+cx*sz,cx*cz-sx*sy*sz,-sx*cy,
                        sx*sz-cx*sy*cz,sx*cz+cx*sy*sz,cx*cy),-1).reshape(*q.shape[:-1],3,3)


def inverse_pose(q):
    r=rotation_xyz(q).transpose(-1,-2)
    return -(r@q[...,:3,None])[...,0],r


@dataclass(frozen=True)
class MotionLimits:
    translation_m: float=0.
    rotation_deg: float=0.
    joint_deg: float=8.

    def __post_init__(self):
        if any(isinstance(x,bool) or not math.isfinite(x) or x<0 for x in
               (self.translation_m,self.rotation_deg,self.joint_deg)) or self.joint_deg<=0:
            raise ValueError('Finite nonnegative root limits and positive joint limit required')


class ContinuationObjective(TemplatePoseObjective):
    def __init__(self,*args,parent_q,limits=MotionLimits(),side_slots=None,**kwargs):
        self.parent_q=np.asarray(parent_q,float).copy()
        self.motion_limits=limits
        self.canonical_side_slots=side_slots
        super().__init__(*args,**kwargs)

    def _search_bounds(self,q):
        super()._search_bounds(q)
        p=torch.as_tensor(self.parent_q,device=self.device,dtype=self.qlo.dtype)
        limits=self.motion_limits
        radius=p.new_tensor([limits.translation_m]*3+[math.radians(limits.rotation_deg)]*3+[math.radians(limits.joint_deg)]*20)
        self.qlo=torch.maximum(self.qlo,p-radius)
        self.qhi=torch.minimum(self.qhi,p+radius)
        if bool((self.qlo>self.qhi).any()):
            raise ValueError('Parent pose does not intersect the continuation articulation bounds')

    def configure_object_regions(self,name,primitive):
        super().configure_object_regions(name,primitive)
        if self.canonical_side_slots is not None and self.side_projection is not None:
            self.side_projection.slots=torch.as_tensor(self.canonical_side_slots,device=self.device,dtype=torch.long)

    def extra_ik_residual(self,q):
        p=torch.as_tensor(self.parent_q,device=q.device,dtype=q.dtype)
        t,r=inverse_pose(q);pt,pr=inverse_pose(p)
        limits=self.motion_limits
        # Root trust is relative to the adjacent result, never a 1x finger prior.
        terms=[(t-pt)/max(limits.translation_m,.001),
               (r-pr).flatten()/max(math.sqrt(2)*math.radians(limits.rotation_deg),.01)]
        terms.append(.1*(q[6:]-p[6:])/math.radians(limits.joint_deg))
        return torch.cat(terms)

    def continuity(self,q):
        p=torch.as_tensor(self.parent_q,device=q.device,dtype=q.dtype)
        t,r=inverse_pose(q);pt,pr=inverse_pose(p)
        displacement=torch.linalg.vector_norm(t-pt,dim=-1)
        chord=torch.linalg.matrix_norm(r-pr,dim=(-2,-1))
        limits=self.motion_limits
        allowed_chord=2*math.sqrt(2)*math.sin(math.radians(limits.rotation_deg)/2)
        valid=(displacement<=limits.translation_m+2e-7)&(chord<=allowed_chord+2e-6)
        valid &= (q[... ,6:]-p[6:]).abs().amax(-1)<=math.radians(limits.joint_deg)+2e-6
        penalty=(displacement/max(limits.translation_m,.001)).square()+(
            chord/max(allowed_chord,.01)).square()
        penalty+=.01*((q[...,6:]-p[6:])/math.radians(limits.joint_deg)).square().mean(-1)
        return penalty,valid

    def _with_continuity(self,cost,controls):
        penalty,valid=self.continuity(controls[:,0].to(device=self.device,dtype=torch.float32))
        self.parts['continuity']=penalty
        self.parts['valid'] &= valid
        return torch.where(valid,cost+penalty,torch.inf)

    def evaluate(self,controls):
        return self._with_continuity(super().evaluate(controls),controls)

    def evaluate_initialization(self,controls):
        return self._with_continuity(super().evaluate_initialization(controls),controls)

    def diagnostics(self,q):
        d=super().diagnostics(q)
        d['cost_parts']['continuity']=float(self.parts['continuity'][0])
        d['adjacent_motion']=pose_change(self.parent_q,np.asarray(q.detach().cpu() if torch.is_tensor(q) else q))
        d['adjacent_motion_limits']=vars(self.motion_limits)
        d['adjacent_motion_valid']=bool(self.continuity(torch.as_tensor(q,device=self.device,dtype=torch.float32))[1])
        return d
