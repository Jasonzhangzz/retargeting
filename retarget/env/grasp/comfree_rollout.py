"""Physical settings and GPU mesh mass properties for the current spring rollout."""
from dataclasses import dataclass
import math
import torch

def matrix_quaternion(matrix):
    """Same four-candidate conversion, using fixed-shape gather for CUDA graphs."""
    a,b,c,d,e,f,g,h,i=matrix.flatten(-2).unbind(-1)
    q=torch.stack((1+a+e+i,1+a-e-i,1-a+e-i,1-a-e+i),-1).clamp_min(0).sqrt()
    candidates=torch.stack((torch.stack((q[...,0]**2,h-f,c-g,d-b),-1),
        torch.stack((h-f,q[...,1]**2,d+b,c+g),-1),
        torch.stack((c-g,d+b,q[...,2]**2,f+h),-1),
        torch.stack((d-b,c+g,f+h,q[...,3]**2),-1)),-2)/(2*q[...,None].clamp_min(.1))
    index=q.argmax(-1)[...,None,None].expand(*q.shape[:-1],1,4)
    result=candidates.gather(-2,index).squeeze(-2)
    return torch.where(result[...,:1]<0,-result,result)


from retarget.solver.cem import RolloutSettings


def mesh_mass_properties(vertices, faces, mass):
    """Exact uniform-density closed triangle-mesh COM and inertia, on device."""
    tri = vertices.to(torch.float64)[faces.long()]
    vol = (tri[:, 0] * torch.linalg.cross(tri[:, 1], tri[:, 2])).sum(-1) / 6
    volume = vol.sum()
    center = (vol[:, None] * tri.sum(1) / 4).sum(0) / volume
    sums = tri.sum(1)
    second = (vol[:, None, None] * (
        torch.einsum('fvi,fvj->fij', tri, tri) + sums[:, :, None] * sums[:, None, :]
    ) / 20).sum(0) - volume * center[:, None] * center[None, :]
    second = second * (mass / volume)
    inertia = torch.eye(3, device=vertices.device, dtype=torch.float64) * second.trace() - second
    if not bool((volume > 0) & torch.isfinite(inertia).all() & (torch.linalg.eigvalsh(inertia) > 0).all()):
        raise ValueError('ComFree requires a closed outward mesh with positive inertia')
    return center.float(), inertia.float()
