"""Normal-pressure resultant and candidate-frame Coulomb force support.

Glove ADC supplies relative scalar pressure, not shear or Newtons. The vector
estimate assumes each taxel pushes along its registered outward surface normal.
Force fitting is an analytic latent-force solve, not a robot force controller.
"""
import math
import numpy as np
import torch


def estimate_normal_resultant(normals, response):
    normals=np.asarray(normals,float);response=np.asarray(response,float)
    if normals.shape!=(len(response),3) or response.ndim!=1 or not np.isfinite(normals).all() or not np.isfinite(response).all() or np.any(response<0):
        raise ValueError('Finite nonnegative pressures and matching normals required')
    length=np.linalg.norm(normals,axis=-1)
    if np.any(length<1e-10):raise ValueError('Nonzero taxel normals required')
    normals=normals/length[:,None]
    resultant=(normals*response[:,None]).sum(0);load=float(response.sum());magnitude=float(np.linalg.norm(resultant))
    valid=bool(load>1e-12 and magnitude>max(1e-12,load*1e-8))
    return dict(model='normal_pressure_resultant_v1',units='relative_response_sum',
                vector_relative=resultant.tolist(),magnitude_relative=magnitude,
                scalar_normal_load_relative=load,direction=(resultant/magnitude).tolist() if valid else None,
                coherence=min(1.,magnitude/load) if load>1e-12 else 0.,direction_valid=valid,
                shear_measured=False,force_calibrated=False)


def fit_coulomb_force(desired, normal, mu=1., normal_weight=0., normal_target=None):
    """Solve min ||f-f_des||² + w*(n.f-N_des)² inside a Coulomb cone.

    Fn>=0, ||Ft||<=mu*Fn. Rotation reduces this convex problem to a weighted
    2-D cone projection; no sampling or pinch/contact-pair labels are used.
    The normal target defaults to the desired resultant magnitude, a local
    normal-push assumption, not another independent measurement.
    """
    if not math.isfinite(mu) or mu<0 or not math.isfinite(normal_weight) or normal_weight<0:
        raise ValueError('Finite nonnegative friction and normal-force weights required')
    normal=torch.nn.functional.normalize(normal,dim=-1)
    desired=torch.broadcast_to(desired,normal.shape)
    a=(desired*normal).sum(-1);tangent=desired-a[...,None]*normal;b=tangent.norm(dim=-1)
    target=desired.norm(dim=-1) if normal_target is None else torch.as_tensor(normal_target,device=a.device,dtype=a.dtype).expand_as(a)
    h=1+normal_weight;centre=(a+normal_weight*target)/h
    inside=(centre>=0)&(b<=mu*centre)
    boundary=((h*centre+mu*b)/(h+mu*mu)).clamp_min(0)
    fn=torch.where(inside,centre,boundary);ft=torch.where(inside,b,mu*boundary)
    force=fn[...,None]*normal+(ft/b.clamp_min(1e-20))[...,None]*tangent
    return dict(force=force,normal_force=fn,tangential_force=ft,
                vector_error=(force-desired).square().sum(-1),
                normal_error=normal_weight*(fn-target).square(),
                desired_normal=a,desired_tangent=b,
                desired_in_cone=(a>=0)&(b<=mu*a+1e-7),
                projected_in_cone=(fn>=0)&(ft<=mu*fn+1e-7))
