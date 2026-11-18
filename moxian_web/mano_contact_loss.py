"""Fixed-material contact geometry and tolerance-scaled training residuals."""
import math
import torch
import torch.nn.functional as F

REVISION='contact_normal_cone_v2'


def vertex_normals(vertices,faces):
    """Area-weighted normals, differentiable through the original triangles."""
    tri=vertices[...,faces,:]
    area=torch.linalg.cross(tri[...,1,:]-tri[...,0,:],tri[...,2,:]-tri[...,0,:],dim=-1)
    normals=torch.zeros_like(vertices)
    for k in range(3):normals=normals.index_add(-2,faces[:,k],area)
    return F.normalize(normals,dim=-1)


def normal_residual(a,b,tolerance_deg=15.,scale_deg=15.):
    """Zero within the antiparallel cone; chord excess in angular units.

    Vector chords avoid acos derivatives at exact alignment. Squaring the
    residual gives a continuously differentiable dead zone at the boundary.
    """
    chord=F.normalize(a,dim=-1)+F.normalize(b,dim=-1)
    length=chord.square().sum(-1,keepdim=True).clamp_min(1e-12).sqrt()
    tolerance=2*math.sin(math.radians(tolerance_deg)/2)
    scale=2*math.sin(math.radians(scale_deg)/2)
    return chord*(1-tolerance/length).clamp_min(0)/scale


def normal_error_deg(a,b):
    """Deviation from opposing outward normals, in degrees (diagnostics only)."""
    dot=(F.normalize(a,dim=-1)*F.normalize(b,dim=-1)).sum(-1)
    return torch.rad2deg(torch.acos((-dot).clamp(-1,1)))


def loss_spec(settings):
    return dict(revision=REVISION,distance_scale_mm=settings.calibration_contact_distance_scale_m*1000,
                normal_tolerance_deg=settings.calibration_contact_normal_tolerance_deg,
                normal_scale_deg=settings.calibration_contact_normal_scale_deg,
                normals='area_weighted_vertex_barycentric_then_fixed_taxel_response_average',
                evidence='confirmed_touch_fresh_bilateral_fixed_taxels',
                normal_weight=settings.calibration_contact_normal_weight,
                position_weight=settings.calibration_contact_position_weight)


def gradient_report(objective,vector):
    """Train-only diagnostic; never feeds gradients to an optimizer."""
    if any(r.get('partition','train')!='train' for r in objective.rows):raise ValueError('接触梯度诊断禁止读取留出数据')
    result={}
    try:
        with torch.enable_grad():
            z=vector.detach().clone().requires_grad_(True)
            objective.residual(z,record=True)
            for name,residual in objective.last_contact_residuals.items():
                grad=torch.autograd.grad(residual.square().sum(),z,retain_graph=True)[0]
                result[name]=dict(shared_gradient_norm=float(grad[:objective.shared_size].norm()),finite=bool(torch.isfinite(grad).all()))
    finally:objective.last_contact_residuals={}
    return dict(source='train_only',scope='full_shared_vector_including_frozen_coordinates; diagnostic_only',terms=result)
