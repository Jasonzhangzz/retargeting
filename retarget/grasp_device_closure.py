"""Capture-safe geometric closure certificate; all branching stays on CUDA."""


def closure_tensors(points, outward, active, *, friction, radius, edges=16, tolerance=1e-7):
    import torch
    p, outward = points.double(), outward.double()
    n = -torch.nn.functional.normalize(outward, dim=-1)
    axes = torch.eye(3, device=p.device, dtype=p.dtype)[n.abs().argmin(-1)]
    t1 = torch.nn.functional.normalize(torch.linalg.cross(n, axes), dim=-1)
    t2 = torch.linalg.cross(n, t1)
    angle = torch.arange(edges, device=p.device, dtype=p.dtype)*(2*torch.pi/edges)
    rays = n[..., None]+friction*(t1[..., None]*angle.cos()+t2[..., None]*angle.sin())
    rays = rays.transpose(-1, -2).reshape(-1, 3)
    arms = p[:, None, :].expand(-1, edges, -1).reshape(-1, 3)
    mask = active.repeat_interleave(edges).double()
    matrix = torch.cat((rays, torch.linalg.cross(arms, rays)/radius), dim=1).T*mask[None]
    # Complete-pivot modified Gram-Schmidt, with reorthogonalization. Unlike
    # matrix_rank/SVD this uses no cuSOLVER status readback to the host.
    residual = matrix.clone()
    rank = torch.zeros((), dtype=torch.int64, device=p.device)
    for _ in range(6):
        lengths = residual.square().sum(0)
        pivot = lengths.argmax().reshape(1, 1).expand(6, 1)
        basis = residual.gather(1, pivot)
        norm = basis.square().sum().sqrt()
        valid = norm > 1.e-8
        basis = basis/norm.clamp(min=1.e-30)*valid
        for _ in range(2):
            residual = residual-basis@(basis.T@residual)
        rank = rank+valid.long()
    k = matrix.shape[1]
    a = torch.cat((matrix, torch.ones((1, k), device=p.device, dtype=p.dtype)), dim=0)
    a = torch.cat((a, (a*mask[None]).sum(1, keepdim=True)), dim=1)
    rhs = torch.zeros(7, device=p.device, dtype=p.dtype)
    rhs[-1:].fill_(1.)
    cost = torch.zeros(k+1, device=p.device, dtype=p.dtype)
    cost[-1:].fill_(-1.)
    x = torch.full_like(cost, 1./(2*k))
    slack = torch.ones_like(x)
    dual = torch.zeros_like(rhs)
    eye = torch.eye(7, device=p.device, dtype=p.dtype)
    converged = torch.zeros((), device=p.device, dtype=torch.bool)
    failed = torch.zeros_like(converged)
    iterations = torch.zeros((), device=p.device, dtype=torch.int64)

    def direction(rc, rp, rd):
        ratio = x/slack
        system = (a*ratio)@a.T
        residual = -rp+a@(rc/slack-ratio*rd)
        dy, info = torch.linalg.solve_ex(system+1.e-13*eye, residual, check_errors=False)
        dx = -rc/slack+ratio*(rd+a.T@dy)
        ds = -rd-a.T@dy
        good = (info == 0)&torch.isfinite(dx).all()&torch.isfinite(dy).all()&torch.isfinite(ds).all()
        return dx, dy, ds, good

    def step(value, delta, fraction=1.):
        limit = torch.where(delta < 0, -value/delta, torch.full_like(delta, torch.inf)).min()
        return torch.minimum(torch.ones_like(limit), fraction*limit)

    for iteration in range(100):
        rp, rd = a@x-rhs, a.T@dual+slack-cost
        mu = (x*slack).mean()
        reached = (rp.abs().max() < 1.e-9)&(rd.abs().max() < 1.e-9)&(mu < 1.e-9)
        newly = reached&~converged&~failed
        iterations = torch.where(newly, iteration+1, iterations)
        converged = converged|(reached&~failed)
        dx, dy, ds, good1 = direction(x*slack, rp, rd)
        ap, ad = step(x, dx), step(slack, ds)
        affine_mu = ((x+ap*dx)*(slack+ad*ds)).mean()
        sigma = (affine_mu/mu).clamp(0., 1.)**3
        dx, dy, ds, good2 = direction(x*slack+dx*ds-sigma*mu, rp, rd)
        ap, ad = step(x, dx, .995), step(slack, ds, .995)
        enabled = ~converged&~failed&good1&good2
        x = torch.where(enabled, x+ap*dx, x)
        dual = torch.where(enabled, dual+ad*dy, dual)
        slack = torch.where(enabled, slack+ad*ds, slack)
        failed = failed|(~converged&~(good1&good2))
    alpha = (x[:-1]+x[-1]*mask)*mask
    alpha = torch.nan_to_num(alpha/alpha.sum().clamp(min=1.e-30), nan=0., posinf=0., neginf=0.)
    margin = torch.where(mask.bool(), alpha, torch.inf).min()
    margin = torch.where(active.any(), margin, torch.zeros_like(margin))
    equilibrium = torch.linalg.vector_norm(matrix@alpha)
    closed = converged&(rank == 6)&(margin > tolerance)&(equilibrium < 1.e-7)&active.any()
    return dict(force_closure=closed, rank=rank, positive_margin=margin,
                normalized_margin=mask.sum()*margin, equilibrium_residual=equilibrium,
                coefficients=alpha, contacts=active.sum(), converged=converged,
                iterations=torch.where(converged, iterations, torch.full_like(iterations, 100)))


def export_closure(tensors, active, edges):
    """Only the completed certificate crosses the device/host boundary."""
    output = {key: value.detach().cpu().item() for key, value in tensors.items() if key != 'coefficients'}
    mask = active.detach().cpu().repeat_interleave(edges).bool()
    output['coefficients'] = tensors['coefficients'].detach().cpu()[mask].tolist()
    if not output['contacts']:
        output['equilibrium_residual'] = None
    output.update(backend='cuda_graph_primal_dual_LP', certificate_kind='geometric_positive_span',
                  force_units='normalized', actuator_feasibility_checked=False)
    return output
