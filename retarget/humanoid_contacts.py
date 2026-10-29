"""Joint hand/object and foot/ground task-wrench contact proximal block.

The same two-body centroidal wrench residual is present in the GPU action
objective. Contact forces are prospective friction-feasible auxiliaries, not
external forces injected into the simulator.
"""
import numpy as np

from retarget.contact_solvers import contact_basis, project_pressures, project_contact_forces
from retarget.humanoid_model import box_mesh


def project(problem, points):
    out = np.array(points,copy=True)
    if problem.surface:
        geom=problem.object_geom
        import mujoco
        rotation=np.empty(9);mujoco.mju_quat2Mat(rotation,problem.model.geom_quat[geom])
        rotation=rotation.reshape(3,3);origin=problem.model.geom_pos[geom]
        local=(out[...,:2,:]-origin)@rotation
        out[...,:2,:]=problem.surface.project(local)@rotation.T+origin
    out[...,2:,2]=problem.ground_height
    return out


def bounded_project(problem, points, anchors, radius):
    out=project(problem,points)
    radius=np.asarray(radius)
    if np.all(radius<=0):return np.broadcast_to(anchors,out.shape).copy()
    if radius.ndim==1:radius=radius[:,None]
    for _ in range(8):
        delta=out-anchors
        out=project(problem,anchors+delta*np.minimum(1.,radius/np.maximum(np.linalg.norm(delta,axis=-1,keepdims=True),1e-12)))
    invalid=(np.linalg.norm(out-anchors,axis=-1,keepdims=True)>radius+1e-6)[...,0]
    return np.where(invalid[...,None],anchors,out)


def desired_wrench(backend,snapshot,ids):
    p=backend.problem
    com=snapshot['com'][ids]
    target=np.broadcast_to(backend.wrench_ref[ids,None],(len(ids),backend.worlds,2,3)).copy()
    target[:,:,0] += backend.model.robot_mass*20.*(backend.com_ref[ids,None]-com)
    target[:,:,0] += backend.model.robot_mass*4.*(backend.reference_qvel[ids,None,:3]-snapshot['qvel'][ids,:,:3])
    actual=snapshot['torso_rotation'][ids]
    reference=backend.rotations_ref[ids,None,p.torso]
    error=.5*np.cross(actual.swapaxes(-1,-2),reference.swapaxes(-1,-2)).sum(axis=-2)
    target[:,:,1] += 20.*error
    return target


def matrices(backend,snapshot,ids,z):
    """z: (time,world,candidate,4,3), matrix: (...,12,32)."""
    p=backend.problem
    shape=z.shape[:-2]
    n,w,c=shape
    com=snapshot['com'][ids]
    points=z.copy()
    normals=np.zeros_like(points);normals[...,2:,2]=-1.  # inward = ground +Z
    object_com=np.zeros((n,w,3))
    object_pos=np.zeros_like(object_com)
    floor_points=np.zeros((n,w,4,3))
    floor_active=np.zeros((n,w,4),dtype=bool)
    if p.surface:
        body=backend.model.object_body
        object_pos=snapshot['object_position'][ids]
        object_rot=snapshot['object_rotation'][ids]
        object_com=object_pos+np.einsum('nwij,j->nwi',object_rot,p.model.body_ipos[body])
        points[...,:2,:]=np.einsum('nwij,nwcsj->nwcsi',object_rot,z[...,:2,:])+object_pos[:,:,None,None]
        # Outward normal in object-body coordinates, transformed to world.
        geom=p.object_geom
        import mujoco
        geom_rot=np.empty(9);mujoco.mju_quat2Mat(geom_rot,p.model.geom_quat[geom]);geom_rot=geom_rot.reshape(3,3)
        local=(z[...,:2,:]-p.model.geom_pos[geom])@geom_rot
        normal=p.surface.normal(local,local)@geom_rot.T
        normals[...,:2,:]=np.einsum('nwij,nwcsj->nwcsi',object_rot,normal)
        if p.model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX:
            vertices,_=box_mesh(p.model.geom_size[geom])
        else:
            vertices=np.asarray(p.surface.mesh.vertices)
        vertices=vertices@geom_rot.T+p.model.geom_pos[geom]
        world=np.einsum('nwij,vj->nwvi',object_rot,vertices)+object_pos[:,:,None]
        order=np.argsort(world[...,2],axis=-1)[...,:4]
        floor_points=np.take_along_axis(world,order[...,None],axis=-2)
        floor_active=floor_points[...,2]<p.ground_height+0.025
        floor_points[...,2]=p.ground_height
    basis=contact_basis(normals)
    tangents=np.array([[1.,1.,0.],[1.,-1.,0.],[1.,0.,1.],[1.,0.,-1.]])
    rays=np.einsum('...sij,rj->...sri',basis,tangents)
    # Preserve the configured friction coefficients of each physical pair.
    mu=np.array([p.contact_friction[role] for role in range(4)])
    rays=basis[...,0][:,:,:, :,None,:]+(rays-basis[...,0][:,:,:, :,None,:])*mu[None,None,None,:,None,None]
    floor_rays=np.array([[1,0,1],[-1,0,1],[0,1,1],[0,-1,1]],float)
    floor_rays[:,:2]*=p.object_floor_friction
    ground=np.broadcast_to(floor_points[:,:,None],(n,w,c,4,3))
    all_points=np.concatenate((points,ground),axis=-2)
    all_rays=np.concatenate((rays,np.broadcast_to(floor_rays,(n,w,c,4,4,3))),axis=-3)
    cap=np.array([backend.model.robot_mass*9.81]*4+[max(backend.model.object_mass*9.81,1.)]*4)
    scaled=all_rays*cap[None,None,None,:,None,None]
    matrix=np.zeros(shape+(12,8,4))
    for role in range(8):
        force=scaled[...,role,:,:]
        if role<4:
            sign=-1. if role<2 else 1.
            matrix[...,0:3,role,:]=np.swapaxes(sign*force,-1,-2)
            torque=np.cross(all_points[...,role,None,:]-com[:,:,None,None,:],sign*force)
            matrix[...,3:6,role,:]=np.swapaxes(torque,-1,-2)
        if role<2 or role>=4:
            matrix[...,6:9,role,:]=np.swapaxes(force,-1,-2)
            torque=np.cross(all_points[...,role,None,:]-object_com[:,:,None,None,:],force)
            matrix[...,9:12,role,:]=np.swapaxes(torque,-1,-2)
    mask=p.mask[backend.reference_indices[ids]]
    active=np.concatenate((np.broadcast_to(mask[:,None],(n,w,4)),floor_active),axis=-1)
    if not p.surface:active[..., :2]=False;active[...,4:]=False
    matrix*=active[:,:,None,None,:,None]
    target=np.zeros((n,w,12))
    target[...,:6]=desired_wrench(backend,snapshot,ids).reshape(n,w,6)
    if p.surface:target[...,6:]=np.broadcast_to(backend.object_wrench_ref[ids,None].reshape(len(ids),1,6),(n,w,6))
    scale=np.array([max(backend.model.robot_mass*9.81,1.)]*3+[max(backend.model.robot_mass*9.81,1.)*.5]*3+
                   [max(backend.model.object_mass*9.81,1.)]*3+[max(backend.model.object_mass*9.81,1.)*.3]*3)
    matrix/=scale[None,None,None,:,None,None]
    target/=scale
    return matrix.reshape(-1,12,32),np.broadcast_to(target[:,:,None],shape+(12,)).reshape(-1,12),scaled,all_points,active


def force_qp(matrix,target,iterations):
    """Same nonnegative capped friction-pyramid pressures as the Wuji QP."""
    batch=matrix.shape[0]
    alpha=np.zeros((batch,32))
    y=alpha.copy();moment=1.
    # A Frobenius bound is a safe Lipschitz constant for every batch member.
    lipschitz=np.maximum(np.sum(matrix**2,axis=(1,2)),1e-8)+1e-6
    caps=np.ones((batch,8));minimum=np.zeros(batch)
    for _ in range(iterations):
        residual=np.einsum('bij,bj->bi',matrix,y)-target
        gradient=np.einsum('bij,bi->bj',matrix,residual)+1e-6*y
        next_=project_pressures((y-gradient/lipschitz[:,None]).reshape(batch,8,4),caps,minimum).reshape(batch,32)
        next_moment=.5*(1+np.sqrt(1+4*moment**2))
        y=next_+(moment-1)/next_moment*(next_-alpha)
        alpha,moment=next_,next_moment
    residual=np.einsum('bij,bj->bi',matrix,alpha)-target
    return alpha,np.sum(residual**2,axis=1)


def update(backend,snapshot,z,dual,rng):
    p,s=backend.problem,backend.settings
    ids=np.unique(np.r_[np.arange(0,backend.dense_steps+1,s.substeps),backend.dense_steps])
    radii=np.array([s.contact_radius]*2+[s.foot_contact_radius]*2)
    x=snapshot['x'][ids]
    anchor=backend.anchors[ids,None]
    prox=(s.rho*(x+dual[ids])+s.anchor_weight*anchor)/(s.rho+s.anchor_weight)
    proposal=np.broadcast_to(prox[:,:,None],(len(ids),backend.worlds,s.contact_samples,4,3)).copy()
    if s.contact_samples>1:
        proposal[:,:,1]=z[ids]
    if s.contact_samples>2:
        noise=rng.normal(size=proposal[:,:,2:].shape)
        noise*=radii[:,None]/np.maximum(np.linalg.norm(noise,axis=-1,keepdims=True),1e-12)
        proposal[:,:,2:]+=noise
    proposal=bounded_project(p,proposal,anchor[:,:,None],radii)
    matrix,target,rays,points,active=matrices(backend,snapshot,ids,proposal)
    # Complete the square: ||A p - desired||² + lambda||A p - actual||².
    # This is the very same realization residual differentiated in action.
    # Keep the original 12-row CUDA pressure solver and add its exact constant.
    lam=np.full(len(ids),s.contact_realization_weight)
    lam[ids==backend.dense_steps]=0.
    lam=np.broadcast_to(lam[:,None,None],proposal.shape[:3]).reshape(-1)
    actual=np.broadcast_to(snapshot['realized_wrench'][ids,:,None],proposal.shape[:3]+(12,)).reshape(-1,12)
    constant=lam/(1.+lam)*np.sum((target-actual)**2,axis=1)
    root=np.sqrt(1.+lam)
    matrix=matrix*root[:,None,None]
    target=(target+lam[:,None]*actual)/root[:,None]
    alpha,wrench=backend.contact_qp.solve(matrix,target,s.contact_qp_iterations)
    wrench+=constant
    shape=proposal.shape[:3]
    wrench=wrench.reshape(shape)
    mask=p.mask[backend.reference_indices[ids]][:,None,None,:,None]
    anchor_cost=s.anchor_weight*np.sum(mask*((proposal-anchor[:,:,None])/s.contact_scale)**2,axis=(-1,-2))
    consensus=s.rho*np.sum(mask*((x[:,:,None]-proposal+dual[ids,:,None])/s.contact_scale)**2,axis=(-1,-2))
    robust_wrench=2.*wrench/(np.sqrt(1.+wrench/s.wrench_robust_scale**2)+1.) if p.surface else wrench
    merit=p.weights.contact_wrench*robust_wrench+anchor_cost+consensus
    selected=np.argmin(merit,axis=-1)
    row,col=np.indices(selected.shape)
    best=proposal[row,col,selected]
    pressures=alpha.reshape(shape+(8,4))[row,col,selected]
    selected_rays=rays[row,col,selected]
    forces=np.einsum('nwsri,nwsr->nwsi',selected_rays,pressures)
    selected_points=points[row,col,selected]
    floor_force=forces[:,:,4:].sum(axis=-2)
    floor_torque=np.cross(selected_points[:,:,4:],forces[:,:,4:]).sum(axis=-2)
    full=np.arange(backend.dense_steps+1)
    def expand(value):
        flat=value.reshape(len(ids),-1)
        return np.stack([np.interp(full,ids,flat[:,j]) for j in range(flat.shape[1])],axis=1).reshape((len(full),)+value.shape[1:])
    znew=bounded_project(p,expand(best),backend.anchors[:,None],radii)
    znew=np.where(p.mask[backend.reference_indices,:,None][:,None],znew,backend.anchors[:,None])
    force=expand(forces[:,:,:4])
    force*=p.mask[backend.reference_indices][:,None,:,None]
    # Re-project interpolated forces into the cone at the interpolated contact;
    # otherwise a box-face transition could invalidate endpoint-feasible forces.
    normals=np.zeros_like(force);normals[:,:,2:,2]=-1.
    if p.surface:
        import mujoco
        geom=p.object_geom
        rotation=np.empty(9);mujoco.mju_quat2Mat(rotation,p.model.geom_quat[geom]);rotation=rotation.reshape(3,3)
        local=(znew[:,:,:2]-p.model.geom_pos[geom])@rotation
        normal=p.surface.normal(local,local)@rotation.T
        normals[:,:,:2]=np.einsum('nwij,nwsj->nwsi',snapshot['object_rotation'],normal)
    basis=contact_basis(normals)
    coordinates=np.einsum('nwsij,nwsi->nwsj',basis,force)
    mu=np.broadcast_to(np.array(p.contact_friction)[None,None,:,None],coordinates.shape[:-1]+(2,))
    active_full=np.broadcast_to(p.mask[backend.reference_indices][:,None],coordinates.shape[:-1])
    shape_full=coordinates.shape
    coordinates=project_contact_forces(coordinates.reshape(-1,12),mu.reshape(-1,4,2),active_full.reshape(-1,4),backend.model.robot_mass*9.81).reshape(shape_full)
    force=np.einsum('nwsij,nwsj->nwsi',basis,coordinates)
    return znew,force,expand(floor_force),expand(floor_torque),dict(
        method='joint_robot_object_centroidal_wrench_qp',force_qp_backend='warp_cuda',samples=s.contact_samples,
        wrench_penalty='pseudo_huber' if p.surface else 'quadratic',wrench_robust_scale=s.wrench_robust_scale,
        wrench_before=float(np.mean(robust_wrench[:,:,1 if s.contact_samples>1 else 0])),
        wrench_after=float(np.mean(robust_wrench[row,col,selected])),
        raw_wrench_before=float(np.mean(wrench[:,:,1 if s.contact_samples>1 else 0])),
        raw_wrench_after=float(np.mean(wrench[row,col,selected])),
        proximal_merit_before=float(np.mean(merit[:,:,1 if s.contact_samples>1 else 0])),
        proximal_merit_after=float(np.mean(merit[row,col,selected])),
        realization_weight=s.contact_realization_weight,actual_mjwarp_wrench_consistency=True,
        forces_injected_into_simulator=False)
