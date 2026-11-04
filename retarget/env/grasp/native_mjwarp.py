"""GPU operators for the native rigid-hand GraspADMM, with unchanged costs.

This module uses the native mjwarp constraint solver. It does not use the
ComFree replacement environment or the quasi-static MANO hand adapter.
"""
from contextlib import contextmanager
import numpy as np
import torch
import warp as wp
import mujoco
import mujoco_warp as mjw
from mujoco_warp._src import support


class NativePhysics:
    def __init__(self, cpu, worlds, *, device='cuda:0', qpos_to_ctrl=None,
                 hand_qpos=27, hand_bodies=None, object_geom_mask=None,
                 nconmax=512, njmax=2048,frozen_mesh_arrays=None):
        self.cpu=cpu;self.device=torch.device(device);self.worlds=worlds;self.hand_qpos=hand_qpos
        self.frozen_mesh_arrays=frozen_mesh_arrays
        wp.init()
        self.stream=wp.stream_from_torch(torch.cuda.current_stream(self.device))
        with wp.ScopedDevice(str(self.device)),wp.ScopedStream(self.stream):
            self.model=mjw.put_model(cpu)
            if frozen_mesh_arrays is not None:
                # MuJoCo versions can compile a different polygon layout. Move
                # the complete original mesh buffer family together, never
                # combine old indices with newly compiled polygon buffers.
                for name,array in frozen_mesh_arrays.items():
                    if not name.startswith('mesh_') or not hasattr(self.model,name):continue
                    original=getattr(self.model,name)
                    if not isinstance(original,wp.array):continue
                    setattr(self.model,name,wp.array(array,dtype=original.dtype,device=str(self.device)))
                self.model.nmaxpolygon=int(np.append(frozen_mesh_arrays['mesh_polyvertnum'],0).max())
                self.model.nmaxmeshdeg=int(np.append(frozen_mesh_arrays['mesh_polymapnum'],0).max())
            self.data=mjw.make_data(cpu,nworld=worlds,nconmax=nconmax,njmax=njmax)
        # put_model normally relaxes tolerances for float32. Preserve the original values explicitly.
        for key in ('tolerance','ls_tolerance'):
            value=getattr(self.model.opt,key)
            if isinstance(value,wp.array):
                with self.scope():value.fill_(float(getattr(cpu.opt,key)))
            else:setattr(self.model.opt,key,float(getattr(cpu.opt,key)))
        self.qpos_to_ctrl=torch.as_tensor(qpos_to_ctrl,device=self.device,dtype=torch.float32) if qpos_to_ctrl is not None else None
        if qpos_to_ctrl is not None:
            matrix=np.asarray(qpos_to_ctrl);gains=np.ones(cpu.nv,np.float32)
            for i in range(matrix.shape[1]):
                selected=np.flatnonzero(matrix[:,i]!=0)
                if len(selected):
                    a=int(selected[0]);gains[i]=cpu.actuator_gainprm[a,0]*matrix[a].sum()
            self.squeeze_gains=torch.as_tensor(gains,device=self.device)
        self.hand_bodies=torch.as_tensor(hand_bodies,device=self.device,dtype=torch.long) if hand_bodies is not None else None
        self.object_geom_mask=torch.as_tensor(object_geom_mask,device=self.device,dtype=torch.bool) if object_geom_mask is not None else None
        self.geom_bodies=torch.as_tensor(cpu.geom_bodyid.copy(),device=self.device,dtype=torch.long)
        # Explicit pairs keep their own margin/gap instead of geom combination.
        pair_margin=np.full((cpu.ngeom,cpu.ngeom),np.nan,np.float32)
        pair_gap=np.full_like(pair_margin,np.nan)
        for i in range(cpu.npair):
            g1,g2=cpu.pair_geom1[i],cpu.pair_geom2[i]
            pair_margin[g1,g2]=pair_margin[g2,g1]=cpu.pair_margin[i]
            pair_gap[g1,g2]=pair_gap[g2,g1]=cpu.pair_gap[i]
        self.pair_margin=torch.as_tensor(pair_margin,device=self.device)
        self.pair_gap=torch.as_tensor(pair_gap,device=self.device)
        limited=np.flatnonzero(cpu.jnt_limited)
        self.limit_adr=torch.as_tensor(cpu.jnt_qposadr[limited],device=self.device,dtype=torch.long)
        self.limit_ranges=torch.as_tensor(cpu.jnt_range[limited],device=self.device,dtype=torch.float32)
        with self.scope():self.ft=wp.zeros((worlds,cpu.nbody),dtype=wp.spatial_vector,device=str(self.device))
        self.graphs={}

    @contextmanager
    def scope(self):
        with wp.ScopedDevice(str(self.device)),wp.ScopedStream(self.stream):yield

    def tensor(self,name):return wp.to_torch(getattr(self.data,name))

    def forward(self):
        with self.scope():mjw.forward(self.model,self.data)

    def set_ctrl(self,q,skip_mocap=False):
        if not skip_mocap and self.cpu.nmocap:
            self.tensor('mocap_pos')[:,0].copy_(q[:,:3]);self.tensor('mocap_quat')[:,0].copy_(q[:,3:7])
        if self.qpos_to_ctrl is None:raise ValueError('Original hand qpos-to-control matrix is required')
        self.tensor('ctrl').copy_(q[:,7:]@self.qpos_to_ctrl[:,6:].T)

    def reset(self,q):
        if q.shape!=(self.worlds,self.cpu.nq):raise ValueError('Expected complete native model qpos')
        self.tensor('qpos').copy_(q)
        for name in ('qvel','qacc_warmstart','qfrc_applied','xfrc_applied','act','time'):
            self.tensor(name).zero_()
        self.tensor('overflow').zero_()
        self.set_ctrl(q[:,:self.hand_qpos]);self.forward()

    def snapshot(self):
        names=('qpos','qvel','qacc_warmstart','qfrc_applied','xfrc_applied','act','ctrl','mocap_pos','mocap_quat','time',
               'xpos','xquat','xmat','xipos','ximat','subtree_com','cdof')
        return {name:self.tensor(name).clone() for name in names}

    def restore(self,state,mask=None):
        for name,value in state.items():
            target=self.tensor(name)
            if mask is None:target.copy_(value)
            else:target.copy_(torch.where(mask.reshape(-1,*([1]*(value.ndim-1))),value,target))

    def step(self,count=10):
        with self.scope():
            for _ in range(count):mjw.step(self.model,self.data)

    def prepare_graph(self,count=10):
        state=self.snapshot()
        self.step(1);self.restore(state);self.forward()
        # Capture on a Warp-owned stream; CUDA's legacy Torch default stream
        # cannot begin capture. Explicit dependencies preserve the same state.
        capture_stream=wp.Stream(device=str(self.device))
        capture_stream.wait_stream(self.stream)
        with wp.ScopedDevice(str(self.device)),wp.ScopedStream(capture_stream),wp.ScopedCapture(stream=capture_stream) as capture:
            for _ in range(count):mjw.step(self.model,self.data)
        self.graphs[count]=capture.graph
        self.stream.wait_stream(capture_stream)
        self.restore(state);self.forward()

    def advance(self,count=10,mask=None):
        state=self.snapshot() if mask is not None else None
        if count in self.graphs:
            with self.scope():wp.capture_launch(self.graphs[count])
        else:self.step(count)
        if mask is not None:self.restore(state,~mask)

    def positions(self,bodies,local):
        r=self.tensor('xmat')[:,bodies].reshape(self.worlds,-1,3,3)
        return (r@local[...,:3,None]).squeeze(-1)+self.tensor('xpos')[:,bodies]

    def local_points(self,bodies,world):
        r=self.tensor('xmat')[:,bodies].reshape(self.worlds,-1,3,3)
        return (r.transpose(-1,-2)@(world-self.tensor('xpos')[:,bodies])[...,None]).squeeze(-1)

    def contacts(self,bodies,local):
        r=self.tensor('xmat')[:,bodies].reshape(self.worlds,-1,3,3)
        return torch.cat((self.positions(bodies,local),(r@local[...,3:,None]).squeeze(-1)),-1)

    def apply_point_forces(self,bodies,points,force):
        torque=torch.linalg.cross(points-self.tensor('xipos')[:,bodies],force,dim=-1)
        force_torque=torch.cat((force,torque),-1)
        ft=wp.to_torch(self.ft);ft.zero_()
        ft.scatter_add_(1,bodies[None,:,None].expand(self.worlds,-1,6),force_torque)
        self.tensor('qfrc_applied').zero_();self.tensor('xfrc_applied').zero_()
        with self.scope():support.apply_ft(self.model,self.data,self.ft,self.data.qfrc_applied,False)

    def apply_forces(self,bodies,local,targets,*,normal=None,mask=None):
        previous=self.snapshot() if mask is not None else None
        points=self.positions(bodies,local);delta=targets-points
        if normal is None:force=500.*delta
        else:
            perpendicular=(delta*normal).sum(-1,keepdim=True)*normal
            force=500.*perpendicular+100.*(delta-perpendicular)
        if mask is not None:force=force*mask[:,None,None]
        self.apply_point_forces(bodies,points,force)
        self.set_ctrl(self.tensor('qpos')[:,:self.hand_qpos],skip_mocap=True)
        self.tensor('qvel').zero_()
        if previous is not None:self.restore(previous,~mask)
        return delta.norm(dim=-1)

    def keep_stable(self,mask=None):
        previous=self.snapshot() if mask is not None else None
        self.tensor('qfrc_applied').zero_();self.tensor('xfrc_applied').zero_()
        self.set_ctrl(self.tensor('qpos')[:,:self.hand_qpos],skip_mocap=True);self.tensor('qvel').zero_()
        if previous is not None:self.restore(previous,~mask)

    def set_margin(self,margin):
        if self.object_geom_mask is None:raise ValueError('Native object geom ownership is required')
        values=wp.to_torch(self.model.geom_margin)
        values[...,self.object_geom_mask]=margin

    def projected_closure(self,solver,points,outward,signed_distance,hand_geom,object_geom):
        """Margin-aware diagnostic of projected slots, using current model geoms.

        Geom IDs must identify the actual surfaces used by the slot projection.
        Calling the solver without this method still gives all-slot hypothetical
        closure. Neither result replaces native filter QP or dynamic GSR.
        """
        shape=points.shape[:2]
        if shape[0]!=self.worlds:raise ValueError('Expected one slot row per physics world')
        h=torch.as_tensor(hand_geom,device=self.device,dtype=torch.long).expand(shape)
        o=torch.as_tensor(object_geom,device=self.device,dtype=torch.long).expand(shape)
        valid=(h>=0)&(h<self.cpu.ngeom)&(o>=0)&(o<self.cpu.ngeom)
        h=h.clamp(0,self.cpu.ngeom-1);o=o.clamp(0,self.cpu.ngeom-1)
        if self.object_geom_mask is not None:valid&=self.object_geom_mask[o]&~self.object_geom_mask[h]
        def gather(name,ids):
            values=wp.to_torch(getattr(self.model,name))
            return values.reshape(-1,self.cpu.ngeom).expand(self.worlds,-1).gather(1,ids)
        hm,om=gather('geom_margin',h),gather('geom_margin',o)
        hg,og=gather('geom_gap',h),gather('geom_gap',o)
        pm,pg=self.pair_margin[h,o],self.pair_gap[h,o]
        margin=torch.where(torch.isfinite(pm),pm,torch.maximum(hm,om))
        gap=torch.where(torch.isfinite(pg),pg,torch.maximum(hg,og))
        distance=torch.where(valid,signed_distance,torch.nan)
        return solver.solve_with_geom_margins(points,outward,signed_distance=distance,
            hand_margin=hm,object_margin=om,pair_margin=margin,pair_gap=gap)

    def overflow(self):
        if not hasattr(self.data,'overflow'):raise RuntimeError('mjwarp overflow flag is required')
        return self.tensor('overflow')!=0

    def native_contacts(self,hand_body_count,margin=None):
        if margin is not None:
            previous=wp.to_torch(self.model.geom_margin).clone()
            self.set_margin(margin);self.forward()
        contact=self.data.contact
        geom=wp.to_torch(contact.geom).long();world=wp.to_torch(contact.worldid).long()
        valid=(torch.arange(len(world),device=self.device)<self.tensor('nacon')[0])&(geom>=0).all(-1)
        body=self.geom_bodies[geom.clamp_min(0)]
        h1=(body[:,0]>0)&(body[:,0]<=hand_body_count);h2=(body[:,1]>0)&(body[:,1]<=hand_body_count)
        object_geoms=self.object_geom_mask
        if object_geoms is None:raise ValueError('Explicit object geom ownership is required after model serialization')
        o1=object_geoms[geom[:,0].clamp_min(0)];o2=object_geoms[geom[:,1].clamp_min(0)]
        ho=valid&((h1&o2)|(h2&o1))
        hh=valid&h1&h2
        normal=wp.to_torch(contact.frame).reshape(-1,3,3)[:,0]
        normal=normal*torch.where(o2,1.,-1.)[:,None]
        distance=wp.to_torch(contact.dist)
        pos=wp.to_torch(contact.pos)-distance[:,None]*normal
        hand_body=torch.where(h1,body[:,0],body[:,1])
        if margin is not None:wp.to_torch(self.model.geom_margin).copy_(previous)
        return dict(ho=ho,hh=hh,world=world,body=hand_body,normal=normal,position=pos,distance=distance)

    def collision_pass(self,packet,ho_threshold,hh_threshold):
        fail=torch.zeros(self.worlds,device=self.device,dtype=torch.int32)
        bad=(packet['ho']&(packet['distance']<ho_threshold))|(packet['hh']&(packet['distance']<hh_threshold))
        fail.scatter_add_(0,packet['world'].clamp(0,self.worlds-1),bad.to(torch.int32))
        return fail==0

    def limits_pass(self,threshold=0.):
        ranges=self.limit_ranges;q=self.tensor('qpos')[:,self.limit_adr]
        return ((q>=ranges[:,0]+threshold)&(q<=ranges[:,1]-threshold)).all(-1)


@wp.kernel(enable_backward=False)
def nearest_mesh(mesh_ids:wp.array(dtype=wp.uint64),points:wp.array2d(dtype=wp.vec3),
                 result:wp.array2d(dtype=wp.vec3),faces:wp.array2d(dtype=int)):
    world,slot=wp.tid();p=points[world,slot]
    query=wp.mesh_query_point_no_sign(mesh_ids[slot],p,1.e6)
    if query.result:
        result[world,slot]=wp.mesh_eval_position(mesh_ids[slot],query.face,query.u,query.v)
        faces[world,slot]=query.face
    else:
        result[world,slot]=p;faces[world,slot]=-1


class NativeHandSurfaces:
    """The same first collision geom and raw mesh triangles as project_to_hand."""
    def __init__(self,cpu,bodies,device,mesh_arrays=None):
        self.device=torch.device(device);self.storage=[];ids=[];rot=[];pos=[]
        import trimesh
        for body in bodies:
            geoms=np.flatnonzero(cpu.geom_bodyid==body)
            selected=[g for g in geoms if cpu.geom_contype[g]!=0 or cpu.geom_conaffinity[g]!=0]
            if not selected:raise ValueError('No native collision geom for contact body')
            g=selected[0]
            if cpu.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH:raise ValueError('Rigid MANO contact geom must be a mesh')
            mesh=lambda name:mesh_arrays[name] if mesh_arrays is not None else getattr(cpu,name)
            mid=cpu.geom_dataid[g];a=mesh('mesh_vertadr')[mid];n=mesh('mesh_vertnum')[mid];f=mesh('mesh_faceadr')[mid];nf=mesh('mesh_facenum')[mid]
            # Trimesh(process=True) is used in the original create_hand_trimesh.
            tm=trimesh.Trimesh(mesh('mesh_vert')[a:a+n],mesh('mesh_face')[f:f+nf])
            v=wp.array(np.asarray(tm.vertices,np.float32),dtype=wp.vec3,device=str(self.device))
            faces=wp.array(np.asarray(tm.faces,np.int32).reshape(-1),dtype=int,device=str(self.device))
            mesh=wp.Mesh(points=v,indices=faces);self.storage.append((v,faces,mesh));ids.append(mesh.id)
            from scipy.spatial.transform import Rotation
            rot.append(Rotation.from_quat(cpu.geom_quat[g],scalar_first=True).as_matrix());pos.append(cpu.geom_pos[g])
        self.ids=wp.array(ids,dtype=wp.uint64,device=str(self.device))
        self.rot=torch.tensor(np.asarray(rot),device=self.device,dtype=torch.float32)
        self.pos=torch.tensor(np.asarray(pos),device=self.device,dtype=torch.float32)

    def project(self,physics,bodies,world):
        bodylocal=physics.local_points(bodies,world)
        geomlocal=(self.rot.transpose(-1,-2)[None]@(bodylocal-self.pos[None])[...,None]).squeeze(-1).contiguous()
        projected=torch.empty_like(geomlocal);face=torch.empty(world.shape[:2],device=self.device,dtype=torch.int32)
        wp.launch(nearest_mesh,world.shape[:2],inputs=[self.ids,wp.from_torch(geomlocal,dtype=wp.vec3)],
                  outputs=[wp.from_torch(projected,dtype=wp.vec3),wp.from_torch(face)],
                  stream=physics.stream,device=str(self.device))
        local=(self.rot[None]@projected[...,None]).squeeze(-1)+self.pos[None]
        return local,face>=0


def native_admm(physics,q,hand_bodies,hand_local,object_cpn,surfaces,qp,external,center,
                *,settings,hand_body_count,progress=None):
    """Native object search, virtual-force hand integration, then dual update.

    Post-ADMM native filter/squeeze/pregrasp are deliberately separate stages.
    An ADMM state alone is not advertised as a successful native grasp.
    """
    from .native_admm import object_step,update_dual
    physics.reset(q);initial_state=physics.snapshot();local=hand_local.clone()
    cp_h=physics.positions(hand_bodies,local).detach();cp_o=object_cpn[...,:3].clone();lam=torch.zeros_like(cp_h)
    hand_surfaces=NativeHandSurfaces(physics.cpu,hand_bodies.detach().cpu().tolist(),physics.device,physics.frozen_mesh_arrays)
    active=torch.ones(len(q),device=q.device,dtype=torch.bool);fallback=torch.zeros_like(active)
    numerical_failure=torch.zeros_like(active);steps=torch.zeros(len(q),device=q.device,dtype=torch.int32)
    for iteration in range(settings.step):
        for _ in range(settings.step_obj):
            updated,audit,gradient,finite=object_step(cp_o,cp_h,lam,surfaces,qp,external,center,settings)
            failed=active&(~audit['valid']|~finite)
            fallback|=failed;active&=~failed
            cp_o=torch.where(active[:,None,None],updated,cp_o)
        target=cp_o-lam
        if not settings.fix_hand_local_cp:
            new_local,valid=hand_surfaces.project(physics,hand_bodies,target)
            numerical_failure|=active&~valid.all(-1);active&=valid.all(-1)
            local[...,:3]=torch.where(active[:,None,None],new_local,local[...,:3])
        for _ in range(settings.step_hand):
            physics.apply_forces(hand_bodies,local,target,mask=active)
            physics.advance(settings.substep,mask=active)
        cp_h=physics.positions(hand_bodies,local).detach()
        next_lam=update_dual(lam,cp_h,cp_o)
        lam=torch.where(active[:,None,None],next_lam,lam)
        finite=torch.isfinite(physics.tensor('qpos')).all(-1)&torch.isfinite(lam).all((1,2))&~physics.overflow()
        numerical_failure|=active&~finite;active&=finite;steps+=active.to(steps.dtype)
        if progress is not None:progress(iteration+1,steps,fallback,numerical_failure)
    fallback_active=fallback&~numerical_failure
    physics.restore(initial_state,fallback_active);physics.forward()
    local=torch.where(fallback_active[:,None,None],hand_local,local)
    previous_diff=torch.zeros_like(cp_h[...,0])
    for i in range(20):
        if not bool(fallback_active.any()):break
        current=physics.apply_forces(hand_bodies,local,object_cpn[...,:3],normal=object_cpn[...,3:],mask=fallback_active)
        physics.advance(10,mask=fallback_active)
        improved=((previous_diff-current).amax(-1)>1e-4) if i else torch.ones_like(fallback_active)
        previous_diff=current
        packet=physics.native_contacts(hand_body_count,.002)
        stable=physics.collision_pass(packet,-.002,-.002)
        fallback_active&=~(~improved&stable)
    finite=torch.isfinite(physics.tensor('qpos')).all(-1)&~physics.overflow()
    numerical_failure|=~finite
    return dict(qpos=physics.tensor('qpos').clone(),hand_cpn_b=local,cp_o=cp_o,cp_h=cp_h,lam=lam,
                active=~numerical_failure,fallback_used=fallback,numerical_failure=numerical_failure,
                steps=steps,initial_state=initial_state,hand_bodies=hand_bodies)


def native_filter_squeeze_pregrasp(physics,admm,required_groups,external,center,*,hand_body_count):
    """Native signed-distance filters, six-coefficient QP, squeeze and pregrasp.

    Reading distinct contact counts schedules QP batch shapes; pose arithmetic,
    QP solves and all physical updates stay on CUDA.
    """
    from .native_batched_qp import NativeBatchedQP
    from .native_admm import Settings
    device=physics.device;batch=physics.worlds;packet=physics.native_contacts(hand_body_count,.002)
    finite=~admm['numerical_failure'];limits=physics.limits_pass()
    collision=physics.collision_pass(packet,-.002,-.002)
    count=torch.zeros(batch,device=device,dtype=torch.long)
    count.scatter_add_(0,packet['world'].clamp(0,batch-1),packet['ho'].long())
    width=max(1,int(count.max()))
    padded=torch.zeros((batch,width),device=device,dtype=torch.long)
    flat_ids=torch.nonzero(packet['ho'],as_tuple=False).flatten()
    flat_ids=flat_ids[torch.argsort(packet['world'][flat_ids],stable=True)]
    worlds_flat=packet['world'][flat_ids]
    starts=count.cumsum(0)-count
    ranks=torch.arange(len(flat_ids),device=device)-starts[worlds_flat]
    padded[worlds_flat,ranks]=flat_ids
    contacts=count>0;required=torch.ones_like(contacts)
    for group in required_groups:
        hit=packet['ho']&torch.isin(packet['body'],torch.as_tensor(group,device=device,dtype=torch.long))
        n=torch.zeros_like(count);n.scatter_add_(0,packet['world'].clamp(0,batch-1),hit.long());required&=n>0
    qualified=finite&limits&collision&contacts&required
    qp_pass=torch.zeros_like(qualified);wrenches=torch.zeros((len(packet['world']),6),device=device)
    qp_audit=[]
    # Contact count is scheduling metadata, not a CPU force or pose calculation.
    for n in torch.unique(count[qualified]).detach().cpu().tolist():
        if not n:continue
        worlds=torch.nonzero(qualified&(count==n),as_tuple=False).flatten()
        indices=padded[worlds,:n]
        qp=NativeBatchedQP(n,str(device),mode='filter')
        p=packet['position'][indices];normal=packet['normal'][indices]
        solved=qp.solve(p,normal,external[worlds],center[worlds])
        cw,error=qp.contact_wrenches(p,normal,external[worlds],center[worlds],solved['alpha'])
        passed=solved['valid']&(error<=.001)
        qp_pass[worlds]=passed;wrenches[indices]=(10*cw).to(wrenches.dtype)
        qp_audit.append(dict(worlds=worlds,**{k:v for k,v in solved.items() if k!='alpha'},error=error))
    grasp_pass=qualified&qp_pass
    grasp=physics.tensor('qpos')[:,:physics.hand_qpos].clone()
    # Native squeeze applies forces at all actual contact positions, without a pose cost.
    ft=wp.to_torch(physics.ft);ft.zero_()
    ids=torch.nonzero(packet['ho'],as_tuple=False).flatten();wid=packet['world'][ids];body=packet['body'][ids]
    force=wrenches[ids,:3]*grasp_pass[wid,None]
    torque=torch.linalg.cross(packet['position'][ids]-physics.tensor('xipos')[wid,body],force,dim=-1)
    flat=ft.reshape(-1,6)
    flat.index_add_(0,wid*physics.cpu.nbody+body,torch.cat((force,torque),-1))
    physics.tensor('qfrc_applied').zero_()
    with physics.scope():support.apply_ft(physics.model,physics.data,physics.ft,physics.data.qfrc_applied,False)
    delta=physics.tensor('qfrc_applied').clone();physics.tensor('qfrc_applied').zero_()
    delta/=physics.squeeze_gains[None]
    squeeze=torch.cat((grasp[:,:7],grasp[:,7:]+delta[:,6:physics.cpu.nv]),-1)
    # Retain the native hybrid output contact update, independently of the ADMM slots.
    hybrid=physics.contacts(admm['hand_bodies'],admm['hand_cpn_b']).clone()
    valid_padded=torch.arange(width,device=device)[None]<count[:,None]
    positions=packet['position'][padded];normals=packet['normal'][padded];body_ids=packet['body'][padded]
    for slot,body_id in enumerate(admm['hand_bodies']):
        dist=(hybrid[:,slot,None,:3]-positions).norm(dim=-1)
        angle=torch.acos((hybrid[:,slot,None,3:]*normals).sum(-1).clamp(-1,1))
        match=valid_padded&(body_ids==body_id)&(dist<.03)&(angle<torch.pi/4)
        first=match.long().argmax(-1);row_ids=torch.arange(batch,device=device)
        selected=torch.cat((positions[row_ids,first],normals[row_ids,first]),-1)
        hybrid[:,slot]=torch.where(match.any(-1)[:,None],selected,hybrid[:,slot])
    # Every world retains its own original early-stop index.
    pre=torch.full((batch,20,physics.hand_qpos),torch.nan,device=device)
    lengths=torch.zeros(batch,device=device,dtype=torch.long);active=grasp_pass.clone()
    pregrasp_pass=torch.zeros_like(active)
    for i in range(20):
        margin=.045*min((i+1)/20*2,1.)
        physics.set_margin(margin);physics.keep_stable(mask=active);physics.advance(10,mask=active)
        pre[:,i]=physics.tensor('qpos')[:,:physics.hand_qpos];lengths+=active.long()
        if i%2==0 and margin>=.045:
            packet_pre=physics.native_contacts(hand_body_count)
            early=physics.collision_pass(packet_pre,.04,-.002)
            ended=active&early
            pregrasp_pass|=ended&physics.limits_pass()&~physics.overflow()
            active&=~early
    packet_pre=physics.native_contacts(hand_body_count)
    pregrasp_pass|=active&physics.limits_pass()&physics.collision_pass(packet_pre,.04,-.002)&~physics.overflow()
    reverse=lengths[:,None]-1-2*torch.arange(10,device=device)[None]
    pregrasp=pre.gather(1,reverse.clamp_min(0)[...,None].expand(-1,-1,physics.hand_qpos))
    pregrasp=torch.where((reverse>=0)[...,None],pregrasp,torch.nan)
    success=grasp_pass&pregrasp_pass&torch.isfinite(squeeze).all(-1)
    return dict(grasp_qpos=grasp,squeeze_qpos=squeeze,pregrasp_qpos=pregrasp,
        pregrasp_count=(lengths+1)//2,native_output=success,limits_pass=limits,collision_pass=collision,
        required_groups_pass=required,contacts_pass=contacts,qp_pass=qp_pass,pregrasp_pass=pregrasp_pass,
        contact_packet=packet,contact_wrenches=wrenches,hybrid_hand_cpn_w=hybrid,qp_audit=qp_audit)
