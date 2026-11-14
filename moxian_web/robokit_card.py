"""CARD energies as a Robokit cost-only task, with captured elite Gaussian CEM.

The search uses Robokit's fitted capsules. Native mesh validation is applied to
ranked endpoints after sampling; no trajectory or historical-pose gate is used.
"""
from dataclasses import dataclass
import time
import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation
from robokit.opt import PopulationSolver, PopulationSolverConfig, Stage, VarValues, VectorVar
from robokit.opt.var_values import VarKey
from robokit.opt.sampler import GaussianSamplerConfig, GaussianSampler
from robokit.terms.task import Task
from robokit.utils.warp_utils import wp_vec7
from robokit.lie.se3_kernels import se3_apply_func, se3_inverse_func
from .card_geometry import source_wrist


@wp.func
def rotation(t: wp_vec7):
    return wp.quat(t[4], t[5], t[6], t[3])


@wp.func
def squared(v: wp.vec3):
    return wp.dot(v, v)


@wp.func
def segment_distance(a: wp.vec3, b: wp.vec3, c: wp.vec3, d: wp.vec3):
    u = b-a; v = d-c; w = a-c
    aa = wp.dot(u,u); bb = wp.dot(u,v); cc = wp.dot(v,v)
    dd = wp.dot(u,w); ee = wp.dot(v,w); den = aa*cc-bb*bb
    s = (bb*ee-cc*dd)/wp.max(den,1.0e-20)
    t = (aa*ee-bb*dd)/wp.max(den,1.0e-20)
    distance = float(1.0e10)
    if den>1.0e-16 and s>=0.0 and s<=1.0 and t>=0.0 and t<=1.0:
        distance = wp.length(w+s*u-t*v)
    for k in range(4):
        point = a; start = c; edge = d-c
        if k==1: point = b
        if k==2: point = c; start = a; edge = b-a
        if k==3: point = d; start = a; edge = b-a
        alpha = wp.clamp(wp.dot(point-start,edge)/wp.max(squared(edge),1.0e-20),0.0,1.0)
        distance = wp.min(distance,wp.length(point-start-alpha*edge))
    return distance


@wp.struct
class CardData:
    links: wp.array(dtype=int)
    meshes: wp.array(dtype=wp.uint64)
    active: wp.array(dtype=int)
    target: wp.array(dtype=wp.vec3)
    direction: wp.array(dtype=wp.vec3)
    preference: wp.array(dtype=wp.vec3)
    rho: wp.array(dtype=float)
    old_valid: wp.array(dtype=int)
    old_local: wp.array(dtype=wp.vec3)
    apex: wp.array(dtype=wp.vec3)
    belly: wp.array(dtype=wp.vec3)
    human_tips: wp.array(dtype=wp.vec3)
    human_normals: wp.array(dtype=wp.vec3)
    human_bones: wp.array(dtype=wp.vec3)
    free: wp.array(dtype=float)
    cap_a: wp.array(dtype=wp.vec3)
    cap_b: wp.array(dtype=wp.vec3)
    cap_radius: wp.array(dtype=float)
    pairs: wp.array(dtype=wp.vec2i)
    samples: wp.array(dtype=wp.vec3)
    offsets: wp.array(dtype=int)
    planes: wp.array(dtype=wp.vec4)
    plane_count: wp.array(dtype=int)
    wrist_inverse: wp.array(dtype=wp_vec7)
    object_inverse: wp.array(dtype=wp_vec7)
    object_mesh: wp.array(dtype=wp.uint64)
    previous: wp.array2d(dtype=float)
    previous_valid: wp.array(dtype=int)
    motion_joint: wp.array(dtype=float)
    motion_valid: wp.array(dtype=int)
    weights: wp.array(dtype=float)


@wp.kernel(enable_backward=False)
def unpack(x: wp.array2d(dtype=float), mapping: wp.array(dtype=int),
           joints: wp.array2d(dtype=float), base: wp.array(dtype=wp_vec7)):
    i=wp.tid()
    v=wp.vec3(x[i,3],x[i,4],x[i,5]);angle=wp.length(v)
    quat=wp.quat_identity()
    if angle>1.0e-10: quat=wp.quat_from_axis_angle(v/angle,angle)
    base[i]=wp_vec7(x[i,0],x[i,1],x[i,2],quat[3],quat[0],quat[1],quat[2])
    for j in range(20):joints[i,mapping[j]]=x[i,6+j]


# Columns match the independently implemented Python objective for diagnostics.
PARTS=('contact_position','contact_direction','contact_fix','style_tip','style_normal',
       'style_bone','smooth_joint','smooth_translation','smooth_rotation','collision_self','collision_object','motion_joint')


@wp.kernel(enable_backward=False)
def card_cost(x: wp.array2d(dtype=float), poses: wp.array2d(dtype=wp_vec7), data: CardData,
              parts: wp.array2d(dtype=float), raw: wp.array(dtype=float), depths: wp.array2d(dtype=float), costs: wp.array(dtype=float)):
    i=wp.tid();w=data.weights
    for k in range(12):parts[i,k]=0.0
    for region in range(16):
        tf=poses[i,data.links[region]];quat=rotation(tf)
        if data.active[region]!=0:
            alpha=float(0.0)
            if region>0 and (region-1)%3>0:alpha=0.15
            local_target=se3_apply_func(se3_inverse_func(tf),data.target[region])
            query=(local_target+alpha*data.preference[region])/(1.0+alpha)
            hit=wp.mesh_query_point_no_sign(data.meshes[region],query,10.0)
            if hit.result:
                local=wp.mesh_eval_position(data.meshes[region],hit.face,hit.u,hit.v)
                normal=wp.quat_rotate(quat,wp.mesh_eval_face_normal(data.meshes[region],hit.face))
                point=se3_apply_func(tf,local);rho=data.rho[region]
                parts[i,0]+=rho*rho*w[0]*squared((point-data.target[region])*1000.0)
                parts[i,1]+=rho*rho*w[0]*w[1]*(1.0-wp.clamp(wp.dot(normal,data.direction[region]),-1.0,1.0))
                if data.old_valid[region]!=0:
                    parts[i,2]+=rho*rho*w[2]*squared((point-se3_apply_func(tf,data.old_local[region]))*1000.0)
            else:parts[i,0]+=1.0e25
        if region>0:
            finger=(region-1)//3;segment=(region-1)%3
            a=wp.vec3(tf[0],tf[1],tf[2]);b=se3_apply_func(tf,data.apex[region])
            if segment<2:
                next_tf=poses[i,data.links[region+1]];b=wp.vec3(next_tf[0],next_tf[1],next_tf[2])
            axis=wp.normalize(b-a)
            parts[i,5]+=data.free[finger]*w[5]*(1.0-wp.clamp(wp.dot(axis,data.human_bones[3*finger+segment]),-1.0,1.0))
            if segment==2:
                parts[i,3]+=data.free[finger]*w[3]*squared((b-data.human_tips[finger])*1000.0)
                belly=wp.quat_rotate(quat,data.belly[region])
                parts[i,4]+=data.free[finger]*w[4]*(1.0-wp.clamp(wp.dot(belly,data.human_normals[finger]),-1.0,1.0))
    if data.previous_valid[0]!=0:
        for j in range(20):parts[i,6]+=w[6]*(x[i,6+j]-data.previous[0,6+j])**2.0
        for j in range(3):parts[i,7]+=w[7]*((x[i,j]-data.previous[0,j])*1000.0)**2.0
        rv=wp.vec3(x[i,3],x[i,4],x[i,5]);old=wp.vec3(data.previous[0,3],data.previous[0,4],data.previous[0,5])
        r=wp.quat_identity();pr=wp.quat_identity()
        if wp.length(rv)>1.0e-10:r=wp.quat_from_axis_angle(wp.normalize(rv),wp.length(rv))
        if wp.length(old)>1.0e-10:pr=wp.quat_from_axis_angle(wp.normalize(old),wp.length(old))
        relative=wp.quat_inverse(pr)*r
        angle=2.0*wp.atan2(wp.length(wp.vec3(relative[0],relative[1],relative[2])),wp.abs(relative[3]))
        parts[i,8]+=w[8]*angle*angle
    if data.motion_valid[0]!=0:
        for j in range(20):parts[i,11]+=w[14]*(x[i,6+j]-data.motion_joint[j])**2.0
    palm_gap=float(1.0e10)
    for k in range(data.pairs.shape[0]):
        pair=data.pairs[k];ia=pair[0];ib=pair[1]
        ta=poses[i,data.links[ia]];tb=poses[i,data.links[ib]]
        gap=segment_distance(se3_apply_func(ta,data.cap_a[ia]),se3_apply_func(ta,data.cap_b[ia]),
            se3_apply_func(tb,data.cap_a[ib]),se3_apply_func(tb,data.cap_b[ib]))-data.cap_radius[ia]-data.cap_radius[ib]
        parts[i,9]+=w[9]*wp.max(0.0,w[10]-gap*1000.0)**2.0
        if ia==0:palm_gap=wp.min(palm_gap,gap*1000.0)
    max_depth=float(0.0)
    if data.plane_count[0]>0 or data.object_mesh[0]!=wp.uint64(0):
        for region in range(16):
            tf=poses[i,data.links[region]]
            for sample in range(data.offsets[region],data.offsets[region+1]):
                point=se3_apply_func(tf,data.samples[sample]);depth=float(0.0)
                if data.object_mesh[0]!=wp.uint64(0):
                    local=se3_apply_func(data.object_inverse[0],point)
                    hit=wp.mesh_query_point_sign_winding_number(data.object_mesh[0],local,10.0)
                    if hit.result and hit.sign<0.0:
                        depth=wp.length(local-wp.mesh_eval_position(data.object_mesh[0],hit.face,hit.u,hit.v))*1000.0
                else:
                    local=se3_apply_func(data.wrist_inverse[0],point);distance=float(-1.0e10)
                    for k in range(data.plane_count[0]):
                        plane=data.planes[k];distance=wp.max(distance,wp.dot(local,wp.vec3(plane[0],plane[1],plane[2]))+plane[3])
                    depth=wp.max(0.0,-distance)*1000.0
                max_depth=wp.max(max_depth,depth)
                parts[i,10]+=w[11]*wp.max(0.0,depth-w[12])**2.0
    total=float(0.0)
    for k in range(12):total+=parts[i,k]
    raw[i]=total
    depths[i,0]=max_depth;depths[i,1]=palm_gap
    # Preserve the live policy: inferred objects guide; tracked solid meshes gate.
    # Native self-penetration is independently checked on shortlisted endpoints.
    if not wp.isfinite(total):total=1.0e30
    if data.object_mesh[0]!=wp.uint64(0) and max_depth>w[13]:total=1.0e29+wp.min(total,1.0e28)
    costs[i]+=total


@wp.kernel(enable_backward=False)
def describe_material(poses: wp.array2d(dtype=wp_vec7),data: CardData,
                      local: wp.array2d(dtype=wp.vec3),world: wp.array2d(dtype=wp.vec3),
                      normals: wp.array2d(dtype=wp.vec3),faces: wp.array2d(dtype=int),bary: wp.array2d(dtype=wp.vec3)):
    region=wp.tid();tf=poses[0,data.links[region]]
    if data.active[region]!=0:
        alpha=float(0.0)
        if region>0 and (region-1)%3>0:alpha=0.15
        target=se3_apply_func(se3_inverse_func(tf),data.target[region])
        query=(target+alpha*data.preference[region])/(1.0+alpha)
        hit=wp.mesh_query_point_no_sign(data.meshes[region],query,10.0)
        if hit.result:
            point=wp.mesh_eval_position(data.meshes[region],hit.face,hit.u,hit.v)
            local[0,region]=point;world[0,region]=se3_apply_func(tf,point)
            normals[0,region]=wp.quat_rotate(rotation(tf),wp.mesh_eval_face_normal(data.meshes[region],hit.face))
            faces[0,region]=hit.face;bary[0,region]=wp.vec3(hit.u,hit.v,1.0-hit.u-hit.v)


@wp.kernel(enable_backward=False)
def archive(x: wp.array2d(dtype=float), costs: wp.array(dtype=float), counter: wp.array(dtype=int),
            values: wp.array2d(dtype=float), saved_costs: wp.array(dtype=float), widths: wp.array(dtype=int), population: int):
    i=wp.tid();slot=counter[0]%widths.shape[0];offset=slot*population+i
    for j in range(26):values[offset,j]=x[i,j]
    saved_costs[offset]=costs[i]
    if i==0:widths[slot]=x.shape[0]


@wp.kernel(enable_backward=False)
def advance(counter: wp.array(dtype=int)):
    counter[0]+=1


@wp.kernel(enable_backward=False)
def multiply(values: wp.array(dtype=float), scale: float):
    i=wp.tid();values[i]=wp.max(values[i]*scale,1.0e-12)


@wp.kernel(enable_backward=False)
def ball_projection(x: wp.array2d(dtype=float), noise: wp.array2d(dtype=float), lower: wp.array2d(dtype=float), upper: wp.array2d(dtype=float), radius: float):
    i=wp.tid();center=wp.vec3((lower[0,0]+upper[0,0])*0.5,(lower[0,1]+upper[0,1])*0.5,(lower[0,2]+upper[0,2])*0.5)
    delta=wp.vec3(x[i,0],x[i,1],x[i,2])-center
    if wp.length(delta)>radius:
        change=delta*(radius/wp.length(delta)-1.0)
        for j in range(3):x[i,j]+=change[j];noise[i,j]+=change[j]


@wp.kernel(enable_backward=False)
def motion_draws(x: wp.array2d(dtype=float),noise: wp.array2d(dtype=float),centers: wp.array2d(dtype=float),
                 motion: wp.array2d(dtype=float),valid: wp.array(dtype=int),lower: wp.array2d(dtype=float),
                 upper: wp.array2d(dtype=float),count: int,scale: float,std: wp.array2d(dtype=float),
                 active: wp.array(dtype=float),seed: int,epoch: wp.array(dtype=int),salt: int):
    i,j=wp.tid()
    if valid[0]!=0 and i>=x.shape[0]-count:
        # Read the same raw normal as Robokit, before global-centre clamping.
        # Clipped global noise would wrongly truncate an interior local cloud
        # whenever the global CEM mean sits on a native joint limit.
        state=wp.rand_init(seed+epoch[0]*7919,(salt*1000003+i)*x.shape[1]+j)
        value=centers[0,j]
        if active[j]!=0.0:value=wp.clamp(motion[0,j]+scale*std[0,j]*wp.randn(state),lower[0,j],upper[0,j])
        x[i,j]=value;noise[i,j]=value-centers[0,j]


class CardSampler(GaussianSampler):
    def _move(self, scores, out_delta):
        # Reuse Robokit's sorting, elite masking, annealing, weights and update
        # kernels, adding paper eta after its elite-standard-deviation estimate.
        super()._move(scores,out_delta)
        if self.temperature!=1.0:
            wp.launch(multiply,self.temperatures.shape,[self.temperatures,self.temperature],device=self._device)
            block=1 if self._device.is_cpu else 256
            wp.launch(self._weights_kernel,(self.instances,block),[self._elite_costs,self.temperatures,self.epsilon,block],[self.weights],block_dim=block,device=self._device)
            wp.launch(self._update_kernel,(self.instances,self.tangent_dim,block),
                [self.weights,self.noise,self._elite_costs,self.best_indices,0.0,self.min_std,self.learning_rate,self.epsilon,block],
                [self.std,out_delta],block_dim=block,device=self._device)

    def sample(self, var, scores):
        result=super().sample(var,scores)
        if self.lower is not None:
            if getattr(self,'motion_center',None) is not None:
                wp.launch(motion_draws,result.get(self.var_key).x.shape,
                    [result.get(self.var_key).x,self.noise,self._centers.get(self.var_key).x,
                     self.motion_center,self.motion_valid,self.lower,self.upper,self.motion_count,.15,
                     self.std,self.active_dof_mask,self.seed,self.epoch,self._round-1],device=self._device)
            wp.launch(ball_projection,result.batch_size,[result.get(self.var_key).x,self.noise,self.lower,self.upper,self.radius],device=self._device)
        return result


@dataclass(frozen=True)
class CardSamplerConfig(GaussianSamplerConfig):
    def build(self,var,num_candidates,active_dof_mask=None):
        from dataclasses import asdict
        return CardSampler(**asdict(self),tangent_dim=26,instances=var.batch_size,device=var.device,active_dof_mask=active_dof_mask)


class CardCostTask(Task):
    """Cost-only Robokit task; buffers persist through full-solve CUDA capture."""
    def __init__(self, owner):
        self.owner=owner;self.robot=owner.initializer.robot;self.device=str(owner.device)
        self.mapping=wp.array(owner.initializer.to_native.astype(np.int32),dtype=int,device=self.device)
        self.states={};self.buffers={};self.data=CardData();d=self.data;s=owner.settings
        def array(values,dtype):return wp.array(np.asarray(values),dtype=dtype,device=self.device)
        links=np.asarray(owner.initializer.link_ids)[[0]+[1+4*f+k for f in range(5) for k in range(3)]]
        d.links=array(links,int);self.meshes=[]
        for triangles in owner.pad_triangles:
            vertices=triangles.cpu().numpy().reshape(-1,3)
            self.meshes.append(wp.Mesh(points=array(vertices,wp.vec3),indices=array(np.arange(len(vertices)),int)))
        d.meshes=array([m.id for m in self.meshes],wp.uint64)
        d.active=array(np.zeros(16,np.int32),int);d.rho=array(np.ones(16),float);d.old_valid=array(np.zeros(16,np.int32),int)
        for key in ['target','direction','preference','old_local']:setattr(d,key,array(np.zeros((16,3)),wp.vec3))
        d.apex=array(owner.apices.cpu().numpy(),wp.vec3);d.belly=array(owner.surface.normals[owner.region_owners],wp.vec3)
        d.human_tips=array(np.zeros((5,3)),wp.vec3);d.human_normals=array(np.zeros((5,3)),wp.vec3);d.human_bones=array(np.zeros((15,3)),wp.vec3);d.free=array(np.ones(5),float)
        spec=self.robot.spec;caps=spec.local_link_capsule_endpoints[links]
        d.cap_a=array(caps[:,0],wp.vec3);d.cap_b=array(caps[:,1],wp.vec3);d.cap_radius=array(spec.link_capsule_radii[links],float)
        d.pairs=array(owner.cross_pairs+owner.palm_pairs,wp.vec2i)
        samples=[];offsets=[0];local=owner.collision_local.cpu().numpy();regions=owner.collision_regions.cpu().numpy()
        for region in range(16):samples.extend(local[regions==region]);offsets.append(len(samples))
        d.samples=array(samples,wp.vec3);d.offsets=array(offsets,int)
        self.plane_capacity=4096;d.planes=array(np.zeros((self.plane_capacity,4)),wp.vec4);d.plane_count=array([0],int)
        d.wrist_inverse=array([[0,0,0,1,0,0,0]],wp_vec7);d.object_inverse=array([[0,0,0,1,0,0,0]],wp_vec7);d.object_mesh=array([0],wp.uint64);self.object_mesh=None
        d.previous=wp.zeros((1,26),dtype=float,device=self.device);d.previous_valid=array([0],int)
        d.motion_joint=wp.zeros(20,dtype=float,device=self.device);d.motion_valid=array([0],int)
        d.weights=array([s.wc**2,s.direction_price*s.radius_mm**2,s.wfix**2,s.wtip**2,s.wn**2,s.wb**2,s.wtheta**2,s.wt**2,s.womega**2,s.wself**2,s.margin_mm,s.wobj_pseudo**2,s.allowance_mm,s.hard_penetration_mm,s.wmotion_joint**2],float)
        self.counter=wp.zeros(1,dtype=int,device=self.device);slots=s.iterations+8
        self.archive_x=wp.zeros((slots*s.population,26),dtype=float,device=self.device)
        self.archive_cost=wp.full(slots*s.population,wp.inf,dtype=float,device=self.device)
        self.archive_width=wp.zeros(slots,dtype=int,device=self.device)

    def update(self,c):
        d=self.data;o=self.owner
        def put(target,values):wp.copy(target,wp.array(np.asarray(values),dtype=target.dtype,device=self.device))
        active=np.zeros(16,np.int32);target=np.zeros((16,3));direction=target.copy();preference=target.copy();rho=np.ones(16)
        for a in c['anchors']:
            r=a['region'];active[r]=1;target[r]=a['position_m'];direction[r]=a['direction'];rho[r]=np.clip(1+.5*a['force']/o.force_max if o.force_max else 1.,1.,1.5)
            owner=o.region_owners[r];pool=o.pools[r].cpu().numpy();psi=o.surface.axes[owner]*o.surface.lengths[owner]*a['u'];n=o.surface.normals[owner]
            preference[r]=pool.mean(0) if r==0 else psi+n*np.mean((pool-psi)@n)
        for key,value in [('active',active),('target',target),('direction',direction),('preference',preference),('rho',rho)]:put(getattr(d,key),value)
        old=np.zeros((16,3));valid=np.zeros(16,np.int32)
        for k,v in o.previous_contacts.items():old[k]=v.cpu().numpy();valid[k]=1
        put(d.old_local,old);put(d.old_valid,valid)
        for key,source in [('human_tips','tips'),('human_normals','normals'),('human_bones','directions')]:put(getattr(d,key),c[source].cpu().numpy().reshape(-1,3))
        put(d.free,c['free'].cpu().numpy());put(d.previous_valid,[int(c['previous'] is not None)])
        if c['previous'] is not None:put(d.previous,pack(c['previous'])[None])
        put(d.motion_valid,[int(c.get('motion_joint') is not None)])
        if c.get('motion_joint') is not None:put(d.motion_joint,c['motion_joint'].cpu().numpy())
        sdf=c['sdf'];put(d.plane_count,[0]);put(d.object_mesh,[0])
        weights=d.weights.numpy();weights[11]=o.settings.wobj_pseudo**2
        if sdf.kind=='tracked_mesh':
            if self.object_mesh is None or self.object_version!=sdf.version:
                self.object_mesh=wp.Mesh(points=wp.array(sdf.vertices,dtype=wp.vec3,device=self.device),indices=wp.array(sdf.faces.ravel(),dtype=int,device=self.device),support_winding_number=True);self.object_version=sdf.version
            put(d.object_mesh,[self.object_mesh.id]);tf=np.linalg.inv(sdf.transform);quat=Rotation.from_matrix(tf[:3,:3]).as_quat();put(d.object_inverse,[np.r_[tf[:3,3],quat[3],quat[:3]]]);weights[11]=o.settings.wobj_mesh**2
        elif sdf.hull is not None:
            planes=sdf.planes.cpu().numpy()
            if len(planes)>self.plane_capacity:raise ValueError('CARD pseudo-object exceeds captured plane capacity')
            wp.copy(d.planes,wp.array(planes,dtype=wp.vec4,device=self.device),count=len(planes));put(d.plane_count,[len(planes)])
            origin,r=source_wrist(c['joints']);quat=Rotation.from_matrix(r.T).as_quat();put(d.wrist_inverse,[np.r_[-r.T@origin,quat[3],quat[:3]]])
        put(d.weights,weights)

    def precompute(self,var_values,need_gradient=False):
        n=var_values.batch_size
        if n not in self.states:
            self.states[n]=self.robot.state(q=wp.zeros((n,20),dtype=float,device=self.device),T_world_base=wp.zeros(n,dtype=wp_vec7,device=self.device))
            self.buffers[n]=(wp.zeros((n,len(PARTS)),dtype=float,device=self.device),wp.zeros(n,dtype=float,device=self.device),wp.zeros((n,2),dtype=float,device=self.device))
        state=self.states[n];state.invalidate()
        wp.launch(unpack,n,[var_values.get(VarKey.ROBOT).x,self.mapping,state.q,state.T_world_base],device=self.device)
        self.robot.forward_kinematics(state)

    def compute_weighted_cost(self,var_values,*,out_cost):
        n=var_values.batch_size;parts,raw,depths=self.buffers[n]
        wp.launch(card_cost,n,[var_values.get(VarKey.ROBOT).x,self.states[n].T_world_link.reshape((n,self.robot.spec.num_links)),self.data,parts,raw,depths,out_cost],device=self.device)
        wp.launch(archive,n,[var_values.get(VarKey.ROBOT).x,out_cost,self.counter,self.archive_x,self.archive_cost,self.archive_width,self.owner.settings.population],device=self.device)
        wp.launch(advance,1,[self.counter],device=self.device)


def pack(initial,rotation_reference=None):
    t,r,j=(v.detach().cpu().numpy() for v in initial)
    vector=Rotation.from_matrix(r).as_rotvec();angle=np.linalg.norm(vector)
    if rotation_reference is not None and angle>1e-10:
        # Equivalent axis-angle charts straddle +/-pi. All mixed candidates
        # must share the current solve's chart before their vectors are averaged.
        choices=vector+np.array([-2*np.pi,0,2*np.pi])[:,None]*(vector/angle)
        vector=choices[np.linalg.norm(choices-rotation_reference,axis=1).argmin()]
    return np.r_[t,vector,j].astype(np.float32)


class RobokitCardSolver:
    def __init__(self,owner):
        self.owner=owner;self.settings=owner.settings;s=self.settings;device=str(owner.device)
        self.task=CardCostTask(owner);self.var=VarValues.from_var(VectorVar(wp.zeros((1,26),dtype=float,device=device)),VarKey.ROBOT)
        config=CardSamplerConfig(num_draws=s.population,elite_ratio=s.elite_fraction,mean_elites=True,temperature=s.temperature,
            init_std=1.,min_std=1.0e-9,gamma=0.,noise_decay=s.gamma**(1/s.iterations),sample_mode='mean',seed=s.seed)
        self.solver=PopulationSolver([self.task],PopulationSolverConfig(stages=[Stage(s.iterations)],samplers=[config],cuda_graph_mode='full',use_early_stopping=False),device)
        # Preallocate both widths before capture. Context buffers are filled first.
        self.lower=wp.zeros((1,26),dtype=float,device=device);self.upper=wp.zeros((1,26),dtype=float,device=device)
        self.motion_center=wp.zeros((1,26),dtype=float,device=device);self.motion_valid=wp.zeros(1,dtype=int,device=device)
        self.initialized=False

    def solve(self,initial,c):
        began=time.perf_counter();o=self.owner;s=self.settings;center=pack(initial);device=str(o.device)
        self.task.update(c)
        wp.copy(self.var.get(VarKey.ROBOT).x,wp.array(center[None],dtype=float,device=device))
        lo=np.r_[center[:3]-s.translation_step_m,[-1.0e6]*3,np.maximum(o.lower_np[6:],center[6:]-s.joint_step_rad)]
        hi=np.r_[center[:3]+s.translation_step_m,[1.0e6]*3,np.minimum(o.upper_np[6:],center[6:]+s.joint_step_rad)]
        wp.copy(self.lower,wp.array(lo[None],dtype=float,device=device));wp.copy(self.upper,wp.array(hi[None],dtype=float,device=device))
        motion=None
        if c.get('motion_joint') is not None and s.wmotion_joint>0:
            t,r,_=c['previous']
            if c.get('previous_source_wrist') is not None:
                origin,rotation_now=source_wrist(c['joints']);old_origin,old_rotation=c['previous_source_wrist'];delta=o.tensor(rotation_now@old_rotation.T)
                t=o.tensor(origin)+delta@(t-o.tensor(old_origin));r=delta@r
            motion=(t,r,c['motion_joint'])
            wp.copy(self.motion_center,wp.array(pack(motion,center[3:6])[None],dtype=float,device=device))
        self.motion_valid.fill_(int(motion is not None))
        if not self.initialized:
            # Setup must see bounds on the very first draw, before graph capture.
            original=self.solver.config.cuda_graph_mode;self.solver.config.cuda_graph_mode='none';self.solver.setup(self.var)
            sampler=self.solver.samplers[0];sampler.lower=self.lower;sampler.upper=self.upper;sampler.radius=s.translation_step_m
            sampler.motion_center=self.motion_center;sampler.motion_valid=self.motion_valid
            sampler.motion_count=min(s.population-1,int(s.population*s.motion_sample_fraction))
            self.solver.config.cuda_graph_mode=original
            std=np.array([s.translation_sigma_m]*3+[s.rotation_sigma_rad]*3+[s.joint_sigma_rad]*20,np.float32)
            wp.copy(sampler.std_reset,wp.array(std[None],dtype=float,device=device))
            self.solver._warmup_and_capture();self.initialized=True
        self.task.counter.zero_();self.task.archive_width.zero_();self.task.archive_cost.fill_(wp.inf)
        wp.synchronize_device(device);prepared_at=time.perf_counter()
        self.solver.solve(self.var);wp.synchronize_device(device)
        sampled_at=time.perf_counter()
        widths=self.task.archive_width.numpy();costs=self.task.archive_cost.numpy();xs=self.task.archive_x.numpy()
        ids=np.concatenate([np.arange(i*s.population,i*s.population+int(n)) for i,n in enumerate(widths) if n])
        # The previous solution is a candidate, never the sampling centre or an
        # unconditional display fallback. Re-score it with today's anchors,
        # field and temporal reference, within today's Robokit refinement box.
        # Without this incumbent, tiny changes to smoothness can send identical
        # source frames into different CEM basins and return a worse pose.
        incumbent_ids={};incumbent_costs={};proposals=[]
        if c['previous'] is not None:
            proposals.append(('previous',c['previous']))
            if c.get('previous_source_wrist') is not None:
                origin,rotation_now=source_wrist(c['joints']);old_origin,old_rotation=c['previous_source_wrist']
                delta=rotation_now@old_rotation.T;t,r,j=c['previous']
                transported=(o.tensor(origin)+o.tensor(delta)@(t-o.tensor(old_origin)),o.tensor(delta)@r,j)
                if not np.allclose(pack(transported),pack(c['previous']),atol=1e-8,rtol=0):
                    proposals.append(('source_rigid_motion',transported))
            if c.get('motion_joint') is not None:
                t,r,_=proposals[-1][1]
                proposal=(t,r,c['motion_joint'])
                if not any(np.allclose(pack(proposal),pack(p),atol=1e-8,rtol=0) for _,p in proposals):
                    proposals.append(('source_joint_motion',proposal))
        for kind,proposal in proposals:
            old=pack(proposal,center[3:6])
            if (np.isfinite(old).all() and np.all(old[6:]>=lo[6:]) and np.all(old[6:]<=hi[6:])
                    and np.linalg.norm(old[:3]-center[:3])<=s.translation_step_m+1e-7):
                idx=len(costs);cost=self.parts(proposal,total=True)
                incumbent_ids[idx]=kind;incumbent_costs[kind]=cost
                xs=np.vstack((xs,old));costs=np.r_[costs,cost];ids=np.r_[ids,idx]
        ranking=ids[np.argsort(costs[ids],kind='stable')];finite=ranking[np.isfinite(xs[ranking]).all(1)&np.isfinite(costs[ranking])]
        if not len(finite):raise ValueError('Robokit CARD produced no finite current candidate')
        tested=0;rejected=0;chosen=None;valid_found=False
        # Include low-cost, early/mean, and distributed current-frame candidates.
        # A converged infeasible cluster must not hide a feasible initializer.
        representatives=[]
        for slot,n in enumerate(widths):
            if n:
                start=slot*s.population;representatives.extend([start,start+int(np.argmin(costs[start:start+int(n)]))])
        diverse=finite[np.linspace(0,len(finite)-1,min(64,len(finite)),dtype=int)]
        selected=list(dict.fromkeys([*finite[:128],*incumbent_ids,*representatives,*diverse]))[:256]
        selected_set=set(selected)
        for k in finite:
            if len(selected)>=256:break
            if k not in selected_set:selected.append(int(k));selected_set.add(k)
        batch=np.array(selected,dtype=int);batch=batch[np.argsort(costs[batch],kind='stable')]
        values=xs[batch]
        p,rr=o.frames(o.tensor(values[:,:3]),o.tensor(Rotation.from_rotvec(values[:,3:6]).as_matrix()),o.tensor(values[:,6:]))
        coarse,query_valid=o.mesh_collision.query(p,rr,dense=False)
        coarse_depth=coarse.amax((-1,-2)).cpu().numpy()
        coarse_valid=query_valid.cpu().numpy()&(coarse_depth<=s.self_penetration_mm)&(costs[batch]<1.0e29)
        candidates=batch[coarse_valid]
        # Coarse samples are a subset of the dense audit. A coarse rejection
        # cannot discard a result which would pass the same dense threshold.
        for start in range(0,len(candidates),8):
            ids_to_test=candidates[start:start+8];values=xs[ids_to_test]
            ts=o.tensor(values[:,:3]);rs=o.tensor(Rotation.from_rotvec(values[:,3:6]).as_matrix());js=o.tensor(values[:,6:])
            valid=o.verify_mesh(ts,rs,js).cpu().numpy();tested+=len(ids_to_test);rejected+=int((~valid).sum())
            if valid.any():chosen=int(ids_to_test[np.flatnonzero(valid)[0]]);valid_found=True;break
        fallback_violation=0.0
        if chosen is None:
            # Recover from measured current endpoints, never a historical pose.
            rank=np.lexsort((costs[batch],np.maximum(coarse_depth-s.self_penetration_mm,0)))[:8]
            fallback=batch[rank];values=xs[fallback]
            p,rr=o.frames(o.tensor(values[:,:3]),o.tensor(Rotation.from_rotvec(values[:,3:6]).as_matrix()),o.tensor(values[:,6:]))
            depth,query_valid=o.mesh_collision.query(p,rr,dense=True)
            violation=(depth.amax((-1,-2))-s.self_penetration_mm).clamp_min(0).cpu().numpy()
            violation=np.where(query_valid.cpu().numpy(),violation,np.inf)
            rank=np.lexsort((costs[fallback],violation));chosen=int(fallback[rank[0]])
            fallback_violation=float(violation[rank[0]]) if np.isfinite(violation[rank[0]]) else None
        value=xs[chosen];best=(o.tensor(value[:3]),o.tensor(Rotation.from_rotvec(value[3:6]).as_matrix()),o.tensor(value[6:]))
        wp.synchronize_device(device);verified_at=time.perf_counter()
        diagnostic=dict(backend='robokit_population_solver_warp_card',feasible=valid_found,best_cost=float(costs[chosen]) if valid_found else None,
            finite_candidate_found=True,fallback_cost=float(costs[chosen]),fallback_violation=fallback_violation,
            output_selection='best_feasible_current_candidate' if valid_found else 'least_violation_current_candidate',
            search_center='current_frame_retargeting',previous_used_as_search_center=False,previous_evaluated_as_incumbent=bool(incumbent_ids),
            previous_incumbent_selected=chosen in incumbent_ids,previous_incumbent_kind=incumbent_ids.get(chosen),
            previous_incumbent_cost=incumbent_costs.get('previous'),temporal_candidate_costs=incumbent_costs,
            sampled_candidates=s.population*s.iterations,evaluations=int(sum(widths))+len(incumbent_ids),iterations=s.iterations,elite_count=int(s.population*s.elite_fraction),
            source_motion_samples_per_round=min(s.population-1,int(s.population*s.motion_sample_fraction)) if motion is not None else 0,
            coarse_verified_candidates=len(batch),coarse_verification_rejected=int((~coarse_valid).sum()),
            dense_verified_candidates=tested,dense_verification_rejected=rejected,hard_rejected=int((costs[ids]>=1.0e29).sum()),
            seed=s.seed,final_sigma_ratio=s.gamma,cuda_graph=True,collision_search='robokit_native_mesh_fitted_capsules',
            collision_output='native_mesh_0.5mm_endpoint_guard',validation_limit=256,
            output_selection_scope='verified_shortlist_of_current_frame_archive',infeasibility_proven=False,
            fallback_violation_units='native_self_depth_excess_mm',
            initial_cost=float(costs[0]),initial_feasible=None,initial_incumbents=0,empty_elite_restarts=0,feasibility_recovery_iterations=0,
            backend_timing_ms=dict(prepare=(prepared_at-began)*1000,sample=(sampled_at-prepared_at)*1000,
                endpoint_verification=(verified_at-sampled_at)*1000,total=(verified_at-began)*1000))
        return best,diagnostic

    def parts(self,initial,*,total=False):
        """Evaluate the same search cost at one output, without changing context."""
        device=str(self.owner.device)
        var=VarValues.from_var(VectorVar(wp.array(pack(initial)[None],dtype=float,device=device)),VarKey.ROBOT)
        self.task.precompute(var);parts,raw,depths=self.task.buffers[1]
        cost=wp.zeros(1,dtype=float,device=device)
        wp.launch(card_cost,1,[var.get(VarKey.ROBOT).x,self.task.states[1].T_world_link.reshape((1,self.task.robot.spec.num_links)),self.task.data,parts,raw,depths,cost],device=device)
        return float(cost.numpy()[0]) if total else dict(zip(PARTS,parts.numpy()[0].tolist()))

    def details(self,initial,c):
        """Report exactly the search material points, plus the native endpoint audit."""
        import torch
        from .contact_force import fit_coulomb_force
        o=self.owner;device=str(o.device);values=self.parts(initial)
        arrays=[wp.zeros((1,16),dtype=wp.vec3,device=device) for _ in range(4)]
        local,world,normals,bary=arrays;faces=wp.zeros((1,16),dtype=int,device=device)
        wp.launch(describe_material,16,[self.task.states[1].T_world_link.reshape((1,self.task.robot.spec.num_links)),self.task.data,local,world,normals,faces,bary],device=device)
        wp.synchronize_device(device)
        points={};ns={};locals_={};material={};support={};indices=faces.numpy()[0]
        for a in c['anchors']:
            region=a['region'];points[region]=wp.to_torch(world)[:,region];ns[region]=wp.to_torch(normals)[:,region];locals_[region]=wp.to_torch(local)[:,region]
            material[region]=(o.pad_face_ids[region][int(indices[region])][None],wp.to_torch(bary)[:,region])
            estimate=a.get('force_estimate');valid=estimate.get('direction_valid',False) if estimate is not None else a['force']>0
            if valid:
                vector=o.tensor(estimate['vector_relative']) if estimate is not None else o.tensor(a['direction'])*a['force']
                support[region]=fit_coulomb_force(vector,ns[region],o.settings.friction_coefficient,o.settings.normal_force_weight)
        parts={k:o.tensor([v]) for k,v in values.items()}
        for k in ['collision_mesh','contact_force_vector','contact_normal_force','contact_self_closure']:parts[k]=o.tensor([0.])
        groups={key:sum(parts[k] for k in names) for key,names in {
            'contact':['contact_position','contact_direction','contact_fix'],'style':['style_tip','style_normal','style_bone'],
            'smooth':['smooth_joint','smooth_translation','smooth_rotation','motion_joint'],'collision':['collision_self','collision_object']}.items()}
        t,r,j=initial;p,rr=o.frames(t[None],r[None],j[None]);mesh_depth,mesh_valid=o.mesh_collision.query(p,rr,dense=True)
        maximum=mesh_depth.amax((-1,-2));hard_mesh=(maximum>o.settings.self_penetration_mm)|~mesh_valid
        depths=wp.to_torch(self.task.buffers[1][2]);object_depth=depths[:,0]
        hard_object=(object_depth>o.settings.hard_penetration_mm)&(c['sdf'].kind=='tracked_mesh')
        return dict(parts=parts,groups=groups,points=points,normals=ns,local_points=locals_,material_points=material,force_support=support,
            hard=hard_mesh|hard_object,hard_mesh=hard_mesh,hard_object=hard_object,hard_palm=torch.zeros_like(hard_mesh),hard_pair=torch.zeros_like(hard_mesh),
            mesh_depth=mesh_depth,mesh_valid=mesh_valid,pair_gaps={},pair_points={},max_penetration_mm=torch.maximum(maximum,object_depth),
            object_max_penetration_mm=object_depth,min_palm_gap_mm=depths[:,1])
