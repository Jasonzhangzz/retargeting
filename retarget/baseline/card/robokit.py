"""CARD energies as a Robokit cost-only task, with captured elite Gaussian CEM.

The search uses Robokit's fitted capsules and the original collision soft costs.
Native mesh depths are reported only; they never change the selected solution.
"""
from dataclasses import dataclass
import time
import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation
from robokit.opt import PopulationSolver, PopulationSolverConfig, Stage, VarValues, VectorVar
from robokit.opt.var_values import VarKey
from robokit.opt.sampler import GaussianSamplerConfig, GaussianSampler, _compute_order_kernel
from robokit.terms.task import Task
from robokit.utils.warp_utils import wp_vec7
from robokit.lie.se3_kernels import se3_apply_func, se3_inverse_func
from retarget.baseline.card.geometry import source_wrist


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


@wp.func
def rotvec_near(vector: wp.vec3, reference: wp.vec3):
    length=wp.length(vector);ref_length=wp.length(reference)
    axis=wp.vec3(0.0,0.0,0.0)
    if length>1.0e-7:axis=vector/length
    elif ref_length>1.0e-12:axis=reference/ref_length
    turns=wp.round(wp.dot(reference-vector,axis)/(2.0*wp.pi))
    return vector+2.0*wp.pi*turns*axis


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
            else:parts[i,0]=wp.inf
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
        parts[i,8]+=w[8]*squared(rotvec_near(rv,old)-old)
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
    if not wp.isfinite(total):total=wp.inf
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
def ball_projection(x: wp.array2d(dtype=float), noise: wp.array2d(dtype=float), lower: wp.array2d(dtype=float), upper: wp.array2d(dtype=float), radius: float):
    i=wp.tid()
    # Unbounded translation on the first frame; bounds are device buffers and
    # can change between solves without recapturing the graph.
    if wp.isfinite(lower[0,0]):
        center=wp.vec3((lower[0,0]+upper[0,0])*0.5,(lower[0,1]+upper[0,1])*0.5,(lower[0,2]+upper[0,2])*0.5)
        delta=wp.vec3(x[i,0],x[i,1],x[i,2])-center
        if wp.length(delta)>radius:
            change=delta*(radius/wp.length(delta)-1.0)
            for j in range(3):x[i,j]+=change[j];noise[i,j]+=change[j]


@wp.kernel(enable_backward=False)
def gaussian_draws(seed: int, epoch: wp.array(dtype=int), salt: int, draws: int,
                   active: wp.array(dtype=float), std: wp.array2d(dtype=float), noise: wp.array2d(dtype=float)):
    i,j=wp.tid();value=float(0.0)
    if active[j]!=0.0:
        state=wp.rand_init(seed+epoch[0]*7919,(salt*1000003+i)*noise.shape[1]+j)
        value=std[i//draws,j]*wp.randn(state)
    noise[i,j]=value


@wp.kernel(enable_backward=False)
def finite_costs(source: wp.array(dtype=float),target: wp.array(dtype=float)):
    i=wp.tid();target[i]=wp.where(wp.isfinite(source[i]),source[i],wp.inf)


@wp.kernel(enable_backward=False)
def finite_elite_weights(keys: wp.array(dtype=float),order: wp.array(dtype=int),draws: int,elite: int,
                         eta: float,weights: wp.array2d(dtype=float),best: wp.array(dtype=int),empty: wp.array(dtype=int)):
    b=wp.tid();start=b*draws;count=int(0);mean=float(0.0)
    for j in range(draws):weights[b,j]=0.0
    for j in range(elite):
        if wp.isfinite(keys[start+j]):count+=1;mean+=keys[start+j]
    best[b]=order[start]
    if count==0:empty[b]+=1
    if count>0:
        mean=mean/float(count);variance=float(0.0)
        for j in range(count):variance+=(keys[start+j]-mean)**2.0
        scale=wp.max(wp.sqrt(variance/float(count)),1.0e-12)*eta
        total=float(0.0)
        for j in range(count):
            value=wp.exp(-(keys[start+j]-keys[start])/scale)
            weights[b,order[start+j]-start]=value;total+=value
        for j in range(count):weights[b,order[start+j]-start]/=total


@wp.kernel(enable_backward=False)
def elite_step(weights: wp.array2d(dtype=float),noise: wp.array2d(dtype=float),draws: int,
               decay: float,std: wp.array2d(dtype=float),delta: wp.array2d(dtype=float)):
    b,d=wp.tid();step=float(0.0)
    for j in range(draws):step+=weights[b,j]*noise[b*draws+j,d]
    delta[b,d]=step
    std[b,d]=wp.max(std[b,d]*decay,1.0e-12)


class CardSampler(GaussianSampler):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.empty_iterations=wp.zeros(self.instances,dtype=int,device=self._device)

    def reset(self):
        super().reset()
        self.empty_iterations.zero_()

    def fill(self,out_tangent,iteration):
        wp.launch(gaussian_draws,out_tangent.shape,
            [self.seed,self.epoch,iteration,self.num_draws,self.active_dof_mask,self.std,out_tangent],device=self._device)
        self.noise=out_tangent

    def _move(self,scores,out_delta):
        instances=scores.shape[0]//self.num_draws
        wp.launch(finite_costs,scores.shape[0],[scores,self._elite_costs],device=self._device)
        wp.launch(_compute_order_kernel,scores.shape[0],[self._elite_costs,self.num_draws,self._keys,self._order],device=self._device)
        wp.launch(finite_elite_weights,instances,
            [self._keys,self._order,self.num_draws,max(1,int(self.elite_ratio*self.num_draws)),
             self.temperature,self.weights,self.best_indices,self.empty_iterations],device=self._device)
        wp.launch(elite_step,(instances,self.tangent_dim),
            [self.weights,self.noise,self.num_draws,self.noise_decay,self.std,out_delta],device=self._device)

    def sample(self,var,scores):
        result=super().sample(var,scores)
        if self.lower is not None:
            wp.launch(ball_projection,result.batch_size,
                [result.get(self.var_key).x,self.noise,self.lower,self.upper,self.radius],device=self._device)
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
        force_max=max((a['force'] for a in c['anchors']),default=0.)
        for a in c['anchors']:
            r=a['region'];active[r]=1;target[r]=a['position_m'];direction[r]=a['direction'];rho[r]=np.clip(1+.5*a['force']/force_max if force_max else 1.,1.,1.5)
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
    if rotation_reference is not None:
        ref_length=np.linalg.norm(rotation_reference)
        axis=vector/angle if angle>1e-7 else rotation_reference/max(ref_length,1e-12)
        vector=vector+2*np.pi*np.round(np.dot(rotation_reference-vector,axis)/(2*np.pi))*axis
    return np.r_[t,vector,j].astype(np.float32)


class CardPopulationSolver(PopulationSolver):
    def _sample(self,cur,scores):
        sampler=self.samplers[0]
        if scores is not None and sampler._round==self.config.num_iters:
            # Consume the final scores without drawing an unscored 17th cloud.
            centers=sampler._centers
            sampler._move(scores,sampler._step)
            centers.integrate(sampler._step,out=centers)
            self._handed=centers
            return centers
        return super()._sample(cur,scores)


class RobokitCardSolver:
    def __init__(self,owner):
        self.owner=owner;self.settings=owner.settings;s=self.settings;device=str(owner.device)
        self.task=CardCostTask(owner);self.var=VarValues.from_var(VectorVar(wp.zeros((1,26),dtype=float,device=device)),VarKey.ROBOT)
        config=CardSamplerConfig(num_draws=s.population,elite_ratio=s.elite_fraction,mean_elites=True,temperature=s.temperature,
            init_std=1.,min_std=1.0e-9,gamma=0.,noise_decay=s.gamma**(1/s.iterations),sample_mode='mean',seed=s.seed)
        self.solver=CardPopulationSolver([self.task],PopulationSolverConfig(stages=[Stage(s.iterations)],samplers=[config],cuda_graph_mode='full',use_early_stopping=False),device)
        # Preallocate both widths before capture. Context buffers are filled first.
        self.lower=wp.zeros((1,26),dtype=float,device=device);self.upper=wp.zeros((1,26),dtype=float,device=device)
        self.rotation_reference=None
        self.initialized=False

    def solve(self,initial,c,*,elite_archive=None):
        began=time.perf_counter();o=self.owner;s=self.settings;device=str(o.device)
        previous=pack(c['previous']) if c['previous'] is not None else None
        self.rotation_reference=previous[3:6] if previous is not None else None
        current=pack(initial,self.rotation_reference)
        center=(current+previous)*.5 if previous is not None else current.copy()
        lo=np.r_[[-np.inf]*6,o.lower_np[6:]].astype(np.float32)
        hi=np.r_[[np.inf]*6,o.upper_np[6:]].astype(np.float32)
        if previous is not None:
            lo[:3]=previous[:3]-s.translation_step_m;hi[:3]=previous[:3]+s.translation_step_m
            lo[6:]=np.maximum(lo[6:],previous[6:]-s.joint_step_rad)
            hi[6:]=np.minimum(hi[6:],previous[6:]+s.joint_step_rad)
        def project_value(value):
            value=np.clip(value,lo,hi)
            if previous is not None:
                delta=value[:3]-previous[:3];length=np.linalg.norm(delta)
                if length>s.translation_step_m:value[:3]=previous[:3]+delta*(s.translation_step_m/length)
            return value
        center=project_value(center);self.task.update(c)
        wp.copy(self.var.get(VarKey.ROBOT).x,wp.array(center[None],dtype=float,device=device))
        wp.copy(self.lower,wp.array(lo[None],dtype=float,device=device));wp.copy(self.upper,wp.array(hi[None],dtype=float,device=device))
        if not self.initialized:
            original=self.solver.config.cuda_graph_mode;self.solver.config.cuda_graph_mode='none';self.solver.setup(self.var)
            sampler=self.solver.samplers[0];sampler.lower=self.lower;sampler.upper=self.upper;sampler.radius=s.translation_step_m
            self.solver.config.cuda_graph_mode=original
            std=np.array([s.translation_sigma_m]*3+[s.rotation_sigma_rad]*3+[s.joint_sigma_rad]*20,np.float32)
            wp.copy(sampler.std_reset,wp.array(std[None],dtype=float,device=device))
            self.solver._warmup_and_capture();self.initialized=True
        self.task.counter.zero_();self.task.archive_width.zero_();self.task.archive_cost.fill_(wp.inf)
        wp.synchronize_device(device);prepared_at=time.perf_counter()
        self.solver.solve(self.var);wp.synchronize_device(device);sampled_at=time.perf_counter()
        widths=self.task.archive_width.numpy();costs=self.task.archive_cost.numpy();xs=self.task.archive_x.numpy()
        ids=np.concatenate([np.arange(i*s.population,i*s.population+int(n)) for i,n in enumerate(widths) if n])
        # Score the same deterministic seed set as Torch, in addition to all
        # sampled candidates and the final updated mean in the archive.
        seed_costs=[]
        for value in [center]+([previous] if previous is not None else []):
            idx=len(costs);cost=self._parts_vector(value,total=True);seed_costs.append(cost)
            xs=np.vstack((xs,value));costs=np.r_[costs,cost];ids=np.r_[ids,idx]
        finite=ids[np.isfinite(costs[ids])&np.isfinite(xs[ids]).all(1)]
        if elite_archive is not None:
            from .math import retained_elites
            elite_archive.update(retained_elites([o.tensor(xs[ids])],[o.tensor(costs[ids])],
                max(1,int(s.population*s.elite_fraction))))
        feasible=bool(len(finite));chosen=int(finite[np.argmin(costs[finite])]) if feasible else None
        value=xs[chosen] if feasible else project_value(current)
        best=(o.tensor(value[:3]),o.tensor(Rotation.from_rotvec(value[3:6]).as_matrix()),o.tensor(value[6:]))
        wp.synchronize_device(device);selected_at=time.perf_counter()
        prev_cost=seed_costs[1] if previous is not None and np.isfinite(seed_costs[1]) else None
        diagnostic=dict(backend='robokit_population_solver_warp_card',feasible=feasible,
            best_cost=float(costs[chosen]) if feasible else None,finite_candidate_found=feasible,fallback_cost=None,fallback_violation=None,
            output_selection='best_feasible_current_candidate' if feasible else 'current_initializer_no_finite_objective',
            search_center='half_previous_half_initializer' if previous is not None else 'current_frame_retargeting',
            previous_used_as_search_center=previous is not None,previous_evaluated_as_incumbent=previous is not None,
            previous_incumbent_selected=bool(previous is not None and chosen==ids[-1]),
            previous_incumbent_kind='previous' if previous is not None and chosen==ids[-1] else None,
            previous_incumbent_cost=prev_cost,temporal_candidate_costs={'previous':prev_cost} if previous is not None else {},
            sampled_candidates=s.population*s.iterations,evaluations=int(sum(widths))+len(seed_costs),
            iterations=s.iterations,elite_count=max(1,int(s.population*s.elite_fraction)),source_motion_samples_per_round=0,
            coarse_verified_candidates=0,coarse_verification_rejected=0,dense_verified_candidates=0,dense_verification_rejected=0,
            hard_rejected=int((~np.isfinite(costs[ids])).sum()),seed=s.seed,final_sigma_ratio=s.gamma,
            last_sample_sigma_ratio=s.gamma**((s.iterations-1)/s.iterations),cuda_graph=True,
            collision_search='robokit_native_mesh_fitted_capsules',collision_output='soft_collision_costs_only',
            validation_limit=None,output_selection_scope='all_scored_candidates',infeasibility_proven=False,
            initial_cost=seed_costs[0] if np.isfinite(seed_costs[0]) else None,initial_feasible=bool(np.isfinite(seed_costs[0])),
            initial_incumbents=0,empty_elite_restarts=0,empty_elite_iterations=int(self.solver.samplers[0].empty_iterations.numpy()[0]) if self.solver.samplers else 0,feasibility_recovery_iterations=0,
            backend_timing_ms=dict(prepare=(prepared_at-began)*1000,sample=(sampled_at-prepared_at)*1000,
                endpoint_verification=(selected_at-sampled_at)*1000,total=(selected_at-began)*1000))
        return best,diagnostic

    def parts(self,initial,*,total=False):
        """Evaluate the same search cost at one output, without changing context."""
        return self._parts_vector(pack(initial,self.rotation_reference),total=total)

    def _parts_vector(self,value,*,total=False):
        device=str(self.owner.device)
        var=VarValues.from_var(VectorVar(wp.array(value[None],dtype=float,device=device)),VarKey.ROBOT)
        self.task.precompute(var);parts,raw,depths=self.task.buffers[1]
        cost=wp.zeros(1,dtype=float,device=device)
        wp.launch(card_cost,1,[var.get(VarKey.ROBOT).x,self.task.states[1].T_world_link.reshape((1,self.task.robot.spec.num_links)),self.task.data,parts,raw,depths,cost],device=device)
        return float(cost.numpy()[0]) if total else dict(zip(PARTS,parts.numpy()[0].tolist()))

    def details(self,initial,c):
        """Report exactly the search material points, plus the native endpoint audit."""
        import torch
        from retarget.baseline.card.contact_force import fit_coulomb_force
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
        hard_object=torch.zeros_like(hard_mesh)
        hard_palm=torch.zeros_like(hard_mesh)
        return dict(parts=parts,groups=groups,points=points,normals=ns,local_points=locals_,material_points=material,force_support=support,
            hard=hard_palm|hard_object,hard_mesh=hard_mesh,hard_object=hard_object,hard_palm=hard_palm,hard_pair=torch.zeros_like(hard_mesh),
            mesh_depth=mesh_depth,mesh_valid=mesh_valid,pair_gaps={},pair_points={},max_penetration_mm=torch.maximum(maximum,object_depth),
            object_max_penetration_mm=object_depth,min_palm_gap_mm=depths[:,1])
