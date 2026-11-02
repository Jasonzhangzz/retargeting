"""Actual native contact geometry for six-group loads and wrench support."""
import numpy as np
import torch
import mujoco
import mujoco_warp as mjw
import warp as wp
from .spider_seed import GROUPS, finger_group, palm_region
from .hand import query_path

Vec5=wp.types.vector(length=5,dtype=wp.float32)


@wp.kernel
def _pack_contacts(nacon:wp.array[int], worlds:wp.array[int], geoms:wp.array[wp.vec2i],
    position:wp.array[wp.vec3], frame:wp.array[wp.mat33], distance:wp.array[float],
    friction:wp.array(dtype=Vec5), forces:wp.array[wp.spatial_vector],
    object_geom:wp.array[int], geom_group:wp.array[int], geom_base:wp.array[int], geom_body:wp.array[int],
    palm_min:wp.array[wp.vec3],palm_max:wp.array[wp.vec3],palm_normals:wp.array[wp.vec3],
    palm_kind:wp.array[int],palm_long:wp.array[wp.vec3],palm_length:wp.array[float],
    region_count:wp.array[int],region_axes:wp.array2d[wp.vec3],region_tip:wp.array2d[int],
    region_long:wp.array[wp.vec3],region_origin:wp.array[wp.vec3],region_length:wp.array[float],
    xpos:wp.array2d[wp.vec3], xmat:wp.array2d[wp.mat33], palm:int,
    normal:wp.vec3, up:wp.vec3, side:wp.vec3, length:float, lateral:wp.vec2,
    capacity:int, count:wp.array[int], points:wp.array2d[wp.vec3],
    normals:wp.array2d[wp.vec3], mus:wp.array2d[float], groups:wp.array2d[int],
    loads:wp.array2d[float], pressures:wp.array2d[float], hand_bodies:wp.array2d[int],
    object_body:int,wrench:wp.array2d[float],depth:wp.array[float]):
    row=wp.tid()
    if row>=nacon[0]:return
    pair=geoms[row]; a=pair[0];b=pair[1]
    if a<0 or b<0:return
    hand=int(-1);sign=float(0.)
    if object_geom[a]!=0 and object_geom[b]==0:hand=b;sign=-1.
    elif object_geom[b]!=0 and object_geom[a]==0:hand=a;sign=1.
    if hand<0:return
    wp.atomic_max(depth,worlds[row],-distance[row])
    world=worlds[row]; group=geom_group[hand]
    inward=sign*frame[row][0]
    owner=geom_body[hand]
    inv=wp.transpose(xmat[world,owner]);p=inv*(position[row]-xpos[world,owner])
    n=inv*inward
    lo=palm_min[owner];hi=palm_max[owner]
    is_palm=(geom_base[hand]!=0 and wp.dot(n,palm_normals[owner])>.5 and
        p[0]>=lo[0] and p[0]<=hi[0] and p[1]>=lo[1] and p[1]<=hi[1] and p[2]>=lo[2] and p[2]<=hi[2])
    if palm_kind[owner]==3:
        fraction=wp.dot(p,palm_long[owner])/palm_length[owner]
        is_palm=is_palm and fraction>=-.15 and fraction<=1./3.
    if is_palm:group=5
    elif group==5:group=-1
    elif region_count[owner]>=0:
        allowed=int(0)
        for k in range(region_count[owner]):
            aligned=wp.dot(n,region_axes[owner,k])>.5
            if region_tip[owner,k]!=0:
                aligned=aligned and wp.dot(p-region_origin[owner],region_long[owner])>=.75*region_length[owner]
            if aligned:allowed=1
        if allowed==0:group=-1
    if group<0:return
    f=forces[row]
    force=sign*(wp.transpose(frame[row])*wp.vec3(f[0],f[1],f[2]))
    torque=wp.cross(position[row]-xpos[world,object_body],force)
    for axis in range(3):
        wp.atomic_add(wrench,world,axis,force[axis]);wp.atomic_add(wrench,world,axis+3,torque[axis])
    pressure=forces[row][0]
    if pressure>0. or not wp.isfinite(pressure):wp.atomic_add(loads,world,group,pressure)
    if distance[row]>.0002:return
    slot=wp.atomic_add(count,world,1)
    if slot>=capacity:return
    points[world,slot]=position[row];normals[world,slot]=inward
    pressures[world,slot]=pressure
    hand_bodies[world,slot]=geom_body[hand]
    mus[world,slot]=wp.min(friction[row][0],friction[row][1]);groups[world,slot]=group


class NativeContactGeometry:
    """No CPU reads or optimizer calls in sample(); one native row counts once."""
    def __init__(self,physics,seed=None,*,capacity=128):
        from .forceaware_validation import finger_body_labels
        self.physics=physics; self.capacity=capacity
        cpu=physics.cpu;self.region=palm_region(cpu);palm=self.region['palm']
        labels=finger_body_labels(cpu).copy();labels[palm]=5
        base=np.zeros(cpu.nbody,int)
        base[palm]=1
        for i in range(cpu.nbody):
            name=cpu.body(i).name
            if name.startswith('right_finger') and name.endswith(('link1','link2')):base[i]=1
            if i and i!=palm and cpu.body_jntnum[i]==0:
                labels[i]=labels[cpu.body_parentid[i]];base[i]=base[cpu.body_parentid[i]]
        obj=cpu.body('grasp_object').id
        self._object=wp.array((cpu.geom_bodyid==obj).astype(np.int32),dtype=int,device=physics.device)
        self._group=wp.array(labels[cpu.geom_bodyid].astype(np.int32),dtype=int,device=physics.device)
        self._base=wp.array(base[cpu.geom_bodyid].astype(np.int32),dtype=int,device=physics.device)
        owner=np.arange(cpu.nbody,dtype=np.int32)
        for body in range(1,cpu.nbody):
            if cpu.body_jntnum[body]==0 and body!=palm:owner[body]=owner[cpu.body_parentid[body]]
        self._body=wp.array(owner[cpu.geom_bodyid],dtype=int,device=physics.device)
        policy=anatomical_policy(cpu,self.region)
        lower,upper,normals=policy['lower'],policy['upper'],policy['normal']
        self._palm_min=wp.array(lower,dtype=wp.vec3,device=physics.device)
        self._palm_max=wp.array(upper,dtype=wp.vec3,device=physics.device)
        self._palm_normals=wp.array(normals,dtype=wp.vec3,device=physics.device)
        self._palm_kind=wp.array(policy['kind'],dtype=int,device=physics.device)
        self._palm_long=wp.array(policy['long'],dtype=wp.vec3,device=physics.device)
        self._palm_length=wp.array(policy['length'],dtype=float,device=physics.device)
        counts=np.full(cpu.nbody,-1,np.int32);axes=np.zeros((cpu.nbody,8,3));tips=np.zeros((cpu.nbody,8),np.int32)
        longs=np.zeros((cpu.nbody,3));origins=np.zeros_like(longs);lengths=np.ones(cpu.nbody)
        if seed is not None:
            counts[:]=0
            for body,records in ContactRequirements(cpu,seed).region_axes.items():
                counts[body]=min(len(records),8)
                for k,(axis,kind,frame,length) in enumerate(records[:8]):
                    axes[body,k]=axis;tips[body,k]=int(kind=='tip');longs[body]=frame['longitudinal'];origins[body]=frame['origin'];lengths[body]=length
        self._region_count=wp.array(counts,dtype=int,device=physics.device)
        self._region_axes=wp.array(axes,dtype=wp.vec3,device=physics.device)
        self._region_tip=wp.array(tips,dtype=int,device=physics.device)
        self._region_long=wp.array(longs,dtype=wp.vec3,device=physics.device)
        self._region_origin=wp.array(origins,dtype=wp.vec3,device=physics.device)
        self._region_length=wp.array(lengths,dtype=float,device=physics.device)
        W=physics.worlds;dev=physics.device
        self.count=wp.zeros(W,dtype=int,device=dev)
        self.points=wp.zeros((W,capacity),dtype=wp.vec3,device=dev)
        self.normals=wp.zeros((W,capacity),dtype=wp.vec3,device=dev)
        self.mu=wp.zeros((W,capacity),dtype=float,device=dev)
        self.groups=wp.zeros((W,capacity),dtype=int,device=dev)
        self.loads=wp.zeros((W,6),dtype=float,device=dev)
        self.pressures=wp.zeros((W,capacity),dtype=float,device=dev)
        self.hand_bodies=wp.zeros((W,capacity),dtype=int,device=dev)
        self.wrench=wp.zeros((W,6),dtype=float,device=dev)
        self.depth=wp.zeros(W,dtype=float,device=dev)
        self.ids=wp.array(np.arange(physics.data.naconmax),dtype=int,device=dev)
        self.force=wp.zeros(physics.data.naconmax,dtype=wp.spatial_vector,device=dev)
        from retarget.task.dex.cost import grasp_wrench_directions
        self.directions=torch.as_tensor(grasp_wrench_directions(),device=dev)
        self.slots=torch.arange(capacity,device=dev)[None]
        self.object_body=obj
        from retarget.env.geometry import ObjectMeshSurface
        mesh_points=ObjectMeshSurface.from_model(cpu,obj).triangles.reshape(-1,3)
        self.radius=max(float(np.linalg.norm(mesh_points,axis=1).max()),.01)

    def sample(self,*,forces=True):
        p=self.physics;d=p.data;r=self.region
        for a in (self.count,self.points,self.normals,self.mu,self.loads,self.pressures,self.wrench,self.depth):a.zero_()
        self.force.zero_()
        if forces:mjw.contact_force(p.model,d,self.ids,False,self.force)
        wp.launch(_pack_contacts,d.naconmax,inputs=[d.nacon,d.contact.worldid,d.contact.geom,d.contact.pos,
            d.contact.frame,d.contact.dist,d.contact.friction,self.force,self._object,self._group,self._base,self._body,
            self._palm_min,self._palm_max,self._palm_normals,
            self._palm_kind,self._palm_long,self._palm_length,self._region_count,self._region_axes,
            self._region_tip,self._region_long,self._region_origin,self._region_length,
            d.xpos,d.xmat,r['palm'],wp.vec3(*r['normal']),wp.vec3(*r['up']),wp.vec3(*r['side']),
            r['longitudinal'],wp.vec2(*r['lateral']),self.capacity,self.count,self.points,self.normals,self.mu,
            self.groups,self.loads,self.pressures,self.hand_bodies,self.object_body,self.wrench,self.depth],device=p.device)
        return wp.to_torch(self.loads)

    def wrench_cost(self,margin=.05,valid_mask=None):
        p=self.physics;obj=self.object_body
        point=wp.to_torch(self.points);normal=wp.to_torch(self.normals)
        inverse=p.xmat[:,obj].transpose(-1,-2)
        arm=torch.einsum('bij,bcj->bci',inverse,point-p.xpos[:,obj,None])
        normal=torch.einsum('bij,bcj->bci',inverse,normal)
        basis=torch.nn.functional.one_hot(normal.abs().argmin(-1),3).to(normal.dtype)
        tangent=torch.nn.functional.normalize(torch.cross(normal,basis,dim=-1),dim=-1)
        second=torch.cross(normal,tangent,dim=-1)
        tangents=torch.stack((tangent,second,-tangent,-second),dim=-2)
        mu=wp.to_torch(self.mu)
        force=(normal[:,:,None]+mu[:,:,None,None]*tangents)/torch.sqrt(1+mu.square())[:,:,None,None]
        torque=torch.cross(arm[:,:,None].expand_as(force),force,dim=-1)/self.radius
        rays=torch.cat((force,torque),dim=-1).flatten(1,2)
        active=self.slots<wp.to_torch(self.count)[:,None]
        if valid_mask is not None:active=active&valid_mask
        active=active.repeat_interleave(4,dim=1)
        support=(rays@self.directions.T).masked_fill(~active[:,:,None],-torch.inf).amax(1).clamp_min(0.)
        deficit=torch.relu(margin-support).square()
        # Force closure is a worst-direction property. Averaging 256 directions
        # lets a single unsupported separating direction almost disappear.
        loss=(deficit.amax(1)+.25*deficit.mean(1))/(margin*margin)
        return torch.where(wp.to_torch(self.count)<=self.capacity,loss,torch.inf)


class ContactRequirements:
    """CPU acceptance from real contact rows and immutable semantic skin pools."""
    def __init__(self,model,seed):
        self.model=model;self.seed=seed;self.palm=palm_region(model)
        self.policy=anatomical_policy(model,self.palm)
        self.region_axes={}
        for body in np.unique(seed['bodies']):
            indices=np.flatnonzero(seed['bodies']==body)
            if model.body(int(body)).name.startswith('right_finger'):
                from .semantic_patches import _target_pool
                axes=[]
                for i in indices:
                    _,segment,region=seed['labels'][i].split('.')
                    pool=_target_pool(model,int(body),segment,GROUPS[finger_group(model,body)])
                    frame=pool['frame']
                    axis=({'palmar':frame['palmar'],'dorsal':-frame['palmar'],
                        'lateral_positive':frame['lateral'],'lateral_negative':-frame['lateral'],
                        'tip':frame['longitudinal']})[region]
                    axes.append((axis,region,frame,pool['length']))
                self.region_axes[int(body)]=axes

    def owner(self,body):
        while self.model.body_jntnum[body]==0 and self.model.body(body).name!='right_palm_link':
            body=int(self.model.body_parentid[body])
            if body==0:break
        return body

    def classify(self,data,body,position,inward):
        body=self.owner(body)
        R=data.xmat[body].reshape(3,3)
        local=R.T@(position-data.xpos[body]);normal=R.T@inward
        p=self.policy
        is_palm=(p['kind'][body]>0 and normal@p['normal'][body]>.5 and
            np.all(local>=p['lower'][body]) and np.all(local<=p['upper'][body]))
        if p['kind'][body]==3:
            fraction=local@p['long'][body]/p['length'][body]
            is_palm=is_palm and -.15<=fraction<=1/3
        if is_palm:return 5
        # Patch locations guide optimization. Final finger contact means the
        # specified phalanx and anatomical normal region really touch, not that
        # a sparse annotation site remains within an arbitrary 3-mm radius.
        for axis,region,frame,length in self.region_axes.get(body,[]):
            if normal@axis>.5 and (region!='tip' or (local-frame['origin'])@frame['longitudinal']>=.75*length):
                return finger_group(self.model,body)
        return -1

    def observe(self,data,*,loaded_only=False):
        obj=self.model.body('grasp_object').id
        points,normals,groups,forces,frictions,bodies,hand_points,object_points=[],[],[],[],[],[],[],[]
        for i in range(data.ncon):
            c=data.contact[i];a,b=self.model.geom_bodyid[c.geom]
            if (a==obj)==(b==obj) or c.dist>.0002:continue
            sign=-1. if a==obj else 1.;hand=b if a==obj else a
            inward=sign*c.frame.reshape(3,3)[0]
            group=self.classify(data,int(hand),c.pos,inward)
            if group<0:continue
            wrench=np.zeros(6);mujoco.mj_contactForce(self.model,data,i,wrench)
            if loaded_only and wrench[0]<=1e-6:continue
            points.append(c.pos.copy());normals.append(inward);groups.append(group)
            forces.append(wrench[0]);frictions.append(min(c.friction[:2]))
            bodies.append(self.owner(int(hand)))
            hand_points.append(c.pos+.5*c.dist*inward)
            object_points.append(c.pos-.5*c.dist*inward)
        loads=np.zeros(6)
        if groups:np.add.at(loads,groups,np.maximum(forces,0))
        return dict(points=np.asarray(points).reshape(-1,3),normals=np.asarray(normals).reshape(-1,3),
            groups=np.asarray(groups,int),forces=np.asarray(forces),loads=loads,
            bodies=np.asarray(bodies,int),hand_points=np.asarray(hand_points).reshape(-1,3),
            object_points=np.asarray(object_points).reshape(-1,3),
            friction=min(frictions) if frictions else 0.)


def exact_closure(observation,*,center,radius):
    from .contactaware import require_upstream
    require_upstream()
    from contactaware.contact.force_margin import make_force_margin_solver
    if len(observation['points'])<3:
        return dict(success=False,wrench_rank=0,reserve=None,reason='fewer_than_three_real_contacts')
    result=make_force_margin_solver(center=np.asarray(center),radius=radius,friction=observation['friction'])(
        observation['points'],observation['normals'])
    passed=(result.status=='Solved' and result.wrench_rank==6 and result.reserve is not None and
            result.reserve>1e-4 and result.max_balance_error<1e-6)
    return dict(success=bool(passed),status=result.status,wrench_rank=result.wrench_rank,
        reserve=result.reserve,max_balance_error=result.max_balance_error,seconds=result.seconds)


def anatomical_policy(model,region):
    """Identical material-local palm predicates for GPU cost and CPU audit."""
    from .regions import belly_axis
    lower=np.full((model.nbody,3),1e6);upper=-lower.copy();normal=np.zeros((model.nbody,3))
    kind=np.zeros(model.nbody,np.int32);long=np.zeros_like(normal);length=np.ones(model.nbody)
    for body in range(model.nbody):
        name=model.body(body).name
        if body==region['palm']:
            ids=region['bodies']==body;points=region['local'][ids]
            lower[body]=points.min(0)-.003;upper[body]=points.max(0)+.003
            normal[body]=region['normal'];kind[body]=1
        elif name.startswith('right_finger') and name.endswith(('link1','link2')):
            with np.load(query_path(model,body)) as raw:
                points=raw['local_pos']
                if 'source_body' in raw:points=points[raw['source_body'].astype(str)==name]
            lower[body]=points.min(0)-.001;upper[body]=points.max(0)+.001
            normal[body]=belly_axis(model,body);kind[body]=2 if name.endswith('link1') else 3
            if kind[body]==3:
                endpoint=model.body(name[:-1]+'3').pos
                length[body]=np.linalg.norm(endpoint);long[body]=endpoint/length[body]
    return dict(lower=lower,upper=upper,normal=normal,kind=kind,long=long,length=length)
