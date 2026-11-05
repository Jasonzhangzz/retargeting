"""One batched MJWarp collision pass for rigid MANO proxies and objects.

All object/scale assets are compiled once. Per-world geom_dataid and compiled
mesh transforms select the correct object without a CPU loop over candidates.
MuJoCo mesh collision uses convex hulls; triangle-mesh distance diagnostics
remain separate and must not be confused with this collision representation.
"""
import xml.etree.ElementTree as ET
import numpy as np
import torch
import warp as wp
import mujoco
import mujoco_warp as mjw
from .mano_collision_geometry import ManoCollisionGeometry
from .collision import _depths
from .comfree_rollout import matrix_quaternion


@wp.kernel(enable_backward=False)
def pack_contacts(count:wp.array[int],world:wp.array[int],geom:wp.array[wp.vec2i],
                  pos:wp.array[wp.vec3],frame:wp.array[wp.mat33],distance:wp.array[float],
                  object_geom:int,capacity:int,n:wp.array[int],overflow:wp.array[int],
                  hand:wp.array2d[wp.vec3],obj:wp.array2d[wp.vec3],normal:wp.array2d[wp.vec3],
                  gap:wp.array2d[float],eligible:wp.array2d[wp.bool]):
    i=wp.tid()
    if i>=count[0]:return
    pair=geom[i]
    if pair[0]!=object_geom and pair[1]!=object_geom:return
    w=world[i];slot=wp.atomic_add(n,w,1)
    if slot>=capacity:
        overflow[w]=1
        return
    direction=wp.vec3(frame[i][0,0],frame[i][0,1],frame[i][0,2])
    sign=float(1.)
    if pair[1]==object_geom:sign=-1.
    outward=sign*direction
    normal[w,slot]=outward;gap[w,slot]=distance[i]
    obj[w,slot]=pos[i]-.5*distance[i]*outward
    hand[w,slot]=pos[i]+.5*distance[i]*outward
    eligible[w,slot]=True


class ManoContactCollision:
    def __init__(self,decoder,surfaces,worlds,capacity=64):
        self.device=str(decoder.device);self.worlds=worlds;self.capacity=capacity
        # Reuse the exact existing anatomical rigid hull construction and exclusions.
        # Obtain source XML without allocating a second set of large GPU buffers.
        if hasattr(decoder,'collision_geometry'):
            xml=decoder.collision_geometry.xml
        else:
            # The CPU constructor builds the same hulls and does not run simulation.
            class CpuGeometry:
                pass
            proxy=CpuGeometry()
            proxy.device=torch.device('cpu');proxy.rest=decoder.rest.cpu();proxy.joints_rest=decoder.joints_rest.cpu()
            proxy.base=CpuGeometry();proxy.base.weights=decoder.base.weights.cpu();proxy.base.parents=decoder.base.parents
            proxy.ordered_joint_ids=decoder.ordered_joint_ids.cpu()
            xml=ManoCollisionGeometry(proxy,1).xml
        root=ET.fromstring(xml)
        # Native zero-margin multi-point collision supports actual contact patches.
        root.find('option/flag').set('multiccd','enable')
        asset=root.find('asset');world=root.find('worldbody')
        keys=list(dict.fromkeys(surfaces.object_keys));lookup={k:i for i,k in enumerate(keys)}
        self.row_mesh=torch.tensor([lookup[k] for k in surfaces.object_keys],device=self.device,dtype=torch.long)
        for i,(idx,scale) in enumerate(keys):
            geo=surfaces.geometries[idx]
            ET.SubElement(asset,'mesh',name=f'object_{i}',vertex=' '.join(map(str,(np.asarray(geo['vertices'],float)*scale).ravel())),
                          face=' '.join(map(str,np.asarray(geo['faces']).ravel())))
        body=ET.SubElement(world,'body',name='rollout_object');ET.SubElement(body,'freejoint')
        ET.SubElement(body,'geom',type='mesh',mesh='object_0',mass='.1',condim='3',margin='0')
        self.object_geom=16
        self.cpu=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
        # Compiled same-frame shortcuts describe object_0 only. Other per-world
        # meshes can have different COM/principal-axis geom transforms.
        self.cpu.geom_sameframe[self.object_geom]=0
        if any(self.cpu.mesh(f'object_{i}').id != 16+i for i in range(len(keys))):
            raise ValueError('Unexpected compiled object mesh identities')
        self.joint_order=torch.argsort(decoder.ordered_joint_ids)[:16]
        with self.scope():
            self.model=mjw.put_model(self.cpu)
            self.data=mjw.make_data(self.cpu,nworld=worlds,nconmax=capacity+256,nccdmax=capacity+256,njmax=1)
            self.n=wp.zeros(worlds,dtype=int,device=self.device)
            self.pack_overflow=wp.zeros(worlds,dtype=int,device=self.device)
            self.depth=wp.zeros((worlds,2),dtype=float,device=self.device)
            self.flags=wp.array([0]*16+[1],dtype=int,device=self.device)
            self.hand=wp.zeros((worlds,capacity),dtype=wp.vec3,device=self.device)
            self.obj=wp.zeros_like(self.hand);self.normal=wp.zeros_like(self.hand)
            self.gap=wp.zeros((worlds,capacity),dtype=float,device=self.device)
            self.eligible=wp.zeros((worlds,capacity),dtype=wp.bool,device=self.device)
        tables={k:[] for k in ['geom_dataid','geom_pos','geom_quat','geom_size','geom_aabb','geom_rbound']}
        for i in range(len(keys)):
            mid=16+i;v=self.cpu.mesh_vert[self.cpu.mesh_vertadr[mid]:self.cpu.mesh_vertadr[mid]+self.cpu.mesh_vertnum[mid]]
            low,high=v.min(0),v.max(0);extent=(high-low)/2;center=(high+low)/2;size=np.maximum(abs(low),abs(high))
            for k,value in dict(geom_dataid=mid,geom_pos=self.cpu.mesh_pos[mid],geom_quat=self.cpu.mesh_quat[mid],
                geom_size=size,geom_aabb=np.stack((center,extent)),geom_rbound=np.linalg.norm(size)).items():tables[k].append(value)
        self.tables={}
        for k,values in tables.items():
            native=getattr(self.model,k);tensor=wp.to_torch(native)
            own=tensor.expand(worlds,*tensor.shape[1:]).clone()
            setattr(self.model,k,wp.from_torch(own,dtype=native.dtype))
            self.tables[k]=torch.as_tensor(np.array(values),device=self.device,dtype=own.dtype)
        self.object_pose=torch.cat((surfaces.origin,matrix_quaternion(surfaces.rotation)),-1)

    def scope(self):
        from contextlib import ExitStack
        s=ExitStack();s.enter_context(wp.ScopedDevice(self.device))
        s.enter_context(wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(self.device))))
        return s

    def query(self,joints,frames,rows):
        b=self.worlds
        if len(joints)!=b:raise ValueError('Pad candidate collision batches to worlds')
        q=torch.cat((joints[:,self.joint_order],matrix_quaternion(frames)),-1).flatten(1)
        q=torch.cat((q,self.object_pose[rows]),-1)
        with self.scope():
            for k,table in self.tables.items():
                wp.to_torch(getattr(self.model,k))[:,self.object_geom].copy_(table[self.row_mesh[rows]])
            wp.to_torch(self.data.qpos).copy_(q)
            self.data.overflow.zero_()
            mjw.kinematics(self.model,self.data)
            mjw.collision(self.model,self.data)
            self.depth.zero_();self.n.zero_();self.pack_overflow.zero_()
            self.hand.zero_();self.obj.zero_();self.normal.fill_(wp.vec3(1.,0.,0.));self.gap.fill_(1000.);self.eligible.zero_()
            wp.launch(_depths,self.data.naconmax,inputs=[self.data.nacon,self.data.contact.worldid,self.data.contact.geom,
                self.data.contact.dist,self.flags,0.,self.depth],device=self.device)
            wp.launch(pack_contacts,self.data.naconmax,inputs=[self.data.nacon,self.data.contact.worldid,self.data.contact.geom,
                self.data.contact.pos,self.data.contact.frame,self.data.contact.dist,self.object_geom,self.capacity,
                self.n,self.pack_overflow,self.hand,self.obj,self.normal,self.gap,self.eligible],device=self.device)
        overflow=(wp.to_torch(self.data.overflow)!=0)|(wp.to_torch(self.pack_overflow)!=0)
        return dict(hand=wp.to_torch(self.hand),object=wp.to_torch(self.obj),normal=wp.to_torch(self.normal),
                    gap=wp.to_torch(self.gap),eligible=wp.to_torch(self.eligible),count=wp.to_torch(self.n),
                    depth=wp.to_torch(self.depth),overflow=overflow)
