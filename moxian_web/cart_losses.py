"""Contact-first CART losses through exact native rigid-body FK.

Closest surface witnesses are refreshed in detached geometry, then their
positions and normals are differentiated through both moving bodies. No
paired robot labels, learned FK, fixed material correspondence or IK teacher.
"""
import numpy as np
import torch
import trimesh
import warp as wp
from .cart_protocol import FINGERS,stage
from retarget.baseline.geort.adapter import RobotAdapter
from .wuji_surface import WujiSurface
from retarget.env.grasp.pose_gpu import ModelFK

WEIGHTS=dict(penetration=100.,contact=20.,normal=5.,wrap=2.,point=1.,shape=1.,motion=.1,pose=.01)

@wp.kernel(enable_backward=False)
def _signed_witness(meshes:wp.array(dtype=wp.uint64),points:wp.array(dtype=wp.vec3),owners:wp.array(dtype=int),
    allowed:wp.array2d(dtype=int),collar:wp.array(dtype=float),
    positions:wp.array2d(dtype=wp.vec3),rotations:wp.array2d(dtype=wp.mat33),
    hits:wp.array3d(dtype=wp.vec3),normals:wp.array3d(dtype=wp.vec3),valid:wp.array3d(dtype=int)):
    b,s,t=wp.tid();o=owners[s]
    if allowed[o,t]!=0:
        world=positions[b,o]+rotations[b,o]*points[s]
        local=wp.transpose(rotations[b,t])*(world-positions[b,t])
        if not (o==0 and collar[t]>0. and wp.length(local)<=collar[t]) and not (t==0 and collar[o]>0. and wp.length(points[s])<=collar[o]):
            q=wp.mesh_query_point_sign_winding_number(meshes[t],local,1.)
            if q.result:
                hit=wp.mesh_eval_position(meshes[t],q.face,q.u,q.v)
                d=local-hit;n=wp.mesh_eval_face_normal(meshes[t],q.face)
                if wp.length(d)>1.e-8:n=q.sign*wp.normalize(d)
                hits[b,s,t]=hit;normals[b,s,t]=n;valid[b,s,t]=1

class NativeHand:
    def __init__(self,device='cpu'):
        self.device=torch.device(device);self.robot=RobotAdapter();self.surface=WujiSurface()
        self.fk=ModelFK(self.robot.model,device)
        self.low=self.tensor(self.robot.lower);self.high=self.tensor(self.robot.upper)
        self.basis=self.tensor(self.robot.basis)
        self.ids=self.robot.body_ids;self.regions={}
        from retarget.baseline.card.collision import WujiMeshCollision
        self.collision=WujiMeshCollision(self.surface,str(device))
        for f,name in enumerate(FINGERS):
            for part,k in (('proximal',1 if f==0 else 0),('intermediate',1),('distal',2)):
                if f==0 and part=='intermediate':continue
                self._region(name+'_'+part,2+4*f+k)
        for f,name in enumerate(FINGERS[1:],1):self._region('palm_mcp_'+name,0,f)
        self.local=torch.as_tensor(self.surface.local,device=device,dtype=torch.float32)
        self.owners=torch.as_tensor(self.surface.owners,device=device,dtype=torch.long)
        self.audit_points=wp.to_torch(self.collision.samples[False][0]);self.audit_owners=wp.to_torch(self.collision.samples[False][1]).long()
    def tensor(self,x):return torch.as_tensor(np.asarray(x),dtype=torch.float32,device=self.device)
    def _region(self,name,owner,palm_finger=None):
        data=self.surface.contact_surface(owner);tri=np.asarray(data['triangles'])
        if palm_finger is not None:
            body=self.ids[2+4*palm_finger]
            anchor=(self.surface.neutral.xpos[body]-self.surface.neutral.xpos[self.ids[0]])@self.surface.neutral.xmat[self.ids[0]].reshape(3,3)
            tri=tri[np.argsort(np.linalg.norm(tri.mean(1)-anchor,axis=1))[:48]]
        if not len(tri):raise ValueError('Wuji2 缺少语义 patch '+name)
        mesh=trimesh.Trimesh(tri.reshape(-1,3),np.arange(len(tri)*3).reshape(-1,3),process=False)
        # Fixed material samples cover each semantic surface; the target
        # witness is continuous on the complete native triangle union.
        ids=np.linspace(0,len(tri)-1,min(64,len(tri))).astype(int)
        self.regions[name]=dict(owner=owner,mesh=mesh,points=self.tensor(tri[ids].mean(1)),
                                normals=self.tensor(mesh.face_normals[ids]))
    def forward(self,q):
        p,r=self.fk.frames(torch.cat((torch.zeros((*q.shape[:-1],6),device=q.device),q),-1))
        return p[:,self.ids],r[:,self.ids]
    def samples(self,p,r,region):
        d=self.regions[region];o=d['owner']
        return d['points']@r[o].T+p[o],d['normals']@r[o].T
    def nearest(self,points,normals,p,r,region):
        d=self.regions[region];o=d['owner'];local=(points-p[o])@r[o]
        hit,_,faces=trimesh.proximity.closest_point(d['mesh'],local.detach().cpu().numpy())
        target=self.tensor(hit)@r[o].T+p[o];normal=self.tensor(d['mesh'].face_normals[faces])@r[o].T
        distance=(points-target).square().sum(-1)
        angle=(1+(normals*normal).sum(-1)).clamp_min(0)
        score=distance.detach()/.002**2+angle.detach()/(1-np.cos(np.deg2rad(15)))
        i=score.argmin()
        return distance[i],angle[i],points[i],target[i],normals[i],normal[i]
    def pair(self,p,r,a,b):
        pa,na=self.samples(p,r,a);pb,nb=self.samples(p,r,b)
        one=self.nearest(pa,na,p,r,b);two=self.nearest(pb,nb,p,r,a)
        if float(one[0].detach())<=float(two[0].detach()):return one
        return two[0],two[1],two[3],two[2],two[5],two[4]
    def self_penetration(self,p,r):
        b=len(p);ns=len(self.audit_points);shape=(b,ns,21)
        hit=wp.zeros(shape,dtype=wp.vec3,device=str(self.device));normal=wp.zeros(shape,dtype=wp.vec3,device=str(self.device));valid=wp.zeros(shape,dtype=int,device=str(self.device))
        wp.launch(_signed_witness,shape,inputs=[self.collision.mesh_ids,*self.collision.samples[False],self.collision.allowed,
            self.collision.collar,wp.from_torch(p.detach().contiguous(),dtype=wp.vec3),wp.from_torch(r.detach().contiguous(),dtype=wp.mat33),hit,normal,valid],device=str(self.device))
        wp.synchronize_device(str(self.device))
        world=(r[:,self.audit_owners]@self.audit_points[...,None]).squeeze(-1)+p[:,self.audit_owners]
        hp=(r[:,None]@wp.to_torch(hit)[...,None]).squeeze(-1)+p[:,None]
        hn=(r[:,None]@wp.to_torch(normal)[...,None]).squeeze(-1)
        gap=((world[:,:,None]-hp)*hn).sum(-1)
        depth=torch.where(wp.to_torch(valid).bool(),(-gap).clamp_min(0),torch.zeros_like(gap))
        return (depth/.0005).square().amax((1,2)).mean(),depth.amax((1,2))
    def vertices(self,p,r):
        return (r[self.owners]@self.local[...,None]).squeeze(-1)+p[self.owners]
    def audit(self,q,dense=True):
        full=torch.cat((q.new_zeros((1,6)),q[None]),1)
        p,r=self.fk.frames(full)
        depth,valid=self.collision.query(p,r,dense=dense)
        return self.collision.report(depth[0],valid[0])

def internal_angles(j):
    v=j[1:].reshape(5,4,3).diff(dim=1)
    v=torch.nn.functional.normalize(v,dim=-1,eps=1e-8)
    return torch.acos((v[:,:-1]*v[:,1:]).sum(-1).clamp(-1+1e-6,1-1e-6))

def contact_losses(hand,q,translation,rotation,rows):
    p,r=hand.forward(q);zero=q.sum()*0
    parts={k:zero for k in WEIGHTS};metrics=[]
    self_loss,self_depth=hand.self_penetration(p,r);parts['penetration']=self_loss
    basis=hand.basis;origin=hand.robot.model.body(hand.ids[0]).pos
    # All object geometry and source features are in the canonical palm frame.
    canonical_p=(p-p[:,0:1])@basis;canonical_r=basis.T@r
    for b,row in enumerate(rows):
        spec=stage(row['stage']);target=row['target'];pb=canonical_p[b];rb=canonical_r[b]
        active=row['phase'] in ('closing','hold')
        gaps=[];normal_errors=[];directions=[];source_directions=[];contacts=[];object_depth_mm=0.;object_intersection=False
        if target.get('contact_mode')=='hand':
            for pair in target['pairs']:
                a,c=pair['regions'];distance,normal,pa,pc,na,nc=hand.pair(pb,rb,a,c)
                gap=torch.sqrt(distance+1e-16)
                # Preserve opening/closing trajectories. Attraction is zero
                # in open/release and source-surface gap remains the target.
                desired=0. if active and pair['gap_m']<=.002 else pair['gap_m']
                parts['contact']=parts['contact']+((gap-desired)/.002).square()/len(rows)
                if active:parts['normal']=parts['normal']+normal/(1-np.cos(np.deg2rad(15)))/len(rows)
                # GeoRT pinch form, generalized to a palm MCP surface point.
                if pair['gap_m']<.015 and active:parts['point']=parts['point']+distance/.002**2/len(rows)
                gaps.append(float(gap.detach()));normal_errors.append(float(torch.rad2deg(torch.acos((-na@nc).clamp(-1,1))).detach()))
                contacts.append(dict(patch=a,target_patch=c,point_m=pa.detach().cpu().tolist(),target_point_m=pc.detach().cpu().tolist()))
        elif target.get('ready'):
            t=translation[b];rot=rotation[b];mesh=trimesh.Trimesh(target['vertices_m'],target['faces'],process=False)
            center=hand.tensor(target['center_m'])
            hand_normals={}
            for witness in target['contacts']:
                name=witness['patch'];points,normals=hand.samples(pb,rb,name)
                patch_weight=spec['patch_weights'].get(name,1.)
                local=(points-t)@rot+center
                hits,_,faces=trimesh.proximity.closest_point(mesh,local.detach().cpu().numpy())
                hit=(hand.tensor(hits)-center)@rot.T+t;nh=hand.tensor(mesh.face_normals[faces])@rot.T
                distance=(points-hit).square().sum(-1);normal=(1+(normals*nh).sum(-1)).clamp_min(0)
                index=(distance.detach()/.002**2+normal.detach()/(1-np.cos(np.deg2rad(15)))).argmin()
                gap=torch.sqrt(distance[index]+1e-16)
                if active:
                    parts['contact']=parts['contact']+patch_weight*distance[index]/.002**2/len(rows)
                    parts['normal']=parts['normal']+patch_weight*normal[index]/(1-np.cos(np.deg2rad(15)))/len(rows)
                hand_normals[name]=normals[index]
                directions.append(torch.nn.functional.normalize(hit[index]-t,dim=-1))
                source_directions.append(np.asarray(witness['object_point_m'])-np.asarray(target['center_m']))
                gaps.append(float(gap.detach()));normal_errors.append(float(torch.rad2deg(torch.acos((-normals[index]@nh[index]).clamp(-1,1))).detach()))
                contacts.append(dict(patch=name,point_m=points[index].detach().cpu().tolist(),target_point_m=hit[index].detach().cpu().tolist()))
            if len(directions)>1 and active:
                d=torch.stack(directions);source=torch.nn.functional.normalize(hand.tensor(source_directions),dim=-1)
                parts['wrap']=parts['wrap']+((d@d.T-source@source.T)**2).mean()/len(rows)
            if spec['id']=='prismatic' and active:
                for finger in FINGERS[1:]:
                    a=hand_normals.get(finger+'_proximal');c=hand_normals.get(finger+'_distal')
                    if a is not None and c is not None:
                        # Opposing hand patches should align with the object's
                        # parallel support planes; these normals depend on FK.
                        parts['wrap']=parts['wrap']+spec['opposition_weight']*(1+a@c).clamp_min(0)/len(rows)
            # Bidirectional hand/object penetration. Neither targets nor
            # object dimensions can change to make this loss disappear.
            all_points=(rb[hand.audit_owners]@hand.audit_points[...,None]).squeeze(-1)+pb[hand.audit_owners]
            planes=hand.tensor(target['planes']);local=(all_points-t)@rot+center
            sdf=(local@planes[:,:3].T+planes[:,3]).amax(-1)
            obj_depth=(-sdf).clamp_min(0)
            object_depth_mm=float(obj_depth.max().detach()*1000)
            parts['penetration']=parts['penetration']+(obj_depth/.0005).square().amax()/len(rows)
            # Object vertices inside a native link: refresh a signed witness
            # against its full mesh and differentiate the two rigid frames.
            obj=(hand.tensor(target['vertices_m'])-center)@rot.T+t
            for owner,bodymesh in enumerate(hand.surface.meshes):
                local_obj=(obj-pb[owner])@rb[owner]
                query=local_obj.detach().cpu().numpy()
                inside=bodymesh.contains(query)
                if inside.any():
                    hit,_,_=trimesh.proximity.closest_point(bodymesh,query[inside])
                    depth=(local_obj[inside]-hand.tensor(hit)).norm(dim=-1)
                    object_depth_mm=max(object_depth_mm,float(depth.max().detach()*1000))
                    parts['penetration']=parts['penetration']+(depth/.0005).square().max()/len(rows)
                # Full triangle clipping catches crossing surfaces even when
                # neither object's vertices lie inside the other. Refresh a
                # differentiable witness on the intersecting native triangle.
                from .cart_geometry import triangle_interior_point
                posed=(hand.tensor(bodymesh.vertices)@rb[owner].T+pb[owner]-t)@rot+center
                tri=posed.detach().cpu().numpy()[bodymesh.faces]
                equations=np.asarray(target['planes'])
                for equation in equations:
                    tri=tri[(tri@equation[:3]+equation[3]+.0005).min(1)<0]
                    if not len(tri):break
                for triangle in tri:
                    point=triangle_interior_point(triangle,equations,eps=.0005)
                    if point is None:continue
                    object_intersection=True
                    # Freeze the material witness, then differentiate through
                    # both the robot body and the object's rigid pose.
                    local_point=((hand.tensor(point)-center)@rot.detach().T+t.detach()-pb[owner].detach())@rb[owner].detach()
                    world_point=local_point@rb[owner].T+pb[owner]
                    obj_point=(world_point-t)@rot+center
                    depth=-(obj_point@planes[:,:3].T+planes[:,3]).amax()
                    object_depth_mm=max(object_depth_mm,float(depth.detach()*1000))
                    parts['penetration']=parts['penetration']+(depth/.0005).square()/len(rows)
                    break
            parts['pose']=parts['pose']+((t-hand.tensor(target['center_m']))/.05).square().mean()/len(rows)
            parts['pose']=parts['pose']+(rot-torch.eye(3,device=q.device)).square().mean()/len(rows)
        # Relative PIP/DIP bending rather than absolute MCP orientation.
        full_p,full_r=hand.fk.frames(torch.cat((q[b:b+1].new_zeros((1,6)),q[b:b+1]),1))
        values=[full_p[0,hand.ids[0]]]
        for f,name in enumerate(FINGERS):
            values.extend(full_p[0,hand.ids[2+4*f+k]] for k in range(3))
            site=hand.robot.model.site('right_'+name+'_tip');body=int(site.bodyid[0])
            values.append(full_p[0,body]+full_r[0,body]@hand.tensor(site.pos))
        j=torch.stack(values);source=hand.tensor(row['joints_m'])
        canonical_j=(j-j[0])@basis
        robot_v=torch.nn.functional.normalize(canonical_j[1:].reshape(5,4,3).diff(dim=1),dim=-1)
        source_v=torch.nn.functional.normalize(source[1:].reshape(5,4,3).diff(dim=1),dim=-1)
        parts['motion']=parts['motion']+(1-(robot_v*source_v).sum(-1)).mean()/len(rows)
        if not active:
            # A released hand must not remain in the trained closed grasp.
            # Preserve separation as a lower bound, without imposing equal
            # human/robot closing speed or rescaling either hand.
            for f in spec['fingers']:
                if f==0:continue
                a=0 if spec['category']<=2 else 1+4*f
                tip=4+4*f
                robot_gap=(canonical_j[tip]-canonical_j[4 if a==0 else a]).norm()
                source_gap=(source[tip]-source[4 if a==0 else a]).norm()
                parts['motion']=parts['motion']+((source_gap-robot_gap).clamp_min(0)/.002).square()/len(rows)
        shape=((internal_angles(j)-internal_angles(source))/np.deg2rad(10)).square().mean()
        parts['shape']=parts['shape']+shape*spec['shape_weight']/len(rows)
        metrics.append(dict(gaps_mm=[g*1000 for g in gaps],normal_errors_deg=normal_errors,contacts=contacts,
            self_penetration_mm=float(self_depth[b].detach()*1000),object_penetration_mm=object_depth_mm,
            object_intersection_over_05mm=object_intersection,contact_mode=target.get('contact_mode'),
            phase=row['phase'],required=len(gaps),shape_error_deg=float((internal_angles(j)-internal_angles(source)).abs().mean().detach()*180/np.pi)))
    return parts,metrics
