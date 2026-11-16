"""Synchronized source targets and exact-mesh IK losses, outside official GeoRT."""
import itertools
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
import warp as wp
from moxian_web.warp_runtime import serialized
from .adapter import FINGERS,SourceAdapter,digest

PAIRS=list(itertools.combinations(range(5),2))
WEIGHTS=dict(penetration=10.,surface_contact=2.,surface_normal=.2,bone_direction=.5,
             bending=.05,flat_pose=.5,pinch_position=1.,flat_direction=5.)
POLICY='geort_contact_native_mesh_glove_extension_v1'
LOSS_POLICY='pair_balanced_contact_first_v2'


def contact_intent(points,surface_contacts):
    """Retain upstream's 15 mm pinch trigger as well as actual mesh contact."""
    point_contact=np.stack([np.linalg.norm(points[:,a]-points[:,b],axis=-1)<.015 for a,b in PAIRS],1)
    return np.asarray(surface_contacts,dtype=bool)|point_contact


def balanced_contact_mean(values,contacts):
    """Each observed semantic pair has equal weight, independent of duration."""
    mask=contacts.to(values.dtype);counts=mask.sum(0)
    means=(values*mask).sum(0)/counts.clamp_min(1)
    return means.sum()/(counts>0).sum().clamp_min(1)


def true_fk_pinch_loss(tips,source_joints):
    """The upstream zero-tip-distance goal in 15 mm units, using exact FK.

    Only upstream's close-tip frames attract. Actual mesh contact/penetration
    remains a separate objective: tip sites must not be mistaken for surfaces.
    """
    source_tips=source_joints[:,[4,8,12,16,20]]
    mask=torch.stack([(source_tips[:,a]-source_tips[:,b]).norm(dim=-1)<.015 for a,b in PAIRS],1)
    squared=torch.stack([(tips[:,a]-tips[:,b]).square().sum(-1) for a,b in PAIRS],1)
    return balanced_contact_mean(squared/.015**2,mask)


def contact_shape_weights(contacts,bend):
    """Let contacting fingers adapt to robot length; retain genuinely flat poses.

    Adjacent fingers can touch while the whole hand is open, so contact alone
    must never disable the flat-hand constraint. No target or mask uses the
    network's predicted angles (which would let it evade a loss).
    """
    active=torch.stack([contacts[:,[i for i,pair in enumerate(PAIRS) if f in pair]].any(1) for f in range(5)],1)
    hand_flat=bend[:,1:].abs().amax((1,2))<np.deg2rad(10.)
    # Surface/penetration terms were strengthened by 10x. Protect open poses
    # at the same scale; merely retaining the old numerical weight would let
    # gradients on closing frames erase the already-correct open-hand mapping.
    return torch.where(hand_flat[:,None],10.,torch.where(active,.1,1.))


def flat_direction_loss(robot_distal,source_distal,bend):
    """Protect the visible distal direction, without averaging it over bones."""
    flat=(bend[:,1:].abs().amax((1,2))<np.deg2rad(10.)).to(robot_distal.dtype)
    error=(1-(robot_distal*source_distal).sum(-1)).mean(-1)
    return (error*flat).sum()/flat.sum().clamp_min(1)/(1-np.cos(np.deg2rad(15.)))


def sample_contact_frames(contacts,count=32,rng=np.random):
    """Include every available pair, then favor thumb opposition in spare slots."""
    pools=[np.flatnonzero(contacts[:,i]) for i in range(len(PAIRS))]
    available=np.array([i for i,pool in enumerate(pools) if len(pool)],dtype=int)
    if not len(available):return rng.choice(len(contacts),count,replace=True)
    weights=np.array([2. if PAIRS[i][0]==0 else 1. for i in available])
    selected=rng.permutation(available)[:count]
    if len(selected)<count:
        selected=np.r_[selected,rng.choice(available,count-len(selected),p=weights/weights.sum())]
    return np.array([rng.choice(pools[i]) for i in selected])


def synchronized_targets(root,ids,progress,check):
    """Never combine independent finger samples for whole-hand contact loss."""
    from .training import atomic_json,load_recordings
    points,identity,records=load_recordings(root,ids)
    rows=[];all_j=[];all_c=[];all_b=[];offset=0
    for key in ids:
        folder=Path(root)/'recordings'/key
        hashes={name:hashlib.sha256((folder/name).read_bytes()).hexdigest()
                for name in ('motion.npz','topology.json','profile.json','timestamps.npy')}
        fingerprint=digest(dict(policy=POLICY,hashes=hashes,enter_mm=2.,exit_mm=4.))
        cache=Path(root)/'contact_targets'/fingerprint;cache.mkdir(parents=True,exist_ok=True)
        topology=json.loads((folder/'topology.json').read_text());adapter=SourceAdapter(topology)
        with np.load(folder/'motion.npz',allow_pickle=False) as motion:
            raw=motion['joints'];q=motion['qpos']
        stamps=np.load(folder/'timestamps.npy',allow_pickle=False)
        from moxian_web.qnbot_surface import QnbotSurface
        source_model=QnbotSurface(topology['hand_side'],'cpu',finger_twist=True)
        source_model.load_artifact(json.loads((folder/'profile.json').read_text()))
        names=source_model.joint_names;side=topology['hand_side']
        straight={**source_model.thumb_straight,**source_model.joint_limit_policy.get('straight_angles_rad',{})}
        bend_names=[[side+'_thumb_mcp_flexion',side+'_thumb_ip_flexion']]+[[side+'_'+f+'_'+part+'_flexion' for part in ('pip','dip')] for f in ('index','middle','ring','little')]
        bend_ids=np.array([[names.index(n) for n in pair] for pair in bend_names])
        zero=np.array([[straight[n] for n in pair] for pair in bend_names])
        signed_bends=q[:,bend_ids]-zero
        del source_model
        if len(raw)!=len(stamps) or len(q)!=len(raw) or not np.isfinite(raw).all():raise ValueError('约束训练需要完整同步骨架与源关节角')
        j=(raw[:,adapter.indices]-raw[:,adapter.indices[0],None])@adapter.transform
        if not np.allclose(j[:,[4,8,12,16,20]],points[offset:offset+len(j)],atol=2e-6):raise ValueError('骨架与指尖坐标不一致')
        if (cache/'targets.npz').exists():
            with np.load(cache/'targets.npz',allow_pickle=False) as data:contacts=data['contacts']
            if contacts.shape!=(len(j),10):raise ValueError('接触缓存维度不匹配')
        else:
            from .source_targets import BatchSourceContacts,contact_hysteresis
            source_model=QnbotSurface(topology['hand_side'],'cuda:0',finger_twist=True)
            source_model.load_artifact(json.loads((folder/'profile.json').read_text()))
            query=BatchSourceContacts(source_model,topology,batch=16)
            features=np.empty((len(q),10,2),np.float32)
            with torch.inference_mode():
                for start in range(0,len(q),query.batch):
                    check();vertices=source_model(torch.as_tensor(q[start:start+query.batch],device='cuda:0'))[0]
                    gap,depth=query.query(vertices)
                    features[start:start+len(vertices)]=torch.stack((gap,depth),-1).cpu().numpy()
                    if start%128==0:progress('source_surface_targets',completed=offset+start+len(vertices),total=len(points))
            contacts=contact_hysteresis(features[:,:,0],features[:,:,1],stamps)
            tmp=cache/'targets.tmp.npz';np.savez(tmp,contacts=contacts);tmp.replace(cache/'targets.npz')
            atomic_json(cache/'manifest.json',dict(policy=POLICY,hashes=hashes,frames=len(j),fingerprint=fingerprint,
                pairs=PAIRS,contact_frames=contacts.sum(0).tolist(),method='signed_distal_mesh_overlap_or_2mm_gap_4mm_release',
                execution='batched_gpu_distal_dense_original_mesh_no_unused_normals',batch_size=query.batch,
                cuda_graph=query.graph is not None))
            del source_model,query
        all_j.append(j.astype(np.float32));all_c.append(contacts);all_b.append(signed_bends.astype(np.float32))
        rows.append(dict(id=key,fingerprint=fingerprint,hashes=hashes,signed_bend_joints=bend_names,physical_straight_rad=zero.tolist()));offset+=len(j)
    return points,np.concatenate(all_j),np.concatenate(all_c),np.concatenate(all_b),identity,records,rows


@wp.kernel(enable_backward=False)
def _penetration(meshes:wp.array(dtype=wp.uint64),points:wp.array(dtype=wp.vec3),owners:wp.array(dtype=int),
    allowed:wp.array2d(dtype=int),collar:wp.array(dtype=float),lower:wp.array(dtype=wp.vec3),upper:wp.array(dtype=wp.vec3),
    positions:wp.array2d(dtype=wp.vec3),rotations:wp.array2d(dtype=wp.mat33),
    hits:wp.array3d(dtype=wp.vec3),normals:wp.array3d(dtype=wp.vec3),valid:wp.array3d(dtype=int)):
    b,s,t=wp.tid();o=owners[s]
    if allowed[o,t]==0:return
    world=positions[b,o]+rotations[b,o]*points[s];local=wp.transpose(rotations[b,t])*(world-positions[b,t])
    if o==0 and collar[t]>0. and wp.length(local)<=collar[t]:return
    if t==0 and collar[o]>0. and wp.length(points[s])<=collar[o]:return
    lo=lower[t];hi=upper[t]
    if local[0]<lo[0] or local[1]<lo[1] or local[2]<lo[2] or local[0]>hi[0] or local[1]>hi[1] or local[2]>hi[2]:return
    q=wp.mesh_query_point_sign_winding_number(meshes[t],local,1.)
    if q.result and q.sign<0.:
        hit=wp.mesh_eval_position(meshes[t],q.face,q.u,q.v);delta=local-hit
        n=wp.mesh_eval_face_normal(meshes[t],q.face)
        if wp.length(delta)>1.e-8:n=-wp.normalize(delta)
        hits[b,s,t]=hit;normals[b,s,t]=n;valid[b,s,t]=1


@wp.kernel(enable_backward=False)
def _surface(meshes:wp.array(dtype=wp.uint64),points:wp.array(dtype=wp.vec3),source:wp.array(dtype=int),target:wp.array(dtype=int),
    positions:wp.array2d(dtype=wp.vec3),rotations:wp.array2d(dtype=wp.mat33),
    hits:wp.array2d(dtype=wp.vec3),normals:wp.array2d(dtype=wp.vec3)):
    b,s=wp.tid();a=source[s];t=target[s]
    world=positions[b,a]+rotations[b,a]*points[s];local=wp.transpose(rotations[b,t])*(world-positions[b,t])
    q=wp.mesh_query_point_no_sign(meshes[s],local,1.)
    if q.result:
        hits[b,s]=wp.mesh_eval_position(meshes[s],q.face,q.u,q.v)
        normals[b,s]=wp.mesh_eval_face_normal(meshes[s],q.face)


def max_penetration_loss(p,r,points,owners,hits,normals,valid):
    """Same per-frame maximum and tie gradients, graph only for active maxima."""
    with torch.no_grad():
        world=(r[:,owners]@points[...,None]).squeeze(-1)+p[:,owners]
        target=(r[:,None]@hits[...,None]).squeeze(-1)+p[:,None]
        normal=(r[:,None]@normals[...,None]).squeeze(-1)
        depth=torch.where(valid,(-((world[:,:,None]-target)*normal).sum(-1)).clamp_min(0),0.)
        maximum=depth.amax((1,2))
        maxima=(depth==maximum[:,None,None])&(depth>0)&valid
        count=maxima.sum((1,2)).clamp_min(1)
        row,sample,target_id=maxima.nonzero(as_tuple=True)
    source_id=owners[sample]
    world=(r[row,source_id]@points[sample,:,None]).squeeze(-1)+p[row,source_id]
    # Advanced indexing copies witnesses, so cached query buffers may be reused
    # even when two forward graphs are retained before backward.
    target=(r[row,target_id]@hits[row,sample,target_id,:,None]).squeeze(-1)+p[row,target_id]
    normal=(r[row,target_id]@normals[row,sample,target_id,:,None]).squeeze(-1)
    values=(-((world-target)*normal).sum(-1)).clamp_min(0)/count[row]
    maximum=p.new_zeros(len(p)).scatter_add(0,row,values)
    return (maximum/.0005).square().mean()


class ContactGeometry:
    @serialized
    def __init__(self,robot,device='cuda:0'):
        from moxian_web.wuji_surface import WujiSurface
        from retarget.baseline.card.collision import WujiMeshCollision
        from retarget.env.grasp.pose_gpu import ModelFK
        self.device=str(device);self.robot=robot;self.surface=WujiSurface();self.fk=ModelFK(robot.model,device)
        self.collision=WujiMeshCollision(self.surface,device)
        self.low=torch.tensor(robot.lower,device=device);self.high=torch.tensor(robot.upper,device=device)
        self.basis=torch.tensor(robot.basis,device=device,dtype=torch.float32)
        self.audit_points=wp.to_torch(self.collision.samples[False][0]);self.audit_owners=wp.to_torch(self.collision.samples[False][1]).long()
        patches=[];self.meshes=[]
        for f in range(5):
            owner=4+4*f;region=self.surface.contact_surface(owner);tri=np.array(region['triangles'],np.float32)
            if not len(tri):raise ValueError('缺少 Wuji2 distal 掌侧曲面')
            ids=np.linspace(0,len(tri)-1,min(32,len(tri))).astype(int)
            patches.append((tri[ids].mean(1),np.asarray(region['normals'],np.float32)[ids]))
            vertices=tri.reshape(-1,3);faces=np.arange(len(vertices),dtype=np.int32)
            self.meshes.append(wp.Mesh(points=wp.array(vertices,dtype=wp.vec3,device=device),indices=wp.array(faces,dtype=int,device=device)))
        points=[];normals=[];source=[];target=[];mesh_ids=[];pair_ids=[]
        for i,(a,b) in enumerate(PAIRS):
            for a,b in ((a,b),(b,a)):
                p,n=patches[a];points.extend(p);normals.extend(n);source.extend([4+4*a]*len(p));target.extend([4+4*b]*len(p));mesh_ids.extend([self.meshes[b].id]*len(p));pair_ids.extend([i]*len(p))
        self.points=torch.tensor(np.asarray(points),device=device);self.normals=torch.tensor(np.asarray(normals),device=device)
        self.source=torch.tensor(source,device=device);self.target=torch.tensor(target,device=device)
        self.pair_indices=[torch.tensor(np.flatnonzero(np.array(pair_ids)==i),device=device) for i in range(10)]
        self.wp_points=wp.from_torch(self.points,dtype=wp.vec3);self.wp_source=wp.array(source,dtype=int,device=device);self.wp_target=wp.array(target,dtype=int,device=device)
        self.wp_meshes=wp.array(mesh_ids,dtype=wp.uint64,device=device)
        width=max(map(len,self.pair_indices));packed=torch.zeros((10,width),device=device,dtype=torch.long)
        mask=torch.zeros((10,width),device=device,dtype=torch.bool)
        for i,ids in enumerate(self.pair_indices):packed[i,:len(ids)]=ids;mask[i,:len(ids)]=True
        self.packed_pairs=packed;self.packed_mask=mask
        self.distal_ids=torch.tensor([4+4*f for f in range(5)],device=device)
        self.tip_sites=torch.tensor(np.asarray(robot.model.site_pos[robot.tip_ids]),device=device,dtype=torch.float32)
        self._buffers={}

    def _query_buffers(self,batch):
        if batch not in self._buffers:
            shape=(batch,len(self.audit_points),21);surface=(batch,len(self.points))
            self._buffers[batch]=(wp.zeros(shape,dtype=wp.vec3,device=self.device),
                wp.zeros(shape,dtype=wp.vec3,device=self.device),wp.zeros(shape,dtype=int,device=self.device),
                wp.empty(surface,dtype=wp.vec3,device=self.device),wp.empty(surface,dtype=wp.vec3,device=self.device))
        return self._buffers[batch]

    @serialized
    def losses(self,norm,source_joints,contacts,source_bends=None):
        q=(norm/2+.5)*(self.high-self.low)+self.low
        p,r=self.fk.frames(torch.cat((q.new_zeros((len(q),6)),q),1));p=p[:,self.robot.body_ids];r=r[:,self.robot.body_ids]
        shape=(len(q),len(self.audit_points),21)
        hit,normal,valid,surface_hit,surface_normal=self._query_buffers(len(q))
        wp.to_torch(valid).zero_()
        wp_p=wp.from_torch(p.detach().contiguous(),dtype=wp.vec3);wp_r=wp.from_torch(r.detach().contiguous(),dtype=wp.mat33)
        wp.launch(_penetration,shape,inputs=[self.collision.mesh_ids,*self.collision.samples[False],self.collision.allowed,self.collision.collar,self.collision.lower,self.collision.upper,wp_p,wp_r,hit,normal,valid],device=self.device)
        penetration=max_penetration_loss(p,r,self.audit_points,self.audit_owners,
            wp.to_torch(hit),wp.to_torch(normal),wp.to_torch(valid).bool())
        shape=(len(q),len(self.points));hit=surface_hit;normal=surface_normal
        wp.launch(_surface,shape,inputs=[self.wp_meshes,self.wp_points,self.wp_source,self.wp_target,wp_p,wp_r,hit,normal],device=self.device)
        local_hit=wp.to_torch(hit).clone();local_normal=wp.to_torch(normal).clone()
        a=(r[:,self.source]@self.points[...,None]).squeeze(-1)+p[:,self.source]
        b=(r[:,self.target]@local_hit[...,None]).squeeze(-1)+p[:,self.target]
        an=(r[:,self.source]@self.normals[...,None]).squeeze(-1)
        bn=(r[:,self.target]@local_normal[...,None]).squeeze(-1)
        distance=(a-b).square().sum(-1);angle=(1+(an*bn).sum(-1)).clamp_min(0)
        with torch.no_grad():
            score=distance[:,self.packed_pairs]/.002**2+.05*angle[:,self.packed_pairs]
            score=score.masked_fill(~self.packed_mask,float('inf'))
            chosen=self.packed_pairs[None].expand(len(q),-1,-1).gather(2,score.argmin(-1,keepdim=True)).squeeze(-1)
        gaps=distance.gather(1,chosen);angles=angle.gather(1,chosen)
        surface_contact=balanced_contact_mean(gaps/.002**2,contacts)
        surface_normal=balanced_contact_mean(angles/(1-np.cos(np.deg2rad(30))),contacts)
        # Native segment directions, not marker or mesh display frames.
        owner=self.distal_ids
        tip=p[:,owner]+(r[:,owner]@self.tip_sites[...,None]).squeeze(-1)
        pinch_position=true_fk_pinch_loss(tip,source_joints)
        joints=torch.stack((p[:,owner-2],p[:,owner-1],p[:,owner],tip),2)@self.basis
        rv=torch.nn.functional.normalize(joints.diff(dim=2),dim=-1,eps=1e-8)
        source=source_joints[:,1:].reshape(-1,5,4,3).diff(dim=2)
        sv=torch.nn.functional.normalize(source,dim=-1,eps=1e-8)
        bend=torch.atan2(torch.linalg.vector_norm(torch.linalg.cross(sv[:,:,:-1],sv[:,:,1:]),dim=-1),(sv[:,:,:-1]*sv[:,:,1:]).sum(-1))
        if source_bends is not None:bend=source_bends
        flat_direction=flat_direction_loss(rv[:,:,-1],sv[:,:,-1],bend)
        shape_weight=contact_shape_weights(contacts,bend)
        bone_direction=((1-(rv*sv).sum(-1))*shape_weight[:,:,None]).mean()/(1-np.cos(np.deg2rad(15.)))
        angle=q.reshape(-1,5,4)[:,:,[2,3]]
        difference=((angle-bend)/np.deg2rad(10.)).square()
        bending=(difference*shape_weight[:,:,None]).mean()
        flat=(bend.abs().amax(-1)<np.deg2rad(10.)).float()
        flat_pose=(difference.mean(-1)*flat*shape_weight).sum()/flat.sum().clamp_min(1)
        parts=dict(penetration=penetration,surface_contact=surface_contact,surface_normal=surface_normal,
                   bone_direction=bone_direction,bending=bending,flat_pose=flat_pose,pinch_position=pinch_position,
                   flat_direction=flat_direction)
        return sum(WEIGHTS[k]*v for k,v in parts.items()),parts
