"""Mixed offline replay fitting of the official skin and optional 25-DOF Delta.

Only acquisition labels select loss masks. Neither labels nor target contacts
enter policy features. Validation never selects templates, statistics or steps.
"""
from __future__ import annotations
import copy,hashlib,json,math,time,uuid
import numpy as np
import torch
from torch import nn
from .qnbot_surface import QnbotSurface,FINGERS,SOURCE_FINGERS,REVISION,exp_so3
from .qnbot_reference import observations,initialize_registration
from .calibration_targets import target
from .mano_contact_loss import normal_residual,normal_error_deg

FIT_REVISION='qnbot_trajectory_fit_v6'
POLICY_REVISION='qnbot_delta25_v1'
DIMENSION_MODE='finger_dimensions_v1'


def configure_dimensions(model):
    """Fifteen independent log dimensions; preserve each finger's proportions.

    The on-disk ABI stays twenty bone scales. Tied gradients give all phalanges
    of one finger exactly the same update; wrist/metacarpals remain fixed.
    """
    model.dimension_mode=DIMENSION_MODE
    owner=torch.as_tensor(model.bone_finger,device=model.device)
    movable=torch.as_tensor([f>=0 and not n.endswith('_metacarpal') for f,n in zip(model.bone_finger,model.bone_names)],device=model.device)
    model.dimension_owners=owner;model.dimension_movable=movable
    for k,p in model.geometry.items():p.requires_grad_(k in ('length','width','depth'))
    def tied(gradient):
        sums=torch.zeros(5,device=gradient.device,dtype=gradient.dtype)
        sums.scatter_add_(0,owner.clamp_min(0),gradient*movable)
        return sums[owner.clamp_min(0)]*movable
    for key in ('length','width','depth'):model.geometry[key].register_hook(tied)
    return [model.geometry[k] for k in ('length','width','depth')]


def contact_band_residual(a,b,na,nb,tolerance_m=.002):
    """Tangential alignment, zero gap, and up to 2mm normal compression."""
    normal=torch.nn.functional.normalize(na-nb,dim=-1)
    delta=b-a;gap=(delta*normal).sum(-1)
    tangent=delta-gap[...,None]*normal
    excess=gap.clamp_min(0)-(-gap-tolerance_m).clamp_min(0)
    return tangent+excess[...,None]*normal


def collision_violation_loss(residual):
    """Soft mean-violating and worst penalties, invariant to zero witnesses.

    Residuals already subtract the allowed 2mm on designated contact pairs.
    """
    if not residual.numel():return residual.sum()*0
    squared=residual.square()
    return 160*(squared.sum()/(residual>0).sum().clamp_min(1)+squared.max())


def training_collision_loss(model,q,ids,buffer,collision):
    # Preserve shuffled order: sorting ids would preferentially inspect early
    # actions. Multiple real contact poses prevent one pose dominating updates.
    contact_ids=ids[buffer.contact_mask[ids]]
    selected=contact_ids[:3] if contact_ids.numel() else ids[:1]
    vertices=model(q[selected])[0];loss=vertices.sum()*0
    if not hasattr(model,'_collision_allowed'):model._collision_allowed={}
    for row_id,vv in zip(selected.tolist(),vertices):
        row=buffer.rows[row_id];key=(row['stage'],bool(row.get('contact_confirmed')))
        if key not in model._collision_allowed:
            allowed=torch.zeros(20,20,device=model.device,dtype=torch.bool)
            if key[1]:
                spec=target(row['stage'])
                for edge in spec.get('edges',[]):
                    parts=[]
                    for name in edge['regions']:
                        if name.startswith('palm_'):parts.append([i for i,n in enumerate(model.bone_names) if n.endswith('_wrist') or n.endswith('_metacarpal')])
                        else:
                            finger,part=name.split('_');parts.append([model.bone_names.index(model.side+'_'+SOURCE_FINGERS[FINGERS.index(finger)]+'_'+part)])
                    for a in parts[0]:
                        for b in parts[1]:allowed[a,b]=True;allowed[b,a]=True
                if spec['kind']=='distal_cluster':
                    bones=[model.bone_names.index(model.side+'_'+SOURCE_FINGERS[FINGERS.index(f)]+'_distal') for f in spec['active_fingers']]
                    for a in bones:
                        for b in bones:allowed[a,b]=True
            model._collision_allowed[key]=allowed
        witness=collision.query(vv.detach());allowed=model._collision_allowed[key]
        tolerance=torch.where(allowed[witness['source_part'],witness['target_part']],.002,0.)
        loss=loss+collision_violation_loss(collision.residual(vv,witness,tolerance_m=tolerance))
    return loss/max(len(vertices),1)


class DeltaPolicy(nn.Module):
    def __init__(self,inputs=146):
        super().__init__();self.net=nn.Sequential(nn.Linear(inputs,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,25))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):return self.net(x)


def apply_delta(seed,delta,limits):
    # Some existing SDK IK seeds exceed the packaged MJCF range. A zero-head
    # policy must preserve those seeds, not introduce an unlearned 11° jump.
    # Permit movement toward the native range, never farther outside it.
    lower=torch.minimum(seed,limits[:,0]);upper=torch.maximum(seed,limits[:,1])
    return (seed+delta).maximum(lower).minimum(upper)


def tactile_features(rows,device):
    values=[];masks=[]
    for row in rows:
        c=row.get('contact') or {};fresh=c.get('ready') is True and isinstance(row.get('tactile_timestamp_ms'),(int,float)) and abs(row['tactile_timestamp_ms']-row['timestamp_ms'])<=25
        normalized=c.get('normalized') or [];assign=(row.get('mapping') or {}).get('assignments',[])
        vv=[];mm=[]
        for patch in range(20):
            ids=[i for i,x in enumerate(assign) if x==patch and i<len(normalized)] if fresh else []
            good=[normalized[i] for i in ids if isinstance(normalized[i],(float,int)) and math.isfinite(normalized[i])]
            vv.append(float(np.mean(good)) if good else 0.);mm.append(float(bool(good)))
        values.append(vv);masks.append(mm)
    return tuple(torch.tensor(x,device=device,dtype=torch.float32) for x in (values,masks))


class ReplayBuffer:
    def __init__(self,model,rows,reference=None,progress=None,cancel=None):
        self.rows=rows;self.device=model.device;self.q=model.angles(rows);self.rawp,self.rawr,self.valid=observations(rows,self.device)
        self.touch,self.touch_mask=tactile_features(rows,self.device)
        self.mask=torch.tensor([[float(f in target(r['stage'])['active_fingers']) for f in FINGERS] for r in rows],device=self.device)*self.valid
        self.stages=list(dict.fromkeys(r['stage'] for r in rows));self.stage_indices=[torch.tensor([i for i,r in enumerate(rows) if r['stage']==s],device=self.device) for s in self.stages]
        self.prev=torch.arange(len(rows),device=self.device);self.dt=torch.zeros(len(rows),device=self.device);self.flat=self.prev.clone()
        first={};original={r.get('_source_index',i):i for i,r in enumerate(rows)}
        for i,row in enumerate(rows):
            key=(row.get('window_id'),row['stage']);first.setdefault(key,i)
            start=original.get(row.get('trajectory_flat_start_index'),first[key]);self.flat[i]=start
            if i and key==(rows[i-1].get('window_id'),rows[i-1]['stage']):
                dt=(row['timestamp_ms']-rows[i-1]['timestamp_ms'])/1000
                if 0<dt<=.25:self.prev[i]=i-1;self.dt[i]=dt
        # A time integral per action/cycle, independent of sampling density.
        durations=torch.where(self.dt>0,self.dt,torch.full_like(self.dt,.02));self.weight=torch.zeros_like(durations)
        for ids in self.stage_indices:
            cycles=list(dict.fromkeys(rows[int(i)]['cycle'] if 'cycle' in rows[int(i)] else 0 for i in ids.cpu()))
            for cycle in cycles:
                group=ids[torch.tensor([rows[int(i)].get('cycle',0)==cycle for i in ids.cpu()],device=self.device)]
                self.weight[group]=durations[group]/durations[group].sum().clamp_min(1e-12)/len(cycles)/len(self.stages)
        prices={s:2. if s.startswith(('contact_pair_','contact_tripod_','contact_four_')) or s=='contact_five' else 1. for s in self.stages}
        self.observation_weight=self.weight*torch.tensor([prices[r['stage']] for r in rows],device=self.device)/(sum(prices.values())/len(prices))
        self.patch_names=list(model.patch_anchors);self.patch_valid=torch.tensor([[float(n in target(r['stage'])['patches']) for n in self.patch_names] for r in rows],device=self.device)
        self.contact_mask=torch.tensor([bool(r.get('contact_confirmed')) for r in rows],device=self.device)
        self.palm_region_vertices=torch.zeros((len(self.patch_names),len(model.vertices0)),device=self.device,dtype=torch.bool)
        for k,name in enumerate(self.patch_names):
            if name.startswith('palm_'):self.palm_region_vertices[k,torch.unique(model.faces[torch.as_tensor(model.regions[name],device=self.device)])]=True
        self.reference_p=torch.zeros((len(rows),len(self.patch_names),3),device=self.device);self.reference_n=torch.zeros_like(self.reference_p)
        ref=reference or model
        anchors=[ref.patch_anchors[n] for n in self.patch_names];support=torch.unique(ref.faces[torch.tensor([a['face'] for a in anchors],device=self.device)].flatten(),sorted=True)
        tri=torch.searchsorted(support,ref.faces[torch.tensor([a['face'] for a in anchors],device=self.device)])
        with torch.no_grad():
            for start in range(0,len(rows),64):
                if cancel is not None and cancel.is_set():raise InterruptedError('轨迹预处理已取消')
                end=min(len(rows),start+64);v=ref(self.q[start:end],vertex_ids=support)[0];t=v[:,tri]
                self.reference_p[start:end]=t.mean(-2);self.reference_n[start:end]=torch.nn.functional.normalize(torch.linalg.cross(t[:,:,1]-t[:,:,0],t[:,:,2]-t[:,:,0],dim=-1),dim=-1)
                # v4 stores the actual frozen acquisition reference. Never replace
                # it with the candidate (including when resuming from a checkpoint).
                for i in range(start,end):
                    for k,name in enumerate(self.patch_names):
                        recorded=rows[i].get('patch_trajectory',{}).get(name)
                        if recorded and recorded.get('valid'):
                            self.reference_p[i,k]=torch.as_tensor(recorded['position_m'],device=self.device);self.reference_n[i,k]=torch.as_tensor(recorded['normal'],device=self.device)
                if progress:progress(dict(phase='prepare',completed=end,total=len(rows),message='构建官方 patch 连续轨迹 buffer',progress=.05+.1*end/len(rows)))
        self.feature_mean=None;self.feature_scale=None

    def features(self,model):
        with torch.no_grad():
            pred,_=model.sensor_prediction(self.q);align,trans=model.alignment();p=self.rawp@align.T+trans;r=align@self.rawr
            return torch.cat((self.q,p.flatten(1),r.flatten(1),(pred-p).flatten(1),self.touch,self.touch_mask,self.valid,self.dt[:,None]),1)

    def seed_stats(self,model):
        x=self.features(model);self.feature_mean=x.mean(0);self.feature_scale=x.std(0,unbiased=False).clamp_min(1e-3)

    def corrected(self,model,policy):
        if policy is None:self.last_raw_delta=torch.zeros_like(self.q);return self.q
        x=self.features(model);raw=policy((x-self.feature_mean)/self.feature_scale)
        # Same two-sample causal rate rule in fitting and streaming. Previous
        # RAW output, rather than a recursively filtered output, is the state.
        self.last_raw_delta=raw;prev=raw[self.prev];limit=self.dt[:,None]*math.radians(25)
        delta=torch.where((self.dt>0)[:,None],prev+(raw-prev).clamp(-limit,limit),raw)
        return apply_delta(self.q,delta,model.limits)


def policy_artifact(policy,buffer,model):
    if policy is None:return None
    return dict(revision=POLICY_REVISION,inputs=146,outputs=25,joint_names=model.joint_names,asset_hash=model.asset_hash,
        feature_definition=['seed_q','wrist_emf_position','wrist_emf_rotation','sensor_prediction_residual','actual_tactile','tactile_valid','emf_valid','dt'],
        mean=buffer.feature_mean.cpu().tolist(),scale=buffer.feature_scale.cpu().tolist(),state={k:v.detach().cpu().tolist() for k,v in policy.state_dict().items()},rate_deg_s=25.)


def load_policy(artifact,model):
    if artifact.get('revision')!=POLICY_REVISION or artifact.get('asset_hash')!=model.asset_hash or artifact.get('joint_names')!=model.joint_names:raise ValueError('Delta权重与官方25关节模型不兼容')
    p=DeltaPolicy().to(model.device);state=p.state_dict()
    for k,v in state.items():
        t=torch.tensor(artifact['state'][k],device=model.device,dtype=v.dtype)
        if t.shape!=v.shape or not torch.isfinite(t).all():raise ValueError('无效Delta权重')
        state[k]=t
    p.load_state_dict(state);p.eval();return p


def nearest_region_anchors(vertices,faces,regions,a,b):
    import trimesh
    # Blended semantic patches can share joint-boundary vertices. They must not
    # produce a fictitious zero-distance fold contact at the SAME material point.
    ids_a=np.asarray(regions[a]);ids_b=np.asarray(regions[b])
    shared=np.intersect1d(faces[ids_a].ravel(),faces[ids_b].ravel())
    if len(shared):
        ids_a=ids_a[~np.isin(faces[ids_a],shared).any(1)];ids_b=ids_b[~np.isin(faces[ids_b],shared).any(1)]
        if not len(ids_a) or not len(ids_b):raise ValueError('接触区域无独立材料核心：'+a+'/'+b)
    regions={**regions,a:ids_a,b:ids_b}
    best=None
    for aa,bb,reverse in ((a,b,False),(b,a,True)):
        ids=np.asarray(regions[aa]);target_ids=np.asarray(regions[bb]);vids=np.unique(faces[ids]);mesh=trimesh.Trimesh(vertices,faces[target_ids],process=False)
        # Bounded original-triangle queries. KD/collision proxies cannot define
        # the material binding or be the final reported surface distance.
        for start in range(0,len(vids),64):
            vv=vids[start:start+64];closest,distance,which=trimesh.proximity.closest_point_naive(mesh,vertices[vv]);k=int(distance.argmin());vertex=int(vv[k]);face0=int(ids[np.flatnonzero((faces[ids]==vertex).any(1))[0]])
            bc0=np.eye(3)[int(np.flatnonzero(faces[face0]==vertex)[0])];face1=int(target_ids[which[k]])
            bc1=trimesh.triangles.points_to_barycentric(vertices[faces[[face1]]],closest[[k]])[0]
            anchors=[dict(face=face0,barycentric=bc0.tolist()),dict(face=face1,barycentric=bc1.tolist())]
            if reverse:anchors.reverse()
            if best is None or distance[k]<best[0]:best=(float(distance[k]),anchors)
    return best


def prepare_templates(model,buffer,cancel,progress,steps=64):
    """Train-only standard-geometry pose solve; freeze material anchors once."""
    contacts=[i for i,r in enumerate(buffer.rows) if r.get('contact_confirmed')]
    if not contacts:raise ValueError('训练记录缺少接触帧')
    q=nn.Parameter(buffer.q[contacts].clone());opt=torch.optim.Adam([q],lr=.03)
    edges=[target(buffer.rows[i]['stage']).get('edges',[]) for i in contacts]
    from .contact_protocol import cluster_tree
    for k,i in enumerate(contacts):
        spec=target(buffer.rows[i]['stage'])
        if spec['kind']=='distal_cluster':
            names=spec['patches'];ids=[buffer.patch_names.index(n) for n in names];p=buffer.reference_p[i,ids].cpu().numpy();edges[k]=cluster_tree(names,np.linalg.norm(p[:,None]-p[None],axis=-1))
    anchors=[model.patch_anchors[n] for n in buffer.patch_names]
    support=torch.unique(model.faces[torch.tensor([a['face'] for a in anchors],device=model.device)].flatten(),sorted=True);tri=torch.searchsorted(support,model.faces[torch.tensor([a['face'] for a in anchors],device=model.device)])
    pair=[(k,buffer.patch_names.index(e['regions'][0]),buffer.patch_names.index(e['regions'][1])) for k,group in enumerate(edges) for e in group]
    pi=torch.tensor(pair,device=model.device);active=buffer.mask[contacts][:,model.joint_finger_ids]
    for step in range(steps):
        if cancel.is_set():raise InterruptedError('接触模板已取消')
        opt.zero_grad();limited=q.maximum(model.limits[:,0]).minimum(model.limits[:,1]);effective=buffer.q[contacts]+(limited-buffer.q[contacts])*active
        v=model(effective,vertex_ids=support)[0];patch=v[:,tri];p=patch.mean(-2)
        normal=torch.nn.functional.normalize(torch.linalg.cross(patch[:,:,1]-patch[:,:,0],patch[:,:,2]-patch[:,:,0],dim=-1),dim=-1)
        _,sr=model.sensor_prediction(effective);align,_=model.alignment()
        loss=((p[pi[:,0],pi[:,1]]-p[pi[:,0],pi[:,2]])/.002).square().mean()+normal_residual(normal[pi[:,0],pi[:,1]],normal[pi[:,0],pi[:,2]],tolerance_deg=0.).square().mean()+((sr-align@buffer.rawr[contacts])/.15).square().mean()+.01*(q-buffer.q[contacts]).square().mean()
        loss.backward();opt.step()
        # Template optimization changes poses ONLY, never the reference mesh.
        model.zero_grad(set_to_none=True)
        if step%16==0:progress(dict(phase='templates',completed=step,total=steps,message='仅训练帧求解官方曲面接触模板',progress=.15+.05*step/steps))
    with torch.no_grad():v=model(buffer.q[contacts]+(q.maximum(model.limits[:,0]).minimum(model.limits[:,1])-buffer.q[contacts])*active)[0].cpu().numpy()
    actions=[];faces=model.faces.cpu().numpy()
    for stage in buffer.stages:
        ids=[k for k,i in enumerate(contacts) if buffer.rows[i]['stage']==stage]
        if not ids:continue
        k=ids[0];records=[]
        for e in edges[k]:
            gap,binding=nearest_region_anchors(v[k],faces,model.regions,*e['regions']);n=[]
            for a in binding:
                t=v[k,faces[a['face']]];x=np.cross(t[1]-t[0],t[2]-t[0]);n.append(x/max(np.linalg.norm(x),1e-12))
            error=float(np.degrees(np.arccos(np.clip(-np.dot(*n),-1,1))))
            records.append(dict(e,anchors=binding,template_gap_mm=gap*1000,template_normal_error_deg=error,template_penetration_mm=None))
        actions.append(dict(stage=stage,edges=records,feasible=all(e['template_gap_mm']<=5 and e['template_normal_error_deg']<=15 for e in records)))
    return dict(revision='qnbot_link_regions_v2',asset_hash=model.asset_hash,topology_hash=model.topology_hash,region_revision=model.region_revision,source='train_only',validation_used=False,contact_target='moving_original_triangle_closest_points_distance_zero_normals_antiparallel',actions=actions)


class TrajectoryObjective:
    def __init__(self,model,buffer,templates):
        self.model=model;self.buffer=buffer;self.templates={a['stage']:a['edges'] for a in templates['actions']}
        self.anchors=[model.patch_anchors[n] for n in buffer.patch_names];self.contact=[];self.edge_info={}
        for i,row in enumerate(buffer.rows):
            if not row.get('contact_confirmed'):continue
            group=self.templates.get(row['stage'])
            if not group:raise ValueError('目标动作未进入接触损失 '+row['stage'])
            for e in group:
                a=len(self.anchors);self.anchors.extend(e['anchors']);self.contact.append((i,a,a+1,e.get('weight',1.)));self.edge_info[a]=dict(regions=e['regions'],required=e.get('required',True))
        ids=torch.tensor([a['face'] for a in self.anchors],device=model.device);self.support=torch.unique(model.faces[ids].flatten(),sorted=True);self.tri=torch.searchsorted(self.support,model.faces[ids]);self.bc=torch.tensor([a['barycentric'] for a in self.anchors],device=model.device)
        self.contact_by_row={i:[] for i,_,_,_ in self.contact}
        for i,a,b,w in self.contact:self.contact_by_row[i].append((a,b,w))
        capacity=max([len(g) for g in self.contact_by_row.values()]+[1]);count=len(buffer.rows)
        aa=np.zeros((count,capacity),np.int64);bb=aa.copy();weight=np.zeros((count,capacity),np.float32)
        event_counts={s:sum(bool(r.get('contact_confirmed')) and r['stage']==s for r in buffer.rows) for s in buffer.stages}
        for i,group in self.contact_by_row.items():
            total=sum(w for _,_,w in group)
            for k,(a,b,w) in enumerate(group):aa[i,k]=a;bb[i,k]=b;weight[i,k]=w/total/event_counts[buffer.rows[i]['stage']]/len(buffer.stages)
        self.contact_a=torch.as_tensor(aa,device=model.device);self.contact_b=torch.as_tensor(bb,device=model.device);self.contact_weight=torch.as_tensor(weight,device=model.device)
        unique=list(dict.fromkeys(i for i,_,_,_ in self.contact));lookup={i:k for k,i in enumerate(unique)}
        self.link_rows=torch.tensor(unique,device=model.device,dtype=torch.long);self.link_edges=[];link_weight=[]
        self.link_by_row={};self.link_query=None
        for i,a,b,w in self.contact:
            index=len(self.link_edges);self.link_by_row.setdefault(i,[]).append(index)
            self.link_edges.append(dict(row=lookup[i],source_row=i,anchor=a,**self.edge_info[a]))
            k=int(np.flatnonzero(aa[i]==a)[0]);link_weight.append(weight[i,k])
        self.link_weights=torch.tensor(link_weight,device=model.device);self.link_source_rows=torch.tensor([e['source_row'] for e in self.link_edges],device=model.device,dtype=torch.long)
        self.baseline_rest=model.vertices0.detach().clone();edge=model.edges;self.base_edges=(self.baseline_rest[edge[:,1]]-self.baseline_rest[edge[:,0]]).detach()
        from scipy.spatial import cKDTree
        from .qnbot_surface import normals
        with torch.no_grad():base_normal=normals(self.baseline_rest,model.faces).cpu().numpy()
        base=self.baseline_rest.cpu().numpy();weights=model.weights.cpu().numpy();pairs=[]
        for bone in range(20):
            axis=model.bind_r0[bone].detach().cpu().numpy()[:,1];dot=base_normal@axis
            aa=np.flatnonzero((weights[:,bone]>.3)&(dot>.4));bb=np.flatnonzero((weights[:,bone]>.3)&(dot<-.4))
            if not len(aa) or not len(bb):continue
            aa=aa[np.linspace(0,len(aa)-1,min(len(aa),128)).astype(int)]
            plane=model.bind_r0[bone].detach().cpu().numpy()[:,[0,2]];_,nearest=cKDTree(base[bb]@plane).query(base[aa]@plane)
            pairs.extend(zip(aa,bb[nearest]))
        self.thickness_pairs=torch.tensor(pairs,device=model.device,dtype=torch.long).reshape(-1,2)
        d=self.baseline_rest[self.thickness_pairs[:,0]]-self.baseline_rest[self.thickness_pairs[:,1]]
        self.thickness0=d.norm(dim=-1).clamp_min(.001);self.thickness_axis=torch.nn.functional.normalize(d,dim=-1)

    def surface(self,q):
        v,j,r=self.model(q,vertex_ids=self.support);tri=v[:,self.tri];p=(tri*self.bc[None,...,None]).sum(-2)
        n=torch.nn.functional.normalize(torch.linalg.cross(tri[:,:,1]-tri[:,:,0],tri[:,:,2]-tri[:,:,0],dim=-1),dim=-1)
        return p,n,j,r

    def link_contact(self,q,active=None):
        if not self.link_edges:
            empty=torch.empty((0,2,3),device=self.model.device);return empty,empty
        from .qnbot_link_contact import LinkContactQuery
        vertices=self.model(q[self.link_rows])[0]
        if self.link_query is None:self.link_query=LinkContactQuery(vertices.detach(),self.model.faces,self.model.regions,self.link_edges)
        points,normal,witness=self.link_query.evaluate(vertices,active)
        self.last_link_minimum=witness['minimum_squared_distance'].clamp_min(0).sqrt()
        return points,normal

    def loss(self,q,ids,measure=True):
        m=self.model;b=self.buffer;flat=b.flat[ids];selection=torch.cat((ids,flat));p,n,j,r=self.surface(q[selection]);batch=len(ids);pp=p[:batch];nn=n[:batch];fp=p[batch:];fn=n[batch:]
        pred,rotation=m.sensor_prediction(q[ids]);align,trans=m.alignment();mask=b.mask[ids];weights=b.weight[ids]
        robust=lambda x:2*(torch.sqrt(1+x.square())-1)
        denominator=mask.sum(1).clamp_min(1)
        sensor=(robust((pred-(b.rawp[ids]@align.T+trans))/.003).mean(-1)*mask).sum(1)/denominator
        orient=(robust((rotation-align@b.rawr[ids])/.15).mean((-1,-2))*mask).sum(1)/denominator
        terms=dict(sensor_position=320*(sensor*b.observation_weight[ids]).sum(),sensor_orientation=40*(orient*b.observation_weight[ids]).sum())
        patchmask=b.patch_valid[ids];count=patchmask.sum(1).clamp_min(1);k=len(b.patch_names)
        movement=(pp[:,:k]-fp[:,:k])-(b.reference_p[ids]-b.reference_p[flat]);direction=(nn[:,:k]-fn[:,:k])-(b.reference_n[ids]-b.reference_n[flat])
        terms['trajectory_position']=.1*((robust(movement/.01).mean(-1)*patchmask).sum(1)/count*weights).sum()
        terms['trajectory_normal']=.1*((robust(direction/math.radians(10)).mean(-1)*patchmask).sum(1)/count*weights).sum()
        actual=torch.nn.functional.normalize(j[:batch,m.tip_bodies]-j[:batch,m.distal_bodies],dim=-1)
        # -X is the delivered tip direction; include fixed distal->tip rotation.
        with torch.no_grad():
            br,bp=m.fk(m.q0[None],personalized=False);local=(br[0,m.distal_bodies].transpose(-1,-2)@(bp[0,m.tip_bodies]-bp[0,m.distal_bodies])[...,None]).squeeze(-1);local=torch.nn.functional.normalize(local,dim=-1)
        desired=((align@b.rawr[ids])@(m.mount_r0@exp_so3(m.geometry['mount_rot'])).transpose(-1,-2))@local[...,None]
        tip=((actual-desired.squeeze(-1))/.15).square().mean(-1)
        terms['tip_emf_orientation']=40*(tip.mul(mask).sum(1)/denominator*b.observation_weight[ids]).sum()
        multiplicity=torch.bincount(ids,minlength=len(b.rows))[self.link_source_rows];cw=self.link_weights*multiplicity
        contact_points,contact_normals=self.link_contact(q,multiplicity>0)
        diff=contact_points[:,0]-contact_points[:,1];na=contact_normals[:,0];nb=contact_normals[:,1]
        # Contact TARGET is exactly zero distance with antiparallel normals.
        # Up to 2mm penetration is a separate collision tolerance, not a 2mm
        # dead-zone in the distance or normal attraction objective.
        terms['contact_position']=320*((diff/.002).square().mean(-1)*cw).sum()
        terms['contact_normal']=40*(normal_residual(na,nb,tolerance_deg=0.).square().mean(-1)*cw).sum()
        contact_mask=cw>0;den=contact_mask.sum().clamp_min(1)
        gap=(diff.detach().norm(dim=-1)*contact_mask).sum()/den*1000;angle=(normal_error_deg(na,nb).detach()*contact_mask).sum()/den
        temporal=(b.dt[ids]>0);prev=b.prev[ids]
        excess=((q[ids]-b.q[ids])-(q[prev]-b.q[prev])).abs()-b.dt[ids,None]*math.radians(25)
        terms['temporal_rate']=((excess.clamp_min(0)/.035).square().mean(1)*temporal*weights).sum()
        joint_mask=mask[:,m.joint_finger_ids];joint_count=joint_mask.sum(1).clamp_min(1)
        delta=q[ids]-b.q[ids];delta_scale=math.radians(3)
        terms['delta_pose_prior']=320*(((delta/delta_scale).square()*joint_mask).sum(1)/joint_count*weights).sum()
        raw_delta=getattr(b,'last_raw_delta',torch.zeros_like(q))
        valid_joints=b.valid[ids][:,m.joint_finger_ids]
        terms['delta_output_prior']=80*(((raw_delta[ids]/delta_scale).square()*valid_joints).sum(1)/valid_joints.sum(1).clamp_min(1)*weights).sum()
        metrics={}
        if measure:
            orientation_angle=torch.rad2deg(torch.acos((((rotation.detach().transpose(-1,-2)@(align.detach()@b.rawr[ids].detach())).diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1)))
            metrics=dict(sensor_rmse_mm=float((((pred.detach()-(b.rawp[ids].detach()@align.detach().T+trans.detach())).square().sum(-1)*mask).sum()/mask.sum().clamp_min(1)).sqrt())*1000,
                     contact_gap_mm=float(gap) if bool(contact_mask.any()) else None,contact_normal_error_deg=float(angle) if bool(contact_mask.any()) else None,
                     tip_emf_angle_deg=float((torch.rad2deg(torch.acos((actual.detach()*desired.detach().squeeze(-1)).sum(-1).clamp(-1,1)))*mask).sum()/mask.sum().clamp_min(1)))
            metrics.update(emf_orientation_error_deg=float((orientation_angle*mask).sum()/mask.sum().clamp_min(1)),delta_rms_deg=float(((delta.detach().square()*valid_joints).sum()/valid_joints.sum().clamp_min(1)).sqrt())*180/math.pi,
                           delta_max_deg=float((delta.detach().abs()*valid_joints).max())*180/math.pi)
        factor=len(b.rows)/len(ids);terms={k:v*factor for k,v in terms.items()}
        return sum(terms.values()),terms,metrics

    def priors(self):
        m=self.model;rest=m.rest()[0];g=m.geometry;e=m.edges;edge=rest[e[:,1]]-rest[e[:,0]]
        scale=self.base_edges.norm(dim=-1).clamp_min(.0002)
        terms=dict(mesh_displacement=5*(g['residual']/.003).square().mean(),mesh_edge=10*((edge.norm(dim=-1)-scale)/scale).square().mean(),
            local_rigidity=1*((edge-self.base_edges)/scale[:,None]).square().mean(),thickness=.1*(torch.cat((g['width'],g['depth']))/.2).square().mean(),
            geometry_prior=.01*sum(v.square().mean() for k,v in g.items() if k!='residual'))
        if getattr(m,'dimension_mode',None)==DIMENSION_MODE:
            mask=m.dimension_movable
            length=g['length'][mask];width=g['width'][mask];depth=g['depth'][mask]
            terms['dimension_prior']=100*((length/.1).square().mean()+((width/.18).square().mean()+(depth/.18).square().mean()))
            terms['dimension_extreme']=10*((length/.2).pow(4).mean()+(width/.3).pow(4).mean()+(depth/.3).pow(4).mean())
            terms['section_aspect']=100*((width-depth)/.12).square().mean()
        # Laplacian of the directly learned material vertex residual.
        residual=g['residual'];delta=residual[e[:,1]]-residual[e[:,0]]
        terms['mesh_smoothness']=100*(delta/.001).square().mean()
        if len(self.thickness_pairs):
            a,b=self.thickness_pairs.unbind(1);thickness=((rest[a]-rest[b])*self.thickness_axis).sum(-1)/self.thickness0
            terms['mesh_thickness']=320*((.6-thickness).clamp_min(0)*self.thickness0/.001).square().mean()+.1*(thickness-1).square().mean()
        return sum(terms.values()),terms


@torch.no_grad()
def spatial_ema(model,before,before_rest,active,critical_patches=()):
    """EMA on the ACTUAL rest surface, including coupling from bone changes."""
    critical=torch.zeros(len(model.vertices0),device=model.device,dtype=torch.bool)
    for f in range(5):critical|=(model.vertex_finger==f)&(active[f]>0)
    if isinstance(critical_patches,torch.Tensor):critical|=critical_patches
    else:
        for patch in critical_patches:
            if patch.startswith('palm_') and patch in model.regions:
                ids=torch.as_tensor(model.regions[patch],device=model.device);critical[torch.unique(model.faces[ids])]=True
    for k,p in model.geometry.items():
        if k in ('length','width','depth'):
            owners=torch.as_tensor(model.bone_finger,device=model.device);mask=torch.where(owners>=0,active[owners.clamp_min(0)]>0,active.all());p.copy_(torch.where(mask,p,.95*before[k]+.05*p));p[0]=0
        elif k=='axis':
            mask=active[model.joint_finger_ids]>0;p.copy_(torch.where(mask[:,None],p,.95*before[k]+.05*p))
        elif k in ('root','mount_pos','mount_rot'):p.copy_(torch.where((active>0)[:,None],p,.95*before[k]+.05*p))
        elif k in ('alignment','translation'):p.copy_(torch.where(active.all(),p,.95*before[k]+.05*p))
    proposed=model.rest()[0]
    # Compensation is stored as a shared material residual. Inference does
    # not run EMA, update parameters, or reproduce an online compensation step.
    if getattr(model,'dimension_mode',None)!=DIMENSION_MODE:
        model.geometry['residual'].add_(torch.where(critical[:,None],torch.zeros_like(proposed),-.95*(proposed-before_rest)))
    return critical


def fit_capture(model_path,profile,mapping,rows,progress,cancel,method='bone_extrinsics',training_options=None,initial_candidate=None,use_cache=True,compile_loss=False,**kwargs):
    if method=='bone_extrinsics':
        if initial_candidate is not None:raise ValueError('骨长外参标定始终从官方初始化开始')
        from .bone_calibration import fit_capture as fit_bones
        return fit_bones(model_path,profile,mapping,rows,progress,cancel,**kwargs)
    if method=='least_squares':
        if training_options is not None:raise ValueError('最小二乘分支请使用独立solver_options，不使用training_options')
        from .qnbot_least_squares import fit_capture as fit_lsq
        return fit_lsq(model_path,profile,mapping,rows,progress,cancel,initial_candidate=initial_candidate,**kwargs)
    if cancel.is_set():raise InterruptedError('官方mesh拟合已取消')
    if compile_loss:raise ValueError('动态link三角面查询使用独立CUDA Graph；完整Inductor训练编译尚不支持，请关闭compile_loss')
    from .training_options import options,validate_optimizer
    settings=options(training_options);validate_optimizer(method,settings)
    if method not in ('parameters','delta','flashsac'):raise ValueError('未知训练方法')
    began=time.monotonic();device=profile.get('device','cuda:0')
    if str(device).startswith('cuda') and not torch.cuda.is_available():raise ValueError('Qnbot GPU标定需要CUDA')
    run_id=profile.get('run_id') or uuid.uuid4().hex
    def update(value):progress(dict(state='running',surface_backend='qnbot',method=method,run_id=run_id,**value))
    update(dict(phase='prepare',completed=0,total=len(rows),message='准备官方骨架与独立轨迹分区',progress=0.))
    source_hash=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    train=[dict(r,_source_index=i) for i,r in enumerate(rows) if r.get('partition')=='train' and r.get('training_eligible',True)];valid=[dict(r,_source_index=i) for i,r in enumerate(rows) if r.get('partition')=='validation' and r.get('training_eligible',True)]
    if not train or not valid:raise ValueError('需要独立训练与完整复测轨迹，八动作仅用于初始化')
    if any(r.get('partition')=='validation' for r in train):raise ValueError('训练／复测混合')
    model=QnbotSurface(profile['side'],device);model.bind_mapping(mapping)
    if profile.get('material_binding'):model.install_binding(profile['material_binding'])
    # The frozen reference is independent of the current/resumed candidate.
    reference=QnbotSurface(profile['side'],device);reference.bind_mapping(mapping);reference.fixed=True
    acquisition=profile.get('contact_reference')
    if acquisition and acquisition.get('surface'):
        reference.load_artifact(acquisition['surface']);model.load_artifact(acquisition['surface'])
    else:
        init=[r for r in rows if r.get('initialization') and (r.get('skeleton') or {}).get('joint_names')]
        initialize_registration(reference,init or train);initialize_registration(model,init or train)
    if initial_candidate:
        if initial_candidate.get('report',{}).get('fit_revision') not in ('qnbot_trajectory_fit_v5',FIT_REVISION):raise ValueError('link接触目标或拇指限位已更新，请重新训练；旧候选仍可比较')
        if initial_candidate.get('geometry_parameterization')!=DIMENSION_MODE:raise ValueError('旧自由mesh候选不能续训为尺寸模型，请从标准手型重新训练')
        if initial_candidate.get('source_capture_hash')!=source_hash or initial_candidate.get('mapping',{}).get('assignments')!=mapping['assignments']:raise ValueError('续训记录或SID分区不匹配')
        model.load_artifact(initial_candidate)
    geometry=configure_dimensions(model)
    buffer=ReplayBuffer(model,train,reference,update,cancel);buffer.seed_stats(model)
    if initial_candidate and initial_candidate.get('contact_templates'):templates=initial_candidate['contact_templates']
    else:templates=prepare_templates(reference,buffer,cancel,update,steps=kwargs.pop('template_steps',64))
    objective=TrajectoryObjective(model,buffer,templates);policy=DeltaPolicy().to(model.device) if method=='delta' else None
    if initial_candidate and policy is not None:
        if not initial_candidate.get('delta_policy'):raise ValueError('续训缺少官方Delta权重')
        artifact=initial_candidate['delta_policy'];policy=load_policy(artifact,model);buffer.feature_mean=torch.tensor(artifact['mean'],device=model.device);buffer.feature_scale=torch.tensor(artifact['scale'],device=model.device)
    hidden=[]
    if settings['optimizer']=='muon':hidden=[layer.weight for layer in policy.net if isinstance(layer,nn.Linear)][:-1]
    other=[p for p in policy.parameters() if not any(p is h for h in hidden)] if policy else []
    # Only tied finger dimensions are trainable; no vertex-offset optimizer.
    optimizer=torch.optim.Adam([dict(params=geometry,lr=.001),dict(params=other,lr=.00001)])
    muon=torch.optim.Muon(hidden,lr=.00001,weight_decay=0.,adjust_lr_fn='match_rms_adamw') if hidden else None
    compilation_s=0.
    def tensor_step(q,ids):
        loss,terms,_=objective.loss(q,ids,measure=False);regularizer,priors=objective.priors()
        return loss+regularizer,{**terms,**priors}
    loss_step=tensor_step
    if compile_loss and policy is not None:
        update(dict(phase='compile',progress=.2,message='编译固定形状Delta前向',completed=0,total=1,estimated_remaining_s=None))
        t=time.monotonic();policy.forward=torch.compile(policy.forward,options={'triton.cudagraphs':False});policy(buffer.features(model));torch.cuda.synchronize();compilation_s=time.monotonic()-t
    if compile_loss:
        update(dict(phase='compile',progress=.2,message='编译GPU轨迹损失、反向及Adam更新；不计训练步数',completed=0,total=1,estimated_remaining_s=None))
        started=time.monotonic();loss_step=torch.compile(tensor_step,options={'triton.cudagraphs':False},dynamic=False)
        optimizer.step=torch.compile(optimizer.step,options={'triton.cudagraphs':False},dynamic=False)
        warm_ids=torch.arange(min(settings['batch_size'] or 256,len(train)),device=model.device)
        warm_loss,_=loss_step(buffer.corrected(model,policy),warm_ids);warm_loss.backward()
        for p in geometry+list(policy.parameters() if policy else []):
            if p.grad is not None:p.grad.zero_()
        optimizer.step()
        # Warmup must not bias Adam's time/momentum or change any parameters.
        for state in optimizer.state.values():
            for value in state.values():
                if isinstance(value,torch.Tensor):value.zero_()
        optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();compilation_s+=time.monotonic()-started
    # Stratified complete replay coverage; contiguous chunks preserve neighbors.
    rng=np.random.default_rng(27);chunk_size=32;logical=settings['batch_size'] or 256
    groups=[]
    for stage in buffer.stages:
        ids=[i for i,r in enumerate(train) if r['stage']==stage]
        groups.extend(np.asarray(ids[i:i+chunk_size],int) for i in range(0,len(ids),chunk_size))
    packed=np.full((len(groups),chunk_size),-1,np.int64)
    for k,group in enumerate(groups):packed[k,:len(group)]=group
    replay_windows=torch.as_tensor(packed,device=model.device)
    trace=[];last_emit=0.;steps=0;seen=0;best=[];budget=False;collision=None;collision_error=None
    try:
        from .qnbot_collision import QnbotCollision
        collision=QnbotCollision(model)
    except (ValueError,RuntimeError) as exc:collision_error=str(exc)
    order=rng.permutation(len(groups));cursor=0;torch.cuda.reset_peak_memory_stats(model.device) if model.device.type=='cuda' else None
    train_start=time.monotonic();compile_requested=bool(compile_loss)
    preprocessing_s=train_start-began-compilation_s
    # Reserve the full holdout sweep rather than spending 70% regardless of
    # dataset length and routinely timing out during the final safety check.
    validation_reserve_s=min(settings['max_seconds']*.8,max(10.,len(valid)*.006+5.))
    for step in range(settings['adam_steps']):
        if cancel.is_set():raise InterruptedError('官方mesh训练已取消')
        if time.monotonic()-began-compilation_s>settings['max_seconds']-validation_reserve_s:budget=True;break
        if method=='flashsac':
            selected=torch.randint(len(groups),(math.ceil(logical/chunk_size)+1,),device=model.device);ids=replay_windows[selected].flatten();ids=ids[ids>=0][:logical]
        else:
            chosen=[]
            while sum(len(x) for x in chosen)<logical:
                if cursor>=len(order):order=rng.permutation(len(groups));cursor=0
                chosen.append(groups[int(order[cursor])]);cursor+=1
                if len(chosen)>=len(groups):break
            ids=torch.tensor(np.concatenate(chosen)[:logical],device=model.device)
        active=buffer.mask[ids].amax(0);before={k:p.detach().clone() for k,p in model.geometry.items()}
        with torch.no_grad():before_rest=model.rest()[0].clone()
        optimizer.zero_grad(set_to_none=True)
        if muon:muon.zero_grad(set_to_none=True)
        q=buffer.corrected(model,policy)
        if compile_loss:total,terms=loss_step(q,ids);priors={};metrics={}
        else:loss,terms,metrics=objective.loss(q,ids);regularizer,priors=objective.priors();total=loss+regularizer
        if collision is not None:
            collision_loss=training_collision_loss(model,q,ids,buffer,collision)
            total=total+collision_loss;terms['collision']=collision_loss
        total.backward()
        geo_grad=float(torch.sqrt(sum(p.grad.square().sum() for p in geometry if p.grad is not None)));policy_grad=float(torch.sqrt(sum(p.grad.square().sum() for p in policy.parameters() if p.grad is not None))) if policy else 0.
        if not math.isfinite(float(total.detach())) or not math.isfinite(geo_grad) or not math.isfinite(policy_grad):raise ValueError('训练出现非有限损失／梯度；候选未应用')
        optimizer.step()
        if muon:muon.step()
        extra=((buffer.patch_valid[ids].amax(0)>0)[:,None]&buffer.palm_region_vertices).any(0)
        spatial_ema(model,before,before_rest,active,extra)
        steps=step+1;seen+=len(ids);now=time.monotonic();best.append(float(total.detach()));best=best[-20:]
        if now-last_emit>=.75 or steps==settings['adam_steps']:
            if compile_loss:
                with torch.no_grad():_,_,metrics=objective.loss(q.detach(),ids)
            raw=dict(loss=float(total.detach()),beta_kl=None,geometry_kl=None,geometry_prior=float((priors or terms)['geometry_prior'].detach()),geometry_gradient_norm=geo_grad,policy_gradient_norm=policy_grad,
                geometry_learning_rate=.001,mesh_learning_rate=0.,policy_learning_rate=.00001 if policy else 0.,optimizer_name='Muon+Adam' if muon else 'Adam',joint_limit_saturation_fraction=float(((q<=model.limits[:,0]+1e-4)|(q>=model.limits[:,1]-1e-4)).float().mean()),
                terms={k:float(v.detach()) for k,v in {**terms,**priors}.items()},**metrics)
            with torch.no_grad():
                after_q=buffer.corrected(model,policy);after_loss,after_terms,after_metrics=objective.loss(after_q,ids);after_reg,after_priors=objective.priors()
                if collision is not None:
                    after_collision=training_collision_loss(model,after_q,ids,buffer,collision);after_loss=after_loss+after_collision;after_terms['collision']=after_collision
                after=dict(raw,loss=float(after_loss+after_reg),geometry_prior=float(after_priors['geometry_prior']),terms={k:float(v) for k,v in {**after_terms,**after_priors}.items()},**after_metrics)
            previous=(initial_candidate or {}).get('report',{}).get('total_training_steps',(initial_candidate or {}).get('report',{}).get('steps',0))
            point=dict(sample_id=steps,source='train_only',phase='mixed',scope='mixed',step=steps,previous_training_steps=previous,total_training_step=previous+steps,training_frames=int(torch.unique(ids).numel()),batch_samples=len(ids),raw=raw,ema=after,seconds=now-began,epochs_seen=seen/len(train))
            trace.append(point);last_emit=now
            update(dict(phase='mixed',progress=.2+.5*steps/settings['adam_steps'],completed=steps,total=settings['adam_steps'],training_sample=point,message='手指长宽厚与小Delta混训' if policy else '手指长宽厚尺寸混训',estimated_remaining_s=max(0,settings['max_seconds']-(now-began)),eta_scope='budget'))
        if len(best)==20 and max(best)-min(best)<max(abs(best[-1]),1)*1e-5 and geo_grad<1e-3 and policy_grad<1e-3:break
    train_s=time.monotonic()-train_start
    update(dict(phase='validation',progress=.75,completed=0,total=len(valid),message='完整独立复测：曲面、EMF与防穿透检查'))
    artifact=policy_artifact(policy,buffer,model);model.fixed=True;model._rest_cache=None
    validation_started=time.monotonic()
    validation=validate(model,valid,templates,artifact,reference,cancel,update,deadline=began+settings['max_seconds']+compilation_s)
    duration=time.monotonic()-began;reasons=list(validation['rejection_reasons'])
    if collision_error:reasons.append('训练碰撞查询不可用：'+collision_error)
    if duration-compilation_s>60:reasons.append('预热后端到端耗时超过60秒目标')
    accepted=not reasons
    report=dict(validation,fit_revision=FIT_REVISION,surface_backend='qnbot',method=method,run_id=run_id,state='completed',accepted=accepted,
        holdout_validation_passed=accepted,rejection_reasons=reasons,reason='验收通过' if accepted else '；'.join(reasons),frozen_forward=True,online_optimization=False,
        geometry_parameterization=DIMENSION_MODE,independent_geometry_parameters=15,frozen_geometry=['residual','root','axis','mount_pos','mount_rot','alignment','translation'],
        finger_dimension_scales={f:{k:float(model.geometry[k].detach()[torch.nonzero(model.dimension_movable&(model.dimension_owners==i))[0,0]].exp()) for k in ('length','width','depth')} for i,f in enumerate(FINGERS)},
        selected_training_frames=len(train),selected_validation_frames=len(valid),capture_protocol=rows[0].get('capture_protocol'),training_options=settings,steps=steps,epochs_seen=seen/len(train),training_s=train_s,
        preprocessing_s=preprocessing_s,validation_s=time.monotonic()-validation_started,validation_reserve_s=validation_reserve_s,
        compilation_s=compilation_s,end_to_end_s=duration,time_target_met=duration-compilation_s<=60,performance_passed=duration-compilation_s<=60 and accepted,validation_complete=validation['validation_complete'],
        peak_torch_memory_bytes=torch.cuda.max_memory_allocated(model.device) if model.device.type=='cuda' else None,source_capture_hash=source_hash,
        solver=dict(steps=steps,budget_exhausted=budget,converged=steps<settings['adam_steps'] and not budget,joint_geometry_policy=policy is not None,training_trace=dict(points=trace),optimizer_name='Muon+Adam' if muon else 'Adam',learning_rates=dict(geometry=.001,mesh_metres=0.,policy=.00001)),
        loss_spec=dict(contact_position=320.,contact_normal=40.,sensor_position=320.,sensor_orientation=40.,tip_emf_orientation=40.,delta_pose_prior=320.,delta_output_prior=80.,delta_scale_deg=3.,geometry_primary=True,contact_penetration_tolerance_mm=2.,noncontact_penetration_tolerance_mm=0.,contact_distance_target_mm=0.,contact_normal_target_dot=-1.,contact_normal_dead_zone_deg=0.,contact_geometry='current_original_link_region_triangles',collision_mean_violating_weight=160.,collision_worst_weight=160.,collision_training_frames_per_step=3,witness_selection='joint_distance_and_antiparallel_normals_at_same_points'),
        thumb_extension_deg=model.thumb_extension_deg,thumb_joint_limits_rad={n:model.limits[model.joint_names.index(n)].cpu().tolist() for n in model.thumb_straight},
        replay_sampling='gpu_random_windows_with_replacement' if method=='flashsac' else 'shuffled_sequence_coverage',
        contact_templates=templates,geometry_change=dict(rms_vertex_residual_mm=float(model.geometry['residual'].detach().square().sum(-1).mean().sqrt())*1000,
        max_vertex_residual_mm=float(model.geometry['residual'].detach().norm(dim=-1).max())*1000),cache=dict(hit=False,raw_buffer_preprocessing_included=True),compile_requested=compile_requested,compiled_training_step=bool(compile_loss))
    previous=(initial_candidate or {}).get('report',{}).get('total_training_steps',(initial_candidate or {}).get('report',{}).get('steps',0));report.update(previous_training_steps=previous,total_training_steps=previous+steps)
    candidate=dict(model.artifact(),geometry_parameterization=DIMENSION_MODE,serial=profile['serial'],mapping=copy.deepcopy(mapping),source_capture_hash=source_hash,calibration_fingerprint=profile.get('calibration_fingerprint'),
        skeleton_runtime_revision=profile.get('skeleton_runtime_revision'),delta_policy=artifact,contact_templates=templates,report=report,capture_session_id=profile.get('capture_session_id'))
    if artifact:artifact['geometry_hash']=model.version()
    return candidate,report


def validate(model,rows,templates,policy_art,reference,cancel,progress,deadline=None):
    from .qnbot_collision import QnbotBatchCollision
    validation_began=time.monotonic();buffer=ReplayBuffer(model,rows,reference,cancel=cancel)
    if policy_art:
        policy=load_policy(policy_art,model);buffer.feature_mean=torch.tensor(policy_art['mean'],device=model.device);buffer.feature_scale=torch.tensor(policy_art['scale'],device=model.device)
    else:policy=None
    objective=TrajectoryObjective(model,buffer,templates);gaps=[];normals=[];per_action={};sensor_sum=torch.zeros(5,device=model.device);sensor_count=torch.zeros(5,device=model.device)
    maximum=0.;checked=0;collision_error=None;complete=True;release=[];orientation=[];seed_orientation=[];delta_values=[];clusters=[];preferred=[];minimum_gaps=[]
    far_threshold={}
    for stage in buffer.stages:
        values=[r['reference_gap_m'] for r in rows if r['stage']==stage and isinstance(r.get('reference_gap_m'),(float,int))]
        if values:far_threshold[stage]=min(values)+.6*(max(values)-min(values))
    batch_size=32
    try:collision=QnbotBatchCollision(model,batch=batch_size)
    except (ValueError,RuntimeError) as exc:collision=None;collision_error=str(exc)
    with torch.no_grad():
        q=buffer.corrected(model,policy)
        link_points,link_normals=objective.link_contact(q)
        for start in range(0,len(rows),batch_size):
            if cancel.is_set():raise InterruptedError('独立验收已取消')
            if deadline and time.monotonic()>deadline:complete=False;break
            end=min(len(rows),start+batch_size);ids=torch.arange(start,end,device=model.device);p,n,j,r=objective.surface(q[ids]);pred,sr=model.sensor_prediction(q[ids]);_,seed_sr=model.sensor_prediction(buffer.q[ids]);align,trans=model.alignment()
            for rr,result in ((sr,orientation),(seed_sr,seed_orientation)):
                relative=rr.transpose(-1,-2)@(align@buffer.rawr[ids]);angles=torch.rad2deg(torch.acos(((relative.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1)))
                result.extend(angles[buffer.mask[ids]>0].cpu().tolist())
            jm=buffer.valid[ids][:,model.joint_finger_ids]>0
            delta_values.extend(torch.rad2deg((q[ids]-buffer.q[ids]).abs())[jm].cpu().tolist())
            sensor_sum+=(((pred-(buffer.rawp[ids]@align.T+trans)).square().sum(-1))*buffer.mask[ids]).sum(0);sensor_count+=buffer.mask[ids].sum(0)
            vertices=model(q[ids])[0]
            if collision:
                penetration=collision.query(vertices);maximum=max(maximum,float(penetration.max()));checked+=len(ids)
            for k,i in enumerate(range(start,end)):
                group=objective.link_by_row.get(i,[]);records=[]
                for e in group:
                    gap=float((link_points[e,0]-link_points[e,1]).norm())*1000;normal=float(normal_error_deg(link_normals[e,0],link_normals[e,1]));edge=objective.link_edges[e];info=dict(regions=edge['regions'],required=edge['required'])
                    if info['required']:gaps.append(gap);normals.append(normal);minimum_gaps.append(float(objective.last_link_minimum[e])*1000)
                    else:preferred.append(dict(stage=rows[i]['stage'],timestamp_ms=rows[i]['timestamp_ms'],regions=info['regions'],gap_mm=gap))
                    records.append(dict(info,gap_mm=gap,normal_error_deg=normal))
                if records:per_action.setdefault(rows[i]['stage'],[]).extend(records)
                if records and target(rows[i]['stage'])['kind']=='distal_cluster':
                    nodes=target(rows[i]['stage'])['patches'];seen={nodes[0]}
                    for _ in nodes:
                        for record in records:
                            a,b=record['regions']
                            if record['gap_mm']<=5 and (a in seen or b in seen):seen.update((a,b))
                    clusters.append(dict(stage=rows[i]['stage'],timestamp_ms=rows[i]['timestamp_ms'],connected=len(seen)==len(nodes)))
                legacy_far=rows[i].get('capture_protocol')!='contact_motion_v4' and not rows[i].get('contact_confirmed') and rows[i].get('reference_gap_m',-1)>=far_threshold.get(rows[i]['stage'],float('inf'))
                if rows[i].get('flat_valid') or legacy_far:
                    for edge in objective.templates.get(rows[i]['stage'],[]):
                        pp,_=model.anchors(vertices[k:k+1],edge['anchors']);release.append(float((pp[0,0]-pp[0,1]).norm())*1000)
            progress(dict(phase='validation',completed=end,total=len(rows),progress=.75+.25*end/len(rows),estimated_remaining_s=(time.monotonic()-validation_began)*(len(rows)-end)/max(end,1),message='完整留出轨迹逐帧检查原始曲面'))
    gap_mean=float(np.mean(gaps)) if gaps else None;gap_max=max(gaps) if gaps else None;rmse=(sensor_sum/sensor_count.clamp_min(1)).sqrt().cpu().numpy()*1000
    reasons=[]
    if not complete or checked!=len(rows):reasons.append('预算内未完成全部留出帧的原始三角面验收')
    if not gaps:reasons.append('独立复测缺少接触事件')
    if gaps and (gap_mean>3 or gap_max>5):reasons.append('独立复测接触间隙超过3/5mm')
    if normals and max(normals)>15:reasons.append('目标接触法向超出15°容差')
    if maximum>2:reasons.append('原始曲面穿透超过2mm')
    if any(x>15 for x in rmse):reasons.append('目标手指EMF残差超过15mm')
    if release and min(release)<2:reasons.append('展平回位仍错误吸合')
    if orientation and np.mean(orientation)>15:reasons.append('目标手指与EMF的平均朝向误差超过15°')
    if policy_art and orientation and np.mean(orientation)>np.mean(seed_orientation)+2:reasons.append('Delta使EMF朝向误差相对原IK增加超过2°')
    if collision_error:reasons.append('无法建立可靠原始曲面闭合分区：'+collision_error)
    if any(not c['connected'] for c in clusters):reasons.append('多指接触簇未连通')
    return dict(validation_scope='separate_complete_contact_pass',validation_complete=complete and checked==len(rows),validation_frames_checked=checked,
        contact_gap_mm=gap_mean,contact_max_gap_mm=gap_max,contact_normal_error_deg=float(np.mean(normals)) if normals else None,max_penetration_mm=maximum,
        contact_distance_query='moving_original_link_triangles_joint_distance_facing_witnesses',contact_normals='area_weighted_surface_vertex_normals_at_selected_same_points',
        minimum_link_gap_mm=float(np.mean(minimum_gaps)) if minimum_gaps else None,maximum_minimum_link_gap_mm=max(minimum_gaps) if minimum_gaps else None,
        sensor_rmse_mm=dict(zip(FINGERS,map(float,rmse))),contact_stages=[dict(stage=s,edges=e) for s,e in per_action.items()],contact_clusters=clusters,preferred_thumb_contacts=preferred,
        rejection_reasons=reasons,collision_query='original_qnbot_triangles_all_vertices_and_face_centroids',
        release_min_gap_mm=min(release) if release else None,validation_pose_source='fixed_qnbot_geometry_and_same_delta_forward',
        emf_orientation_error_deg=float(np.mean(orientation)) if orientation else None,emf_orientation_max_deg=max(orientation) if orientation else None,
        seed_emf_orientation_error_deg=float(np.mean(seed_orientation)) if seed_orientation else None,
        delta_correction_deg=dict(mean=float(np.mean(delta_values)),rms=float(np.sqrt(np.mean(np.square(delta_values)))),p95=float(np.percentile(delta_values,95)),maximum=max(delta_values)) if delta_values else None)
