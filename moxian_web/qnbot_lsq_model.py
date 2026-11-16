"""Low dimensional, functional native-hand calibration and measured evidence.

All fitted coordinates are dimensionless. Geometry expands into the existing
fixed-forward ABI; no per-vertex or per-frame parameters are deployed.
"""
from __future__ import annotations
import math
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from .qnbot_surface import FINGERS, normals
from .contact import PATCH_IDS
from .calibration_targets import target
from retarget.baseline.card.math import log_so3

PARAMETER_REVISION = 'qnbot_lsq_coordinates_v1'
STAGES = ('extrinsics', 'dimensions', 'kinematics', 'tactile', 'surface')
SLICES = dict(dimensions=slice(0,15),extrinsics=slice(15,51),kinematics=slice(51,69),tactile=slice(69,129),surface=slice(129,149))
ROOT_FINGERS = (0,1,3,4)


def robust(r):
    """Squared result is pseudo-Huber; priors are deliberately not robustified."""
    return r/(1+(r/3).square()).pow(.25)


def observed(rows, device):
    positions=[];rotations=[];valid=[];rotation_valid=[]
    for row in rows:
        pp=[];rr=[];vv=[];rv=[]
        for finger in FINGERS:
            node=(row.get('fingers') or {}).get(finger) or {}
            p=np.asarray(node.get('position',[]),float);q=np.asarray(node.get('quaternion_xyzw',[]),float)
            good=p.shape==(3,) and np.isfinite(p).all() and node.get('valid') is not False and node.get('confidence',1)>0
            orient=good and q.shape==(4,) and np.isfinite(q).all() and np.linalg.norm(q)>.1
            pp.append(p if good else np.zeros(3));rr.append(Rotation.from_quat(q).as_matrix() if orient else np.eye(3));vv.append(good);rv.append(orient)
        positions.append(pp);rotations.append(rr);valid.append(vv);rotation_valid.append(rv)
    return tuple(torch.tensor(np.asarray(v),device=device,dtype=torch.float32) for v in (positions,rotations,valid,rotation_valid))


def contact_evidence(row, assignments):
    """Only fresh, explicit two-sided taxel evidence. No event/proxy fallback."""
    c=row.get('contact') or {};stamp=row.get('tactile_timestamp_ms')
    fresh=c.get('ready') is True and isinstance(stamp,(int,float)) and math.isfinite(stamp) and abs(stamp-row['timestamp_ms'])<=25
    values=np.asarray(c.get('normalized') or [],float)
    cells=c.get('cell_contact') or []
    patch_state={p.get('id'):p.get('contact') for p in c.get('patches',[])}
    weights=np.zeros(460,np.float32);states=np.full(20,-1,np.int8)
    for patch,name in enumerate(PATCH_IDS):
        ids=np.flatnonzero(np.asarray(assignments)==patch)
        good=[i for i in ids if fresh and i<len(values) and np.isfinite(values[i]) and i<len(cells) and isinstance(cells[i],(bool,np.bool_))]
        if not good:continue
        states[patch]=1 if patch_state.get(name) is True and any(cells[i] for i in good) else 0 if patch_state.get(name) is False and not any(cells[i] for i in good) else -1
        active=[i for i in good if cells[i]]
        selected=active if active else good
        ww=np.maximum(values[selected],0) if active else np.ones(len(selected))
        if ww.sum()<=0:ww=np.ones(len(selected))
        weights[selected]=ww/ww.sum()
    edges=[]
    for edge in target(row.get('stage')).get('edges',[]):
        a,b=edge['regions']
        if a not in PATCH_IDS or b not in PATCH_IDS:continue
        a,b=PATCH_IDS.index(a),PATCH_IDS.index(b)
        # Single-sided and unknown readings are ambiguous, never positive labels.
        label=1 if states[a]==states[b]==1 else 0 if states[a]==states[b]==0 else -1
        if label>=0:edges.append((a,b,label))
    return weights,edges,states


class Coordinates:
    size=149
    def __init__(self, model):
        self.model=model;self.device=model.device
        self.base={k:v.detach().clone() for k,v in model.geometry.items()}
        # Calibration always starts with the standard skin, not a learned mesh.
        for value in self.base.values():value.zero_()
        owners=torch.tensor(model.bone_finger,device=self.device)
        movable=torch.tensor([f>=0 and not n.endswith('_metacarpal') for f,n in zip(model.bone_finger,model.bone_names)],device=self.device)
        self.owner=owners.clamp_min(0);self.movable=movable
        self.root_indices=torch.tensor(ROOT_FINGERS,device=self.device)
        axes=model.joint_axis0[:3]
        aux=torch.eye(3,device=self.device)[axes.abs().argmin(-1)]
        first=torch.nn.functional.normalize(torch.linalg.cross(axes,aux),dim=-1)
        second=torch.linalg.cross(axes,first)
        self.axis_basis=torch.stack((first,second),-1)
        self.base_bc=torch.tensor(model.sid_bc,device=self.device)
        self.patch=torch.tensor(model.mapping['assignments'],device=self.device,dtype=torch.long)
        self.sid_tri=model.faces[torch.tensor(model.sid_faces,device=self.device)]
        self.support=torch.unique(self.sid_tri.flatten(),sorted=True)
        self.local_tri=torch.searchsorted(self.support,self.sid_tri)
        self.surface_basis=torch.zeros((len(model.vertices0),3,20),device=self.device)
        with torch.no_grad():self.reference_rest=model.rest(geometry=self.base)[0].detach()
        self.surface_basis_report=dict(enabled=False,reason='not_yet_supported_by_training_residuals')
        self.lower=torch.full((self.size,),-2.,device=self.device);self.upper=-self.lower
        self.lower[:5]=-1.5;self.upper[:5]=1.5
        self.lower[5:15]=-1.;self.upper[5:15]=1.
        self.lower[30:45]=-3.;self.upper[30:45]=3.
        self.lower[63:69]=-math.radians(5)/.05;self.upper[63:69]=-self.lower[63:69]
        self.lower[129:]=-1.;self.upper[129:]=1.
        self.names=[]
        for kind in ('length','width','depth'):
            self.names.extend(f'{kind}.{f}' for f in FINGERS)
        for kind in ('mount_position_cm','mount_rotation_0.1rad'):
            self.names.extend(f'{kind}.{f}.{a}' for f in FINGERS for a in 'xyz')
        self.names.extend(f'{k}.{a}' for k in ('alignment_0.1rad','translation_cm') for a in 'xyz')
        self.names.extend(f'root_mm.{FINGERS[f]}.{a}' for f in ROOT_FINGERS for a in 'xyz')
        self.names.extend(f'thumb_cmc_axis.{j}.{a}' for j in range(3) for a in range(2))
        self.names.extend(f'tactile.{p}.{a}' for p in PATCH_IDS for a in ('barycentric_u','barycentric_v','normal_mm'))
        self.names.extend(f'surface.{p}' for p in PATCH_IDS)

    def geometry(self, x):
        g=dict(self.base)
        for i,key in enumerate(('length','width','depth')):
            g[key]=.1*x[i*5:(i+1)*5][self.owner]*self.movable
        g['mount_pos']=x[15:30].reshape(5,3)
        g['mount_rot']=.1*x[30:45].reshape(5,3)
        g['alignment']=.1*x[45:48];g['translation']=x[48:51]
        g['root']=torch.zeros_like(g['root']).index_copy(0,self.root_indices,.1*x[51:63].reshape(4,3))
        tilt=(self.axis_basis@(.05*x[63:69].reshape(3,2))[...,None]).squeeze(-1)
        g['axis']=torch.cat((tilt,torch.zeros_like(g['axis'][3:])))
        g['residual']=torch.einsum('vck,k->vc',self.surface_basis,.001*x[129:149])
        return g

    def binding(self,x):
        shifts=x[69:129].reshape(20,3)[self.patch]
        logits=self.base_bc.clamp_min(1e-8).log()+torch.cat((.25*shifts[:,:2],torch.zeros_like(shifts[:,:1])),-1)
        return torch.softmax(logits,-1),.001*shifts[:,2]

    def skin(self,x,q):
        v=self.model(q,vertex_ids=self.support,geometry=self.geometry(x))[0]
        tri=v[:,self.local_tri];bc,offset=self.binding(x)
        normal=torch.nn.functional.normalize(torch.linalg.cross(tri[:,:,1]-tri[:,:,0],tri[:,:,2]-tri[:,:,0],dim=-1),dim=-1)
        points=(tri*bc[None,...,None]).sum(-2)+offset[None,:,None]*normal
        return points,normal

    def install(self,x):
        with torch.no_grad():
            for k,value in self.geometry(x).items():self.model.geometry[k].copy_(value)
        bc,offset=self.binding(x)
        self.model.install_binding(dict(assignments=self.model.mapping['assignments'],face_ids=self.model.sid_faces.tolist(),barycentric=bc.detach().cpu().tolist(),normal_offset_m=offset.detach().cpu().tolist()))
        self.model._rest_cache=None

    def build_surface_basis(self,x):
        """Smooth regional normal modes, orthogonal to rigid/size/kinematic modes."""
        import scipy.sparse as sp
        from scipy.sparse.linalg import spsolve
        m=self.model;faces=m.faces.cpu().numpy();edges=m.edges.cpu().numpy();n=len(m.vertices0)
        with torch.no_grad():rest=m.rest(geometry=self.geometry(x))[0];normal=normals(rest,m.faces)
        labels=m.mapping['assignments'];weights=np.zeros((n,20))
        for k in range(20):
            ids=np.flatnonzero(np.asarray(labels)==k)
            if len(ids):weights[np.unique(self.sid_tri[ids].cpu()) ,k]=1.
        a=sp.coo_matrix((np.ones(len(edges)*2),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(n,n)).tocsr()
        lap=sp.diags(np.asarray(a.sum(1)).ravel())-a
        smooth=spsolve(sp.eye(n,format='csc')+4*lap.tocsc(),weights)
        basis=torch.tensor(smooth,device=self.device,dtype=torch.float32)[:,None,:]*normal[:,:,None]
        # Rest-space nuisance Jacobian, independent of any held-out observations.
        ids=torch.tensor(list(range(15))+list(range(51,69)),device=self.device)
        def rest_fn(z):return m.rest(geometry=self.geometry(x.index_copy(0,ids,z)))[0]
        jac=torch.func.jacfwd(rest_fn)(x[ids]).reshape(-1,len(ids))
        rigid=[torch.eye(3,device=self.device)[i].expand(n,3) for i in range(3)]
        rigid += [torch.linalg.cross(torch.eye(3,device=self.device)[i].expand(n,3),rest) for i in range(3)]
        nuisance=torch.cat((jac,torch.stack(rigid,-1).reshape(-1,6)),1)
        u,s,_=torch.linalg.svd(nuisance,full_matrices=False);u=u[:,s>s.max()*1e-5]
        flat=basis.reshape(-1,20);flat=flat-u@(u.T@flat)
        u2,s2,_=torch.linalg.svd(flat,full_matrices=False);rank=int((s2>s2.max()*1e-4).sum())
        out=torch.zeros_like(flat);out[:,:rank]=u2[:,:rank]
        out=out.reshape(n,3,20);out=out/out.norm(dim=1).amax(0).clamp_min(1e-10)[None,None,:]
        self.surface_basis=out.detach()
        self.surface_basis_report=dict(enabled=True,rank=rank,maximum_modes=20,construction='graph_smoothed_regional_normals_projected_off_rigid_dimensions_and_kinematics',heldout_used=False)


class Measurements:
    def __init__(self,model,rows,training=True,noise=None):
        self.model=model;self.rows=rows;self.device=model.device
        if training and any(r.get('partition')!='train' for r in rows):raise ValueError('最小二乘统计和参数更新只能读取训练分区')
        self.p,self.r,self.valid,self.rotation_valid=observed(rows,self.device)
        self.q0=model.angles(rows).detach()
        self.low=torch.minimum(model.limits[:,0],self.q0);self.high=torch.maximum(model.limits[:,1],self.q0)
        specs=[target(r.get('stage')) for r in rows]
        self.mask=torch.tensor([[float(f in s.get('active_fingers',FINGERS)) for f in FINGERS] for s in specs],device=self.device)*self.valid
        if not training:self.mask=self.valid.clone()
        self.rotation_mask=self.mask*self.rotation_valid
        counts={s:sum(r['stage']==s for r in rows) for s in {r['stage'] for r in rows}}
        self.weight=torch.tensor([1/counts[r['stage']]/len(counts) for r in rows],device=self.device)
        self.prev=torch.arange(len(rows),device=self.device);self.dt=torch.zeros(len(rows),device=self.device)
        self.previous2=self.prev.clone()
        for i,r in enumerate(rows):
            if not i:continue
            before=rows[i-1];delta=(r['timestamp_ms']-before['timestamp_ms'])/1000
            if all(r.get(k)==before.get(k) for k in ('window_id','stage','partition','cycle')) and 0<delta<=.25:
                self.prev[i]=i-1;self.dt[i]=delta
                self.previous2[i]=i-1
                if self.dt[i-1]>0:self.previous2[i]=self.prev[i-1]
        evidence=[contact_evidence(r,model.mapping['assignments']) for r in rows]
        self.sid_weight=torch.tensor(np.array([e[0] for e in evidence]),device=self.device)
        self.states=np.array([e[2] for e in evidence])
        self.edges=[e[1] for e in evidence]
        # Four required edges at most in the current reviewed action contract.
        edge_ids=np.zeros((len(rows),4,2),np.int64);labels=np.full((len(rows),4),-1,np.float32)
        for i,(_,edges,_) in enumerate(evidence):
            if len(edges)>4:raise ValueError('接触边容量不足，不能静默截断')
            for k,(a,b,label) in enumerate(edges):edge_ids[i,k]=[a,b];labels[i,k]=label
        self.edge_ids=torch.tensor(edge_ids,device=self.device);self.labels=torch.tensor(labels,device=self.device)
        self.noise=noise or self.estimate_noise()
        self.noise=dict(self.noise)

    def estimate_noise(self):
        # Robust high-pass statistics on training measurements only; model errors
        # are not interpreted as independent measurement noise.
        good=[i for i in range(2,len(self.rows)) if int(self.prev[i])==i-1 and int(self.prev[i-1])==i-2]
        sigma_p=.003;sigma_r=.15
        if good:
            ids=torch.tensor(good,device=self.device)
            d=self.p[ids]-2*self.p[ids-1]+self.p[ids-2]
            valid=self.valid[ids]*self.valid[ids-1]*self.valid[ids-2]
            a=d[valid>0]
            if a.numel():sigma_p=float(a.abs().median()/.67449/math.sqrt(6))
            relative=self.r[ids-1].transpose(-1,-2)@self.r[ids]
            angles=log_so3(relative)[self.rotation_valid[ids]*self.rotation_valid[ids-1]>0]
            if angles.numel():sigma_r=float(angles.abs().median()/.67449/math.sqrt(2))
        return dict(position_scale_m=float(np.clip(sigma_p,.001,.01)),rotation_scale_rad=float(np.clip(sigma_r,.05,.3)),source='train_only_robust_temporal_highpass_with_floors')

    def summary(self):
        return dict(frames=len(self.rows),positive_contact_edges=int((self.labels==1).sum()),negative_contact_edges=int((self.labels==0).sum()),
                    positive_contact_frames=int((self.labels==1).any(1).sum()),contact_stages=sorted({r['stage'] for i,r in enumerate(self.rows) if bool((self.labels[i]==1).any())}),
                    temporal_links=int((self.dt>0).sum()),noise=self.noise,
                    disabled=dict(joint_supervision='EMF_derived_IK_is_only_a_seed',object_sdf='no_known_object_pose',force_gain='no_known_force_reference',joint_offsets_and_coupling='confounded_with_latent_pose'),
                    skipped_contact_stages=sorted({r['stage'] for i,r in enumerate(self.rows) if target(r['stage']).get('patches') and not bool((self.labels[i]==1).any())}-{r['stage'] for i,r in enumerate(self.rows) if bool((self.labels[i]==1).any())}))


class Residuals:
    def __init__(self,coords,data):
        self.c=coords;self.d=data;self.m=coords.model;self.witnesses=[]

    def measurement(self,x,q,ids,contact=True):
        c,d,m=self.c,self.d,self.m;g=c.geometry(x)
        pred,rot=m.sensor_prediction(q,geometry=g);align,trans=m.alignment(g)
        position=robust((pred-(d.p[ids]@align.T+trans))/d.noise['position_scale_m'])*d.mask[ids,:,None]
        orientation=robust(log_so3(rot.transpose(-1,-2)@(align@d.r[ids]))/d.noise['rotation_scale_rad'])*d.rotation_mask[ids,:,None]
        parts=[position.flatten(1),orientation.flatten(1)]
        if contact:
            points,normal=c.skin(x,q)
            weight=d.sid_weight[ids]
            patch=c.patch[None,:,None].expand(len(q),-1,3)
            shape=(len(q),20,3)
            p=torch.zeros(shape,device=q.device).scatter_add(1,patch,points*weight[...,None])
            n=torch.nn.functional.normalize(torch.zeros(shape,device=q.device).scatter_add(1,patch,normal*weight[...,None]),dim=-1)
            edges=d.edge_ids[ids];b=torch.arange(len(q),device=q.device)[:,None]
            pa,pb=p[b,edges[:,:,0]],p[b,edges[:,:,1]]
            na,nb=n[b,edges[:,:,0]],n[b,edges[:,:,1]]
            labels=d.labels[ids];positive=(labels==1)[...,None]
            parts.append(2*robust((pa-pb)/.002)*positive)
            # Outward normals oppose at a self contact.
            parts.append((na+nb)/math.radians(15)*positive)
            distance=(pa-pb).norm(dim=-1)
            logits=(.002-distance)/.002
            bce=torch.nn.functional.binary_cross_entropy_with_logits(logits,labels.clamp_min(0),reduction='none')
            parts.append(.5*torch.sqrt(2*bce+1e-8)*(labels>=0))
        return torch.cat([v.flatten(1) for v in parts],1)

    def frame(self,x,q,ids,previous=None,contact=True):
        d=self.d
        parts=[self.measurement(x,q,ids,contact)]
        valid=d.mask[ids][:,self.m.joint_finger_ids]
        parts.append(.35*(q-d.q0[ids])/.5*valid)
        if previous is not None:
            # Timestamp-aware acceleration of the correction, not of the raw IK.
            correction=q-d.q0[ids];prior=previous-d.q0
            p=d.prev[ids];p2=d.previous2[ids];dt=d.dt[ids]
            old_dt=d.dt[p].clamp_min(.001)
            velocity=(correction-prior[p])/dt.clamp_min(.001)[:,None]
            old_velocity=(prior[p]-prior[p2])/old_dt[:,None]
            active=(dt>0)&(p2!=p)
            parts.append(.02*(velocity-old_velocity)*active[:,None]*valid)
        if contact and self.witnesses:
            width=max(len(w['sign']) for _,w in self.witnesses)
            if width:
                local={i:k for k,i in enumerate(ids.cpu().tolist())};result=torch.zeros((len(q),width),device=q.device)
                for row,w in self.witnesses:
                    if row not in local or not len(w['sign']):continue
                    support=torch.unique(torch.cat((w['source'].flatten(),w['target'].flatten())),sorted=True)
                    vv=self.m(q[local[row]:local[row]+1],vertex_ids=support,geometry=self.c.geometry(x))[0][0]
                    a=torch.searchsorted(support,w['source']);b=torch.searchsorted(support,w['target'])
                    pa=(vv[a]*w['source_bc'][...,None]).sum(-2);pb=(vv[b]*w['target_bc'][...,None]).sum(-2)
                    signed=(pa-pb).norm(dim=-1)*w['sign']
                    rr=4*((-signed-w.get('tolerance',.002))/.002).clamp_min(0)/math.sqrt(len(signed))
                    padded=torch.nn.functional.pad(rr,(0,width-len(rr)))
                    result=result.index_copy(0,torch.tensor([local[row]],device=q.device),padded[None])
                parts.append(result)
        return torch.cat(parts,1)

    def prior(self,x):
        dims=x[:15]
        rest=self.m.rest(geometry=self.c.geometry(x))[0]
        base=self.c.reference_rest;edges=self.m.edges
        length=(rest[edges[:,1]]-rest[edges[:,0]]).norm(dim=-1)
        initial=(base[edges[:,1]]-base[edges[:,0]]).norm(dim=-1)
        surface=self.c.geometry(x)['residual']
        parts=[.2*dims, .1*x[15:51], .3*x[51:69], .5*x[69:129],x[129:149],
               ((length-initial)/initial.clamp_min(.0002)/.2)/math.sqrt(len(edges)),
               (surface[edges[:,0]]-surface[edges[:,1]]).flatten()/.001/math.sqrt(3*len(edges)),
               ((surface.norm(dim=-1)-.003).clamp_min(0)/.001)/math.sqrt(len(rest))]
        return torch.cat(parts)

    def collision(self,x,q_all):
        result=[]
        for row,w in self.witnesses:
            ids=torch.unique(torch.cat((w['source'].flatten(),w['target'].flatten())),sorted=True)
            v=self.m(q_all[row:row+1],vertex_ids=ids,geometry=self.c.geometry(x))[0][0]
            a=torch.searchsorted(ids,w['source']);b=torch.searchsorted(ids,w['target'])
            pa=(v[a]*w['source_bc'][...,None]).sum(-2);pb=(v[b]*w['target_bc'][...,None]).sum(-2)
            signed=(pa-pb).norm(dim=-1)*w['sign']
            r=((-signed-w.get('tolerance',.002))/.002).clamp_min(0)
            if r.numel():result.append(4*r/math.sqrt(len(r)))
        return torch.cat(result)/math.sqrt(max(1,len(result))) if result else x[:0]
