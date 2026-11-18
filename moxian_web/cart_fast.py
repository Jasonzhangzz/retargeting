"""GPU observation storage and batched, semantically balanced CARD costs.

Only immutable prepared observations enter this path. Policy contact witnesses
come from continuous fields; native selection is restricted to independent audit.
"""
from collections import Counter
import numpy as np
import torch
from torch.nn import functional as F
from .cart_contacts import link_owner

DEFAULT_BATCH_SIZE = 0  # actual simultaneous windows, selected from free memory


class ObservationStore:
    def __init__(self, rows, device):
        self.device=device
        names=dict(future_emf='emf_pose',emf_valid='emf_valid',tactile_positions='tactile_positions_m',
                   tactile_state='tactile_state',tactile_valid='tactile_valid')
        self.values={k:torch.as_tensor(np.asarray([r[v] for r in rows]),device=device,
            dtype=torch.bool if k.endswith('valid') else torch.float32) for k,v in names.items()}
        self.stamps=torch.as_tensor([r['timestamp_ms'] for r in rows],device=device,dtype=torch.float64)
        self.cpu_stamps=np.asarray([r['timestamp_ms'] for r in rows],np.float64)
        self.initial_states=torch.as_tensor(np.asarray([r.get('ik_q_rad',[0.]*20) for r in rows]),device=device,dtype=torch.float32) if any('ik_q_rad' in r for r in rows) else None
        v=self.values;ev=v['emf_valid'];tv=v['tactile_valid']
        if not torch.isfinite(v['future_emf'][ev]).all() or not torch.isfinite(v['tactile_positions'][tv]).all():
            raise ValueError('Nonfinite prepared CART observation')
        state=v['tactile_state'][tv]
        if not ((state==0)|(state==1)).all():raise ValueError('Prepared tactile state is not binary')
        if not torch.isfinite(self.stamps).all():raise ValueError('Invalid observation timestamps')
        for i,row in enumerate(rows):
            row['_cart_observation']={k:value[i] for k,value in v.items()}
            active=np.asarray(row['tactile_valid'],bool)&np.asarray(row['tactile_state'],bool)
            ids=torch.as_tensor(np.flatnonzero(active),device=device)
            row['_cart_active_points']=v['tactile_positions'][i,ids]


class PreparedBatch(list):
    def __init__(self, windows, device, normalizers=None):
        super().__init__(windows);self.device=device;self.normalizers=normalizers

    def future_inputs(self, offsets):
        # Group by session so each field has one device gather per session,
        # rather than B*6 host conversions on every rollout step.
        groups={}
        for b,(window,i) in enumerate(zip(self,offsets)):
            store=window['store'];groups.setdefault(store,[]).append((b,window['indices'][i:i+11]))
        outputs={};order=[]
        for store,items in groups.items():
            ids=torch.as_tensor([idx for _,idx in items],device=self.device,dtype=torch.long)
            order.extend(b for b,_ in items)
            for key,value in store.values.items():outputs.setdefault(key,[]).append(value[ids[:,1:]])
            stamps=store.stamps[ids]
            outputs.setdefault('future_dt',[]).append(((stamps[:,1:]-stamps[:,:1])/1000).float())
        inverse=torch.as_tensor(np.argsort(order),device=self.device)
        return {k:torch.cat(values,0)[inverse] for k,values in outputs.items()}


def prepare_windows(sessions, windows, device):
    stores={meta['id']:ObservationStore(rows,device) for meta,rows,_ in sessions}
    for window in windows:
        # Window rows retain their exact session index, including boundary warmup.
        if 'indices' not in window:raise ValueError('Prepared window requires session indices')
        store=stores[window['dataset']];delta=np.diff(store.cpu_stamps[window['indices']])
        if np.any(delta<=0) or np.any(delta>250):raise ValueError('Prepared window crosses a stream boundary')
        window['store']=store
        if 'initial_q' in window:
            if store.initial_states is None:raise ValueError('缺少常驻 IK 初始状态')
            current=window['indices'][len(window['rows'])-window['length']-10]
            window['initial_q']=store.initial_states[current]
    return stores


def choose_micro_batch(architecture, device, requested=0, per_window_mb=None,workspace_mb=0):
    if isinstance(requested,bool) or not isinstance(requested,int) or requested<0:
        raise ValueError('GPU 分块必须为非负整数；0 表示自动')
    if requested:return requested
    if not str(device).startswith('cuda'):return 8
    free,_=torch.cuda.mem_get_info(device)
    # Measured FP32 attention at T=16 uses about 331 MiB/window. Reserve
    # 20% of currently free memory for mesh labels, optimizer and variation.
    per_window=(per_window_mb or (350 if architecture=='attention' else 100))*1024**2
    limit=max(1,int((.8*free-workspace_mb*1024**2)/per_window))
    # Multiples of eight use the budget without discarding almost half of it
    # solely because the next power of two does not fit.
    return max(1,limit//8*8)


def choose_parallel_batch(architecture,device,requested=0,memory_limit=0,length=16,
                          attention_precision='float32',checkpoint_observations=False):
    """Resolve one real batch, never a larger batch split for accumulation.

    Explicit requests are upper bounds: a request of 4096 on a small GPU must
    not silently mean 4096 accumulated windows. Budget scales with BPTT length.
    The old micro-batch argument is retained only as a simultaneous-window cap.
    """
    for value in (requested,memory_limit):
        if isinstance(value,bool) or not isinstance(value,int) or value<0:
            raise ValueError('并行窗口数必须为非负整数；0 表示按显存自动选择')
    # The bounded encoder workspace dominates the recomputed BF16 path:
    # measured T=16 peaks are 6304 MiB at B=112 and 6990 MiB at B=512.
    # Reserve 6 GiB plus 3 MiB/window, AND 20% of device memory. There is no
    # artificial window cap: enough memory permits the complete dataset.
    recomputed=architecture=='attention' and checkpoint_observations and attention_precision=='bfloat16'
    estimate=3 if recomputed else 100 if architecture=='attention' and checkpoint_observations else 240 if architecture=='attention' and attention_precision=='bfloat16' else None
    workspace=6144 if recomputed else 0
    capacity=max(1,int(choose_micro_batch(architecture,device,0,estimate,workspace)*16/max(16,length)))
    return min([capacity]+[v for v in (requested,memory_limit) if v])


def semantic_key(contact, self_contact=False):
    return ('self',*sorted(contact['regions'])) if self_contact else ('object',contact['region'])


def group_normalizers(windows):
    """Balance semantic groups independently at each rollout step."""
    length=min(w['length'] for w in windows);result=[]
    for t in range(length):
        counts=Counter()
        for w in windows:
            row=w['rows'][len(w['rows'])-w['length']-9+t]
            c=row['contacts'];keys={semantic_key(x,True) for x in c['self_pairs']}
            keys.update(semantic_key(x) for x in c['object_contacts']);counts.update(keys)
        result.append({key:len(windows)/(len(counts)*n) for key,n in counts.items()})
    return result


def fast_objective(objective,q,rows,references=None,acceleration=None,previous_rows=None,audit=False,normalizers=None):
    h=objective.hand;s=objective.settings;frames=h.frames(q);p,r,j=frames[:3];size=len(q)
    constants=[objective.fast_constants(row) for row in rows]
    # prepare_targets may remove degenerate edges; group by the exact edge set.
    edge_groups={}
    for i,c in enumerate(constants):edge_groups.setdefault(c['edge_key'],[]).append(i)
    dir_cost=q.new_zeros(size);dir_error=q.new_zeros(size)
    for ids in edge_groups.values():
        edges=constants[ids[0]]['edges'];desired=torch.stack([constants[i]['desired'] for i in ids])
        dirs=F.normalize(j[ids][:,edges[:,1]]-j[ids][:,edges[:,0]],dim=-1,eps=1.e-8)
        index=torch.as_tensor(ids,device=q.device)
        dir_cost=dir_cost.index_copy(0,index,.5*s.lambda_pose*(dirs-desired).square().sum((1,2)))
        dir_error=dir_error.index_copy(0,index,torch.rad2deg(torch.acos((dirs.detach()*desired).sum(-1).clamp(-1,1))).mean(-1))
    smooth=.5*s.lambda_smooth*(q-references).square().sum(-1) if references is not None else q.new_zeros(size)
    self_depth,point_depth=objective.collision.penalties(q,rows,frames)
    depth=torch.maximum(self_depth,point_depth);valid=torch.ones(size,device=q.device,dtype=torch.bool)
    if audit:depth,valid=objective.collision.audit(q,rows,frames)
    barrier=(depth/s.penetration_scale_m).square() if audit else (self_depth/s.penetration_scale_m).square()+(point_depth/s.penetration_scale_m).square()
    groups={};contact_points=[[] for _ in rows]
    for i,c in enumerate(constants):
        for item in c['contacts']:groups.setdefault(item['key'],[]).append((i,item))
    contact=q.new_zeros(size);normal_cost=q.new_zeros(size);max_gap=q.new_zeros(size);max_angle=q.new_zeros(size)
    unresolved=torch.zeros(size,device=q.device,dtype=torch.long);counts=[0]*size
    # A time-batched rollout retains each step's semantic denominators. Mixing
    # time into the group counts would silently change the contact objective.
    weights=normalizers if normalizers is not None else {key:size/(len(groups)*len({i for i,_ in items})) for key,items in groups.items()}
    angle_scale=1-np.cos(np.deg2rad(s.normal_deg))
    for key,items in groups.items():
        ids=torch.as_tensor([i for i,_ in items],device=q.device);duplicates=Counter(i for i,_ in items)
        target=torch.stack([item['target'] for _,item in items])
        if key[0]=='self':
            a,b=items[0][1]['owners']
            if audit:
                distance,angle,pa,pb,na,nb,ok=h.pair(p[ids],r[ids],a,b,True)
                centre=((pa+pb)*.5-target).square().sum(-1)
            else:
                distance,angle,pa,pb,na,nb,ok,centre=objective.collision.soft_pair(p[ids],r[ids],a,b,target,s.contact_scale_m,angle_scale)
            position=(distance+.02*centre)/s.contact_scale_m**2
            norm=angle/angle_scale
            degrees=torch.rad2deg(torch.acos((-na.detach()*nb.detach()).sum(-1).clamp(-1,1)))
        else:
            owner=items[0][1]['owner'];desired=torch.stack([item['normal'] for _,item in items])
            query=(r[ids,owner].transpose(-1,-2)@(target-p[ids,owner])[...,None]).squeeze(-1)
            escape=(r[ids,owner].transpose(-1,-2)@desired[...,None]).squeeze(-1)
            if audit:local,ln,ok=h.nearest(query,owner,palmar=True,escape=escape)
            else:
                local,ln=objective.collision.contact(query,escape,owner,palmar=True)
                ok=torch.ones(len(query),device=q.device,dtype=torch.bool)
            pa=(r[ids,owner]@local[...,None]).squeeze(-1)+p[ids,owner];pb=target
            nn=(r[ids,owner]@ln[...,None]).squeeze(-1)
            dot=(nn*desired).sum(-1);distance=(pa-target).square().sum(-1)
            rho=q.new_tensor([item['rho'] for _,item in items])
            position=rho.square()*distance/s.contact_scale_m**2;norm=rho.square()*(1-dot)/angle_scale
            degrees=torch.rad2deg(torch.acos(dot.detach().clamp(-1,1)))
        weight=q.new_tensor([(weights[i][key] if isinstance(weights,list) else weights[key])/duplicates[i] for i,_ in items])
        contact=contact.index_add(0,ids,torch.where(ok,position,0.)*weight)
        normal_cost=normal_cost.index_add(0,ids,torch.where(ok,norm,0.)*weight)
        max_gap=max_gap.scatter_reduce(0,ids,torch.where(ok,distance.detach().clamp_min(0).sqrt(),0.),reduce='amax',include_self=True)
        max_angle=max_angle.scatter_reduce(0,ids,degrees,reduce='amax',include_self=True)
        unresolved=unresolved.index_add(0,ids,(~ok).long())
        for k,(i,_) in enumerate(items):counts[i]+=1;contact_points[i].append((pa[k:k+1],pb[k:k+1]))
    valid=valid&(unresolved==0)
    keep,keep_counts=objective.collision.retention(q,rows,previous_rows or [None]*size,frames,s.contact_scale_m)
    total=dir_cost+smooth+s.contact*contact+s.normal*normal_cost+s.penetration*barrier+s.contact_keep*keep
    parts=dict(direction=dir_cost.mean(),smooth=smooth.mean(),contact=contact.mean(),normal=normal_cost.mean(),
        penetration=barrier.mean(),self_penetration=(self_depth/s.penetration_scale_m).square().mean(),
        tactile_penetration=(point_depth/s.penetration_scale_m).square().mean(),contact_keep=keep.mean())
    for key in ('laplacian','antireverse','thumb_relative'):parts[key]=q.new_zeros(())
    if acceleration is not None:
        parts['acceleration']=acceleration.square().mean();total=total+s.acceleration*acceleration.square().mean(-1)
    feasible=valid&torch.isfinite(total)&(depth<=s.penetration_limit_m)&(max_gap<=s.contact_limit_m)&(max_angle<=s.normal_deg)&(q>=h.low-1.e-6).all(-1)&(q<=h.high+1.e-6).all(-1)
    metrics=[dict(feasible=feasible[i:i+1],contact_count=counts[i],max_gap_m=max_gap[i:i+1],max_normal_deg=max_angle[i:i+1],
        max_penetration_m=depth[i:i+1],bone_direction_deg=dir_error[i:i+1],points=contact_points[i],query_valid=valid[i:i+1],
        contact_query_valid=unresolved[i:i+1]==0,unresolved_contact_count=unresolved[i:i+1],groups={},
        contact_keep_count=keep_counts[i],contact_keep_loss=keep[i:i+1].detach()) for i in range(size)]
    return total.mean(),parts,metrics
