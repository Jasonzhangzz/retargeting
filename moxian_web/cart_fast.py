"""GPU observation storage and memory budgeting for measured-EMF CART.

Only immutable prepared observations enter this path. Policy contact witnesses
come from continuous fields; native selection is restricted to independent audit.
"""
import numpy as np
import torch

DEFAULT_BATCH_SIZE = 0  # actual simultaneous windows, selected from free memory


class ObservationStore:
    def __init__(self, rows, device):
        self.device=device
        if any('emf_pose' not in r or 'emf_valid' not in r for r in rows):
            raise ValueError('常驻观测缺少原始 EMF 完整位姿')
        names=dict(tactile_positions='tactile_positions_m',
                   tactile_state='tactile_state',tactile_valid='tactile_valid')
        self.values={k:torch.as_tensor(np.asarray([r[v] for r in rows]),device=device,
            dtype=torch.bool if k.endswith('valid') else torch.float32) for k,v in names.items()}
        self.values.update(future_emf=torch.as_tensor(np.asarray([r['emf_pose'] for r in rows]),device=device,dtype=torch.float32),
            emf_valid=torch.as_tensor([r['emf_valid'] for r in rows],device=device,dtype=torch.bool))
        self.stamps=torch.as_tensor([r['timestamp_ms'] for r in rows],device=device,dtype=torch.float64)
        self.cpu_stamps=np.asarray([r['timestamp_ms'] for r in rows],np.float64)
        self.initial_states=torch.as_tensor(np.asarray([r.get('ik_q_rad',[0.]*20) for r in rows]),device=device,dtype=torch.float32) if any('ik_q_rad' in r for r in rows) else None
        v=self.values;ev=v['emf_valid'];tv=v['tactile_valid']
        if v['future_emf'].shape!=(len(rows),5,9) or ev.shape!=(len(rows),5):raise ValueError('Invalid EMF observation shapes')
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
    if architecture!='attention':raise ValueError('CART requires attention')
    recomputed=architecture=='attention' and checkpoint_observations and attention_precision=='bfloat16'
    estimate=3 if recomputed else 100 if architecture=='attention' and checkpoint_observations else 240 if architecture=='attention' and attention_precision=='bfloat16' else None
    workspace=6144 if recomputed else 0
    capacity=max(1,int(choose_micro_batch(architecture,device,0,estimate,workspace)*16/max(16,length)))
    return min([capacity]+[v for v in (requested,memory_limit) if v])
