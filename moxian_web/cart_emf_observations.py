"""Shared train/runtime packing for ten measured future EMF frames."""
import numpy as np
import torch
from .cart_protocol import LOOKAHEAD


def pack_future(current_stamp,rows,device='cpu'):
    """Read only actual t+1..t+10 measured EMF/taxel observations, never IK."""
    if len(rows)!=LOOKAHEAD:raise ValueError('CART requires exactly ten actual future frames')
    stamps=np.asarray([r['timestamp_ms'] for r in rows],np.float64)
    gaps=np.diff(np.r_[current_stamp,stamps])
    if not np.isfinite(stamps).all() or not np.isfinite(current_stamp) or np.any(gaps<=0) or np.any(gaps>250) or any(r.get('stream_epoch')!=rows[0].get('stream_epoch') for r in rows):
        raise ValueError('CART future window crosses a sensor boundary')
    if any('emf_pose' not in r or 'emf_valid' not in r for r in rows):
        raise ValueError('未来窗口缺少原始 EMF 完整位姿；禁止使用骨架或 IK 替代')
    values=dict(future_emf=[r['emf_pose'] for r in rows],emf_valid=[r['emf_valid'] for r in rows],
        future_dt=(stamps-current_stamp)/1000,tactile_positions=[r['tactile_positions_m'] for r in rows],
        tactile_state=[r['tactile_state'] for r in rows],tactile_valid=[r['tactile_valid'] for r in rows])
    packed={k:torch.as_tensor(np.asarray(v),device=device,dtype=torch.bool if k.endswith('valid') else torch.float32) for k,v in values.items()}
    if packed['future_emf'].shape!=(10,5,9) or packed['emf_valid'].shape!=(10,5):raise ValueError('Expected ten EMF poses[5,9] and validity[5]')
    return packed


def batch_future(windows,offsets,device):
    packed=[pack_future(w['rows'][i]['timestamp_ms'],w['rows'][i+1:i+11],device) for w,i in zip(windows,offsets)]
    return {k:torch.stack([v[k] for v in packed]) for k in packed[0]}
