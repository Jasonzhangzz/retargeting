"""Validated offline training controls. These never change runtime geometry."""
import math

DEFAULTS=dict(adam_steps=480,max_seconds=45.,batch_size=0,optimizer='adam')

def options(value=None):
    if value is None:return dict(DEFAULTS)
    if not isinstance(value,dict) or set(value)-set(DEFAULTS):raise ValueError('无效的训练选项')
    result={**DEFAULTS,**value}
    if result['optimizer'] not in ('adam','muon'):raise ValueError('优化器须为 adam 或 muon')
    if type(result['adam_steps']) is not int or not 128<=result['adam_steps']<=8192:raise ValueError('每阶段步数须为128–8192')
    if type(result['batch_size']) is not int or result['batch_size'] not in (0,512,768):raise ValueError('训练批次须为覆盖默认、512或768个独立训练帧')
    seconds=result['max_seconds']
    if type(seconds) not in (int,float) or not math.isfinite(seconds) or not 15<=seconds<=600:raise ValueError('优化预算须为15–600秒')
    return result


def validate_optimizer(method,settings):
    if settings['optimizer']=='muon' and method!='delta':
        raise ValueError('Muon 仅用于 MANO＋Delta 隐藏层；个体参数版及 FlashSAC 请选 Adam')


def training_rows(rows,batch_size=0,full_buffer=False):
    """Expand the existing coverage set using unique, train-only records."""
    from .mano_fit import choose_rows
    import copy,numpy as np
    if full_buffer:
        if not rows or any(r.get('capture_protocol') not in ('geort_motion_v2','contact_motion_v3') for r in rows):
            raise ValueError('完全混训需要带独立分区的连续标定记录')
        if any(r.get('partition') not in ('train','validation','initialization') for r in rows):
            raise ValueError('完全混训记录缺少明确的训练／留出分区')
        selected=[r for r in rows if r.get('partition')=='train' and r.get('training_eligible',True)]
        if not selected:raise ValueError('完全混训缺少训练分区')
        return sorted(copy.deepcopy(selected),key=lambda r:r['timestamp_ms'])
    baseline=choose_rows(rows)[0]
    if not batch_size:return baseline
    if batch_size<len(baseline):raise ValueError(f'批次至少需要{len(baseline)}帧才能保留当前动作覆盖')
    key=lambda r:(r['timestamp_ms'],r['stage'],r.get('cycle',-1))
    selected={key(r):r for r in baseline};groups={}
    for row in rows:
        if row.get('partition')!='train':continue
        if key(row) in selected:continue
        group=(row['stage'],row.get('cycle',-1),row.get('motion_phase','static'))
        groups.setdefault(group,[]).append(row)
    # Round-robin groups, and evenly spaced time coverage within each group.
    queues=[]
    for group in groups.values():
        ordered=sorted(group,key=lambda r:r['timestamp_ms'])
        # Recursive midpoint ordering yields temporal spread at every prefix.
        indices=[];pending=[(0,len(ordered))]
        while pending:
            lo,hi=pending.pop(0)
            if lo>=hi:continue
            mid=(lo+hi)//2;indices.append(mid);pending.extend(((lo,mid),(mid+1,hi)))
        queues.append([ordered[i] for i in indices])
    while len(selected)<batch_size and any(queues):
        for group in queues:
            if group and len(selected)<batch_size:
                row=group.pop(0);selected[key(row)]=row
    return sorted(copy.deepcopy(list(selected.values())),key=lambda r:r['timestamp_ms'])
