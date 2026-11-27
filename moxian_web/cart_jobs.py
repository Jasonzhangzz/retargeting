"""Web/CLI adapters for the current noise-free CART training route."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch
from .cart_data import load_sessions
from .cart_initialization import attach_ik
from .cart_training import windows,predict_trajectory
from .cart_fast import PreparedBatch,prepare_windows
from .cart_wrist import FloatingCartHand
from .cart_palm_alignment import PalmAlignedNetwork
from .cart_reach_loss import ReachObjective
from retarget.baseline.geort.training import atomic_json


def train(root,train_ids,validation_ids,run_id,epochs=1000,*,
          policy_width=384,decoder_width=744,micro_batch=512,
          project='cart',entity=None,base_url=None,mode='online',check=lambda:None,progress=None):
    workers=min(2,torch.cuda.device_count())
    if workers<1:raise ValueError('CART 训练需要 CUDA GPU')
    if not train_ids or not validation_ids or set(train_ids)&set(validation_ids):
        raise ValueError('需要互不重叠的训练集和验证集')
    root=Path(root).resolve();folder=root/'models'/run_id
    folder.mkdir(parents=True,exist_ok=False)
    ref=dict(train_ids=train_ids,validation_ids=validation_ids,seed=714,
        tracking=dict(project=project,entity=entity,base_url=base_url,mode=mode))
    reference=folder/'reference_config.json';atomic_json(reference,ref)
    script=Path(__file__).resolve().parents[1]/'scripts'/'train_cart_reach_policy.py'
    cmd=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1',f'--nproc-per-node={workers}',str(script),
         '--root',str(root),'--reference-config',str(reference),'--run-id',run_id,'--epochs',str(epochs),
         '--unbounded-wrist','--effective-batch','8192','--micro-batch',str(micro_batch),
         '--policy-width',str(policy_width),'--decoder-width',str(decoder_width)]
    cmd+=['--tactile-encoder','pointnet','--single-step','--on-policy-state-depth','16',
          '--delta-direction-weight','0.01','--delta-direction-samples','128',
          '--initial-noise-rad','0','--step-noise-rad','0']
    child=subprocess.Popen(cmd,start_new_session=True,env={**os.environ,'OMP_NUM_THREADS':'1'})
    last=None
    try:
        while child.poll() is None:
            check();path=folder/'status.json'
            if progress and path.exists():
                raw=path.read_text()
                if raw!=last:
                    state=json.loads(raw);progress(state.get('phase','training'),policy_updates=state.get('policy_updates',0),model=run_id);last=raw
            time.sleep(.5)
        if child.returncode:raise RuntimeError(f'CART distributed training failed ({child.returncode}); see model logs')
        status=json.loads((folder/'status.json').read_text())
        if status['status']!='finished':raise InterruptedError('CART training stopped')
    except BaseException:
        (folder/'STOP_REQUESTED').write_text('Parent job cancelled\n')
        try:child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            import signal
            os.killpg(child.pid,signal.SIGTERM)
        raise
    return folder


def load_current_policy(folder,device='cpu'):
    folder=Path(folder)
    path=folder/'best.pth'
    if not path.exists():path=folder/'resume.pth'
    saved=torch.load(path,map_location='cpu',weights_only=False);config=saved['config']
    hand=FloatingCartHand(device);network=config.get('policy_network',{})
    model=PalmAlignedNetwork(hand.low,hand.high,alignment=config['palm_alignment'],
        width=network.get('width',128),decoder_width=network.get('decoder_width',256),
        tactile_encoder=config.get('tactile_encoder','dense'),tactile_groups=config.get('tactile_groups')).to(device)
    model.load_state_dict(saved['model'],strict=True);model.eval()
    model.set_attention_precision('bfloat16' if str(device).startswith('cuda') else 'float32')
    return model,hand,config


def evaluate(root,model_id,dataset_ids,check=lambda:None):
    from .cart_card_cost import attach_card_targets
    root=Path(root);folder=root/'models'/model_id
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    model,hand,config=load_current_policy(folder,device)
    if set(dataset_ids)&set(config['train_ids']):raise ValueError('训练集不能作为独立验证集')
    sessions,_=load_sessions(root,dataset_ids,'validation',require_complete=True)
    attach_ik(root,sessions);attach_card_targets(root,sessions,device,check)
    batches=windows(sessions,16,16);prepare_windows(sessions,batches,device)
    objective=ReachObjective(hand);totals={};count=0
    with torch.no_grad():
        for start in range(0,len(batches),16):
            check();batch=PreparedBatch(batches[start:start+16],device)
            tr=predict_trajectory(model,objective,batch,step_noise_rad=0.)
            _,parts=objective.trajectory_loss(tr)
            for k,v in parts.items():totals[k]=totals.get(k,0.)+float(v)*len(batch)
            count+=len(batch)
    if not count:raise ValueError('没有完整的验证窗口')
    result=dict(model=model_id,datasets=dataset_ids,windows=count,metrics={k:v/count for k,v in totals.items()},
                geometry='native semantic palmar mesh',evaluation='16-step noise-free heldout windows')
    atomic_json(folder/'evaluation.json',result);return result
