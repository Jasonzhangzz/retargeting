"""Optional scalar-only W&B tracking; no model or recording uploads."""
import os
import json
from pathlib import Path


class CartTracking:
    def __init__(self,folder,config,*,project=None,entity=None,base_url=None,mode='online',resume=False):
        self.run=None
        if not project:return
        import wandb
        settings=dict(console='off',disable_code=True,init_timeout=60)
        host=base_url or os.environ.get('WANDB_BASE_URL')
        if host:settings['base_url']=host.rstrip('/')
        self.run=wandb.init(project=project,entity=entity,id=config['run_id'],name=config['run_id'],
            resume='must' if resume and mode=='online' else 'never',mode=mode,
            dir=str(Path(folder)),config=config,allow_val_change=resume,save_code=False,settings=wandb.Settings(**settings))
        if resume and 'error' in self.run.summary:del self.run.summary['error']
        (Path(folder)/'wandb_run.json').write_text(json.dumps(dict(id=self.run.id,url=self.run.url,project=project,
            entity=entity,base_url=host,mode=mode),indent=2))
        for namespace,axis in (('train','optimizer_step'),('geometry','update'),('validation','epoch')):
            self.run.define_metric(f'{namespace}/{axis}')
            self.run.define_metric(f'{namespace}/*',step_metric=f'{namespace}/{axis}')

    def log(self,phase,values):
        if self.run is None:return
        payload={}
        if phase=='retargeting':
            payload.update({'train/'+key:float(value) for key,value in values['batch_losses'].items()})
            payload.update({'train/optimizer_step':values['optimizer_step'],'train/epoch':values['epoch'],
                'train/parallel_windows':values['parallel_batch_size'],'train/transitions_per_s':values['transitions_per_s']})
            payload.update({'timing/'+key:float(value) for key,value in values['timings'].items()})
        if phase in ('retargeting','sdf_warmup'):
            payload.update({'geometry/'+key:float(value) for key,value in values['sdf_losses'].items()})
            payload['geometry/update']=values.get('sdf_updates',values.get('completed',0))
        if phase=='validation':
            payload.update({'validation/'+key:float(value) for key,value in values['metrics'].items() if value is not None})
            payload.update({'validation/sdf_'+key:float(value) for key,value in values['sdf_validation'].items() if value is not None})
            payload.update({'validation/epoch':values['epoch'],'validation/loss':values['validation_loss'],
                'validation/best_loss':values['best_validation_loss']})
        if payload:
            payload['wall/elapsed_s']=values.get('elapsed_s',0.)
            self.run.log(payload)

    def finish(self,*,report=None,error=None):
        if self.run is None:return
        if report:
            self.run.summary.update({'final/'+key:value for key,value in report.items()})
        if error:self.run.summary['error']=error
        self.run.finish(exit_code=1 if error else 0)
