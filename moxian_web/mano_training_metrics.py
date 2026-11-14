"""Training-only raw/EMA diagnostics with comparable phase replay objectives."""
import time
import torch

class TrainingMetrics:
    def __init__(self,objective,progress):
        if any(r.get('partition','train')!='train' for r in objective.rows):raise ValueError('训练曲线不能读取留出数据')
        self.objective=objective;self.progress=progress;self.began=time.monotonic();self.last=-1.;self.points=[];self.step=0

    @torch.no_grad()
    def emit(self,raw,average,phase,phase_step,local=None,force=False,advance=1,full_replay=False):
        self.step+=advance;now=time.monotonic()
        if not force and now-self.last<1.:return
        self.last=now;o=self.objective
        # Current-pair losses and full phase replay losses are separate series.
        scope=local.stages[local.index] if local else phase
        saved=None
        if local and full_replay:
            scope='all_pinches'
            saved=(o.phase_active.clone(),o.weights.clone(),o.local_vertex_weight.clone())
            o.phase_active.fill_(1)
            gate=o.t([r['stage'] in local.stages for r in o.rows])
            o.weights.copy_(local.original_weights*gate[:,None]);o.weights.div_(o.weights.sum().clamp_min(1e-12));o.local_vertex_weight.zero_()
        def evaluate(vector):
            residual=o.residual(vector,record=True)
            from .mano_shape_prior import GEOMETRY_SIGMA
            geometry=vector[46:o.shared_size]
            return dict(loss=float(residual.square().sum()),terms=dict(o.last_terms),
                beta_kl=float(.5*vector[:10].square().sum()),
                geometry_kl=float(.5*(geometry/o.t(GEOMETRY_SIGMA)).square().sum()) if len(geometry)==43 else None,
                **getattr(o,'last_training_metrics',{}),**getattr(o,'optimizer_diagnostics',{}))
        try:
            raw_values=evaluate(raw)
            ema_values=raw_values if getattr(o,'accepted_parameters_direct',False) else evaluate(average)
        finally:
            if saved:
                o.phase_active.copy_(saved[0]);o.weights.copy_(saved[1]);o.local_vertex_weight.copy_(saved[2])
        point=dict(sample_id=len(self.points)+1,elapsed_s=now-self.began,phase=phase,scope=scope,step=self.step,phase_step=phase_step,
                   raw=raw_values,ema=ema_values,training_frames=len(o.rows),source='train_only')
        if getattr(o,'resume_info',None):
            point.update(previous_training_steps=o.resume_info['previous_steps'],total_training_step=o.resume_info['previous_steps']+self.step)
        self.points.append(point)
        self.progress(dict(state='running',phase=phase,training_sample=point,message=f'训练曲线 · {scope} · {phase_step} 步'))

    def report(self):
        return dict(revision='mano_training_trace_v1',points=self.points,source='train_only',
                    kl_definition='equal_unit_covariance_Gaussian_mean_prior; geometry normalized by GEOMETRY_SIGMA; not a policy KL',
                    note='Compare losses only within the same phase and scope. beta KL stays flat when betas are frozen.')
