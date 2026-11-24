"""Validation selection for a fixed geometry budget before policy training."""
import math
import json
from pathlib import Path


COMPLETION = 'full_budget_best'
CONVERGED_COMPLETION = 'converged_best'
EARLY_STOP = dict(minimum_updates=5000,validation_interval=500,patience=6,minimum_relative_improvement=.01)
SELECTION = 'accepted_first_mean_normalized_geometry_error_v1'


def selection_key(report, rejection_reasons):
    """Passing every physical gate takes priority over the scalar ranking.

    The score only ranks checkpoints; it never replaces or relaxes a gate.
    Errors are divided by their acceptance tolerances, including cosine and
    descent errors expressed as 1 - metric. Missing/nonfinite data ranks last.
    """
    errors = {
        'distance_p95_mm': .5,
        'penetration_false_negative_rate': .01,
        'contact_position_p95_mm': .5,
        'contact_normal_p95_deg': 30.,
        'contact_jacobian_error_mean': .25,
        'contact_far_position_p95_mm': 1.,
        'contact_far_jacobian_error_mean': .25,
        'q_false_negative_rate': .01,
    }
    similarities = {
        'spatial_gradient_cosine_mean': .9,
        'q_gradient_cosine_mean': .8,
        'q_escape_cost_descent_rate': .9,
        'q_cap_gradient_cosine_mean': .8,
        'q_combined_escape_cost_descent_rate': .9,
        'q_combined_escape_cap_descent_rate': .9,
    }
    if report.get('geometry_validation') in ('link_sdf_distance_gradient_v1','conditional_sdf_distance_gradient_v2'):
        errors={k:v for k,v in errors.items() if not k.startswith('contact_')}
        similarities['articulated_gradient_cosine_mean']=.9
    values = []
    for name, tolerance in errors.items():
        value = report.get(name)
        if value is None or not math.isfinite(value):
            return (1, math.inf)
        values.append(max(0., value) / tolerance)
    for name, minimum in similarities.items():
        value = report.get(name)
        if value is None or not math.isfinite(value):
            return (1, math.inf)
        values.append(max(0., 1. - value) / (1. - minimum))
    norm = report.get('spatial_gradient_norm_mean')
    if norm is None or not math.isfinite(norm):
        return (1, math.inf)
    values.append(abs(norm - 1.) / .2)
    return (int(bool(rejection_reasons)), sum(values) / len(values))


def candidate_is_better(report, reasons, previous):
    current = selection_key(report, reasons)
    if not math.isfinite(current[1]):
        return False
    if previous is None:
        return True
    return current < selection_key(previous['metrics'], previous['rejection_reasons'])


class FullBudgetSelection:
    """Atomically retain candidates and recheck selected weights at the budget."""
    def __init__(self,folder,save_checkpoint,atomic_json,reject):
        import torch
        self.folder=Path(folder);self.save=save_checkpoint;self.write=atomic_json;self.reject=reject
        self.path=self.folder/'geometry_best.pth';self.index=self.folder/'geometry_best.json'
        if self.path.exists():
            saved=torch.load(self.path,map_location='cpu',weights_only=True)
            if saved['selection']!=SELECTION:raise ValueError('Geometry selection rule changed')
            self.write(self.index,{k:v for k,v in saved.items() if k!='collision'})
        elif self.index.exists():
            raise ValueError('Geometry best-checkpoint index has no weights')

    def observe(self,model,report,update):
        reasons=self.reject(report)
        previous=json.loads(self.index.read_text()) if self.index.exists() else None
        if candidate_is_better(report,reasons,previous):
            candidate=dict(accepted=not reasons,update=update,metrics=report,rejection_reasons=reasons,
                selection=SELECTION,score=selection_key(report,reasons)[1])
            self.save(self.path,dict(candidate,collision=model.state_dict()))
            self.write(self.index,candidate)

    def finish(self,model,validate,completed,required):
        import torch
        import hashlib
        if completed!=required:raise ValueError('Geometry budget is incomplete; policy training is forbidden')
        if not self.path.exists():raise ValueError('No finite geometry validation checkpoint; policy training was not started')
        return self._finish_selected(model,validate,completed,required,'maximum_updates_reached')

    def _finish_selected(self,model,validate,completed,required,termination):
        import torch
        import hashlib
        if not self.path.exists():raise ValueError('No validated geometry checkpoint')
        selected=torch.load(self.path,map_location='cpu',weights_only=True)
        if selected['update']>completed:raise ValueError('Selected weights are newer than the restored training state')
        model.load_state_dict(selected['collision'])
        report=validate();reasons=self.reject(report)
        with self.path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        decision=dict(accepted=not reasons,update=selected['update'],training_completed_updates=completed,
            maximum_updates=required,completion=getattr(self,'completion',COMPLETION),termination=termination,
            metrics=report,rejection_reasons=reasons,selection=SELECTION,
            checkpoint=self.path.name,checkpoint_sha256=digest)
        self.write(self.folder/'geometry_acceptance.json',decision)
        self.write(self.folder/'geometry_schedule_status.json',dict(decision,stage='policy_ready' if not reasons else 'rejected'))
        return decision


def convergence_state(history,policy=None):
    """Passing checks and a plateau are both necessary; failures reset patience.

    Only regularly spaced distinct validation updates count. Significant
    improvement is cumulative relative to the last significant best score.
    Thus many small improvements cannot falsely appear to be a plateau.
    """
    policy=EARLY_STOP if policy is None else policy
    best=math.inf;stale=0;streak=0;previous=None;last=None
    for row in sorted(history,key=lambda r:r['update']):
        update=row['update'];score=row['score']
        if update%policy['validation_interval']:continue
        if previous is not None and update==previous:raise ValueError('Duplicate validation update')
        if previous is not None and update-previous!=policy['validation_interval']:
            best=math.inf;stale=0;streak=0
        previous=update;last=row
        if not row['accepted'] or not math.isfinite(score):
            best=math.inf;stale=0;streak=0;continue
        streak+=1
        if not math.isfinite(best) or score<best*(1-policy['minimum_relative_improvement']):
            best=score;stale=0
        else:stale+=1
    completed=last['update'] if last else 0
    return dict(converged=bool(last and last['accepted'] and completed>=policy['minimum_updates'] and stale>=policy['patience']),
        update=completed,consecutive_accepted_checks=streak,checks_without_significant_improvement=stale,
        significant_best_score=best if math.isfinite(best) else None)


class ConvergedSelection(FullBudgetSelection):
    """Same best-weight/gate rules, with an explicitly authorized early budget."""
    completion=CONVERGED_COMPLETION

    def __init__(self,folder,save_checkpoint,atomic_json,reject):
        super().__init__(folder,save_checkpoint,atomic_json,reject)
        self.convergence_path=self.folder/'geometry_convergence.json'
        if self.convergence_path.exists():
            saved=json.loads(self.convergence_path.read_text())
            if saved['policy']!=EARLY_STOP:raise ValueError('Convergence policy changed')
            self.history=saved['history']
        else:
            # Reviewed schedule migration can retain the original deterministic
            # holdout validations. A resume observation prunes later entries.
            self.history=[];trace=self.folder/'trace.jsonl'
            if trace.exists():
                for line in trace.open():
                    row=json.loads(line)
                    if row.get('phase')=='geometry_validation':
                        self._record(row['sdf_validation'],row['completed'])

    def _record(self,report,update):
        self.history=[row for row in self.history if row['update']<update]
        if update%EARLY_STOP['validation_interval']:return
        reasons=self.reject(report);score=selection_key(report,reasons)[1]
        self.history.append(dict(update=update,accepted=not reasons,score=score if math.isfinite(score) else None))

    def observe(self,model,report,update):
        super().observe(model,report,update)
        self._record(report,update)
        self.write(self.convergence_path,dict(policy=EARLY_STOP,history=self.history,**self.state()))

    def state(self):
        history=[dict(row,score=row['score'] if row['score'] is not None else math.inf) for row in self.history]
        return convergence_state(history)

    def should_stop(self,completed):
        state=self.state()
        return state['update']==completed and state['converged']

    def finish(self,model,validate,completed,required):
        if completed>required:raise ValueError('Geometry exceeded the maximum budget')
        if completed<required and not self.should_stop(completed):
            raise ValueError('Geometry has not converged; early policy training is forbidden')
        reason='validation_plateau_after_acceptance' if completed<required else 'maximum_updates_reached'
        return self._finish_selected(model,validate,completed,required,reason)
