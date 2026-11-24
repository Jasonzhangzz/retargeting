"""Check full-budget selection, physical rejection, and strict resume migration."""
import copy
import json

import pytest
import torch

from moxian_web.cart_collision import geometry_rejection_reasons
from moxian_web.cart_geometry_schedule import FullBudgetSelection,candidate_is_better
from moxian_web.cart_training import save_checkpoint
from scripts.continue_cart_after_geometry import prepare_config,sha256


def passing_report():
    return dict(distance_p95_mm=.2,penetration_cases=30,penetration_false_negative_rate=0.,
        spatial_gradient_norm_mean=1.,spatial_gradient_cosine_mean=.98,
        contact_samples=100,contact_position_p95_mm=.2,contact_normal_p95_deg=10.,
        contact_jacobian_samples=100,contact_jacobian_error_mean=.1,
        contact_derivative_label_valid_fraction=.98,contact_far_samples=100,
        contact_far_position_p95_mm=.4,contact_far_jacobian_error_mean=.1,
        contact_far_derivative_label_valid_fraction=.98,q_collision_cases=30,
        q_false_negative_rate=0.,q_gradient_cosine_mean=.95,q_escape_cost_descent_rate=.98,
        q_cap_cases=30,q_cap_gradient_cosine_mean=.95,
        q_combined_escape_cost_descent_rate=.98,q_combined_escape_cap_descent_rate=.98)


def selector(folder):
    return FullBudgetSelection(folder,save_checkpoint,
        lambda path,data:path.write_text(json.dumps(data)),geometry_rejection_reasons)


def test_full_budget_restores_earlier_best_and_never_finishes_early(tmp_path):
    model=torch.nn.Linear(1,1,bias=False)
    select=selector(tmp_path)
    with torch.no_grad():model.weight.fill_(1.)
    best=passing_report()
    assert geometry_rejection_reasons(best)==[]
    select.observe(model,best,2000)
    with pytest.raises(ValueError,match='budget is incomplete'):
        select.finish(model,lambda:best,2000,30000)
    with torch.no_grad():model.weight.fill_(2.)
    worse=dict(best,contact_position_p95_mm=2.)
    select.observe(model,worse,30000)
    decision=select.finish(model,lambda:best if model.weight.item()==1. else worse,30000,30000)
    assert model.weight.item()==1. and decision['accepted'] and decision['update']==2000
    assert decision['training_completed_updates']==30000
    assert json.loads((tmp_path/'geometry_schedule_status.json').read_text())['stage']=='policy_ready'


def test_selection_rechecks_weights_instead_of_trusting_old_acceptance(tmp_path):
    model=torch.nn.Linear(1,1,bias=False)
    select=selector(tmp_path);select.observe(model,passing_report(),2000)
    failed=dict(passing_report(),q_cap_gradient_cosine_mean=.2)
    decision=select.finish(model,lambda:failed,30000,30000)
    assert not decision['accepted'] and 'q_cap_gradient_cosine_mean' in decision['rejection_reasons']
    assert json.loads((tmp_path/'geometry_schedule_status.json').read_text())['stage']=='rejected'


def test_any_passing_checkpoint_outranks_a_lower_score_failing_checkpoint():
    passing=passing_report()
    failing=dict(passing,contact_samples=1,distance_p95_mm=0.,contact_position_p95_mm=0.)
    previous=dict(metrics=failing,rejection_reasons=geometry_rejection_reasons(failing))
    assert candidate_is_better(passing,[],previous)
    assert not candidate_is_better(dict(passing,distance_p95_mm=float('nan')),['distance_p95_mm'],None)


def test_best_index_recovers_from_atomic_checkpoint_after_interruption(tmp_path):
    model=torch.nn.Linear(1,1,bias=False);select=selector(tmp_path)
    select.observe(model,passing_report(),2500)
    (tmp_path/'geometry_best.json').write_text('{}')
    selector(tmp_path)
    assert json.loads((tmp_path/'geometry_best.json').read_text())['update']==2500


def test_resume_migration_only_allows_archived_driver_and_schedule(tmp_path):
    old=tmp_path/'old.py';old.write_text('old reviewed training driver')
    config=dict(geometry=dict(files={'cart_training.py':sha256(old),'model.py':'unchanged'}),
        geometry_gate=dict(required=True,maximum_updates=30000))
    current=copy.deepcopy(config['geometry']);current['files']['cart_training.py']='new-driver'
    migrated=prepare_config(config,current,'schedule-hash',old)
    assert config['geometry_gate'].get('completion') is None
    assert migrated['geometry_gate']['completion']=='full_budget_best'
    assert prepare_config(migrated,current,'schedule-hash')==migrated
    with pytest.raises(ValueError,match='Archived training source'):
        prepare_config(config,current,'schedule-hash')
    changed=copy.deepcopy(current);changed['files']['model.py']='unreviewed-model-change'
    with pytest.raises(ValueError,match='beyond the training driver'):
        prepare_config(config,changed,'schedule-hash',old)
