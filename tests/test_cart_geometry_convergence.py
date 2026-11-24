"""Early stopping requires both accepted geometry and a validation plateau."""
import json
import pytest
import torch
from moxian_web.cart_geometry_schedule import ConvergedSelection,convergence_state,EARLY_STOP
from moxian_web.cart_collision import geometry_rejection_reasons
from moxian_web.cart_training import save_checkpoint


def passing():
    return dict(distance_p95_mm=.2,penetration_cases=30,penetration_false_negative_rate=0.,
        spatial_gradient_norm_mean=1.,spatial_gradient_cosine_mean=.98,
        contact_samples=100,contact_position_p95_mm=.2,contact_normal_p95_deg=10.,
        contact_jacobian_samples=100,contact_jacobian_error_mean=.1,
        contact_far_samples=100,contact_far_position_p95_mm=.4,contact_far_jacobian_error_mean=.1,
        q_collision_cases=30,q_false_negative_rate=0.,q_gradient_cosine_mean=.95,q_escape_cost_descent_rate=.98,
        q_cap_cases=30,q_cap_gradient_cosine_mean=.95,q_combined_escape_cost_descent_rate=.98,
        q_combined_escape_cap_descent_rate=.98)


def selector(folder):
    return ConvergedSelection(folder,save_checkpoint,lambda p,d:p.write_text(json.dumps(d)),geometry_rejection_reasons)


def test_passing_plateau_restores_best_and_revalidates_before_policy(tmp_path):
    model=torch.nn.Linear(1,1,bias=False);select=selector(tmp_path)
    with torch.no_grad():model.weight.fill_(1.)
    select.observe(model,passing(),2000)
    with torch.no_grad():model.weight.fill_(2.)
    for update in range(2500,5001,500):select.observe(model,dict(passing(),distance_p95_mm=.21),update)
    assert select.should_stop(5000)
    decision=select.finish(model,lambda:passing() if model.weight.item()==1. else dict(passing(),distance_p95_mm=2.),5000,30000)
    assert decision['accepted'] and model.weight.item()==1.
    assert decision['update']==2000 and decision['training_completed_updates']==5000
    assert decision['maximum_updates']==30000 and decision['termination']=='validation_plateau_after_acceptance'


def test_failing_plateau_does_not_start_policy(tmp_path):
    model=torch.nn.Linear(1,1);select=selector(tmp_path)
    for update in range(2000,8001,500):select.observe(model,dict(passing(),q_cap_gradient_cosine_mean=.2),update)
    assert not select.should_stop(8000)
    with pytest.raises(ValueError,match='has not converged'):select.finish(model,passing,8000,30000)


def test_early_selected_weights_must_pass_fresh_validation(tmp_path):
    model=torch.nn.Linear(1,1);select=selector(tmp_path)
    for update in range(2000,5001,500):select.observe(model,passing(),update)
    decision=select.finish(model,lambda:dict(passing(),distance_p95_mm=2.),5000,30000)
    assert not decision['accepted']
    assert json.loads((tmp_path/'geometry_schedule_status.json').read_text())['stage']=='rejected'


def test_cumulative_small_improvements_reset_patience():
    history=[dict(update=2000+500*i,accepted=True,score=.4*.997**i) for i in range(15)]
    state=convergence_state(history)
    assert not state['converged'] and state['checks_without_significant_improvement']<EARLY_STOP['patience']


def test_resume_is_idempotent_and_prunes_uncommitted_validation_history(tmp_path):
    model=torch.nn.Linear(1,1);select=selector(tmp_path)
    for update in range(2000,5001,500):select.observe(model,passing(),update)
    select=selector(tmp_path);select.observe(model,passing(),5000)
    assert len(select.history)==7 and select.should_stop(5000)
    select.observe(model,passing(),4001)
    assert max(row['update'] for row in select.history)==4000 and not select.should_stop(4001)


def test_minimum_updates_and_failures_and_missing_checks_prevent_early_stop():
    h=[dict(update=500+500*i,accepted=True,score=.4) for i in range(7)]
    assert not convergence_state(h)['converged']
    h=[dict(update=2000+500*i,accepted=True,score=.4) for i in range(7)]
    h[-2]['accepted']=False
    assert not convergence_state(h)['converged']
    h=[dict(update=2000+500*i,accepted=True,score=.4) for i in range(7)]
    h[-1]['update']=5500
    assert not convergence_state(h)['converged']


def test_schedule_migration_changes_only_driver_hash_and_gate():
    import copy
    from scripts.enable_cart_sdf_early_stop import prepare_config
    config=dict(objective_version='card_style_link_sdf_contacts_v6_wrist',
        geometry={'files':{'cart_training.py':'old','network.py':'unchanged'},'link_sdf':{'model':'unchanged'}},
        geometry_gate={'completion':'full_budget_best','maximum_updates':30000,'required':True,'schedule_sha256':'old-schedule'},
        optimizer_kind='muon_sgd',seed=714)
    previous=copy.deepcopy(config)
    new=prepare_config(config,'old','new','old-schedule','new-schedule')
    assert config==previous and new['geometry']['files']['network.py']=='unchanged'
    assert new['geometry']['link_sdf']==config['geometry']['link_sdf']
    assert new['geometry_gate']['completion']=='converged_best' and new['geometry_gate']['early_stop']==EARLY_STOP
    with pytest.raises(ValueError,match='Archived driver'):prepare_config(config,'wrong','new','old-schedule','new-schedule')
    with pytest.raises(ValueError,match='Only the new'):prepare_config(dict(config,objective_version='v9'),'old','new','old-schedule','new-schedule')
