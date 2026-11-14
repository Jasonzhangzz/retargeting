import numpy as np
import pytest
from retarget.solver.base import Settings, solve, Problem, NoValidCandidate
from retarget.solver.contact_cem import Branch, ContactProblem
import torch


def test_dispatch_requires_explicit_contact_problem():
    assert Settings('contact_cem').parameters.top_k == 4
    with pytest.raises(TypeError, match='ContactProblem'):
        solve(Problem(torch.zeros(2), torch.ones(2), lambda x: x.sum(-1)), Settings('contact_cem'))


def test_branches_survive_and_native_cost_selects():
    seen = []
    branches = [Branch([], np.zeros(1), np.ones(1), str(i)) for i in range(2)]
    def decode(x, i):
        return np.tile(x[:, None], (1, 2, 1)) + i * 10, [[{'branch': i}] for _ in x]
    def evaluate(u, events):
        return u[:, 0, 0]**2
    def validate(u, events):
        seen.extend(events)
        return [dict(cost=-e[0]['branch'] * 1000 + abs(v[0, 0])) for v, e in zip(u, events)]
    p = ContactProblem(branches, decode, evaluate, validate, 12, 3)
    r = solve(p, Settings('contact_cem'), seed=4)
    assert r.metadata['branch'] == '1'
    assert {e[0]['branch'] for e in seen} == {0, 1}
    assert r.sampled == 36 and r.evaluated <= 44
    again = solve(p, Settings('contact_cem'), seed=4)
    assert torch.equal(r.parameters, again.parameters)


def test_no_valid_candidates_raises_without_fallback():
    p = ContactProblem([Branch([], np.zeros(1), np.ones(1), 'bad')],
        lambda x, i: (np.zeros((len(x), 2, 1)), [[] for _ in x]),
        lambda u, e: np.full(len(u), np.inf), lambda u, e: [], 4, 2)
    with pytest.raises(NoValidCandidate):
        solve(p, Settings('contact_cem'))


def test_retained_plan_keeps_its_own_events_during_native_review():
    retained = [{'mode': 'roll', 'start_time': .1}]
    observed = []
    def validate(controls, events):
        observed.extend(events)
        return [dict(cost=0. if row == retained else 1.) for row in events]
    p = ContactProblem([Branch([], np.zeros(1), np.ones(1), 'new')],
        lambda x, i: (np.zeros((len(x), 2, 1)), [[] for _ in x]),
        lambda u, e: np.ones(len(u)), validate, 4, 1,
        previous=np.ones((2, 1)), previous_events=retained)
    result = solve(p, Settings('contact_cem'))
    assert observed[-1] == retained
    assert result.metadata['branch'] == 'previous' and result.metadata['events'] == retained


def test_identical_controls_do_not_erase_distinct_contact_modes():
    seen = []
    def verify(controls, events):
        seen.extend(events)
        return [dict(cost=-row[0]['mode_id']) for row in events]
    problem = ContactProblem([Branch([], np.zeros(1), np.ones(1), str(i)) for i in range(2)],
        lambda x, i: (np.zeros((len(x), 2, 1)), [[dict(mode_id=i)] for _ in x]),
        lambda u, e: np.ones(len(u)), verify, 4, 1)
    result = solve(problem, Settings('contact_cem'))
    assert {row[0]['mode_id'] for row in seen} == {0, 1}
    assert result.metadata['branch'] == '1'


def test_nominal_survives_gpu_shortlist_without_increasing_native_quota():
    seen = []
    def verify(controls, events):
        seen.extend(controls[:, 0, 0])
        return [dict(cost=abs(float(u[0, 0]))) for u in controls]
    p = ContactProblem([Branch([], np.zeros(1), np.ones(1), 'residual')],
        lambda x, i: (np.tile(x[:, None], (1, 2, 1)), [[] for _ in x]),
        lambda u, e: -abs(u[:, 0, 0]), verify, 32, 1, retain_nominal=True)
    result = solve(p, Settings('contact_cem'), seed=0)
    assert len(seen) == 8 and 0. in seen
    assert result.metadata['nominal_selected'] and result.cost == 0.
