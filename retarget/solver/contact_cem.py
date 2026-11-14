"""Branch-preserving contact-space CEM; physics belongs to the supplied adapter."""
from dataclasses import dataclass, field, asdict
from typing import Callable
import warp as wp
from enum import Enum
from robokit.opt.population_solver import PopulationSolver, PopulationSolverConfig, Stage
from robokit.opt.var_values import VarValues, VectorVar
from robokit.terms.task import Task
from .robokit_cem import CEMSamplerConfig, CEMPopulationSolver
import numpy as np
import torch
from .base import Result, NoValidCandidate
from .contact_defaults import Parameters, DEFAULTS


@dataclass
class ContactEvent:
    geom1: int
    geom2: int
    link_name: str
    object_part: str
    link_local_point: list
    object_local_point: list
    link_local_normal: list
    object_local_normal: list
    start_time: float
    duration: float
    approach_direction: list
    normal_velocity: float = -.02
    approach_normal_velocity: float | None = None
    tangential_velocity: list = field(default_factory=lambda: [0., 0., 0.])
    mode: str = 'stick'

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Branch:
    events: list
    mean: np.ndarray
    sigma: np.ndarray
    name: str


@dataclass
class ContactProblem:
    branches: list
    decode: Callable
    evaluate: Callable
    validate: Callable
    population: int
    iterations: int
    previous: np.ndarray | None = None
    previous_events: list | None = None
    device: str = "cpu"
    retain_nominal: bool = False


def solve(problem: ContactProblem, *, seed: int = 0) -> Result:
    if not isinstance(problem, ContactProblem):
        raise TypeError('contact_cem requires ContactProblem; ordinary joint Problem is unsupported')
    if not problem.branches or len(problem.branches) > problem.population or problem.population < 2 or problem.iterations < 1:
        raise ValueError('Nonempty contact branches and a positive search budget are required')
    if problem.population % len(problem.branches):
        raise ValueError('Robokit branch populations require a budget divisible by branch count')
    class Key(Enum):
        CONTACT = 0
    history, archive, sampled, evaluated = [], [], 0, 0
    class ContactScore(Task):
        preparing = True
        def compute_weighted_cost(self, values, *, out_cost):
            nonlocal sampled, evaluated
            if self.preparing:
                out_cost.zero_()
                return
            parameters = values.get(Key.CONTACT).x.numpy()
            count = problem.population // len(problem.branches)
            controls, events = [], []
            for branch in range(len(problem.branches)):
                begin = branch * count
                u, e = problem.decode(parameters[begin:begin + count], branch)
                controls.extend(u); events.extend(e)
            controls = np.asarray(controls)
            cost = np.asarray(problem.evaluate(controls, events), dtype=float)
            cost[~np.isfinite(cost)] = np.inf
            sampled += len(cost); evaluated += len(cost)
            record = dict(iteration=len(history), branches=[], finite=int(np.isfinite(cost).sum()))
            for branch, template in enumerate(problem.branches):
                ids = np.arange(branch * count, (branch + 1) * count)
                ids = ids[np.isfinite(cost[ids])]
                ids = ids[np.argsort(cost[ids], kind='stable')]
                record['branches'].append(dict(name=template.name, finite=len(ids),
                    cost=float(cost[ids[0]]) if len(ids) else None))
                keep = ids[:DEFAULTS.cpu_candidates].tolist()
                if problem.retain_nominal:
                    keep.extend(j for j in ids if np.all(parameters[j] == 0.) and j not in keep)
                archive.extend((float(cost[j]), controls[j].copy(), events[j], parameters[j].copy(), branch)
                    for j in keep)
            history.append(record)
            out_cost.assign(cost.astype(np.float32))
    means = np.asarray([branch.mean for branch in problem.branches], np.float32)
    sigmas = np.asarray([branch.sigma for branch in problem.branches], np.float32)
    values = VarValues({Key.CONTACT: VectorVar(wp.array(means, dtype=float, device=problem.device))})
    sampler = CEMSamplerConfig(var_key=Key.CONTACT,
        num_draws=problem.population // len(problem.branches), branch_sigma=sigmas, seed=seed,
        cem_elite_fraction=DEFAULTS.elite_fraction, cem_refit=DEFAULTS.refit,
        cem_min_sigma_fraction=DEFAULTS.min_sigma_fraction)
    scorer = ContactScore()
    engine = CEMPopulationSolver([scorer], PopulationSolverConfig(
        stages=[Stage(problem.iterations, sample_every=1)], samplers=[sampler], cuda_graph_mode='none'),
        problem.device)
    engine.setup(values)
    scorer.preparing = False
    engine.solve(values)
    distribution = engine.samplers[0]
    delta = wp.zeros_like(distribution.std)
    distribution._move(engine.candidate_scores, delta)
    means = distribution._centers.get(Key.CONTACT).x.numpy() + delta.numpy()
    sigmas = distribution.std.numpy()
    if not archive:
        raise NoValidCandidate(history)
    # Reserve a slot for each distinct branch, then fill by score. Deduplicate controls.
    archive.sort(key=lambda row: row[0])
    selected, seen, covered = [], set(), set()
    ordered = [next(row for row in archive if row[4] == i) for i in sorted({row[4] for row in archive})]
    if problem.retain_nominal:
        nominal = next((row for row in archive if np.all(row[3] == 0.)), None)
        if nominal is not None:
            ordered.insert(0, nominal)
    ordered.extend(archive)
    for row in ordered:
        key = (row[1].tobytes(), repr(row[2]))
        if key in seen:
            continue
        selected.append(row); seen.add(key)
        if len(selected) == DEFAULTS.cpu_candidates:
            break
    candidate_controls = [row[1] for row in selected]
    candidate_events = [row[2] for row in selected]
    if problem.previous is not None:
        candidate_controls.append(problem.previous)
        if problem.previous_events is None:
            raise ValueError('A retained contact plan requires its shifted contact events')
        candidate_events.append(problem.previous_events)
    validation = problem.validate(np.asarray(candidate_controls), candidate_events)
    native_cost = np.asarray([row['cost'] for row in validation], dtype=float)
    native_cost[~np.isfinite(native_cost)] = np.inf
    evaluated += len(native_cost)
    if not np.isfinite(native_cost).any():
        raise NoValidCandidate(history + [dict(cpu4_valid=0)])
    winner = int(native_cost.argmin())
    row = selected[winner] if winner < len(selected) else None
    params = row[3] if row is not None else problem.branches[0].mean
    return Result(torch.as_tensor(params), float(native_cost[winner]), history, sampled, evaluated,
        dict(controls=candidate_controls[winner], events=candidate_events[winner],
             optimizer_backend="robokit.PopulationSolver/CEMSampler(GaussianSampler)",
             branch=problem.branches[row[4]].name if row is not None else 'previous',
             nominal_selected=bool(row is not None and np.all(row[3] == 0.)),
             cpu4=validation[winner], cpu4_evaluated=len(native_cost),
             branch_distributions=[dict(name=b.name, mean=m.tolist(), sigma=s.tolist())
                                   for b, m, s in zip(problem.branches, means, sigmas)]))
