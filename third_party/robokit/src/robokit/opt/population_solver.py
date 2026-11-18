# pyright: reportGeneralTypeIssues=false
# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportCallIssue=false
# pyright: reportOperatorIssue=false
# pyright: reportInvalidTypeForm=false
"""One population solver. A round is sample -> optimize -> score -> select; the config says which happen when."""

from dataclasses import dataclass, field, replace
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union, cast

import numpy as np
import warp as wp

from robokit.opt.lm_optimizer import LMOptimizerConfig
from robokit.opt.optimizer import IterationCallback, Optimizer, OptimizerConfig, aggregate_residuals_to_costs_kernel
from robokit.opt.sampler import SamplerConfig, _prefix
from robokit.opt.var_values import VarValues
from robokit.terms.task import ResidualTask
from robokit.utils.warp_utils import gather, wp_device_type


@wp.kernel(enable_backward=False)
def _select_best_k_kernel(costs: wp.array1d(dtype=wp.float32), width: int, best_indices: wp.array2d(dtype=wp.int32)):
    """Select each instance's `best_indices.shape[1]` lowest-cost flat indices among `width` candidates.

    Sort keys, in order:
    1. Cost, ascending, with NaN last.
    2. Candidate index, ascending, to break ties.
    """
    b = wp.tid()
    prev_cost = float(-wp.inf)
    prev_j = int(-1)
    for i in range(best_indices.shape[1]):
        best_cost = float(wp.inf)
        best_j = int(-1)
        for j in range(width):
            raw = costs[b * width + j]
            cost = wp.where(wp.isnan(raw), wp.inf, raw)
            after_prev = cost > prev_cost or (cost == prev_cost and j > prev_j)
            if after_prev and (best_j < 0 or cost < best_cost or (cost == best_cost and j < best_j)):
                best_cost = cost
                best_j = j
        best_indices[b, i] = b * width + best_j
        prev_cost = best_cost
        prev_j = best_j


# Warp 1.15+ also records CPU launches and replays them in a native loop, saving each launch's Python cost.
_CPU_CAPTURE = tuple(int(x) for x in wp.__version__.split(".")[:2]) >= (1, 15)

# Tile sort is faster on wide populations, but warp 1.8's CPU tile sort segfaults; 1.8 uses the plain loop above.
_TILE_SELECT = not wp.__version__.startswith("1.8.")


@dataclass
class Stage:
    """A run of `iters` iterations at one population width. Entering a stage keeps each instance's best `num_candidates` (the first stage's is the input width); unset fields keep the previous stage's."""

    iters: int = 0
    """Iterations of this stage."""
    num_candidates: Optional[int] = None
    """Population width per instance; None: the previous stage's, or 1."""
    optimizer: Optional[OptimizerConfig] = None
    """Gradient update; None: the previous stage's. With none in any stage: drawing samplers (MPPI/ES) update by sampling, other solvers by dense LM."""
    sample_every: int = 0
    """Draw with the samplers every this many iterations from the stage's start; 0 in every stage: a drawing sampler draws before every score and once after the last, others at each select."""


@dataclass
class PopulationSolverConfig:
    """What a round does, and when: `stages` run in order, iterations counted across them."""

    stages: List[Stage] = field(default_factory=lambda: [Stage(iters=10)])
    """The schedule, e.g. `[Stage(6, 64, LMOptimizerConfig(lm_lambda=10.0)), Stage(10, 4), Stage(0, 1)]`: 64 candidates for 6 iterations, the best 4 for 10 more, then the best one."""
    samplers: Sequence[SamplerConfig] = field(default_factory=tuple)
    """Built during setup, then applied in order where the stages sample.

    At each sampling iteration:
    1. Sample the population before optimizing.
    2. Restart the optimizer on that population.
    3. At the last iteration, sample after the final score and end on the moved candidates.
    """
    cuda_graph_mode: Literal["none", "full", "iter"] = "iter"
    """"iter" replays one optimizer step per width; "full" the whole solve (no callbacks or early stopping)."""
    use_early_stopping: bool = True
    """Let a segment end at its optimizer's stop rule (`OptimizerConfig.early_stopping_interval`); not under "full"."""
    rank_final: bool = False
    """Score only the final population and return it best first; earlier selects keep the optimization cost."""
    min_improvement: float = 0.0
    """If positive, a drawing sampler's solve ends once each instance's cheapest draw beats its first one (the center) by less than this, `patience` scores in a row; the last sample still moves the candidates. Reads scores on the host."""
    patience: int = 2

    @property
    def num_draws(self) -> int:
        """The widest the population gets per candidate: the drawing sampler's draw count, or 1."""
        return max((s.num_draws for s in self.samplers), default=1)

    @property
    def num_iters(self) -> int:
        """Iterations of the whole schedule."""
        return sum(stage.iters for stage in self.stages)

    @property
    def select_at(self) -> List[int]:
        """Iterations (1-based, after the update) where a stage ends and the next one's width is kept."""
        return [int(i) for i in np.cumsum([stage.iters for stage in self.stages])[:-1]]

    @property
    def num_candidates(self) -> List[int]:
        """Population width per instance in each stage."""
        widths = [1]
        for stage in self.stages:
            widths.append(widths[-1] if stage.num_candidates is None else stage.num_candidates)
        return widths[1:]

    @property
    def optimizer(self) -> Optional[List[Optional[OptimizerConfig]]]:
        """Each stage's gradient update, or None if no stage sets one."""
        configs: List[Optional[OptimizerConfig]] = [None]
        for stage in self.stages:
            configs.append(stage.optimizer or configs[-1])
        return configs[1:] if any(configs) else None

    @property
    def sample_at(self) -> List[int]:
        """Zero-based sampling iterations; empty: the default (see `Stage.sample_every`)."""
        starts = [0, *np.cumsum([stage.iters for stage in self.stages])]
        return [
            i
            for k, s in enumerate(self.stages)
            if s.sample_every
            for i in range(starts[k], starts[k + 1], s.sample_every)
        ]

    def __post_init__(self):
        self.stages = list(self.stages)
        if not self.stages or any(stage.iters < 0 for stage in self.stages):
            raise ValueError("stages must be a non-empty list with non-negative iters")
        if any(not isinstance(s, SamplerConfig) for s in self.samplers):
            raise TypeError("samplers must contain SamplerConfig objects, not runtime samplers")
        if sum(s.num_draws > 1 for s in self.samplers) > 1:  # a second one would draw around the first one's draws
            raise ValueError("at most one sampler may draw more than one candidate")
        if self.min_improvement > 0 and max(self.num_candidates) > 1:  # it compares each instance's one center
            raise ValueError("min_improvement needs one candidate per instance")


class PopulationSolver:
    """Refine `batch * num_candidates[0]` VarValues, reusing one var, allocation, and graph per population width.

    Each scheduled round:
    1. `sample`: move candidates using prior scores, then draw around them.
    2. `optimize`: move each candidate toward lower cost.
    3. `score`: evaluate the terms or score term.
    4. `select`: keep each problem's best `num_candidates[k + 1]` candidates.
    """

    def __init__(
        self,
        terms: Union[List[ResidualTask], List[List[ResidualTask]]],
        config: PopulationSolverConfig,
        device: wp_device_type,
        score_terms: Optional[Union[ResidualTask, List[Optional[ResidualTask]]]] = None,
        *,
        active_dof_mask: Optional[wp.array] = None,
    ):
        self.config = config
        self.device: wp_device_type = wp.get_device(device) if isinstance(device, str) else device
        self._active_dof_mask = active_dof_mask
        num_segments = len(config.select_at) + 1
        per_segment = bool(terms) and isinstance(terms[0], list) and len(terms) > 1
        self.terms: List[List[ResidualTask]] = (
            list(terms)
            if per_segment
            else [list(terms[0] if terms and isinstance(terms[0], list) else terms)] * num_segments
        )
        if len(self.terms) != num_segments:
            raise ValueError(f"terms has {len(self.terms)} segments, schedule has {num_segments}")
        if isinstance(score_terms, list) and len(score_terms) == num_segments:
            self.score_terms: List[Optional[ResidualTask]] = score_terms
        elif isinstance(score_terms, list):
            self.score_terms = [score_terms[0] if score_terms else None] * num_segments
        else:
            self.score_terms = [score_terms] * num_segments
        if config.rank_final:
            self.score_terms = [None] * (num_segments - 1) + self.score_terms[-1:]
        self._can_capture = (
            (self.device.is_cuda or _CPU_CAPTURE)
            and all(t.SUPPORTS_CUDA_GRAPH for seg in self.terms for t in seg)
            and all(s is None or s.SUPPORTS_CUDA_GRAPH for s in self.score_terms)
        )
        self._initialized = False
        self._iteration_callback: Optional[IterationCallback] = None

    # ------------------------------------------------------------------ setup
    def setup(self, var: VarValues):
        cfg = self.config
        num_segments = len(cfg.select_at) + 1
        if len(self.terms) == 1 and num_segments > 1:  # terms assigned after construction: one list for all segments
            self.terms = self.terms * num_segments
        if len(self.score_terms) == 1 and num_segments > 1:
            self.score_terms = self.score_terms * num_segments
        mode = cfg.cuda_graph_mode
        self._use_sub_graph = mode == "iter"
        self._use_early_stopping = cfg.use_early_stopping and mode != "full"
        # Segment k runs iterations [start_k, select_at[k]) at width widths[k]; the input width is segment 0's.
        self._pop = var
        self.final_var = var
        mask = self._active_dof_mask
        if mask is not None and (mask.shape, mask.dtype, mask.device) != (
            (var.tangent_dim,),
            wp.float32,
            wp.get_device(self.device),
        ):
            raise ValueError("active_dof_mask must match the variable tangent dimensions, dtype and device")
        self.samplers = [s.build(var, cfg.num_candidates[0], self._active_dof_mask) for s in cfg.samplers]
        candidates = list(cfg.num_candidates)
        if var.batch_size % candidates[0]:
            raise ValueError(f"input batch {var.batch_size} is not a multiple of num_candidates[0]={candidates[0]}")
        self.batch_size = var.batch_size // candidates[0]
        self._widths = [self.batch_size * w for w in candidates]
        # Keep one var address per population width for captured graphs; the input var has the widest width.
        self._vars: Dict[int, VarValues] = {var.batch_size: var}
        for width in self._widths[1:]:
            if width not in self._vars:
                self._vars[width] = var.gather(wp.zeros((width,), dtype=wp.int32, device=self.device))
        max_width = var.batch_size * cfg.num_draws  # a drawing sampler returns its own wider population
        self.best_indices: List[wp.array] = [
            wp.empty((self.batch_size, w), dtype=wp.int32, device=self.device) for w in cfg.num_candidates[1:]
        ]
        self.best_costs = wp.empty((self.batch_size,), dtype=wp.float32, device=self.device)
        ranked_width = cfg.num_candidates[-1] if cfg.rank_final else 1
        self.ranked_indices = wp.from_numpy(  # final rows per instance; ranking all is opt-in
            np.arange(self.batch_size * ranked_width, dtype=np.int32).reshape((self.batch_size, -1)),
            dtype=wp.int32,
            device=self.device,
        )
        self._select_kernels = {}
        # Build one optimizer per term list at its widest width; narrower solves use a prefix of its buffers.
        self._optimizers: List = [None] * len(self.terms)
        if cfg.optimizer is not None or cfg.num_draws == 1:  # no config and draws: the sampling is the update
            by_key: Dict[Tuple[int, int], object] = {}
            for k, seg_terms in enumerate(self.terms):
                key = (id(seg_terms), id(self._optimizer_config(k)))
                if key not in by_key:  # a drawing sampler hands the optimizer its draws, so build at their width
                    by_key[key] = self._build_optimizer(self._draws(self._vars[self._widths[k]]), k)
                self._optimizers[k] = by_key[key]
        # score buffers at the widest width; prefix at narrower ones
        self.score_residuals: List[Optional[wp.array]] = []
        self._score_costs: List[Optional[wp.array]] = []
        for score_term in self.score_terms:
            if score_term is None:
                self.score_residuals.append(None)
                self._score_costs.append(None)
            else:
                self.score_residuals.append(
                    wp.empty((max_width, score_term.residual_dim), dtype=wp.float32, device=self.device)
                )
                self._score_costs.append(wp.empty((max_width,), dtype=wp.float32, device=self.device))
        self._costs = wp.zeros((max_width,), dtype=wp.float32, device=self.device)
        # one per segment, as the cost kernel sums every column; one zero column if every term is cost-only
        widths = [max(sum(t.residual_dim for t in ts if isinstance(t, ResidualTask)), 1) for ts in self.terms]
        self._residuals = [wp.zeros((max_width, w), dtype=wp.float32, device=self.device) for w in widths]
        self.candidate_scores: Optional[wp.array] = None  # live scores of the current population
        # terms size their buffers to the first population they see: make it the widest
        self._score(self._draws(var), 0, None, rank=False)
        self._full_graph = None
        if mode == "full" and self._can_capture:
            self._warmup_and_capture()
        self._initialized = True

    def _draws(self, var: VarValues) -> VarValues:
        """The population the drawing samplers return for `var`, before any score: the widest it gets."""
        for sampler in self.samplers:
            if sampler.num_draws > 1:
                var = sampler.sample(var, None)
        return var

    def _optimizer_config(self, segment: int):
        c = self.config.optimizer
        return c[min(segment, len(c) - 1)] if isinstance(c, (list, tuple)) else c

    def _build_optimizer(self, var: VarValues, segment: int = 0):
        c = self._optimizer_config(segment) or LMOptimizerConfig()  # dense LM, the default update rule
        c = replace(
            c,
            max_iter=self.config.num_iters,
            early_stopping_interval=c.early_stopping_interval if self._use_early_stopping else 0,
            cuda_graph_mode="iter" if self._use_sub_graph else "none",
        )
        return c.build(self.terms[segment], self.device, self._active_dof_mask)

    # ------------------------------------------------------------------ the round
    def _run_solve_body(self, sync_each_stage: bool = False):
        """Run sample -> optimize -> score -> select, `num_iters` times.

        Execution modes:
        - Normal: follow the configured iteration counts.
        - `sync_each_stage`: eager warmup for "full" capture; drain each select and cap segments at two iterations.
        """
        cfg = self.config
        capturing = self.device.is_capturing
        selects = list(cfg.select_at)
        num_iters = cfg.num_iters
        if sync_each_stage:  # warmup schedule: same selects, at most two iterations per segment
            bounds = [0, *selects, cfg.num_iters]
            lengths = [min(2, b - a) for a, b in zip(bounds[:-1], bounds[1:])]
            selects = [int(g) for g in np.cumsum(lengths)[:-1]]
            num_iters = int(sum(lengths))
        # by default a drawing sampler draws before every score and once after the last; noise lands after selects
        default_at = range(cfg.num_iters + 1) if cfg.num_draws > 1 else cfg.select_at
        samples = set(cfg.sample_at or (default_at if cfg.samplers else ()))
        cur = self._pop
        cur.invalidate()
        segment = it = 0
        ends = sorted({*selects, *samples, num_iters})
        # an optimizer runs until the next select, sample or the end; L-BFGS finishes its last step there
        steps = lambda: min([e for e in ends if e > it], default=it) - it  # noqa: E731
        optimizer = self._optimizers[0]
        reset_on = None  # the population the optimizer was last reset on; a sampler may hand back a wider one
        scores: Optional[wp.array] = None
        self._handed = None  # the population a sampler was handed and keeps moving; the solve ends on it
        for sampler in self.samplers:
            sampler.reset()
        if self._iteration_callback is not None:
            self._iteration_callback(0, cur, self._score(cur, segment, optimizer, rank=False))
        while segment < len(selects) and selects[segment] == 0:  # a select before any optimize
            scores = self._score(cur, segment, optimizer)
            cur = self._select(cur, scores, segment)
            segment += 1
            optimizer = self._optimizers[segment]
            if optimizer is not None:
                optimizer.reset(cur, steps())
                reset_on = cur
            scores = self._score(cur, segment, optimizer, rank=False)
        ranked, recent = False, []  # with `min_improvement`, each score of this solve
        while it < num_iters:
            if it in samples:  # a sampled population is a fresh start for the optimizer (damping, history)
                cur, reset_on = self._sample(cur, scores), None
            if optimizer is not None:
                if cur is not reset_on:
                    optimizer.reset(cur, steps())
                    reset_on = cur
                optimizer.step(cur)
            it += 1
            selecting = segment < len(selects) and it == selects[segment]
            ranked = selecting or it == num_iters
            scores = self._score(cur, segment, optimizer, rank=ranked)
            if cfg.min_improvement > 0 and cfg.num_draws > 1 and not capturing:  # a plateau: skip to the last sample
                recent.append(wp.clone(scores))
                if cfg.patience <= it < num_iters:  # only the last `patience` scores decide; other checks cannot stop
                    costs = np.stack([x.numpy().reshape(self.batch_size, -1) for x in recent[-cfg.patience :]])
                    it = num_iters if np.all(costs[..., 0] - costs.min(-1) < cfg.min_improvement) else it
            if self._use_early_stopping and optimizer is not None and not capturing:
                # converged, or a term asked to stop: skip to this segment's select, as a per-stage solve would
                interval = optimizer.config.early_stopping_interval
                stop = optimizer._task_stop
                if not stop and interval > 0 and it % interval == 0:
                    stop = bool(np.all(optimizer.costs.numpy() <= optimizer.config.cost_tol))
                if stop:
                    it = selects[segment] if segment < len(selects) else num_iters
                    scores = self._score(cur, segment, optimizer)  # rank at the jump target
            if segment < len(selects) and it == selects[segment]:
                cur = self._select(cur, scores, segment)
                segment += 1
                # Next optimizer scores selected candidates on reset; streaming HEC lacks fallback residuals.
                optimizer = self._optimizers[segment]
                if optimizer is not None:
                    optimizer.reset(cur, steps())
                    reset_on = cur
                ranked = it == num_iters
                if ranked or self._iteration_callback is not None:
                    scores = self._score(cur, segment, optimizer, rank=ranked)
                if sync_each_stage:
                    wp.synchronize_device(self.device)
            if self._iteration_callback is not None:
                self._iteration_callback(it, cur, scores)
        if num_iters in samples:  # a last sample moves the candidates by the final scores
            self._sample(cur, scores)
            cur = self._handed if self._handed is not None else cur
            optimizer, ranked = None, False  # the optimizer's costs describe the population before that sample
        if not ranked:  # nothing ranked the final population yet
            scores = self._score(cur, segment, optimizer)
        self.final_var = cur
        self.candidate_scores = scores
        # final: one winner per instance
        width = cur.batch_size // self.batch_size
        if width == 1:
            wp.copy(self.best_costs, scores)
            self._best_var = cur
            return
        ranked = min(width, self.ranked_indices.shape[1])
        self._launch_select(scores, width, ranked, self.ranked_indices, stable=self.config.rank_final)
        winners = self.ranked_indices[:, 0]
        if self.batch_size not in self._vars:
            self._vars[self.batch_size] = cur.gather(winners)
        self._best_var = self._vars[self.batch_size]
        cur.gather(winners, out=self._best_var)
        gather(scores, winners, self.best_costs)

    @property
    def winner_indices(self) -> wp.array:
        """`(batch, 1)` row of each instance's winner: the first ranked index."""
        return cast(wp.array, self.ranked_indices[:, :1])

    def _sample(self, cur: VarValues, scores: Optional[wp.array]) -> VarValues:
        """Move and draw candidates through each sampler in turn; the final draw becomes the population."""
        for sampler in self.samplers:
            out = sampler.sample(cur, scores)
            if out is not cur:  # the sampler returned draws; `cur` is what it keeps moving
                self._handed = cur
            cur = out
        return cur

    def _score(self, cur: VarValues, segment: int, optimizer: Optional[Optimizer], rank: bool = True) -> wp.array:
        """Publish `candidate_scores` from the score term if ranking, else optimizer cost, else terms' cost."""
        b = cur.batch_size
        score_term = self.score_terms[min(segment, len(self.score_terms) - 1)] if rank else None
        if score_term is not None:
            if optimizer is not None:  # a score term may read the plain costs it ranks with (the candidate score does)
                self._score(cur, segment, optimizer, rank=False)
            residual = _prefix(cast(wp.array, self.score_residuals[segment]), b)
            scores = _prefix(cast(wp.array, self._score_costs[segment]), b)
            score_term.precompute(cur, need_gradient=False)
            score_term.compute_weighted_residual(cur, out_residual=residual, row_offset=0)
            wp.launch(aggregate_residuals_to_costs_kernel, dim=b, inputs=[residual, scores], device=self.device)
        elif optimizer is not None and optimizer._iter > 0:  # stepped, so its costs are the population's
            scores = optimizer.costs
        else:
            terms = self.terms[min(segment, len(self.terms) - 1)]
            residual, scores = _prefix(self._residuals[segment], b), _prefix(self._costs, b)
            self._accumulate_cost(cur, terms, residual, scores)
        self.candidate_scores = scores
        return scores

    def _select(self, cur: VarValues, scores: wp.array, k: int) -> VarValues:
        """Keep the best `num_candidates[k + 1]` candidates per problem."""
        width_in = cur.batch_size // self.batch_size
        width_out = self.config.num_candidates[k + 1]
        if width_out > width_in:
            raise ValueError(f"select {k} grows {width_in} -> {width_out}; selects only narrow")
        if width_out == width_in:
            return cur
        self._launch_select(scores, width_in, width_out, self.best_indices[k])
        width = self.batch_size * width_out
        out = self._vars[width]
        cur.gather(self.best_indices[k].reshape((width,)), out=out)
        return out

    def _launch_select(self, scores: wp.array, width_in: int, width_out: int, out: wp.array, stable: bool = False):
        """Write each instance's best `width_out` of `width_in` candidate indices into `out`, best first."""
        if not _TILE_SELECT:  # the plain loop already keeps equal costs in index order
            wp.launch(_select_best_k_kernel, dim=self.batch_size, inputs=[scores, width_in, out], device=self.device)
            return
        key = (width_in, width_out, stable)
        if key not in self._select_kernels:
            threads = 32 * ((width_in + 31) // 32)  # whole warps: CUDA tile sorts mis-rank partial ones
            self._select_kernels[key] = (self._build_select_best_k_kernel(width_in, width_out, stable), threads)
        kernel, threads = self._select_kernels[key]
        wp.launch_tiled(kernel, dim=[self.batch_size], inputs=[scores, out], block_dim=threads, device=self.device)

    def _accumulate_cost(self, var: VarValues, terms: List[ResidualTask], residual: wp.array, cost: wp.array):
        """Sum the residual terms' costs, then add each other term's `compute_weighted_cost`, e.g. a rollout's."""
        for term in terms:
            term.precompute(var, need_gradient=False)
        offset = 0
        for term in terms:
            if isinstance(term, ResidualTask):
                term.compute_weighted_residual(var, out_residual=residual, row_offset=offset)
                offset += term.residual_dim
        wp.launch(aggregate_residuals_to_costs_kernel, dim=cost.shape[0], inputs=[residual, cost], device=self.device)
        for term in terms:
            if not isinstance(term, ResidualTask):
                term.compute_weighted_cost(var, out_cost=cost)

    def _warmup_and_capture(self):
        initial_var = self._pop.clone()
        self._restore_mask = wp.ones(self._pop.batch_size, dtype=wp.int32, device=self.device)
        self._run_solve_body(sync_each_stage=True)
        wp.synchronize_device(self.device)
        self._pop.accept(self._restore_mask, initial_var)
        wp.synchronize_device(self.device)
        self._pop.invalidate()
        with wp.ScopedCapture(device=self.device) as capture:
            self._run_solve_body(sync_each_stage=False)
        self._full_graph = capture.graph
        self._pop.invalidate()

    @staticmethod
    def _build_select_best_k_kernel(num_candidates: int, beam_width: int, stable: bool = False):
        NUM_CANDIDATES = int(num_candidates)
        BEAM_WIDTH = int(beam_width)

        def _template(
            costs: wp.array1d(dtype=wp.float32),
            best_indices: wp.array2d(dtype=wp.int32),
        ):
            batch_idx = wp.tid()
            start = batch_idx * NUM_CANDIDATES

            cost_tile = wp.tile_load(costs, shape=(NUM_CANDIDATES,), offset=(start,), storage="shared")
            index_tile = wp.tile_arange(0, NUM_CANDIDATES, dtype=wp.int32, storage="shared")

            if wp.static(stable):
                # Real infinities must sort before the tile sort's infinity padding; `tile_full` needs warp >= 1.9.
                cost_tile = wp.tile_map(
                    wp.min, cost_tile, wp.tile_ones(shape=(NUM_CANDIDATES,), dtype=wp.float32) * 3.402823e38
                )
            wp.tile_sort(cost_tile, index_tile)
            if wp.static(stable):
                # Preserve input order within equal-cost groups after the tile sort.
                for i in range(1, NUM_CANDIDATES):
                    score = wp.tile_extract(cost_tile, i)
                    index = wp.tile_extract(index_tile, i)
                    j = i - 1
                    while j >= 0:
                        if wp.tile_extract(cost_tile, j) != score or wp.tile_extract(index_tile, j) < index:
                            break
                        index_tile[j + 1] = wp.tile_extract(index_tile, j)
                        j -= 1
                    index_tile[j + 1] = index

            for i in range(BEAM_WIDTH):
                best_indices[batch_idx, i] = start + wp.tile_extract(index_tile, i)

        _template.__name__ = f"_simple_select_best_k_{NUM_CANDIDATES}to{BEAM_WIDTH}_{int(stable)}"
        _template.__qualname__ = f"_simple_select_best_k_{NUM_CANDIDATES}to{BEAM_WIDTH}_{int(stable)}"
        return wp.kernel(enable_backward=False, module="unique")(_template)

    def solve(
        self, var: VarValues, iteration_callback: Optional[IterationCallback] = None
    ) -> Tuple[VarValues, wp.array]:
        """Run the schedule on widest `var`; return each instance's best candidate and cost."""
        if iteration_callback is not None and self.config.cuda_graph_mode == "full" and self._can_capture:
            raise ValueError("iteration_callback is incompatible with full-solve CUDA graph capture")
        if not self._initialized:
            self.setup(var)
        self._iteration_callback = iteration_callback
        if self._full_graph is not None:  # replay reads the population it was recorded on
            if var.batch_size != self._pop.batch_size:
                raise ValueError(f"the recorded solve takes {self._pop.batch_size} candidates, got {var.batch_size}")
            if var is not self._pop:
                self._pop.accept(self._restore_mask, var)
            if self.device.is_capturing:
                self._run_solve_body()  # CUDA cannot launch a graph while capturing; record its kernels instead
            else:
                wp.capture_launch(self._full_graph)
            if var is not self._pop:
                var.accept(self._restore_mask, self._pop)
        else:
            self._pop = var
            self._run_solve_body()
        return self._best_var, self.best_costs

    def compute_costs(self, var: VarValues) -> Dict[str, wp.array]:
        """Sum final-segment term costs by `cost_name`, including duplicate names such as ConfigurationTask."""
        terms = self.terms[-1]
        b = var.batch_size
        for term in terms:
            term.precompute(var, need_gradient=False)
        dims: Dict[str, int] = {}
        for term in terms:
            dims[term.cost_name] = dims.get(term.cost_name, 0) + term.residual_dim
        residuals = {n: wp.zeros((b, d), dtype=wp.float32, device=self.device) for n, d in dims.items()}
        offsets = dict.fromkeys(dims, 0)
        for term in terms:
            term.compute_weighted_residual(
                var, out_residual=residuals[term.cost_name], row_offset=offsets[term.cost_name]
            )
            offsets[term.cost_name] += term.residual_dim
        costs = {n: wp.zeros((b,), dtype=wp.float32, device=self.device) for n in dims}
        for name, residual in residuals.items():
            wp.launch(aggregate_residuals_to_costs_kernel, dim=b, inputs=[residual, costs[name]], device=self.device)
        return costs
