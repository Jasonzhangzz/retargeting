# pyright: reportGeneralTypeIssues=false
# pyright: reportCallIssue=false
# pyright: reportIndexIssue=false
# pyright: reportArgumentType=false
# pyright: reportOperatorIssue=false
"""Draw manifold-aware candidates for one named variable via tangent noise and `VarValues.integrate`.

Per-width lifecycle:
1. Allocate during the first eager `sample` call.
2. Launch kernels only on subsequent calls, including captured CUDA graphs.
"""

from dataclasses import asdict, dataclass
from enum import Enum
from functools import lru_cache
from typing import Dict, Literal, Optional, Protocol, Sequence, Union, cast

import numpy as np
import warp as wp

from robokit.opt.var_values import VarKey, VarValues, VectorVar
from robokit.utils.warp_utils import wp_device_type


Weighting = Literal["softmax", "fitness_shaping"]
"""How scored draws become weights: a Boltzmann softmax over negative cost, or standardized costs."""


# --- kernels ------------------------------------------------------------------
@wp.kernel
def _tile_offsets_kernel(
    offsets: wp.array2d(dtype=wp.float32),
    scale: wp.array1d(dtype=wp.float32),
    num_candidates: wp.int32,
    out_tangent: wp.array2d(dtype=wp.float32),
):
    """Broadcast one offset per candidate across every instance, scaled per tangent column."""
    batch_idx, col = wp.tid()  # pyright: ignore[reportAssignmentType]
    out_tangent[batch_idx, col] = offsets[batch_idx % num_candidates, col] * scale[col]


@wp.kernel
def _gaussian_sample_kernel(
    seed: int,
    epoch: wp.array1d(dtype=wp.int32),
    salt: int,
    num_draws: int,
    tangent_dim: int,
    active_dof_mask: wp.array1d(dtype=wp.float32),
    std: wp.array2d(dtype=wp.float32),
    num_wide: int,
    wide_scale: float,
    noise: wp.array2d(dtype=wp.float32),
):
    batch_idx, dof_idx = wp.tid()  # pyright: ignore[reportAssignmentType]
    if (batch_idx % num_draws) == 0 or active_dof_mask[dof_idx] == 0.0:
        noise[batch_idx, dof_idx] = 0.0
    else:
        state = wp.rand_init(seed + epoch[0] * 7919, (salt * 1000003 + batch_idx) * tangent_dim + dof_idx)
        wide = salt == 0 and batch_idx % num_draws >= num_draws - num_wide  # a solve's first round: the last draws
        noise[batch_idx, dof_idx] = (
            std[batch_idx / num_draws, dof_idx] * wp.randn(state) * wp.where(wide, wide_scale, 1.0)
        )


@lru_cache(maxsize=None)  # one kernel per config; on warp 1.8, freeing one sampler's copy crashed the others
def _build_weights_kernel(num_draws: int, weighting: Weighting):
    """Per center: cost weights over its draws. Each thread of the block strides over the draws, so shared memory stays one value per thread at any `num_draws`."""
    draws = wp.constant(num_draws)
    softmax = weighting == "softmax"

    def _kernel(
        costs: wp.array1d(dtype=wp.float32),
        temperature: wp.array1d(dtype=wp.float32),
        epsilon: float,
        block: int,
        weights: wp.array2d(dtype=wp.float32),
    ):
        batch_idx, lane = wp.tid()  # pyright: ignore[reportAssignmentType]
        start = batch_idx * draws
        if wp.static(softmax):  # Boltzmann softmax over negative cost
            low = wp.float32(3.4e38)
            for k in range(lane, draws, block):
                low = wp.min(low, costs[start + k])
            low = wp.tile_min(wp.tile(low))[0]
            total = wp.float32(0.0)
            for k in range(lane, draws, block):
                total += wp.exp((low - costs[start + k]) / temperature[batch_idx])
            total = wp.tile_sum(wp.tile(total))[0]
            for k in range(lane, draws, block):
                weights[batch_idx, k] = wp.exp((low - costs[start + k]) / temperature[batch_idx]) / total
        else:  # fitness shaping: standardize the costs
            total = wp.float32(0.0)
            for k in range(lane, draws, block):
                total += costs[start + k]
            mean = wp.tile_sum(wp.tile(total))[0] / float(draws)
            squares = wp.float32(0.0)
            for k in range(lane, draws, block):
                squares += (mean - costs[start + k]) * (mean - costs[start + k])
            spread = wp.sqrt(wp.tile_sum(wp.tile(squares))[0] / float(max(draws - 1, 1))) + epsilon
            for k in range(lane, draws, block):
                weights[batch_idx, k] = (mean - costs[start + k]) / spread

    _kernel.__name__ = _kernel.__qualname__ = f"_{weighting}_weights_{num_draws}"
    return wp.kernel(enable_backward=False, module="unique")(_kernel)


@lru_cache(maxsize=None)
def _build_update_kernel(num_draws: int, weighting: Weighting, best: bool):
    """Per (center, dim): the step to take from the weighted noise, re-fitting `std` for MPPI; strided like weights."""
    draws = wp.constant(num_draws)
    softmax = weighting == "softmax"

    def _kernel(
        weights: wp.array2d(dtype=wp.float32),
        noise: wp.array2d(dtype=wp.float32),
        costs: wp.array1d(dtype=wp.float32),
        best_indices: wp.array1d(dtype=wp.int32),
        gamma: float,
        min_std: float,
        learning_rate: float,
        epsilon: float,
        block: int,
        std: wp.array2d(dtype=wp.float32),
        delta: wp.array2d(dtype=wp.float32),
    ):
        batch_idx, dim_idx, lane = wp.tid()  # pyright: ignore[reportAssignmentType]
        step = wp.float32(0.0)
        if wp.static(softmax or not best):  # the weighted mean of the noise, and for softmax the refit of std
            second = wp.float32(0.0)  # softmax: the weighted spread; fitness shaping: the squared weights
            for k in range(lane, draws, block):
                w, n = weights[batch_idx, k], noise[batch_idx * draws + k, dim_idx]
                step += w * n
                if wp.static(softmax):
                    second += w * n * n
                else:
                    second += w * w
            step = wp.tile_sum(wp.tile(step))[0]
            second = wp.tile_sum(wp.tile(second))[0]
            if wp.static(softmax):  # move toward the weighted spread and adapt std to it
                old_std = std[batch_idx, dim_idx]
                if lane == 0:
                    std[batch_idx, dim_idx] = max(wp.sqrt((1.0 - gamma) * old_std * old_std + gamma * second), min_std)
            else:  # natural-gradient step scaled by the weight spread; std stays fixed
                variance = std[batch_idx, dim_idx] * std[batch_idx, dim_idx]
                step = learning_rate * step / (float(draws) * wp.sqrt(second / float(draws)) * variance + epsilon)
        if wp.static(best):  # land exactly on the cheapest draw, the first one on ties
            low, low_idx = wp.float32(3.4e38), int(0)  # all draws inf or NaN: the first one
            for k in range(lane, draws, block):
                if costs[batch_idx * draws + k] < low:
                    low = costs[batch_idx * draws + k]
                    low_idx = k
            lowest = wp.tile_min(wp.tile(low))[0]
            best_idx = batch_idx * draws + wp.tile_min(wp.tile(wp.where(low == lowest, low_idx, int(draws))))[0]
            if wp.static(not softmax):  # nothing per dim above, so one block per parent writes every dim
                for d in range(lane, delta.shape[1], block):
                    delta[batch_idx, d] = noise[best_idx, d]
                if lane == 0:
                    best_indices[batch_idx] = best_idx
                return
            if dim_idx == 0 and lane == 0:
                best_indices[batch_idx] = best_idx
            step = noise[best_idx, dim_idx]
        if lane == 0:
            delta[batch_idx, dim_idx] = step

    _kernel.__name__ = _kernel.__qualname__ = f"_{weighting}_update_{num_draws}_{int(best)}"
    return wp.kernel(enable_backward=False, module="unique")(_kernel)


@wp.kernel
def _compute_order_kernel(
    cost: wp.array(dtype=float), num_draws: int, keys: wp.array(dtype=float), order: wp.array(dtype=int)
):
    """Stable sort of each center's draw costs into `keys`, with the draws in `order`; NaN sorts as inf."""
    i = wp.tid()
    start, rank = (i // num_draws) * num_draws, int(0)
    c = wp.where(wp.isnan(cost[i]), wp.inf, cost[i])
    for j in range(start, start + num_draws):
        other = wp.where(wp.isnan(cost[j]), wp.inf, cost[j])
        if other < c or (other == c and j < i):
            rank += 1
    keys[start + rank] = c
    order[start + rank] = i


@wp.kernel(enable_backward=False)
def _compute_elite_kernel(
    keys: wp.array(dtype=float),
    order: wp.array(dtype=int),
    num_draws: int,
    num_elite: int,
    block: int,
    temperature: wp.array(dtype=wp.float32),
    best_indices: wp.array(dtype=wp.int32),
):
    """Per center, from its sorted costs: the temperature (cost spread of the elite) and the cheapest draw."""
    b, lane = wp.tid()  # pyright: ignore[reportAssignmentType]
    start = b * num_draws
    total = float(0.0)
    for j in range(lane, num_elite, block):
        total += keys[start + j]
    mean = wp.tile_sum(wp.tile(total))[0] / float(num_elite)
    spread = float(0.0)
    for j in range(lane, num_elite, block):
        spread += (keys[start + j] - mean) * (keys[start + j] - mean)
    spread = wp.tile_sum(wp.tile(spread))[0]
    if lane == 0:
        temperature[b] = wp.sqrt(spread / float(num_elite)) + 1.0e-9
        best_indices[b] = order[start]


@wp.kernel
def _set_dropped_cost_kernel(order: wp.array(dtype=int), num_draws: int, num_elite: int, cost: wp.array(dtype=float)):
    i = wp.tid()
    dropped = num_draws - num_elite
    cost[order[(i // dropped) * num_draws + num_elite + i % dropped]] = 1.0e30


@wp.kernel
def _set_decayed_std_kernel(decay: float, min_std: float, std: wp.array2d(dtype=wp.float32)):
    i, j = wp.tid()  # pyright: ignore[reportAssignmentType]
    std[i, j] = wp.max(std[i, j] * decay, min_std)


@wp.kernel
def _set_bounded_draws_kernel(
    lower: wp.array2d(dtype=wp.float32),
    upper: wp.array2d(dtype=wp.float32),
    num_draws: int,
    draws: wp.array2d(dtype=wp.float32),
    noise: wp.array2d(dtype=wp.float32),
):
    i, j = wp.tid()  # pyright: ignore[reportAssignmentType]
    value = wp.clamp(draws[i, j], lower[i // num_draws, j], upper[i // num_draws, j])
    noise[i, j] += value - draws[i, j]
    draws[i, j] = value


# --- helpers ------------------------------------------------------------------
def _prefix(buf: wp.array, batch_size: int) -> wp.array:
    """Return first `batch_size` rows of a capacity buffer, or all rows at full width for zero-column support."""
    return buf if batch_size == buf.shape[0] else cast(wp.array, buf[:batch_size])


def roberts_samples(num_points: int, dim: int, offset: int = 0) -> np.ndarray:
    """Low-discrepancy points in `[0, 1)` of shape `(num_points, dim)`.

    Example:
        >>> roberts_samples(5, 3).shape
        (5, 3)
        >>> bool(roberts_samples(64, 2).max() < 1.0)
        True
    """
    root = 1.0
    for _ in range(10000):
        f = root ** (dim + 1) - root - 1
        root = root - f / ((dim + 1) * root**dim - 1)
        if abs(f) < 1e-10:
            break
    basis = 1 - (1 / root ** (1 + np.arange(dim)))
    n = np.arange(num_points) + offset
    samples, _ = np.modf(n[:, None] * basis[None, :])
    return samples


# --- sampling interface -------------------------------------------------------
class Sampler(Protocol):
    """Draws new candidates around each candidate of one named variable."""

    var_key: Enum
    num_draws: int
    """Candidates drawn around each one; 1 draws in place."""

    def reset(self) -> None:
        """Reopen the distribution at the start of a solve."""
        ...

    def sample(self, var: VarValues, scores: Optional[wp.array]) -> VarValues:
        """Move candidates by `scores`; return `var` or sampler-owned draws for scoring."""
        ...


@dataclass(frozen=True)
class SamplerConfig:
    """Sampler settings; owners supply variables, population width, device and masks at build time."""

    var_key: Enum = VarKey.ROBOT
    num_draws = 1

    def build(self, var: VarValues, num_candidates: int, active_dof_mask: Optional[wp.array] = None) -> Sampler:
        """Construct an independent runtime sampler for these variables."""
        raise NotImplementedError


# --- Roberts sampler ----------------------------------------------------------
@dataclass(frozen=True)
class RobertsSamplerConfig(SamplerConfig):
    """Low-discrepancy initialization settings; helpers may scale the spread by joint ranges.

    Example:
        >>> config = RobertsSamplerConfig(scale=0.5, keep_first=True)
        >>> config.scale, config.keep_first, config.num_draws
        (0.5, True, 1)
    """

    scale: Union[float, Sequence[float]] = 0.2
    seed: Optional[int] = None
    keep_first: bool = False

    def build(self, var: VarValues, num_candidates: int, active_dof_mask: Optional[wp.array] = None) -> Sampler:
        width = var.get(self.var_key).tangent_dim
        scale = np.broadcast_to(self.scale, (width,)).copy()
        if active_dof_mask is not None:
            offset = var.tangent_offset(self.var_key)
            scale *= active_dof_mask.numpy()[offset : offset + width]
        return RobertsSampler(num_candidates, scale, self.seed, self.keep_first, var.device, self.var_key)


class RobertsSampler:
    """Seed candidates once with a Roberts low-discrepancy sequence, covering a box more evenly than random draws.

    Sampling options:
    - `scale`: half-width per tangent column; joint columns usually use joint range.
    - `sampler.scale.assign(...)`: change the device-side scale between solves.
    - `keep_first`: leave candidate zero at the seed by assigning it zero noise.

    Example:
        >>> sampler = RobertsSampler(num_candidates=4, scale=[0.5, 0.5], device="cpu")
        >>> tangent = wp.zeros((8, 2), dtype=wp.float32, device="cpu")
        >>> sampler.fill(tangent)
        >>> tangent.numpy().shape
        (8, 2)
        >>> bool(np.abs(tangent.numpy()).max() <= 0.25)
        True
    """

    def __init__(
        self,
        num_candidates: int,
        scale: Sequence[float],
        seed: Optional[int] = None,
        keep_first: bool = False,
        device: wp_device_type = "cpu",
        var_key: Enum = VarKey.ROBOT,
    ):
        self.num_candidates, self.var_key = num_candidates, var_key
        self.tangent_dim = len(scale)
        offsets = roberts_samples(num_candidates, self.tangent_dim, (seed % 10000) if seed is not None else 0) - 0.5
        if keep_first:
            offsets = np.vstack([np.zeros_like(offsets[:1]), offsets[:-1]])
        self._offsets = wp.from_numpy(offsets.astype(np.float32), dtype=wp.float32, device=device)
        self.scale = wp.from_numpy(np.asarray(scale, dtype=np.float32), dtype=wp.float32, device=device)
        self._device = device
        self._noise: Dict[int, wp.array] = {}  # per population width

    num_draws = 1

    def reset(self) -> None:
        """A Roberts spread is fixed."""

    def sample(self, var: VarValues, scores: Optional[wp.array]) -> VarValues:
        """A Roberts spread is fixed, so `scores` tell it nothing: spread `var` in place."""
        noise = self._noise.get(var.batch_size)
        if noise is None:
            noise = self._noise[var.batch_size] = wp.zeros(
                (var.batch_size, var.tangent_dim), dtype=wp.float32, device=self._device
            )
        offset = var.tangent_offset(self.var_key)
        self.fill(noise[:, offset : offset + self.tangent_dim])
        var.integrate(noise, out=var)
        return var

    def fill(self, out_tangent: wp.array, iteration: int = 0) -> None:
        if out_tangent.shape[1] != self.tangent_dim:
            raise ValueError(f"expected tangent_dim {self.tangent_dim}, got {out_tangent.shape[1]}")
        wp.launch(
            _tile_offsets_kernel,
            dim=out_tangent.shape,
            inputs=[self._offsets, self.scale, self.num_candidates, out_tangent],
            device=self._device,
        )


# --- Gaussian sampler ---------------------------------------------------------
@dataclass(frozen=True)
class GaussianSamplerConfig(SamplerConfig):
    """Gaussian draw/update settings, independent of runtime population buffers.

    Example:
        >>> from robokit.opt.var_values import VectorVar
        >>> class Key(Enum):
        ...     X = 0
        >>> var = VarValues({Key.X: VectorVar(wp.zeros((1, 2), dtype=wp.float32, device="cpu"))})
        >>> config = GaussianSamplerConfig(var_key=Key.X, num_draws=8)
        >>> first, second = config.build(var, 1), config.build(var, 1)
        >>> first is second
        False
        >>> first.noise.shape
        (8, 2)
    """

    num_draws: int = 64
    """Candidates drawn around each center, the center itself first."""
    weighting: Weighting = "softmax"
    """How scored draws become weights: MPPI's softmax, or evolution strategies' fitness shaping."""
    init_std: float = 0.1
    """Noise standard deviation at the start of a solve."""
    min_std: float = 0.01
    """Floor the softmax refit keeps the standard deviation above."""
    temperature: float = 0.1
    """Softmax temperature; lower follows the cheapest draws more closely."""
    gamma: float = 0.9
    """Share of the weighted spread the softmax refit moves the standard deviation toward."""
    learning_rate: float = 0.1
    """Fitness-shaping step size."""
    epsilon: float = 1e-6
    """Guard against division by a zero spread."""
    seed: int = 0
    """Noise seed; the same seed replays the same draws."""
    sample_mode: Literal["mean", "best"] = "mean"
    """Move each center to its draws' weighted mean, or onto its cheapest draw."""
    elite_ratio: float = 0.0
    """If positive, the softmax temperature is each center's cost spread over this best share of its draws (at least two) instead of `temperature`, and `best_draws` keeps its cheapest draw."""
    mean_elites: bool = False
    """With `elite_ratio`, move each center to the weighted mean of that share alone."""
    noise_decay: float = 1.0
    """Each scored round scales `std` by this before the refit, never below `min_std`."""
    wide_ratio: float = 0.0
    """Share of a solve's first-round draws (its last ones) drawn at `wide_scale` times `std`, to explore."""
    wide_scale: float = 1.0

    def build(self, var: VarValues, num_candidates: int, active_dof_mask: Optional[wp.array] = None) -> Sampler:
        width = var.get(self.var_key).tangent_dim
        offset = var.tangent_offset(self.var_key)
        return GaussianSampler(
            **asdict(self),
            tangent_dim=width,
            instances=var.batch_size,
            active_dof_mask=None if active_dof_mask is None else active_dof_mask[offset : offset + width],
            device=var.device,
        )


class GaussianSampler:
    """Fit a sampling distribution from costs, as in MPPI and evolution strategies.

    Each round:
    1. Move centers using scored draws and refit `std`.
    2. Draw `num_draws` std-scaled normals per center; draw zero has no noise and preserves the center.

    Center updates:
    - `sample_mode=mean`: cost-weighted mean of noise.
    - `sample_mode=best`: cheapest draw's current position, including any optimizer movement.

    Cost weighting:
    - softmax: Boltzmann weights over negative cost at `temperature`.
    - fitness_shaping: standardized costs scaled by `learning_rate`.

    Capacity/reset controls:
    - `instances`: maximum center population; narrower solves use prefixes.
    - `std_reset` and `temperatures`: caller-adjustable per-instance settings; `reset` restores std from `std_reset`.
    - `lower` and `upper`: optional `(instances, tangent_dim)` bounds on a vector variable's drawn values; each draw is clamped to its center's and its noise follows.
    """

    def __init__(
        self,
        num_draws: int,
        tangent_dim: int,
        instances: int,
        weighting: Weighting = "softmax",
        init_std: float = 0.1,
        min_std: float = 0.01,
        temperature: float = 0.1,
        gamma: float = 0.9,
        learning_rate: float = 0.1,
        epsilon: float = 1e-6,
        seed: int = 0,
        sample_mode: Literal["mean", "best"] = "mean",
        elite_ratio: float = 0.0,
        mean_elites: bool = False,
        noise_decay: float = 1.0,
        wide_ratio: float = 0.0,
        wide_scale: float = 1.0,
        active_dof_mask: Optional[wp.array] = None,
        device: wp_device_type = "cpu",
        var_key: Enum = VarKey.ROBOT,
    ):
        device = wp.get_device(device) if isinstance(device, str) else device
        self.num_draws, self.tangent_dim, self.instances, self.var_key = num_draws, tangent_dim, instances, var_key
        self.weighting, self.init_std, self.min_std = weighting, init_std, min_std
        self.temperature, self.gamma = temperature, gamma
        self.learning_rate, self.epsilon = learning_rate, epsilon
        self.seed, self.sample_mode = seed, sample_mode
        self.elite_ratio, self.mean_elites, self.noise_decay = elite_ratio, mean_elites, noise_decay
        self.num_wide, self.wide_scale = int(wide_ratio * (num_draws - 1)), wide_scale
        self._device = device
        self.active_dof_mask = (
            active_dof_mask if active_dof_mask is not None else wp.ones(tangent_dim, dtype=wp.float32, device=device)
        )
        if self.active_dof_mask.shape[0] != tangent_dim:
            raise ValueError(f"active_dof_mask must have length {tangent_dim}")
        self.noise = wp.zeros((instances * num_draws, tangent_dim), dtype=wp.float32, device=device)
        self.std = wp.full((instances, tangent_dim), init_std, dtype=wp.float32, device=device)
        self.std_reset = wp.full((instances, tangent_dim), init_std, dtype=wp.float32, device=device)
        # read on device, so an owner advancing it between solves draws fresh noise through a captured graph
        self.epoch = wp.zeros(1, dtype=wp.int32, device=device)
        self.temperatures = wp.full((instances,), temperature, dtype=wp.float32, device=device)
        self.weights = wp.zeros((instances, num_draws), dtype=wp.float32, device=device)
        self.best_indices = wp.zeros((instances,), dtype=wp.int32, device=device)
        if elite_ratio > 0:  # sorted costs, their draws, and with `mean_elites` the costs with the rest dropped
            n = instances * num_draws
            self._keys, self._elite_costs = wp.zeros(n, device=device), wp.zeros(n, device=device)
            self._order = wp.zeros(n, dtype=int, device=device)
        self.lower: Optional[wp.array] = None
        self.upper: Optional[wp.array] = None
        self.best_draws: Optional[VarValues] = None  # with `elite_ratio`, each center's cheapest draw last scored
        best = sample_mode == "best"
        self._weights_kernel = (
            None if best and weighting == "fitness_shaping" else _build_weights_kernel(num_draws, weighting)
        )
        self._update_kernel = _build_update_kernel(num_draws, weighting, best)
        # Sized to the whole tangent on the first `sample`; the draws var is one gather of the centers per width.
        self._noise_full: Optional[wp.array] = None
        self._step: Optional[wp.array] = None
        self._draws_by_width: Dict[int, VarValues] = {}
        self._repeat_idx: Dict[int, wp.array] = {}
        self.draws: Optional[VarValues] = None  # the population `sample` returned last
        self._centers: Optional[VarValues] = None
        self._round = 0

    def reset(self) -> None:
        """Reopen the spread to `std_reset` and replay the same noise sequence."""
        wp.copy(self.std, self.std_reset)
        self._round = 0

    def sample(self, var: VarValues, scores: Optional[wp.array]) -> VarValues:
        offset, width = var.tangent_offset(self.var_key), var.tangent_dim
        if self._noise_full is None or self._step is None:
            self._noise_full = wp.zeros((self.instances * self.num_draws, width), dtype=wp.float32, device=self._device)
            self._step = wp.zeros((self.instances, width), dtype=wp.float32, device=self._device)
        if var is self.draws and scores is not None:  # its draws came back scored: move their centers
            centers = cast(VarValues, self._centers)
            step = _prefix(self._step, centers.batch_size)
            self._move(scores, step[:, offset : offset + self.tangent_dim])
            if self.elite_ratio > 0:  # before the redraw overwrites the draws
                best = _prefix(self.best_indices, centers.batch_size)
                self.best_draws = var.gather(best, out=self.best_draws) if self.best_draws else var.gather(best)
            if self.sample_mode == "best":  # the draw as it is now, optimizer steps included
                var.gather(_prefix(self.best_indices, centers.batch_size), out=centers)
            else:
                centers.integrate(step, out=centers)
        else:
            centers = var
        n = centers.batch_size
        if n not in self._draws_by_width:
            idx = np.repeat(np.arange(n, dtype=np.int32), self.num_draws)
            self._repeat_idx[n] = wp.array(idx, dtype=wp.int32, device=self._device)
            self._draws_by_width[n] = centers.gather(self._repeat_idx[n])
        draws = self._draws_by_width[n]
        centers.gather(self._repeat_idx[n], out=draws)
        noise = _prefix(self._noise_full, draws.batch_size)
        self.fill(noise[:, offset : offset + self.tangent_dim], self._round)
        self._round += 1
        draws.integrate(noise, out=draws)
        if self.lower is not None and self.upper is not None:
            values = cast(VectorVar, draws.get(self.var_key)).x
            inputs = [self.lower, self.upper, self.num_draws]
            wp.launch(_set_bounded_draws_kernel, values.shape, inputs, [values, self.noise], device=self._device)
        self._centers, self.draws = centers, draws
        return draws

    def fill(self, out_tangent: wp.array, iteration: int = 0) -> None:
        wp.launch(
            _gaussian_sample_kernel,
            dim=out_tangent.shape,
            inputs=[
                self.seed,
                self.epoch,
                iteration,
                self.num_draws,
                self.tangent_dim,
                self.active_dof_mask,
                self.std,
                self.num_wide,
                self.wide_scale,
                out_tangent,
            ],
            device=self._device,
        )
        self.noise = out_tangent  # `update` weights the noise it produced

    def _move(self, scores: wp.array, out_delta: wp.array) -> None:
        """Weight the drawn noise by cost, re-fit `std`, and write each center's step into `out_delta`."""
        instances = scores.shape[0] // self.num_draws
        weights, best = _prefix(self.weights, instances), _prefix(self.best_indices, instances)
        block = 1 if self._device.is_cpu else 256  # a CPU tile spans one thread
        if self.elite_ratio > 0:  # the elite's cost spread as temperature, and the cheapest draw
            n, elite = self.num_draws, max(2, int(self.elite_ratio * self.num_draws))
            keys, order = _prefix(self._keys, instances * n), _prefix(self._order, instances * n)
            wp.launch(_compute_order_kernel, instances * n, [scores, n], [keys, order], device=self._device)
            inputs, outputs = [keys, order, n, elite, block], [_prefix(self.temperatures, instances), best]
            wp.launch(_compute_elite_kernel, (instances, block), inputs, outputs, block_dim=block, device=self._device)
            if self.mean_elites:  # the other draws get no weight
                wp.copy(_prefix(self._elite_costs, instances * n), scores)
                scores = _prefix(self._elite_costs, instances * n)
                dropped = instances * (n - elite)
                wp.launch(_set_dropped_cost_kernel, dropped, [order, n, elite], [scores], device=self._device)
        if self.noise_decay != 1.0:
            inputs = [self.noise_decay, self.min_std]
            wp.launch(_set_decayed_std_kernel, self.std.shape, inputs, [self.std], device=self._device)
        if self._weights_kernel is not None:  # ES in best mode needs only the argmin, taken in the update kernel
            wp.launch(
                self._weights_kernel,
                dim=[instances, block],
                inputs=[scores, _prefix(self.temperatures, instances), self.epsilon, block],
                outputs=[weights],
                block_dim=block,
                device=self._device,
            )
        wp.launch(
            self._update_kernel,
            dim=[instances, 1 if self._weights_kernel is None else self.tangent_dim, block],
            inputs=[
                weights,
                self.noise,
                scores,
                best,
                self.gamma,
                self.min_std,
                self.learning_rate,
                self.epsilon,
                block,
            ],
            outputs=[_prefix(self.std, instances), out_delta],
            block_dim=block,
            device=self._device,
        )


__all__ = [
    "Sampler",
    "SamplerConfig",
    "RobertsSamplerConfig",
    "RobertsSampler",
    "GaussianSamplerConfig",
    "GaussianSampler",
    "roberts_samples",
]
