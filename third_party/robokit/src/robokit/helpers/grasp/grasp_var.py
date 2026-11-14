# pyright: reportOperatorIssue=false
# pyright: reportArgumentType=false
# pyright: reportCallIssue=false
# pyright: reportGeneralTypeIssues=false
"""Discrete hand-contact selection, with no tangent for gradient updates.

Redraw mechanisms:
- `ContactVar.integrate`: propose contacts alongside pose moves in annealed search.
- `ContactSampler`: redraw contacts periodically during LM and keep the better set, as graspkit does.
"""

import copy
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, cast

import numpy as np
import warp as wp

from robokit.opt.sampler import SamplerConfig
from robokit.opt.var_values import Var, VarKey, VarValues
from robokit.types import ArrayLike
from robokit.utils.warp_utils import gather, masked_copy, seed_rand_states_kernel, wp_device_type


@wp.func
def _draw_points(
    state: wp.uint32, num_points: wp.int32, batch_idx: wp.int32, out_indices: wp.array2d(dtype=wp.int32)
) -> wp.uint32:
    """Fill `out_indices[batch_idx]` with distinct hand points: rejection-sample, then scan linearly if unlucky."""
    for slot in range(out_indices.shape[1]):
        selected = wp.int32(0)
        unique = wp.int32(0)
        for _attempt in range(8):
            selected = wp.randi(state, wp.int32(0), num_points)
            unique = wp.int32(1)
            for prev in range(slot):
                if selected == out_indices[batch_idx, prev]:
                    unique = wp.int32(0)
            if unique == wp.int32(1):
                break
        if unique == wp.int32(0):
            for shift in range(num_points):
                point = (selected + wp.int32(shift)) % num_points
                unique = wp.int32(1)
                for prev in range(slot):
                    if point == out_indices[batch_idx, prev]:
                        unique = wp.int32(0)
                if unique == wp.int32(1):
                    selected = point
                    break
        out_indices[batch_idx, slot] = selected
    return state


@wp.kernel
def _propose_indices_kernel(
    rand_states: wp.array1d(dtype=wp.uint32),
    num_points: wp.int32,
    switch_probability: wp.float32,
    current: wp.array2d(dtype=wp.int32),
    out_indices: wp.array2d(dtype=wp.int32),
    out_rand_states: wp.array1d(dtype=wp.uint32),
):
    """Redraw a batch element's set with probability `switch_probability`, else keep it; `>= 1` forces a redraw."""
    tid = wp.tid()
    state = rand_states[tid]
    if wp.randf(state) < switch_probability:
        state = _draw_points(state, num_points, tid, out_indices)
    else:
        for slot in range(current.shape[1]):
            out_indices[tid, slot] = current[tid, slot]
    rand_states[tid] = state
    out_rand_states[tid] = state


@wp.kernel
def _redraw_second_draw_kernel(
    seed: wp.int32, salt: wp.int32, num_points: wp.int32, out_indices: wp.array2d(dtype=wp.int32)
):
    """Each candidate's second draw, the odd batch element of its pair, gets fresh contacts."""
    batch_idx = 2 * wp.tid() + 1
    _draw_points(wp.rand_init(seed, salt * 1000003 + batch_idx), num_points, batch_idx, out_indices)


@wp.kernel
def _better_draw_kernel(scores: wp.array1d(dtype=wp.float32), best: wp.array1d(dtype=wp.int32)):
    tid = wp.tid()
    best[tid] = 2 * tid
    if scores[2 * tid + 1] < scores[2 * tid]:
        best[tid] = 2 * tid + 1


class ContactVar(Var):
    """Select one hand-point index per contact slot and batch element, with derived `points` and `link_indices`.

    Update mechanisms:
    - `integrate`: ignore velocity (zero tangent); redraw the whole set with `switch_probability`.
    - Metropolis `accept`: keep or revert the proposed set, as for continuous variables.
    - Sampler integration: also redraws because it integrates all of `VarValues`.
    - `ContactSampler`: the sole updater when `switch_probability` is 0.

    Example:
        >>> import numpy as np, warp as wp
        >>> wp.init()
        >>> var = ContactVar(np.zeros((4, 3), np.float32), np.zeros(4, np.int32), batch_size=2, num_slots=2)
        >>> var.indices.numpy().shape, var.points.numpy().shape
        ((2, 2), (2, 2, 3))
        >>> bool((np.unique(var.indices.numpy()[0]).size == 2))
        True
    """

    def __init__(
        self,
        hand_points: np.ndarray,
        hand_link_indices: np.ndarray,
        batch_size: int,
        num_slots: int,
        switch_probability: float = 0.5,
        seed: int = 0,
        device: wp_device_type = "cpu",
    ):
        self.switch_probability = switch_probability
        self.num_hand_points = int(hand_points.shape[0])
        self._device = wp.get_device(device) if isinstance(device, str) else device
        self.hand_points = wp.from_numpy(hand_points.astype(np.float32), dtype=wp.vec3, device=self._device)
        self.hand_link_indices = wp.from_numpy(hand_link_indices.astype(np.int32), dtype=wp.int32, device=self._device)
        self.indices = wp.zeros((batch_size, num_slots), dtype=wp.int32, device=self._device)
        self.points = wp.zeros((batch_size, num_slots), dtype=wp.vec3, device=self._device)
        self.link_indices = wp.zeros((batch_size, num_slots), dtype=wp.int32, device=self._device)
        self.rand_states = wp.zeros((batch_size,), dtype=wp.uint32, device=self._device)
        self.reset(seed)

    def reset(self, seed: int = 0) -> None:
        """Seed the RNG and draw a fresh distinct set for every batch element."""
        wp.launch(seed_rand_states_kernel, dim=self.batch_size, inputs=[seed, self.rand_states], device=self._device)
        self._propose(1.0, self)

    def _propose(self, switch_probability: float, out: "ContactVar") -> None:
        # Copy `out` with its RNG state so accepting a warmup restore also rewinds contact draws.
        wp.launch(
            _propose_indices_kernel,
            dim=self.batch_size,
            inputs=[self.rand_states, self.num_hand_points, wp.float32(switch_probability), self.indices],
            outputs=[out.indices, out.rand_states],
            device=self._device,
        )
        out.lookup()

    def lookup(self) -> None:
        """Refresh `points` and `link_indices` from `indices`."""
        flat = self.indices.reshape((self.indices.size,))
        gather(self.hand_points, flat, self.points.reshape((flat.shape[0],)))
        gather(self.hand_link_indices, flat, self.link_indices.reshape((flat.shape[0],)))

    @property
    def tangent_dim(self) -> int:
        return 0

    @property
    def num_slots(self) -> int:
        return int(self.indices.shape[1])

    @property
    def batch_size(self) -> int:
        return int(self.indices.shape[0])

    @property
    def device(self) -> wp_device_type:
        return self._device

    def clone(self) -> "ContactVar":
        clone = copy.copy(self)  # hand points are shared; per-element state is copied
        for name in ("indices", "points", "link_indices", "rand_states"):
            setattr(clone, name, wp.clone(getattr(self, name)))
        return clone

    def view(self, batch_size: int, offset: int = 0) -> "ContactVar":
        view = copy.copy(self)
        for name in ("indices", "points", "link_indices", "rand_states"):
            setattr(view, name, getattr(self, name)[offset : offset + batch_size])
        return view

    def integrate(
        self,
        velocity: ArrayLike,
        out: Optional["ContactVar"] = None,
        tangent_mask: Optional[wp.array] = None,
        weight_decay: float = 0.0,
        out_velocity: Optional[ArrayLike] = None,
    ) -> "ContactVar":
        """Hold contacts and RNG state for grad-requiring probes; otherwise propose a new set."""
        out = out if out is not None else self.clone()
        if velocity.requires_grad:
            for name in ("indices", "points", "link_indices", "rand_states"):
                wp.copy(getattr(out, name), getattr(self, name))
        else:
            self._propose(self.switch_probability, out)
        return out

    def gather(self, indices: ArrayLike, out: Optional["ContactVar"] = None) -> "ContactVar":
        if out is None:
            out = copy.copy(self)
            out.indices, out.rand_states = gather(self.indices, indices), gather(self.rand_states, indices)
            out.points = wp.zeros(out.indices.shape, dtype=wp.vec3, device=self._device)
            out.link_indices = wp.zeros(out.indices.shape, dtype=wp.int32, device=self._device)
        else:
            gather(self.indices, indices, out.indices)
            gather(self.rand_states, indices, out.rand_states)
        out.lookup()
        return out

    def accept(self, accept_mask: ArrayLike, proposed: "ContactVar") -> "ContactVar":
        masked_copy(accept_mask, proposed.indices, self.indices)
        masked_copy(accept_mask, proposed.rand_states, self.rand_states)
        self.lookup()
        return self


@dataclass(frozen=True)
class ContactSamplerConfig(SamplerConfig):
    """Contact redraw settings; each candidate competes with one redraw."""

    var_key: Enum = VarKey.CONTACTS
    seed: int = 0
    num_draws = 2

    def build(
        self, var: VarValues, num_candidates: int, active_dof_mask: Optional[wp.array] = None
    ) -> "ContactSampler":
        return ContactSampler(self.var_key, self.seed, var.device)


class ContactSampler:
    """Graspkit-style LM contact search with two draws per candidate (`num_draws=2`).

    Each round:
    1. Return the candidate and a copy with redrawn contacts.
    2. Let the optimizer step this doubled population.
    3. Keep the better scored draw per candidate and draw again.
    """

    num_draws = 2

    def __init__(self, var_key: Enum = VarKey.CONTACTS, seed: int = 0, device: wp_device_type = "cpu"):
        self.var_key, self.seed = var_key, seed
        self._device = wp.get_device(device) if isinstance(device, str) else device
        self._draws_by_width: Dict[int, VarValues] = {}
        self._index: Dict[int, tuple] = {}  # per center width: repeat index, better-draw buffer
        self.draws: Optional[VarValues] = None
        self._centers: Optional[VarValues] = None
        self._round = 0

    def reset(self) -> None:
        """Replay the same redraw sequence."""
        self._round = 0

    def sample(self, var: VarValues, scores: Optional[wp.array]) -> VarValues:
        if var is self.draws and scores is not None:  # its draws came back scored: keep the better of each pair
            centers = cast(VarValues, self._centers)
            better = self._index[centers.batch_size][1]
            wp.launch(_better_draw_kernel, dim=centers.batch_size, inputs=[scores, better], device=self._device)
            var.gather(better, out=centers)
        else:
            centers = var
        n = centers.batch_size
        if n not in self._draws_by_width:
            i32 = lambda a: wp.array(np.asarray(a, np.int32), dtype=wp.int32, device=self._device)  # noqa: E731
            self._index[n] = (i32(np.repeat(np.arange(n), 2)), wp.zeros((n,), dtype=wp.int32, device=self._device))
            self._draws_by_width[n] = centers.gather(self._index[n][0])
        repeat, _ = self._index[n]
        draws = self._draws_by_width[n]
        centers.gather(repeat, out=draws)
        contacts = cast(ContactVar, draws.get(self.var_key))
        wp.launch(
            _redraw_second_draw_kernel,
            dim=n,
            inputs=[self.seed, self._round, contacts.num_hand_points, contacts.indices],
            device=self._device,
        )
        contacts.lookup()
        self._round += 1
        self._centers, self.draws = centers, draws
        return draws


__all__ = ["ContactSampler", "ContactVar"]
