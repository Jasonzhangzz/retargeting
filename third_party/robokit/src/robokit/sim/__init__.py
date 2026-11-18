"""Batched simulation: `nworld` worlds of one model or of several scenes stepped together, fields as Warp arrays."""

from typing import TYPE_CHECKING, Any, Optional, Protocol

import warp as wp

from robokit.sim.dynamics import QposVar, SimDynamics, SimKey


if TYPE_CHECKING:
    from robokit.sim.mjwarp import CoarseConfig, MjWarpSim


def __getattr__(name: str):
    if name in ("CoarseConfig", "MjWarpSim"):  # imported on first use: MuJoCo Warp needs Python >= 3.10
        from robokit.sim import mjwarp

        return getattr(mjwarp, name)
    raise AttributeError(f"module 'robokit.sim' has no attribute {name!r}")


class Sim(Protocol):
    """What `MjWarpSim` and `newton.NewtonSim` offer: device work only, also in a capture on Warp's current stream."""

    mjm: Any
    """The `mujoco.MjModel` whose layout `data` follows: bodies, joints, sensors (for several scenes, their merge)."""
    nworld: int
    substeps: int
    stream: Optional[wp.Stream]
    """The CUDA stream it launches on (None on the CPU); other GPU work on it stays ordered without host syncs."""
    # at least time, qpos, qvel, act, qacc_warmstart, ctrl, sensordata, xpos, xmat, nacon, contact.worldid/geom/dist/pos
    data: Any
    """Warp arrays named and laid out as in `mujoco.MjData`, with a leading world axis `(nworld, ...)`."""
    model: Any  # a per-world array assigned to a field stays
    """`mujoco.MjModel`'s fields (`NewtonSim`: none), a row per world copied from the shared row on first access."""

    def step(self):
        """Advance `substeps` physics steps; `data.substep_sensordata[k]`: `sensordata` after substep `k`."""

    def forward(self):
        """Recompute the derived quantities (`xpos`, `sensordata`, ...) from the current state."""

    def reset(self, mask: wp.array):
        """Return the worlds where `mask` is set to the initial state."""

    def resized(self, nworld: int) -> "Sim":
        """A new simulator of the same scenes and settings with `nworld` worlds."""
        ...


__all__ = ["CoarseConfig", "MjWarpSim", "QposVar", "Sim", "SimDynamics", "SimKey"]
