"""SimMPC settings; build a planner with `SimMPC(config.add(cost), MjWarpSim(model, 1), ctrl_ref)`.

Predictive sampling: https://arxiv.org/abs/2212.00541. CEM: https://doi.org/10.1023/A:1010091220143.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import warp as wp

from robokit.terms.task import Task


if TYPE_CHECKING:
    from robokit.sim import Sim


@dataclass(frozen=True)
class SimMPCStage:
    """A run of each plan's iterations whose rollouts step at `dt` seconds (None: the simulator's timestep), or along a schedule of `(fraction of the horizon it runs to, dt)`, e.g. `[(0.25, None), (1.0, 0.04)]`; each dt a whole multiple of the timestep, each step's costs weighted by the timesteps it spans."""

    iters: int
    dt: Union[None, float, Sequence[Tuple[float, Optional[float]]]] = None


@dataclass(frozen=True)
class SimMPCConfig:
    """SimMPC's per-task settings; times are in sim steps.

    Example:
        >>> config = SimMPCConfig(num_samples=64, horizon_steps=30)
        >>> config.num_samples, config.node_steps
        (64, 50)
        >>> SimMPCConfig(stages=[SimMPCStage(3, dt=0.04), SimMPCStage(1)]).max_iterations  # 3 coarse iterations, 1 fine
        4
    """

    terms: List[Tuple[Task, float, Optional[float]]] = field(default_factory=list)
    """The cost terms with their running and terminal scales, added with `add`."""
    ctrl_std: Any = 0.1
    """Node noise per command at the first free node, one value or one per command."""
    ctrl_range: Optional[np.ndarray] = None
    """`(nu, 2)` bounds of the commands; None: the actuator ranges MuJoCo clamps to."""
    qpos_std: Any = 0.0
    """Each replica's start noise on `qpos`, one value or one per coordinate."""
    qvel_std: Any = 0.0
    """Each replica's start noise on `qvel`, one value or one per dof."""
    control_map: Optional[Callable[["Sim", wp.array], None]] = None
    """`control_map(sim, clock)` turns the commands in `sim.data.ctrl` into actuator controls before every step."""
    batch_size: int = 1
    """Instances planned at once; they share the model and the config."""
    num_samples: int = 512
    """Draws per iteration, one of them reserved to score the moved plan. A sim step costs about the same up to some hundreds of worlds (about 512 on an RTX 3090) and grows past them, so fill that many before adding iterations."""
    num_replicas: int = 1
    """Worlds per draw, which average their costs (e.g. over `qpos_std` noise)."""
    horizon_steps: int = 300
    """Steps each rollout simulates and scores."""
    node_steps: int = 50
    """Steps between nodes, the points the control offsets are interpolated through."""
    ctrl_steps: int = 100
    """Steps executed per plan; a multiple of `node_steps` keeps shifted nodes exact."""
    max_iterations: int = 12
    """Sampling iterations per plan, at most; with `stages`, the sum of their iters."""
    init_iterations: int = 0
    """If positive, the first plan's iterations instead of `max_iterations`: the stages' last ones, the first stage run longer or cut."""
    stages: Optional[List[SimMPCStage]] = None
    """Coarse to fine: the stages each plan's iterations run in order; the moved plans are scored at the simulator's timestep. None: the first rollouts pick `[SimMPCStage(max_iterations, [(0.25, None), (1.0, k * timestep)])]` for the largest k of 4 and 2 whose draws end within 11% of their spread of the fine ones, else one fine stage."""
    min_improvement: float = 0.005
    """Stop once the best draw beats the plan by under this, `patience` times running."""
    patience: int = 2
    """Iterations under `min_improvement` in a row that stop the plan."""
    std_growth: float = 4.0
    """Node noise grows from `ctrl_std` at the first free node to this multiple at the last."""
    noise_decay: float = 1.0
    """Iteration `i` scales the node noise by `noise_decay**i`."""
    interpolation: Union[Literal["linear", "cubic", "zero"], np.ndarray] = "linear"
    """Node interpolation, or a `(steps, num_nodes)` basis of each step's node weights; "zero" also draws the first node."""
    draw_first_node: bool = False
    """Draw the first node for any interpolation, so a plan can change the command it executes now (predictive sampling, CEM)."""
    wide_ratio: float = 0.0
    """Share of each plan's first-round draws taken at `wide_scale` times the noise, as predictive sampling explores."""
    wide_scale: float = 6.0
    std_range: Optional[Tuple[float, float]] = None
    """If set, a plan of one iteration starts from the last plan's refit noise, as CEM carries its spread: one scale of the configured noise (root mean square over the drawn nodes), clipped to this range."""
    min_std: float = 1e-3
    """Floor of the node noise."""
    std_gamma: float = 0.5
    """Share of the draws' weighted spread each iteration moves the node noise toward."""
    elite_ratio: float = 0.2
    """Share of the cheapest draws whose cost spread is the temperature and whose weighted mean is the moved plan."""
    best_draw_margin: float = 4.0
    """Execute the best draw instead of the moved plan when the moved plan costs this many temperatures more."""
    settle_steps: int = 0
    """Unscored zero-control steps before each rollout."""
    terminal_scale: float = 10.0
    """Cost scale at the horizon's last step, unless `add` sets a term's own."""
    seed: int = 0
    """Seed of the node draws and the replicas' initial-state noise."""
    cuda_graph_mode: Literal["none", "step"] = "step"
    """"step" replays one captured graph per sim step; "none" launches every kernel."""
    planner: Literal["mppi", "ilqr"] = "mppi"
    """"ilqr": `ILQROptimizer` refines commands held for `node_steps`, `max_iterations` times a plan, instead of nodes."""
    ilqr: Optional[Any] = None
    """"ilqr"'s `ILQROptimizerConfig`, but `max_iter`; None: damped (robot costs seldom penalize controls) and cautious."""

    def __post_init__(self):
        if self.stages is not None:
            object.__setattr__(self, "max_iterations", sum(stage.iters for stage in self.stages))

    def add(self, term: Task, running_scale: float = 1.0, terminal_scale: Optional[float] = None) -> "SimMPCConfig":
        """Add a cost term, scaled by `running_scale` per step and by `terminal_scale` (default: the config's) at the horizon's last step; `running_scale=0` makes a terminal-only cost."""
        self.terms.append((term, running_scale, terminal_scale))
        return self
