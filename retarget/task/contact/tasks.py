"""Task references and strict acceptance, independent of the search method."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Profile:
    steps: int
    horizon: int
    node_steps: int
    ctrl_steps: int
    samples: int
    iterations: int
    initial_iterations: int = 0
    interpolation: str = 'cubic'
    noise_decay: float = 1.
    coarse_fraction: float = 1.
    coarse_stride: int = 1


PROFILES = {
    'sharpa_pen': Profile(3000, 100, 25, 8, 128, 2, coarse_fraction=.5, coarse_stride=2),
    'sharpa_coin': Profile(2400, 176, 50, 12, 128, 3, noise_decay=.6),
    'h1_jog': Profile(400, 25, 5, 1, 2048, 4, 10, 'quadratic', .5, .25, 2),
}



def __getattr__(name: str):
    if name == 'Task':
        from .robokit_tasks import Task
        return Task
    raise AttributeError(name)
