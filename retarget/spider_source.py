"""Access the pinned SPIDER implementation without changing its source files."""
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace

SOURCE = Path(__file__).resolve().parents[1]/'data/spider/source'
COMMIT = '44717007de41cbef7565dff7ff9f4453557a2d3d'
# From this commit's pyproject.toml and uv.lock, verified against GitHub.
UPSTREAM_ENGINE_VERSIONS = {'mujoco':'3.7.0', 'mujoco-warp':'3.7.0.1', 'warp-lang':'1.12.1'}


def engine_version_audit():
    installed = {name:version(name) for name in UPSTREAM_ENGINE_VERSIONS}
    return dict(upstream=UPSTREAM_ENGINE_VERSIONS.copy(), installed=installed,
                match=installed == UPSTREAM_ENGINE_VERSIONS)


@lru_cache(None)
def modules():
    if str(SOURCE) not in sys.path:
        sys.path.insert(0, str(SOURCE))
    import warp as wp
    from spider import config, interp, io
    from spider.simulators import mjwp
    from spider.optimizers import sampling

    class ArrayCopyAPI:
        def __getattr__(self, name):
            return getattr(wp, name)

        def copy(self, destination, source, *args, **kwargs):
            if destination.dtype != source.dtype:
                source = wp.from_torch(wp.to_torch(source), dtype=destination.dtype)
            return wp.copy(destination, source, *args, **kwargs)

    # Old SPIDER copies scalar torch views into compound Warp arrays. Rebind
    # only its private wp.copy dispatch for current Warp dtype checking.
    scope = dict(vars(mjwp)); scope['wp'] = ArrayCopyAPI()
    for name, function in list(scope.items()):
        if isinstance(function, FunctionType) and function.__module__ == mjwp.__name__:
            clone = FunctionType(function.__code__, scope, function.__name__,
                                 function.__defaults__, function.__closure__)
            clone.__kwdefaults__ = function.__kwdefaults__
            scope[name] = clone
    return SimpleNamespace(config=config, interp=interp, io=io,
                           sampling=sampling, mjwp=SimpleNamespace(**scope))
