"""Keep cyclic Warp-array destruction outside CUDA graph capture windows."""
from contextlib import contextmanager
import gc


@contextmanager
def capture_guard():
    # Old scene/graph cycles may become unreachable during preparation. Python
    # can otherwise run their Warp CUDA deallocators in an unrelated capture,
    # invalidating that capture. Explicit per-scene/group collection stays at
    # the existing boundary; reference-counted local temporaries are unaffected.
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()
