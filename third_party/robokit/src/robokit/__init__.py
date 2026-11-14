import logging
import os
import tempfile

import rich
from rich.logging import RichHandler


# --- CUDA: coredump mode avoids NVIDIA's multi-process, multi-stream CUDA graph fault (TensorRT#3633) -------
os.environ.setdefault("CUDA_ENABLE_COREDUMP_ON_EXCEPTION", "1")
os.environ.setdefault("CUDA_ENABLE_LIGHTWEIGHT_COREDUMP", "1")
os.environ.setdefault("CUDA_COREDUMP_FILE", os.path.join(tempfile.gettempdir(), "robokit-cuda-%h-%p.nvcudmp"))

# --- logging ---------------------------------------------------------------
rich.reconfigure(log_path=False)

logger = logging.getLogger("robokit")
logger.propagate = False
logger.setLevel("INFO")
if not logger.handlers:
    logger.addHandler(RichHandler(console=rich.get_console(), show_path=False, log_time_format="[%X]"))


__version__ = "1.0.0"

__all__ = ["__version__"]
