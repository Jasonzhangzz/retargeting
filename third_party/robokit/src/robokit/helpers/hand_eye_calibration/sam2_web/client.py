# pyright: reportMissingImports=false
"""Request browser-reviewed binary masks from the SAM2 annotation service."""

import json
from typing import Optional, Sequence, Tuple

import numpy as np


def segment(
    image: np.ndarray,
    address: str = "tcp://localhost:5555",
    timeout_s: float = 3600.0,
    label: str = "",
    box: Optional[Sequence[float]] = None,
    points: Optional[Sequence[Tuple[float, float, int]]] = None,
) -> np.ndarray:
    """Request a binary mask from a running SAM2 annotation server.

    Args:
        label: Browser label identifying the frame and requested object.
        box: `(x1, y1, x2, y2)` prompt SAM2 starts from, so a mask is already there to correct.
        points: `(x, y, positive)` prompts, alongside or instead of `box`.

    Wire protocol (two-frame ZMQ multipart, both directions):
        - Frame 0: JSON header `{"dtype": str, "shape": [...], "label": str, "box": [...], "points": [...]}`.
        - Frame 1: Raw `ndarray.tobytes()` payload.

    Example (requires a running server and a human in the browser)::

        mask = segment(image, label="ARM 3/20", box=(120, 80, 400, 360))
    """
    import zmq

    assert image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3
    image = np.ascontiguousarray(image)
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000))
    sock.connect(address)
    header = json.dumps(
        {
            "dtype": "uint8",
            "shape": list(image.shape),
            "label": label,
            "box": list(box) if box is not None else None,
            "points": [list(p) for p in points] if points is not None else None,
        }
    ).encode()
    sock.send_multipart([header, image.tobytes()])
    reply_header, reply_body = sock.recv_multipart()
    sock.close()
    meta = json.loads(reply_header.decode())
    mask = np.frombuffer(reply_body, dtype=np.dtype(meta["dtype"])).reshape(meta["shape"])
    return np.ascontiguousarray(mask)
