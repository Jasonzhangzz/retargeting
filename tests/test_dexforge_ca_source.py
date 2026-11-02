"""The ContactAware reference remains byte-for-byte pinned upstream source."""

import hashlib
import json
from pathlib import Path


def test_contactaware_source_matches_pinned_manifest():
    root = Path(__file__).resolve().parents[1] / "third_party/DexForge"
    manifest = json.loads((root / "CONTACTAWARE_UPSTREAM.json").read_text())
    assert manifest["commit"] == "fa6d671948ef03d18514062748b94e445b63cd49"
    assert "contactaware/contactaware/keys/key_sqp.py" in manifest["sha256"]
    assert "contactaware/contactaware/keys/surface_retraction.py" in manifest["sha256"]
    assert "contactaware/third_party/manopth/LICENSE" in manifest["sha256"]
    for relative, expected in manifest["sha256"].items():
        assert relative.startswith("contactaware/")
        assert Path(relative).suffix not in (".pkl", ".npy", ".npz", ".mp4")
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected
