#!/usr/bin/env python3
"""Build the generic Moxian Studio-style source solver (g++ and Eigen3)."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from moxian_web.studio_emf_solver import native_library

if __name__=='__main__':
    _,checksum,path=native_library()
    print(json.dumps(dict(library=str(path),source_sha256=checksum,
        implementation='compiled_studio_style_bounded_qp',official_sdk_binary=False)))
