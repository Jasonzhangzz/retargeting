# WUJI Hand 2.2 right hand for grasp synthesis

Source: [official wuji-description](https://github.com/wuji-technology/wuji-description/tree/5c4dfa16917355d073e2e1d7e3d03897b6cf9508/hand2/hand2_2/body),
MIT licensed. `source.json` pins the commit, source file hashes and interface
aliases. `official/right.xml` is the unmodified upstream MJCF. All 27 referenced
meshes are included, including the five fixed fingertip sensor pads and mount.

`right.xml` is the local grasp adapter. It uses the existing six slide/XYZ wrist
coordinates and canonical `right_fingerN_linkK` names, retaining the official
20 finger joint transforms, limits, inertias and position actuators. The wrist
frame is the official `r_wrist`; the mount is kept at its exact relative pose.
The local native ComFree executor requires Euler integration. The numeric XML
marker `grasp_hand_generation=2` identifies archived scenes independently of
the current CLI default.

`qp_right/` contains 8880 deterministic, area-weighted samples on the actual
body meshes, including rigidly attached sensor pads and mount. Normals are
mesh-local surface normals. Sampling seed: 20261029. Distances use metres.
Contact selection resolves the second-generation MCP abduction and palmar
normal conventions separately from first-generation geometry.

Rebuild derived assets offline using the existing vendored source:

```bash
python scripts/build_wuji2_assets.py
```

Add `--download` only when initially fetching the pinned official source.
`run_grasp.py` defaults to `--hand-model wuji2`; `--hand-model wuji1` selects
the original assets. Archived first-generation scenes remain first-generation
models during replay and refinement.
