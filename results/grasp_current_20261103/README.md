# Final grasp poses and dense interpolation

31 grids · 640 solved poses · 39,007 interpolated poses · approximately 6 MB.

Only **one shared scene XML** is stored. Primitive geometry is shared by shape;
each grid contains pose and size arrays, without per-size XML/OBJ files.
All pose and dimension values retain float64 precision. Dense linear contact
projections are reconstructed exactly from their source knots.

```bash
python scripts/verify_grasp_cache.py results/grasp_current_20261103
python visualize.py --port 8089
```

The viewer creates its local output link automatically. It reads the packed data
directly, without extracting temporary scenes. Slider updates reuse scene nodes;
no meshes are rebuilt and the camera stays fixed during size changes.

`poses.npz` stores original `qpos`, `grasp_qpos`, `case_names`, `radius_m`,
`height_m`, `dimensions_m`, `size_scale`, source `projected_points` and counts.
`dense/*.npz` stores dense pose/size arrays and exact source knots.
The first 3 q coordinates are root translation in metres, the next 3 are
intrinsic XYZ root angles, and the final 20 are finger joint angles in radians.

Final closure: 612/640. Interpolated poses are previews and have not been solved
or physically evaluated. Original failed cases are retained. Shared geometry
reconstruction error against the expanded source is at most 1.31e-8 metres.

See [storage and validation](../../docs/grasp_pose_cache.md), `source_verification.json`,
`verification.json`, `preservation.json.gz` and `checksums.json` for provenance.
Initialization chains, retries, optimization histories and repeated patch clouds
are intentionally excluded from this final pose cache.
