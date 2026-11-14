"""Retarget ParaHome with 10's held-grasp pipeline; default to two batches of 32."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh
import warp as wp

from robokit.assets import REVISION
from robokit.assets.datasets import parahome
from robokit.helpers.full_body_retargeting import FullBodyRetargeter, solve_grasp_refine_batch
from robokit.helpers.full_body_retargeting.parahome_utils import (
    ParaHomeInterval,
    find_parahome_intervals,
    load_parahome_clip,
)
from robokit.robo import Robot
from robokit.utils.visualize_utils import RetargetingPlayback, view_retargeting


def _view_results(output_dir: Path, robot: Optional[Robot] = None) -> None:
    files = {str(path.relative_to(output_dir)): path for path in sorted(output_dir.glob("*/*.npz"))}
    if not files:
        print("No saved clips to visualize.")
        return
    if robot is None:
        from robokit.helpers.humanoid_retargeting.presets.g1_inspire_smplh import g1_inspire_smplh

        robot = g1_inspire_smplh.load_robot(load_meshes=True)
    num_dofs = robot.num_actuated_joints
    mesh_dir = parahome.mesh_dir()

    def load_clip(path: Path) -> RetargetingPlayback:
        with np.load(path, allow_pickle=False) as result:
            qpos = result["qpos"]
            sides = [side for side in ("right", "left") if f"{side}_contact_mask" in result.files]
            return RetargetingPlayback(
                qpos=qpos,
                fps=float(result["fps"]),
                object_meshes=[trimesh.load_mesh(mesh_dir / f"{name}.obj") for name in result["object_names"]],
                T_world_object=qpos[:, 7 + num_dofs :].reshape(len(qpos), -1, 7),
                targets_world=np.concatenate([result[f"{side}_targets_world"] for side in sides], axis=1),
                contacts_world=np.concatenate([result[f"{side}_contacts_world"] for side in sides], axis=1),
                contact_mask=np.concatenate([result[f"{side}_contact_mask"] for side in sides], axis=1),
            )

    view_retargeting(robot, files, load_clip)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/parahome"), help="NPZ output directory (default: %(default)s)"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="clips per solve (default: 32)")
    parser.add_argument("--limit", type=int, help="first N indexed clips (default: two batches); zero processes all")
    viewer = parser.add_mutually_exclusive_group()
    viewer.add_argument("--no-viz", action="store_true", help="save results and exit without playback")
    viewer.add_argument("--viz-only", action="store_true", help="browse saved results without solving")
    args = parser.parse_args()
    if args.limit is None:
        args.limit = 2 * args.batch_size
    if args.batch_size < 1 or args.limit < 0:
        parser.error("--batch-size must be positive and --limit must be nonnegative")
    if args.viz_only:
        _view_results(args.output_dir)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    start = time.perf_counter()

    # --- dataset: index the pinned ParaHome annotations once ---
    data_root = parahome.data_root()
    manifest = args.output_dir / "intervals.json"
    if manifest.exists():
        index = json.loads(manifest.read_text())
        if index["asset_revision"] != REVISION:
            parser.error("output directory contains a different asset revision")
        intervals = [ParaHomeInterval(**row) for row in index["intervals"]]
    else:
        sequences = sorted((data_root / "seq").iterdir(), key=lambda path: int(path.name[1:]))
        intervals = []
        for i, sequence in enumerate(sequences):
            intervals.extend(find_parahome_intervals(str(data_root), sequence.name))
            print(f"indexed {i + 1}/{len(sequences)} sequences: {len(intervals)} interactions", flush=True)
        temporary = manifest.with_suffix(".tmp")
        temporary.write_text(json.dumps({"asset_revision": REVISION, "intervals": [asdict(row) for row in intervals]}))
        temporary.replace(manifest)
    if args.limit:
        intervals = intervals[: args.limit]
    pending = [
        row for row in intervals if not (args.output_dir / row.sequence / f"{row.start:06d}_{row.end:06d}.npz").exists()
    ]
    pending.sort(key=lambda row: (row.end - row.start, int(row.sequence[1:]), row.start))
    print(f"{len(intervals)} interactions, {len(pending)} pending, batch size {args.batch_size}", flush=True)
    if not pending:
        if not args.no_viz:
            _view_results(args.output_dir)
        return

    # --- batches: helpers pad to the longest clip, mask temporal tasks, and swap object scenes ---
    retargeter = FullBodyRetargeter(device=device, load_meshes=True)
    frames_solved = 0
    for offset in range(0, len(pending), args.batch_size):
        batch_start = time.perf_counter()
        rows = pending[offset : offset + args.batch_size]
        clips = [load_parahome_clip(str(data_root), row, device=device) for row in rows]
        loaded = time.perf_counter()
        results = solve_grasp_refine_batch(clips, retargeter)
        solved = time.perf_counter()

        # --- outputs: save valid frames and source metadata atomically; reruns skip completed clips ---
        for row, clip, result in zip(rows, clips, results):
            if not np.isfinite(result.qpos).all():
                raise RuntimeError(f"nonfinite result for {row.sequence} {row.start}:{row.end}")
            output = args.output_dir / row.sequence / f"{row.start:06d}_{row.end:06d}.npz"
            output.parent.mkdir(exist_ok=True)
            fields = {
                "qpos": result.qpos,
                "frame_indices": clip.frame_indices,
                "object_names": np.asarray(clip.object_names),
                "sequence": row.sequence,
                "start": row.start,
                "end": row.end,
                "text": row.text,
                "asset_revision": REVISION,
                "fps": clip.fps,
                "human_height": clip.human_height,
                "scale": clip.scale,
            }
            for side in clip.sides:
                fields[f"{side}_targets_world"] = result.targets_world[side]
                fields[f"{side}_contacts_world"] = result.contacts_world[side]
                fields[f"{side}_contact_mask"] = result.contact_masks[side]
            temporary = output.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **fields)
            temporary.replace(output)
        frames_solved += sum(clip.num_frames for clip in clips)
        print(
            f"solved {offset + len(rows)}/{len(pending)} clips, {frames_solved} frames: "
            f"load {loaded - batch_start:.2f}s, solve {solved - loaded:.2f}s, "
            f"save {time.perf_counter() - solved:.2f}s",
            flush=True,
        )
    print(f"finished in {time.perf_counter() - start:.2f}s", flush=True)
    if not args.no_viz:
        _view_results(args.output_dir, retargeter.body_robot)


if __name__ == "__main__":
    main()
