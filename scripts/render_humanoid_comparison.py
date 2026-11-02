#!/usr/bin/env python3
"""Render synchronized reference, native FA and independent MuJoCo trajectories.

The camera stays fixed in world coordinates and is identical for all panels.
Only saved states are rendered; this script never simulates or repairs a pose.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def _load(batch):
    path = batch / "trajectory.npz"
    with np.load(path, allow_pickle=False) as saved:
        times = np.asarray(saved["time"], np.float64).copy()
        actual = np.asarray(saved["qpos"], np.float64).copy()
        reference = np.asarray(saved["qpos_ref"], np.float64).copy()
    if times.ndim != 1 or len(times) != len(actual) or reference.shape != actual.shape:
        raise ValueError("Saved actual/reference states must share one timestamp per row")
    if len(times) < 1 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Trajectory timestamps must be finite and strictly increasing")
    panels = [("REFERENCE", reference, (172, 196, 222)),
              ("FA NATIVE", actual, (91, 220, 211))]
    independent = batch / "independent_mujoco/trajectory.npz"
    if independent.is_file():
        with np.load(independent, allow_pickle=False) as saved:
            other = np.asarray(saved["qpos"], np.float64).copy()
            other_times = np.asarray(saved["time"], np.float64).copy()
        if other.shape != actual.shape or not np.allclose(other_times, times, atol=1e-8, rtol=0):
            raise ValueError("Independent replay does not match the native saved time grid")
        panels.append(("INDEPENDENT MUJOCO", other, (248, 192, 104)))
    if not all(np.isfinite(qpos).all() for _, qpos, _ in panels):
        raise ValueError("Cannot render nonfinite poses")
    return times, panels


def _fixed_camera(reference):
    camera = mujoco.MjvCamera()
    # Root and object reference positions determine a shared world-space view.
    # Include the floor so a fall stays visible rather than following the torso.
    positions = np.concatenate((reference[:, :3], reference[:, -7:-4]), axis=0)
    lower, upper = positions.min(0), positions.max(0)
    camera.lookat[:] = [*(0.5*(lower[:2]+upper[:2])), 0.5]
    camera.distance = max(2.9, float(np.linalg.norm(upper[:2]-lower[:2]))+2.1)
    camera.azimuth = 140.0
    camera.elevation = -22.0
    return camera


def render_comparison(batch, *, output=None, width=480, height=360, fps=30., speed=.5,
                      snapshots=6):
    batch = Path(batch).expanduser().resolve()
    output = batch if output is None else Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if width < 240 or height < 180 or width % 2 or height % 2:
        raise ValueError("Panel dimensions must be even and at least 240x180")
    if not np.isfinite(fps) or fps <= 0 or not np.isfinite(speed) or speed <= 0 or snapshots < 1:
        raise ValueError("FPS, playback speed and snapshot count must be positive")
    times, panels = _load(batch)
    model_path = batch / "native_scene/source_scene.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    model.vis.global_.offwidth = max(width, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(height, model.vis.global_.offheight)
    data = mujoco.MjData(model)
    camera = _fixed_camera(panels[0][1])
    options = mujoco.MjvOption()
    # Sites and physical proxies are observations, not an extra visual robot.
    options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    header, footer = 56, 34
    canvas_shape = (width*len(panels), height+header+footer)
    heading, small = _font(18), _font(14)
    duration = float(times[-1]-times[0])
    summary_path = batch / "summary.json"
    complete = False
    if summary_path.is_file():
        complete = bool(json.loads(summary_path.read_text()).get("full_clip_completed", False))
    status = "FULL CLIP" if complete else "DIAGNOSTIC PREFIX"
    video_path = output / "video_comparison.mp4"
    sheet_path = output / "comparison.png"
    preview_path = output / "comparison_preview.png"
    final_path = output / "comparison_final.png"

    with mujoco.Renderer(model, height=height, width=width) as renderer:
        def frame(index):
            canvas = Image.new("RGB", canvas_shape, (18, 23, 31))
            draw = ImageDraw.Draw(canvas)
            for column, (label, qpos, color) in enumerate(panels):
                data.qpos[:] = qpos[index]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera, scene_option=options)
                canvas.paste(Image.fromarray(renderer.render()), (column*width, header))
                left = column*width+12
                draw.text((left, 7), label, fill=color, font=heading)
                draw.text((left, 32), f"t = {times[index]:.2f} s", fill=(221, 226, 235), font=small)
                if column:
                    error = np.linalg.norm(qpos[index, -7:-4]-panels[0][1][index, -7:-4])
                    draw.text((left, header+height+8), f"box error: {error:.3f} m", fill=(221, 226, 235), font=small)
                else:
                    draw.text((left, header+height+8), f"{status}  |  {speed:g}x playback", fill=(221, 226, 235), font=small)
                if column:
                    draw.line((column*width, 0, column*width, canvas.height), fill=(64, 73, 90), width=2)
            return canvas

        # Use nearest saved state on a regular display time grid. Always include
        # the final actual state; never extend simulation to fill a requested clip.
        count = max(1, int(np.ceil(duration*fps/speed)))
        displayed = np.linspace(times[0], times[-1], count+1)
        right = np.searchsorted(times, displayed).clip(0, len(times)-1)
        left = np.maximum(right-1, 0)
        indices = np.where(np.abs(times[left]-displayed) <= np.abs(times[right]-displayed), left, right)
        with imageio.get_writer(video_path, fps=fps, codec="libx264", quality=8,
                                macro_block_size=2, ffmpeg_log_level="error") as writer:
            for index in indices:
                writer.append_data(np.asarray(frame(int(index))))
        chosen = np.unique(np.rint(np.linspace(0, len(times)-1, min(snapshots, len(times)))).astype(int))
        sheet = Image.new("RGB", (canvas_shape[0], canvas_shape[1]*len(chosen)), (18, 23, 31))
        for row, index in enumerate(chosen):
            sheet.paste(frame(int(index)), (0, row*canvas_shape[1]))
        sheet.save(sheet_path)
        frame(int(chosen[min(len(chosen)-1, len(chosen)//2)])).save(preview_path)
        frame(len(times)-1).save(final_path)
    metadata = dict(video=str(video_path), comparison=str(sheet_path), preview=str(preview_path), final=str(final_path),
                    duration_seconds=duration, saved_state_count=len(times), playback_speed=speed,
                    fps=fps, video_frame_count=len(indices), camera=dict(lookat=camera.lookat.tolist(),
                    distance=camera.distance, azimuth=camera.azimuth, elevation=camera.elevation),
                    panels=[name for name, _, _ in panels], full_clip_completed=complete,
                    rendered_saved_states_only=True, snapshot_times=times[chosen].tolist())
    (output / "comparison_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=float, default=30.)
    parser.add_argument("--speed", type=float, default=.5)
    parser.add_argument("--snapshots", type=int, default=6)
    args = parser.parse_args()
    result = render_comparison(args.batch, output=args.output, width=args.width, height=args.height,
                               fps=args.fps, speed=args.speed, snapshots=args.snapshots)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
