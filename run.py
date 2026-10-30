"""Unified task entry point: python run.py --config task.yaml."""

import argparse
import importlib
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
from retarget.config.run import load, ROOT
from retarget.common.results import prepare_output, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "task.yaml")
    args = parser.parse_args()
    config = load(args.config)
    if config.mode in ("web", "replay"):
        from retarget.viewer.runtime import run

        return run(config)
    module = importlib.import_module(f"retarget.env.{config.task['type']}.runner")
    module.validate(config)
    output = prepare_output(config)
    started = time.perf_counter()
    try:
        cases = module.run(config, output)
        from retarget.common.provenance import record

        write_json(output / "provenance.json", record(config, cases, output))
        write_json(
            output / "manifest.json",
            dict(
                version=1,
                task=config.task["type"],
                method=config.solver.type,
                cases=cases,
            ),
        )
        if config.visualization.get("save_video", True):
            from retarget.viewer.runtime import video

            for case in cases:
                video(output, case)
        write_json(
            output / "status.json",
            dict(state="completed", wall_seconds=time.perf_counter() - started),
        )
        if config.visualization.get("live", False):
            from retarget.viewer.runtime import replay

            replay(config)
    except BaseException as exc:
        write_json(
            output / "status.json",
            dict(
                state="failed",
                error=f"{type(exc).__name__}: {exc}",
                wall_seconds=time.perf_counter() - started,
            ),
        )
        if hasattr(exc, "history"):
            write_json(output / "failure_iterations.json", exc.history)
        raise
    print(output, flush=True)
    return output


if __name__ == "__main__":
    main()
