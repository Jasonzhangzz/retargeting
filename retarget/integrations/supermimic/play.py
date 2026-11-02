"""Live Viser playback or finite headless replay of a shared grasp policy."""
import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import torch

from .dataset import GraspDataset
from .environment import GraspHoldEnv, make_env_cfg
from .train import FEATURE_SCHEMA, FORMAT


@torch.no_grad()
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest")
    p.add_argument("--case-id")
    p.add_argument("--device", default="cuda:3")
    p.add_argument("--steps", type=int, default=0)
    p.add_argument("--viser", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--port", type=int, default=8082)
    p.add_argument("--output", type=Path)
    args = p.parse_args(argv)
    if args.steps < 0 or (args.steps == 0 and not args.viser):
        p.error("Headless playback needs positive --steps")
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    manifest = args.manifest or checkpoint["config"]["manifest"]
    ds = GraspDataset(manifest)
    if checkpoint["format_version"] != FORMAT or checkpoint["feature_schema"] != FEATURE_SCHEMA or checkpoint["dataset_signature"] != ds.signature:
        raise ValueError("Checkpoint/cache mismatch")
    case = ds.ids[args.case_id] if args.case_id else 0
    env = GraspHoldEnv(make_env_cfg(1, manifest=manifest, case_ids=[case]), args.device)
    env.force_scale = 1.
    env.reset(split="test")
    from dex_supermimic.rl.dex_supermimic.networks import Actor
    actor = Actor(checkpoint["obs_dim"]).to(args.device)
    actor.load_state_dict(checkpoint["modules"]["actor"])
    actor.eval()
    server = viewer = None
    if args.viser:
        import viser
        from mjviser import ViserMujocoScene
        server = viser.ViserServer(host="0.0.0.0", port=args.port)
        viewer = ViserMujocoScene(server, env.sim.mj_model, 1)
        print(f"Grasp viewer: http://127.0.0.1:{args.port} (forward this port in VS Code)", flush=True)
    cpu_data = mujoco.MjData(env.sim.mj_model)
    states, commands = [], []
    i = 0
    try:
        while args.steps == 0 or i < args.steps:
            started = time.monotonic()
            result = env.step(actor(env.observe()), training=False)
            if args.output:
                states.append(env.qpos[0].cpu().numpy().copy())
                commands.append(result["u"][0].cpu().numpy())
            if viewer is not None:
                cpu_data.qpos[:] = env.qpos[0].cpu().numpy()
                mujoco.mj_kinematics(env.sim.mj_model, cpu_data)
                viewer.update_from_mjdata(cpu_data)
                time.sleep(max(0., .02 - (time.monotonic() - started)))
            i += 1
            if result["done"][0]:
                env.reset(split="test")
        print(json.dumps(dict(task="hand2-grasp-hold", case_id=ds.cases[case]["case_id"], steps=i,
                              finite=bool(result["valid"][0]), strict_success_evaluated=False)))
    finally:
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.output, qpos=states, ctrl=commands, dt=.02)
        if server is not None:
            server.stop()
        env.close()


if __name__ == "__main__":
    main()
