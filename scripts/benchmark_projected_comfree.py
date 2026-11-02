"""Separate eager launch overhead, CUDA graph setup, and warmed solve latency."""

import argparse
import json
import math
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retarget.env.grasp.projected_comfree import ClosureSettings, ProjectedComFreeClosure


def fixture(batch, contacts):
    if contacts == 2:
        normals = torch.tensor([[1., 0., 0.], [-1., 0., 0.]], device="cuda")
    elif contacts == 6:
        normals = torch.cat((torch.eye(3, device="cuda"), -torch.eye(3, device="cuda")))
    else:
        i = torch.arange(contacts, device="cuda")
        z = 1. - 2. * (i + .5) / contacts
        angle = i * (math.pi * (3. - math.sqrt(5.)))
        radius = (1. - z.square()).sqrt()
        normals = torch.stack((radius * angle.cos(), radius * angle.sin(), z), -1)
    normals = normals[None].expand(batch, -1, -1).contiguous()
    return normals * .05, normals


def measure(function, repeats):
    function()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin = time.perf_counter()
    start.record()
    for _ in range(repeats):
        function()
    end.record()
    end.synchronize()
    return dict(wall_ms=(time.perf_counter() - begin) * 1000. / repeats,
                cuda_span_ms=start.elapsed_time(end) / repeats)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--contacts", type=int, nargs="+", default=[2, 6, 16])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--response-mode", choices=["equilibrium", "finite_horizon"], default="equilibrium")
    parser.add_argument("--profile", action="store_true", help="Profile one eager solve's CPU launches; CUDA spans include idle gaps")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The benchmark requires CUDA")
    print(json.dumps(dict(device=torch.cuda.get_device_name(), batch=args.batch, response_mode=args.response_mode)), flush=True)
    for contacts in args.contacts:
        points, normals = fixture(args.batch, contacts)
        closure = ProjectedComFreeClosure(ClosureSettings(response_mode=args.response_mode), center=[0., 0., 0.], radius=.05,
                                         object_mass=.1, object_inertia=[.0001] * 3)
        geometry = closure._geometry(points, normals)
        preloads = points.new_full(points.shape[:2], closure.settings.initial_preload_n)
        timings = dict(
            rollout=measure(lambda: closure._rollout(preloads, geometry), args.repeats),
            jacobian_rollout=measure(lambda: closure._rollout(preloads, geometry, jacobian=True), args.repeats),
            eager_solve=measure(lambda: closure.solve_eager(points, normals), args.repeats),
        )
        torch.cuda.synchronize()
        begin = time.perf_counter()
        captured = closure.solve(points, normals)
        torch.cuda.synchronize()
        timings["first_graph_call_ms"] = (time.perf_counter() - begin) * 1000.
        timings["graph_warmup_ms"] = captured.metadata["cuda_graph_warmup_seconds"] * 1000.
        timings["graph_capture_ms"] = captured.metadata["cuda_graph_capture_seconds"] * 1000.
        timings["graph_solve"] = measure(lambda: closure.solve(points, normals), args.repeats)
        eager = closure.solve_eager(points, normals)
        torch.testing.assert_close(captured.cost, eager.cost)
        torch.testing.assert_close(captured.preloads, eager.preloads)
        timings["speedup"] = timings["eager_solve"]["wall_ms"] / timings["graph_solve"]["wall_ms"]
        print(json.dumps(dict(contacts=contacts, **timings)), flush=True)
        if args.profile and contacts == args.contacts[0]:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
                closure.solve_eager(points, normals)
            print(profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=8), flush=True)


if __name__ == "__main__":
    main()
