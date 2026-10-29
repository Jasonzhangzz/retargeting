#!/usr/bin/env python3
"""Profile only the device DAG replay, using existing saved grasp inputs."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    import warp as wp
    from types import SimpleNamespace
    from retarget.grasp_device import GraspDeviceNode, CudaNodeGraph
    from retarget.grasp_runtime import GraspRuntime
    from retarget.grasp_cem import CardCemNode
    from retarget.grasp import StaticHand, build_scene
    from retarget.grasp_audit import GraspAudit
    import numpy as np
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path)
    parser.add_argument('--nodes', type=int, default=3)
    parser.add_argument('--torch-trace', type=Path,
                        help='Use the PyTorch CUDA profiler and save a Chrome trace')
    args = parser.parse_args()
    nodes = []
    for row in json.loads(args.summary.read_text())[:args.nodes]:
        case = Path(row['case'])
        source = json.loads((case/'result.json').read_text())
        settings = SimpleNamespace(**source['settings'])
        runtime = GraspRuntime(case/f"input_{source['selected']['start']:02d}", settings)
        model, surface, _, _ = build_scene(source['object'], source['scale'], source.get('hand_model','wuji1'))
        with np.load(case/'grasp.npz') as data:
            hand = StaticHand(model, surface, data['hand_body_ids'],
                data.get('initial_hand_local_points', data['hand_local_points']),
                data.get('initial_hand_local_normals', data['hand_local_normals']))
        runtime.audit_context = GraspAudit(surface, settings.init_device), hand, settings
        nodes.append((CardCemNode if runtime.grasp_optimizer == 'cem' else GraspDeviceNode)(runtime))
    dag = CudaNodeGraph(nodes)
    # Instantiate/upload and execute before collecting the warm replay.
    dag.launch()
    wp.synchronize_stream(dag.stream)
    if args.torch_trace:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as profile:
            dag.launch()
            wp.synchronize_stream(dag.stream)
        profile.export_chrome_trace(str(args.torch_trace))
    else:
        torch.cuda.cudart().cudaProfilerStart()
        dag.launch()
        wp.synchronize_stream(dag.stream)
        torch.cuda.cudart().cudaProfilerStop()
    dag.close()


if __name__ == '__main__':
    main()
