#!/usr/bin/env python3
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MUJOCO_GL', 'egl')
import argparse
import fcntl
import signal
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import queue
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from retarget.viewer.mano_results import ManoSnapshot, read, write


def worker(number, device, tasks, options):
    import collections
    import mujoco
    import torch
    import trimesh
    import warp as wp
    from retarget.env.grasp.pose_gpu import WujiPoseAdapter,matrix_to_euler_xyz,euler_xyz_to_matrix
    from retarget.initialization.dexonomy.util.torch_util import torch_quaternion_to_matrix
    from retarget.env.grasp.pose_objective import TemplatePoseObjective, PoseObjectiveSettings
    from retarget.env.grasp.pose_inputs import TemplateContacts
    from retarget.env.grasp.graspadmm_scene import required_body_groups
    from retarget.solver.base import Problem, Settings, solve, NoValidCandidate
    torch.set_num_threads(1)
    torch.cuda.set_device(device)
    wp.init()
    output = Path(options['output'])
    mapping = read(options['mapping'])['mappings']
    cache = collections.OrderedDict()
    settings = Settings(type='cem', population=options['population'], worlds=options['population'])
    count = 0
    failures = 0
    started = time.monotonic()
    while not (output/'STOP').exists():
        try:
            group = tasks.get(timeout=2)
        except queue.Empty:
            continue
        if group is None:
            break
        source = Path(group['path'])
        typ = group['type']
        dest = output/typ/'shards'/f"{group['part']:07d}"
        dest.mkdir(parents=True, exist_ok=True)
        try:
            checksums = read(source/'checksums.json')
            for name in ('poses.npz', 'geometry.npz'):
                if hashlib.sha256((source/name).read_bytes()).hexdigest() != checksums[name]:
                    raise ValueError('Input checksum mismatch: '+str(source/name))
            snapshot = ManoSnapshot(source)
            records = mapping[typ]
            names = tuple(r['wuji_body'] for r in records)
            local = np.array([r['wuji_local'] for r in records])
            normals = np.array([r['wuji_local_normal'] for r in records])
            total = len(snapshot.poses['mano_joints'])
            if options['max_rows']:
                total = min(total, options['max_rows'])
            completed = []
            stored = {}
            if (dest/'progress.json').exists():
                completed = read(dest/'progress.json')['rows']
                with np.load(dest/'poses.npz', allow_pickle=False) as archive:
                    stored = {k: list(archive[k]) for k in archive.files}
            for row in range(len(completed), total):
                if (output/'STOP').exists():
                    break
                begun = time.monotonic()
                pose = torch.as_tensor(snapshot.poses['object_pose_world'][row],device=device,dtype=torch.float64)
                rotation = torch_quaternion_to_matrix(pose[3:])
                geometry_key='object_'+str(int(snapshot.poses['object_geometry_ids'][row]))
                source_vertices=snapshot.geometry[geometry_key+'_vertices']
                faces=snapshot.geometry[geometry_key+'_faces']
                scale=torch.as_tensor(snapshot.poses['scene_scale'][row],device=device,dtype=torch.float64)
                vertices=(torch.as_tensor(source_vertices,device=device,dtype=torch.float64)*scale).cpu().numpy()
                identity = hashlib.sha256(vertices.tobytes()+faces.tobytes()).hexdigest()
                if identity not in cache:
                    mesh_path = output/'meshes'/f'{identity}.obj'
                    mesh_path.parent.mkdir(exist_ok=True)
                    if not mesh_path.exists():
                        temp = mesh_path.with_name(f'{identity}.{number}.obj')
                        trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(temp)
                        temp.replace(mesh_path)
                    objective = TemplatePoseObjective(mesh_path, PoseObjectiveSettings(), device=device)
                    objective.set_batch_size(options['population'])
                    objective.pose_adapter=WujiPoseAdapter(objective.cpu,device)
                    cache[identity] = objective
                    if len(cache) > 2:
                        cache.popitem(last=False)
                objective = cache[identity]
                cache.move_to_end(identity)
                joints = (torch.as_tensor(snapshot.poses['mano_joints'][row],device=device,dtype=torch.float64)-pose[:3])@rotation
                initial = objective.pose_adapter.initialize(joints)[0].to(torch.float32).clamp(objective.low,objective.high)
                template = TemplateContacts(names, local, normals, initial[6:],
                    tuple(map(tuple, required_body_groups(names))), dict(source=str(source), row=row))
                objective.bind(template, initial)
                mean = torch.as_tensor(initial, device=device, dtype=torch.float32)
                sigma = mean.new_tensor([objective.settings.pos_noise]*3+
                    [objective.settings.rot_noise]*3+[objective.settings.joint_noise]*20)
                problem = Problem(mean, sigma, lambda values: objective.evaluate(values[:, None]), objective.project)
                no_valid = False
                try:
                    result = solve(problem, settings, seed=7+row)
                    final = result.parameters
                except NoValidCandidate:
                    final = initial
                    no_valid = True
                with torch.no_grad():
                    report = objective.diagnostics(final)
                final=final.to(torch.float64)
                final_world=final.clone()
                final_world[:3]=rotation@final[:3]+pose[:3]
                final_world[3:6]=matrix_to_euler_xyz(rotation@euler_xyz_to_matrix(final[3:6]))
                keypoints=objective.pose_adapter.keypoints(final)
                parts=objective.parts
                values=dict(qpos=final_world,keypoints=keypoints@rotation.T+pose[:3],
                    object_pose_world=pose,object_scale=torch.as_tensor(snapshot.poses['object_scale'][row],device=device),
                    scene_scale=torch.as_tensor(snapshot.poses['scene_scale'][row],device=device),
                    hand_contact_points=parts['hand_points'][0].to(torch.float64)@rotation.T+pose[:3],
                    object_contact_points=parts['projected_points'][0].to(torch.float64)@rotation.T+pose[:3],
                    object_contact_normals=parts['object_normals'][0].to(torch.float64)@rotation.T,
                    projected_closure=parts['closure_certified'][0],geometry_success=report['geometry_success'],
                    soft_contact_success=report['soft_contact_success'],no_valid_candidate=no_valid)
                if not all(bool(torch.isfinite(value).all()) for value in values.values() if isinstance(value,torch.Tensor)):
                    raise ValueError('Nonfinite final output')
                position,frame=objective.pose_adapter.fk.frames(final_world)
                body_ids=[objective.cpu.body(r['wuji_body']).id for r in records]
                native=position[body_ids]+(frame[body_ids]@final.new_tensor(local)[...,None]).squeeze(-1)
                torch.testing.assert_close(native,values['hand_contact_points'],atol=3.e-6,rtol=0)
                values={key:value.detach().cpu().numpy() if isinstance(value,torch.Tensor) else value for key,value in values.items()}
                for key, value in values.items():
                    stored.setdefault(key, []).append(value)
                seconds = time.monotonic()-begun
                completed.append(dict(row=row, seconds=seconds, projected_closure=report['projected_closure'],
                    geometry_success=report['geometry_success'], no_valid_candidate=no_valid,
                    cost=report['cost'] if np.isfinite(report['cost']) else None, cost_parts=report['cost_parts']))
                count += 1
                if len(completed)%8 == 0 or row+1 == total or count <= 2:
                    temp = dest/'poses.tmp.npz'
                    np.savez_compressed(temp, **{k: np.asarray(v) for k, v in stored.items()})
                    temp.replace(dest/'poses.npz')
                    write(dest/'progress.json', dict(source=str(source), rows=completed, total=total))
                    write(output/f'worker_{number}.json', dict(pid=os.getpid(), device=device, solved=count,
                        failures=failures, source=str(source), row=row, seconds=seconds,
                        projected_closure=report['projected_closure'], cost_parts=report['cost_parts']))
                    print(json.dumps(dict(worker=number, source=typ+'/'+str(group['part']), row=row,
                        seconds=seconds, projected_closure=report['projected_closure'], cost_parts=report['cost_parts'])), flush=True)
            if len(completed) == total:
                write(dest/'verification.json', dict(complete=True, samples=total,
                    projected_closure=sum(r['projected_closure'] for r in completed), source=str(source)))
        except Exception:
            import traceback
            failures += 1
            error = traceback.format_exc()
            write(dest/'failure.json', dict(error=error))
            print(error, flush=True)
            raise
    write(output/f'worker_{number}.json', dict(pid=os.getpid(), device=device, state='stopped' if (output/'STOP').exists() else 'complete',
        solved=count, failures=failures, seconds=time.monotonic()-started))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mapping', type=Path, default=ROOT/'output/mano_wuji2_mapping_20261104/mapping.json')
    parser.add_argument('--devices', default='cuda:0,cuda:1,cuda:2,cuda:3')
    parser.add_argument('--population', type=int, default=1024)
    parser.add_argument('--workers-per-device', type=int, default=1)
    parser.add_argument('--max-shards', type=int, default=0)
    parser.add_argument('--max-rows', type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output/'.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def request_stop(*_):
        (args.output/'STOP').touch()
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    options = dict(source=str(args.source.resolve()), output=str(args.output.resolve()), mapping=str(args.mapping.resolve()),
        population=args.population, max_rows=args.max_rows, workers_per_device=args.workers_per_device)
    immutable = ['scripts/retarget_mano_wuji2.py', 'retarget/solver/cem.py', 'retarget/solver/base.py', 'retarget/env/grasp/pose_objective.py',
        'retarget/env/grasp/closure_types.py','retarget/env/grasp/projected_qp.py','retarget/env/grasp/pose_gpu.py','retarget/env/grasp/pose_inputs.py','retarget/env/grasp/qp_projection.py', 'retarget/initialization/ik.py']
    write(args.output/'configuration.json', dict(**options, solver='retarget.solver.cem.solve',
        objective='retarget.env.grasp.pose_objective.TemplatePoseObjective.evaluate',
        source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in immutable}))
    write(args.output/'process.json', dict(pid=os.getpid(), options=options))
    print('INDEXING', str(args.source), flush=True)
    references = args.source/'reused_shards.json'
    if args.max_shards and references.exists():
        groups = []
        for item in read(references)[:args.max_shards]:
            path = Path(item['path'])
            groups.append(dict(path=str(path), type=path.parents[1].name, part=int(path.name)))
    else:
        indexed = {}
        if references.exists():
            for item in read(references):
                path = Path(item['path'])
                key = (path.parents[1].name, int(path.name))
                indexed[key] = dict(path=str(path), type=key[0], part=key[1])
        for verification in args.source.glob('*/shards/*/verification.json'):
            path = verification.parent.resolve()
            key = (path.parents[1].name, int(path.name))
            indexed[key] = dict(path=str(path), type=key[0], part=key[1])
        groups = list(indexed.values())
    print('SOURCE_SHARDS', len(groups), flush=True)
    groups.sort(key=lambda g: (int(g['type'].split('_')[0]), g['part']))
    if args.max_shards:
        groups = groups[:args.max_shards]
    context = mp.get_context('spawn')
    tasks = context.Queue()
    for g in groups:
        if not (args.output/g['type']/'shards'/f"{g['part']:07d}"/'verification.json').exists():
            tasks.put(g)
    if args.workers_per_device < 1:
        raise ValueError('workers-per-device must be positive')
    devices = [d for d in args.devices.split(',') for _ in range(args.workers_per_device)]
    for _ in devices:
        tasks.put(None)
    workers = [context.Process(target=worker, args=(i, d, tasks, options)) for i, d in enumerate(devices)]
    for process in workers:
        process.start()
    write(args.output/'status.json', dict(state='running', workers=[dict(pid=p.pid, device=d) for p,d in zip(workers, devices)], source_shards=len(groups)))
    while any(process.is_alive() for process in workers):
        if any(process.exitcode not in (None, 0) for process in workers):
            (args.output/'STOP').touch()
        time.sleep(2)
    for process in workers:
        process.join()
    tasks.cancel_join_thread()
    failed = [p.exitcode for p in workers if p.exitcode]
    write(args.output/'status.json', dict(state='failed' if failed else 'stopped' if (args.output/'STOP').exists() else 'completed', exitcodes=[p.exitcode for p in workers]))
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
