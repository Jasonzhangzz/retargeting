"""Read-only, same-pose native-mesh versus configuration-SDF collision audit."""
import argparse
import contextlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from retarget.learning.wuji2_sdf.common import fingerprint, sha256
from retarget.learning.wuji2_sdf.geometry import Geometry
from retarget.learning.wuji2_sdf.model import load_model

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / 'output/wuji2_sdf/mlp100k_v1/best.pt'
THRESHOLD_MM = .5


def pair_depths(source, target, distances):
    depth = torch.zeros(21 * 21, device=distances.device)
    depth.scatter_reduce_(0, source * 21 + target, (-distances * 1000).clamp_min(0), reduce='amax')
    depth = depth.reshape(21, 21)
    return torch.maximum(depth, depth.T).cpu().numpy()


def visual_witnesses(points, source, target, mesh_sdf, neural_sdf, per_group=4, limit=768):
    """Keep deepest witnesses per link pair AND error category, without inventing contacts."""
    x, a, b, real = (v.detach().cpu().numpy() for v in (points, source, target, mesh_sdf))
    pred = neural_sdf.detach().cpu().numpy() if neural_sdf is not None else None
    mesh_hit = real < -THRESHOLD_MM / 1000
    model_hit = pred < -THRESHOLD_MM / 1000 if pred is not None else np.zeros(len(real), bool)
    ids = np.flatnonzero(mesh_hit | model_hit)
    # 0=both, 1=mesh only, 2=neural only; reserve points for disagreements.
    category = np.where(mesh_hit & model_hit, 0, np.where(mesh_hit, 1, 2))
    keys = (np.minimum(a, b) * 21 + np.maximum(a, b)) * 3 + category
    selected = []
    for key in np.unique(keys[ids]):
        group = ids[keys[ids] == key]
        score = np.maximum(-real[group], -pred[group]) if pred is not None else -real[group]
        selected.extend(group[np.argsort(-score, kind='stable')[:per_group]].tolist())
    selected = np.asarray(selected[:limit], dtype=int)
    return dict(positions_m=x[selected].tolist(), source_links=a[selected].tolist(), target_links=b[selected].tolist(),
                mesh_sdf_mm=(real[selected] * 1000).tolist(),
                neural_sdf_mm=(pred[selected] * 1000).tolist() if pred is not None else None,
                total_collision_queries=len(ids), displayed_queries=len(selected), sampling='deepest_per_pair_and_hit_category')


class CollisionComparison:
    def __init__(self, model_path=DEFAULT_MODEL, device='cpu', per_link=128, batch_size=8192):
        if per_link < 54 or batch_size < 1:
            raise ValueError('Require at least 54 surface samples per link and a positive query batch')
        self.device = str(device)
        self.geometry = Geometry(device)
        self.batch_size = batch_size
        self.model = None
        self.model_error = None
        self.model_info = None
        try:
            model, saved = load_model(model_path, device)
            if fingerprint(saved['geometry']) != fingerprint(self.geometry.spec()):
                raise ValueError('SDF 权重与当前 Wuji2 几何、坐标或关节域不匹配')
            self.model = model.requires_grad_(False)
            self.model_info = dict(name=Path(model_path).parent.name, epoch=saved.get('epoch'),
                                   checkpoint_sha256=sha256(model_path))
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            self.model_error = str(exc)
        auditor = self.geometry.collision
        material, owner = (a.numpy() for a in auditor.samples[False])
        samples = []
        rng = np.random.default_rng(23891)
        for i, mesh in enumerate(self.geometry.surface.meshes):
            fixed = material[owner == i]
            extra = self.geometry.surface_points(mesh, per_link - len(fixed), rng)
            samples.append(np.concatenate((fixed, extra)))
        local = torch.as_tensor(np.concatenate(samples), device=device, dtype=torch.float32)
        owners = torch.arange(21, device=device).repeat_interleave(per_link)
        allowed = torch.as_tensor(self.geometry.allowed, device=device)
        si, target = allowed[owners].nonzero(as_tuple=True)
        self.source = owners[si]
        self.target = target
        self.local = local[si]
        self.collar = torch.as_tensor(auditor.collar_radii_m, device=device)
        self.samples_per_link = per_link
        self.context = None
        self.frames = self.disagreements = self.matched_disagreements = 0

    def sync(self):
        if self.device.startswith('cuda'):
            torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def query(self, q, joint_names=None, context_key=None):
        q = np.asarray(q, np.float32)
        if q.shape != (20,) or not np.isfinite(q).all():
            raise ValueError('碰撞对照需要同帧的 20 个有限 Wuji2 关节角')
        if joint_names is not None and joint_names != self.geometry.joint_names:
            raise ValueError('当前求解器关节顺序与 Wuji2 SDF 不匹配')
        if self.context != context_key:
            self.context = context_key
            self.frames = self.disagreements = self.matched_disagreements = 0
        self.sync()
        began = time.perf_counter()
        frames = self.geometry.frames(q[None])
        self.sync()
        fk_done = time.perf_counter()
        raw, valid = self.geometry.collision.query(*frames, dense=True)
        self.sync()
        dense = raw[0].cpu().numpy()
        dense = np.maximum(dense, dense.T)
        query_valid = bool(valid[0])
        if not query_valid:
            raise ValueError('原生密集网格查询无效，不能生成碰撞对照结论')
        dense_done = time.perf_counter()
        p, r = (a[0, self.geometry.surface.body_ids] for a in frames)
        world = (r[self.source] @ self.local[..., None]).squeeze(-1) + p[self.source]
        target_local = (r[self.target].transpose(-1, -2) @ (world - p[self.target])[..., None]).squeeze(-1)
        # Exactly the same mechanical collar exclusions as the native auditor.
        exclude = ((self.source == 0) & (self.collar[self.target] > 0) &
                   (target_local.norm(dim=-1) <= self.collar[self.target]))
        exclude |= ((self.target == 0) & (self.collar[self.source] > 0) &
                    (self.local.norm(dim=-1) <= self.collar[self.source]))
        points, a, b = world[~exclude], self.source[~exclude], self.target[~exclude]
        self.sync()
        preparation_done = time.perf_counter()
        truth = self.geometry.label(points[None], b[None], frames)[0]
        actual = torch.as_tensor(truth, device=self.device)
        matched = pair_depths(a, b, actual)
        mesh_done = time.perf_counter()
        predicted = None
        neural = None
        if self.model is not None:
            qq = torch.as_tensor(q, device=self.device)
            predicted = torch.cat([self.model(qq, points[i:i + self.batch_size], b[i:i + self.batch_size])
                                   for i in range(0, len(points), self.batch_size)])
            if not bool(torch.isfinite(predicted).all()):
                raise ValueError('SDF 输出非有限距离')
            neural = pair_depths(a, b, predicted)
        self.sync()
        neural_done = time.perf_counter()
        lower = np.asarray(self.geometry.domain['lower'])
        upper = np.asarray(self.geometry.domain['upper'])
        outside = np.flatnonzero((q < lower - 1e-6) | (q > upper + 1e-6))
        def result(depth, milliseconds):
            return dict(query_valid=query_valid, max_penetration_mm=float(depth.max()),
                        collision=bool(depth.max() > THRESHOLD_MM) if query_valid else None,
                        collision_pairs=int(np.count_nonzero(np.triu(depth > THRESHOLD_MM, 1))),
                        processing_ms=milliseconds)
        dense_result = result(dense, (dense_done - fk_done) * 1000)
        matched_result = result(matched, (mesh_done - preparation_done) * 1000)
        neural_result = result(neural, (neural_done - mesh_done) * 1000) if neural is not None else None
        self.frames += 1
        comparison = None
        if neural is not None:
            self.disagreements += dense_result['collision'] != neural_result['collision']
            self.matched_disagreements += matched_result['collision'] != neural_result['collision']
            difference = abs(predicted - actual) * 1000
            near = actual.abs() <= .01
            comparison = dict(dense_agrees=dense_result['collision'] == neural_result['collision'],
                              matched_agrees=matched_result['collision'] == neural_result['collision'],
                              dense_depth_error_mm=abs(float(dense.max() - neural.max())),
                              matched_depth_error_mm=abs(float(matched.max() - neural.max())),
                              point_mae_mm=float(difference.mean()),
                              near_10mm_mae_mm=float(difference[near].mean()) if bool(near.any()) else None,
                              frames=self.frames, dense_disagreement_frames=self.disagreements,
                              matched_disagreement_frames=self.matched_disagreements)
        pairs = []
        for i in range(21):
            for j in range(i + 1, 21):
                predicted_depth = float(neural[i, j]) if neural is not None else None
                if max(float(dense[i, j]), float(matched[i, j]), predicted_depth or 0) <= .1:
                    continue
                pairs.append(dict(a=i, b=j, link_a=self.geometry.collision.names[i], link_b=self.geometry.collision.names[j],
                                  dense_mm=float(dense[i, j]), matched_mm=float(matched[i, j]), neural_mm=predicted_depth,
                                  agrees=bool((dense[i, j] > THRESHOLD_MM) == (predicted_depth > THRESHOLD_MM))
                                         if predicted_depth is not None else None))
        pairs.sort(key=lambda row: max(row['dense_mm'], row['neural_mm'] or 0), reverse=True)
        visualization = dict(asset_sha256=self.geometry.surface.asset_hash,
                             link_positions_m=p.cpu().numpy().tolist(), link_rotations=r.cpu().numpy().tolist(),
                             witnesses=visual_witnesses(points, a, b, actual, predicted))
        return dict(ready=True, status='ready', device=self.device, threshold_mm=THRESHOLD_MM,
                    dense_mesh=dense_result, matched_mesh=matched_result, neural=neural_result,
                    comparison=comparison, pairs=pairs, query_points=len(points),
                    samples_per_link=self.samples_per_link, model=self.model_info, model_error=self.model_error,
                    in_training_domain=not len(outside), out_of_domain_joints=[self.geometry.joint_names[i] for i in outside],
                    fk_ms=(fk_done - began) * 1000, query_preparation_ms=(preparation_done - dense_done) * 1000,
                    processing_ms=(time.perf_counter() - began) * 1000, qpos=q.tolist(), visualization=visualization)

    def export_visual_model(self, folder):
        from retarget.learning.wuji2_sdf.common import atomic_json
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        meshes = [dict(name=name, vertices_m=np.asarray(mesh.vertices, np.float32).tolist(),
                       faces=np.asarray(mesh.faces, np.int32).tolist())
                  for name, mesh in zip(self.geometry.collision.names, self.geometry.surface.meshes)]
        atomic_json(folder / 'visual_model.json', dict(ready=True, format='wuji-collision-visual-v1',
                    asset_sha256=self.geometry.surface.asset_hash, units='m', coordinate_frame='native_fixed_wrist', meshes=meshes))

    def export_examples(self, folder, data):
        from retarget.learning.wuji2_sdf.common import atomic_json
        from retarget.learning.wuji2_sdf.data import read_manifest
        manifest = read_manifest(data, verify=True)
        if fingerprint(manifest['spec']['geometry']) != fingerprint(self.geometry.spec()):
            raise ValueError('测试姿态资产与当前几何不匹配')
        with np.load(Path(data) / manifest['shards'][0]['file'], allow_pickle=False) as rows:
            q, ids, depth = rows['qpos'], rows['pose_id'], rows['max_penetration_mm']
        chosen = list(dict.fromkeys([int(np.argmin(depth)), int(np.argmax(depth))] +
                                    [int(np.argmin(abs(depth - d))) for d in (.5, .75, 2., 5.)]))
        folder = Path(folder)
        (folder / 'examples').mkdir(parents=True, exist_ok=True)
        catalog = []
        for index, row in enumerate(chosen):
            result = self.query(q[row], context_key='test-example-' + str(ids[row]))
            disagreements = sum((p['matched_mm'] > THRESHOLD_MM) != (p['neural_mm'] > THRESHOLD_MM)
                                for p in result['pairs']) if result['neural'] else 0
            label = f'测试姿态 {index + 1} · #{ids[row]} · 网格 {result["dense_mesh"]["max_penetration_mm"]:.2f} mm'
            if disagreements:
                label += f' · {disagreements} 对判定不同'
            example = dict(index=index, pose_id=int(ids[row]), label=label, disagreement_pairs=disagreements,
                           dataset_fingerprint=manifest['fingerprint'])
            result.update(enabled=True, input_kind='test', solver_backend='test_dataset', timestamp_ms=None,
                          age_ms=None, example=example)
            atomic_json(folder / 'examples' / f'{index}.json', result)
            catalog.append(example)
        atomic_json(folder / 'examples.json', dict(ready=True, items=catalog, model=self.model_info,
                    asset_sha256=self.geometry.surface.asset_hash, dataset_fingerprint=manifest['fingerprint']))
        return catalog


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--worker', action='store_true')
    p.add_argument('--visual-root', type=Path)
    p.add_argument('--export-visuals', type=Path)
    p.add_argument('--example-data', type=Path)
    p.add_argument('--model', default=str(DEFAULT_MODEL))
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.export_visuals:
        with contextlib.redirect_stdout(sys.stderr):
            worker = CollisionComparison(args.model, args.device)
            worker.export_visual_model(args.export_visuals)
            examples = worker.export_examples(args.export_visuals, args.example_data) if args.example_data else []
        print(json.dumps(dict(ready=True, examples=len(examples))), flush=True)
        return
    if not args.worker:
        p.error('Use --worker or --export-visuals')
    worker = None
    for line in sys.stdin:
        packet = {}
        try:
            packet = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                if worker is None:
                    worker = CollisionComparison(args.model, args.device)
                    if args.visual_root:
                        worker.export_visual_model(args.visual_root)
                result = worker.query(packet['frame']['qpos'], packet['frame'].get('joint_names'), packet['context_key'])
            result.update(request_id=packet['request_id'], context_key=packet['context_key'], timestamp_ms=packet['timestamp_ms'],
                          input_kind=packet['input_kind'], solver_backend=packet['frame']['solver_backend'],
                          solver_session_id=packet['frame'].get('solver_session_id'),
                          model_run_id=packet['frame'].get('model_run_id'))
        except Exception as exc:
            import traceback
            traceback.print_exc(file=sys.stderr)
            result = dict(ready=False, status='error', request_id=packet.get('request_id'), error=str(exc))
        print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
