"""Lightweight web-process transport; torch and meshes stay in a child process."""
import os
import subprocess
import threading

from .geort_diagnostics_service import DiagnosticService, ROOT


def comparison_packet(w):
    """Take only one already-published robot pose; never substitute live Glove q."""
    import math
    from retarget.baseline.geort.adapter import digest
    if not (w.get('ready') or w.get('display_available')):
        return None
    q = w.get('qpos')
    stamp = w.get('source_timestamp_ms')
    if (not isinstance(q, list) or len(q) != 26 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in q)
            or not isinstance(stamp, (int, float)) or not math.isfinite(stamp)):
        return None
    playback = w.get('playback') or {}
    context = digest([w.get(k) for k in ('solver_backend', 'solver_session_id', 'active_solver_session_id',
                                      'mapping_version', 'geometry_version', 'model_run_id')] +
                     [bool(playback.get('active')), playback.get('id')])
    names = w.get('joint_names')
    if isinstance(names, list) and len(names) == 26:
        names = names[6:]
    return dict(context_key=context, timestamp_ms=stamp, robot=True,
                input_kind='replay' if (w.get('playback') or {}).get('active') else 'live',
                frame=dict(qpos=q[6:], joint_names=names, solver_backend=w.get('solver_backend'),
                           solver_session_id=w.get('solver_session_id'), model_run_id=w.get('model_run_id')))


class CollisionComparisonService(DiagnosticService):
    def _start(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / 'collision-comparison.log').open('a') as log:
            proc = subprocess.Popen([os.environ.get('MOXIAN_GEORT_PYTHON', str(ROOT / '.venv-contact/bin/python')),
                '-u', '-m', 'moxian_web.wuji_sdf_comparison', '--worker', '--model',
                os.environ.get('MOXIAN_WUJI_SDF_MODEL', str(ROOT / 'output/wuji2_sdf/mlp100k_v1/best.pt')),
                '--visual-root', str(self.root)],
                cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1,
                env={**os.environ, 'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1'})
        self.proc = proc
        self.context_key = None
        threading.Thread(target=self._read, args=(proc,), daemon=True).start()
        return proc

    def snapshot(self, wuji=None):
        result = super().snapshot()
        if wuji is not None and result.get('ready'):
            current = comparison_packet(wuji)
            if current is None or current['context_key'] != result.get('context_key'):
                result.update(ready=False, status='waiting_input')
        if result.get('status') == 'waiting_input':
            result['error'] = '等待当前 Wuji2 求解器的有效姿态；切换到 Moxian／Wuji2 对照'
        if not result.get('ready'):
            for key in ('dense_mesh', 'matched_mesh', 'neural', 'comparison', 'visualization'):
                result[key] = None
            result['pairs'] = []
        return result

    def visual_asset(self, kind, index=None):
        if kind == 'example':
            if not isinstance(index, int) or not 0 <= index < 32:
                raise ValueError('无效测试姿态编号')
            path = self.root / 'examples' / f'{index}.json'
        elif kind in ('model', 'examples'):
            path = self.root / ('visual_model.json' if kind == 'model' else 'examples.json')
        else:
            raise ValueError('未知可视化资源')
        if not path.is_file():
            return dict(ready=False, error='暂无可视化资源，请先初始化碰撞检测或生成测试姿态')
        return path.read_bytes()
