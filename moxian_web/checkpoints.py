"""Read-only checkpoint discovery for explicit, reversible live comparison."""
import copy
import hashlib
import json
import time
from pathlib import Path

LABELS={'moxian_wuji':'Moxian六动作骨长与EMF外参','wuji_native':'Wuji原生六动作标定','bone_extrinsics':'Moxian 17动作标定（历史）','parameters':'xMimic 思路版（历史）','flashsac':'FlashSAC buffer 版（历史）','delta':'MANO＋Delta 联合训练（历史）','least_squares':'分阶段最小二乘标定（历史）'}

def preview_revision_compatible(candidate):
    if candidate.get('surface_backend')=='qnbot':
        r=candidate.get('report') or {}
        if r.get('method')=='moxian_wuji':return r.get('fit_revision') in ('moxian_native25_wuji_six_v1','moxian_wuji_six_v2_distal_twist','moxian_wuji_six_v3_bounded_stable') and candidate.get('bone_extrinsics',{}).get('schema_revision')=='moxian_bone_extrinsics_v1'
        if r.get('method')=='wuji_native':return r.get('fit_revision')=='wuji_studio_calibration_2026_7_1_v1' and candidate.get('wuji_native',{}).get('schema_revision')=='moxian_wuji_native21_v1'
        if r.get('method')=='bone_extrinsics':
            return r.get('fit_revision')=='moxian_static_bone_extrinsics_v1' and candidate.get('bone_extrinsics',{}).get('schema_revision')=='moxian_bone_extrinsics_v1'
        if r.get('method')=='least_squares':
            # Discovery runs in the SDK web environment, without NumPy/Torch.
            # Keep the fixed-forward ABI check independent of solver imports.
            return r.get('fit_revision')=='qnbot_least_squares_v1' and r.get('frozen_forward') is True and r.get('online_optimization') is False and candidate.get('delta_policy') is None
        return r.get('fit_revision') in ('qnbot_trajectory_fit_v1','qnbot_trajectory_fit_v2','qnbot_trajectory_fit_v3','qnbot_trajectory_fit_v4','qnbot_trajectory_fit_v5','qnbot_trajectory_fit_v6') and r.get('frozen_forward') is True and r.get('online_optimization') is False
    from .hand_model import MANO_FIT_REVISION
    report=candidate.get('report') or {}
    if report.get('fit_revision')==MANO_FIT_REVISION:return True
    # These revisions share a fixed-forward geometry ABI; old objectives are
    # diagnostic comparisons only, never accepted by the current apply gate.
    return bool(report.get('fit_revision') in ('glove_motion_joint_fit_v14','glove_motion_joint_fit_v15','glove_motion_joint_fit_v16','glove_motion_joint_fit_v17','glove_motion_joint_fit_v18','glove_motion_joint_fit_v19','glove_motion_joint_fit_v20') and
        candidate.get('runtime_backend')=='fixed_mano_v1' and report.get('frozen_forward') is True and
        report.get('online_optimization') is False and report.get('shape_validation',{}).get('accepted') is True)


class CheckpointCatalog:
    def __init__(self, output):
        self.output=Path(output).resolve();self.cache={};self.file_cache={}

    def _paths(self,serial):
        key=hashlib.sha256(serial.encode()).hexdigest()[:16];home=self.output/'hand_model'/key
        paths=[home/'candidate.json',*sorted((home/'methods').glob('*/candidate.json'))]
        paths.extend(sorted((home/'qnbot/methods').glob('*/candidate.json')))
        # Only completed six-trial comparisons; interrupted/smoke/debug runs are
        # not presented as fully trained alternatives.
        for summary in sorted((self.output/'debug').glob('*/summary.json')):
            if (summary.parent/'DIAGNOSTIC_ONLY.json').exists():continue
            try:records=json.loads(summary.read_text())
            except (OSError,ValueError):continue
            if not isinstance(records,list):continue
            for method in LABELS:
                completed=[r for r in records if isinstance(r,dict) and r.get('method')==method and not r.get('error')]
                if { (r.get('cache'),r.get('run')) for r in completed } != {(cache,run) for cache in ('cold','warm') for run in (1,2,3)}:continue
                paths.extend(summary.parent/f"{method}-{r['cache']}-{r['run']}"/'candidate.json' for r in completed)
        return paths

    def _scan(self,serial,side):
        key=(serial,side);cached=self.cache.get(key)
        if cached and time.monotonic()-cached[0]<5:return cached[1]
        result={}
        for source in self._paths(serial):
            try:
                path=source.resolve();relative=path.relative_to(self.output)
                stat=path.stat();signature=(stat.st_ino,stat.st_size,stat.st_mtime_ns)
                cached_file=self.file_cache.get(path)
                if cached_file and cached_file[0]==signature:
                    item,summary=cached_file[1]
                    if summary['serial']==serial and summary['side']==side:result[item['id']]=(item,summary)
                    continue
                content=path.read_bytes();candidate=json.loads(content)
                if candidate.get('serial')!=serial or candidate.get('side')!=side:continue
                report=candidate.get('report') or {};method=report.get('method','legacy')
                if method not in LABELS:continue
                if report.get('fit_revision') is None:continue
                digest=hashlib.sha256(str(relative).encode()+b'\0'+content).hexdigest()[:24]
                label=LABELS[method]
                item=dict(id=digest,label=label,method=method,source=str(relative),accepted=bool(report.get('accepted')),
                    contact_gap_mm=report.get('contact_gap_mm'),contact_max_gap_mm=report.get('contact_max_gap_mm'),
                    max_penetration_mm=report.get('max_penetration_mm'),end_to_end_s=report.get('end_to_end_s',report.get('seconds')),
                    calibration_fingerprint=candidate.get('calibration_fingerprint'),skeleton_runtime_revision=candidate.get('skeleton_runtime_revision'),
                    weights_hash=candidate.get('weights_hash'),modified_at_ms=path.stat().st_mtime*1000)
                item['surface_backend']=candidate.get('surface_backend','mano')
                # Discovery needs compatibility metadata, not every training
                # frame/residual in every historical checkpoint. Retaining
                # those Python trees causes long GC pauses in serial dispatch.
                summary=dict(serial=serial,side=side,preview_compatible=preview_revision_compatible(candidate),
                    bone_extrinsics=bool(candidate.get('bone_extrinsics')),wuji_native=bool(candidate.get('wuji_native')))
                result[digest]=(item,summary)
                self.file_cache[path]=(signature,(item,summary))
            except (OSError,ValueError,KeyError,TypeError):continue
        self.cache[key]=(time.monotonic(),result);return result

    def list(self,serial,side,frame=None):
        from .hand_model import MANO_FIT_REVISION
        frame=frame or {};items=[]
        for item,candidate in self._scan(serial,side).values():
            if not candidate['preview_compatible']:continue
            item=dict(item);reason='Wuji内置手型已停用，仅保留诊断' if candidate.get('wuji_native') else None
            if frame.get('calibration_fingerprint') and item['calibration_fingerprint']!=frame['calibration_fingerprint']:reason='骨架标定版本不匹配'
            if not candidate.get('bone_extrinsics') and not candidate.get('wuji_native') and frame.get('skeleton_runtime_revision') and item['skeleton_runtime_revision']!=frame['skeleton_runtime_revision']:reason='上游 IK 版本不匹配'
            item.update(compatible=reason is None,incompatible_reason=reason);items.append(item)
        selected=[]
        for backend in ('qnbot','mano'):
            for method in LABELS:
                alternatives=[item for item in items if item['method']==method and item['surface_backend']==backend]
                if not alternatives:continue
                item=max(alternatives,key=lambda item:(item['modified_at_ms'],item['source']))
                item['label']=LABELS[method];selected.append(item)
        return selected

    def get(self,identifier,serial,side):
        record=self._scan(serial,side).get(identifier)
        if record is None:raise ValueError('权重不存在或已更新，请刷新列表')
        item,_=record
        path=(self.output/item['source']).resolve()
        relative=path.relative_to(self.output);content=path.read_bytes()
        digest=hashlib.sha256(str(relative).encode()+b'\0'+content).hexdigest()[:24]
        if digest!=identifier:raise ValueError('权重已更新，请刷新列表')
        return copy.deepcopy(item),json.loads(content)


def comparison_profile(candidate,model,frame,mapping,serial,side):
    """Validate compatibility without requiring quality acceptance for preview."""
    from .hand_anatomy import GEOMETRY_REVISION
    from .mano_local_geometry import validate_rest_residual
    validate_rest_residual(candidate.get('rest_vertex_residual_m'))
    from .hand_model import MANO_FIT_REVISION,tracking_registration_matches
    if candidate.get('serial')!=serial or candidate.get('side')!=side or candidate.get('weights_hash')!=model.weights_hash:raise ValueError('权重与设备、手侧或 MANO 不匹配')
    if candidate.get('geometry_revision')!=GEOMETRY_REVISION or not preview_revision_compatible(candidate):raise ValueError('候选模型版本过旧，不能实时对比')
    if not frame or not tracking_registration_matches(candidate,frame):raise ValueError('权重与当前骨架标定／IK 版本不匹配')
    if candidate.get('mapping',{}).get('assignments')!=(mapping or {}).get('assignments'):raise ValueError('权重的触觉 SID 分区与当前手套不匹配')
    if candidate.get('glove_parameters') is not None:
        from .glove_geometry import validate_payload
        validate_payload(candidate.get('skeleton_model'),side,candidate['glove_parameters'])
    result=copy.deepcopy(candidate)
    result.update(calibrated=False,revision=None,applied_at_ms=None,automatic_application=False,
                  mounting_calibrated=True,mounting_estimated=True)
    # Candidate mount estimates are enabled exactly as in validation; this flag
    # is temporary and must never be interpreted as acceptance or persisted.
    return result
