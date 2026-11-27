"""Explicit imported CART packages; training acceptance remains unchanged."""
import hashlib
import json
from pathlib import Path

from .cart_protocol import ARCHITECTURES, FINGERS, POLICY_INPUT, POLICY_OUTPUT, POLICY_TRANSITION, PROTOCOL

ROOT = Path(__file__).resolve().parents[1]
INFERENCE_FILES = (
    'moxian_web/cart_deployment.py', 'moxian_web/cart_network.py',
    'moxian_web/cart_training.py', 'moxian_web/cart_protocol.py', 'moxian_web/cart_data.py',
    'moxian_web/cart_emf_observations.py', 'moxian_web/cart_observations.py',
    'moxian_web/cart_runtime.py', 'moxian_web/cart_objective.py',
    'moxian_web/cart_live.py', 'moxian_web/wuji_worker.py', 'moxian_web/wuji_window.py',
    'moxian_web/cart_initialization.py', 'moxian_web/cart_kinematics.py',
    'moxian_web/cart_collision.py', 'moxian_web/cart_mesh_audit.py',
    'moxian_web/cart_contacts.py',
    'moxian_web/cart_card_cost.py',
    'moxian_web/cart_wrist.py',
    'moxian_web/wuji_surface.py', 'moxian_web/qnbot_surface.py',
    'moxian_web/source_ik_initialization.py', 'moxian_web/cart_contact_witness.py',
    'retarget/env/grasp/pose_gpu.py', 'retarget/baseline/card/collision.py',
    'retarget/baseline/geort/adapter.py', 'retarget/baseline/geort/runtime.py',
)
PACKAGE_FILES = ('last.pth', 'config.json', 'status.json', 'source_profile.json', 'source_topology.json')


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inference_identity():
    return {name: sha256(ROOT / name) for name in INFERENCE_FILES}


def validate_architecture(config):
    name = config.get('architecture_name')
    if (config.get('protocol') != PROTOCOL or name not in ARCHITECTURES
            or config.get('architecture') != ARCHITECTURES[name]
            or config.get('policy_input') != POLICY_INPUT
            or config.get('policy_output') != POLICY_OUTPUT
            or config.get('policy_transition') != POLICY_TRANSITION):
        raise ValueError('CART 只支持未来十帧 EMF 与五指独立 decoder')
    from .cart_protocol import V9_OBJECTIVE
    if config.get('objective_version')!=V9_OBJECTIVE:
        raise ValueError('本地指关节运行时只加载 v9；腕部版需要腕部控制接口')
    decoder = config.get('decoder', {})
    if (decoder.get('layout') != 'independent_finger_mlps_v1'
            or decoder.get('fingers') != list(FINGERS) or decoder.get('joints_per_finger') != 4):
        raise ValueError('CART decoder 布局不匹配')


def register_import(folder, *, source, experimental=False):
    """Call only after strict policy/geometry loading and inference verification."""
    folder = Path(folder)
    config = json.loads((folder / 'config.json').read_text())
    status = json.loads((folder / 'status.json').read_text())
    validate_architecture(config)
    if config.get('profile') != 'full' or status.get('status') not in ('completed', 'rejected', 'cancelled'):
        raise ValueError('只能导入正式规模的已完成、未通过验收或已停止任务的 checkpoint')
    if status['status'] != 'completed' and not experimental:
        raise ValueError('未通过验收的权重必须显式注册为实验模型')
    manifest = dict(schema='cart_imported_inference_v1', source=source,
                    experimental=bool(experimental), acceptance_status=status['status'],
                    inference=inference_identity(), files={name: sha256(folder / name) for name in PACKAGE_FILES})
    temp = folder / 'deployment.json.tmp'
    temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    temp.replace(folder / 'deployment.json')
    return manifest


def validate_package(folder, *, allow_smoke=False):
    """Validate imported exceptions without relabeling rejected training runs."""
    folder = Path(folder)
    config = json.loads((folder / 'config.json').read_text())
    status = json.loads((folder / 'status.json').read_text())
    validate_architecture(config)
    path = folder / 'deployment.json'
    if not path.exists():
        allowed = ('completed', 'smoke_test') if allow_smoke else ('completed',)
        if status.get('status') not in allowed:
            raise ValueError('模型尚未通过真实几何验收，也未显式导入为实验模型')
        return None
    manifest = json.loads(path.read_text())
    if (manifest.get('schema') != 'cart_imported_inference_v1'
            or manifest.get('inference') != inference_identity()):
        raise ValueError('导入模型的推理实现已变化，请重新核验并导入')
    if manifest.get('files') != {name: sha256(folder / name) for name in PACKAGE_FILES}:
        raise ValueError('导入 CART 权重或配置校验和不匹配')
    if status.get('status') not in ('completed', 'rejected', 'cancelled') or config.get('profile') != 'full':
        raise ValueError('导入 CART checkpoint 的来源状态不兼容')
    if manifest.get('acceptance_status') != status['status']:
        raise ValueError('导入模型的验收状态已变化')
    if status['status'] != 'completed' and manifest.get('experimental') is not True:
        raise ValueError('未通过验收的模型只能用于显式实验推理')
    return manifest
