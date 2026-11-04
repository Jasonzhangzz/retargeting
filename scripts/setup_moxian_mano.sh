#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
mano_python="${MOXIAN_MANO_PYTHON:-$project_root/.venv-contact/bin/python}"
if [[ ! -x "$mano_python" ]]; then
  mano_python="$project_root/.venv-moxian-mano/bin/python"
  if [[ ! -x "$mano_python" ]]; then python3 -m venv "$project_root/.venv-moxian-mano"; fi
  "$mano_python" -m pip install -r requirements-moxian-mano.txt
fi
"$mano_python" - <<'PY'
import os
from pathlib import Path
import torch,mujoco,mujoco_warp,warp,trimesh
model=Path(os.environ.get('MOXIAN_MANO_MODEL','/home/zz/GraspSONIC/thirdparty/DexForge/assets/mano_v1_2/models/MANO_LEFT.pkl'))
if not model.is_file():raise SystemExit('请通过 MOXIAN_MANO_MODEL 指定已有、获许可的 MANO_LEFT.pkl；脚本不会下载模型权重。')
if not torch.cuda.is_available():raise SystemExit('MANO 碰撞 worker 需要 CUDA；请检查 PyTorch 与 NVIDIA 驱动。')
print('MANO runtime ready:',torch.cuda.get_device_name(),mujoco.__version__,model)
PY
