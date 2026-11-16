#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
geort_python="${MOXIAN_GEORT_PYTHON:-$project_root/.venv-contact/bin/python}"
if [[ ! -x "$geort_python" ]]; then
  echo '请先运行 scripts/setup_moxian_mano.sh，或通过 MOXIAN_GEORT_PYTHON 指定已有 PyTorch CUDA 环境。' >&2
  exit 1
fi
"$geort_python" -m pip install -r requirements.txt
"$geort_python" -c 'import torch,open3d,mujoco; from retarget.baseline.geort.adapter import RobotAdapter; print("GeoRT ready:",torch.cuda.is_available(),RobotAdapter().key)'
