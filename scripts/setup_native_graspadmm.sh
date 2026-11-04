#!/usr/bin/env bash
set -euo pipefail
task_root=${1:?Usage: setup_native_graspadmm.sh EXPERIMENT_ROOT BASE_PYTHON}
base_python=${2:?Base Python interpreter is required}
mkdir -p "$task_root"
if [ ! -x "$task_root/venv/bin/python" ]; then
  "$base_python" -m venv "$task_root/venv"
fi
export PIP_NO_CACHE_DIR=1
"$base_python" "$task_root/fetch_graspadmm_torch.py" "$task_root/wheels"
"$base_python" "$task_root/fetch_graspadmm_dependencies.py" "$task_root"
"$task_root/venv/bin/python" -m pip install --no-index --no-deps "$task_root"/wheels/*.whl
"$task_root/venv/bin/python" -m pip install --index-url https://mirrors.aliyun.com/pypi/simple --find-links "$task_root/wheels" -r "$task_root/requirements-graspadmm.txt"
"$task_root/venv/bin/python" - <<'PY'
import importlib.metadata,json,sys
names=['torch','mujoco','numpy','scipy','clarabel','qpsolvers','warp-lang','trimesh']
print(json.dumps({'python':sys.version,'packages':{name:importlib.metadata.version(name) for name in names}},indent=2))
assert not sys.prefix == sys.base_prefix
assert 'include-system-site-packages = false' in __import__('pathlib').Path(sys.prefix,'pyvenv.cfg').read_text()
PY
