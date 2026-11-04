#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
test -f MoxianEMF-Windows-20260917/MoxianEMF/sdk/qnbot_sdk/core/sdk.py || {
  echo '缺少厂商 SDK，请先解压 MoxianEMF-Windows-20260917.zip 到项目目录。' >&2
  exit 1
}
if [[ ! -x .venv-moxian/bin/python ]]; then
  python3 -m venv .venv-moxian
fi
.venv-moxian/bin/python -m pip install -r requirements-moxian.txt
.venv-moxian/bin/python - <<'PY'
import site
from pathlib import Path
sdk = Path('MoxianEMF-Windows-20260917/MoxianEMF/sdk').resolve()
Path(site.getsitepackages()[0], 'moxian-sdk.pth').write_text(str(sdk) + '\n')
PY
.venv-moxian/bin/python -c 'from qnbot_sdk.core import QnBotSDK; from qnbot_sdk.glove import GloveModule; print("Moxian SDK ready")'
scripts/setup_moxian_ik.sh
scripts/setup_moxian_mano.sh
echo '启动：scripts/start_moxian_web.sh'
