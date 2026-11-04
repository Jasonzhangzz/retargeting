#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
wine_root="$project_root/.cache/moxian-wine"
wine_bin="$wine_root/root/usr/lib/wine/wine64"
[[ -x "$wine_bin" && -f "$wine_root/prefix/system.reg" ]] || {
  echo '请先运行 scripts/setup_gloveipc.sh 安装本地 Wine 环境。' >&2
  exit 1
}
export WINEPREFIX="$wine_root/prefix" WINEARCH=win64 WINEDEBUG=-all
export WINEDLLOVERRIDES='mscoree,mshtml=' QT_OPENGL=software
export DISPLAY="${DISPLAY:-:1}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
if [[ -z "${XAUTHORITY:-}" && -f "$XDG_RUNTIME_DIR/gdm/Xauthority" ]]; then
  export XAUTHORITY="$XDG_RUNTIME_DIR/gdm/Xauthority"
fi
if [[ "${1:-}" == '--take-serial' ]]; then
  echo '暂停 WebUI 的 SDK 串口连接，将串口交给 GloveIPC；退出后可在网页点击连接。'
  curl --fail --silent --show-error --max-time 7 -H 'Content-Type: application/json' \
    -d '{"command":"disconnect"}' http://127.0.0.1:8890/api/command
  echo
else
  echo 'GloveIPC 启动为空闲模式；连接串口前，请先在 WebUI 点击「断开手套」。'
fi
echo 'Wine 串口映射：'
find "$WINEPREFIX/dosdevices" -maxdepth 1 -lname '/dev/ttyACM*' -printf '%f → %l\n'
cd "$project_root/GloveIPC_pack"
exec "$wine_bin" ./GloveIPC_V1.0.1.exe
