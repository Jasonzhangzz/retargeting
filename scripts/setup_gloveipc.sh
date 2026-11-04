#!/usr/bin/env bash
# Local extraction of Ubuntu 24.04 packages; no root or system Wine changes.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
wine_root="$project_root/.cache/moxian-wine"
mkdir -p "$wine_root/debs" "$wine_root/root"
cd "$wine_root/debs"
packages=(
  'wine64=9.0~repack-4build3|wine64_9.0~repack-4build3_amd64.deb|7f4f912b21917f28c281c99e3244ee801be2e20f0a23cd9f6d37acc97f5041ff'
  'libwine=9.0~repack-4build3|libwine_9.0~repack-4build3_amd64.deb|eb421a3660bfdc090d70ce16b06aadfde1c2bce2593292023e520988f45d009e'
  'libz-mingw-w64=1.3.1+dfsg-1|libz-mingw-w64_1.3.1+dfsg-1_all.deb|eb4ffd7e941578c58a323331b6b6953cef8a0bd61d6e0ba82f1b9b6e66650d16'
)
for entry in "${packages[@]}"; do
  IFS='|' read -r spec filename checksum <<< "$entry"
  if ! echo "$checksum  $filename" | sha256sum --check --status 2>/dev/null; then
    apt-get download "$spec"
  fi
  echo "$checksum  $filename" | sha256sum --check
  dpkg-deb -x "$filename" "$wine_root/root"
done
# Debian's wrapper has absolute /usr paths; the extracted install needs a local link.
ln -sfn wineserver64 "$wine_root/root/usr/lib/wine/wineserver"
ln -sfn ../../../../x86_64-w64-mingw32/lib/zlib1.dll \
  "$wine_root/root/usr/lib/x86_64-linux-gnu/wine/x86_64-windows/zlib1.dll"
export WINEPREFIX="$wine_root/prefix" WINEARCH=win64 WINEDEBUG=-all
export WINEDLLOVERRIDES='mscoree,mshtml='
export DISPLAY="${DISPLAY:-:1}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
if [[ -z "${XAUTHORITY:-}" && -f "$XDG_RUNTIME_DIR/gdm/Xauthority" ]]; then
  export XAUTHORITY="$XDG_RUNTIME_DIR/gdm/Xauthority"
fi
mkdir -p "$WINEPREFIX/drive_c/windows/system32"
cp --remove-destination "$wine_root/root/usr/x86_64-w64-mingw32/lib/zlib1.dll" \
  "$WINEPREFIX/drive_c/windows/system32/zlib1.dll"
"$wine_root/root/usr/lib/wine/wine64" wineboot.exe -u
echo 'GloveIPC ready. 启动：scripts/start_gloveipc.sh'
