#!/bin/zsh
# Double-click this file in Finder to start the local KUKA Slicer UI.
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
python_bin="$repo_root/.venv/bin/python"
ui_url="http://127.0.0.1:8765"

if [[ ! -x "$python_bin" ]]; then
  echo "Python virtual environment is missing: $python_bin"
  echo "Please complete the macOS development setup first."
  read -r '?Press Return to close...'
  exit 1
fi

cd "$repo_root"
export KUKA_SLICER_MAX_CPU_CORES="${KUKA_SLICER_MAX_CPU_CORES:-4}"

if lsof -nP -iTCP:8765 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "KUKA Slicer UI is already running at $ui_url"
  open "$ui_url"
  exit 0
fi

echo "Starting KUKA Slicer UI at $ui_url"
echo "Output directory: $repo_root/outputs/mac-ui"
echo "Press Control-C in this Terminal window to stop it."

"$python_bin" -m kuka_slicer ui \
  --host 127.0.0.1 \
  --port 8765 \
  --output-dir "$repo_root/outputs/mac-ui" &
server_pid=$!

cleanup() {
  if kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

for _ in {1..50}; do
  if curl --fail --silent --show-error "$ui_url/" >/dev/null 2>&1; then
    open "$ui_url"
    wait "$server_pid"
    exit $?
  fi
  sleep 0.2
done

echo "The UI did not respond at $ui_url. Check the errors above."
exit 1
