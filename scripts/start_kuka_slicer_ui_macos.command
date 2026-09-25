#!/bin/zsh
# Start the local KUKA Slicer UI without leaving a Terminal window behind.
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
python_bin="$repo_root/.venv/bin/python"
ui_url="http://127.0.0.1:8765"
output_dir="$repo_root/outputs/mac-ui"
log_file="$output_dir/kuka_slicer_ui.log"
service_label="com.jayson.kuka-slicer.ui"
mode="${1:---background}"

if [[ ! -x "$python_bin" ]]; then
  echo "Python virtual environment is missing: $python_bin"
  echo "Please complete the macOS development setup first."
  read -r '?Press Return to close...'
  exit 1
fi

cd "$repo_root"
export KUKA_SLICER_MAX_CPU_CORES="${KUKA_SLICER_MAX_CPU_CORES:-4}"

ui_is_healthy() {
  curl --fail --silent --show-error --max-time 2 \
    "$ui_url/core-warmup-status" >/dev/null 2>&1
}

if lsof -nP -iTCP:8765 -sTCP:LISTEN >/dev/null 2>&1; then
  if ui_is_healthy; then
    echo "KUKA Slicer UI is already running at $ui_url"
    open "$ui_url"
    exit 0
  fi
  # Do not report success for an unrelated listener or a hung former UI.
  # A labelled launchd service can be safely replaced; an unknown process is
  # left untouched and reported instead of being terminated blindly.
  if /bin/launchctl print "gui/$UID/$service_label" >/dev/null 2>&1; then
    /bin/launchctl remove "$service_label" >/dev/null 2>&1 || true
    sleep 0.2
  fi
  if lsof -nP -iTCP:8765 -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Port 8765 is occupied by a process that is not responding as KUKA Slicer."
    echo "Close that process or select a different port before retrying."
    exit 1
  fi
fi

mkdir -p "$output_dir"

if [[ "$mode" == "--foreground" ]]; then
  echo "Starting KUKA Slicer UI at $ui_url"
  echo "Output directory: $output_dir"
  echo "Press Control-C to stop it."
  exec "$python_bin" -m kuka_slicer ui \
    --host 127.0.0.1 \
    --port 8765 \
    --output-dir "$output_dir"
fi

/bin/launchctl remove "$service_label" >/dev/null 2>&1 || true
/bin/launchctl submit \
  -l "$service_label" \
  -o "$log_file" \
  -e "$log_file" \
  -- "$python_bin" -m kuka_slicer ui \
    --host 127.0.0.1 \
    --port 8765 \
    --output-dir "$output_dir"

for _ in {1..50}; do
  if ui_is_healthy; then
    open "$ui_url"
    exit 0
  fi
  if ! /bin/launchctl print "gui/$UID/$service_label" >/dev/null 2>&1; then
    echo "KUKA Slicer UI stopped during startup. Log: $log_file"
    tail -n 20 "$log_file" 2>/dev/null || true
    exit 1
  fi
  sleep 0.2
done

/bin/launchctl remove "$service_label" >/dev/null 2>&1 || true
echo "The UI did not respond at $ui_url. Log: $log_file"
tail -n 20 "$log_file" 2>/dev/null || true
exit 1
