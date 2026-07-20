#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")" && pwd)"
host_path="$project_dir/native/native_host.py"
host_name="com.diegopacheco.localhost_radar.json"
case "$(uname -s)" in
  Darwin) host_dir="$HOME/Library/Application Support/Google/Chrome/NativeMessagingHosts" ;;
  Linux) host_dir="$HOME/.config/google-chrome/NativeMessagingHosts" ;;
  *) printf 'Unsupported operating system.\n'; exit 1 ;;
esac
mkdir -p "$host_dir"
chmod +x "$host_path"
sed "s|__HOST_PATH__|$host_path|g" "$project_dir/native/$host_name" > "$host_dir/$host_name"
if test "$(uname -s)" = "Darwin"; then
  open -a "Google Chrome" "chrome://extensions"
  open "$project_dir"
else
  xdg-open "chrome://extensions"
  xdg-open "$project_dir"
fi
printf 'Native host installed at:\n%s\n' "$host_dir/$host_name"
printf 'Enable Developer mode, select Load unpacked, and choose:\n%s\n' "$project_dir"
