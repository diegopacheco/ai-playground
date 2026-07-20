#!/usr/bin/env bash
set -euo pipefail
host_name="com.diegopacheco.localhost_radar.json"
case "$(uname -s)" in
  Darwin) host_dir="$HOME/Library/Application Support/Google/Chrome/NativeMessagingHosts" ;;
  Linux) host_dir="$HOME/.config/google-chrome/NativeMessagingHosts" ;;
  *) printf 'Unsupported operating system.\n'; exit 1 ;;
esac
rm -f "$host_dir/$host_name"
if test "$(uname -s)" = "Darwin"; then
  open -a "Google Chrome" "chrome://extensions"
else
  xdg-open "chrome://extensions"
fi
printf 'Native host removed. Select Localhost Radar in Chrome and choose Remove.\n'
