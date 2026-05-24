#!/usr/bin/env bash
set -euo pipefail

AUTOSTART=0
NO_PATHS=0
for arg in "$@"; do
  case "$arg" in
    --autostart) AUTOSTART=1 ;;
    --no-paths)  NO_PATHS=1 ;;
    -h|--help)
      printf 'usage: %s [--autostart] [--no-paths]\n' "$0"; exit 0 ;;
    *) printf 'unknown flag: %s\n' "$arg" >&2; exit 2 ;;
  esac
done

SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$HOME/.cc-token-bar"
SETTINGS="$HOME/.claude/settings.json"
APP_DIR="/Applications/cc-token-bar.app"
LAUNCH_AGENT="$HOME/Library/LaunchAgents/com.cc-token-bar.plist"
LABEL="com.cc-token-bar"

for bin in jq swift; do
  if ! command -v "$bin" >/dev/null 2>&1; then
    printf 'missing required tool: %s\n' "$bin" >&2
    exit 1
  fi
done

printf 'creating %s\n' "$DATA_DIR"
mkdir -p "$DATA_DIR/bin" "$DATA_DIR/sessions" "$DATA_DIR/tools" "$DATA_DIR/state"

printf 'installing hook\n'
install -m 755 "$SRC_DIR/hook/hook.sh" "$DATA_DIR/bin/hook.sh"

if [[ ! -f "$DATA_DIR/config.json" ]]; then
  printf 'writing default config\n'
  cat > "$DATA_DIR/config.json" <<'JSON'
{
  "version": 1,
  "store_project_paths": true,
  "pricing": {
    "claude-opus-4":   {"input": 15.0, "output": 75.0, "cache_write": 18.75, "cache_read": 1.50},
    "claude-sonnet-4": {"input":  3.0, "output": 15.0, "cache_write":  3.75, "cache_read": 0.30},
    "claude-haiku-4":  {"input":  1.0, "output":  5.0, "cache_write":  1.25, "cache_read": 0.10}
  }
}
JSON
fi

if [[ "$NO_PATHS" -eq 1 ]]; then
  jq '.store_project_paths = false' "$DATA_DIR/config.json" > "$DATA_DIR/config.json.tmp"
  mv "$DATA_DIR/config.json.tmp" "$DATA_DIR/config.json"
fi

mkdir -p "$(dirname "$SETTINGS")"
[[ -f "$SETTINGS" ]] || printf '{}\n' > "$SETTINGS"
ts="$(date +%Y%m%d_%H%M%S)"
cp "$SETTINGS" "$SETTINGS.bak.$ts"
printf 'backed up settings to %s.bak.%s\n' "$SETTINGS" "$ts"

HOOK_CMD="$DATA_DIR/bin/hook.sh"
merge_hook() {
  local event="$1"
  local subcmd="$2"
  local matcher="$3"
  local cmd="$HOOK_CMD $subcmd"
  jq --arg ev "$event" --arg m "$matcher" --arg cmd "$cmd" '
    .hooks = (.hooks // {}) |
    .hooks[$ev] = (.hooks[$ev] // []) |
    if any(.hooks[$ev][]?.hooks[]?.command; . == $cmd) then .
    else .hooks[$ev] += [{"matcher": $m, "hooks": [{"type":"command","command":$cmd}]}]
    end
  ' "$SETTINGS" > "$SETTINGS.tmp"
  mv "$SETTINGS.tmp" "$SETTINGS"
}
merge_hook PostToolUse post_tool_use "*"
merge_hook Stop        stop          ""
merge_hook SessionEnd  session_end   ""
printf 'hooks merged into %s\n' "$SETTINGS"

printf 'building app (swift build -c release)\n'
(cd "$SRC_DIR/app" && swift build -c release >/dev/null)
BUILT="$SRC_DIR/app/.build/release/cc-token-bar"
if [[ ! -x "$BUILT" ]]; then
  printf 'build did not produce %s\n' "$BUILT" >&2
  exit 1
fi

printf 'installing app bundle at %s\n' "$APP_DIR"
rm -rf "$APP_DIR"
mkdir -p "$APP_DIR/Contents/MacOS" "$APP_DIR/Contents/Resources"
cp "$BUILT" "$APP_DIR/Contents/MacOS/cc-token-bar"
cp "$SRC_DIR/app/Info.plist" "$APP_DIR/Contents/Info.plist"

backfill() {
  local count=0
  local started_iso
  printf 'backfilling from existing transcripts\n'
  shopt -s nullglob
  for proj_dir in "$HOME"/.claude/projects/*/; do
    for tpath in "$proj_dir"*.jsonl; do
      sid="$(basename "$tpath" .jsonl)"
      decoded_cwd="$(printf '%s' "$(basename "$proj_dir")" | tr '-' '/')"
      payload="$(jq -nc --arg s "$sid" --arg t "$tpath" --arg c "/$decoded_cwd" \
        '{session_id:$s, transcript_path:$t, cwd:$c}')"
      printf '%s' "$payload" | "$DATA_DIR/bin/hook.sh" backfill || true
      count=$((count+1))
    done
  done
  shopt -u nullglob
  printf 'backfilled %d session files\n' "$count"
}
backfill

if [[ "$AUTOSTART" -eq 1 ]]; then
  printf 'installing LaunchAgent\n'
  mkdir -p "$(dirname "$LAUNCH_AGENT")"
  cat > "$LAUNCH_AGENT" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>$APP_DIR/Contents/MacOS/cc-token-bar</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><false/>
</dict>
</plist>
PLIST
  launchctl unload "$LAUNCH_AGENT" 2>/dev/null || true
  launchctl load "$LAUNCH_AGENT"
fi

printf 'launching app\n'
open "$APP_DIR"
printf 'install complete\n'