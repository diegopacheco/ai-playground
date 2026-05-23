#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

command -v bun >/dev/null || { echo "ERROR: bun not installed. Install: curl -fsSL https://bun.sh/install | bash"; exit 1; }
command -v ollama >/dev/null || { echo "ERROR: ollama not installed. Install: brew install ollama"; exit 1; }
command -v podman >/dev/null || echo "WARN: podman not installed; container path will be unavailable"

if ! curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  echo "starting ollama serve in background"
  nohup ollama serve >/tmp/qa2pw-ollama.log 2>&1 &
  echo $! > /tmp/qa2pw-ollama.pid
  for i in $(seq 1 30); do
    if curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then break; fi
    sleep 1
  done
fi
curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1 || { echo "ERROR: ollama did not come up"; exit 1; }
echo "ollama: up at http://127.0.0.1:11434"

MODEL="${QA2PW_MODEL:-qwen2.5vl:32b}"
if ! ollama list 2>/dev/null | awk '{print $1}' | grep -qx "$MODEL"; then
  echo "WARN: model $MODEL not pulled. Pull: ollama pull $MODEL"
  echo "       smaller alternative: ollama pull qwen2.5vl:7b (set QA2PW_MODEL=qwen2.5vl:7b)"
fi

echo "installing runner deps"
(cd runner && bun install --silent)

if [ -d web ]; then
  echo "installing web deps"
  (cd web && bun install --silent)
  echo "starting web on http://127.0.0.1:3000"
  (cd web && nohup bun run dev >/tmp/qa2pw-web.log 2>&1 & echo $! > /tmp/qa2pw-web.pid)
  for i in $(seq 1 30); do
    if curl -fsS http://127.0.0.1:3000 >/dev/null 2>&1; then break; fi
    sleep 1
  done
  curl -fsS http://127.0.0.1:3000 >/dev/null 2>&1 && echo "web: up at http://127.0.0.1:3000" || echo "WARN: web did not respond on :3000"
else
  echo "web/ not yet scaffolded (task T6); skipping web boot"
fi

echo "qa2pw is up. open http://127.0.0.1:3000 once web is implemented. logs in /tmp/qa2pw-*.log"
