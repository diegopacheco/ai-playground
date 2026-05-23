#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

command -v bun >/dev/null || { echo "ERROR: bun not installed"; exit 1; }
command -v ollama >/dev/null || { echo "ERROR: ollama not installed"; exit 1; }

MODEL="${QA2PW_MODEL:-qwen2.5vl:32b}"
OLLAMA_URL="${QA2PW_OLLAMA_URL:-http://127.0.0.1:11434}"

if ! curl -fsS "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
  echo "starting ollama serve in background"
  nohup ollama serve >/tmp/qa2pw-ollama.log 2>&1 &
  echo $! > /tmp/qa2pw-ollama.pid
  for i in $(seq 1 30); do
    if curl -fsS "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then break; fi
    sleep 1
  done
fi
curl -fsS "$OLLAMA_URL/api/tags" >/dev/null 2>&1 || { echo "ERROR: ollama did not come up at $OLLAMA_URL"; exit 1; }
echo "ollama: up at $OLLAMA_URL"

if ! ollama list 2>/dev/null | awk '{print $1}' | grep -qx "$MODEL"; then
  echo "ERROR: model $MODEL not pulled. Run: ollama pull $MODEL"
  exit 1
fi
echo "model: $MODEL ready"

(cd runner && bun install --silent)

if [ ! -d "$HOME/Library/Caches/ms-playwright" ] && [ ! -d "$HOME/.cache/ms-playwright" ]; then
  echo "installing playwright chromium (one-time)"
  (cd runner && bunx playwright install chromium)
fi

echo ""
if [ -n "${EVAL_CASE_ID:-}" ]; then
  echo "running single case: $EVAL_CASE_ID"
else
  echo "running all 10 eval cases against $MODEL"
fi
echo ""

QA2PW_MODEL="$MODEL" QA2PW_OLLAMA_URL="$OLLAMA_URL" \
  bun run --cwd runner eval
