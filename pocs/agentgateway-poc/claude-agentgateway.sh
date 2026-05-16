#!/usr/bin/env bash
set -euo pipefail

if ! command -v claude >/dev/null 2>&1; then
  echo "claude CLI not found. Install: npm install -g @anthropic-ai/claude-code"
  exit 1
fi

if ! command -v ccr >/dev/null 2>&1; then
  echo "claude-code-router (ccr) not found. Install: npm install -g @musistudio/claude-code-router"
  exit 1
fi

if ! curl -sf -o /dev/null -m 2 "http://localhost:8080/v1/chat/completions" -X POST \
     -H "content-type: application/json" -d '{}' && \
   [ "$(curl -s -o /dev/null -w '%{http_code}' -m 2 http://localhost:8080/)" = "000" ]; then
  echo "agentgateway is not reachable at http://localhost:8080. run ./start.sh first."
  exit 1
fi

HERE="$(cd "$(dirname "$0")" && pwd)"

pkill -f "node .*strip-proxy.js" 2>/dev/null || true
PORT=8090 UPSTREAM="http://localhost:8080" \
  nohup node "$HERE/app/strip-proxy.js" >/tmp/strip-proxy.log 2>&1 &
echo $! > /tmp/strip-proxy.pid
until curl -s -o /dev/null -m 1 -X POST -d '{}' -H "content-type: application/json" \
  http://localhost:8090/v1/chat/completions; do sleep 1; done

mkdir -p "$HOME/.claude-code-router"
cat > "$HOME/.claude-code-router/config.json" <<EOF
{
  "LOG": false,
  "Providers": [
    {
      "name": "agentgateway",
      "api_base_url": "http://localhost:8090/v1/chat/completions",
      "api_key": "unused-handled-by-agentgateway",
      "models": ["gpt-4o-mini"],
      "transformer": {
        "use": [
          "openai",
          ["maxtoken", { "max_tokens": 16384 }]
        ]
      }
    }
  ],
  "Router": {
    "default": "agentgateway,gpt-4o-mini",
    "background": "agentgateway,gpt-4o-mini",
    "think": "agentgateway,gpt-4o-mini",
    "longContext": "agentgateway,gpt-4o-mini"
  }
}
EOF

ccr restart >/dev/null 2>&1 || true

export CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1
export DISABLE_THINKING=1

echo "claude -> ccr -> strip-proxy -> agentgateway -> OpenAI"
exec ccr code "$@"
