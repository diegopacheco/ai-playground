#!/bin/bash

if [ -z "$OPENAI_API_KEY" ]; then
  echo "ERROR: OPENAI_API_KEY env var is not set."
  echo "Usage: OPENAI_API_KEY=sk-xxx ./run.sh"
  exit 1
fi

if ! command -v dapr &> /dev/null; then
  echo "Installing Dapr CLI..."
  brew install dapr/tap/dapr-cli
fi

dapr_version=$(dapr --version 2>/dev/null | head -1)
echo "Dapr CLI: $dapr_version"

if ! podman ps 2>/dev/null | grep -q dapr_redis; then
  echo "Initializing Dapr runtime (requires podman)..."
  dapr init --container-runtime podman
fi

PYTHON_BIN=""
for p in python3.13 python3.12 python3.11; do
  if command -v $p &> /dev/null; then
    PYTHON_BIN=$p
    break
  fi
done

if [ -z "$PYTHON_BIN" ]; then
  echo "Python 3.11-3.13 not found. Installing python@3.13 via brew..."
  brew install python@3.13
  PYTHON_BIN=$(brew --prefix python@3.13)/bin/python3.13
fi

echo "Using: $PYTHON_BIN ($($PYTHON_BIN --version))"

sed -i '' "s|\${OPENAI_API_KEY}|${OPENAI_API_KEY}|g" components/llm-provider.yaml

rm -rf venv
$PYTHON_BIN -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Starting Dapr Dashboard on http://localhost:9999 ..."
dapr dashboard -p 9999 &
DASHBOARD_PID=$!

echo ""
echo "Running agent with Dapr sidecar..."
echo "==========================================="
dapr run --app-id assistant-agent --resources-path components -- python agent.py

echo ""
echo "Agent finished. Dashboard still running at http://localhost:9999"
echo "Press Ctrl+C to stop the dashboard."
wait $DASHBOARD_PID
