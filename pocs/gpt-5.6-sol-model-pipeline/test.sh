#!/usr/bin/env bash
set -euo pipefail
port="${UI_PORT:-8091}"
OPEN_BROWSER=0 ./train.sh
OPEN_BROWSER=0 ./start-ui.sh
health=$(curl -fsS "http://localhost:${port}/health")
prediction=$(curl -fsS -X POST "http://localhost:${port}/predict" -H 'Content-Type: application/json' -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}')
python3 -c 'import json,sys; data=json.loads(sys.argv[1]); assert data["status"] == "ready" and data["model"]' "$health"
python3 -c 'import json,sys; data=json.loads(sys.argv[1]); assert data["species"] == "Iris setosa" and data["confidence"] > 90' "$prediction"
echo "$health"
echo "$prediction"
echo "All checks passed"
