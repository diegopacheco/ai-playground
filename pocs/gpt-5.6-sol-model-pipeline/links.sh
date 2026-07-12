#!/usr/bin/env bash
set -euo pipefail
port="${UI_PORT:-8091}"
echo "Inference UI        http://localhost:${port}"
echo "Inference health    http://localhost:${port}/health"
echo "Prediction endpoint http://localhost:${port}/predict"
echo "Temporal UI         http://localhost:8233"
echo "Temporal gRPC       localhost:7233"
