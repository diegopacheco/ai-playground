#!/bin/bash
set -e

CLUSTER_NAME="sre-agent-cluster"
if [ -f /tmp/sre-agent-portforward.pid ]; then
    kill "$(cat /tmp/sre-agent-portforward.pid)" 2>/dev/null || true
    rm -f /tmp/sre-agent-portforward.pid
fi
kind delete cluster --name "$CLUSTER_NAME"
echo "Cluster $CLUSTER_NAME deleted."
