#!/bin/bash
set -e

CLUSTER_NAME="sre-agent-cluster"
kind delete cluster --name "$CLUSTER_NAME"
echo "Cluster $CLUSTER_NAME deleted."
