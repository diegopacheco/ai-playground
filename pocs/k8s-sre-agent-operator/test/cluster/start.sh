#!/bin/bash
set -e

CLUSTER_NAME="kovalski-test"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

kind create cluster --name "$CLUSTER_NAME" --config "$SCRIPT_DIR/kind-config.yaml"
kubectl cluster-info --context "kind-$CLUSTER_NAME"

kubectl apply -f https://raw.githubusercontent.com/metallb/metallb/v0.14.9/config/manifests/metallb-native.yaml

echo "Waiting for MetalLB controller to be ready..."
kubectl wait --namespace metallb-system --for=condition=ready pod --selector=app=metallb --timeout=120s

SUBNET=$(podman network inspect kind | grep -oP '"subnet":\s*"\K[0-9]+\.[0-9]+\.[0-9]+' | head -1)
if [ -z "$SUBNET" ]; then
    SUBNET="172.18.0"
fi

kubectl apply -f - <<EOF
apiVersion: metallb.io/v1beta1
kind: IPAddressPool
metadata:
  name: kovalski-pool
  namespace: metallb-system
spec:
  addresses:
    - ${SUBNET}.200-${SUBNET}.250
---
apiVersion: metallb.io/v1beta1
kind: L2Advertisement
metadata:
  name: kovalski-l2
  namespace: metallb-system
EOF

echo ""
echo "Cluster '$CLUSTER_NAME' is ready with MetalLB."
echo "Run: kovalski deploy"
