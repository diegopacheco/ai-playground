#!/bin/bash
set -e

CLUSTER_NAME="kovalski-test"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

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

cd "$SCRIPT_DIR/../k8s"
podman build -t test-app:latest -f Containerfile .
podman save test-app:latest -o /tmp/test-app.tar
kind load image-archive /tmp/test-app.tar --name "$CLUSTER_NAME"
rm -f /tmp/test-app.tar

cd "$PROJECT_ROOT/operator"
podman build -t sre-agent-operator:latest -f Containerfile .
podman save sre-agent-operator:latest -o /tmp/sre-agent-operator.tar
kind load image-archive /tmp/sre-agent-operator.tar --name "$CLUSTER_NAME"
rm -f /tmp/sre-agent-operator.tar

echo ""
echo "Cluster '$CLUSTER_NAME' is ready with MetalLB."
echo "Run: kovalski deploy"
echo "Run: kovalski k8s --name test-app --image test-app:latest --port 8080"
