#!/bin/bash
set -e

CLUSTER_NAME="test-app-cluster"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

kind create cluster --name "$CLUSTER_NAME"
kubectl cluster-info --context "kind-$CLUSTER_NAME"

cd "$SCRIPT_DIR"
podman build -t localhost/test-app:latest -f Containerfile .
podman save localhost/test-app:latest -o /tmp/test-app.tar
kind load image-archive /tmp/test-app.tar --name "$CLUSTER_NAME"
rm -f /tmp/test-app.tar

for f in "$SCRIPT_DIR/specs/"*.yaml; do
echo "Applying $f"
kubectl apply -f "$f"
done

echo "Waiting for test-app pod to be ready..."
while true; do
READY=$(kubectl get pods -l app=test-app -o jsonpath='{.items[0].status.conditions[?(@.type=="Ready")].status}' 2>/dev/null)
if [ "$READY" = "True" ]; then
break
fi
sleep 1
done

echo "test-app is running."
kubectl get pods -A
