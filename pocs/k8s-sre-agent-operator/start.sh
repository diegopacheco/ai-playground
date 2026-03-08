#!/bin/bash
set -e

CLUSTER_NAME="sre-agent-cluster"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

kind create cluster --name "$CLUSTER_NAME" --config "$SCRIPT_DIR/kind-config.yaml"

kubectl cluster-info --context "kind-$CLUSTER_NAME"

cd "$SCRIPT_DIR/ui"
bun install
bun run build
rm -rf "$SCRIPT_DIR/operator/ui-dist"
cp -r "$SCRIPT_DIR/ui/dist" "$SCRIPT_DIR/operator/ui-dist"

cd "$SCRIPT_DIR/operator"
podman build -t sre-agent-operator:latest -f Containerfile .

podman save sre-agent-operator:latest -o /tmp/sre-agent-operator.tar
kind load image-archive /tmp/sre-agent-operator.tar --name "$CLUSTER_NAME"
rm -f /tmp/sre-agent-operator.tar

for f in "$SCRIPT_DIR/specs/"*.yaml; do
    echo "Applying $f"
    kubectl apply -f "$f"
done

echo "Waiting for sre-agent-operator pod to be ready..."
while true; do
    READY=$(kubectl get pods -l app=sre-agent-operator -o jsonpath='{.items[0].status.conditions[?(@.type=="Ready")].status}' 2>/dev/null)
    if [ "$READY" = "True" ]; then
        break
    fi
    sleep 1
done

kubectl port-forward svc/sre-agent-operator 30080:8080 &
PF_PID=$!
echo "$PF_PID" > /tmp/sre-agent-portforward.pid
sleep 1

echo ""
echo "SRE Agent Operator is running."
echo "  GET  /logs -> kovalski logs"
echo "  POST /fix  -> kovalski fix"
echo "  Port-forward PID: $PF_PID"
echo ""
kubectl get pods -A
