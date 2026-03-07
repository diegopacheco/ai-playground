#!/bin/bash
set -e

CLUSTER_NAME="sre-agent-cluster"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

kind create cluster --name "$CLUSTER_NAME" --config "$SCRIPT_DIR/kind-config.yaml"

kubectl cluster-info --context "kind-$CLUSTER_NAME"

cd "$SCRIPT_DIR/operator"
podman build -t sre-agent-operator:latest -f Containerfile .

podman save sre-agent-operator:latest -o /tmp/sre-agent-operator.tar
kind load image-archive /tmp/sre-agent-operator.tar --name "$CLUSTER_NAME"
rm -f /tmp/sre-agent-operator.tar

if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "WARNING: ANTHROPIC_API_KEY not set. /fix endpoint will not work."
    kubectl create secret generic anthropic-secret --from-literal=api-key="not-set" 2>/dev/null || true
else
    kubectl create secret generic anthropic-secret --from-literal=api-key="$ANTHROPIC_API_KEY" 2>/dev/null || true
fi

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
