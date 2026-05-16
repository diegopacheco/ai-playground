#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "OPENAI_API_KEY is required"
  exit 1
fi

CLUSTER_NAME=agentgateway
NS=agentgateway-system

export KIND_EXPERIMENTAL_PROVIDER=podman

if command -v podman >/dev/null 2>&1; then
  if ! podman machine list --format '{{.Name}} {{.Running}}' | grep -q ' true$'; then
    if ! podman machine list --format '{{.Name}}' | grep -q .; then
      podman machine init
    fi
    podman machine start
    until podman info >/dev/null 2>&1; do sleep 1; done
  fi
fi

if ! kind get clusters | grep -qx "$CLUSTER_NAME"; then
  kind create cluster --config k8s/kind-cluster.yaml
fi

kubectl config use-context "kind-$CLUSTER_NAME"

kubectl apply --server-side --force-conflicts -f https://github.com/kubernetes-sigs/gateway-api/releases/download/v1.5.0/standard-install.yaml

helm upgrade -i agentgateway-crds oci://cr.agentgateway.dev/charts/agentgateway-crds \
  --create-namespace --namespace "$NS" \
  --version v1.2.0 \
  --set controller.image.pullPolicy=Always

helm upgrade -i agentgateway oci://cr.agentgateway.dev/charts/agentgateway \
  --namespace "$NS" \
  --version v1.2.0 \
  --set controller.image.pullPolicy=Always \
  --wait

until kubectl get deployment -n "$NS" agentgateway >/dev/null 2>&1; do sleep 1; done
kubectl rollout status -n "$NS" deployment/agentgateway --timeout=120s

kubectl create secret generic openai-secret -n "$NS" \
  --from-literal=Authorization="Bearer $OPENAI_API_KEY" \
  --dry-run=client -o yaml | kubectl apply -f -

if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
  kubectl create secret generic anthropic-secret -n "$NS" \
    --from-literal=Authorization="$ANTHROPIC_API_KEY" \
    --dry-run=client -o yaml | kubectl apply -f -
fi

kubectl apply -f k8s/gateway.yaml
kubectl apply -f k8s/openai-backend.yaml
if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
  kubectl apply -f k8s/anthropic-backend.yaml
fi

until kubectl get deployment -n "$NS" agentgateway-proxy >/dev/null 2>&1; do sleep 1; done
kubectl rollout status -n "$NS" deployment/agentgateway-proxy --timeout=180s

PF_LOG=/tmp/agentgateway-pf.log
pkill -f "port-forward .*agentgateway-proxy.*8080:80" 2>/dev/null || true
nohup kubectl port-forward -n "$NS" deployment/agentgateway-proxy 8080:80 >"$PF_LOG" 2>&1 &
echo $! > /tmp/agentgateway-pf.pid

until curl -sf -o /dev/null -m 1 http://localhost:8080/ -H "host: any" || [ $? -eq 22 ]; do sleep 1; done

echo "agentgateway is up on http://localhost:8080"
echo "run ./ui.sh for the admin UI"
echo "run ./app-cli.sh '<prompt>' to use the OpenAI SDK"
echo "run ./claude-agentgateway.sh to route Claude Code through agentgateway"
