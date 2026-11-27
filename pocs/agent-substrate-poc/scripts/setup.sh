#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require go
require bun
require kind
require kubectl
require podman
podman_env

if [ "$(git -C "$SUBSTRATE" rev-parse HEAD 2>/dev/null || true)" != "$SUBSTRATE_SHA" ]; then
  log "fetching substrate $SUBSTRATE_SHA"
  rm -rf "$SUBSTRATE"
  git init -q "$SUBSTRATE"
  git -C "$SUBSTRATE" fetch -q --depth 1 "$SUBSTRATE_REPO" "$SUBSTRATE_SHA"
  git -C "$SUBSTRATE" checkout -q FETCH_HEAD
fi

log "building kubectl-ate"
(cd "$SUBSTRATE" && go build -o "$BIN/kubectl-ate" ./cmd/kubectl-ate)

log "installing frontend dependencies"
(cd "$ROOT/frontend" && bun install)

if ! cpuset_delegated; then
  printf "%s\n" "ERROR: the podman machine does not delegate the cpuset cgroup controller, gVisor sandboxes cannot start." \
    "Fix it once, this restarts the podman machine and every container in it:" \
    "  podman machine ssh \"printf '[Service]\\nDelegate=cpu cpuset io memory pids\\n' | sudo tee /etc/systemd/system/user@.service.d/delegate.conf\"" \
    "  podman machine stop && podman machine start" >&2
  exit 1
fi

log "raising the kernel keyring quota inside the podman machine"
podman machine ssh "sudo sysctl -q -w kernel.keys.maxkeys=20000 kernel.keys.maxbytes=2000000" || fail "could not raise the keyring quota"

if [ "$(podman inspect -f '{{.State.Running}}' kind-registry 2>/dev/null || true)" != "true" ]; then
  log "starting the local registry"
  podman rm -f kind-registry >/dev/null 2>&1 || true
  podman run -d --restart=always --label created-by=agent-substrate -p "127.0.0.1:$(service_port registry):5000" --network bridge --name kind-registry registry:3 >/dev/null
fi

if ! kind get clusters 2>/dev/null | grep -qx "$KIND_CLUSTER_NAME"; then
  log "creating kind cluster $KIND_CLUSTER_NAME"
  (cd "$SUBSTRATE" && KIND_REGISTRY_PORT="$(service_port registry)" hack/create-kind-cluster.sh)
fi

for node in $(kind get nodes --name "$KIND_CLUSTER_NAME"); do
  podman exec "$node" sysctl -q net.ipv4.conf.all.proxy_arp=1
done

if ! kube -n ate-system get deploy ate-api-server >/dev/null 2>&1; then
  log "installing agent substrate, this takes a while the first time"
  kubectl config use-context "$KCTX" >/dev/null
  (cd "$SUBSTRATE" && go run ./cmd/ate-setup deploy ate-system --credential-provider='{"name":"k8s.io"}' \
    --ingress-allowed-spiffe-ids=spiffe://cluster.local/ns/ate-system/sa/atenet-client)
fi

"$SCRIPTS/deploy-agent.sh"
log "setup done"
