#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

podman_env
KO="$(ko_bin)"
PLATFORM="$(node_platform)"
REPO="localhost:$(service_port registry)"

log "building the worker image and applying the worker pool"
(cd "$SUBSTRATE" && KO_DOCKER_REPO="$REPO" "$KO" resolve --base-import-paths --platform="$PLATFORM" -f "$ROOT/deploy/workerpool.yaml") | kube apply -f - >/dev/null

log "building the memory agent image"
IMAGE="$(cd "$ROOT/backend" && KO_DOCKER_REPO="$REPO" KO_DEFAULTBASEIMAGE="$AGENT_BASE_IMAGE" "$KO" build ./cmd/agent --base-import-paths --platform="$PLATFORM" 2>"$LOGS/ko-agent.log")"
[ -n "$IMAGE" ] || fail "agent image build failed, see $LOGS/ko-agent.log"
log "agent image $IMAGE"

ate get atespace "$ATESPACE" >/dev/null 2>&1 || ate create atespace "$ATESPACE" >/dev/null

template_field() {
  ate get actor-template "$TEMPLATE" -a "$ATESPACE" -o json 2>/dev/null | jq -r "$1 // empty" || true
}

CURRENT="$(template_field '.containers[0].image')"
FAILED="$(template_field '.status.goldenSnapshotStatus.errorMessage')"
if [ "$CURRENT" != "$IMAGE" ] || [ -n "$FAILED" ]; then
  if [ -n "$CURRENT" ]; then
    log "recreating template $TEMPLATE"
    ate delete actor-template "$TEMPLATE" -a "$ATESPACE" >/dev/null || fail "delete the actors of $TEMPLATE first, templates are immutable"
  fi
  sed "s|\${AGENT_IMAGE}|$IMAGE|" "$ROOT/deploy/template.yaml.tmpl" | ate create actor-template -f - >/dev/null
  log "template $TEMPLATE created"
fi

tries=60
while [ "$tries" -gt 0 ]; do
  golden="$(template_field '.status.goldenSnapshotStatus.goldenTag.name')"
  [ -n "$golden" ] && break
  failed="$(template_field '.status.goldenSnapshotStatus.errorMessage')"
  [ -z "$failed" ] || fail "golden snapshot failed: $failed"
  sleep 1
  tries=$((tries - 1))
done
[ -n "$golden" ] || fail "golden snapshot of $TEMPLATE is not ready yet, run scripts/deploy-agent.sh again"
log "template $TEMPLATE ready with golden snapshot $golden"
