#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require curl
FRONTEND="$(service_url frontend)"
port_up "$(service_port frontend)" || fail "app is not running, run scripts/start-all.sh first"
log "uploading samples/table-page.pdf through the UI proxy"
body="$(curl -sf -X POST "$FRONTEND/api/conversions?ocr=false" -H "X-Filename: table-page.pdf" --data-binary "@$ROOT/samples/table-page.pdf")" || fail "upload failed"
printf "%s" "$body" | grep -q '"status": "success"' || fail "conversion did not succeed"
printf "%s" "$body" | grep -q '"tables": \[{' || fail "no table extracted from the sample pdf"
id="$(printf "%s" "$body" | sed -E 's/^\{"id": "([a-z0-9]+)".*/\1/')"
curl -sf "$FRONTEND/api/conversions" | grep -q "$id" || fail "conversion $id missing from history"
curl -s -o /dev/null -w "%{http_code}" -X POST "$FRONTEND/api/conversions" -H "X-Filename: virus.exe" --data-binary "x" | grep -q 415 || fail "unsupported file was not rejected"
log "e2e passed for conversion $id"
