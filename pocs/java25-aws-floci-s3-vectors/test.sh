#!/usr/bin/env bash
set -euo pipefail
app_port="${APP_PORT:-8082}"
base_url="${BASE_URL:-http://localhost:$app_port}"
request() {
  method="$1"
  path="$2"
  body="${3:-}"
  if [ -n "$body" ]; then
    curl -fsS -X "$method" "$base_url$path" -H "Content-Type: application/json" -d "$body"
  else
    curl -fsS -X "$method" "$base_url$path"
  fi
}
echo "Status"
request GET /api/v1/status
echo
echo "Store vectors"
request PUT /api/v1/vectors/java '{"values":[1.0,0.0,0.0]}'
echo
request PUT /api/v1/vectors/rust '{"values":[0.8,0.2,0.0]}'
echo
request PUT /api/v1/vectors/cooking '{"values":[0.0,0.0,1.0]}'
echo
echo "Get vectors"
java_vector="$(request GET /api/v1/vectors/java)"
rust_vector="$(request GET /api/v1/vectors/rust)"
cooking_vector="$(request GET /api/v1/vectors/cooking)"
echo "$java_vector"
echo "$rust_vector"
echo "$cooking_vector"
case "$java_vector$rust_vector$cooking_vector" in
  *java*rust*cooking*) ;;
  *) exit 1 ;;
esac
echo "Nearest vectors"
matches="$(request POST /api/v1/vectors/search '{"values":[0.9,0.1,0.0],"topK":2}')"
echo "$matches"
case "$matches" in
  *java*rust*|*rust*java*) ;;
  *) exit 1 ;;
esac
echo "Delete vectors"
request DELETE /api/v1/vectors/java
request DELETE /api/v1/vectors/rust
request DELETE /api/v1/vectors/cooking
echo
echo "All REST checks passed"
