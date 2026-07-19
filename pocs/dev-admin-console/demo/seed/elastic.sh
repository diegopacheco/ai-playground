#!/usr/bin/env bash
set -euo pipefail
base="${1:-http://localhost:9200}"
curl -fsS -X DELETE "$base/products" > /dev/null 2>&1 || true
curl -fsS -X PUT "$base/products" -H 'Content-Type: application/json' -d '{
  "settings": { "index.max_result_window": 200 },
  "mappings": {
    "properties": {
      "sku": { "type": "keyword" },
      "name": { "type": "text" },
      "price_cents": { "type": "integer" },
      "in_stock": { "type": "boolean" },
      "category": { "type": "keyword" },
      "supplier": {
        "properties": {
          "name": { "type": "keyword" },
          "country": { "type": "keyword" }
        }
      }
    }
  }
}' > /dev/null
batch=/tmp/elastic-bulk.ndjson
: > "$batch"
for index in $(seq 1 ${DOCS:-1000}); do
  category=$(( index % 6 ))
  printf '{"index":{"_index":"products","_id":"%s"}}\n' "$index" >> "$batch"
  printf '{"sku":"SKU-%04d","name":"Product %d","price_cents":%d,"in_stock":%s,"category":"cat-%d","supplier":{"name":"Supplier %d","country":"BR"}}\n' \
    "$index" "$index" "$(( 500 + index * 13 % 50000 ))" "$([ $(( index % 3 )) -eq 0 ] && echo false || echo true)" "$category" "$(( index % 20 ))" >> "$batch"
done
split -l 2000 "$batch" /tmp/elastic-part-
for part in /tmp/elastic-part-*; do
  curl -fsS -X POST "$base/_bulk" -H 'Content-Type: application/x-ndjson' --data-binary "@$part" > /dev/null
done
rm -f /tmp/elastic-part-* "$batch"
curl -fsS -X POST "$base/products/_refresh" > /dev/null
curl -fsS "$base/products/_count"
echo
