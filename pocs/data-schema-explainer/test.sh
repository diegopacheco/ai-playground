#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
BASE=http://localhost:8080
printf "GET /api/samples\n"
curl -sf $BASE/api/samples >/dev/null && printf "  samples endpoint OK\n"
for s in orders.parquet orders.arrow orders.orc users.avro order.proto iceberg_table.metadata.json delta_table.log.json; do
  curl -sf -X POST "$BASE/api/explain?sample=$s" | python3 -c "import sys,json; d=json.load(sys.stdin); print('  {:32} -> {:24} {} fields'.format('$s', d['format'], len(d['fields'])))"
done
printf "all formats explained\n"
