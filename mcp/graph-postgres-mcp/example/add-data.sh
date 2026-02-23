#!/bin/bash
cd "$(dirname "$0")"

PG_CONTAINER="${PG_CONTAINER_NAME:-graphmcp-postgres}"
PG_USER_VALUE="${PG_USER:-graphmcp}"
PG_DATABASE_VALUE="${PG_DATABASE:-graphmcpdb}"

if ! podman exec "$PG_CONTAINER" pg_isready -U "$PG_USER_VALUE" -d "$PG_DATABASE_VALUE" > /dev/null 2>&1; then
  echo "PostgreSQL is not running. Start it first with ../start.sh"
  exit 1
fi

podman exec -i "$PG_CONTAINER" psql -U "$PG_USER_VALUE" -d "$PG_DATABASE_VALUE" <<'SQL'
DROP TABLE IF EXISTS shipments;
CREATE TABLE shipments (
    id SERIAL PRIMARY KEY,
    tracking_code TEXT NOT NULL UNIQUE,
    carrier TEXT NOT NULL,
    status TEXT NOT NULL,
    destination_city TEXT NOT NULL,
    shipped_at TIMESTAMPTZ DEFAULT NOW()
);
INSERT INTO shipments (tracking_code, carrier, status, destination_city) VALUES
    ('TRK-1001', 'UPS', 'shipped', 'Seattle'),
    ('TRK-1002', 'FedEx', 'in_transit', 'Portland'),
    ('TRK-1003', 'DHL', 'delivered', 'Denver'),
    ('TRK-1004', 'UPS', 'processing', 'Austin'),
    ('TRK-1005', 'FedEx', 'shipped', 'Chicago');
SELECT * FROM shipments ORDER BY id;
SQL
