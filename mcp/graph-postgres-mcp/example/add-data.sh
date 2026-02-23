#!/bin/bash
cd "$(dirname "$0")"

if ! podman exec graphmcp-example-postgres pg_isready -U exampleuser -d exampledb > /dev/null 2>&1; then
  echo "PostgreSQL is not running. Start it first with ./run-pgsql.sh"
  exit 1
fi

podman exec -i graphmcp-example-postgres psql -U exampleuser -d exampledb <<'SQL'
DROP TABLE IF EXISTS shipments;
CREATE TABLE shipments (
    id SERIAL PRIMARY KEY,
    order_id INTEGER NOT NULL REFERENCES orders(id),
    tracking_code TEXT NOT NULL UNIQUE,
    carrier TEXT NOT NULL,
    status TEXT NOT NULL,
    shipped_at TIMESTAMPTZ DEFAULT NOW()
);
INSERT INTO shipments (order_id, tracking_code, carrier, status) VALUES
    (1, 'TRK-1001', 'UPS', 'shipped'),
    (2, 'TRK-1002', 'FedEx', 'in_transit'),
    (3, 'TRK-1003', 'DHL', 'delivered'),
    (4, 'TRK-1004', 'UPS', 'processing'),
    (5, 'TRK-1005', 'FedEx', 'shipped');
SELECT * FROM shipments ORDER BY id;
SQL
