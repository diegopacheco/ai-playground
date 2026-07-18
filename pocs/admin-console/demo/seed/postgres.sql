DROP TABLE IF EXISTS order_items;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS customers;

CREATE TABLE customers (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    full_name VARCHAR(255) NOT NULL,
    country CHAR(2) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    status VARCHAR(16) NOT NULL,
    total_cents INTEGER NOT NULL,
    placed_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE order_items (
    id SERIAL PRIMARY KEY,
    order_id INTEGER NOT NULL REFERENCES orders(id),
    sku VARCHAR(64) NOT NULL,
    quantity INTEGER NOT NULL,
    unit_cents INTEGER NOT NULL
);

INSERT INTO customers (email, full_name, country)
SELECT 'customer' || n || '@example.com', 'Customer ' || n,
       (ARRAY['BR','US','DE','JP','PT'])[1 + (n % 5)]
FROM generate_series(1, 500) AS n;

INSERT INTO orders (customer_id, status, total_cents)
SELECT 1 + (n % 500), (ARRAY['placed','paid','shipped','cancelled'])[1 + (n % 4)], 1000 + (n * 37) % 90000
FROM generate_series(1, 2000) AS n;

INSERT INTO order_items (order_id, sku, quantity, unit_cents)
SELECT 1 + (n % 2000), 'SKU-' || lpad(((n % 300))::text, 4, '0'), 1 + (n % 5), 500 + (n * 13) % 20000
FROM generate_series(1, 6000) AS n;

CREATE OR REPLACE VIEW order_totals AS
SELECT o.id AS order_id, c.email, o.status, o.total_cents
FROM orders o JOIN customers c ON c.id = o.customer_id;

DROP USER IF EXISTS console_reader;
CREATE USER console_reader WITH PASSWORD 'console_reader';
GRANT CONNECT ON DATABASE shop TO console_reader;
GRANT USAGE ON SCHEMA public TO console_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO console_reader;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO console_reader;
