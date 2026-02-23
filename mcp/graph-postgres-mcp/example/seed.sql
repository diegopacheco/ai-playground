INSERT INTO salesmen (name, region, commission_rate) VALUES
    ('Carlos Silva', 'South', 5.50),
    ('Maria Santos', 'North', 6.00),
    ('John Parker', 'West', 5.75),
    ('Ana Costa', 'East', 6.25);

INSERT INTO buyers (name, email, city) VALUES
    ('TechCorp', 'buy@techcorp.com', 'Seattle'),
    ('DataWorks', 'orders@dataworks.io', 'Portland'),
    ('CloudInc', 'purchasing@cloudinc.com', 'Denver'),
    ('NetSoft', 'buy@netsoft.com', 'Austin'),
    ('DevHouse', 'orders@devhouse.co', 'Chicago');

INSERT INTO products (name, description, price, stock) VALUES
    ('Laptop Pro 15', '15 inch laptop with 32GB RAM', 1899.99, 50),
    ('Wireless Mouse', 'Ergonomic bluetooth mouse', 49.99, 200),
    ('USB-C Hub', '7-port USB-C docking station', 89.99, 150),
    ('Mechanical Keyboard', 'Cherry MX Blue switches', 129.99, 80),
    ('Monitor 27', '4K IPS 27 inch display', 549.99, 30),
    ('Webcam HD', '1080p webcam with microphone', 79.99, 120);

INSERT INTO orders (buyer_id, salesman_id, product_id, quantity, total) VALUES
    (1, 1, 1, 10, 18999.90),
    (1, 1, 5, 20, 10999.80),
    (2, 2, 2, 50, 2499.50),
    (2, 2, 4, 15, 1949.85),
    (3, 3, 3, 30, 2699.70),
    (3, 1, 6, 25, 1999.75),
    (4, 4, 1, 5, 9499.95),
    (4, 4, 2, 100, 4999.00),
    (5, 3, 5, 10, 5499.90),
    (5, 2, 3, 40, 3599.60);
