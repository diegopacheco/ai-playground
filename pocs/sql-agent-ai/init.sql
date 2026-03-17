CREATE TABLE IF NOT EXISTS states (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    abbreviation VARCHAR(2) NOT NULL
);

CREATE TABLE IF NOT EXISTS salesmen (
    id SERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    email VARCHAR(200) NOT NULL,
    state_id INTEGER REFERENCES states(id),
    hire_date DATE NOT NULL
);

CREATE TABLE IF NOT EXISTS products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    category VARCHAR(100) NOT NULL,
    price DECIMAL(10,2) NOT NULL
);

CREATE TABLE IF NOT EXISTS sales (
    id SERIAL PRIMARY KEY,
    salesman_id INTEGER REFERENCES salesmen(id),
    product_id INTEGER REFERENCES products(id),
    state_id INTEGER REFERENCES states(id),
    quantity INTEGER NOT NULL,
    total_amount DECIMAL(12,2) NOT NULL,
    sale_date DATE NOT NULL
);

INSERT INTO states (name, abbreviation) VALUES
('California', 'CA'), ('Texas', 'TX'), ('New York', 'NY'),
('Florida', 'FL'), ('Illinois', 'IL'), ('Pennsylvania', 'PA'),
('Ohio', 'OH'), ('Georgia', 'GA'), ('North Carolina', 'NC'),
('Michigan', 'MI'), ('Washington', 'WA'), ('Arizona', 'AZ'),
('Massachusetts', 'MA'), ('Tennessee', 'TN'), ('Indiana', 'IN');

INSERT INTO products (name, category, price) VALUES
('Laptop Pro 15', 'Electronics', 1299.99),
('Wireless Mouse', 'Electronics', 29.99),
('USB-C Hub', 'Electronics', 49.99),
('Mechanical Keyboard', 'Electronics', 149.99),
('Monitor 27inch', 'Electronics', 399.99),
('Standing Desk', 'Furniture', 599.99),
('Office Chair', 'Furniture', 349.99),
('Desk Lamp', 'Furniture', 79.99),
('Webcam HD', 'Electronics', 89.99),
('Headphones NC', 'Electronics', 249.99),
('Tablet 10inch', 'Electronics', 449.99),
('Phone Case', 'Accessories', 19.99),
('Screen Protector', 'Accessories', 9.99),
('Laptop Bag', 'Accessories', 59.99),
('Power Bank', 'Electronics', 39.99),
('Bluetooth Speaker', 'Electronics', 79.99),
('Smart Watch', 'Electronics', 299.99),
('Drawing Tablet', 'Electronics', 199.99),
('Printer Inkjet', 'Electronics', 179.99),
('Paper Shredder', 'Office', 129.99);

INSERT INTO salesmen (name, email, state_id, hire_date) VALUES
('Alice Johnson', 'alice@sales.com', 1, '2022-01-15'),
('Bob Smith', 'bob@sales.com', 2, '2021-06-20'),
('Carol Davis', 'carol@sales.com', 3, '2023-03-10'),
('David Wilson', 'david@sales.com', 4, '2022-08-05'),
('Eve Martinez', 'eve@sales.com', 5, '2021-11-30'),
('Frank Brown', 'frank@sales.com', 6, '2023-01-22'),
('Grace Lee', 'grace@sales.com', 7, '2022-04-18'),
('Henry Taylor', 'henry@sales.com', 8, '2021-09-12'),
('Ivy Anderson', 'ivy@sales.com', 9, '2023-06-01'),
('Jack Thomas', 'jack@sales.com', 10, '2022-12-08'),
('Karen White', 'karen@sales.com', 11, '2021-03-25'),
('Leo Harris', 'leo@sales.com', 12, '2023-02-14'),
('Maria Clark', 'maria@sales.com', 13, '2022-07-19'),
('Nathan Lewis', 'nathan@sales.com', 14, '2021-05-11'),
('Olivia Walker', 'olivia@sales.com', 15, '2023-04-28');

INSERT INTO sales (salesman_id, product_id, state_id, quantity, total_amount, sale_date)
SELECT
    (random() * 14 + 1)::int AS salesman_id,
    (random() * 19 + 1)::int AS product_id,
    (random() * 14 + 1)::int AS state_id,
    (random() * 20 + 1)::int AS quantity,
    round((random() * 5000 + 50)::numeric, 2) AS total_amount,
    DATE '2025-01-01' + (random() * 365)::int AS sale_date
FROM generate_series(1, 500);

INSERT INTO sales (salesman_id, product_id, state_id, quantity, total_amount, sale_date)
SELECT
    (random() * 14 + 1)::int AS salesman_id,
    (random() * 19 + 1)::int AS product_id,
    (random() * 14 + 1)::int AS state_id,
    (random() * 20 + 1)::int AS quantity,
    round((random() * 5000 + 50)::numeric, 2) AS total_amount,
    DATE '2026-01-01' + (random() * 75)::int AS sale_date
FROM generate_series(1, 200);
