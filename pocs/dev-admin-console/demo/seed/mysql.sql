SET SESSION cte_max_recursion_depth = 100000;

DROP TABLE IF EXISTS payments;
DROP TABLE IF EXISTS invoices;

CREATE TABLE invoices (
    id INT AUTO_INCREMENT PRIMARY KEY,
    number VARCHAR(32) NOT NULL UNIQUE,
    customer_email VARCHAR(255) NOT NULL,
    amount_cents INT NOT NULL,
    status ENUM('draft','sent','paid','void') NOT NULL DEFAULT 'draft',
    issued_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE payments (
    id INT AUTO_INCREMENT PRIMARY KEY,
    invoice_id INT NOT NULL,
    method VARCHAR(32) NOT NULL,
    amount_cents INT NOT NULL,
    paid_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (invoice_id) REFERENCES invoices(id)
);

INSERT INTO invoices (number, customer_email, amount_cents, status)
WITH RECURSIVE seq(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM seq WHERE n < 800)
SELECT CONCAT('INV-', LPAD(n, 6, '0')), CONCAT('customer', n % 500, '@example.com'),
       1000 + (n * 41) % 90000,
       ELT(1 + (n % 4), 'draft', 'sent', 'paid', 'void')
FROM seq;

INSERT INTO payments (invoice_id, method, amount_cents)
WITH RECURSIVE seq(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM seq WHERE n < 1200)
SELECT 1 + (n % 800), ELT(1 + (n % 3), 'card', 'pix', 'transfer'), 500 + (n * 17) % 50000
FROM seq;

DROP USER IF EXISTS 'console_reader'@'%';
CREATE USER 'console_reader'@'%' IDENTIFIED BY 'console_reader';
GRANT SELECT ON shop.* TO 'console_reader'@'%';
FLUSH PRIVILEGES;
