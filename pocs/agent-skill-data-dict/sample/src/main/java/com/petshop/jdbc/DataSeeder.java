package com.petshop.jdbc;

import org.springframework.boot.CommandLineRunner;
import org.springframework.core.annotation.Order;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Component;

@Component
@Order(2)
public class DataSeeder implements CommandLineRunner {

    private final JdbcTemplate jdbc;

    public DataSeeder(JdbcTemplate jdbc) {
        this.jdbc = jdbc;
    }

    private static final String[] SEED = {
        "INSERT INTO owner (first_name, last_name, email, phone, address) VALUES ('Ada', 'Lovelace', 'ada@petshop.test', '555-0101', '12 Analytical Ave')",
        "INSERT INTO owner (first_name, last_name, email, phone, address) VALUES ('Alan', 'Turing', 'alan@petshop.test', '555-0102', '7 Enigma Road')",
        "INSERT INTO supplier (name, contact_email, phone, country) VALUES ('Acme Pet Foods', 'sales@acme.test', '555-0900', 'US')",
        "INSERT INTO supplier (name, contact_email, phone, country) VALUES ('Happy Tails Supplies', 'hello@happytails.test', '555-0901', 'CA')",
        "INSERT INTO pet (name, species, breed, birth_date, weight_kg, owner_id) VALUES ('Byron', 'Dog', 'Beagle', '2021-05-01', 11.40, 1)",
        "INSERT INTO pet (name, species, breed, birth_date, weight_kg, owner_id) VALUES ('Pixel', 'Cat', 'Tabby', '2020-11-12', 4.20, 1)",
        "INSERT INTO pet (name, species, breed, birth_date, weight_kg, owner_id) VALUES ('Bombe', 'Dog', 'Labrador', '2022-02-20', 27.10, 2)",
        "INSERT INTO product (sku, name, category, price, stock_qty, supplier_id) VALUES ('FD-001', 'Grain-Free Kibble 5kg', 'Food', 39.90, 120, 1)",
        "INSERT INTO product (sku, name, category, price, stock_qty, supplier_id) VALUES ('TY-014', 'Rope Chew Toy', 'Toys', 8.50, 300, 2)",
        "INSERT INTO visit (pet_id, visit_date, reason, cost) VALUES (1, '2024-03-10', 'Annual checkup', 60.00)",
        "INSERT INTO visit (pet_id, visit_date, reason, cost) VALUES (2, '2024-04-02', 'Vaccination follow-up', 45.00)",
        "INSERT INTO appointment (pet_id, staff_name, scheduled_at, service, status, notes) VALUES (1, 'Dr. Grace Hopper', '2024-06-01 09:30:00', 'Grooming', 'BOOKED', 'Nervous around clippers')",
        "INSERT INTO appointment (pet_id, staff_name, scheduled_at, service, status, notes) VALUES (3, 'Dr. Grace Hopper', '2024-06-03 14:00:00', 'Dental', 'COMPLETED', NULL)",
        "INSERT INTO vaccination (pet_id, vaccine_name, administered_on, next_due_on, batch_number) VALUES (1, 'Rabies', '2024-03-10', '2025-03-10', 'RB-7781')",
        "INSERT INTO vaccination (pet_id, vaccine_name, administered_on, next_due_on, batch_number) VALUES (2, 'Feline Distemper', '2024-04-02', '2025-04-02', 'FD-2210')"
    };

    @Override
    public void run(String... args) {
        for (String statement : SEED) {
            jdbc.execute(statement);
        }
    }
}
