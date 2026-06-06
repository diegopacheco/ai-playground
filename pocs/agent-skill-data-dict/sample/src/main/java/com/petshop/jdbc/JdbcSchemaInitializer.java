package com.petshop.jdbc;

import org.springframework.boot.CommandLineRunner;
import org.springframework.core.annotation.Order;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Component;

@Component
@Order(1)
public class JdbcSchemaInitializer implements CommandLineRunner {

    private final JdbcTemplate jdbc;

    public JdbcSchemaInitializer(JdbcTemplate jdbc) {
        this.jdbc = jdbc;
    }

    private static final String CREATE_SUPPLIER = """
            CREATE TABLE IF NOT EXISTS supplier (
                id          BIGINT       AUTO_INCREMENT PRIMARY KEY,
                name        VARCHAR(160) NOT NULL,
                contact_email VARCHAR(160),
                phone       VARCHAR(40),
                country     VARCHAR(80)  NOT NULL DEFAULT 'US',
                active      BOOLEAN      NOT NULL DEFAULT TRUE
            )
            """;

    private static final String CREATE_PRODUCT = """
            CREATE TABLE IF NOT EXISTS product (
                id          BIGINT        AUTO_INCREMENT PRIMARY KEY,
                sku         VARCHAR(40)   NOT NULL UNIQUE,
                name        VARCHAR(200)  NOT NULL,
                category    VARCHAR(80)   NOT NULL,
                price       NUMERIC(10,2) NOT NULL,
                stock_qty   INT           NOT NULL DEFAULT 0,
                supplier_id BIGINT        NOT NULL,
                CONSTRAINT fk_product_supplier FOREIGN KEY (supplier_id) REFERENCES supplier (id)
            )
            """;

    @Override
    public void run(String... args) {
        jdbc.execute(CREATE_SUPPLIER);
        jdbc.execute(CREATE_PRODUCT);
    }
}
