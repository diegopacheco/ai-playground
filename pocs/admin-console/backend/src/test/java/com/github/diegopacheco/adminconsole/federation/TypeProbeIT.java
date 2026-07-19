package com.github.diegopacheco.adminconsole.federation;

import com.github.diegopacheco.adminconsole.engine.EngineRegistry;
import com.github.diegopacheco.adminconsole.engine.PageRequest;
import com.github.diegopacheco.adminconsole.engine.QueryResult;
import com.github.diegopacheco.adminconsole.project.ConnectionConfig;
import com.github.diegopacheco.adminconsole.project.ConnectionKind;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

@Tag("integration-test")
@SpringBootTest
class TypeProbeIT {
    @Autowired
    private EngineRegistry engines;

    private void probe(String label, ConnectionConfig config, String statement) {
        QueryResult result = engines.of(config.kind()).query(config, statement, PageRequest.first(2));
        System.out.println("PROBE " + label + " columns=" + result.columns());
        for (Map<String, Object> row : result.rows()) {
            StringBuilder line = new StringBuilder("PROBE " + label + " ");
            row.forEach((key, value) -> line.append(key).append('=')
                    .append(value == null ? "null" : value.getClass().getSimpleName() + "(" + value + ") "));
            System.out.println(line);
        }
    }

    @Test
    void printsJavaTypesForEveryJoinKeyColumn() {
        probe("postgres", new ConnectionConfig(1L, 1L, "demo-postgres", ConnectionKind.POSTGRES, "localhost", 5432,
                "shop", "public", null, "console_reader", "console_reader", null, "t"),
                "SELECT id, customer_id, total_cents, placed_at FROM orders");
        probe("postgres-items", new ConnectionConfig(1L, 1L, "demo-postgres", ConnectionKind.POSTGRES, "localhost",
                5432, "shop", "public", null, "console_reader", "console_reader", null, "t"),
                "SELECT id, sku, unit_cents FROM order_items");
        probe("mysql", new ConnectionConfig(2L, 1L, "demo-mysql", ConnectionKind.MYSQL, "localhost", 3306, "shop",
                null, null, "console_reader", "console_reader", null, "t"),
                "SELECT id, customer_email, amount_cents, issued_at, status FROM invoices");
        probe("cassandra", new ConnectionConfig(3L, 1L, "demo-cassandra", ConnectionKind.CASSANDRA, "localhost", 9042,
                null, "shop", "datacenter1", null, null, null, "t"),
                "SELECT session_id, customer_id, started_at FROM sessions");
        probe("elastic", new ConnectionConfig(7L, 1L, "demo-elasticsearch", ConnectionKind.ELASTICSEARCH, "localhost",
                9200, null, null, null, null, null, null, "t"), "GET /products/_search");
        probe("kafka", new ConnectionConfig(6L, 1L, "demo-kafka", ConnectionKind.KAFKA, "localhost", 9092, null,
                null, null, null, null, null, "t"), "consume orders.events --limit 2");
        probe("etcd", new ConnectionConfig(5L, 1L, "demo-etcd", ConnectionKind.ETCD, "localhost", 2379, null, null,
                null, null, null, null, "t"), "get /config --prefix");
        probe("redis", new ConnectionConfig(4L, 1L, "demo-redis", ConnectionKind.REDIS, "localhost", 6379, null, null,
                null, null, null, null, "t"), "HGETALL session:abc123");
    }
}
