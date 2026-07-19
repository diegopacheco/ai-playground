package com.github.diegopacheco.devadminconsole.engine.jdbc;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.github.diegopacheco.devadminconsole.engine.PageRequest;
import com.github.diegopacheco.devadminconsole.engine.QueryResult;
import com.github.diegopacheco.devadminconsole.engine.ReadOnlyViolation;
import com.github.diegopacheco.devadminconsole.engine.SchemaNode;
import com.github.diegopacheco.devadminconsole.project.ConnectionConfig;
import com.github.diegopacheco.devadminconsole.project.ConnectionKind;
import java.util.List;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

@Tag("integration-test")
@SpringBootTest
class JdbcEngineIT {
    @Autowired
    private PostgresEngine postgres;
    @Autowired
    private MysqlEngine mysql;

    private final ConnectionConfig postgresConfig = new ConnectionConfig(9001L, 1L, "demo-postgres",
            ConnectionKind.POSTGRES, "localhost", 5432, "shop", "public", null, "console_reader", "console_reader",
            null, "tester");

    private final ConnectionConfig mysqlConfig = new ConnectionConfig(9002L, 1L, "demo-mysql",
            ConnectionKind.MYSQL, "localhost", 3306, "shop", null, null, "console_reader", "console_reader",
            null, "tester");

    @Test
    void listsPostgresTablesWithTheirColumnsSoTheLeftPanelCanFold() {
        List<SchemaNode> tables = postgres.schema(postgresConfig);
        assertThat(tables).extracting(SchemaNode::name).contains("customers", "orders", "order_items");
        SchemaNode customers = tables.stream().filter(node -> node.name().equals("customers")).findFirst().orElseThrow();
        assertThat(customers.children()).extracting(SchemaNode::name).contains("id", "email", "full_name", "country");
        assertThat(customers.children()).extracting(SchemaNode::detail).anyMatch(detail -> detail.contains("not null"));
    }

    @Test
    void listsMysqlTablesWithTheirColumns() {
        List<SchemaNode> tables = mysql.schema(mysqlConfig);
        assertThat(tables).extracting(SchemaNode::name).contains("invoices", "payments");
        SchemaNode invoices = tables.stream().filter(node -> node.name().equals("invoices")).findFirst().orElseThrow();
        assertThat(invoices.children()).extracting(SchemaNode::name).contains("number", "customer_email", "status");
    }

    @Test
    void runsASelectAgainstPostgresAndReturnsColumnsAndRows() {
        QueryResult result = postgres.query(postgresConfig, "SELECT id, email FROM customers ORDER BY id",
                PageRequest.first(10));
        assertThat(result.columns()).containsExactly("id", "email");
        assertThat(result.rows()).hasSize(10);
        assertThat(result.rows().getFirst().get("email")).isEqualTo("customer1@example.com");
    }

    @Test
    void runsASelectAgainstMysql() {
        QueryResult result = mysql.query(mysqlConfig, "SELECT number, status FROM invoices ORDER BY id",
                PageRequest.first(5));
        assertThat(result.columns()).containsExactly("number", "status");
        assertThat(result.rows()).hasSize(5);
        assertThat(result.rows().getFirst().get("number")).isEqualTo("INV-000001");
    }

    @Test
    void pagesForwardWithoutGapsOrRepeatsBecauseSilentlyWrongRowsWouldMisleadAnOperator() {
        PageRequest first = PageRequest.first(10);
        QueryResult pageOne = postgres.query(postgresConfig, "SELECT id FROM customers ORDER BY id", first);
        assertThat(pageOne.hasMore()).isTrue();
        QueryResult pageTwo = postgres.query(postgresConfig, "SELECT id FROM customers ORDER BY id",
                new PageRequest(10, pageOne.nextCursor(), 2));
        List<String> firstIds = pageOne.rows().stream().map(row -> String.valueOf(row.get("id"))).toList();
        List<String> secondIds = pageTwo.rows().stream().map(row -> String.valueOf(row.get("id"))).toList();
        assertThat(firstIds).containsExactly("1", "2", "3", "4", "5", "6", "7", "8", "9", "10");
        assertThat(secondIds).containsExactly("11", "12", "13", "14", "15", "16", "17", "18", "19", "20");
        assertThat(firstIds.stream().collect(Collectors.toSet()))
                .doesNotContainAnyElementsOf(secondIds);
    }

    @Test
    void reportsNoMorePagesOnTheFinalPageSoTheNextButtonCanBeDisabled() {
        QueryResult result = postgres.query(postgresConfig, "SELECT id FROM customers ORDER BY id LIMIT 3",
                PageRequest.first(10));
        assertThat(result.rows()).hasSize(3);
        assertThat(result.hasMore()).isFalse();
        assertThat(result.nextCursor()).isNull();
    }

    @Test
    void refusesAWriteBeforeItEverReachesTheDatabase() {
        assertThatThrownBy(() -> postgres.query(postgresConfig, "DELETE FROM customers", PageRequest.first(10)))
                .isInstanceOf(ReadOnlyViolation.class);
        assertThatThrownBy(() -> mysql.query(mysqlConfig, "DROP TABLE invoices", PageRequest.first(10)))
                .isInstanceOf(ReadOnlyViolation.class);
    }

    @Test
    void theDatabaseItselfRejectsAWriteEvenWhenTheGuardIsBypassedWhichIsTheLayerThatMustNotFail() {
        assertThatThrownBy(() -> postgres.query(postgresConfig,
                "WITH x AS (SELECT 1) INSERT INTO customers (email, full_name, country) VALUES ('h@x','h','BR')",
                PageRequest.first(10)))
                .isInstanceOf(RuntimeException.class);
        assertThat(postgres.query(postgresConfig, "SELECT count(*) AS c FROM customers WHERE email = 'h@x'",
                PageRequest.first(10)).rows().getFirst().get("c")).isEqualTo("0");
    }

    @Test
    void connectsAndReportsHealthySoABrokenTargetIsVisibleInTheUi() {
        assertThat(postgres.ping(postgresConfig)).isTrue();
        assertThat(mysql.ping(mysqlConfig)).isTrue();
    }
}
