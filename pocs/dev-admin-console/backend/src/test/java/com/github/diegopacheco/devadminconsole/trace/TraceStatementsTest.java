package com.github.diegopacheco.devadminconsole.trace;

import static org.assertj.core.api.Assertions.assertThat;

import com.github.diegopacheco.devadminconsole.engine.SchemaNode;
import com.github.diegopacheco.devadminconsole.project.ConnectionKind;
import java.util.List;
import org.junit.jupiter.api.Test;

class TraceStatementsTest {
    private final TraceStatements statements = new TraceStatements();
    private final TraceBudget budget = TraceBudget.defaults();

    private final List<SchemaNode> sqlSchema = List.of(
            new SchemaNode("customers", "table", "BASE TABLE", List.of(
                    SchemaNode.leaf("id", "column", "integer not null"),
                    SchemaNode.leaf("email", "column", "character varying not null"),
                    SchemaNode.leaf("total_cents", "column", "integer not null"))),
            new SchemaNode("metrics", "table", "BASE TABLE", List.of(
                    SchemaNode.leaf("value", "column", "double precision"))));

    @Test
    void probesOnlyKeyOrTextColumnsSoATraceDoesNotScanEveryNumericColumn() {
        List<String> candidates = statements.candidateColumns(sqlSchema.getFirst());
        assertThat(candidates).contains("id", "email").doesNotContain("total_cents");
    }

    @Test
    void skipsTablesWithNothingWorthSearching() {
        List<TraceStatements.Probe> probes = statements.probes(ConnectionKind.POSTGRES, sqlSchema, "42", budget);
        assertThat(probes).extracting(TraceStatements.Probe::source).containsExactly("customers");
    }

    @Test
    void boundsEverySqlProbeSoOneTraceCannotPullAWholeTable() {
        List<TraceStatements.Probe> probes = statements.probes(ConnectionKind.POSTGRES, sqlSchema, "42", budget);
        assertThat(probes.getFirst().statement()).contains("LIMIT " + budget.rowsPerSource());
    }

    @Test
    void escapesQuotesSoATermCannotTerminateTheGeneratedLiteral() {
        List<TraceStatements.Probe> probes = statements.probes(ConnectionKind.POSTGRES, sqlSchema, "o'brien", budget);
        assertThat(probes.getFirst().statement()).contains("'o''brien'").doesNotContain("'o'brien'");
    }

    @Test
    void onlyProbesCassandraPartitionKeysThatCanHoldTheTerm() {
        List<SchemaNode> cassandra = List.of(
                new SchemaNode("events", "table", "", List.of(
                        SchemaNode.leaf("customer_id", "column", "int (partition_key)"),
                        SchemaNode.leaf("event_type", "column", "text (regular)"))),
                new SchemaNode("sessions", "table", "", List.of(
                        SchemaNode.leaf("session_id", "column", "uuid (partition_key)"))));

        assertThat(statements.probes(ConnectionKind.CASSANDRA, cassandra, "42", budget))
                .extracting(TraceStatements.Probe::source).containsExactly("events");

        assertThat(statements.probes(ConnectionKind.CASSANDRA, cassandra, "order-42", budget))
                .isEmpty();

        assertThat(statements.probes(ConnectionKind.CASSANDRA, cassandra,
                "123e4567-e89b-12d3-a456-426614174000", budget))
                .extracting(TraceStatements.Probe::source).containsExactly("sessions");
    }

    @Test
    void neverProbesACassandraNonKeyColumnBecauseThatWouldNeedAllowFiltering() {
        List<SchemaNode> cassandra = List.of(new SchemaNode("events", "table", "", List.of(
                SchemaNode.leaf("payload", "column", "text (regular)"))));
        assertThat(statements.probes(ConnectionKind.CASSANDRA, cassandra, "anything", budget)).isEmpty();
    }

    @Test
    void scansAWideKafkaWindowSoAMatchIsNotMissedJustBecauseItIsNotRecent() {
        List<SchemaNode> topics = List.of(SchemaNode.leaf("orders.events", "topic", "3 partitions"));
        List<TraceStatements.Probe> probes = statements.probes(ConnectionKind.KAFKA, topics, "order-42", budget);
        assertThat(probes.getFirst().statement()).isEqualTo("consume orders.events --limit " + budget.scanWindow());
        assertThat(budget.scanWindow()).isGreaterThan(budget.rowsPerSource());
    }

    @Test
    void asksElasticsearchToSearchEveryFieldWhichIsWhatItIsFor() {
        List<SchemaNode> indices = List.of(SchemaNode.leaf("products", "index", "6 fields"));
        assertThat(statements.probes(ConnectionKind.ELASTICSEARCH, indices, "SKU-0042", budget).getFirst().statement())
                .contains("query_string").contains("SKU-0042");
    }

    @Test
    void escapesQuotesInTheElasticsearchJsonBody() {
        List<SchemaNode> indices = List.of(SchemaNode.leaf("products", "index", ""));
        assertThat(statements.probes(ConnectionKind.ELASTICSEARCH, indices, "say \"hi\"", budget).getFirst().statement())
                .contains("\\\"hi\\\"");
    }

    @Test
    void looksUpRedisBothAsAnExactKeyAndAsAPattern() {
        List<TraceStatements.Probe> probes = statements.probes(ConnectionKind.REDIS, List.of(), "session:abc", budget);
        assertThat(probes).extracting(TraceStatements.Probe::statement)
                .anyMatch(statement -> statement.startsWith("TYPE session:abc"))
                .anyMatch(statement -> statement.contains("MATCH *session:abc*"));
    }

    @Test
    void looksUpEtcdBothExactlyAndByPrefix() {
        List<TraceStatements.Probe> probes = statements.probes(ConnectionKind.ETCD, List.of(), "/config/app", budget);
        assertThat(probes).extracting(TraceStatements.Probe::statement)
                .containsExactly("get /config/app", "get /config/app --prefix");
    }

    @Test
    void neverExceedsTheSourceBudgetOnAWideSchema() {
        List<SchemaNode> wide = java.util.stream.IntStream.range(0, 100)
                .mapToObj(index -> new SchemaNode("t" + index, "table", "", List.of(
                        SchemaNode.leaf("id", "column", "integer"))))
                .toList();
        assertThat(statements.probes(ConnectionKind.POSTGRES, wide, "42", budget))
                .hasSize(budget.sourcesPerConnection());
    }
}
