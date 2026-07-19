package com.github.diegopacheco.adminconsole.ai;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.github.diegopacheco.adminconsole.engine.SchemaNode;
import com.github.diegopacheco.adminconsole.project.ConnectionConfig;
import com.github.diegopacheco.adminconsole.project.ConnectionKind;
import java.util.List;
import org.junit.jupiter.api.Test;

class PromptBuilderTest {
    private final PromptBuilder builder = new PromptBuilder();

    private ConnectionConfig connection(ConnectionKind kind) {
        return new ConnectionConfig(1L, 2L, "prod-db", kind, "db.internal.example.com", 5432, "shop", "public",
                "dc1", "reporting_user", "hunter2-super-secret", null, "diego");
    }

    private final List<SchemaNode> schema = List.of(
            new SchemaNode("customers", "table", "BASE TABLE", List.of(
                    SchemaNode.leaf("id", "column", "integer not null"),
                    SchemaNode.leaf("email", "column", "text not null"))),
            new SchemaNode("orders", "table", "BASE TABLE", List.of(
                    SchemaNode.leaf("total_cents", "column", "integer not null"))));

    @Test
    void neverSendsThePasswordToAThirdPartyModel() {
        String prompt = builder.build(connection(ConnectionKind.POSTGRES), schema, "count the customers");
        assertThat(prompt).doesNotContain("hunter2-super-secret");
    }

    @Test
    void neverSendsTheHostnameOrUsernameBecauseTheyDescribeInfrastructureNotSchema() {
        String prompt = builder.build(connection(ConnectionKind.POSTGRES), schema, "count the customers");
        assertThat(prompt).doesNotContain("db.internal.example.com");
        assertThat(prompt).doesNotContain("reporting_user");
    }

    @Test
    void sendsTableAndColumnNamesSoTheModelCanWriteARealQuery() {
        String prompt = builder.build(connection(ConnectionKind.POSTGRES), schema, "count the customers");
        assertThat(prompt).contains("customers").contains("email").contains("orders").contains("total_cents");
    }

    @Test
    void carriesTheUserRequestVerbatim() {
        String prompt = builder.build(connection(ConnectionKind.POSTGRES), schema, "top 10 customers by order total");
        assertThat(prompt).contains("top 10 customers by order total");
    }

    @Test
    void tellsTheModelToWriteOnlyReadQueries() {
        String prompt = builder.build(connection(ConnectionKind.POSTGRES), schema, "count the customers");
        assertThat(prompt.toLowerCase()).contains("only read");
        assertThat(prompt.toLowerCase()).contains("never write");
    }

    @Test
    void asksForEachEnginesOwnGrammarSoCqlIsNotWrittenAsPostgres() {
        assertThat(builder.grammar(ConnectionKind.POSTGRES)).contains("PostgreSQL");
        assertThat(builder.grammar(ConnectionKind.MYSQL)).contains("MySQL");
        assertThat(builder.grammar(ConnectionKind.CASSANDRA)).contains("CQL");
        assertThat(builder.grammar(ConnectionKind.REDIS)).contains("Redis command");
        assertThat(builder.grammar(ConnectionKind.ETCD)).contains("etcdctl");
        assertThat(builder.grammar(ConnectionKind.KAFKA)).contains("consume");
        assertThat(builder.grammar(ConnectionKind.ELASTICSEARCH)).contains("_search");
    }

    @Test
    void coversEveryEngineSoAnEighthOneCannotBeAddedWithoutAGrammar() {
        for (ConnectionKind kind : ConnectionKind.values()) {
            assertThat(builder.grammar(kind)).isNotBlank();
        }
    }

    @Test
    void rejectsAnEmptyRequestRatherThanAskingTheModelForNothing() {
        assertThatThrownBy(() -> builder.build(connection(ConnectionKind.POSTGRES), schema, "  "))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void capsAVeryWideSchemaSoThePromptCannotGrowWithoutBound() {
        List<SchemaNode> wide = java.util.stream.IntStream.range(0, 500)
                .mapToObj(index -> SchemaNode.leaf("table_" + index, "table", "BASE TABLE"))
                .toList();
        String prompt = builder.build(connection(ConnectionKind.POSTGRES), wide, "count everything");
        assertThat(prompt).contains("more omitted");
        assertThat(prompt.getBytes(java.nio.charset.StandardCharsets.UTF_8).length)
                .isLessThan(AgentCliRunner.MAX_PROMPT_BYTES);
    }
}
