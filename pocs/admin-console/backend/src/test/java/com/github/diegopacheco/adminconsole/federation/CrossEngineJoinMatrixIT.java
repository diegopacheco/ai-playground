package com.github.diegopacheco.adminconsole.federation;

import static org.assertj.core.api.Assertions.assertThat;

import com.github.diegopacheco.adminconsole.project.ConnectionConfig;
import com.github.diegopacheco.adminconsole.project.ConnectionKind;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

@Tag("integration-test")
@SpringBootTest
class CrossEngineJoinMatrixIT {
    private record Keyed(String connection, String source, String column) {}

    private static final List<Keyed> KEYED = List.of(
            new Keyed("demo-postgres", "customers", "id"),
            new Keyed("demo-mysql", "invoices", "id"),
            new Keyed("demo-cassandra", "sessions", "customer_id"),
            new Keyed("demo-elasticsearch", "products", "_id"),
            new Keyed("demo-kafka", "orders.events", "partition"),
            new Keyed("demo-etcd", "config", "version"));

    @Autowired
    private FederatedQueryParser parser;
    @Autowired
    private FederatedExecutor executor;

    private final List<ConnectionConfig> connections = List.of(
            new ConnectionConfig(9501L, 1L, "demo-postgres", ConnectionKind.POSTGRES, "localhost", 5432, "shop",
                    "public", null, "console_reader", "console_reader", null, "tester"),
            new ConnectionConfig(9502L, 1L, "demo-mysql", ConnectionKind.MYSQL, "localhost", 3306, "shop", null, null,
                    "console_reader", "console_reader", null, "tester"),
            new ConnectionConfig(9503L, 1L, "demo-cassandra", ConnectionKind.CASSANDRA, "localhost", 9042, null,
                    "shop", "datacenter1", null, null, null, "tester"),
            new ConnectionConfig(9504L, 1L, "demo-redis", ConnectionKind.REDIS, "localhost", 6379, null, null, null,
                    null, null, null, "tester"),
            new ConnectionConfig(9505L, 1L, "demo-etcd", ConnectionKind.ETCD, "localhost", 2379, null, null, null,
                    null, null, null, "tester"),
            new ConnectionConfig(9506L, 1L, "demo-kafka", ConnectionKind.KAFKA, "localhost", 9092, null, null, null,
                    null, null, null, "tester"),
            new ConnectionConfig(9507L, 1L, "demo-elasticsearch", ConnectionKind.ELASTICSEARCH, "localhost", 9200,
                    null, null, null, null, null, null, "tester"));

    private FederatedExecutor.Result run(String statement) {
        return executor.execute(parser.parse(statement), connections);
    }

    static Stream<Arguments> everyOrderedPair() {
        List<Arguments> pairs = new ArrayList<>();
        for (Keyed left : KEYED) {
            for (Keyed right : KEYED) {
                pairs.add(Arguments.of(left, right));
            }
        }
        return pairs.stream();
    }

    @ParameterizedTest(name = "{0} joined to {1}")
    @MethodSource("everyOrderedPair")
    void anyEngineCanDriveAJoinIntoAnyOtherEngineOnEqualValues(Keyed left, Keyed right) {
        FederatedExecutor.Result result = run("SELECT l." + left.column() + ", r." + right.column()
                + " FROM " + left.connection() + "." + left.source() + " l"
                + " JOIN " + right.connection() + "." + right.source() + " r"
                + " ON l." + left.column() + " = r." + right.column()
                + " LIMIT 20");

        assertThat(result.rows()).isNotEmpty();
        assertThat(result.diagnostic()).isNull();
        assertThat(result.rows()).allSatisfy(row ->
                assertThat(row.get("l." + left.column())).isEqualTo(row.get("r." + right.column())));
    }

    @Test
    void redisCanDriveAJoinIntoAnotherEngine() {
        FederatedExecutor.Result result = run("""
                SELECT r.value, e.key
                FROM demo-redis.config:app:version r
                JOIN demo-etcd.config e ON r.value = e.value
                LIMIT 10""");

        assertThat(result.rows()).hasSize(1);
        assertThat(result.rows().getFirst().get("e.key")).isEqualTo("/config/app/version");
    }

    @Test
    void redisCanBeDrivenIntoAsTheRightSideOfAJoin() {
        FederatedExecutor.Result result = run("""
                SELECT e.key, r.value
                FROM demo-etcd.config e
                JOIN demo-redis.config:app:version r ON e.value = r.value
                LIMIT 10""");

        assertThat(result.rows()).hasSize(1);
        assertThat(result.rows().getFirst().get("r.value")).isEqualTo("1.0.0");
    }

    @Test
    void timestampsNeverMatchAcrossEnginesBecauseEachFormatsThemDifferently() {
        FederatedExecutor.Result result = run("""
                SELECT o.placed_at, i.issued_at
                FROM demo-postgres.orders o
                JOIN demo-mysql.invoices i ON o.placed_at = i.issued_at
                LIMIT 10""");

        assertThat(result.rows()).isEmpty();
        assertThat(result.diagnostic())
                .contains("nothing matched on o.placed_at = i.issued_at")
                .contains("do not overlap");
    }

    @Test
    void postgresAndCassandraTimestampsAlsoDisagreeOnFormat() {
        FederatedExecutor.Result result = run("""
                SELECT o.placed_at, s.started_at
                FROM demo-postgres.orders o
                JOIN demo-cassandra.sessions s ON o.placed_at = s.started_at
                LIMIT 10""");

        assertThat(result.rows()).isEmpty();
        assertThat(result.diagnostic()).contains("nothing matched");
    }

    @Test
    void nestedElasticsearchObjectsArriveAsJsonStringsNotJoinableFields() {
        FederatedExecutor.Result result = run("""
                SELECT p.supplier, p.sku
                FROM demo-postgres.order_items oi
                JOIN demo-elasticsearch.products p ON oi.sku = p.sku
                LIMIT 5""");

        assertThat(result.rows()).allSatisfy(row ->
                assertThat(row.get("p.supplier")).asString().startsWith("{").contains("\"country\":\"BR\""));
    }

    @Test
    void nullJoinKeysNeverMatchEachOtherEvenWhenBothSidesAreNull() {
        FederatedExecutor.Result result = run("""
                SELECT a._id, b._id
                FROM demo-elasticsearch.products a
                JOIN demo-elasticsearch.products b ON a._score = b._score
                LIMIT 10""");

        assertThat(result.rows()).isEmpty();
    }

    @Test
    void aLeftJoinFollowedByAnInnerJoinDropsTheRowsTheLeftJoinPreserved() {
        FederatedExecutor.Result left = run("""
                SELECT c.id, s.user_agent
                FROM demo-postgres.customers c
                LEFT JOIN demo-cassandra.sessions s ON c.id = s.customer_id
                LIMIT 1000""");

        FederatedExecutor.Result chained = run("""
                SELECT c.id, s.user_agent, i.number
                FROM demo-postgres.customers c
                LEFT JOIN demo-cassandra.sessions s ON c.id = s.customer_id
                JOIN demo-mysql.invoices i ON c.email = i.customer_email
                LIMIT 1000""");

        assertThat(left.rows()).hasSize(559);
        assertThat(chained.rows()).hasSize(917);
        assertThat(chained.rows()).allSatisfy(row ->
                assertThat(row.get("i.number")).asString().startsWith("INV-"));
    }

    @Test
    void joinsOnValuesAreCaseSensitiveSoCategoryLabelsMustMatchExactly() {
        FederatedExecutor.Result matching = run("""
                SELECT a.category, b.category
                FROM demo-elasticsearch.products a
                JOIN demo-elasticsearch.products b ON a.category = b.category
                LIMIT 5""");

        assertThat(matching.rows()).isNotEmpty();
        assertThat(matching.rows()).allSatisfy(row ->
                assertThat(row.get("a.category")).asString().startsWith("cat-"));
    }
}
