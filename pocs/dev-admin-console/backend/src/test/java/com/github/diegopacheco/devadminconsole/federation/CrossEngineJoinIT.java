package com.github.diegopacheco.devadminconsole.federation;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.github.diegopacheco.devadminconsole.project.ConnectionConfig;
import com.github.diegopacheco.devadminconsole.project.ConnectionKind;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

@Tag("integration-test")
@SpringBootTest
class CrossEngineJoinIT {
    @Autowired
    private FederatedQueryParser parser;
    @Autowired
    private FederatedExecutor executor;

    private final ConnectionConfig postgres = new ConnectionConfig(9401L, 1L, "demo-postgres",
            ConnectionKind.POSTGRES, "localhost", 5432, "shop", "public", null, "console_reader", "console_reader",
            null, "tester");

    private final ConnectionConfig mysql = new ConnectionConfig(9402L, 1L, "demo-mysql",
            ConnectionKind.MYSQL, "localhost", 3306, "shop", null, null, "console_reader", "console_reader",
            null, "tester");

    private final ConnectionConfig cassandra = new ConnectionConfig(9403L, 1L, "demo-cassandra",
            ConnectionKind.CASSANDRA, "localhost", 9042, null, "shop", "datacenter1", null, null, null, "tester");

    private final ConnectionConfig redis = new ConnectionConfig(9404L, 1L, "demo-redis",
            ConnectionKind.REDIS, "localhost", 6379, null, null, null, null, null, null, "tester");

    private final ConnectionConfig etcd = new ConnectionConfig(9405L, 1L, "demo-etcd",
            ConnectionKind.ETCD, "localhost", 2379, null, null, null, null, null, null, "tester");

    private final ConnectionConfig kafka = new ConnectionConfig(9406L, 1L, "demo-kafka",
            ConnectionKind.KAFKA, "localhost", 9092, null, null, null, null, null, null, "tester");

    private final ConnectionConfig elasticsearch = new ConnectionConfig(9407L, 1L, "demo-elasticsearch",
            ConnectionKind.ELASTICSEARCH, "localhost", 9200, null, null, null, null, null, null, "tester");

    private final List<ConnectionConfig> connections =
            List.of(postgres, mysql, cassandra, redis, etcd, kafka, elasticsearch);

    private FederatedExecutor.Result run(String statement) {
        return executor.execute(parser.parse(statement), connections);
    }

    @Test
    void joinsPostgresCustomersToMysqlInvoicesOnTheSharedEmail() {
        FederatedExecutor.Result result = run("""
                SELECT c.email, c.full_name, i.number, i.amount_cents
                FROM demo-postgres.customers c
                JOIN demo-mysql.invoices i ON c.email = i.customer_email
                LIMIT 50""");

        assertThat(result.rows()).hasSize(50);
        assertThat(result.columns()).containsExactly("c.email", "c.full_name", "i.number", "i.amount_cents");
        assertThat(result.diagnostic()).isNull();
        assertThat(result.rows()).allSatisfy(row ->
                assertThat(row.get("i.number")).asString().startsWith("INV-"));
    }

    @Test
    void joinsMysqlInvoicesToElasticsearchProductsOnTheDocumentId() {
        FederatedExecutor.Result result = run("""
                SELECT a.number, b.sku, b.price_cents
                FROM demo-mysql.invoices a
                JOIN demo-elasticsearch.products b ON a.id = b._id
                LIMIT 100""");

        assertThat(result.rows()).hasSize(100);
        assertThat(result.rows()).allSatisfy(row -> {
            assertThat(row.get("b.sku")).asString().startsWith("SKU-");
            assertThat(row.get("a.number")).asString().startsWith("INV-");
        });
    }

    @Test
    void joinsPostgresOrderItemsToElasticsearchProductsOnTheSku() {
        FederatedExecutor.Result result = run("""
                SELECT oi.sku, oi.quantity, p.sku, p.name, p.category
                FROM demo-postgres.order_items oi
                JOIN demo-elasticsearch.products p ON oi.sku = p.sku
                LIMIT 200""");

        assertThat(result.rows()).hasSize(200);
        assertThat(result.rows()).allSatisfy(row -> {
            assertThat(row.get("oi.sku")).isEqualTo(row.get("p.sku"));
            assertThat(row.get("p.name")).asString().startsWith("Product ");
        });
    }

    @Test
    void joinsPostgresCustomersToCassandraSessionsOnTheCustomerId() {
        FederatedExecutor.Result result = run("""
                SELECT c.id, c.email, s.user_agent
                FROM demo-postgres.customers c
                JOIN demo-cassandra.sessions s ON c.id = s.customer_id
                LIMIT 1000""");

        assertThat(result.rows()).hasSize(98);
        assertThat(result.rows()).allSatisfy(row ->
                assertThat(row.get("s.user_agent")).asString().startsWith("agent-"));
    }

    @Test
    void joinsMysqlInvoicesToCassandraEventsOnANumericKey() {
        FederatedExecutor.Result result = run("""
                SELECT i.number, e.event_type, e.payload
                FROM demo-mysql.invoices i
                JOIN demo-cassandra.events_by_customer e ON i.id = e.customer_id
                LIMIT 500""");

        assertThat(result.rows()).isNotEmpty();
        assertThat(result.rows()).allSatisfy(row -> assertThat(row.get("e.event_type")).isEqualTo("view"));
    }

    @Test
    void joinsRedisToEtcdOnTheSharedConfigValue() {
        FederatedExecutor.Result result = run("""
                SELECT r.value, e.key, e.value
                FROM demo-redis.config:app:version r
                JOIN demo-etcd.config e ON r.value = e.value
                LIMIT 10""");

        assertThat(result.rows()).hasSize(1);
        assertThat(result.rows().getFirst().get("e.key")).isEqualTo("/config/app/version");
        assertThat(result.rows().getFirst().get("r.value")).isEqualTo("1.0.0");
    }

    @Test
    void joinsKafkaRecordsToPostgresRowsOnANumericColumn() {
        FederatedExecutor.Result result = run("""
                SELECT k.key, k.partition, c.email
                FROM demo-kafka.orders.events k
                JOIN demo-postgres.customers c ON k.partition = c.id
                LIMIT 100""");

        assertThat(result.rows()).hasSize(100);
        assertThat(result.rows()).allSatisfy(row -> {
            assertThat(row.get("k.key")).asString().startsWith("order-");
            assertThat(row.get("k.partition")).asString().isIn("1", "2");
        });
    }

    @Test
    void joinsRedisHashFieldsToEtcdServiceKeys() {
        FederatedExecutor.Result result = run("""
                SELECT h.field, h.value, e.key
                FROM demo-redis.session:abc123 h
                LEFT JOIN demo-etcd.service e ON h.value = e.value
                LIMIT 20""");

        assertThat(result.rows()).hasSize(4);
        assertThat(result.rows()).extracting(row -> row.get("h.field"))
                .contains("user", "ip", "agent", "expires");
    }

    @Test
    void chainsThreeEnginesInOneQuery() {
        FederatedExecutor.Result result = run("""
                SELECT c.email, i.number, s.user_agent
                FROM demo-postgres.customers c
                JOIN demo-mysql.invoices i ON c.email = i.customer_email
                JOIN demo-cassandra.sessions s ON c.id = s.customer_id
                LIMIT 500""");

        assertThat(result.sides()).hasSize(3);
        assertThat(result.sides()).extracting(FederatedExecutor.SideResult::kind)
                .containsExactly("postgres", "mysql", "cassandra");
        assertThat(result.rows()).isNotEmpty();
        assertThat(result.rows()).allSatisfy(row -> {
            assertThat(row.get("c.email")).asString().endsWith("@example.com");
            assertThat(row.get("i.number")).asString().startsWith("INV-");
            assertThat(row.get("s.user_agent")).asString().startsWith("agent-");
        });
    }

    @Test
    void chainsTheMaximumOfFiveSourcesAcrossThreeEngines() {
        FederatedExecutor.Result result = run("""
                SELECT o.id, oi.sku, p.name, c.email, i.number
                FROM demo-postgres.orders o
                JOIN demo-postgres.order_items oi ON o.id = oi.order_id
                JOIN demo-elasticsearch.products p ON oi.sku = p.sku
                JOIN demo-postgres.customers c ON o.customer_id = c.id
                JOIN demo-mysql.invoices i ON c.email = i.customer_email
                LIMIT 100""");

        assertThat(result.sides()).hasSize(5);
        assertThat(result.rows()).hasSize(100);
        assertThat(result.columns()).containsExactly("o.id", "oi.sku", "p.name", "c.email", "i.number");
    }

    @Test
    void rejectsMoreSourcesThanTheJoinPlannerWillHold() {
        assertThatThrownBy(() -> run("""
                SELECT o.id
                FROM demo-postgres.orders o
                JOIN demo-postgres.order_items oi ON o.id = oi.order_id
                JOIN demo-elasticsearch.products p ON oi.sku = p.sku
                JOIN demo-postgres.customers c ON o.customer_id = c.id
                JOIN demo-mysql.invoices i ON c.email = i.customer_email
                JOIN demo-cassandra.sessions s ON c.id = s.customer_id"""))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("at most 5");
    }

    @Test
    void leftJoinKeepsUnmatchedLeftRowsThatInnerJoinDrops() {
        FederatedExecutor.Result inner = run("""
                SELECT c.id, s.user_agent
                FROM demo-postgres.customers c
                JOIN demo-cassandra.sessions s ON c.id = s.customer_id
                LIMIT 1000""");

        FederatedExecutor.Result left = run("""
                SELECT c.id, s.user_agent
                FROM demo-postgres.customers c
                LEFT JOIN demo-cassandra.sessions s ON c.id = s.customer_id
                LIMIT 1000""");

        assertThat(inner.rows()).hasSize(98);
        assertThat(left.rows()).hasSize(559);
        assertThat(left.rows()).filteredOn(row -> row.get("s.user_agent") == null).hasSize(461);
    }

    @Test
    void leftJoinLeavesTheRightColumnsNullWhenNothingMatches() {
        FederatedExecutor.Result result = run("""
                SELECT c.email, p.sku
                FROM demo-postgres.customers c
                LEFT JOIN demo-elasticsearch.products p ON c.email = p.sku
                LIMIT 25""");

        assertThat(result.rows()).hasSize(25);
        assertThat(result.rows()).allSatisfy(row -> assertThat(row.get("p.sku")).isNull());
        assertThat(result.columns()).containsExactly("c.email", "p.sku");
    }

    @Test
    void starProjectsEveryColumnOfEverySidePrefixedByItsAlias() {
        FederatedExecutor.Result result = run("""
                SELECT *
                FROM demo-postgres.customers c
                JOIN demo-mysql.invoices i ON c.email = i.customer_email
                LIMIT 5""");

        assertThat(result.columns()).contains("c.id", "c.email", "c.full_name", "c.country",
                "i.id", "i.number", "i.customer_email", "i.amount_cents", "i.status");
    }

    @Test
    void aliasStarProjectsOnlyThatSidesColumns() {
        FederatedExecutor.Result result = run("""
                SELECT i.*
                FROM demo-postgres.customers c
                JOIN demo-mysql.invoices i ON c.email = i.customer_email
                LIMIT 5""");

        assertThat(result.columns()).allSatisfy(column -> assertThat(column).startsWith("i."));
        assertThat(result.columns()).contains("i.number", "i.customer_email");
    }

    @Test
    void keepsIdenticallyNamedColumnsFromBothSidesApart() {
        FederatedExecutor.Result result = run("""
                SELECT c.id, o.id, c.email, o.status
                FROM demo-postgres.orders o
                JOIN demo-postgres.customers c ON o.customer_id = c.id
                LIMIT 10""");

        assertThat(result.columns()).containsExactly("c.id", "o.id", "c.email", "o.status");
        assertThat(result.rows()).allSatisfy(row -> {
            assertThat(row.get("c.email")).asString().endsWith("@example.com");
            assertThat(row.get("o.status")).asString().isIn("placed", "paid", "shipped", "cancelled");
        });
    }

    @Test
    void limitCapsTheProjectedRowsAndClampsAtTheHardMaximum() {
        assertThat(run("""
                SELECT c.email, i.number
                FROM demo-postgres.customers c
                JOIN demo-mysql.invoices i ON c.email = i.customer_email
                LIMIT 7""").rows()).hasSize(7);

        assertThat(run("""
                SELECT oi.sku, p.name
                FROM demo-postgres.order_items oi
                JOIN demo-elasticsearch.products p ON oi.sku = p.sku
                LIMIT 99999""").rows()).hasSize(1000);
    }

    @Test
    void defaultsToOneHundredRowsWhenNoLimitIsGiven() {
        FederatedExecutor.Result result = run("""
                SELECT c.email, i.number
                FROM demo-postgres.customers c
                JOIN demo-mysql.invoices i ON c.email = i.customer_email""");

        assertThat(result.rows()).hasSize(100);
    }

    @Test
    void reportsWhatEachSideContributedSoTheUiCanShowTheFanOut() {
        FederatedExecutor.Result result = run("""
                SELECT c.email, p.sku
                FROM demo-postgres.order_items c
                JOIN demo-elasticsearch.products p ON c.sku = p.sku
                LIMIT 10""");

        FederatedExecutor.SideResult left = result.sides().getFirst();
        FederatedExecutor.SideResult right = result.sides().get(1);
        assertThat(left.connectionName()).isEqualTo("demo-postgres");
        assertThat(left.source()).isEqualTo("order_items");
        assertThat(left.rows()).isEqualTo(5000);
        assertThat(left.truncated()).isTrue();
        assertThat(right.kind()).isEqualTo("elasticsearch");
        assertThat(right.rows()).isEqualTo(1000);
        assertThat(right.truncated()).isFalse();
    }

    @Test
    void explainsWhyAJoinAcrossUnrelatedValuesFoundNothing() {
        FederatedExecutor.Result result = run("""
                SELECT c.email, p.sku
                FROM demo-postgres.customers c
                JOIN demo-elasticsearch.products p ON c.email = p.sku
                LIMIT 10""");

        assertThat(result.rows()).isEmpty();
        assertThat(result.diagnostic())
                .contains("nothing matched on c.email = p.sku")
                .contains("@example.com")
                .contains("SKU-");
    }

    @Test
    void blamesTheJoinThatEmptiedTheResultRatherThanTheFirstOneInTheQuery() {
        FederatedExecutor.Result result = run("""
                SELECT a.customer_id, b._id, e.value
                FROM demo-cassandra.events_by_customer a
                JOIN demo-elasticsearch.products b ON a.customer_id = b._id
                JOIN demo-redis.cache:customer:1 e ON a.customer_id = e.value
                LIMIT 25""");

        assertThat(result.rows()).isEmpty();
        assertThat(result.diagnostic())
                .contains("a.customer_id = e.value")
                .doesNotContain("b._id");
    }

    @Test
    void eachJoinOfAChainMatchesOnItsOwnSoOnlyTheGuiltyOneIsReported() {
        assertThat(run("""
                SELECT a.customer_id, b._id
                FROM demo-cassandra.events_by_customer a
                JOIN demo-elasticsearch.products b ON a.customer_id = b._id
                LIMIT 25""").rows()).hasSize(25);

        assertThat(run("""
                SELECT a.customer_id, d.id
                FROM demo-cassandra.events_by_customer a
                JOIN demo-mysql.invoices d ON a.customer_id = d.id
                LIMIT 25""").rows()).hasSize(25);
    }

    @Test
    void namesTheAvailableColumnsWhenTheJoinKeyDoesNotExist() {
        assertThatThrownBy(() -> run("""
                SELECT c.email, i.number
                FROM demo-postgres.customers c
                JOIN demo-mysql.invoices i ON c.emai = i.customer_email"""))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("no column named \"emai\"")
                .hasMessageContaining("did you mean \"email\"")
                .hasMessageContaining("full_name");
    }

    @Test
    void namesTheAvailableSourcesWhenTheTableDoesNotExist() {
        assertThatThrownBy(() -> run("""
                SELECT c.email, i.number
                FROM demo-postgres.custmers c
                JOIN demo-mysql.invoices i ON c.email = i.customer_email"""))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("no source named \"custmers\"")
                .hasMessageContaining("customers");
    }

    @Test
    void rejectsACassandraKeyspaceUsedWhereATableBelongs() {
        assertThatThrownBy(() -> run("""
                SELECT c.id, s.customer_id
                FROM demo-postgres.customers c
                JOIN demo-cassandra.shop s ON c.id = s.customer_id"""))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("is the keyspace, not a table");
    }

    @Test
    void rejectsARedisKeyThatDoesNotExist() {
        assertThatThrownBy(() -> run("""
                SELECT r.value, e.key
                FROM demo-redis.config:app:missing r
                JOIN demo-etcd.config e ON r.value = e.value"""))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("no Redis key named \"config:app:missing\"");
    }

    @Test
    void readsRedisCollectionTypesAsJoinSides() {
        FederatedExecutor.Result result = run("""
                SELECT h.field, h.value
                FROM demo-redis.session:abc123 h
                LEFT JOIN demo-redis.queue:emails q ON h.value = q.value
                LIMIT 50""");

        assertThat(result.sides().getFirst().rows()).isEqualTo(4);
        assertThat(result.sides().get(1).rows()).isEqualTo(5);
        assertThat(result.rows()).hasSize(4);
    }

    @Test
    void rejectsAConnectionThatIsNotInTheProject() {
        assertThatThrownBy(() -> run("""
                SELECT c.email, x.id
                FROM demo-postgres.customers c
                JOIN demo-oracle.things x ON c.email = x.id"""))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("no connection named demo-oracle");
    }

    @Test
    void rejectsAProjectionColumnThatIsNotQualifiedByAnAlias() {
        assertThatThrownBy(() -> run("""
                SELECT email, i.number
                FROM demo-postgres.customers c
                JOIN demo-mysql.invoices i ON c.email = i.customer_email"""))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("qualify every column with its alias");
    }

    @Test
    void rejectsTheSameAliasUsedForTwoSources() {
        assertThatThrownBy(() -> run("""
                SELECT a.email, b.number
                FROM demo-postgres.customers a
                JOIN demo-mysql.invoices b ON a.email = b.customer_email
                JOIN demo-cassandra.sessions b ON a.id = b.customer_id"""))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("is used twice");
    }

    @Test
    void rejectsAnOnClauseThatDoesNotReachBackToAnEarlierSource() {
        assertThatThrownBy(() -> run("""
                SELECT a.email, a.number
                FROM demo-postgres.customers a
                JOIN demo-mysql.invoices a ON a.email = a.customer_email"""))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("must compare it to an earlier source");
    }

    @Test
    void matchesAliasesRegardlessOfTheCaseTheyWereTypedIn() {
        FederatedExecutor.Result result = run("""
                SELECT C.email, I.number
                FROM demo-postgres.customers c
                JOIN demo-mysql.invoices i ON C.email = I.customer_email
                LIMIT 5""");

        assertThat(result.rows()).hasSize(5);
        assertThat(result.rows()).allSatisfy(row -> assertThat(row.values()).doesNotContainNull());
    }

    @Test
    void readsTheSameResultWhenTheSameJoinRunsTwice() {
        String statement = """
                SELECT c.email, i.number
                FROM demo-postgres.customers c
                JOIN demo-mysql.invoices i ON c.email = i.customer_email
                LIMIT 30""";

        List<Map<String, Object>> first = run(statement).rows();
        List<Map<String, Object>> second = run(statement).rows();
        assertThat(first).isEqualTo(second);
    }
}
