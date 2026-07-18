package com.github.diegopacheco.adminconsole.federation;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Test;

class FederatedQueryParserTest {
    private final FederatedQueryParser parser = new FederatedQueryParser();

    @Test
    void parsesAJoinBetweenTwoDifferentEngines() {
        FederatedQuery query = parser.parse("""
                SELECT c.email, p.name
                FROM demo-postgres.customers c
                JOIN demo-elasticsearch.products p ON p.sku = c.country
                LIMIT 50""");
        assertThat(query.projection()).containsExactly("c.email", "p.name");
        assertThat(query.sides().get(0).connectionName()).isEqualTo("demo-postgres");
        assertThat(query.sides().get(0).source()).isEqualTo("customers");
        assertThat(query.sides().get(1).connectionName()).isEqualTo("demo-elasticsearch");
        assertThat(query.sides().get(1).source()).isEqualTo("products");
        assertThat(query.limit()).isEqualTo(50);
    }

    @Test
    void mapsEachJoinKeyToItsOwnSideRegardlessOfTheOrderTheyAreWritten() {
        FederatedQuery written = parser.parse(
                "SELECT a.id FROM demo-postgres.orders a JOIN demo-kafka.orders.events b ON a.id = b.key");
        FederatedQuery reversed = parser.parse(
                "SELECT a.id FROM demo-postgres.orders a JOIN demo-kafka.orders.events b ON b.key = a.id");
        assertThat(written.joins().get(0).leftKey()).isEqualTo("id");
        assertThat(written.joins().get(0).rightKey()).isEqualTo("key");
        assertThat(reversed.joins().get(0).leftKey()).isEqualTo("id");
        assertThat(reversed.joins().get(0).rightKey()).isEqualTo("key");
    }

    @Test
    void keepsDottedSourceNamesLikeKafkaTopicsIntact() {
        FederatedQuery query = parser.parse(
                "SELECT a.id FROM demo-kafka.orders.events a JOIN demo-postgres.orders b ON a.key = b.id");
        assertThat(query.sides().get(0).connectionName()).isEqualTo("demo-kafka");
        assertThat(query.sides().get(0).source()).isEqualTo("orders.events");
    }

    @Test
    void recognisesLeftJoinSoUnmatchedRowsCanBeKept() {
        assertThat(parser.parse("SELECT a.id FROM x.a a LEFT JOIN y.b b ON a.id = b.id").joins().get(0).leftJoin()).isTrue();
        assertThat(parser.parse("SELECT a.id FROM x.a a JOIN y.b b ON a.id = b.id").joins().get(0).leftJoin()).isFalse();
        assertThat(parser.parse("SELECT a.id FROM x.a a INNER JOIN y.b b ON a.id = b.id").joins().get(0).leftJoin()).isFalse();
    }

    @Test
    void capsTheLimitSoOneQueryCannotAskForAnUnboundedJoin() {
        assertThat(parser.parse("SELECT a.id FROM x.a a JOIN y.b b ON a.id = b.id LIMIT 999999").limit())
                .isEqualTo(1000);
    }

    @Test
    void defaultsTheLimitWhenNoneIsGiven() {
        assertThat(parser.parse("SELECT a.id FROM x.a a JOIN y.b b ON a.id = b.id").limit()).isEqualTo(100);
    }

    @Test
    void acceptsEtcdPathsAndRedisKeysAsSourcesSoEveryEngineCanBeJoined() {
        FederatedQuery etcd = parser.parse(
                "SELECT a.key FROM demo-etcd./config/app a JOIN demo-postgres.orders b ON a.key = b.id");
        assertThat(etcd.sides().get(0).connectionName()).isEqualTo("demo-etcd");
        assertThat(etcd.sides().get(0).source()).isEqualTo("/config/app");

        FederatedQuery redis = parser.parse(
                "SELECT a.field FROM demo-redis.session:abc123 a JOIN demo-postgres.orders b ON a.field = b.id");
        assertThat(redis.sides().get(0).connectionName()).isEqualTo("demo-redis");
        assertThat(redis.sides().get(0).source()).isEqualTo("session:abc123");
    }

    @Test
    void rejectsASingleSourceQueryBecauseThatIsWhatTheNormalConsoleIsFor() {
        assertThatThrownBy(() -> parser.parse("SELECT * FROM demo-postgres.customers c"))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("no JOIN found");
    }

    @Test
    void rejectsASourceThatDoesNotNameItsConnection() {
        assertThatThrownBy(() -> parser.parse("SELECT a.id FROM customers a JOIN y.b b ON a.id = b.id"))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void rejectsAnOnClauseThatOnlyMentionsOneSide() {
        assertThatThrownBy(() -> parser.parse("SELECT a.id FROM x.a a JOIN y.b b ON a.id = a.other"))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("must compare it to an earlier source");
    }

    @Test
    void rejectsTwoSidesSharingAnAliasBecauseProjectionWouldBeAmbiguous() {
        assertThatThrownBy(() -> parser.parse("SELECT a.id FROM x.a a JOIN y.b a ON a.id = a.id"))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("must compare it to an earlier source");
    }

    @Test
    void rejectsAnEmptyStatement() {
        assertThatThrownBy(() -> parser.parse("   ")).isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> parser.parse(null)).isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void toleratesTrailingSemicolonsAndUntidyWhitespace() {
        FederatedQuery query = parser.parse("  SELECT a.id   FROM x.a  a JOIN\n y.b b ON a.id = b.id ;  ");
        assertThat(query.projection()).containsExactly("a.id");
    }
}
