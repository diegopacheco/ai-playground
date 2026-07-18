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
        assertThat(query.left().connectionName()).isEqualTo("demo-postgres");
        assertThat(query.left().source()).isEqualTo("customers");
        assertThat(query.right().connectionName()).isEqualTo("demo-elasticsearch");
        assertThat(query.right().source()).isEqualTo("products");
        assertThat(query.limit()).isEqualTo(50);
    }

    @Test
    void mapsEachJoinKeyToItsOwnSideRegardlessOfTheOrderTheyAreWritten() {
        FederatedQuery written = parser.parse(
                "SELECT a.id FROM demo-postgres.orders a JOIN demo-kafka.orders.events b ON a.id = b.key");
        FederatedQuery reversed = parser.parse(
                "SELECT a.id FROM demo-postgres.orders a JOIN demo-kafka.orders.events b ON b.key = a.id");
        assertThat(written.leftKey()).isEqualTo("id");
        assertThat(written.rightKey()).isEqualTo("key");
        assertThat(reversed.leftKey()).isEqualTo("id");
        assertThat(reversed.rightKey()).isEqualTo("key");
    }

    @Test
    void keepsDottedSourceNamesLikeKafkaTopicsIntact() {
        FederatedQuery query = parser.parse(
                "SELECT a.id FROM demo-kafka.orders.events a JOIN demo-postgres.orders b ON a.key = b.id");
        assertThat(query.left().connectionName()).isEqualTo("demo-kafka");
        assertThat(query.left().source()).isEqualTo("orders.events");
    }

    @Test
    void recognisesLeftJoinSoUnmatchedRowsCanBeKept() {
        assertThat(parser.parse("SELECT a.id FROM x.a a LEFT JOIN y.b b ON a.id = b.id").leftJoin()).isTrue();
        assertThat(parser.parse("SELECT a.id FROM x.a a JOIN y.b b ON a.id = b.id").leftJoin()).isFalse();
        assertThat(parser.parse("SELECT a.id FROM x.a a INNER JOIN y.b b ON a.id = b.id").leftJoin()).isFalse();
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
    void carriesAPerSideWhereClause() {
        FederatedQuery query = parser.parse(
                "SELECT a.id FROM x.a a WHERE a.country = 'BR' JOIN y.b b ON a.id = b.id");
        assertThat(query.left().where()).isEqualTo("a.country = 'BR'");
    }

    @Test
    void rejectsASingleSourceQueryBecauseThatIsWhatTheNormalConsoleIsFor() {
        assertThatThrownBy(() -> parser.parse("SELECT * FROM demo-postgres.customers c"))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("expected");
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
                .hasMessageContaining("both aliases");
    }

    @Test
    void rejectsTwoSidesSharingAnAliasBecauseProjectionWouldBeAmbiguous() {
        assertThatThrownBy(() -> parser.parse("SELECT a.id FROM x.a a JOIN y.b a ON a.id = a.id"))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("different aliases");
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
