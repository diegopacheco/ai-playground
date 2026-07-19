package com.github.diegopacheco.devadminconsole.federation;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Test;

class FederatedQueryParserTest {
    private final FederatedQueryParser parser = new FederatedQueryParser();

    @Test
    void resolvesAliasesWrittenInADifferentCaseThanTheyWereDeclared() {
        FederatedQuery query = parser.parse(
                "SELECT C.email, I.number FROM demo-postgres.customers c JOIN demo-mysql.invoices i "
                        + "ON C.email = I.customer_email");
        assertThat(query.projection()).containsExactly("c.email", "i.number");
        assertThat(query.joins().get(0).leftAlias()).isEqualTo("c");
        assertThat(query.joins().get(0).rightAlias()).isEqualTo("i");
    }

    @Test
    void keepsTheDeclaredAliasSpellingSoProjectedColumnsMatchTheJoinedRows() {
        FederatedQuery query = parser.parse(
                "SELECT cust.email FROM demo-postgres.customers CUST JOIN demo-mysql.invoices i "
                        + "ON cust.email = i.customer_email");
        assertThat(query.sides().get(0).alias()).isEqualTo("CUST");
        assertThat(query.projection()).containsExactly("CUST.email");
        assertThat(query.joins().get(0).leftAlias()).isEqualTo("CUST");
    }

    @Test
    void acceptsTheAsKeywordBeforeAnAlias() {
        FederatedQuery query = parser.parse(
                "SELECT a.id FROM demo-postgres.orders AS a JOIN demo-mysql.invoices AS b ON a.id = b.id");
        assertThat(query.sides().get(0).alias()).isEqualTo("a");
        assertThat(query.sides().get(1).alias()).isEqualTo("b");
    }

    @Test
    void readsKeywordsWhateverCaseTheyAreTypedIn() {
        FederatedQuery query = parser.parse(
                "select a.id, b.id from demo-postgres.orders a left join demo-mysql.invoices b on a.id = b.id limit 5");
        assertThat(query.sides()).hasSize(2);
        assertThat(query.joins().get(0).leftJoin()).isTrue();
        assertThat(query.limit()).isEqualTo(5);
    }

    @Test
    void chainsUpToFiveSourcesAndRejectsTheSixth() {
        FederatedQuery five = parser.parse("""
                SELECT a.id FROM x.a a
                JOIN x.b b ON a.id = b.id
                JOIN x.c c ON b.id = c.id
                JOIN x.d d ON c.id = d.id
                JOIN x.e e ON d.id = e.id""");
        assertThat(five.sides()).hasSize(5);
        assertThat(five.joins()).hasSize(4);

        assertThatThrownBy(() -> parser.parse("""
                SELECT a.id FROM x.a a
                JOIN x.b b ON a.id = b.id
                JOIN x.c c ON b.id = c.id
                JOIN x.d d ON c.id = d.id
                JOIN x.e e ON d.id = e.id
                JOIN x.f f ON e.id = f.id"""))
                .hasMessageContaining("at most 5");
    }

    @Test
    void letsALaterJoinHangOffAnyEarlierAliasNotOnlyTheFirst() {
        FederatedQuery query = parser.parse(
                "SELECT a.id FROM x.a a JOIN x.b b ON a.id = b.id JOIN x.c c ON b.name = c.name");
        assertThat(query.joins().get(1).leftAlias()).isEqualTo("b");
        assertThat(query.joins().get(1).leftKey()).isEqualTo("name");
        assertThat(query.joins().get(1).rightAlias()).isEqualTo("c");
    }

    @Test
    void rejectsAJoinThatReferencesAnAliasNobodyDeclared() {
        assertThatThrownBy(() -> parser.parse(
                "SELECT a.id FROM x.a a JOIN x.b b ON zz.id = b.id"))
                .hasMessageContaining("unknown alias");
    }

    @Test
    void keepsWildcardProjectionsIntact() {
        assertThat(parser.parse("SELECT * FROM x.a a JOIN x.b b ON a.id = b.id").projection())
                .containsExactly("*");
        assertThat(parser.parse("SELECT B.* FROM x.a a JOIN x.b b ON a.id = b.id").projection())
                .containsExactly("b.*");
    }

    @Test
    void rejectsALimitThatIsNotAWholeNumber() {
        assertThatThrownBy(() -> parser.parse(
                "SELECT a.id FROM x.a a JOIN x.b b ON a.id = b.id LIMIT ten"))
                .hasMessageContaining("LIMIT must be a whole number");
    }

    @Test
    void rejectsACompoundOnConditionBecauseOnlyOneEqualityIsSupported() {
        assertThatThrownBy(() -> parser.parse(
                "SELECT a.id FROM x.a a JOIN x.b b ON a.id = b.id AND a.name = b.name"))
                .hasMessageContaining("only one equality");
    }

    @Test
    void explainsAStatementThatNeverMentionsSelectOrFrom() {
        assertThatThrownBy(() -> parser.parse("DELETE FROM x.a"))
                .hasMessageContaining("must start with SELECT");
        assertThatThrownBy(() -> parser.parse("SELECT a.id"))
                .hasMessageContaining("no FROM clause");
    }

    @Test
    void explainsAJoinThatForgotItsOnClause() {
        assertThatThrownBy(() -> parser.parse("SELECT a.id FROM x.a a JOIN x.b b"))
                .hasMessageContaining("no ON clause");
    }

    @Test
    void explainsASourceThatForgotItsAlias() {
        assertThatThrownBy(() -> parser.parse("SELECT a.id FROM x.a JOIN x.b b ON a.id = b.id"))
                .hasMessageContaining("alias");
    }

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
