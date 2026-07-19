package com.github.diegopacheco.adminconsole.engine;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.github.diegopacheco.adminconsole.engine.cassandra.CassandraEngine;
import com.github.diegopacheco.adminconsole.engine.redis.RedisEngine;
import com.github.diegopacheco.adminconsole.project.ConnectionConfig;
import com.github.diegopacheco.adminconsole.project.ConnectionKind;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

@Tag("integration-test")
@SpringBootTest
class CassandraRedisEngineIT {
    @Autowired
    private CassandraEngine cassandra;
    @Autowired
    private RedisEngine redis;

    private final ConnectionConfig cassandraConfig = new ConnectionConfig(9101L, 1L, "demo-cassandra",
            ConnectionKind.CASSANDRA, "localhost", 9042, null, "shop", "datacenter1", null, null, null, "tester");

    private final ConnectionConfig redisConfig = new ConnectionConfig(9102L, 1L, "demo-redis",
            ConnectionKind.REDIS, "localhost", 6379, null, null, null, null, null, null, "tester");

    @Test
    void listsCassandraTablesWithPartitionAndClusteringKeysMarkedSoOperatorsSeeTheKeyStructure() {
        List<SchemaNode> tables = cassandra.schema(cassandraConfig);
        assertThat(tables).extracting(SchemaNode::name).contains("events_by_customer", "sessions");
        SchemaNode events = tables.stream().filter(node -> node.name().equals("events_by_customer"))
                .findFirst().orElseThrow();
        assertThat(events.children()).extracting(SchemaNode::detail)
                .anyMatch(detail -> detail.contains("partition_key"))
                .anyMatch(detail -> detail.contains("clustering"));
    }

    @Test
    void runsASelectAgainstCassandra() {
        QueryResult result = cassandra.query(cassandraConfig,
                "SELECT customer_id, event_type FROM events_by_customer", PageRequest.first(10));
        assertThat(result.columns()).containsExactly("customer_id", "event_type");
        assertThat(result.rows()).hasSize(10);
    }

    @Test
    void pagesCassandraWithItsNativePagingStateWithoutRepeatingRows() {
        QueryResult first = cassandra.query(cassandraConfig,
                "SELECT event_id FROM events_by_customer", PageRequest.first(25));
        assertThat(first.rows()).hasSize(25);
        assertThat(first.hasMore()).isTrue();
        QueryResult second = cassandra.query(cassandraConfig, "SELECT event_id FROM events_by_customer",
                new PageRequest(25, first.nextCursor(), 2));
        Set<Object> firstIds = new HashSet<>(first.rows().stream().map(row -> row.get("event_id")).toList());
        Set<Object> secondIds = new HashSet<>(second.rows().stream().map(row -> row.get("event_id")).toList());
        assertThat(second.rows()).hasSize(25);
        assertThat(firstIds).doesNotContainAnyElementsOf(secondIds);
    }

    @Test
    void rejectsEveryCassandraWriteIncludingTruncateAndDdl() {
        assertThatThrownBy(() -> cassandra.query(cassandraConfig,
                "INSERT INTO sessions (session_id) VALUES (uuid())", PageRequest.first(10)))
                .isInstanceOf(ReadOnlyViolation.class);
        assertThatThrownBy(() -> cassandra.query(cassandraConfig, "TRUNCATE sessions", PageRequest.first(10)))
                .isInstanceOf(ReadOnlyViolation.class);
        assertThatThrownBy(() -> cassandra.query(cassandraConfig, "DROP TABLE sessions", PageRequest.first(10)))
                .isInstanceOf(ReadOnlyViolation.class);
    }

    @Test
    void listsRedisKeysWithTheirTypesSoTheLeftPanelCanBadgeThem() {
        List<SchemaNode> keys = redis.schema(redisConfig);
        assertThat(keys).extracting(SchemaNode::name).contains("session:abc123", "queue:emails", "leaderboard");
        assertThat(keys).extracting(SchemaNode::kind).contains("hash", "list", "zset", "set", "string", "stream");
    }

    @Test
    void expandsAHashIntoFieldsSoFoldingAKeyShowsItsContents() {
        SchemaNode session = redis.schema(redisConfig).stream()
                .filter(node -> node.name().equals("session:abc123")).findFirst().orElseThrow();
        assertThat(session.children()).extracting(SchemaNode::name).contains("user", "ip", "agent");
        assertThat(session.children()).extracting(SchemaNode::detail).contains("diego");
    }

    @Test
    void runsAReadOnlyRedisCommand() {
        QueryResult result = redis.query(redisConfig, "GET config:app:name", PageRequest.first(10));
        assertThat(result.rows().getFirst().get("value")).isEqualTo("admin-console-demo");
    }

    @Test
    void shapesAHashIntoFieldAndValueColumns() {
        QueryResult result = redis.query(redisConfig, "HGETALL session:abc123", PageRequest.first(10));
        assertThat(result.columns()).containsExactly("field", "value");
        assertThat(result.rows()).extracting(row -> row.get("field")).contains("user", "ip");
    }

    @Test
    void rejectsRedisWritesUsingTheServersOwnCommandFlagsRatherThanAHandMaintainedList() {
        assertThatThrownBy(() -> redis.query(redisConfig, "SET config:app:name hacked", PageRequest.first(10)))
                .isInstanceOf(ReadOnlyViolation.class);
        assertThatThrownBy(() -> redis.query(redisConfig, "DEL config:app:name", PageRequest.first(10)))
                .isInstanceOf(ReadOnlyViolation.class);
        assertThatThrownBy(() -> redis.query(redisConfig, "FLUSHALL", PageRequest.first(10)))
                .isInstanceOf(ReadOnlyViolation.class);
    }

    @Test
    void rejectsLuaEvalBecauseAScriptCanWriteEvenThoughTheCommandLooksHarmless() {
        assertThatThrownBy(() -> redis.query(redisConfig,
                "EVAL \"redis.call('set', KEYS[1], 'pwned')\" 1 config:app:name", PageRequest.first(10)))
                .isInstanceOf(ReadOnlyViolation.class);
    }

    @Test
    void leavesRedisDataUntouchedAfterAllRejectedWriteAttempts() {
        assertThat(redis.query(redisConfig, "GET config:app:name", PageRequest.first(10))
                .rows().getFirst().get("value")).isEqualTo("admin-console-demo");
    }
}
