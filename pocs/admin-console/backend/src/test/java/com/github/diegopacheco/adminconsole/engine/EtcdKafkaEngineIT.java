package com.github.diegopacheco.adminconsole.engine;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.github.diegopacheco.adminconsole.engine.etcd.EtcdEngine;
import com.github.diegopacheco.adminconsole.engine.kafka.KafkaEngine;
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
class EtcdKafkaEngineIT {
    @Autowired
    private EtcdEngine etcd;
    @Autowired
    private KafkaEngine kafka;

    private final ConnectionConfig etcdConfig = new ConnectionConfig(9201L, 1L, "demo-etcd", ConnectionKind.ETCD,
            "localhost", 2379, null, null, null, null, null, null, "tester");

    private final ConnectionConfig kafkaConfig = new ConnectionConfig(9202L, 1L, "demo-kafka", ConnectionKind.KAFKA,
            "localhost", 9092, null, null, null, null, null, null, "tester");

    @Test
    void foldsEtcdKeysIntoAPrefixTreeSinceEtcdHasNoTablesOrColumns() {
        List<SchemaNode> roots = etcd.schema(etcdConfig);
        assertThat(roots).extracting(SchemaNode::name).contains("config", "service", "leases");
        SchemaNode config = roots.stream().filter(node -> node.name().equals("config")).findFirst().orElseThrow();
        assertThat(config.children()).extracting(SchemaNode::name).contains("app", "database", "features");
        SchemaNode app = config.children().stream().filter(node -> node.name().equals("app")).findFirst().orElseThrow();
        assertThat(app.children()).extracting(SchemaNode::name).contains("name", "version", "log-level");
        assertThat(app.children()).extracting(SchemaNode::detail).contains("admin-console");
    }

    @Test
    void readsASingleEtcdKey() {
        QueryResult result = etcd.query(etcdConfig, "get /config/app/name", PageRequest.first(10));
        assertThat(result.columns()).containsExactly("key", "value", "version", "mod_revision");
        assertThat(result.rows().getFirst().get("value")).isEqualTo("admin-console");
    }

    @Test
    void readsAnEtcdPrefix() {
        QueryResult result = etcd.query(etcdConfig, "get /config/database --prefix", PageRequest.first(10));
        assertThat(result.rows()).extracting(row -> row.get("key"))
                .contains("/config/database/host", "/config/database/port", "/config/database/pool-size");
    }

    @Test
    void pagesEtcdPrefixResultsWithoutRepeats() {
        QueryResult first = etcd.query(etcdConfig, "get /leases --prefix", PageRequest.first(15));
        assertThat(first.rows()).hasSize(15);
        assertThat(first.hasMore()).isTrue();
        QueryResult second = etcd.query(etcdConfig, "get /leases --prefix",
                new PageRequest(15, first.nextCursor(), 2));
        Set<Object> firstKeys = new HashSet<>(first.rows().stream().map(row -> row.get("key")).toList());
        Set<Object> secondKeys = new HashSet<>(second.rows().stream().map(row -> row.get("key")).toList());
        assertThat(firstKeys).doesNotContainAnyElementsOf(secondKeys);
    }

    @Test
    void rejectsEveryEtcdWriteOperation() {
        assertThatThrownBy(() -> etcd.query(etcdConfig, "put /config/app/name hacked", PageRequest.first(10)))
                .isInstanceOf(ReadOnlyViolation.class);
        assertThatThrownBy(() -> etcd.query(etcdConfig, "del /config/app/name", PageRequest.first(10)))
                .isInstanceOf(ReadOnlyViolation.class);
        assertThatThrownBy(() -> etcd.query(etcdConfig, "txn", PageRequest.first(10)))
                .isInstanceOf(ReadOnlyViolation.class);
        assertThat(etcd.query(etcdConfig, "get /config/app/name", PageRequest.first(10))
                .rows().getFirst().get("value")).isEqualTo("admin-console");
    }

    @Test
    void listsKafkaTopicsWithPartitionOffsetsSoLagIsVisibleInTheTree() {
        List<SchemaNode> topics = kafka.schema(kafkaConfig);
        assertThat(topics).extracting(SchemaNode::name).contains("orders.events", "payments.events");
        SchemaNode orders = topics.stream().filter(node -> node.name().equals("orders.events"))
                .findFirst().orElseThrow();
        assertThat(orders.children()).hasSize(3);
        assertThat(orders.children()).extracting(SchemaNode::detail).allMatch(detail -> detail.contains("offsets"));
    }

    @Test
    void consumesABoundedWindowOfMessagesAsATable() {
        QueryResult result = kafka.query(kafkaConfig, "consume orders.events --limit 10", PageRequest.first(10));
        assertThat(result.columns()).containsExactly("partition", "offset", "timestamp", "key", "value");
        assertThat(result.rows()).hasSize(10);
        assertThat(result.rows().getFirst().get("key")).asString().startsWith("order-");
    }

    @Test
    void pagesKafkaByOffsetWithoutRereadingTheSameRecords() {
        QueryResult first = kafka.query(kafkaConfig, "consume orders.events", PageRequest.first(20));
        assertThat(first.rows()).hasSize(20);
        assertThat(first.hasMore()).isTrue();
        QueryResult second = kafka.query(kafkaConfig, "consume orders.events",
                new PageRequest(20, first.nextCursor(), 2));
        Set<String> firstIds = first.rows().stream()
                .map(row -> row.get("partition") + ":" + row.get("offset")).collect(java.util.stream.Collectors.toSet());
        Set<String> secondIds = second.rows().stream()
                .map(row -> row.get("partition") + ":" + row.get("offset")).collect(java.util.stream.Collectors.toSet());
        assertThat(firstIds).doesNotContainAnyElementsOf(secondIds);
    }

    @Test
    void describesTopicPartitionsAndOffsets() {
        assertThat(kafka.query(kafkaConfig, "describe topic orders.events", PageRequest.first(10)).rows()).hasSize(3);
        QueryResult offsets = kafka.query(kafkaConfig, "offsets orders.events", PageRequest.first(10));
        assertThat(offsets.columns()).containsExactly("partition", "earliest", "latest", "records");
        assertThat(offsets.rows()).hasSize(3);
    }

    @Test
    void rejectsEveryKafkaWriteOperationIncludingOffsetResetsThatWouldDisturbRealConsumers() {
        assertThatThrownBy(() -> kafka.query(kafkaConfig, "produce orders.events hello", PageRequest.first(10)))
                .isInstanceOf(ReadOnlyViolation.class);
        assertThatThrownBy(() -> kafka.query(kafkaConfig, "delete topic orders.events", PageRequest.first(10)))
                .isInstanceOf(ReadOnlyViolation.class);
        assertThatThrownBy(() -> kafka.query(kafkaConfig, "reset offsets shop-workers", PageRequest.first(10)))
                .isInstanceOf(ReadOnlyViolation.class);
    }

    @Test
    void doesNotDisturbAnExistingConsumerGroupOffsetWhichIsTheWholePointOfObserverSemantics() throws Exception {
        var admin = org.apache.kafka.clients.admin.Admin.create(java.util.Map.of("bootstrap.servers", "localhost:9092"));
        var before = admin.listConsumerGroupOffsets("shop-workers").partitionsToOffsetAndMetadata().get();
        kafka.query(kafkaConfig, "consume orders.events --limit 50", PageRequest.first(50));
        var after = admin.listConsumerGroupOffsets("shop-workers").partitionsToOffsetAndMetadata().get();
        admin.close();
        assertThat(after).isEqualTo(before);
    }
}
