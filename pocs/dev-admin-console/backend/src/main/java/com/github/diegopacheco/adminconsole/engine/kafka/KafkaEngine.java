package com.github.diegopacheco.adminconsole.engine.kafka;

import com.github.diegopacheco.adminconsole.engine.Engine;
import com.github.diegopacheco.adminconsole.engine.PageRequest;
import com.github.diegopacheco.adminconsole.engine.QueryResult;
import com.github.diegopacheco.adminconsole.engine.SchemaNode;
import com.github.diegopacheco.adminconsole.project.ConnectionConfig;
import com.github.diegopacheco.adminconsole.project.ConnectionKind;
import com.github.diegopacheco.adminconsole.registry.ConnectionRegistry;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.apache.kafka.clients.admin.Admin;
import org.apache.kafka.clients.admin.OffsetSpec;
import org.apache.kafka.clients.admin.TopicDescription;
import org.apache.kafka.clients.consumer.ConsumerRecord;
import org.apache.kafka.clients.consumer.ConsumerRecords;
import org.apache.kafka.clients.consumer.KafkaConsumer;
import org.apache.kafka.common.TopicPartition;
import org.springframework.stereotype.Component;

@Component
public class KafkaEngine implements Engine {
    private final ConnectionRegistry registry;
    private final KafkaCommandParser parser;
    private final KafkaReadOnlyGuard guard;
    private final ObserverConsumerFactory consumers;

    public KafkaEngine(ConnectionRegistry registry, KafkaCommandParser parser, KafkaReadOnlyGuard guard,
                       ObserverConsumerFactory consumers) {
        this.registry = registry;
        this.parser = parser;
        this.guard = guard;
        this.consumers = consumers;
    }

    @Override
    public ConnectionKind kind() {
        return ConnectionKind.KAFKA;
    }

    @Override
    public void assertReadOnly(String statement) {
        guard.assertOperationAllowed(parser.operation(statement));
    }

    @Override
    public List<SchemaNode> schema(ConnectionConfig config) {
        Admin admin = registry.kafka(config);
        List<SchemaNode> topics = new ArrayList<>();
        try {
            Set<String> names = admin.listTopics().names().get();
            Map<String, TopicDescription> described = admin.describeTopics(names).allTopicNames().get();
            for (String name : names.stream().sorted().toList()) {
                TopicDescription description = described.get(name);
                List<SchemaNode> partitions = new ArrayList<>();
                for (var partition : description.partitions()) {
                    TopicPartition topicPartition = new TopicPartition(name, partition.partition());
                    long earliest = offset(admin, topicPartition, OffsetSpec.earliest());
                    long latest = offset(admin, topicPartition, OffsetSpec.latest());
                    partitions.add(SchemaNode.leaf("partition-" + partition.partition(), "partition",
                            "offsets " + earliest + ".." + latest + " (" + (latest - earliest) + " records), leader "
                                    + partition.leader().id()));
                }
                topics.add(new SchemaNode(name, "topic", partitions.size() + " partitions", partitions));
            }
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("kafka request was interrupted", error);
        } catch (Exception error) {
            throw new IllegalStateException("kafka request failed: " + error.getMessage(), error);
        }
        return topics;
    }

    @Override
    public QueryResult query(ConnectionConfig config, String statement, PageRequest page) {
        guard.assertOperationAllowed(parser.operation(statement));
        KafkaCommand command = parser.parse(statement);
        return switch (command.operation()) {
            case "list" -> list(config, command, page);
            case "describe" -> describe(config, command, page);
            case "offsets" -> offsets(config, command, page);
            case "consume" -> consume(config, command, page);
            default -> throw new IllegalArgumentException("unsupported operation: " + command.operation());
        };
    }

    private QueryResult list(ConnectionConfig config, KafkaCommand command, PageRequest page) {
        Admin admin = registry.kafka(config);
        String target = command.target() == null ? "topics" : command.target();
        try {
            if (target.startsWith("group")) {
                List<Map<String, Object>> rows = new ArrayList<>();
                admin.listConsumerGroups().all().get().forEach(group -> {
                    Map<String, Object> row = new LinkedHashMap<>();
                    row.put("group", group.groupId());
                    row.put("state", group.state().map(Enum::name).orElse("unknown"));
                    rows.add(row);
                });
                return window(List.of("group", "state"), rows, page);
            }
            List<Map<String, Object>> rows = admin.listTopics().names().get().stream().sorted()
                    .map(name -> {
                        Map<String, Object> row = new LinkedHashMap<>();
                        row.put("topic", name);
                        return row;
                    }).toList();
            return window(List.of("topic"), new ArrayList<>(rows), page);
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("kafka request was interrupted", error);
        } catch (Exception error) {
            throw new IllegalStateException("kafka request failed: " + error.getMessage(), error);
        }
    }

    private QueryResult describe(ConnectionConfig config, KafkaCommand command, PageRequest page) {
        Admin admin = registry.kafka(config);
        String target = command.target() == null ? "" : command.target();
        String topic = target.startsWith("topic ") ? target.substring(6) : target;
        try {
            TopicDescription description = admin.describeTopics(List.of(topic)).allTopicNames().get().get(topic);
            List<Map<String, Object>> rows = new ArrayList<>();
            for (var partition : description.partitions()) {
                Map<String, Object> row = new LinkedHashMap<>();
                row.put("partition", String.valueOf(partition.partition()));
                row.put("leader", String.valueOf(partition.leader().id()));
                row.put("replicas", String.valueOf(partition.replicas().size()));
                row.put("isr", String.valueOf(partition.isr().size()));
                rows.add(row);
            }
            return window(List.of("partition", "leader", "replicas", "isr"), rows, page);
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("kafka request was interrupted", error);
        } catch (Exception error) {
            throw new IllegalStateException("kafka request failed: " + error.getMessage(), error);
        }
    }

    private QueryResult offsets(ConnectionConfig config, KafkaCommand command, PageRequest page) {
        Admin admin = registry.kafka(config);
        try {
            TopicDescription description = admin.describeTopics(List.of(command.target()))
                    .allTopicNames().get().get(command.target());
            List<Map<String, Object>> rows = new ArrayList<>();
            for (var partition : description.partitions()) {
                TopicPartition topicPartition = new TopicPartition(command.target(), partition.partition());
                long earliest = offset(admin, topicPartition, OffsetSpec.earliest());
                long latest = offset(admin, topicPartition, OffsetSpec.latest());
                Map<String, Object> row = new LinkedHashMap<>();
                row.put("partition", String.valueOf(partition.partition()));
                row.put("earliest", String.valueOf(earliest));
                row.put("latest", String.valueOf(latest));
                row.put("records", String.valueOf(latest - earliest));
                rows.add(row);
            }
            return window(List.of("partition", "earliest", "latest", "records"), rows, page);
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("kafka request was interrupted", error);
        } catch (Exception error) {
            throw new IllegalStateException("kafka request failed: " + error.getMessage(), error);
        }
    }

    private QueryResult consume(ConnectionConfig config, KafkaCommand command, PageRequest page) {
        Admin admin = registry.kafka(config);
        int limit = command.limit() > 0 ? Math.min(command.limit(), page.size()) : page.size();
        List<String> columns = List.of("partition", "offset", "timestamp", "key", "value");
        try (KafkaConsumer<String, String> consumer = consumers.create(config, limit)) {
            TopicDescription description = admin.describeTopics(List.of(command.target()))
                    .allTopicNames().get().get(command.target());
            List<TopicPartition> partitions = description.partitions().stream()
                    .filter(partition -> command.partition() == null || partition.partition() == command.partition())
                    .map(partition -> new TopicPartition(command.target(), partition.partition()))
                    .toList();
            consumer.assign(partitions);
            Map<TopicPartition, Long> starts = startOffsets(consumer, partitions, command, page);
            starts.forEach(consumer::seek);
            List<Map<String, Object>> rows = new ArrayList<>();
            Map<TopicPartition, Long> next = new LinkedHashMap<>(starts);
            long deadline = System.currentTimeMillis() + 5000;
            while (rows.size() < limit && System.currentTimeMillis() < deadline) {
                ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(500));
                if (records.isEmpty()) {
                    break;
                }
                for (ConsumerRecord<String, String> record : records) {
                    if (rows.size() == limit) {
                        break;
                    }
                    Map<String, Object> row = new LinkedHashMap<>();
                    row.put("partition", String.valueOf(record.partition()));
                    row.put("offset", String.valueOf(record.offset()));
                    row.put("timestamp", String.valueOf(record.timestamp()));
                    row.put("key", record.key());
                    row.put("value", record.value());
                    rows.add(row);
                    next.put(new TopicPartition(record.topic(), record.partition()), record.offset() + 1);
                }
            }
            Map<TopicPartition, Long> ends = consumer.endOffsets(partitions);
            boolean hasMore = ends.entrySet().stream()
                    .anyMatch(entry -> next.getOrDefault(entry.getKey(), 0L) < entry.getValue());
            return QueryResult.of(columns, rows, page.pageNumber(), hasMore ? encode(next) : null, hasMore);
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("kafka request was interrupted", error);
        } catch (Exception error) {
            throw new IllegalStateException("kafka request failed: " + error.getMessage(), error);
        }
    }

    private Map<TopicPartition, Long> startOffsets(KafkaConsumer<String, String> consumer,
                                                   List<TopicPartition> partitions, KafkaCommand command,
                                                   PageRequest page) {
        Map<TopicPartition, Long> starts = new LinkedHashMap<>();
        if (!page.isFirst()) {
            Map<String, Long> decoded = decode(page.cursor());
            for (TopicPartition partition : partitions) {
                starts.put(partition, decoded.getOrDefault(key(partition), 0L));
            }
            return starts;
        }
        if (command.offset() != null) {
            partitions.forEach(partition -> starts.put(partition, command.offset()));
            return starts;
        }
        Map<TopicPartition, Long> bounds = "latest".equals(command.from())
                ? consumer.endOffsets(partitions)
                : consumer.beginningOffsets(partitions);
        bounds.forEach(starts::put);
        return starts;
    }

    private long offset(Admin admin, TopicPartition partition, OffsetSpec spec) throws Exception {
        return admin.listOffsets(Map.of(partition, spec)).all().get().get(partition).offset();
    }

    private QueryResult window(List<String> columns, List<Map<String, Object>> rows, PageRequest page) {
        int offset = page.isFirst() ? 0 : Integer.parseInt(page.cursor());
        List<Map<String, Object>> window = rows.stream().skip(offset).limit(page.size()).toList();
        boolean hasMore = rows.size() > offset + window.size();
        return QueryResult.of(columns, window, page.pageNumber(),
                hasMore ? String.valueOf(offset + window.size()) : null, hasMore);
    }

    private String key(TopicPartition partition) {
        return partition.topic() + ":" + partition.partition();
    }

    private String encode(Map<TopicPartition, Long> offsets) {
        return offsets.entrySet().stream()
                .map(entry -> key(entry.getKey()) + "=" + entry.getValue())
                .reduce((left, right) -> left + "," + right)
                .orElse("");
    }

    private Map<String, Long> decode(String cursor) {
        Map<String, Long> offsets = new LinkedHashMap<>();
        for (String part : cursor.split(",")) {
            int separator = part.lastIndexOf('=');
            if (separator > 0) {
                offsets.put(part.substring(0, separator), Long.parseLong(part.substring(separator + 1)));
            }
        }
        return offsets;
    }
}
