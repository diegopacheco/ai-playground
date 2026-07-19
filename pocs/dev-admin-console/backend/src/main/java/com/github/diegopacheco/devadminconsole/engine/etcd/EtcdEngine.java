package com.github.diegopacheco.devadminconsole.engine.etcd;

import com.github.diegopacheco.devadminconsole.engine.Engine;
import com.github.diegopacheco.devadminconsole.engine.PageRequest;
import com.github.diegopacheco.devadminconsole.engine.QueryResult;
import com.github.diegopacheco.devadminconsole.engine.SchemaNode;
import com.github.diegopacheco.devadminconsole.project.ConnectionConfig;
import com.github.diegopacheco.devadminconsole.project.ConnectionKind;
import com.github.diegopacheco.devadminconsole.registry.ConnectionRegistry;
import io.etcd.jetcd.ByteSequence;
import io.etcd.jetcd.KeyValue;
import io.etcd.jetcd.options.GetOption;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import org.springframework.stereotype.Component;

@Component
public class EtcdEngine implements Engine {
    private static final ByteSequence ALL_KEYS = ByteSequence.from(new byte[]{0});

    private final ConnectionRegistry registry;
    private final EtcdCommandParser parser;
    private final EtcdReadOnlyGuard guard;
    private final PrefixTreeBuilder treeBuilder;

    public EtcdEngine(ConnectionRegistry registry, EtcdCommandParser parser, EtcdReadOnlyGuard guard,
                      PrefixTreeBuilder treeBuilder) {
        this.registry = registry;
        this.parser = parser;
        this.guard = guard;
        this.treeBuilder = treeBuilder;
    }

    @Override
    public ConnectionKind kind() {
        return ConnectionKind.ETCD;
    }

    @Override
    public void assertReadOnly(String statement) {
        guard.assertReadOnly(parser.parse(statement));
    }

    @Override
    public List<SchemaNode> schema(ConnectionConfig config) {
        Map<String, String> keys = new LinkedHashMap<>();
        for (KeyValue entry : all(config)) {
            keys.put(entry.getKey().toString(StandardCharsets.UTF_8), entry.getValue().toString(StandardCharsets.UTF_8));
        }
        return treeBuilder.build(keys);
    }

    @Override
    public QueryResult query(ConnectionConfig config, String statement, PageRequest page) {
        EtcdCommand command = parser.parse(statement);
        guard.assertReadOnly(command);
        GetOption.Builder options = GetOption.builder();
        if (command.prefix()) {
            options.isPrefix(true);
        }
        if (command.rangeEnd() != null) {
            options.withRange(ByteSequence.from(command.rangeEnd(), StandardCharsets.UTF_8));
        }
        List<KeyValue> entries = get(config, ByteSequence.from(command.key(), StandardCharsets.UTF_8), options.build());
        List<String> columns = List.of("key", "value", "version", "mod_revision");
        int offset = page.isFirst() ? 0 : Integer.parseInt(page.cursor());
        List<Map<String, Object>> rows = new ArrayList<>();
        for (KeyValue entry : entries.stream().skip(offset).limit(page.size()).toList()) {
            Map<String, Object> row = new LinkedHashMap<>();
            row.put("key", entry.getKey().toString(StandardCharsets.UTF_8));
            row.put("value", entry.getValue().toString(StandardCharsets.UTF_8));
            row.put("version", String.valueOf(entry.getVersion()));
            row.put("mod_revision", String.valueOf(entry.getModRevision()));
            rows.add(row);
        }
        boolean hasMore = entries.size() > offset + rows.size();
        return QueryResult.of(columns, rows, page.pageNumber(),
                hasMore ? String.valueOf(offset + rows.size()) : null, hasMore);
    }

    private List<KeyValue> all(ConnectionConfig config) {
        return get(config, ALL_KEYS, GetOption.builder().isPrefix(false).withRange(ALL_KEYS).build());
    }

    private List<KeyValue> get(ConnectionConfig config, ByteSequence key, GetOption options) {
        try {
            return registry.etcd(config).getKVClient().get(key, options).get(10, TimeUnit.SECONDS).getKvs();
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("etcd request was interrupted", error);
        } catch (Exception error) {
            throw new IllegalStateException("etcd request failed: " + error.getMessage(), error);
        }
    }
}
