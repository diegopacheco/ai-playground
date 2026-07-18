package com.github.diegopacheco.adminconsole.engine.redis;

import com.github.diegopacheco.adminconsole.engine.Engine;
import com.github.diegopacheco.adminconsole.engine.PageRequest;
import com.github.diegopacheco.adminconsole.engine.QueryResult;
import com.github.diegopacheco.adminconsole.engine.SchemaNode;
import com.github.diegopacheco.adminconsole.project.ConnectionConfig;
import com.github.diegopacheco.adminconsole.project.ConnectionKind;
import com.github.diegopacheco.adminconsole.registry.ConnectionRegistry;
import io.lettuce.core.KeyScanCursor;
import io.lettuce.core.ScanArgs;
import io.lettuce.core.ScanCursor;
import io.lettuce.core.api.sync.RedisCommands;
import io.lettuce.core.codec.StringCodec;
import io.lettuce.core.output.NestedMultiOutput;
import io.lettuce.core.protocol.CommandArgs;
import io.lettuce.core.protocol.CommandType;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.springframework.stereotype.Component;

@Component
public class RedisEngine implements Engine {
    private final ConnectionRegistry registry;
    private final RedisCommandParser parser;
    private final RedisReadOnlyGuard guard;

    public RedisEngine(ConnectionRegistry registry, RedisCommandParser parser, RedisReadOnlyGuard guard) {
        this.registry = registry;
        this.parser = parser;
        this.guard = guard;
    }

    @Override
    public ConnectionKind kind() {
        return ConnectionKind.REDIS;
    }

    @Override
    public void assertReadOnly(String statement) {
        throw new UnsupportedOperationException("redis read-only checks require a live connection");
    }

    public void assertReadOnly(ConnectionConfig config, String statement) {
        List<String> parts = parser.parse(statement);
        guard.assertReadOnly(parts.getFirst(), commandFlags(config, parts.getFirst()));
    }

    @Override
    public List<SchemaNode> schema(ConnectionConfig config) {
        RedisCommands<String, String> commands = registry.redis(config).sync();
        List<SchemaNode> nodes = new ArrayList<>();
        KeyScanCursor<String> cursor = commands.scan(ScanArgs.Builder.limit(200));
        int guardrail = 0;
        while (guardrail++ < 25) {
            for (String key : cursor.getKeys()) {
                String type = commands.type(key);
                nodes.add(new SchemaNode(key, type, describe(commands, key, type), children(commands, key, type)));
            }
            if (cursor.isFinished()) {
                break;
            }
            cursor = commands.scan(cursor, ScanArgs.Builder.limit(200));
        }
        nodes.sort((left, right) -> left.name().compareTo(right.name()));
        return nodes;
    }

    @Override
    public QueryResult query(ConnectionConfig config, String statement, PageRequest page) {
        List<String> parts = parser.parse(statement);
        String operation = parts.getFirst().toUpperCase();
        guard.assertReadOnly(operation, commandFlags(config, operation));
        RedisCommands<String, String> commands = registry.redis(config).sync();
        CommandArgs<String, String> arguments = new CommandArgs<>(StringCodec.UTF8);
        parts.stream().skip(1).forEach(arguments::add);
        Object raw = commands.dispatch(CommandType.valueOf(operation),
                new NestedMultiOutput<>(StringCodec.UTF8), arguments);
        return shape(operation, raw, page);
    }

    private QueryResult shape(String operation, Object raw, PageRequest page) {
        List<Map<String, Object>> rows = new ArrayList<>();
        List<String> columns;
        if (raw instanceof List<?> values) {
            if (operation.equals("HGETALL")) {
                columns = List.of("field", "value");
                for (int index = 0; index + 1 < values.size(); index += 2) {
                    rows.add(pair("field", values.get(index), "value", values.get(index + 1)));
                }
            } else {
                columns = List.of("value");
                values.forEach(value -> rows.add(single("value", value)));
            }
        } else {
            columns = List.of("value");
            rows.add(single("value", raw));
        }
        int offset = page.isFirst() ? 0 : Integer.parseInt(page.cursor());
        List<Map<String, Object>> window = rows.stream().skip(offset).limit(page.size()).toList();
        boolean hasMore = rows.size() > offset + window.size();
        return QueryResult.of(columns, window, page.pageNumber(),
                hasMore ? String.valueOf(offset + window.size()) : null, hasMore);
    }

    private Map<String, Object> single(String column, Object value) {
        Map<String, Object> row = new LinkedHashMap<>();
        row.put(column, value == null ? null : String.valueOf(value));
        return row;
    }

    private Map<String, Object> pair(String first, Object firstValue, String second, Object secondValue) {
        Map<String, Object> row = new LinkedHashMap<>();
        row.put(first, String.valueOf(firstValue));
        row.put(second, String.valueOf(secondValue));
        return row;
    }

    private String describe(RedisCommands<String, String> commands, String key, String type) {
        return switch (type) {
            case "hash" -> commands.hlen(key) + " fields";
            case "list" -> commands.llen(key) + " items";
            case "set" -> commands.scard(key) + " members";
            case "zset" -> commands.zcard(key) + " members";
            case "stream" -> commands.xlen(key) + " entries";
            default -> "string";
        };
    }

    private List<SchemaNode> children(RedisCommands<String, String> commands, String key, String type) {
        return switch (type) {
            case "hash" -> commands.hgetall(key).entrySet().stream()
                    .limit(100)
                    .map(entry -> SchemaNode.leaf(entry.getKey(), "field", entry.getValue()))
                    .toList();
            case "list" -> commands.lrange(key, 0, 99).stream()
                    .map(value -> SchemaNode.leaf(value, "item", ""))
                    .toList();
            case "set" -> commands.smembers(key).stream()
                    .limit(100)
                    .map(value -> SchemaNode.leaf(value, "member", ""))
                    .toList();
            case "zset" -> commands.zrangeWithScores(key, 0, 99).stream()
                    .map(value -> SchemaNode.leaf(value.getValue(), "member", String.valueOf(value.getScore())))
                    .toList();
            default -> List.of();
        };
    }

    private Set<String> commandFlags(ConnectionConfig config, String command) {
        RedisCommands<String, String> commands = registry.redis(config).sync();
        CommandArgs<String, String> arguments = new CommandArgs<>(StringCodec.UTF8)
                .add("INFO").add(command);
        Object raw = commands.dispatch(CommandType.COMMAND, new NestedMultiOutput<>(StringCodec.UTF8), arguments);
        Set<String> flags = new LinkedHashSet<>();
        if (raw instanceof List<?> entries && !entries.isEmpty() && entries.getFirst() instanceof List<?> info
                && info.size() > 2 && info.get(2) instanceof List<?> rawFlags) {
            rawFlags.forEach(flag -> flags.add(String.valueOf(flag)));
        }
        return flags;
    }
}
