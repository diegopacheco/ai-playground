package com.github.diegopacheco.devadminconsole.federation;

import com.github.diegopacheco.devadminconsole.engine.EngineRegistry;
import com.github.diegopacheco.devadminconsole.engine.PageRequest;
import com.github.diegopacheco.devadminconsole.engine.QueryResult;
import com.github.diegopacheco.devadminconsole.engine.SchemaNode;
import com.github.diegopacheco.devadminconsole.project.ConnectionConfig;
import com.github.diegopacheco.devadminconsole.project.ConnectionKind;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

@Service
public class FederatedExecutor {
    public record SideResult(String alias, String connectionName, String kind, String source, int rows,
                             boolean truncated) {}

    public record Result(List<String> columns, List<Map<String, Object>> rows, List<SideResult> sides,
                         long elapsedMs, String diagnostic) {}

    private static final int FETCH_PAGE = 100;

    private final EngineRegistry engines;
    private final int maxRowsPerSide;

    public FederatedExecutor(EngineRegistry engines,
                             @Value("${app.federation.max-rows-per-side:5000}") int maxRowsPerSide) {
        this.engines = engines;
        this.maxRowsPerSide = maxRowsPerSide;
    }

    public Result execute(FederatedQuery query, List<ConnectionConfig> available) {
        long started = System.currentTimeMillis();

        Map<String, ConnectionConfig> connectionByAlias = new LinkedHashMap<>();
        Map<String, QueryResult> fetched = new LinkedHashMap<>();
        List<SideResult> sideResults = new ArrayList<>();

        for (FederatedQuery.Side side : query.sides()) {
            ConnectionConfig connection = resolve(side.connectionName(), available);
            validateSource(connection, side);
            QueryResult rows = fetch(connection, side);
            connectionByAlias.put(side.alias().toLowerCase(), connection);
            fetched.put(side.alias().toLowerCase(), rows);
            sideResults.add(new SideResult(side.alias(), connection.name(), connection.kind().wireName(),
                    side.source(), rows.rows().size(), rows.hasMore()));
        }

        for (FederatedQuery.Join join : query.joins()) {
            requireKey(join.leftAlias(), connectionByAlias, query, join.leftKey(), fetched);
            requireKey(join.rightAlias(), connectionByAlias, query, join.rightKey(), fetched);
        }

        FederatedQuery.Side first = query.sides().getFirst();
        List<Map<String, Object>> accumulated = prefix(first.alias(), fetched.get(first.alias().toLowerCase()).rows());

        FederatedQuery.Join emptiedBy = null;
        for (FederatedQuery.Join join : query.joins()) {
            List<Map<String, Object>> right = prefix(join.rightAlias(),
                    fetched.get(join.rightAlias().toLowerCase()).rows());
            accumulated = hashJoin(accumulated, join.leftAlias() + "." + join.leftKey(),
                    right, join.rightAlias() + "." + join.rightKey(), join.leftJoin());
            if (accumulated.isEmpty()) {
                emptiedBy = join;
                break;
            }
            if (accumulated.size() > maxRowsPerSide) {
                accumulated = new ArrayList<>(accumulated.subList(0, maxRowsPerSide));
            }
        }

        List<Map<String, Object>> projected = new ArrayList<>();
        Set<String> columns = new LinkedHashSet<>();
        for (Map<String, Object> row : accumulated) {
            if (projected.size() >= query.limit()) {
                break;
            }
            Map<String, Object> out = project(query, row);
            columns.addAll(out.keySet());
            projected.add(out);
        }

        String diagnostic = projected.isEmpty()
                ? explainEmpty(query, fetched, emptiedBy)
                : null;
        return new Result(new ArrayList<>(columns), projected, sideResults,
                System.currentTimeMillis() - started, diagnostic);
    }

    private String explainEmpty(FederatedQuery query, Map<String, QueryResult> fetched,
                               FederatedQuery.Join emptiedBy) {
        for (FederatedQuery.Join join : emptiedBy == null ? query.joins() : List.of(emptiedBy)) {
            String left = samples(fetched.get(join.leftAlias().toLowerCase()), join.leftKey());
            String right = samples(fetched.get(join.rightAlias().toLowerCase()), join.rightKey());
            if (left == null || right == null) {
                continue;
            }
            return "nothing matched on " + join.leftAlias() + "." + join.leftKey() + " = "
                    + join.rightAlias() + "." + join.rightKey() + ". "
                    + join.leftAlias() + "." + join.leftKey() + " looks like: " + left + " · "
                    + join.rightAlias() + "." + join.rightKey() + " looks like: " + right
                    + " — these values do not overlap, so the join key is probably wrong.";
        }
        return "the join produced no rows.";
    }

    private String samples(QueryResult rows, String column) {
        if (rows == null || rows.rows().isEmpty()) {
            return null;
        }
        List<String> seen = new ArrayList<>();
        for (Map<String, Object> row : rows.rows()) {
            Object value = row.get(column);
            if (value == null) {
                continue;
            }
            String text = String.valueOf(value);
            if (text.length() > 28) {
                text = text.substring(0, 28) + "…";
            }
            if (!seen.contains(text)) {
                seen.add(text);
            }
            if (seen.size() == 3) {
                break;
            }
        }
        return seen.isEmpty() ? null : String.join(", ", seen);
    }

    private List<Map<String, Object>> prefix(String alias, List<Map<String, Object>> rows) {
        List<Map<String, Object>> prefixed = new ArrayList<>(rows.size());
        for (Map<String, Object> row : rows) {
            Map<String, Object> copy = new LinkedHashMap<>();
            row.forEach((key, value) -> copy.put(alias + "." + key, value));
            prefixed.add(copy);
        }
        return prefixed;
    }

    private List<Map<String, Object>> hashJoin(List<Map<String, Object>> left, String leftKey,
                                               List<Map<String, Object>> right, String rightKey, boolean leftJoin) {
        Map<String, List<Map<String, Object>>> index = new LinkedHashMap<>();
        for (Map<String, Object> row : right) {
            Object key = row.get(rightKey);
            if (key != null) {
                index.computeIfAbsent(String.valueOf(key), ignored -> new ArrayList<>()).add(row);
            }
        }
        List<Map<String, Object>> joined = new ArrayList<>();
        for (Map<String, Object> row : left) {
            Object key = row.get(leftKey);
            List<Map<String, Object>> matches = key == null
                    ? List.of()
                    : index.getOrDefault(String.valueOf(key), List.of());
            if (matches.isEmpty()) {
                if (leftJoin) {
                    joined.add(row);
                }
                continue;
            }
            for (Map<String, Object> match : matches) {
                Map<String, Object> merged = new LinkedHashMap<>(row);
                merged.putAll(match);
                joined.add(merged);
            }
        }
        return joined;
    }

    private Map<String, Object> project(FederatedQuery query, Map<String, Object> row) {
        Map<String, Object> out = new LinkedHashMap<>();
        for (String selected : query.projection()) {
            if (selected.equals("*")) {
                out.putAll(row);
                continue;
            }
            if (selected.endsWith(".*")) {
                String alias = selected.substring(0, selected.length() - 2);
                row.forEach((key, value) -> {
                    if (key.regionMatches(true, 0, alias + ".", 0, alias.length() + 1)) {
                        out.put(key, value);
                    }
                });
                continue;
            }
            if (!selected.contains(".")) {
                throw new IllegalArgumentException("qualify every column with its alias, for example "
                        + query.sides().getFirst().alias() + "." + selected);
            }
            out.put(selected, row.get(selected));
        }
        return out;
    }

    private ConnectionConfig resolve(String name, List<ConnectionConfig> available) {
        return available.stream()
                .filter(connection -> connection.name().equalsIgnoreCase(name))
                .findFirst()
                .orElseThrow(() -> new IllegalArgumentException("no connection named " + name + " in this project"));
    }

    private void requireKey(String alias, Map<String, ConnectionConfig> connections, FederatedQuery query,
                            String key, Map<String, QueryResult> fetched) {
        QueryResult rows = fetched.get(alias.toLowerCase());
        if (rows == null || rows.rows().isEmpty() || rows.columns().isEmpty()) {
            return;
        }
        boolean present = rows.columns().stream().anyMatch(column -> column.equalsIgnoreCase(key))
                || rows.rows().getFirst().containsKey(key);
        if (present) {
            return;
        }
        ConnectionConfig connection = connections.get(alias.toLowerCase());
        FederatedQuery.Side side = query.sideOf(alias);
        String suggestion = closest(key, rows.columns());
        throw new IllegalArgumentException("no column named \"" + key + "\" on " + connection.name() + "."
                + side.source() + " (alias " + alias + ")"
                + (suggestion == null ? "" : " — did you mean \"" + suggestion + "\"?")
                + " — available: " + String.join(", ", rows.columns()));
    }

    static String closest(String wanted, List<String> candidates) {
        String needle = wanted.toLowerCase();
        return candidates.stream()
                .filter(candidate -> {
                    String name = candidate.toLowerCase();
                    return name.contains(needle) || needle.contains(name)
                            || name.replace("_", "").equals(needle.replace("_", ""));
                })
                .min((left, right) -> Integer.compare(left.length(), right.length()))
                .orElse(null);
    }

    static String normalizeSource(String source) {
        return source.startsWith("/") ? source.substring(1) : source;
    }

    private void validateSource(ConnectionConfig connection, FederatedQuery.Side side) {
        if (connection.kind() == ConnectionKind.REDIS) {
            return;
        }
        List<String> available;
        try {
            available = engines.of(connection.kind()).schema(connection).stream().map(SchemaNode::name).toList();
        } catch (RuntimeException error) {
            return;
        }
        String wanted = normalizeSource(side.source());
        if (available.isEmpty() || available.stream().anyMatch(name -> normalizeSource(name).equalsIgnoreCase(wanted))) {
            return;
        }
        throw new IllegalArgumentException("no source named \"" + side.source() + "\" on " + connection.name()
                + " — available: " + String.join(", ", available)
                + (connection.kind() == ConnectionKind.CASSANDRA && side.source().equalsIgnoreCase(connection.keyspace())
                        ? " (\"" + side.source() + "\" is the keyspace, not a table)"
                        : ""));
    }

    private QueryResult fetch(ConnectionConfig connection, FederatedQuery.Side side) {
        String statement = connection.kind() == ConnectionKind.REDIS
                ? redisStatement(connection, side.source())
                : nativeStatement(connection.kind(), side);
        var engine = engines.of(connection.kind());
        List<Map<String, Object>> rows = new ArrayList<>();
        List<String> columns = List.of();
        String cursor = null;
        int page = 1;
        while (rows.size() < maxRowsPerSide) {
            QueryResult result = engine.query(connection, statement,
                    new PageRequest(Math.min(FETCH_PAGE, maxRowsPerSide - rows.size()), cursor, page));
            if (columns.isEmpty()) {
                columns = result.columns();
            }
            rows.addAll(result.rows());
            if (!result.hasMore() || result.nextCursor() == null || result.rows().isEmpty()) {
                break;
            }
            cursor = result.nextCursor();
            page++;
        }
        return new QueryResult(columns, rows, 0, 1, null, rows.size() >= maxRowsPerSide, null);
    }

    String redisStatement(ConnectionConfig connection, String key) {
        String type;
        try {
            QueryResult typed = engines.of(ConnectionKind.REDIS)
                    .query(connection, "TYPE " + key, PageRequest.first(1));
            type = typed.rows().isEmpty() ? "none" : String.valueOf(typed.rows().getFirst().get("value"));
        } catch (RuntimeException error) {
            type = "none";
        }
        return switch (type) {
            case "hash" -> "HGETALL " + key;
            case "string" -> "GET " + key;
            case "list" -> "LRANGE " + key + " 0 -1";
            case "set" -> "SMEMBERS " + key;
            case "zset" -> "ZRANGE " + key + " 0 -1";
            case "stream" -> "XRANGE " + key + " - +";
            case "none" -> throw new IllegalArgumentException("no Redis key named \"" + key + "\" on "
                    + connection.name() + " — check the key exists, or browse it in the schema panel");
            default -> throw new IllegalArgumentException("Redis key \"" + key + "\" holds a " + type
                    + ", which cannot be read as rows for a join");
        };
    }

    String nativeStatement(ConnectionKind kind, FederatedQuery.Side side) {
        String where = side.where() == null || side.where().isBlank() ? "" : " WHERE " + side.where();
        return switch (kind) {
            case POSTGRES, MYSQL, CASSANDRA -> "SELECT * FROM " + side.source() + where;
            case ELASTICSEARCH -> "GET /" + side.source() + "/_search";
            case KAFKA -> "consume " + side.source() + " --limit " + FETCH_PAGE;
            case REDIS -> "HGETALL " + side.source();
            case ETCD -> "get " + (side.source().startsWith("/") ? side.source() : "/" + side.source()) + " --prefix";
        };
    }
}
