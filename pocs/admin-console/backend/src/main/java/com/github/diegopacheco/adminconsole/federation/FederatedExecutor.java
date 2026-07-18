package com.github.diegopacheco.adminconsole.federation;

import com.github.diegopacheco.adminconsole.engine.EngineRegistry;
import com.github.diegopacheco.adminconsole.engine.PageRequest;
import com.github.diegopacheco.adminconsole.engine.QueryResult;
import com.github.diegopacheco.adminconsole.engine.SchemaNode;
import com.github.diegopacheco.adminconsole.project.ConnectionConfig;
import com.github.diegopacheco.adminconsole.project.ConnectionKind;
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
                         long elapsedMs) {}

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

        for (FederatedQuery.Join join : query.joins()) {
            List<Map<String, Object>> right = prefix(join.rightAlias(),
                    fetched.get(join.rightAlias().toLowerCase()).rows());
            accumulated = hashJoin(accumulated, join.leftAlias() + "." + join.leftKey(),
                    right, join.rightAlias() + "." + join.rightKey(), join.leftJoin());
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

        return new Result(new ArrayList<>(columns), projected, sideResults, System.currentTimeMillis() - started);
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
        String statement = nativeStatement(connection.kind(), side);
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
