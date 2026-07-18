package com.github.diegopacheco.adminconsole.federation;

import com.github.diegopacheco.adminconsole.engine.EngineRegistry;
import com.github.diegopacheco.adminconsole.engine.PageRequest;
import com.github.diegopacheco.adminconsole.engine.QueryResult;
import com.github.diegopacheco.adminconsole.project.ConnectionConfig;
import com.github.diegopacheco.adminconsole.project.ConnectionKind;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

@Service
public class FederatedExecutor {
    public record SideResult(String alias, String connectionName, String kind, String source, int rows,
                             boolean truncated) {}

    public record Result(List<String> columns, List<Map<String, Object>> rows, List<SideResult> sides,
                         long elapsedMs) {}

    private final EngineRegistry engines;
    private final int maxRowsPerSide;

    public FederatedExecutor(EngineRegistry engines,
                             @Value("${app.federation.max-rows-per-side:5000}") int maxRowsPerSide) {
        this.engines = engines;
        this.maxRowsPerSide = maxRowsPerSide;
    }

    public Result execute(FederatedQuery query, List<ConnectionConfig> available) {
        long started = System.currentTimeMillis();
        ConnectionConfig leftConnection = resolve(query.left().connectionName(), available);
        ConnectionConfig rightConnection = resolve(query.right().connectionName(), available);

        QueryResult leftRows = fetch(leftConnection, query.left());
        QueryResult rightRows = fetch(rightConnection, query.right());

        Map<String, List<Map<String, Object>>> index = new LinkedHashMap<>();
        for (Map<String, Object> row : rightRows.rows()) {
            Object key = row.get(query.rightKey());
            if (key == null) {
                continue;
            }
            index.computeIfAbsent(String.valueOf(key), ignored -> new ArrayList<>()).add(row);
        }

        List<Map<String, Object>> joined = new ArrayList<>();
        for (Map<String, Object> row : leftRows.rows()) {
            if (joined.size() >= query.limit()) {
                break;
            }
            Object key = row.get(query.leftKey());
            List<Map<String, Object>> matches = key == null ? List.of() : index.getOrDefault(String.valueOf(key), List.of());
            if (matches.isEmpty()) {
                if (query.leftJoin()) {
                    joined.add(project(query, row, Map.of()));
                }
                continue;
            }
            for (Map<String, Object> match : matches) {
                if (joined.size() >= query.limit()) {
                    break;
                }
                joined.add(project(query, row, match));
            }
        }

        List<String> columns = columns(query);
        List<SideResult> sides = List.of(
                new SideResult(query.left().alias(), leftConnection.name(), leftConnection.kind().wireName(),
                        query.left().source(), leftRows.rows().size(), leftRows.hasMore()),
                new SideResult(query.right().alias(), rightConnection.name(), rightConnection.kind().wireName(),
                        query.right().source(), rightRows.rows().size(), rightRows.hasMore()));
        return new Result(columns, joined, sides, System.currentTimeMillis() - started);
    }

    private ConnectionConfig resolve(String name, List<ConnectionConfig> available) {
        return available.stream()
                .filter(connection -> connection.name().equalsIgnoreCase(name))
                .findFirst()
                .orElseThrow(() -> new IllegalArgumentException("no connection named " + name + " in this project"));
    }

    private static final int FETCH_PAGE = 100;

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
            case ETCD -> "get " + side.source() + " --prefix";
        };
    }

    private List<String> columns(FederatedQuery query) {
        List<String> columns = new ArrayList<>();
        for (String selected : query.projection()) {
            columns.add(selected.equals("*") ? "*" : selected);
        }
        return columns;
    }

    private Map<String, Object> project(FederatedQuery query, Map<String, Object> left, Map<String, Object> right) {
        Map<String, Object> row = new LinkedHashMap<>();
        for (String selected : query.projection()) {
            if (selected.equals("*")) {
                left.forEach((key, value) -> row.put(query.left().alias() + "." + key, value));
                right.forEach((key, value) -> row.put(query.right().alias() + "." + key, value));
                continue;
            }
            String[] parts = selected.split("\\.", 2);
            if (parts.length != 2) {
                throw new IllegalArgumentException("qualify every column with its alias, for example "
                        + query.left().alias() + "." + selected);
            }
            Map<String, Object> source = parts[0].equalsIgnoreCase(query.left().alias()) ? left : right;
            row.put(selected, source.get(parts[1]));
        }
        return row;
    }
}
