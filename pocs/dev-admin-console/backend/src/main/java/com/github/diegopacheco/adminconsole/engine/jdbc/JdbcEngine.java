package com.github.diegopacheco.adminconsole.engine.jdbc;

import com.github.diegopacheco.adminconsole.engine.Engine;
import com.github.diegopacheco.adminconsole.engine.PageRequest;
import com.github.diegopacheco.adminconsole.engine.QueryResult;
import com.github.diegopacheco.adminconsole.engine.SchemaNode;
import com.github.diegopacheco.adminconsole.project.ConnectionConfig;
import com.github.diegopacheco.adminconsole.project.ConnectionKind;
import com.github.diegopacheco.adminconsole.registry.ConnectionRegistry;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public abstract class JdbcEngine implements Engine {
    private final ConnectionRegistry registry;
    private final SqlReadOnlyGuard guard;
    private final int statementTimeoutSeconds;

    protected JdbcEngine(ConnectionRegistry registry, SqlReadOnlyGuard guard, int statementTimeoutSeconds) {
        this.registry = registry;
        this.guard = guard;
        this.statementTimeoutSeconds = statementTimeoutSeconds;
    }

    protected abstract String schemaQuery();

    protected abstract String columnQuery();

    protected abstract String schemaFilter(ConnectionConfig config);

    @Override
    public void assertReadOnly(String statement) {
        guard.assertReadOnly(statement);
    }

    @Override
    public List<SchemaNode> schema(ConnectionConfig config) {
        Map<String, List<SchemaNode>> columnsByTable = new LinkedHashMap<>();
        each(config, columnQuery(), schemaFilter(config), row -> {
            String table = row.getString(1);
            columnsByTable.computeIfAbsent(table, key -> new ArrayList<>())
                    .add(SchemaNode.leaf(row.getString(2), "column", detail(row)));
        });
        List<SchemaNode> tables = new ArrayList<>();
        each(config, schemaQuery(), schemaFilter(config), row -> {
            String table = row.getString(1);
            tables.add(new SchemaNode(table, "table", row.getString(2),
                    columnsByTable.getOrDefault(table, List.of())));
        });
        return tables;
    }

    @Override
    public QueryResult query(ConnectionConfig config, String statement, PageRequest page) {
        assertReadOnly(statement);
        int offset = page.isFirst() ? 0 : Integer.parseInt(page.cursor());
        try (Connection connection = registry.jdbc(config).getConnection()) {
            connection.setReadOnly(true);
            connection.setAutoCommit(false);
            try (PreparedStatement prepared = connection.prepareStatement(statement)) {
                prepared.setQueryTimeout(statementTimeoutSeconds);
                prepared.setFetchSize(page.size());
                try (ResultSet result = prepared.executeQuery()) {
                    List<String> columns = columns(result.getMetaData());
                    for (int skipped = 0; skipped < offset && result.next(); skipped++) {
                        continue;
                    }
                    List<Map<String, Object>> rows = new ArrayList<>();
                    boolean hasMore = false;
                    while (result.next()) {
                        if (rows.size() == page.size()) {
                            hasMore = true;
                            break;
                        }
                        rows.add(row(result, columns));
                    }
                    String nextCursor = hasMore ? String.valueOf(offset + rows.size()) : null;
                    return QueryResult.of(columns, rows, page.pageNumber(), nextCursor, hasMore);
                }
            } finally {
                connection.rollback();
            }
        } catch (SQLException error) {
            throw new IllegalStateException(error.getMessage(), error);
        }
    }

    private interface RowConsumer {
        void accept(ResultSet row) throws SQLException;
    }

    private void each(ConnectionConfig config, String sql, String filter, RowConsumer consumer) {
        try (Connection connection = registry.jdbc(config).getConnection();
             PreparedStatement prepared = connection.prepareStatement(sql)) {
            prepared.setQueryTimeout(statementTimeoutSeconds);
            prepared.setString(1, filter);
            try (ResultSet result = prepared.executeQuery()) {
                while (result.next()) {
                    consumer.accept(result);
                }
            }
        } catch (SQLException error) {
            throw new IllegalStateException(error.getMessage(), error);
        }
    }

    private String detail(ResultSet row) throws SQLException {
        String type = row.getString(3);
        String nullable = row.getString(4);
        return "YES".equalsIgnoreCase(nullable) ? type : type + " not null";
    }

    private List<String> columns(ResultSetMetaData metadata) throws SQLException {
        List<String> columns = new ArrayList<>();
        for (int index = 1; index <= metadata.getColumnCount(); index++) {
            columns.add(metadata.getColumnLabel(index));
        }
        return columns;
    }

    private Map<String, Object> row(ResultSet result, List<String> columns) throws SQLException {
        Map<String, Object> values = new LinkedHashMap<>();
        for (int index = 1; index <= columns.size(); index++) {
            Object value = result.getObject(index);
            values.put(columns.get(index - 1), value == null ? null : String.valueOf(value));
        }
        return values;
    }

    protected ConnectionKind kindOf() {
        return kind();
    }
}
