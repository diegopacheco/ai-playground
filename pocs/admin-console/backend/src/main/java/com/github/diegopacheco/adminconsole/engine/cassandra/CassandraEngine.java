package com.github.diegopacheco.adminconsole.engine.cassandra;

import com.datastax.oss.driver.api.core.CqlSession;
import com.datastax.oss.driver.api.core.cql.ColumnDefinition;
import com.datastax.oss.driver.api.core.cql.PagingState;
import com.datastax.oss.driver.api.core.cql.ResultSet;
import com.datastax.oss.driver.api.core.cql.Row;
import com.datastax.oss.driver.api.core.cql.SimpleStatement;
import com.github.diegopacheco.adminconsole.engine.Engine;
import com.github.diegopacheco.adminconsole.engine.PageRequest;
import com.github.diegopacheco.adminconsole.engine.QueryResult;
import com.github.diegopacheco.adminconsole.engine.SchemaNode;
import com.github.diegopacheco.adminconsole.project.ConnectionConfig;
import com.github.diegopacheco.adminconsole.project.ConnectionKind;
import com.github.diegopacheco.adminconsole.registry.ConnectionRegistry;
import java.util.ArrayList;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.springframework.stereotype.Component;

@Component
public class CassandraEngine implements Engine {
    private final ConnectionRegistry registry;
    private final CqlReadOnlyGuard guard;

    public CassandraEngine(ConnectionRegistry registry, CqlReadOnlyGuard guard) {
        this.registry = registry;
        this.guard = guard;
    }

    @Override
    public ConnectionKind kind() {
        return ConnectionKind.CASSANDRA;
    }

    @Override
    public void assertReadOnly(String statement) {
        guard.assertReadOnly(statement);
    }

    @Override
    public List<SchemaNode> schema(ConnectionConfig config) {
        CqlSession session = registry.cassandra(config);
        String keyspace = config.keyspace();
        Map<String, List<SchemaNode>> columns = new LinkedHashMap<>();
        session.execute(SimpleStatement.newInstance(
                "SELECT table_name, column_name, type, kind FROM system_schema.columns WHERE keyspace_name = ?",
                keyspace)).forEach(row -> columns
                .computeIfAbsent(row.getString("table_name"), key -> new ArrayList<>())
                .add(SchemaNode.leaf(row.getString("column_name"), "column",
                        row.getString("type") + " (" + row.getString("kind") + ")")));
        List<SchemaNode> tables = new ArrayList<>();
        session.execute(SimpleStatement.newInstance(
                "SELECT table_name FROM system_schema.tables WHERE keyspace_name = ?", keyspace)).forEach(row -> {
            String name = row.getString("table_name");
            List<SchemaNode> tableColumns = columns.getOrDefault(name, List.of());
            long keys = tableColumns.stream().filter(column -> column.detail().contains("key")).count();
            tables.add(new SchemaNode(name, "table",
                    tableColumns.size() + " columns, " + keys + " key", tableColumns));
        });
        tables.sort((left, right) -> left.name().compareTo(right.name()));
        return tables;
    }

    @Override
    public QueryResult query(ConnectionConfig config, String statement, PageRequest page) {
        assertReadOnly(statement);
        CqlSession session = registry.cassandra(config);
        SimpleStatement prepared = SimpleStatement.builder(statement).setPageSize(page.size()).build();
        if (!page.isFirst()) {
            prepared = prepared.setPagingState(PagingState.fromBytes(Base64.getDecoder().decode(page.cursor())));
        }
        ResultSet result = session.execute(prepared);
        List<String> columns = new ArrayList<>();
        for (ColumnDefinition column : result.getColumnDefinitions()) {
            columns.add(column.getName().asInternal());
        }
        List<Map<String, Object>> rows = new ArrayList<>();
        int available = result.getAvailableWithoutFetching();
        for (int index = 0; index < available; index++) {
            Row row = result.one();
            if (row == null) {
                break;
            }
            Map<String, Object> values = new LinkedHashMap<>();
            for (int column = 0; column < columns.size(); column++) {
                Object value = row.getObject(column);
                values.put(columns.get(column), value == null ? null : String.valueOf(value));
            }
            rows.add(values);
        }
        PagingState pagingState = result.getExecutionInfo().getSafePagingState();
        String nextCursor = pagingState == null ? null : Base64.getEncoder().encodeToString(pagingState.toBytes());
        return QueryResult.of(columns, rows, page.pageNumber(), nextCursor, nextCursor != null);
    }
}
