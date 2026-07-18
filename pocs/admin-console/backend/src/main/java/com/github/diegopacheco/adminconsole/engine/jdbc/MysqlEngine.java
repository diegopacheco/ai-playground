package com.github.diegopacheco.adminconsole.engine.jdbc;

import com.github.diegopacheco.adminconsole.project.ConnectionConfig;
import com.github.diegopacheco.adminconsole.project.ConnectionKind;
import com.github.diegopacheco.adminconsole.registry.ConnectionRegistry;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

@Component
public class MysqlEngine extends JdbcEngine {
    public MysqlEngine(ConnectionRegistry registry, SqlReadOnlyGuard guard,
                       @Value("${app.query.statement-timeout-seconds}") int statementTimeoutSeconds) {
        super(registry, guard, statementTimeoutSeconds);
    }

    @Override
    public ConnectionKind kind() {
        return ConnectionKind.MYSQL;
    }

    @Override
    protected String schemaQuery() {
        return """
                SELECT table_name, concat(table_type, coalesce(concat(' ~', table_rows, ' rows'), ''))
                FROM information_schema.tables WHERE table_schema = ? ORDER BY table_name""";
    }

    @Override
    protected String columnQuery() {
        return """
                SELECT table_name, column_name, column_type, is_nullable
                FROM information_schema.columns WHERE table_schema = ? ORDER BY table_name, ordinal_position""";
    }

    @Override
    protected String schemaFilter(ConnectionConfig config) {
        return config.database();
    }
}
