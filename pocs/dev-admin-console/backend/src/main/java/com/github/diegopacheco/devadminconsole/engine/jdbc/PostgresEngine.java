package com.github.diegopacheco.devadminconsole.engine.jdbc;

import com.github.diegopacheco.devadminconsole.project.ConnectionConfig;
import com.github.diegopacheco.devadminconsole.project.ConnectionKind;
import com.github.diegopacheco.devadminconsole.registry.ConnectionRegistry;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

@Component
public class PostgresEngine extends JdbcEngine {
    public PostgresEngine(ConnectionRegistry registry, SqlReadOnlyGuard guard,
                          @Value("${app.query.statement-timeout-seconds}") int statementTimeoutSeconds) {
        super(registry, guard, statementTimeoutSeconds);
    }

    @Override
    public ConnectionKind kind() {
        return ConnectionKind.POSTGRES;
    }

    @Override
    protected String schemaQuery() {
        return """
                SELECT table_name, table_type FROM information_schema.tables
                WHERE table_schema = ? ORDER BY table_name""";
    }

    @Override
    protected String columnQuery() {
        return """
                SELECT table_name, column_name, data_type, is_nullable FROM information_schema.columns
                WHERE table_schema = ? ORDER BY table_name, ordinal_position""";
    }

    @Override
    protected String schemaFilter(ConnectionConfig config) {
        return config.keyspace() == null || config.keyspace().isBlank() ? "public" : config.keyspace();
    }
}
