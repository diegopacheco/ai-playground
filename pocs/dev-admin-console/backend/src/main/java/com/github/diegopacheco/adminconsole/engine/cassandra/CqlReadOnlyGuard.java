package com.github.diegopacheco.adminconsole.engine.cassandra;

import com.github.diegopacheco.adminconsole.engine.ReadOnlyViolation;
import org.springframework.stereotype.Component;

@Component
public class CqlReadOnlyGuard {
    private final com.github.diegopacheco.adminconsole.engine.jdbc.SqlReadOnlyGuard sql;

    public CqlReadOnlyGuard(com.github.diegopacheco.adminconsole.engine.jdbc.SqlReadOnlyGuard sql) {
        this.sql = sql;
    }

    public void assertReadOnly(String statement) {
        String normalized = sql.normalize(statement);
        if (normalized.isEmpty()) {
            throw new ReadOnlyViolation("statement is empty");
        }
        String first = normalized.split("\\s+")[0].toUpperCase();
        if (!first.equals("SELECT")) {
            throw new ReadOnlyViolation("only SELECT is allowed on Cassandra, found: " + first);
        }
        sql.assertReadOnly(statement);
    }
}
