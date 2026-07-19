package com.github.diegopacheco.devadminconsole.audit;

import java.sql.ResultSet;
import java.sql.SQLException;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.core.RowMapper;
import org.springframework.stereotype.Repository;

@Repository
public class AuditRepository {
    private static final RowMapper<AuditEntry> MAPPER = AuditRepository::map;

    private final JdbcTemplate jdbc;

    public AuditRepository(JdbcTemplate jdbc) {
        this.jdbc = jdbc;
    }

    public void insert(AuditEntry entry) {
        jdbc.update("""
                INSERT INTO audit_log (query_id, page, username, connection_id, project_id, kind, statement,
                                       allowed, denial_reason, elapsed_ms, row_count, error, client_ip)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                entry.queryId(), entry.page(), entry.username(), entry.connectionId(), entry.projectId(),
                entry.kind(), entry.statement(), entry.allowed(), entry.denialReason(), entry.elapsedMs(),
                entry.rowCount(), entry.error(), entry.clientIp());
    }

    public void insertSuggestion(AuditEntry entry, String cli, String model, String userPrompt) {
        jdbc.update("""
                INSERT INTO audit_log (query_id, page, username, connection_id, project_id, kind, statement,
                                       allowed, denial_reason, client_ip, ai_cli, ai_model, ai_prompt)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                entry.queryId(), entry.page(), entry.username(), entry.connectionId(), entry.projectId(),
                entry.kind(), entry.statement(), entry.allowed(), entry.denialReason(), entry.clientIp(),
                cli, model, userPrompt);
    }

    public void insertFederated(java.util.UUID queryId, String username, Long projectId, String statement,
                                Long elapsedMs, Integer rowCount, String error, String clientIp) {
        jdbc.update("""
                INSERT INTO audit_log (query_id, page, username, connection_id, project_id, kind, statement,
                                       allowed, denial_reason, elapsed_ms, row_count, error, client_ip)
                VALUES (?, 1, ?, NULL, ?, 'federated', ?, true, NULL, ?, ?, ?, ?)""",
                queryId, username, projectId, statement, elapsedMs, rowCount, error, clientIp);
    }

    public List<String> recentFederated(String username, Long projectId, int limit) {
        return jdbc.queryForList("""
                SELECT statement FROM audit_log
                WHERE username = ? AND kind = 'federated' AND project_id = ? AND error IS NULL
                GROUP BY statement ORDER BY max(at) DESC LIMIT ?""",
                String.class, username, projectId, limit);
    }

    public List<AuditEntry> search(String username, Long connectionId, Boolean allowed, Instant from, Instant to,
                                   int limit, int offset) {
        StringBuilder sql = new StringBuilder("SELECT * FROM audit_log WHERE 1 = 1");
        List<Object> arguments = new ArrayList<>();
        if (username != null && !username.isBlank()) {
            sql.append(" AND username = ?");
            arguments.add(username);
        }
        if (connectionId != null) {
            sql.append(" AND connection_id = ?");
            arguments.add(connectionId);
        }
        if (allowed != null) {
            sql.append(" AND allowed = ?");
            arguments.add(allowed);
        }
        if (from != null) {
            sql.append(" AND at >= ?");
            arguments.add(java.sql.Timestamp.from(from));
        }
        if (to != null) {
            sql.append(" AND at <= ?");
            arguments.add(java.sql.Timestamp.from(to));
        }
        sql.append(" ORDER BY at DESC, id DESC LIMIT ? OFFSET ?");
        arguments.add(limit);
        arguments.add(offset);
        return jdbc.query(sql.toString(), MAPPER, arguments.toArray());
    }

    public List<String> recentStatements(String username, Long connectionId, int limit) {
        if (connectionId == null) {
            return jdbc.queryForList("""
                    SELECT statement FROM audit_log WHERE username = ? AND allowed = true
                    GROUP BY statement ORDER BY max(at) DESC LIMIT ?""",
                    String.class, username, limit);
        }
        return jdbc.queryForList("""
                SELECT statement FROM audit_log WHERE username = ? AND connection_id = ? AND allowed = true
                GROUP BY statement ORDER BY max(at) DESC LIMIT ?""",
                String.class, username, connectionId, limit);
    }

    private static AuditEntry map(ResultSet row, int index) throws SQLException {
        return new AuditEntry(
                row.getLong("id"),
                UUID.fromString(row.getString("query_id")),
                row.getInt("page"),
                row.getTimestamp("at").toInstant(),
                row.getString("username"),
                nullableLong(row, "connection_id"),
                nullableLong(row, "project_id"),
                row.getString("kind"),
                row.getString("statement"),
                row.getBoolean("allowed"),
                row.getString("denial_reason"),
                nullableLong(row, "elapsed_ms"),
                nullableInt(row, "row_count"),
                row.getString("error"),
                row.getString("client_ip"));
    }

    private static Long nullableLong(ResultSet row, String column) throws SQLException {
        long value = row.getLong(column);
        return row.wasNull() ? null : value;
    }

    private static Integer nullableInt(ResultSet row, String column) throws SQLException {
        int value = row.getInt(column);
        return row.wasNull() ? null : value;
    }
}
