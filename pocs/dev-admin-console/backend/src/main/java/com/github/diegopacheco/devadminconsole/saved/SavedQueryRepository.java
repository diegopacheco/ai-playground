package com.github.diegopacheco.devadminconsole.saved;

import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.List;
import java.util.Optional;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.core.RowMapper;
import org.springframework.stereotype.Repository;

@Repository
public class SavedQueryRepository {
    private static final RowMapper<SavedQuery> MAPPER = SavedQueryRepository::map;

    private final JdbcTemplate jdbc;

    public SavedQueryRepository(JdbcTemplate jdbc) {
        this.jdbc = jdbc;
    }

    public List<SavedQuery> findByProject(long projectId) {
        return jdbc.query("SELECT * FROM saved_queries WHERE project_id = ? ORDER BY name", MAPPER, projectId);
    }

    public Optional<SavedQuery> findById(long id) {
        return jdbc.query("SELECT * FROM saved_queries WHERE id = ?", MAPPER, id).stream().findFirst();
    }

    public SavedQuery create(SavedQuery query) {
        Long id = jdbc.queryForObject("""
                INSERT INTO saved_queries (project_id, connection_id, name, statement, kind, description, created_by)
                VALUES (?, ?, ?, ?, ?, ?, ?) RETURNING id""",
                Long.class, query.projectId(), query.connectionId(), query.name(), query.statement(),
                query.kind(), query.description(), query.createdBy());
        return findById(id).orElseThrow();
    }

    public void update(long id, String name, String statement, String description, Long connectionId) {
        jdbc.update("""
                UPDATE saved_queries SET name = ?, statement = ?, description = ?, connection_id = ?, updated_at = now()
                WHERE id = ?""",
                name, statement, description, connectionId, id);
    }

    public void delete(long id) {
        jdbc.update("DELETE FROM saved_queries WHERE id = ?", id);
    }

    private static SavedQuery map(ResultSet row, int index) throws SQLException {
        long rawConnectionId = row.getLong("connection_id");
        Long connectionId = row.wasNull() ? null : rawConnectionId;
        return new SavedQuery(
                row.getLong("id"),
                row.getLong("project_id"),
                connectionId,
                row.getString("name"),
                row.getString("statement"),
                row.getString("kind"),
                row.getString("description"),
                row.getString("created_by"),
                row.getTimestamp("created_at").toInstant(),
                row.getTimestamp("updated_at").toInstant());
    }
}
