package com.github.diegopacheco.devadminconsole.project;

import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.List;
import java.util.Optional;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.core.RowMapper;
import org.springframework.stereotype.Repository;

@Repository
public class ProjectRepository {
    private static final RowMapper<Project> MAPPER = ProjectRepository::map;

    private final JdbcTemplate jdbc;

    public ProjectRepository(JdbcTemplate jdbc) {
        this.jdbc = jdbc;
    }

    public List<Project> findAll() {
        return jdbc.query("SELECT * FROM projects ORDER BY name", MAPPER);
    }

    public Optional<Project> findById(long id) {
        return jdbc.query("SELECT * FROM projects WHERE id = ?", MAPPER, id).stream().findFirst();
    }

    public Project create(String name, String createdBy) {
        Long id = jdbc.queryForObject("INSERT INTO projects (name, created_by) VALUES (?, ?) RETURNING id",
                Long.class, name, createdBy);
        return findById(id).orElseThrow();
    }

    public void rename(long id, String name) {
        jdbc.update("UPDATE projects SET name = ? WHERE id = ?", name, id);
    }

    public void delete(long id) {
        jdbc.update("DELETE FROM projects WHERE id = ?", id);
    }

    private static Project map(ResultSet row, int index) throws SQLException {
        return new Project(row.getLong("id"), row.getString("name"),
                row.getTimestamp("created_at").toInstant(), row.getString("created_by"));
    }
}
