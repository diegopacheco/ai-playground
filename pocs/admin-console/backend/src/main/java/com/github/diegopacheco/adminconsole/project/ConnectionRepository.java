package com.github.diegopacheco.adminconsole.project;

import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.List;
import java.util.Optional;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.core.RowMapper;
import org.springframework.stereotype.Repository;

@Repository
public class ConnectionRepository {
    private final JdbcTemplate jdbc;
    private final SecretPayload secrets;
    private final RowMapper<ConnectionConfig> mapper = this::map;

    public ConnectionRepository(JdbcTemplate jdbc, SecretPayload secrets) {
        this.jdbc = jdbc;
        this.secrets = secrets;
    }

    public List<ConnectionConfig> findByProject(long projectId) {
        return jdbc.query("SELECT * FROM connections WHERE project_id = ? ORDER BY name", mapper, projectId);
    }

    public List<ConnectionConfig> findAll() {
        return jdbc.query("SELECT * FROM connections ORDER BY project_id, name", mapper);
    }

    public Optional<ConnectionConfig> findById(long id) {
        return jdbc.query("SELECT * FROM connections WHERE id = ?", mapper, id).stream().findFirst();
    }

    public ConnectionConfig create(ConnectionConfig config) {
        Long id = jdbc.queryForObject("""
                INSERT INTO connections (project_id, name, kind, host, port, database, keyspace, datacenter,
                                         username, secret_ciphertext, options_json, created_by)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id""",
                Long.class, config.projectId(), config.name(), config.kind().wireName(), config.host(), config.port(),
                config.database(), config.keyspace(), config.datacenter(), config.username(),
                secrets.seal(config.password()), config.options(), config.createdBy());
        return findById(id).orElseThrow();
    }

    public void update(long id, ConnectionConfig config, boolean replacePassword) {
        jdbc.update("""
                UPDATE connections SET name = ?, kind = ?, host = ?, port = ?, database = ?, keyspace = ?,
                                       datacenter = ?, username = ?, options_json = ?
                WHERE id = ?""",
                config.name(), config.kind().wireName(), config.host(), config.port(), config.database(),
                config.keyspace(), config.datacenter(), config.username(), config.options(), id);
        if (replacePassword) {
            jdbc.update("UPDATE connections SET secret_ciphertext = ? WHERE id = ?", secrets.seal(config.password()), id);
        }
    }

    public void delete(long id) {
        jdbc.update("DELETE FROM connections WHERE id = ?", id);
    }

    private ConnectionConfig map(ResultSet row, int index) throws SQLException {
        return new ConnectionConfig(
                row.getLong("id"),
                row.getLong("project_id"),
                row.getString("name"),
                ConnectionKind.of(row.getString("kind")),
                row.getString("host"),
                row.getInt("port"),
                row.getString("database"),
                row.getString("keyspace"),
                row.getString("datacenter"),
                row.getString("username"),
                secrets.openPassword(row.getBytes("secret_ciphertext")),
                row.getString("options_json"),
                row.getString("created_by"));
    }
}
