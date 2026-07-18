package com.github.diegopacheco.adminconsole.user;

import java.sql.ResultSet;
import java.sql.SQLException;
import java.time.Instant;
import java.util.List;
import java.util.Optional;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.core.RowMapper;
import org.springframework.stereotype.Repository;

@Repository
public class UserRepository {
    private static final RowMapper<User> MAPPER = UserRepository::map;

    private final JdbcTemplate jdbc;

    public UserRepository(JdbcTemplate jdbc) {
        this.jdbc = jdbc;
    }

    public Optional<User> findByUsername(String username) {
        return jdbc.query("SELECT * FROM users WHERE username = ?", MAPPER, username).stream().findFirst();
    }

    public Optional<User> findById(long id) {
        return jdbc.query("SELECT * FROM users WHERE id = ?", MAPPER, id).stream().findFirst();
    }

    public List<User> findAll() {
        return jdbc.query("SELECT * FROM users ORDER BY username", MAPPER);
    }

    public long count() {
        return jdbc.queryForObject("SELECT count(*) FROM users", Long.class);
    }

    public User create(String username, byte[] hash, byte[] salt, String role) {
        Long id = jdbc.queryForObject(
                "INSERT INTO users (username, password_hash, password_salt, role) VALUES (?, ?, ?, ?) RETURNING id",
                Long.class, username, hash, salt, role);
        return findById(id).orElseThrow();
    }

    public void updatePassword(long id, byte[] hash, byte[] salt) {
        jdbc.update("UPDATE users SET password_hash = ?, password_salt = ? WHERE id = ?", hash, salt, id);
    }

    public void updateRoleAndEnabled(long id, String role, boolean enabled) {
        jdbc.update("UPDATE users SET role = ?, enabled = ? WHERE id = ?", role, enabled, id);
    }

    public void touchLogin(long id) {
        jdbc.update("UPDATE users SET last_login_at = now() WHERE id = ?", id);
    }

    public void delete(long id) {
        jdbc.update("DELETE FROM users WHERE id = ?", id);
    }

    private static User map(ResultSet row, int index) throws SQLException {
        return new User(
                row.getLong("id"),
                row.getString("username"),
                row.getBytes("password_hash"),
                row.getBytes("password_salt"),
                row.getString("role"),
                row.getBoolean("enabled"),
                instant(row, "created_at"),
                instant(row, "last_login_at"));
    }

    private static Instant instant(ResultSet row, String column) throws SQLException {
        var value = row.getTimestamp(column);
        return value == null ? null : value.toInstant();
    }
}
