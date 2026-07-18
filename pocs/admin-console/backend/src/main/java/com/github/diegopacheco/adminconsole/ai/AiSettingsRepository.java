package com.github.diegopacheco.adminconsole.ai;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Repository;

@Repository
public class AiSettingsRepository {
    public record Choice(AgentCli cli, String model) {}

    private static final String GLOBAL = "global";
    private static final String USER = "user";

    private final JdbcTemplate jdbc;

    public AiSettingsRepository(JdbcTemplate jdbc) {
        this.jdbc = jdbc;
    }

    public Map<AgentCli, String> globalModels() {
        Map<AgentCli, String> models = new LinkedHashMap<>();
        for (AgentCli cli : AgentCli.values()) {
            models.put(cli, cli.defaultModel());
        }
        jdbc.query("SELECT cli, model FROM ai_settings WHERE scope = ? AND username IS NULL", row -> {
            models.put(AgentCli.of(row.getString("cli")), row.getString("model"));
        }, GLOBAL);
        return models;
    }

    public boolean enabled(AgentCli cli) {
        Boolean enabled = jdbc.query(
                "SELECT enabled FROM ai_settings WHERE scope = ? AND username IS NULL AND cli = ?",
                row -> row.next() ? row.getBoolean("enabled") : null, GLOBAL, cli.wireName());
        return enabled == null || enabled;
    }

    public void saveGlobal(AgentCli cli, String model, boolean enabled) {
        jdbc.update("DELETE FROM ai_settings WHERE scope = ? AND username IS NULL AND cli = ?", GLOBAL, cli.wireName());
        jdbc.update("INSERT INTO ai_settings (scope, username, cli, model, enabled) VALUES (?, NULL, ?, ?, ?)",
                GLOBAL, cli.wireName(), model, enabled);
    }

    public Optional<Choice> forUser(String username) {
        return jdbc.query("SELECT cli, model FROM ai_settings WHERE scope = ? AND username = ? ORDER BY updated_at DESC",
                row -> row.next()
                        ? Optional.of(new Choice(AgentCli.of(row.getString("cli")), row.getString("model")))
                        : Optional.<Choice>empty(),
                USER, username);
    }

    public void saveForUser(String username, AgentCli cli, String model) {
        jdbc.update("DELETE FROM ai_settings WHERE scope = ? AND username = ?", USER, username);
        jdbc.update("INSERT INTO ai_settings (scope, username, cli, model, enabled) VALUES (?, ?, ?, ?, TRUE)",
                USER, username, cli.wireName(), model);
    }
}
