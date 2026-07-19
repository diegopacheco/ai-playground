package com.github.diegopacheco.devadminconsole.ai;

import com.github.diegopacheco.devadminconsole.audit.AuditService;
import com.github.diegopacheco.devadminconsole.auth.CurrentUser;
import com.github.diegopacheco.devadminconsole.engine.Engine;
import com.github.diegopacheco.devadminconsole.engine.EngineRegistry;
import com.github.diegopacheco.devadminconsole.engine.ReadOnlyViolation;
import com.github.diegopacheco.devadminconsole.project.ConnectionConfig;
import com.github.diegopacheco.devadminconsole.project.ConnectionRepository;
import io.swagger.v3.oas.annotations.Operation;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.constraints.NotBlank;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/ai")
public class AiController {
    public record SettingsRequest(@NotBlank String cli, String model) {}

    public record GlobalSettingsRequest(@NotBlank String cli, String model, boolean enabled) {}

    public record AskRequest(@NotBlank String prompt) {}

    private final CliAvailability availability;
    private final AiSettingsRepository settings;
    private final AgentCliRunner runner;
    private final PromptBuilder prompts;
    private final SuggestionParser parser;
    private final ConnectionRepository connections;
    private final EngineRegistry engines;
    private final AuditService audit;
    private final CurrentUser current;

    public AiController(CliAvailability availability, AiSettingsRepository settings, AgentCliRunner runner,
                        PromptBuilder prompts, SuggestionParser parser, ConnectionRepository connections,
                        EngineRegistry engines, AuditService audit, CurrentUser current) {
        this.availability = availability;
        this.settings = settings;
        this.runner = runner;
        this.prompts = prompts;
        this.parser = parser;
        this.connections = connections;
        this.engines = engines;
        this.audit = audit;
        this.current = current;
    }

    @GetMapping("/clis")
    @Operation(summary = "List agent CLIs, their models and whether they are installed")
    public List<Map<String, Object>> clis() {
        Map<AgentCli, String> models = settings.globalModels();
        List<Map<String, Object>> result = new ArrayList<>();
        availability.all().forEach((cli, found) -> {
            Map<String, Object> body = new LinkedHashMap<>();
            body.put("cli", cli.wireName());
            body.put("label", cli.label());
            body.put("model", models.get(cli));
            body.put("installed", found.available());
            body.put("enabled", settings.enabled(cli));
            body.put("reason", found.reason());
            result.add(body);
        });
        return result;
    }

    @GetMapping("/settings")
    @Operation(summary = "Read the AI choice remembered for the current user")
    public Map<String, Object> mySettings() {
        Map<AgentCli, String> models = settings.globalModels();
        return settings.forUser(current.username())
                .map(choice -> view(choice.cli(), choice.model(), false))
                .orElseGet(() -> {
                    AgentCli fallback = firstUsable();
                    return view(fallback, models.get(fallback), true);
                });
    }

    @PutMapping("/settings")
    @Operation(summary = "Remember the AI choice of the current user")
    public Map<String, Object> saveMySettings(@RequestBody SettingsRequest request) {
        AgentCli cli = AgentCli.of(request.cli());
        String model = request.model() == null || request.model().isBlank()
                ? settings.globalModels().get(cli)
                : request.model();
        settings.saveForUser(current.username(), cli, model);
        return view(cli, model, false);
    }

    @PutMapping("/settings/global")
    @Operation(summary = "Set the default model for a CLI and whether it is enabled")
    public Map<String, Object> saveGlobal(@RequestBody GlobalSettingsRequest request) {
        AgentCli cli = AgentCli.of(request.cli());
        settings.saveGlobal(cli, request.model(), request.enabled());
        return Map.of("updated", true);
    }

    @PostMapping("/connections/{connectionId}/query")
    @Operation(summary = "Ask the configured agent CLI to write a read-only query")
    public Map<String, Object> ask(@PathVariable long connectionId, @RequestBody AskRequest request,
                                   HttpServletRequest servletRequest) {
        ConnectionConfig config = connections.findById(connectionId)
                .orElseThrow(() -> new IllegalArgumentException("connection not found"));
        Engine engine = engines.of(config.kind());
        AiSettingsRepository.Choice choice = settings.forUser(current.username())
                .orElseGet(() -> {
                    AgentCli fallback = firstUsable();
                    return new AiSettingsRepository.Choice(fallback, settings.globalModels().get(fallback));
                });
        if (!settings.enabled(choice.cli())) {
            throw new IllegalStateException(choice.cli().label() + " is disabled by an administrator");
        }

        String prompt = prompts.build(config, engine.schema(config), request.prompt());
        String statement = parser.extract(runner.run(choice.cli(), choice.model(), prompt));

        boolean readOnlyOk = true;
        String denialReason = null;
        try {
            engine.assertReadOnly(statement);
        } catch (ReadOnlyViolation violation) {
            readOnlyOk = false;
            denialReason = violation.getMessage();
        } catch (RuntimeException error) {
            readOnlyOk = false;
            denialReason = error.getMessage();
        }

        audit.suggested(UUID.randomUUID(), current.username(), config, statement, readOnlyOk, denialReason,
                choice.cli().wireName(), choice.model(), request.prompt(), servletRequest.getRemoteAddr());

        Map<String, Object> body = new LinkedHashMap<>();
        body.put("statement", statement);
        body.put("cli", choice.cli().wireName());
        body.put("model", choice.model());
        body.put("readOnlyOk", readOnlyOk);
        body.put("denialReason", denialReason);
        return body;
    }

    private AgentCli firstUsable() {
        return availability.all().entrySet().stream()
                .filter(entry -> entry.getValue().available())
                .map(Map.Entry::getKey)
                .filter(settings::enabled)
                .findFirst()
                .orElse(AgentCli.CLAUDE);
    }

    private Map<String, Object> view(AgentCli cli, String model, boolean isDefault) {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("cli", cli.wireName());
        body.put("label", cli.label());
        body.put("model", model);
        body.put("usingDefault", isDefault);
        body.put("installed", availability.of(cli).available());
        return body;
    }
}
