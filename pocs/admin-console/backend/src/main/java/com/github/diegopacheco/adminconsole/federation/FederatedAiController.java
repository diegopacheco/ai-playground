package com.github.diegopacheco.adminconsole.federation;

import com.github.diegopacheco.adminconsole.ai.AgentCli;
import com.github.diegopacheco.adminconsole.ai.AgentCliRunner;
import com.github.diegopacheco.adminconsole.ai.AiSettingsRepository;
import com.github.diegopacheco.adminconsole.ai.CliAvailability;
import com.github.diegopacheco.adminconsole.ai.PromptBuilder;
import com.github.diegopacheco.adminconsole.ai.SuggestionParser;
import com.github.diegopacheco.adminconsole.auth.CurrentUser;
import com.github.diegopacheco.adminconsole.engine.EngineRegistry;
import com.github.diegopacheco.adminconsole.engine.SchemaNode;
import com.github.diegopacheco.adminconsole.project.ConnectionConfig;
import com.github.diegopacheco.adminconsole.project.ConnectionRepository;
import io.swagger.v3.oas.annotations.Operation;
import jakarta.validation.constraints.NotBlank;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/projects/{projectId}/federated/ai")
public class FederatedAiController {
    public record AskRequest(@NotBlank String prompt) {}

    private static final int MAX_SOURCES = 24;
    private static final int MAX_COLUMNS = 24;

    private final ConnectionRepository connections;
    private final EngineRegistry engines;
    private final PromptBuilder prompts;
    private final AgentCliRunner runner;
    private final SuggestionParser parser;
    private final FederatedQueryParser federatedParser;
    private final AiSettingsRepository settings;
    private final CliAvailability availability;
    private final CurrentUser current;

    public FederatedAiController(ConnectionRepository connections, EngineRegistry engines, PromptBuilder prompts,
                                 AgentCliRunner runner, SuggestionParser parser, FederatedQueryParser federatedParser,
                                 AiSettingsRepository settings, CliAvailability availability, CurrentUser current) {
        this.connections = connections;
        this.engines = engines;
        this.prompts = prompts;
        this.runner = runner;
        this.parser = parser;
        this.federatedParser = federatedParser;
        this.settings = settings;
        this.availability = availability;
        this.current = current;
    }

    @PostMapping
    @Operation(summary = "Ask the agent CLI to write a cross-engine join")
    public Map<String, Object> ask(@PathVariable long projectId, @RequestBody AskRequest request) {
        List<ConnectionConfig> available = connections.findByProject(projectId);
        if (available.size() < 2) {
            throw new IllegalArgumentException("a join needs at least two connections in the project");
        }

        List<PromptBuilder.FederatedSource> sources = new ArrayList<>();
        for (ConnectionConfig connection : available) {
            try {
                for (SchemaNode node : engines.of(connection.kind()).schema(connection)) {
                    if (sources.size() >= MAX_SOURCES) {
                        break;
                    }
                    List<String> columns = node.children().stream()
                            .map(SchemaNode::name)
                            .limit(MAX_COLUMNS)
                            .toList();
                    sources.add(new PromptBuilder.FederatedSource(connection.name(), connection.kind().wireName(),
                            node.name(), columns));
                }
            } catch (RuntimeException ignored) {
                continue;
            }
        }
        if (sources.isEmpty()) {
            throw new IllegalStateException("no connection in this project could be inspected");
        }

        AiSettingsRepository.Choice choice = settings.forUser(current.username())
                .orElseGet(() -> {
                    AgentCli fallback = availability.all().entrySet().stream()
                            .filter(entry -> entry.getValue().available())
                            .map(Map.Entry::getKey)
                            .findFirst()
                            .orElse(AgentCli.CLAUDE);
                    return new AiSettingsRepository.Choice(fallback, settings.globalModels().get(fallback));
                });

        String statement = parser.extract(
                runner.run(choice.cli(), choice.model(), prompts.buildFederated(sources, request.prompt())));

        boolean looksLikeQuery = statement.toUpperCase().contains("SELECT") && statement.toUpperCase().contains("JOIN");
        boolean parses = true;
        String problem = null;
        if (!looksLikeQuery) {
            parses = false;
        } else {
            try {
                federatedParser.parse(statement);
            } catch (RuntimeException error) {
                parses = false;
                problem = error.getMessage();
            }
        }

        Map<String, Object> body = new LinkedHashMap<>();
        body.put("statement", statement);
        body.put("cli", choice.cli().wireName());
        body.put("model", choice.model());
        body.put("parses", parses);
        body.put("problem", problem);
        body.put("declined", !looksLikeQuery);
        return body;
    }
}
