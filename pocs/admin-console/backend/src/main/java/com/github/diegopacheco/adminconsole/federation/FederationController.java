package com.github.diegopacheco.adminconsole.federation;

import com.github.diegopacheco.adminconsole.project.ConnectionRepository;
import io.swagger.v3.oas.annotations.Operation;
import jakarta.validation.constraints.NotBlank;
import java.util.LinkedHashMap;
import java.util.Map;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/projects/{projectId}/federated")
public class FederationController {
    public record FederatedRequest(@NotBlank String statement) {}

    private final FederatedQueryParser parser;
    private final FederatedExecutor executor;
    private final ConnectionRepository connections;

    public FederationController(FederatedQueryParser parser, FederatedExecutor executor,
                                ConnectionRepository connections) {
        this.parser = parser;
        this.executor = executor;
        this.connections = connections;
    }

    @PostMapping
    @Operation(summary = "Join two sources that live on different engines")
    public Map<String, Object> query(@PathVariable long projectId, @RequestBody FederatedRequest request) {
        FederatedQuery query = parser.parse(request.statement());
        FederatedExecutor.Result result = executor.execute(query, connections.findByProject(projectId));
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("columns", result.columns());
        body.put("rows", result.rows());
        body.put("elapsedMs", result.elapsedMs());
        body.put("sides", result.sides().stream().map(side -> {
            Map<String, Object> view = new LinkedHashMap<>();
            view.put("alias", side.alias());
            view.put("connectionName", side.connectionName());
            view.put("kind", side.kind());
            view.put("source", side.source());
            view.put("rows", side.rows());
            view.put("truncated", side.truncated());
            return view;
        }).toList());
        return body;
    }
}
