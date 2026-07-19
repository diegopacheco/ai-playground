package com.github.diegopacheco.adminconsole.console;

import com.github.diegopacheco.adminconsole.audit.AuditService;
import com.github.diegopacheco.adminconsole.auth.CurrentUser;
import com.github.diegopacheco.adminconsole.engine.Engine;
import com.github.diegopacheco.adminconsole.engine.EngineRegistry;
import com.github.diegopacheco.adminconsole.engine.PageRequest;
import com.github.diegopacheco.adminconsole.engine.QueryResult;
import com.github.diegopacheco.adminconsole.engine.ReadOnlyViolation;
import com.github.diegopacheco.adminconsole.engine.SchemaNode;
import com.github.diegopacheco.adminconsole.project.ConnectionConfig;
import com.github.diegopacheco.adminconsole.project.ConnectionRepository;
import io.swagger.v3.oas.annotations.Operation;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.constraints.NotBlank;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/connections")
public class ConsoleController {
    public record QueryRequest(@NotBlank String statement, Integer pageSize, String cursor, Integer pageNumber,
                               String queryId) {}

    private final ConnectionRepository connections;
    private final EngineRegistry engines;
    private final AuditService audit;
    private final CurrentUser current;
    private final int defaultPageSize;
    private final int maxPageSize;

    public ConsoleController(ConnectionRepository connections, EngineRegistry engines, AuditService audit,
                             CurrentUser current,
                             @Value("${app.query.page-size}") int defaultPageSize,
                             @Value("${app.query.max-page-size}") int maxPageSize) {
        this.connections = connections;
        this.engines = engines;
        this.audit = audit;
        this.current = current;
        this.defaultPageSize = defaultPageSize;
        this.maxPageSize = maxPageSize;
    }

    @GetMapping("/{id}/schema")
    @Operation(summary = "Read the schema tree of a connection")
    public List<SchemaNode> schema(@PathVariable long id) {
        ConnectionConfig config = connection(id);
        return engines.of(config.kind()).schema(config);
    }

    @GetMapping("/{id}/ping")
    @Operation(summary = "Check that a connection is reachable")
    public Map<String, Object> ping(@PathVariable long id) {
        ConnectionConfig config = connection(id);
        Map<String, Object> body = new LinkedHashMap<>();
        try {
            engines.of(config.kind()).ping(config);
            body.put("healthy", true);
        } catch (RuntimeException error) {
            body.put("healthy", false);
            body.put("error", error.getMessage());
        }
        return body;
    }

    @PostMapping("/{id}/query")
    @Operation(summary = "Run a read-only statement and read one page of results")
    public Map<String, Object> query(@PathVariable long id, @RequestBody QueryRequest request,
                                     HttpServletRequest servletRequest) {
        ConnectionConfig config = connection(id);
        Engine engine = engines.of(config.kind());
        UUID queryId = request.queryId() == null ? UUID.randomUUID() : UUID.fromString(request.queryId());
        int pageNumber = request.pageNumber() == null ? 1 : Math.max(request.pageNumber(), 1);
        int size = Math.min(request.pageSize() == null ? defaultPageSize : request.pageSize(), maxPageSize);
        String clientIp = servletRequest.getRemoteAddr();
        long started = System.currentTimeMillis();
        try {
            QueryResult result = engine.query(config, request.statement(),
                    new PageRequest(size, request.cursor(), pageNumber));
            long elapsed = System.currentTimeMillis() - started;
            audit.allowed(queryId, pageNumber, current.username(), config, request.statement(), elapsed,
                    result.rows().size(), clientIp);
            return view(queryId, result.withElapsed(elapsed));
        } catch (ReadOnlyViolation violation) {
            audit.denied(queryId, current.username(), config, request.statement(), violation.getMessage(), clientIp);
            throw violation;
        } catch (RuntimeException error) {
            audit.failed(queryId, pageNumber, current.username(), config, request.statement(),
                    System.currentTimeMillis() - started, error.getMessage(), clientIp);
            throw error;
        }
    }

    @GetMapping("/{id}/count")
    @Operation(summary = "Count rows for a statement, SQL connections only")
    public Map<String, Object> count(@PathVariable long id, @RequestParam String statement) {
        ConnectionConfig config = connection(id);
        Engine engine = engines.of(config.kind());
        engine.assertReadOnly(statement);
        QueryResult result = engine.query(config, "SELECT count(*) AS total FROM (" + statement + ") AS counted",
                PageRequest.first(1));
        return Map.of("total", result.rows().isEmpty() ? 0 : result.rows().getFirst().get("total"));
    }

    private ConnectionConfig connection(long id) {
        return connections.findById(id).orElseThrow(() -> new IllegalArgumentException("connection not found"));
    }

    private Map<String, Object> view(UUID queryId, QueryResult result) {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("queryId", queryId);
        body.put("columns", result.columns());
        body.put("rows", result.rows());
        body.put("elapsedMs", result.elapsedMs());
        body.put("pageNumber", result.pageNumber());
        body.put("nextCursor", result.nextCursor());
        body.put("hasMore", result.hasMore());
        body.put("totalRows", result.totalRows());
        return body;
    }
}
