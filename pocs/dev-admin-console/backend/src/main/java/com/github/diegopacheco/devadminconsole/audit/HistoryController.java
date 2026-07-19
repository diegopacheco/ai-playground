package com.github.diegopacheco.devadminconsole.audit;

import com.github.diegopacheco.devadminconsole.auth.CurrentUser;
import io.swagger.v3.oas.annotations.Operation;
import java.util.List;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/history")
public class HistoryController {
    private final AuditRepository repository;
    private final CurrentUser current;

    public HistoryController(AuditRepository repository, CurrentUser current) {
        this.repository = repository;
        this.current = current;
    }

    @GetMapping
    @Operation(summary = "Recent statements run by the current user")
    public List<String> history(@RequestParam(required = false) Long connection,
                                @RequestParam(defaultValue = "20") int limit) {
        return repository.recentStatements(current.username(), connection, Math.min(Math.max(limit, 1), 100));
    }
}
