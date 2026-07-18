package com.github.diegopacheco.adminconsole.audit;

import com.github.diegopacheco.adminconsole.project.ConnectionConfig;
import java.time.Instant;
import java.util.UUID;
import org.springframework.stereotype.Service;

@Service
public class AuditService {
    private final AuditRepository repository;

    public AuditService(AuditRepository repository) {
        this.repository = repository;
    }

    public void allowed(UUID queryId, int page, String username, ConnectionConfig connection, String statement,
                        long elapsedMs, int rowCount, String clientIp) {
        repository.insert(new AuditEntry(null, queryId, page, Instant.now(), username, connection.id(),
                connection.projectId(), connection.kind().wireName(), statement, true, null, elapsedMs, rowCount,
                null, clientIp));
    }

    public void denied(UUID queryId, String username, ConnectionConfig connection, String statement, String reason,
                       String clientIp) {
        repository.insert(new AuditEntry(null, queryId, 1, Instant.now(), username, connection.id(),
                connection.projectId(), connection.kind().wireName(), statement, false, reason, null, null,
                null, clientIp));
    }

    public void suggested(UUID queryId, String username, ConnectionConfig connection, String statement,
                          boolean readOnlyOk, String denialReason, String cli, String model, String userPrompt,
                          String clientIp) {
        repository.insertSuggestion(new AuditEntry(null, queryId, 1, Instant.now(), username, connection.id(),
                connection.projectId(), connection.kind().wireName(), statement, readOnlyOk, denialReason, null, null,
                null, clientIp), cli, model, userPrompt);
    }

    public void federated(UUID queryId, String username, Long projectId, String statement, long elapsedMs,
                          Integer rowCount, String error, String clientIp) {
        repository.insertFederated(queryId, username, projectId, statement, elapsedMs, rowCount, error, clientIp);
    }

    public void failed(UUID queryId, int page, String username, ConnectionConfig connection, String statement,
                       long elapsedMs, String error, String clientIp) {
        repository.insert(new AuditEntry(null, queryId, page, Instant.now(), username, connection.id(),
                connection.projectId(), connection.kind().wireName(), statement, true, null, elapsedMs, null,
                error, clientIp));
    }
}
