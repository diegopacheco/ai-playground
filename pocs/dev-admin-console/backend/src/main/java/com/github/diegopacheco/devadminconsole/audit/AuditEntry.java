package com.github.diegopacheco.devadminconsole.audit;

import java.time.Instant;
import java.util.UUID;

public record AuditEntry(Long id, UUID queryId, int page, Instant at, String username, Long connectionId,
                         Long projectId, String kind, String statement, boolean allowed, String denialReason,
                         Long elapsedMs, Integer rowCount, String error, String clientIp) {}
