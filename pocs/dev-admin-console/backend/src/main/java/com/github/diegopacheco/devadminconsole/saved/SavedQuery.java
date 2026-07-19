package com.github.diegopacheco.devadminconsole.saved;

import java.time.Instant;

public record SavedQuery(Long id, Long projectId, Long connectionId, String name, String statement, String kind,
                         String description, String createdBy, Instant createdAt, Instant updatedAt) {}
