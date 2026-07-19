package com.github.diegopacheco.devadminconsole.project;

import java.time.Instant;

public record Project(Long id, String name, Instant createdAt, String createdBy) {}
