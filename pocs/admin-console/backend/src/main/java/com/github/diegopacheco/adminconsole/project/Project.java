package com.github.diegopacheco.adminconsole.project;

import java.time.Instant;

public record Project(Long id, String name, Instant createdAt, String createdBy) {}
