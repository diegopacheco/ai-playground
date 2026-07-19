package com.github.diegopacheco.devadminconsole.trace;

import java.time.Instant;
import java.util.List;
import java.util.Map;

public record TraceHit(String connectionName, String kind, String source, String label, Instant at,
                       List<String> columns, Map<String, Object> row) {}
