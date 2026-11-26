package com.github.diegopacheco.decisions.guardrail;

import java.util.List;
import java.util.Map;

public record GuardrailResult(Battery battery, long latencyMs, String outcome, boolean blocked, double severity,
        Map<String, Double> scores, List<String> triggered, List<String> flagged, String summary) {
}
