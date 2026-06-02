package com.diegopacheco.autotune.tune;

public record RunSummary(
        int total,
        int success,
        int failure,
        int shortCircuited,
        int rateLimited,
        int rejected,
        double meanLatencyMs,
        double p95LatencyMs
) {}
