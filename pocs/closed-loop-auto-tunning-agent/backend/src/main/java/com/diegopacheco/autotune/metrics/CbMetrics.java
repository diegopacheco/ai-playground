package com.diegopacheco.autotune.metrics;

public record CbMetrics(
        String state,
        float failureRate,
        float slowCallRate,
        int bufferedCalls,
        int failedCalls,
        int slowCalls,
        int successfulCalls,
        long notPermittedCalls,
        long ts
) {}
