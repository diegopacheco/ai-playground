package com.diegopacheco.autotune.config;

public record CircuitBreakerSettings(
        double failureRateThreshold,
        double slowCallRateThreshold,
        long slowCallDurationThresholdMs,
        String slidingWindowType,
        int slidingWindowSize,
        int minimumNumberOfCalls,
        long waitDurationInOpenStateSeconds,
        int permittedNumberOfCallsInHalfOpenState
) {}
