package com.diegopacheco.autotune.tune;

public record Proposal(
        double failureRateThreshold,
        double slowCallRateThreshold,
        long slowCallDurationThresholdMs,
        String slidingWindowType,
        int slidingWindowSize,
        int minimumNumberOfCalls,
        long waitDurationInOpenStateSeconds,
        int permittedNumberOfCallsInHalfOpenState,
        String rationale
) {}
