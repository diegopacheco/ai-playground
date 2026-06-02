package com.diegopacheco.autotune.tune;

import com.diegopacheco.autotune.config.CircuitBreakerSettings;
import com.diegopacheco.autotune.config.Clamp;
import com.diegopacheco.autotune.metrics.CbMetrics;

import java.util.List;

public record TuneResult(
        CircuitBreakerSettings current,
        CircuitBreakerSettings proposed,
        CircuitBreakerSettings clamped,
        List<Clamp.FieldClamp> clamps,
        String rationale,
        String model,
        CbMetrics metrics
) {}
