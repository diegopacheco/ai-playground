package com.diegopacheco.autotune.downstream;

public record Scenario(double failRate, long latencyMs, long jitterMs) {}
