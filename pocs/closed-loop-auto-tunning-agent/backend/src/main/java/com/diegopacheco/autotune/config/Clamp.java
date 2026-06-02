package com.diegopacheco.autotune.config;

import java.util.ArrayList;
import java.util.List;

public final class Clamp {

    public record FieldClamp(String field, double proposed, double applied, boolean wasClamped) {}

    public record Result(CircuitBreakerSettings settings, List<FieldClamp> fields) {}

    private Clamp() {}

    public static Result apply(CircuitBreakerSettings p) {
        List<FieldClamp> fields = new ArrayList<>();

        double failureRate = bound("failureRateThreshold", p.failureRateThreshold(), 40, 70, fields);
        double slowRate = bound("slowCallRateThreshold", p.slowCallRateThreshold(), 50, 100, fields);
        long slowDuration = (long) bound("slowCallDurationThresholdMs", p.slowCallDurationThresholdMs(), 200, 5000, fields);
        int windowSize = (int) bound("slidingWindowSize", p.slidingWindowSize(), 10, 200, fields);
        int minCalls = (int) bound("minimumNumberOfCalls", p.minimumNumberOfCalls(), 5, 100, fields);
        long wait = (long) bound("waitDurationInOpenStateSeconds", p.waitDurationInOpenStateSeconds(), 5, 60, fields);
        int halfOpen = (int) bound("permittedNumberOfCallsInHalfOpenState", p.permittedNumberOfCallsInHalfOpenState(), 2, 20, fields);

        String windowType = "TIME".equalsIgnoreCase(p.slidingWindowType()) ? "TIME" : "COUNT";

        CircuitBreakerSettings clamped = new CircuitBreakerSettings(
                failureRate, slowRate, slowDuration, windowType,
                windowSize, minCalls, wait, halfOpen);

        return new Result(clamped, fields);
    }

    private static double bound(String field, double value, double min, double max, List<FieldClamp> fields) {
        double applied = Math.max(min, Math.min(max, value));
        fields.add(new FieldClamp(field, value, applied, applied != value));
        return applied;
    }
}
